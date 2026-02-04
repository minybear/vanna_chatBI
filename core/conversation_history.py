"""ChromaDB 持久化会话与消息、收藏、用户设置。"""
import os
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any


class SimpleEmbeddingFunction:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def name(self) -> str:
        return "simple"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in input]

    def _embed_text(self, text: str) -> List[float]:
        import hashlib
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]


class ChromaConversationHistory:
    """使用 ChromaDB 持久化保存会话与消息，支持 operator_id 租户隔离"""

    def __init__(self, chroma_client, max_history_per_session=10):
        self.max_history = max_history_per_session
        self.embedding_function = SimpleEmbeddingFunction()
        self.messages = chroma_client.get_or_create_collection(
            name="chat_messages",
            embedding_function=self.embedding_function,
        )
        self.sessions = chroma_client.get_or_create_collection(
            name="chat_sessions",
            embedding_function=self.embedding_function,
        )
        self.pins = chroma_client.get_or_create_collection(
            name="pinned_charts",
            embedding_function=self.embedding_function,
        )
        self.pin_blacklist = chroma_client.get_or_create_collection(
            name="pin_blacklist",
            embedding_function=self.embedding_function,
        )
        self.user_settings = chroma_client.get_or_create_collection(
            name="user_settings",
            embedding_function=self.embedding_function,
        )

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _get_session_record(self, session_id: str) -> Optional[Dict]:
        data = self.sessions.get(ids=[session_id])
        if data and data.get("ids"):
            meta = (data.get("metadatas") or [{}])[0]
            title = (data.get("documents") or [None])[0] or meta.get("title")
            return {
                "title": title,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "message_count": meta.get("message_count", 0),
                "operator_id": meta.get("operator_id", ""),
                "username": meta.get("username", ""),
            }
        return None

    def _upsert_session(
        self,
        session_id: str,
        title: str,
        created_at: str,
        updated_at: str,
        message_count: int,
        operator_id: str = "",
        username: str = "",
    ):
        self.sessions.upsert(
            ids=[session_id],
            documents=[title],
            metadatas=[{
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": message_count,
                "operator_id": operator_id,
                "username": username,
            }],
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        data: Optional[Dict] = None,
        operator_id: str = "",
        username: str = "",
    ) -> str:
        now = self._now()

        def _sanitize_for_json(value):
            try:
                import pandas as pd
                if isinstance(value, Decimal):
                    return float(value)
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            if isinstance(value, dict):
                return {k: _sanitize_for_json(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sanitize_for_json(v) for v in value]
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return value
            return value

        payload = {"role": role, "content": content, "timestamp": now}
        if data:
            payload.update(_sanitize_for_json(data))

        msg_id = f"{session_id}-{uuid.uuid4().hex}"
        self.messages.add(
            documents=[json.dumps(payload, ensure_ascii=False)],
            ids=[msg_id],
            metadatas=[{
                "session_id": session_id,
                "timestamp": now,
                "role": role,
                "operator_id": operator_id,
                "username": username,
            }],
        )

        session_record = self._get_session_record(session_id)
        if session_record is None:
            created_at = now
            message_count = 0
            title = f"Session {session_id[:8]}"
            session_operator_id = operator_id
            session_username = username
        else:
            created_at = session_record.get("created_at") or now
            message_count = session_record.get("message_count", 0)
            title = session_record.get("title") or f"Session {session_id[:8]}"
            session_operator_id = session_record.get("operator_id") or operator_id
            session_username = session_record.get("username") or username

        message_count += 1
        if role == "user" and message_count == 1:
            title = content[:30] + ("..." if len(content) > 30 else "")

        self._upsert_session(session_id, title, created_at, now, message_count, session_operator_id, session_username)
        return msg_id

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        operator_id: str = "",
        username: str = "",
    ) -> List[Dict]:
        session_record = self._get_session_record(session_id)
        if session_record and operator_id:
            if session_record.get("operator_id", "") and session_record.get("operator_id") != operator_id:
                return []
        if session_record and username:
            if session_record.get("username", "") and session_record.get("username") != username:
                return []

        data = self.messages.get(where={"session_id": session_id})
        if not data or not data.get("ids"):
            return []

        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        messages = []
        for msg_id, doc, meta in zip(ids, documents, metadatas):
            try:
                msg = json.loads(doc) if isinstance(doc, str) else {"content": doc}
            except Exception:
                msg = {"content": doc}
            msg.setdefault("role", meta.get("role"))
            msg.setdefault("timestamp", meta.get("timestamp"))
            msg["id"] = msg_id
            messages.append(msg)

        messages.sort(key=lambda x: x.get("timestamp") or "")
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def list_recent_messages(
        self,
        operator_id: str = "",
        username: str = "",
        limit: int = 200,
        since_days: int = 7,
    ) -> List[Dict]:
        if username and operator_id:
            data = self.messages.get(where={"$and": [{"operator_id": operator_id}, {"username": username}]})
        elif username:
            data = self.messages.get(where={"username": username})
        elif operator_id:
            data = self.messages.get(where={"operator_id": operator_id})
        else:
            data = self.messages.get()
        if not data or not data.get("ids"):
            return []

        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        messages: List[Dict] = []
        cutoff = datetime.now() - timedelta(days=since_days)

        for msg_id, doc, meta in zip(ids, documents, metadatas):
            try:
                msg = json.loads(doc) if isinstance(doc, str) else {"content": doc}
            except Exception:
                msg = {"content": doc}
            msg.setdefault("role", meta.get("role"))
            msg.setdefault("timestamp", meta.get("timestamp"))
            msg["id"] = msg_id
            ts = msg.get("timestamp")
            try:
                ts_dt = datetime.fromisoformat(ts) if ts else None
            except Exception:
                ts_dt = None
            if ts_dt and ts_dt < cutoff:
                continue
            messages.append(msg)

        messages.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return messages[:limit]

    def get_message_by_id(self, message_id: str, operator_id: str = "", username: str = "") -> Optional[Dict]:
        data = self.messages.get(ids=[message_id])
        if not data or not data.get("ids"):
            return None
        meta = (data.get("metadatas") or [{}])[0] or {}
        if operator_id and meta.get("operator_id", "") and meta.get("operator_id") != operator_id:
            return None
        if username and meta.get("username", "") and meta.get("username") != username:
            return None
        doc = (data.get("documents") or [None])[0]
        try:
            msg = json.loads(doc) if isinstance(doc, str) else {"content": doc}
        except Exception:
            msg = {"content": doc}
        msg.setdefault("role", meta.get("role"))
        msg.setdefault("timestamp", meta.get("timestamp"))
        msg["id"] = message_id
        return msg

    def set_message_feedback(self, message_id: str, feedback_type: str, operator_id: str = "") -> bool:
        data = self.messages.get(ids=[message_id])
        if not data or not data.get("ids"):
            return False
        meta = (data.get("metadatas") or [{}])[0]
        if operator_id and meta.get("operator_id", "") and meta.get("operator_id") != operator_id:
            return False
        doc = (data.get("documents") or [None])[0]
        try:
            payload = json.loads(doc) if isinstance(doc, str) else {"content": doc}
        except Exception:
            payload = {"content": doc}
        payload["feedback_type"] = feedback_type
        payload["feedback_at"] = self._now()
        self.messages.upsert(
            ids=[message_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[meta],
        )
        return True

    def list_pinned(self, operator_id: str = "", username: str = "") -> List[Dict[str, Any]]:
        all_data = self.pins.get()
        documents = all_data.get("documents") if all_data else []
        pinned_items = []
        for doc in documents or []:
            try:
                item = json.loads(doc)
                item_operator = item.get("operator_id") or ""
                item_username = item.get("username") or ""
                if operator_id and item_operator and item_operator != operator_id:
                    continue
                if username and item_username and item_username != username:
                    continue
                if username and not item_username and operator_id and item_operator == operator_id:
                    pass
                elif username and not item_username:
                    continue
                pinned_items.append(item)
            except Exception:
                continue
        pinned_items.sort(key=lambda x: x.get("position", 0))
        return pinned_items

    def clear_pins(self, operator_id: str = "", username: str = "") -> int:
        if username and operator_id:
            data = self.pins.get(where={"$and": [{"operator_id": operator_id}, {"username": username}]})
        elif username:
            data = self.pins.get(where={"username": username})
        elif operator_id:
            data = self.pins.get(where={"operator_id": operator_id})
        else:
            data = self.pins.get()
        ids = data.get("ids") if data else []
        if not ids:
            return 0
        self.pins.delete(ids=ids)
        return len(ids)

    def pin_message(self, message_id: str, operator_id: str = "", username: str = "") -> Dict[str, Any]:
        existing = self.pins.get(ids=[message_id])
        if existing and existing.get("ids"):
            doc = (existing.get("documents") or [None])[0]
            return json.loads(doc) if isinstance(doc, str) else {"message_id": message_id}

        message = self.get_message_by_id(message_id, operator_id=operator_id, username=username)
        if not message:
            raise ValueError("消息不存在或无权访问")
        has_chart = bool(message.get("chart"))
        has_table = bool(message.get("result")) and bool(message.get("columns"))
        if not (has_chart or has_table):
            raise ValueError("该消息没有可收藏的图表")

        pinned_items = self.list_pinned(operator_id=operator_id, username=username)
        max_pins = 6
        if len(pinned_items) >= max_pins:
            sorted_items = sorted(pinned_items, key=lambda x: x.get("pinned_at") or "")
            oldest_id = sorted_items[0].get("message_id")
            if oldest_id:
                self.unpin_message(oldest_id, operator_id=operator_id, username=username, add_to_blacklist=False)
                pinned_items = self.list_pinned(operator_id=operator_id, username=username)

        used_positions = {item.get("position") for item in pinned_items}
        position = next(pos for pos in range(max_pins) if pos not in used_positions)

        payload = {
            "message_id": message_id,
            "question": message.get("question") or message.get("content") or "",
            "sql": message.get("sql") or "",
            "chart": message.get("chart"),
            "columns": message.get("columns"),
            "result": message.get("result"),
            "datasource_id": message.get("datasource_id") or "",
            "operator_id": operator_id or "",
            "username": username or "",
            "pinned_at": self._now(),
            "last_refreshed": self._now(),
            "position": position,
        }
        self.pins.upsert(
            ids=[message_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[{"operator_id": operator_id or "", "username": username or "", "position": position}],
        )
        return payload

    def unpin_message(self, message_id: str, operator_id: str = "", username: str = "", add_to_blacklist: bool = True) -> bool:
        data = self.pins.get(ids=[message_id])
        if not data or not data.get("ids"):
            return False
        meta = (data.get("metadatas") or [{}])[0]
        if operator_id and meta.get("operator_id", "") and meta.get("operator_id") != operator_id:
            return False
        if username and meta.get("username", "") and meta.get("username") != username:
            return False
        self.pins.delete(ids=[message_id])
        if add_to_blacklist:
            self._add_to_pin_blacklist(message_id, operator_id, username)
        return True

    def _add_to_pin_blacklist(self, message_id: str, operator_id: str = "", username: str = "") -> None:
        self.pin_blacklist.upsert(
            ids=[message_id],
            documents=[message_id],
            metadatas=[{"operator_id": operator_id or "", "username": username or "", "removed_at": self._now()}],
        )

    def is_in_pin_blacklist(self, message_id: str, operator_id: str = "", username: str = "") -> bool:
        data = self.pin_blacklist.get(ids=[message_id])
        if not data or not data.get("ids"):
            return False
        meta = (data.get("metadatas") or [{}])[0]
        if operator_id and meta.get("operator_id", "") and meta.get("operator_id") != operator_id:
            return False
        if username and meta.get("username", "") and meta.get("username") != username:
            return False
        return True

    def clear_pin_blacklist(self, operator_id: str = "", username: str = "") -> int:
        all_data = self.pin_blacklist.get()
        if not all_data or not all_data.get("ids"):
            return 0
        ids_to_delete = []
        for i, mid in enumerate(all_data.get("ids", [])):
            meta = (all_data.get("metadatas") or [])[i] if i < len(all_data.get("metadatas", [])) else {}
            if operator_id and meta.get("operator_id", "") and meta.get("operator_id") != operator_id:
                continue
            if username and meta.get("username", "") and meta.get("username") != username:
                continue
            ids_to_delete.append(mid)
        if ids_to_delete:
            self.pin_blacklist.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def _get_settings_id(self, operator_id: str, username: str) -> str:
        return f"settings-{operator_id or 'default'}-{username or 'default'}"

    def get_user_settings(self, operator_id: str = "", username: str = "") -> Dict[str, Any]:
        settings_id = self._get_settings_id(operator_id, username)
        data = self.user_settings.get(ids=[settings_id])
        if data and data.get("ids"):
            doc = (data.get("documents") or [None])[0]
            if doc:
                try:
                    return json.loads(doc)
                except Exception:
                    pass
        return self.get_default_settings()

    def get_default_settings(self) -> Dict[str, Any]:
        return {
            "auto_pin_enabled": os.getenv("ANALYTICS_AUTO_PIN_ENABLED", "true").lower() == "true",
            "auto_pin_require_helpful": os.getenv("ANALYTICS_AUTO_PIN_REQUIRE_HELPFUL", "true").lower() == "true",
            "auto_pin_limit": int(os.getenv("ANALYTICS_AUTO_PIN_LIMIT", "3")),
            "auto_pin_days": int(os.getenv("ANALYTICS_AUTO_PIN_DAYS", "7")),
            "dedup_enabled": os.getenv("ANALYTICS_PIN_DEDUP_ENABLED", "true").lower() == "true",
            "dedup_threshold": float(os.getenv("ANALYTICS_PIN_DEDUP_THRESHOLD", "0.70")),
            "dedup_mode": os.getenv("ANALYTICS_PIN_DEDUP_MODE", "rule").lower(),
            "max_pins": 6,
            "insight_enabled": os.getenv("ANALYTICS_INSIGHT_ENABLED", "true").lower() == "true",
            "alert_fatigue_enabled": os.getenv("ALERT_FATIGUE_ENABLED", "true").lower() == "true",
            "alert_fatigue_cooldown_hours": float(os.getenv("ALERT_FATIGUE_COOLDOWN_HOURS", "24")),
        }

    def save_user_settings(self, settings: Dict[str, Any], operator_id: str = "", username: str = "") -> Dict[str, Any]:
        settings_id = self._get_settings_id(operator_id, username)
        merged = {**self.get_default_settings(), **settings, "updated_at": self._now()}
        self.user_settings.upsert(
            ids=[settings_id],
            documents=[json.dumps(merged, ensure_ascii=False)],
            metadatas=[{"operator_id": operator_id or "", "username": username or ""}],
        )
        return merged

    def update_pinned(self, message_id: str, payload: Dict[str, Any], operator_id: str = "", username: str = "") -> None:
        self.pins.upsert(
            ids=[message_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[{"operator_id": operator_id or "", "username": username or "", "position": payload.get("position", 0)}],
        )

    def get_context_history(self, session_id: str, operator_id: str = "", username: str = "") -> List[Dict]:
        return self.get_history(session_id, limit=self.max_history, operator_id=operator_id, username=username)

    def clear_session(self, session_id: str, operator_id: str = "", username: str = "") -> bool:
        session_record = self._get_session_record(session_id)
        if session_record and operator_id and session_record.get("operator_id", "") and session_record.get("operator_id") != operator_id:
            return False
        if session_record and username and session_record.get("username", "") and session_record.get("username") != username:
            return False
        self.messages.delete(where={"session_id": session_id})
        self.sessions.delete(ids=[session_id])
        return True

    def get_sessions(self, operator_id: str = "", username: str = "") -> List[Dict]:
        if username and operator_id:
            data = self.sessions.get(where={"$and": [{"operator_id": operator_id}, {"username": username}]})
        elif username:
            data = self.sessions.get(where={"username": username})
        elif operator_id:
            data = self.sessions.get(where={"operator_id": operator_id})
        else:
            data = self.sessions.get()
        if not data or not data.get("ids"):
            return []
        sessions_list = []
        for session_id, doc, meta in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []):
            meta = meta or {}
            sessions_list.append({
                "id": session_id,
                "title": doc or meta.get("title") or f"Session {session_id}",
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at", meta.get("created_at")),
                "message_count": meta.get("message_count", 0),
            })
        return sorted(sessions_list, key=lambda x: x.get("updated_at") or "", reverse=True)

    def list_operator_ids(self) -> List[str]:
        data = self.sessions.get()
        if not data or not data.get("ids"):
            return []
        operator_ids = {m.get("operator_id", "") for m in (data.get("metadatas") or []) if isinstance(m, dict)}
        if not operator_ids:
            operator_ids.add("")
        return sorted(operator_ids)

    def list_usernames(self, operator_id: str = "") -> List[str]:
        data = self.sessions.get(where={"operator_id": operator_id}) if operator_id else self.sessions.get()
        if not data or not data.get("ids"):
            return []
        usernames = {m.get("username", "") for m in (data.get("metadatas") or []) if isinstance(m, dict)}
        usernames.discard("")
        return sorted(usernames)

    def rename_session(self, session_id: str, new_title: str, operator_id: str = "", username: str = "") -> bool:
        session_record = self._get_session_record(session_id)
        if session_record is None:
            return False
        if operator_id and session_record.get("operator_id", "") and session_record.get("operator_id") != operator_id:
            return False
        if username and session_record.get("username", "") and session_record.get("username") != username:
            return False
        created_at = session_record.get("created_at") or self._now()
        updated_at = self._now()
        message_count = session_record.get("message_count", 0)
        title = new_title or session_record.get("title") or f"Session {session_id}"
        session_operator_id = session_record.get("operator_id") or operator_id
        session_username = session_record.get("username") or username
        self._upsert_session(session_id, title, created_at, updated_at, message_count, session_operator_id, session_username)
        return True
