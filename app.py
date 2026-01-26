import os
import re
import time
import json
import uuid
import requests
import secrets
import jwt
import difflib
from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Response, Body, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pymysql
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional, List, Dict, Any

# 数据源和知识库管理
from datasource_manager import init_datasource_manager, get_datasource_manager
from knowledge_base_manager import init_kb_manager, get_kb_manager
from analytics_manager import init_analytics_manager, get_analytics_manager
from insight_engine import init_insight_engine, get_insight_engine
from insight_scheduler import start_insight_scheduler

# Load environment variables
load_dotenv()

# 【调试】打印关键环境变量加载状态
print("="*60)
print("[ENV CHECK] 环境变量加载状态:")
print(f"  - ZHIPU_API_KEY: {'已设置' if os.getenv('ZHIPU_API_KEY') else '❌ 未设置'}")
print(f"  - DB_HOST: {'已设置' if os.getenv('DB_HOST') else '❌ 未设置'}")
print(f"  - LARK_WEBHOOK_URL: {'已设置 ✅' if os.getenv('LARK_WEBHOOK_URL') else '❌ 未设置'}")
if os.getenv('LARK_WEBHOOK_URL'):
    webhook_url = os.getenv('LARK_WEBHOOK_URL')
    print(f"  - Webhook URL (前40字符): {webhook_url[:40]}...")
# 调试 SHOW_SQL_DEBUG
show_sql_raw = os.getenv('SHOW_SQL_DEBUG', '未设置')
print(f"  - SHOW_SQL_DEBUG: '{show_sql_raw}' (类型: {type(show_sql_raw).__name__}, 长度: {len(show_sql_raw)})")
print("="*60)

# === Auth & Tenant Isolation Config ===
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
AUTH_USERS_JSON = os.getenv("AUTH_USERS_JSON", "")
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))
AUTH_TOKEN_SLIDING_ENABLED = os.getenv("AUTH_TOKEN_SLIDING_ENABLED", "true").lower() == "true"
AUTH_TOKEN_REFRESH_SECONDS = int(os.getenv("AUTH_TOKEN_REFRESH_SECONDS", "1800"))
# JWT 密钥，生产环境务必通过环境变量设置一个强随机字符串
JWT_SECRET = os.getenv("JWT_SECRET", "chatbi-default-secret-change-in-production")
JWT_ALGORITHM = "HS256"

TENANT_ISOLATION_ENABLED = os.getenv("ENABLE_TENANT_ISOLATION", "false").lower() == "true"
TENANT_COLUMN = os.getenv("TENANT_COLUMN", "operator_id")

_AUTH_USERS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_auth_users() -> Dict[str, Dict[str, Any]]:
    if not AUTH_USERS_JSON:
        return {}
    try:
        raw = json.loads(AUTH_USERS_JSON)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AUTH_USERS_JSON 解析失败: {exc}") from exc

    users: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(raw.values())
    else:
        raise ValueError("AUTH_USERS_JSON 必须是数组或对象")

    for item in items:
        if not isinstance(item, dict):
            continue
        username = str(item.get("username", "")).strip()
        password = str(item.get("password", "")).strip()
        operator_id = str(item.get("operator_id", "")).strip()
        if not username or not password or not operator_id:
            raise ValueError("AUTH_USERS_JSON 中每个用户必须包含 username/password/operator_id")
        users[username] = {
            "username": username,
            "password": password,
            "operator_id": operator_id,
        }
    return users


def _get_auth_users() -> Dict[str, Dict[str, Any]]:
    global _AUTH_USERS_CACHE
    if _AUTH_USERS_CACHE is None:
        _AUTH_USERS_CACHE = _load_auth_users()
    return _AUTH_USERS_CACHE


def _authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    users = _get_auth_users()
    user = users.get(username)
    if not user:
        return None
    if secrets.compare_digest(user.get("password", ""), password):
        return {
            "username": user["username"],
            "operator_id": user["operator_id"],
        }
    return None


def _issue_token(user: Dict[str, Any]) -> str:
    """生成 JWT token，包含用户信息和过期时间"""
    payload = {
        "username": user["username"],
        "operator_id": user["operator_id"],
        "exp": datetime.utcnow() + timedelta(seconds=AUTH_TOKEN_TTL_SECONDS),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def _parse_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _maybe_refresh_token(payload: Dict[str, Any], response: Response) -> None:
    if not AUTH_TOKEN_SLIDING_ENABLED:
        return
    exp_ts = payload.get("exp")
    if not exp_ts:
        return
    try:
        exp_ts = int(exp_ts)
    except Exception:
        return
    now_ts = int(datetime.utcnow().timestamp())
    remaining = exp_ts - now_ts
    if remaining > AUTH_TOKEN_REFRESH_SECONDS:
        return
    refreshed = _issue_token({
        "username": payload.get("username"),
        "operator_id": payload.get("operator_id"),
    })
    response.headers["X-Auth-Token"] = refreshed
    response.headers["X-Auth-Expires-In"] = str(AUTH_TOKEN_TTL_SECONDS)


def get_current_user(
    response: Response,
    authorization: Optional[str] = Header(None),
    x_auth_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        return {"username": "anonymous", "operator_id": ""}

    token = x_auth_token or _parse_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="未登录或缺少访问令牌")

    try:
        # JWT 无状态验证，不依赖服务器内存
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        _maybe_refresh_token(payload, response)
        return {
            "username": payload.get("username"),
            "operator_id": payload.get("operator_id"),
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="访问令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="访问令牌无效")


def _has_tenant_filter(sql: str, tenant_column: str, operator_id: str) -> bool:
    if not sql:
        return False
    pattern = rf"\b{re.escape(tenant_column)}\b\s*=\s*'{re.escape(operator_id)}'"
    return re.search(pattern, sql, re.IGNORECASE) is not None


def _inject_tenant_filter(sql: str, tenant_column: str, operator_id: str) -> str:
    sql_stripped = sql.strip()
    if not sql_stripped:
        return sql_stripped

    has_semicolon = sql_stripped.endswith(";")
    if has_semicolon:
        sql_stripped = sql_stripped[:-1].rstrip()

    sql_upper = sql_stripped.upper()
    if "WHERE" in sql_upper:
        replaced = re.sub(
            r"\bWHERE\b",
            f"WHERE {tenant_column} = '{operator_id}' AND ",
            sql_stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"{replaced};" if has_semicolon else replaced

    insert_pos = len(sql_stripped)
    for keyword in [" GROUP BY", " ORDER BY", " LIMIT", " HAVING", " UNION ", " INTERSECT ", " EXCEPT "]:
        pos = sql_upper.find(keyword)
        if pos != -1 and pos < insert_pos:
            insert_pos = pos

    if insert_pos == len(sql_stripped):
        injected = f"{sql_stripped} WHERE {tenant_column} = '{operator_id}'"
    else:
        before = sql_stripped[:insert_pos].rstrip()
        after = sql_stripped[insert_pos:]
        injected = f"{before} WHERE {tenant_column} = '{operator_id}' {after.lstrip()}"

    return f"{injected};" if has_semicolon else injected


def transform_sql_with_tenant(sql: str, operator_id: str) -> str:
    if not TENANT_ISOLATION_ENABLED:
        return sql
    if not operator_id:
        raise ValueError("租户隔离已启用，但 operator_id 为空")
    if _has_tenant_filter(sql, TENANT_COLUMN, operator_id):
        return sql
    return _inject_tenant_filter(sql, TENANT_COLUMN, operator_id)

# 对话历史管理（ChromaDB 持久化存储）
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
        """添加一条消息到会话历史（带租户隔离）"""
        now = self._now()
        def _sanitize_for_json(value):
            try:
                import pandas as pd
                from decimal import Decimal
                
                # 处理 Decimal 类型
                if isinstance(value, Decimal):
                    return float(value)
                
                # 处理 pandas Timestamp
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            
            # 递归处理字典
            if isinstance(value, dict):
                return {k: _sanitize_for_json(v) for k, v in value.items()}
            
            # 递归处理列表
            if isinstance(value, list):
                return [_sanitize_for_json(v) for v in value]
            
            # 处理其他日期时间类型
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return value
            
            return value

        payload = {
            "role": role,
            "content": content,
            "timestamp": now,
        }
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
                "operator_id": operator_id,  # 添加租户标识
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
            # 保留原有的 operator_id，如果没有则使用传入的
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
        """获取指定会话的历史记录（带租户隔离验证）"""
        # 先验证会话归属
        session_record = self._get_session_record(session_id)
        if session_record and operator_id:
            session_owner = session_record.get("operator_id", "")
            if session_owner and session_owner != operator_id:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_owner}，当前用户 {operator_id} 无权访问")
                return []  # 无权访问其他用户的会话
        if session_record and username:
            session_user = session_record.get("username", "")
            if session_user and session_user != username:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_user}，当前用户 {username} 无权访问")
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
        """获取最近消息（按 operator_id/username 过滤）"""
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
        if operator_id:
            owner_id = meta.get("operator_id", "")
            if owner_id and owner_id != operator_id:
                return None
        if username:
            owner_name = meta.get("username", "")
            if owner_name and owner_name != username:
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
        if operator_id:
            owner_id = meta.get("operator_id", "")
            if owner_id and owner_id != operator_id:
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
        data = None
        if username and operator_id:
            data = self.pins.get(where={"$and": [{"operator_id": operator_id}, {"username": username}]})
        elif username:
            data = self.pins.get(where={"username": username})
        elif operator_id:
            data = self.pins.get(where={"operator_id": operator_id})
        if data is None:
            data = self.pins.get()
        documents = data.get("documents") if data else []
        pinned_items = []
        for doc in documents or []:
            try:
                pinned_items.append(json.loads(doc))
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
        used_positions = {item.get("position") for item in pinned_items}
        if len(used_positions) >= 6:
            raise ValueError("看板最多支持 6 个收藏图表")
        position = next(pos for pos in range(6) if pos not in used_positions)

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

    def unpin_message(self, message_id: str, operator_id: str = "", username: str = "") -> bool:
        if operator_id or username:
            data = self.pins.get(ids=[message_id])
            if not data or not data.get("ids"):
                return False
            meta = (data.get("metadatas") or [{}])[0]
            owner_id = meta.get("operator_id", "")
            owner_name = meta.get("username", "")
            if operator_id and owner_id and owner_id != operator_id:
                return False
            if username and owner_name and owner_name != username:
                return False
        self.pins.delete(ids=[message_id])
        return True

    def update_pinned(self, message_id: str, payload: Dict[str, Any], operator_id: str = "", username: str = "") -> None:
        self.pins.upsert(
            ids=[message_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[{"operator_id": operator_id or "", "username": username or "", "position": payload.get("position", 0)}],
        )

    def get_context_history(self, session_id: str, operator_id: str = "", username: str = "") -> List[Dict]:
        return self.get_history(session_id, limit=self.max_history, operator_id=operator_id, username=username)

    def clear_session(self, session_id: str, operator_id: str = "", username: str = "") -> bool:
        """清除指定会话的历史（带租户隔离验证）"""
        # 先验证会话归属
        session_record = self._get_session_record(session_id)
        if session_record and operator_id:
            session_owner = session_record.get("operator_id", "")
            if session_owner and session_owner != operator_id:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_owner}，当前用户 {operator_id} 无权删除")
                return False  # 无权删除其他用户的会话
        if session_record and username:
            session_user = session_record.get("username", "")
            if session_user and session_user != username:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_user}，当前用户 {username} 无权删除")
                return False
        
        self.messages.delete(where={"session_id": session_id})
        self.sessions.delete(ids=[session_id])
        return True

    def get_sessions(self, operator_id: str = "", username: str = "") -> List[Dict]:
        """获取用户的会话列表（按 operator_id/username 过滤）"""
        # 如果提供了 username 或 operator_id，只返回该用户的会话
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
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        for session_id, doc, meta in zip(ids, documents, metadatas):
            title = doc or (meta or {}).get("title") or f"Session {session_id}"
            sessions_list.append({
                "id": session_id,
                "title": title,
                "created_at": (meta or {}).get("created_at"),
                "updated_at": (meta or {}).get("updated_at", (meta or {}).get("created_at")),
                "message_count": (meta or {}).get("message_count", 0),
            })

        return sorted(sessions_list, key=lambda x: x.get("updated_at") or "", reverse=True)

    def list_operator_ids(self) -> List[str]:
        """获取所有存在会话的 operator_id 列表"""
        data = self.sessions.get()
        if not data or not data.get("ids"):
            return []
        metadatas = data.get("metadatas") or []
        operator_ids = {meta.get("operator_id", "") for meta in metadatas if isinstance(meta, dict)}
        if not operator_ids:
            operator_ids.add("")
        return sorted(operator_ids)

    def list_usernames(self, operator_id: str = "") -> List[str]:
        """获取指定 operator 下的 username 列表"""
        if operator_id:
            data = self.sessions.get(where={"operator_id": operator_id})
        else:
            data = self.sessions.get()
        if not data or not data.get("ids"):
            return []
        metadatas = data.get("metadatas") or []
        usernames = {meta.get("username", "") for meta in metadatas if isinstance(meta, dict)}
        usernames.discard("")
        return sorted(usernames)

    def rename_session(self, session_id: str, new_title: str, operator_id: str = "", username: str = "") -> bool:
        """重命名会话（带租户隔离验证）"""
        session_record = self._get_session_record(session_id)
        if session_record is None:
            return False
        
        # 验证会话归属
        if operator_id:
            session_owner = session_record.get("operator_id", "")
            if session_owner and session_owner != operator_id:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_owner}，当前用户 {operator_id} 无权重命名")
                return False  # 无权重命名其他用户的会话
        if username:
            session_user = session_record.get("username", "")
            if session_user and session_user != username:
                print(f"[DEBUG] 会话 {session_id} 属于 {session_user}，当前用户 {username} 无权重命名")
                return False
        
        created_at = session_record.get("created_at") or self._now()
        updated_at = self._now()
        message_count = session_record.get("message_count", 0)
        title = new_title or session_record.get("title") or f"Session {session_id}"
        session_operator_id = session_record.get("operator_id") or operator_id
        session_username = session_record.get("username") or username
        self._upsert_session(session_id, title, created_at, updated_at, message_count, session_operator_id, session_username)
        return True

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Initialize ChromaDB
        ChromaDB_VectorStore.__init__(self, config=config)

        # Initialize OpenAI Client for Zhipu AI
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['api_base']
        )

        self.sql_model = os.getenv("OPENROUTER_SQL_MODEL", "openai/gpt-5.1")
        self.sql_client = self._build_openrouter_client()

        # Remove api_base from config to avoid Vanna legacy check error
        vanna_config = config.copy()
        if 'api_base' in vanna_config:
            del vanna_config['api_base']

        # Initialize OpenAI_Chat with the client
        OpenAI_Chat.__init__(self, client=client, config=vanna_config)

    def _build_openrouter_client(self) -> Optional[OpenAI]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        referer = os.getenv("OPENROUTER_REFERER")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name

        if headers:
            return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        return OpenAI(api_key=api_key, base_url=base_url)

    def _submit_sql_prompt(self, prompt, **kwargs) -> str:
        if self.sql_client is None:
            return self.submit_prompt(prompt, **kwargs)

        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        model = kwargs.get("model", self.sql_model)
        response = self.sql_client.chat.completions.create(
            model=model,
            messages=prompt,
            stop=None,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        重写父类方法，支持相似度阈值过滤
        ChromaDB使用L2距离，距离越小越相似
        """
        # 获取原始检索结果（包含距离信息）
        raw_results = self.sql_collection.query(
            query_texts=[question],
            n_results=self.n_results_sql,
        )
        
        # 相似度阈值：L2距离小于此值才保留（可通过config配置）
        distance_threshold = kwargs.get('sql_distance_threshold', 
                                       getattr(self, 'sql_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=True)
    
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """重写父类方法，支持相似度阈值过滤"""
        raw_results = self.ddl_collection.query(
            query_texts=[question],
            n_results=self.n_results_ddl,
        )
        
        distance_threshold = kwargs.get('ddl_distance_threshold', 
                                       getattr(self, 'ddl_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=False)
    
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """重写父类方法，支持相似度阈值过滤"""
        raw_results = self.documentation_collection.query(
            query_texts=[question],
            n_results=self.n_results_documentation,
        )
        
        distance_threshold = kwargs.get('doc_distance_threshold', 
                                       getattr(self, 'doc_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=False)
    
    def _filter_by_distance(self, query_results, threshold: float, parse_json: bool = False) -> list:
        """
        根据L2距离阈值过滤结果
        
        Args:
            query_results: ChromaDB查询结果
            threshold: 距离阈值，小于此值才保留（L2距离越小越相似）
            parse_json: 是否需要解析JSON（SQL类型需要）
        
        Returns:
            过滤后的文档列表
        """
        if query_results is None or 'documents' not in query_results:
            return []
        
        documents = query_results.get('documents', [[]])[0]
        distances = query_results.get('distances', [[]])[0]
        
        # 过滤：只保留距离小于阈值的结果
        filtered_docs = []
        for doc, dist in zip(documents, distances):
            if dist <= threshold:
                if parse_json:
                    try:
                        filtered_docs.append(json.loads(doc))
                    except:
                        filtered_docs.append(doc)
                else:
                    filtered_docs.append(doc)
            else:
                print(f"[DEBUG] 过滤掉相似度低的结果 (距离={dist:.3f} > 阈值={threshold})")
        
        print(f"[DEBUG] 相似度过滤: {len(documents)}条 -> {len(filtered_docs)}条 (阈值={threshold})")
        return filtered_docs

    def detect_intent(self, text: str) -> Dict[str, str]:
        """
        简单意图识别（规则优先）
        返回: intent, preferred_chart, time_grain, agg_hint
        """
        q = (text or "").lower()
        intent = "detail"
        preferred_chart = "table"
        time_grain = ""
        agg_hint = ""

        # 关键词优先级调整：明细优先级最高
        detail_keywords = ["明细", "详情", "列表", "详细信息", "明细数据", "数据明细"]
        trend_keywords = ["趋势", "变化", "走势", "曲线", "按日", "按月", "按周", "按小时", "同比", "环比"]
        distribution_keywords = ["分布", "占比", "比例", "构成", "份额"]
        compare_keywords = ["对比", "比较", "差异"]
        ranking_keywords = ["top", "排行", "排名", "前几"]

        # 【优先级1】明细数据 - 必须优先判断
        if any(k in q for k in detail_keywords):
            intent = "detail"
            preferred_chart = "table"
        # 【优先级2】排行数据
        elif any(k in q for k in ranking_keywords):
            intent = "ranking"
            preferred_chart = "bar"
        # 【优先级3】趋势分析
        elif any(k in q for k in trend_keywords):
            intent = "trend"
            preferred_chart = "line"
            agg_hint = "sum"
        # 【优先级4】分布占比
        elif any(k in q for k in distribution_keywords):
            intent = "distribution"
            preferred_chart = "pie"
        # 【优先级5】对比分析
        elif any(k in q for k in compare_keywords):
            intent = "comparison"
            preferred_chart = "bar"

        if "按日" in q or "每日" in q or "日" in q:
            time_grain = "day"
        elif "按月" in q or "每月" in q or "月" in q:
            time_grain = "month"
        elif "按周" in q or "每周" in q or "周" in q:
            time_grain = "week"
        elif "按小时" in q or "每小时" in q or "小时" in q:
            time_grain = "hour"

        return {
            "intent": intent,
            "preferred_chart": preferred_chart,
            "time_grain": time_grain,
            "agg_hint": agg_hint,
        }

    def _recommend_chart_type(self, df: pd.DataFrame, question: str) -> str:
        """
        基于数据特征和问题内容，智能推荐图表类型
        """
        row_count = len(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        question_lower = question.lower()
        
        # 【优先级1】明细数据场景 - 不画图，返回 None
        # 强匹配：如果问题中包含"明细"或"详情"，一定不画图
        strong_detail_keywords = ['明细', '详情', '列表', '详细信息', 'detail', 'list', 'records']
        if any(kw in question_lower for kw in strong_detail_keywords):
            print(f"[DEBUG] 检测到明细关键词，不生成图表")
            return None  # 明细数据不应该画图
        
        # 【优先级2】单值场景
        if row_count == 1 and len(numeric_cols) == 1:
            return "Indicator (go.Indicator) - 单一指标卡片"
        
        # 【优先级3】排行场景（优先于趋势判断）
        if any(kw in question_lower for kw in ['top', 'rank', '排名', '排行', '前', '最']):
            return "Bar Chart (px.bar) - 横向柱状图（水平）"
        
        # 【优先级4】占比/分布场景
        if any(kw in question_lower for kw in ['占比', '比例', '分布', 'proportion', 'distribution', '构成']):
            if len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() <= 8:
                return "Pie Chart (px.pie) - 饼图"
            else:
                return "Bar Chart (px.bar) - 柱状图"
        
        # 【优先级5】时间序列场景（趋势分析）
        trend_keywords = ['趋势', '变化', '走势', '曲线', 'trend', '按日', '按月', '按周']
        if datetime_cols or any(kw in question_lower for kw in trend_keywords):
            return "Line Chart (px.line) - 时间序列折线图"
        
        # 【优先级6】对比场景
        if any(kw in question_lower for kw in ['对比', '比较', 'compare', 'vs']):
            return "Bar Chart (px.bar) - 分组柱状图"
        
        # 【默认】根据数据结构推荐
        if len(numeric_cols) >= 2:
            return "Scatter Plot (px.scatter) - 散点图"
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            return "Bar Chart (px.bar) - 柱状图"
        else:
            return None  # 无合适图表，返回表格

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df: pd.DataFrame = None, chart_recommendation: str = None, **kwargs
    ) -> str:
        """
        优化版可视化代码生成，传递真实数据样本给LLM
        
        Args:
            question: 用户原始问题
            sql: 生成的SQL查询
            df: 查询结果DataFrame（真实数据）
            chart_recommendation: 外部传入的图表推荐（如果为None则表示不需要图表）
            **kwargs: 其他参数
        """
        if df is None:
            raise ValueError("DataFrame cannot be None for visualization generation")
        
        # 【关键】如果外部传入 chart_recommendation=None，说明是明细数据，直接返回 None
        if chart_recommendation is None:
            print(f"[DEBUG] 明细数据场景（外部传入），不生成可视化代码")
            return None
        
        # 检测用户问题的语言
        user_lang = self.detect_language(question) if question else 'zh'
        
        # 1. 数据分析
        row_count = len(df)
        col_count = len(df.columns)
        
        # 识别数值列、分类列、时间列
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 数据样本（前5行，转为JSON格式）
        sample_data = df.head(min(5, row_count)).to_dict('records')
        
        # 数据统计信息
        stats_info = []
        for col in numeric_cols[:5]:  # 最多展示5个数值列的统计
            try:
                stats_info.append(
                    f"  - {col}: 最小值={df[col].min()}, 最大值={df[col].max()}, "
                    f"平均值={df[col].mean():.2f}, 唯一值数={df[col].nunique()}"
                )
            except Exception:
                pass
        
        # 2. 如果没有传入推荐，则调用内部推荐逻辑
        if not chart_recommendation:
            chart_recommendation = self._recommend_chart_type(df, question or "")
            # 【关键】如果推荐结果是 None，说明是明细数据，不应该生成图表
            if chart_recommendation is None:
                print(f"[DEBUG] 明细数据场景（内部检测），不生成可视化代码")
                return None  # 返回 None，不生成图表

        # 2.5 趋势且存在分类维度时，优先生成多序列折线图（每个分类一条线）
        question_text = (question or "").lower()
        wants_category_series = any(k in question_text for k in ["每一项", "每个", "各", "分别", "按", "按类", "按类型", "按名称"])
        is_trend = "line" in chart_recommendation.lower() or "trend" in question_text or "趋势" in question_text
        if (is_trend or wants_category_series) and categorical_cols and numeric_cols:
            time_col = ""
            if datetime_cols:
                time_col = datetime_cols[0]
            else:
                for col in df.columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ['date', 'time', 'dt', 'day', 'month']):
                        time_col = col
                        break

            if time_col:
                cat_col = ""
                for preferred in ["license_name", "name", "type", "category"]:
                    if preferred in df.columns:
                        cat_col = preferred
                        break
                if not cat_col:
                    cat_col = categorical_cols[0]

                if "license_total_capacity" in df.columns and ("容量" in question_text or "capacity" in question_text):
                    value_col = "license_total_capacity"
                elif "daily_total_used" in df.columns:
                    value_col = "daily_total_used"
                else:
                    preferred_metrics = [c for c in numeric_cols if any(k in c.lower() for k in ["used", "count", "total", "amount"])]
                    value_col = preferred_metrics[0] if preferred_metrics else numeric_cols[0]
                title = (question or "趋势分析").replace("'", "").strip() or "趋势分析"
                return (
                    "import pandas as pd\n"
                    "import plotly.express as px\n\n"
                    f"df['{time_col}'] = pd.to_datetime(df['{time_col}'], errors='coerce')\n"
                    "df = df.dropna(subset=["
                    f"'{time_col}', '{cat_col}', '{value_col}'"
                    "])\n"
                    f"fig = px.line(\n"
                    f"    df,\n"
                    f"    x='{time_col}',\n"
                    f"    y='{value_col}',\n"
                    f"    color='{cat_col}',\n"
                    f"    title='{title}',\n"
                    f"    labels={{'{time_col}': '日期', '{value_col}': '数值', '{cat_col}': '类型'}},\n"
                    f")\n"
                    "fig.update_layout(font=dict(size=14), hovermode='x unified')\n"
                )
        
        # 3. 构建提示词
        if user_lang == 'zh':
            system_prompt = f"""你是一位数据可视化专家，精通 Plotly 图表库。

===用户问题===
{question or '数据可视化'}

===SQL查询===
{sql or 'N/A'}

===数据集信息===
- 数据维度: {row_count} 行 × {col_count} 列
- 数值列: {', '.join(numeric_cols) if numeric_cols else '无'}
- 分类列: {', '.join(categorical_cols) if categorical_cols else '无'}
- 时间列: {', '.join(datetime_cols) if datetime_cols else '无'}

===数据类型===
{df.dtypes.to_string()}

===数据样本（前{min(5, row_count)}行真实数据）===
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

===数值列统计===
{chr(10).join(stats_info) if stats_info else '无数值列'}

===推荐图表类型===
{chart_recommendation}
"""
            
            user_prompt = """请为这个数据集生成专业的 Plotly 可视化代码，要求：

**图表选择原则**：
1. 时间序列数据 → 使用 px.line()（折线图），X轴为时间字段
2. 分类占比数据 → 使用 px.pie() 或 px.bar()（饼图/柱状图）
3. 多维度对比 → 使用 px.bar()（分组柱状图）
4. 单一指标 → 使用 go.Indicator()（指示器卡片）
5. 排名数据 → 使用 px.bar()（横向柱状图）

**代码质量要求**：
1. 必须正确识别 X轴 和 Y轴 字段（参考上面的数据样本）
2. 为图表设置有意义的中文标题（基于用户问题）
3. 设置清晰的轴标签（使用 labels 参数）
4. 如果有时间字段，确保正确识别并作为X轴
5. 优化布局（font 字体大小、hovermode 悬停模式）
6. 对于时间类数据，使用 pd.to_datetime() 转换（如果需要）
7. 对于大数据集（>100行），考虑适当聚合

**输出格式**：
- 只输出可执行的 Python 代码，不要任何文字解释
- 代码中 DataFrame 变量名必须是 'df'
- 图表对象变量名必须是 'fig'
- 不要包含 fig.show()
- 必须 import 所需的库（import plotly.express as px 或 import plotly.graph_objects as go）

**示例代码结构**：
```python
import plotly.express as px

# 如果需要预处理时间字段
# df['date_col'] = pd.to_datetime(df['date_col'], format='%Y%m%d')

# 生成图表
fig = px.line(
    df, 
    x='时间字段名',  # 替换为实际字段
    y='数值字段名',  # 替换为实际字段
    title='用户友好的中文标题',
    labels={'时间字段名': '时间', '数值字段名': '数值'}
)

# 优化布局
fig.update_layout(
    font=dict(size=14),
    hovermode='x unified'
)
```

请立即生成代码（只输出代码，不要任何解释）：
"""
        else:  # 英文版
            system_prompt = f"""You are a data visualization expert proficient in Plotly.

===User Question===
{question or 'Data Visualization'}

===SQL Query===
{sql or 'N/A'}

===Dataset Information===
- Dimensions: {row_count} rows × {col_count} columns
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
- Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'None'}

===Data Types===
{df.dtypes.to_string()}

===Data Sample (first {min(5, row_count)} rows)===
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

===Numeric Statistics===
{chr(10).join(stats_info) if stats_info else 'No numeric columns'}

===Recommended Chart Type===
{chart_recommendation}
"""
            
            user_prompt = """Generate professional Plotly visualization code for this dataset:

**Chart Selection Rules**:
1. Time series → px.line() with time on X-axis
2. Category proportions → px.pie() or px.bar()
3. Multi-dimensional comparison → px.bar() (grouped)
4. Single metric → go.Indicator()
5. Rankings → px.bar() (horizontal)

**Code Quality Requirements**:
1. Correctly identify X and Y axis fields (refer to data sample above)
2. Set meaningful chart title
3. Add clear axis labels (use labels parameter)
4. If datetime column exists, use it as X-axis
5. Optimize layout (font size, hovermode)
6. Convert datetime if needed using pd.to_datetime()
7. Consider aggregation for large datasets (>100 rows)

**Output Format**:
- Python code only, no explanations
- DataFrame variable name: 'df'
- Figure variable name: 'fig'
- Do not include fig.show()
- Must import required libraries (import plotly.express as px or import plotly.graph_objects as go)

Generate code now (code only, no explanations):
"""
        
        message_log = [
            self.system_message(system_prompt),
            self.user_message(user_prompt)
        ]
        
        # 提交给LLM生成代码（优先使用 OpenRouter）
        if self.sql_client is not None:
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = os.getenv("OPENROUTER_SQL_MODEL", self.sql_model)
            plotly_code = self._submit_sql_prompt(message_log, **call_kwargs)
        else:
            plotly_code = self.submit_prompt(message_log, **kwargs)
        
        # 提取并清理代码
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def validate_trend_sql(self, sql: str) -> bool:
        """
        趋势类SQL最低校验: 时间字段 + 聚合 + group by + order by
        """
        if not sql:
            return False
        sql_upper = sql.upper()
        has_group_by = "GROUP BY" in sql_upper
        has_order_by = "ORDER BY" in sql_upper
        has_agg = any(k in sql_upper for k in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("])
        has_time_col = any(k in sql_upper for k in [
            "DATE", "TIME", "DAY", "MONTH", "YEAR", "HOUR", "WEEK",
            "STAT_DATE", "COLLECT_TIME", "DT"
        ])
        return has_group_by and has_order_by and has_agg and has_time_col

    def detect_language(self, text: str) -> str:
        """
        检测文本的主要语言(中文或英文)
        返回: 'zh' (中文) 或 'en' (英文)
        """
        # 统计中文字符数量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字母数量
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        # 如果中文字符占比超过30%,判定为中文
        total_chars = len(text.strip())
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3 or chinese_chars > english_chars:
            return 'zh'
        else:
            return 'en'

    def generate_sql_optimized(
        self,
        question: str,
        allow_llm_to_see_data=False,
        conversation_history: Optional[List[Dict]] = None,
        intent: Optional[Dict] = None,
        operator_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Optimized version of generate_sql that returns both SQL and Explanation in a single pass.
        支持对话历史上下文。
        
        Args:
            question: 用户问题
            allow_llm_to_see_data: 是否允许LLM查看数据
            conversation_history: 对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 其他参数
        """
        # 0. 检测用户问题的语言
        user_lang = self.detect_language(question)
        print(f"[DEBUG] 检测到用户语言: {'中文' if user_lang == 'zh' else '英文'}")
        
        # 1. Retrieve Context
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        # 去重：避免提示词重复导致长度膨胀与响应变慢
        def _dedupe_strings(values: List[str]) -> List[str]:
            seen = set()
            result = []
            for item in values:
                key = (item or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        def _dedupe_question_sql(values: List[Dict]) -> List[Dict]:
            seen = set()
            result = []
            for item in values or []:
                if not isinstance(item, dict):
                    continue
                q = (item.get("question") or "").strip()
                s = (item.get("sql") or "").strip()
                key = (q, s)
                if q and s and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        ddl_list = _dedupe_strings(ddl_list or [])
        doc_list = _dedupe_strings(doc_list or [])
        question_sql_list = _dedupe_question_sql(question_sql_list or [])

        # 2. Construct Prompt (根据语言动态生成)
        
        # 获取当前时间信息
        from datetime import datetime
        current_time = datetime.now()
        current_date_str = current_time.strftime('%Y-%m-%d')
        current_datetime_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        current_weekday_en = current_time.strftime('%A')
        
        # 中文星期映射
        weekday_map = {
            'Monday': '星期一',
            'Tuesday': '星期二',
            'Wednesday': '星期三',
            'Thursday': '星期四',
            'Friday': '星期五',
            'Saturday': '星期六',
            'Sunday': '星期日'
        }
        current_weekday_zh = weekday_map.get(current_weekday_en, current_weekday_en)
        
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        if initial_prompt is None:
            if user_lang == 'zh':
                # 中文提示词 - 添加当前时间信息
                time_context = (
                    f"===当前时间信息===\n"
                    f"当前日期: {current_date_str}\n"
                    f"当前时间: {current_datetime_str}\n"
                    f"星期: {current_weekday_zh}\n"
                    f"注意: 在生成涉及时间的SQL查询时，请使用上述当前时间作为参考基准。\n\n"
                )
                initial_prompt = (
                    time_context
                    + f"你是一个 {self.dialect} 数据库专家。"
                    + "请帮助生成 SQL 查询来回答用户的问题。"
                    + "你的回复应该仅基于给定的上下文，并遵循响应指南和格式说明。"
                )
            else:
                # 英文提示词 - 添加当前时间信息
                time_context = (
                    f"===Current Time Information===\n"
                    f"Current Date: {current_date_str}\n"
                    f"Current Time: {current_datetime_str}\n"
                    f"Day of Week: {current_weekday_en}\n"
                    f"Note: When generating SQL queries involving time, use the above current time as the reference point.\n\n"
                )
                initial_prompt = (
                    time_context
                    + f"You are a {self.dialect} expert. "
                    + "Please help to generate a SQL query to answer the question. "
                    + "Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
                )

        # RAG优化：降低token限制，避免prompt过长
        # DDL部分限制在3000 tokens（约3-5个表结构）
        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=3000)

        if self.static_documentation != "":
            static_doc_key = self.static_documentation.strip()
            if static_doc_key and static_doc_key not in {d.strip() for d in doc_list}:
                doc_list.append(self.static_documentation)

        # Documentation部分限制在5000 tokens（约5-8条业务说明）
        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=5000)

        # Custom Guidelines for Single-Pass Explanation (根据语言切换)
        if user_lang == 'zh':
            initial_prompt += (
                "===响应指南 \n"
                "1. 如果提供的上下文足够，请生成有效的 SQL 查询。\n"
                "2. **重要**: 你必须用中文提供简短的解释（1-2句话），说明查询的作用。\n"
                "3. 按照以下格式响应:\n"
                "   解释: [你的中文解释]\n"
                "   ```sql\n   [你的SQL查询]\n   ```\n"
                "4. 如果提供的上下文几乎足够，但需要了解特定列中的特定字符串，请生成一个中间SQL查询来查找该列中的不同字符串。在查询前添加注释 intermediate_sql。\n"
                "5. 如果提供的上下文不足，请用中文解释为什么无法生成。\n"
                f"6. 确保输出的 SQL 符合 {self.dialect} 标准且可执行。\n"
                "7. **MySQL关键提示**: 不要使用带格式字符串的 `UNIX_TIMESTAMP`（例如 `UNIX_TIMESTAMP(col, '%Y-%m-%d')` 是无效的）。`UNIX_TIMESTAMP()` 只接受 0 或 1 个参数。要格式化时间戳，请使用 `DATE_FORMAT(FROM_UNIXTIME(timestamp_col/1000), '%Y-%m-%d')`（如果是毫秒）或 `DATE_FORMAT(FROM_UNIXTIME(timestamp_col), '%Y-%m-%d')`（如果是秒）。\n"
                "8. **语言要求**: 所有解释和说明文字必须使用中文。\n"
                "9. **多数据库架构要求**: 本系统使用多数据库架构，SQL中的所有表名必须使用完全限定格式 `数据库名.表名`（例如: `SELECT * FROM database_name.table_name`）。绝不能省略数据库名，否则会导致执行错误。\n"
                "10. **严禁参数化查询占位符**: SQL查询必须是完整可执行的语句，不要使用参数占位符如 `?` 或 `:param`。所有值必须直接嵌入SQL中（字符串用单引号包裹）。\n"
                "11. **对话上下文**: 如果用户的问题中出现\"11月\"、\"上个月\"、\"本月\"等相对时间表述，请结合之前的对话历史和当前时间信息来理解用户的意图。\n"
            )
        else:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. If the provided context is sufficient, please generate a valid SQL query. \n"
                "2. **IMPORTANT**: You must provide a brief explanation (1-2 sentences) in English of what the query does. \n"
                "3. Format your response as follows:\n"
                "   Explanation: [Your explanation here in English]\n"
                "   ```sql\n   [Your SQL here]\n   ```\n"
                "4. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
                "5. If the provided context is insufficient, please explain in English why it can't be generated. \n"
                f"6. Ensure that the output SQL is {self.dialect}-compliant and executable. \n"
                "7. **CRITICAL for MySQL**: DO NOT use `UNIX_TIMESTAMP` with a format string (e.g., `UNIX_TIMESTAMP(col, '%Y-%m-%d')` is INVALID). `UNIX_TIMESTAMP()` only accepts 0 or 1 argument. To format a timestamp, use `DATE_FORMAT(FROM_UNIXTIME(timestamp_col/1000), '%Y-%m-%d')` (if ms) or `DATE_FORMAT(FROM_UNIXTIME(timestamp_col), '%Y-%m-%d')` (if seconds). \n"
                "8. **Language requirement**: All explanations must be in English. \n"
                "9. **Multi-Database Architecture Requirement**: This system uses a multi-database architecture. ALL table names in SQL queries MUST use the fully qualified format `database_name.table_name` (e.g., `SELECT * FROM database_name.table_name`). Never omit the database name or the query will fail. \n"
                "10. **NO Parameterized Queries**: The SQL must be a complete executable statement. DO NOT use parameter placeholders like `?` or `:param`. All values must be embedded directly in the SQL (strings wrapped in single quotes). \n"
                "11. **Conversation Context**: If the user mentions relative time expressions like \"last month\" or \"this month\", use the conversation history and current time information to understand their intent. \n"
            )

        # 意图增强提示（趋势类强约束）
        if intent and intent.get("intent") == "trend":
            if user_lang == 'zh':
                initial_prompt += (
                    "12. **趋势类SQL强制要求**: 必须包含时间维度字段（如日期/时间列），并进行聚合统计（如 SUM/COUNT/AVG），"
                    "且必须包含 `GROUP BY 时间字段` 与 `ORDER BY 时间字段`，按时间升序排列。\n"
                )
            else:
                initial_prompt += (
                    "12. **Trend SQL requirements**: Must include a time dimension column, an aggregate metric (SUM/COUNT/AVG), "
                    "and include `GROUP BY time` plus `ORDER BY time` in ascending order.\n"
                )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        # 添加对话历史到消息日志（在示例之后，当前问题之前）
        if conversation_history and len(conversation_history) > 0:
            print(f"[DEBUG] 添加 {len(conversation_history)} 条对话历史到上下文")
            for msg in conversation_history:
                if msg['role'] == 'user':
                    message_log.append(self.user_message(msg['content']))
                elif msg['role'] == 'assistant':
                    message_log.append(self.assistant_message(msg['content']))

        message_log.append(self.user_message(question))

        # 【调试日志】输出提示词与消息日志，便于排查
        print(f"\n{'='*60}")
        print("[DEBUG] Prompt (system):")
        print(initial_prompt)
        print(f"[DEBUG] message_log size: {len(message_log)}")
        for i, msg in enumerate(message_log):
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            print(f"[DEBUG] message_log[{i}] role={role}")
            print(content)
        print(f"{'='*60}\n")

        # 3. Submit Prompt (SQL生成使用专用模型)
        call_kwargs = dict(kwargs)
        if self.sql_client is not None:
            call_kwargs["model"] = os.getenv("OPENROUTER_SQL_MODEL", self.sql_model)
            llm_response = self._submit_sql_prompt(message_log, **call_kwargs)
        else:
            call_kwargs["model"] = os.getenv("ZHIPU_MODEL", "GLM-4.7")
            llm_response = self.submit_prompt(message_log, **call_kwargs)

        # 4. Handle Intermediate SQL (Simplified for optimization)
        if "intermediate_sql" in llm_response and allow_llm_to_see_data:
            # If intermediate SQL is needed, we might have to do the standard flow or just return.
            # For now, let's use the standard extract_sql to get the query, run it, and re-prompt.
            # This is the slow path, but necessary for correctness if hit.
            intermediate_sql = self.extract_sql(llm_response)
            try:
                df = self.run_sql(intermediate_sql, operator_id=operator_id)
                # Re-prompt with data (根据语言切换提示文本)
                message_log.append(self.assistant_message(llm_response))
                if user_lang == 'zh':
                    message_log.append(self.user_message(
                        f"以下是中间SQL查询 {intermediate_sql} 的结果（pandas DataFrame格式）: \n"
                        + df.to_markdown()
                    ))
                else:
                    message_log.append(self.user_message(
                        f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n"
                        + df.to_markdown()
                    ))
                if self.sql_client is not None:
                    llm_response = self._submit_sql_prompt(message_log, **call_kwargs)
                else:
                    llm_response = self.submit_prompt(message_log, **call_kwargs)
            except Exception as e:
                if user_lang == 'zh':
                    return f"执行中间SQL时出错: {e}"
                else:
                    return f"Error running intermediate SQL: {e}"

        return llm_response

    def run_sql(self, sql: str, **kwargs):
        # Custom SQL Runner to handle SSL and Multi-DB
        operator_id = kwargs.get("operator_id")
        datasource_id = kwargs.get("datasource_id")  # 支持指定数据源
        
        try:
            sql = transform_sql_with_tenant(sql, operator_id)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        # 尝试使用数据源管理器获取连接配置
        try:
            ds_manager = get_datasource_manager()
            db_config = ds_manager.get_connection_config(datasource_id)
        except Exception as e:
            print(f"[run_sql] 使用数据源管理器失败，降级到环境变量配置: {e}")
            # 降级到环境变量配置
            db_config = {
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'ssl': {
                    'check_hostname': False,
                    'verify_mode': False  # equivalent to CERT_NONE
                } if os.getenv('DB_SSL_ENABLED', 'True').lower() == 'true' else None
            }

        try:
            cnx = pymysql.connect(**db_config)
            cursor = cnx.cursor()
            cursor.execute(sql)

            # Fetch column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch data
            data = cursor.fetchall()

            # Convert to pandas DataFrame
            df = pd.DataFrame(data, columns=columns)

            cursor.close()
            cnx.close()
            return df

        except pymysql.Error as err:
            print(f"SQL Error: {err}")
            raise err

# Configuration for Zhipu AI (GLM-4)
zhipu_embedding = ZhipuAIEmbeddingFunction(config={
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'api_base': os.getenv('ZHIPU_API_BASE')  # Pass api_base to embedding function
})

base_dir = os.path.dirname(os.path.abspath(__file__))
default_chroma_path = os.path.join(base_dir, "chroma_db")
config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': os.getenv('CHROMA_DB_PATH', default_chroma_path),  # Path for ChromaDB storage
    'embedding_function': zhipu_embedding,
    'dialect': 'MySQL', # 显式指定 MySQL 方言，防止 LLM 生成不兼容的 SQL 函数 (如 UNIX_TIMESTAMP 多参数)
    # RAG优化：限制检索数量，避免把整个知识库塞进prompt
    'n_results_ddl': 3,           # DDL只检索最相关的3条表结构
    'n_results_documentation': 5,  # 文档检索5条最相关的业务说明
    'n_results_sql': 5,            # SQL示例检索5条最相似的问答对
}

# Initialize Vanna
vn = MyVanna(config=config)

# 设置相似度阈值（L2距离，越小越相似）
# 调优建议：
# - 1.0: 非常严格，只保留高度相关的结果
# - 1.5: 适中（推荐起始值）
# - 2.0: 宽松，保留更多候选结果
vn.sql_distance_threshold = float(os.getenv('RAG_SQL_THRESHOLD', '1.5'))
vn.ddl_distance_threshold = float(os.getenv('RAG_DDL_THRESHOLD', '1.5'))
vn.doc_distance_threshold = float(os.getenv('RAG_DOC_THRESHOLD', '1.5'))

# 初始化对话历史管理器（使用 ChromaDB 持久化）
conversation_history = ChromaConversationHistory(
    chroma_client=vn.chroma_client,
    max_history_per_session=10
)

# 初始化数据源管理器
print("[INIT] 初始化数据源管理器...")
ds_manager = init_datasource_manager(vn.chroma_client)

# 初始化知识库管理器
print("[INIT] 初始化知识库管理器...")
kb_manager = init_kb_manager(vn.chroma_client, zhipu_embedding)
print("[INIT] 知识库管理器初始化完成")

# 初始化系统指标与洞察引擎
print("[INIT] 初始化 Analytics 管理器...")
analytics_manager = init_analytics_manager(vn.chroma_client)
print("[INIT] 初始化 Insight 引擎...")
insight_engine = init_insight_engine(vn.chroma_client)

def run_query_and_chart(
    sql: str,
    question: Optional[str] = None,
    operator_id: Optional[str] = None,
    datasource_id: Optional[str] = None,
):
    df = vn.run_sql(sql=sql, operator_id=operator_id, datasource_id=datasource_id)

    # 清理不符合 JSON 规范的数据类型
    import numpy as np
    from decimal import Decimal
    
    # 1. 将 Decimal 类型转换为 float
    for col in df.columns:
        if df[col].dtype == 'object':  # 可能包含 Decimal
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    
    # 2. 清理 inf, -inf, nan
    df = df.replace([np.inf, -np.inf], None)
    df = df.replace({np.nan: None})
    
    # 3. 智能识别并转换日期列（针对字符串类型的日期字段）
    potential_date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'dt', 'day', 'month'])]
    for col in potential_date_cols:
        if df[col].dtype == 'object':  # 字符串类型
            try:
                # 先检查数据格式
                sample_value = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                print(f"[DEBUG] 列 '{col}' 的样本值: {sample_value}")
                
                # 尝试多种日期格式
                converted = False
                
                # 格式1: yyyyMMdd (8位数字字符串，如 20251224)
                if len(sample_value) == 8 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
                    print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (格式: yyyyMMdd)")
                    converted = True
                
                # 格式2: yyyyMMddHH (10位数字字符串)
                elif len(sample_value) == 10 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d%H', errors='coerce')
                    print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (格式: yyyyMMddHH)")
                    converted = True
                
                # 格式3: ISO 格式或其他常见格式
                if not converted:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (自动识别格式)")
                    else:
                        print(f"[DEBUG] 列 '{col}' 转换后全部为 NaT，可能格式不支持")
                        
            except Exception as e:
                print(f"[DEBUG] 列 '{col}' 无法转换为 datetime: {e}")
    
    # 4. 将时间类型转换为字符串（用于 JSON 序列化，必须在图表生成之前）
    try:
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            df[col] = df[col].apply(lambda v: v.isoformat() if pd.notna(v) and hasattr(v, "isoformat") else v)
            print(f"[DEBUG] 已将列 '{col}' 的 datetime 转换为 ISO 字符串格式")
    except Exception as e:
        print(f"[DEBUG] 时间列序列化处理失败: {e}")
    
    # 5. 进行意图检测，判断是否需要生成图表
    chart_json = None
    if question:
        try:
            # 意图检测
            intent = vn.detect_intent(question)
            print(f"[DEBUG] 问题意图: {intent}")
            
            # 如果意图是明细数据，直接跳过图表生成
            if intent.get('intent') == 'detail':
                print(f"[DEBUG] 明细数据场景，跳过图表生成")
            else:
                # 打印 DataFrame 信息用于调试
                print(f"\n[DEBUG] ===== DataFrame 信息 =====")
                print(f"[DEBUG] 列名: {df.columns.tolist()}")
                print(f"[DEBUG] 数据类型:\n{df.dtypes}")
                print(f"[DEBUG] 数据预览:\n{df.head()}")
                print(f"[DEBUG] 数据值:\n{df.to_dict('records')}")
                print(f"[DEBUG] ========================\n")
                
                # 先推荐图表类型
                chart_recommendation = vn._recommend_chart_type(df, question)
                
                # 如果推荐为 None（明细数据），不生成图表
                if chart_recommendation is None:
                    print(f"[DEBUG] 图表推荐为 None，不生成图表")
                else:
                    plotly_code = vn.generate_plotly_code(
                        question=question, 
                        sql=sql, 
                        df=df,
                        chart_recommendation=chart_recommendation
                    )
                    
                    # 如果返回 None，说明是明细数据，不生成图表
                    if plotly_code is None:
                        print(f"[DEBUG] 明细数据不生成图表，只返回表格")
                    else:
                        print(f"[DEBUG] 生成的 Plotly 代码:\n{plotly_code}\n")
                        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
                        if fig:
                            chart_json = fig.to_json()
        except Exception as e:
            print(f"可视化生成失败: {e}")
            import traceback
            traceback.print_exc()

    return df, chart_json

# Setup FastAPI
app = FastAPI(title="redtea chatBi MVP")

# Mount static files (for serving images)
app.mount("/img", StaticFiles(directory="img"), name="img")

# Import and Include Knowledge Base Router
from knowledge_base_api import router as kb_router
app.include_router(
    kb_router,
    prefix="/api/v0",
    tags=["Knowledge Base"],
    dependencies=[Depends(get_current_user)],
)

# Import and Include DataSource Router
from datasource_api import router as ds_router
app.include_router(
    ds_router,
    prefix="/api/v0",
    tags=["DataSource"],
    dependencies=[Depends(get_current_user)],
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Auth-Token", "X-Auth-Expires-In"],
)

# --- Manual API Routes Implementation ---

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # 支持会话ID以跟踪对话历史
    datasource_id: Optional[str] = None  # 指定数据源ID
    kb_id: Optional[str] = None  # 指定知识库ID

class SqlRequest(BaseModel):
    sql: str
    question: str = None
    datasource_id: Optional[str] = None  # 指定数据源ID

class LoginRequest(BaseModel):
    username: str
    password: str

class FeedbackRequest(BaseModel):
    question: str
    sql: str
    explanation: str = None
    feedback_type: str  # "up" (点赞) 或 "down" (点踩)
    comment: str = None
    message_id: Optional[str] = None

def send_lark_alert(feedback: FeedbackRequest):
    """发送点踩反馈到飞书群"""
    url = os.getenv("LARK_WEBHOOK_URL")
    
    print(f"\n[飞书通知] 准备发送点踩反馈...")
    print(f"[飞书通知] Webhook URL 状态: {'已配置 ✅' if url else '❌ 未配置'}")
    
    if not url:
        print("=" * 60)
        print("❌ 错误: LARK_WEBHOOK_URL 环境变量未设置")
        print("解决方法:")
        print("  1. 在 .env 文件中添加: LARK_WEBHOOK_URL=你的webhook地址")
        print("  2. 确保 .env 文件在项目根目录")
        print("  3. 重启应用使配置生效")
        print("=" * 60)
        return

    # 构造飞书卡片消息
    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "🚨 用户反馈：回答不准确 (点踩)"},
                "template": "red"
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**用户提问:**\n{feedback.question}"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**生成的 SQL:**\n```sql\n{feedback.sql}\n```"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**解释:**\n{feedback.explanation or '无'}"}},
                {"tag": "hr"},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": f"反馈时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}]}
            ]
        }
    }

    try:
        print(f"[飞书通知] 正在发送到: {url[:50]}...")
        resp = requests.post(url, json=payload, timeout=5)
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 0:
                print(f"[飞书通知] ✅ 发送成功！")
            else:
                print(f"[飞书通知] ⚠️ 飞书返回错误: {result}")
        else:
            print(f"[飞书通知] ❌ HTTP错误: {resp.status_code}, 响应: {resp.text}")
            
    except requests.exceptions.Timeout:
        print(f"[飞书通知] ❌ 请求超时，请检查网络连接")
    except requests.exceptions.RequestException as e:
        print(f"[飞书通知] ❌ 网络错误: {e}")
    except Exception as e:
        print(f"[飞书通知] ❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()

def _require_operator_id(current_user: Dict[str, Any]) -> str:
    operator_id = (current_user or {}).get("operator_id", "")
    if TENANT_ISOLATION_ENABLED and not operator_id:
        raise HTTPException(status_code=401, detail="租户隔离已启用，但未绑定 operator_id")
    return operator_id

@app.post("/api/v0/login")
def login(request: LoginRequest):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="认证功能已关闭")
    user = _authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = _issue_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "operator_id": user["operator_id"],
        "expires_in": AUTH_TOKEN_TTL_SECONDS,
    }

@app.get("/api/v0/config")
def get_config():
    # Controls whether the frontend displays the raw SQL code block
    # Default to False for production/business users
    raw_value = os.getenv('SHOW_SQL_DEBUG', 'False')
    show_sql = raw_value.lower() == 'true'
    print(f"[DEBUG] SHOW_SQL_DEBUG 原始值: '{raw_value}', 解析后: {show_sql}")
    return {"show_sql": show_sql}

@app.post("/api/v0/generate_sql")
def generate_sql(
    request: QuestionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        print(f"\n{'='*60}")
        print(f"[DEBUG] 接收到问题: {request.question}")
        print(f"[DEBUG] 入参 session_id: {request.session_id}")
        print(f"[DEBUG] 入参 datasource_id: {request.datasource_id}")
        print(f"[DEBUG] 入参 kb_id: {request.kb_id}")
        print(f"{'='*60}\n")
        # 0. 获取或生成 session_id
        import uuid
        session_id = request.session_id or str(uuid.uuid4())
        datasource_id = request.datasource_id  # 数据源ID
        kb_id = request.kb_id  # 知识库ID
        
        # 0.1 检测用户问题的语言
        user_lang = vn.detect_language(request.question)

        # 0.2 意图识别（用于趋势类SQL约束）
        intent = vn.detect_intent(request.question)
        
        # 0.3 获取当前用户的 operator_id
        operator_id = _require_operator_id(current_user)
        
        # 0.4 获取该会话的对话历史（带租户隔离）
        username = (current_user or {}).get("username", "")
        history = conversation_history.get_context_history(session_id, operator_id=operator_id, username=username)
        print(f"[DEBUG] Session ID: {session_id}, 历史记录数: {len(history)}")
        print(f"[DEBUG] Intent: {intent}")
        print(f"[DEBUG] Operator ID: {operator_id}")
        print(f"[DEBUG] DataSource ID: {datasource_id}")
        print(f"[DEBUG] KB ID: {kb_id}")
        
        # 1. Generate SQL using Optimized Vanna Method (Single Pass for SQL + Explanation)
        # 将对话历史传递给 generate_sql_optimized
        raw_result = vn.generate_sql_optimized(
            question=request.question, 
            allow_llm_to_see_data=True,
            conversation_history=history,
            intent=intent,
            operator_id=operator_id,
        )
        
        # 【调试日志】记录LLM原始输出,方便排查问题
        print(f"\n{'='*60}")
        print(f"[DEBUG] 用户问题: {request.question}")
        print(f"[DEBUG] Session ID: {session_id}")
        print(f"[DEBUG] 检测语言: {'中文' if user_lang == 'zh' else '英文'}")
        print(f"[DEBUG] LLM原始回复:\n{raw_result}")
        print(f"{'='*60}\n")

        # 2. Parse Result to extract SQL and Explanation
        # Expected format: "Explanation: ... ```sql ... ```"
        
        # 尝试多种SQL提取模式(增强鲁棒性)
        sql_match = None
        explanation = ""
        
        # 模式1: 标准的 ```sql ... ```
        sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        
        # 模式2: 通用的 ``` ... ```
        if not sql_match:
            sql_match = re.search(r"```\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        
        # 模式3: 检测SELECT/INSERT/UPDATE/DELETE等SQL关键字(无代码块包裹的情况)
        if not sql_match:
            sql_keywords_pattern = r'(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s+.*?(?:;|$)'
            potential_sql = re.search(sql_keywords_pattern, raw_result, re.DOTALL | re.IGNORECASE)
            if potential_sql:
                print(f"[DEBUG] {'检测到无代码块包裹的SQL，尝试提取...' if user_lang == 'zh' else 'Detected SQL without code blocks, attempting extraction...'}")
                
                # 提取从SQL关键字开始到分号或字符串末尾的内容
                sql_start = potential_sql.start()
                
                # 尝试找到SQL结束位置(分号或两个连续换行)
                sql_end_match = re.search(r';|\n\n', raw_result[sql_start:])
                if sql_end_match:
                    sql_end = sql_start + sql_end_match.end()
                else:
                    sql_end = len(raw_result)
                
                extracted_sql = raw_result[sql_start:sql_end].strip()
                
                # 只清理开头的注释行（以 -- 或 /* 开头的行），保留SQL语句本身
                lines = extracted_sql.split('\n')
                sql_start_index = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # 跳过空行和注释行
                    if stripped and not stripped.startswith('--') and not stripped.startswith('/*'):
                        sql_start_index = i
                        break
                extracted_sql = '\n'.join(lines[sql_start_index:]).strip()
                
                # 创建一个模拟的match对象
                class MockMatch:
                    def __init__(self, text):
                        self._text = text
                    def group(self, n):
                        return self._text
                
                sql_match = MockMatch(extracted_sql)
                
                # 提取SQL之前的内容作为解释
                explanation = raw_result[:sql_start].strip()

        if sql_match:
            clean_sql = sql_match.group(1).strip()
            
            # Remove "intermediate_sql" marker if present
            clean_sql = re.sub(r'^intermediate_sql\s*', '', clean_sql, flags=re.IGNORECASE)
            # 清理SQL中的注释行(-- 开头),但保留内联注释
            clean_sql_lines = []
            for line in clean_sql.split('\n'):
                stripped = line.strip()
                # 保留非注释行,或者是有代码的注释行(内联注释)
                if not stripped.startswith('--') or ' -- ' in line:
                    clean_sql_lines.append(line)
            clean_sql = '\n'.join(clean_sql_lines).strip()
            
            # Extract explanation (everything before the SQL block usually)
            # Or look for "Explanation:" or "解释:" prefix
            if not explanation:  # 如果之前没有提取到explanation
                # 尝试匹配中文"解释:"
                explanation_match = re.search(r"解释[:：]\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE)
                if not explanation_match:
                    # 尝试匹配英文"Explanation:"
                    explanation_match = re.search(r"Explanation:\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE)
                
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    # Fallback: Use text before code block
                    parts = raw_result.split("```")
                    if len(parts) > 0:
                        explanation = parts[0].replace("Explanation:", "").replace("解释:", "").replace("解释：", "").strip()
            
            if not explanation:
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == 'zh' else "Here is the SQL query for your request."
            
            # 确保explanation中不包含SQL代码
            if "SELECT" in explanation.upper() or "FROM" in explanation.upper():
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == 'zh' else "Here is the SQL query for your request."

            print(f"[DEBUG] {'成功提取SQL' if user_lang == 'zh' else 'Successfully extracted SQL'}: {clean_sql[:100]}...")
            print(f"[DEBUG] {'解释' if user_lang == 'zh' else 'Explanation'}: {explanation[:100]}...")

            # 趋势类SQL校验，不通过则尝试重试一次
            if intent.get("intent") == "trend" and not vn.validate_trend_sql(clean_sql):
                print("[DEBUG] Trend SQL validation failed, retrying with stronger constraints...")
                retry_prompt = request.question + "（请严格输出趋势类SQL：时间维度+聚合+按时间分组排序）"
                raw_retry = vn.generate_sql_optimized(
                    question=retry_prompt,
                    allow_llm_to_see_data=True,
                    conversation_history=history,
                    intent=intent,
                    operator_id=operator_id,
                )
                retry_match = re.search(r"```sql\s*(.*?)\s*```", raw_retry, re.DOTALL | re.IGNORECASE)
                if retry_match:
                    retry_sql = retry_match.group(1).strip()
                    retry_sql = re.sub(r'^intermediate_sql\s*', '', retry_sql, flags=re.IGNORECASE).strip()
                    if vn.validate_trend_sql(retry_sql):
                        clean_sql = retry_sql
                        print("[DEBUG] Trend SQL validation passed after retry.")
            
            # 保存对话历史（带租户隔离）
            conversation_history.add_message(
                session_id,
                "user",
                request.question,
                operator_id=operator_id,
                username=(current_user or {}).get("username", ""),
            )
            print(f"[DEBUG] 已保存对话历史到 session {session_id}, operator_id={operator_id}")

            try:
                clean_sql = transform_sql_with_tenant(clean_sql, operator_id)
                start_time = time.perf_counter()
                df, chart_json = run_query_and_chart(
                    clean_sql,
                    request.question,
                    operator_id=operator_id,
                    datasource_id=datasource_id,
                )
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                get_analytics_manager().record_query(
                    latency_ms=latency_ms,
                    operator_id=operator_id,
                    datasource_id=datasource_id,
                    sql=clean_sql,
                )
            except pymysql.Error as e:
                error_msg = f"数据库错误: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            result_rows = df.to_dict(orient='records')
            columns = df.columns.tolist()

            assistant_msg_id = conversation_history.add_message(
                session_id,
                "assistant",
                f"{explanation}\n\nSQL: {clean_sql}",
                data={
                    "question": request.question,
                    "sql": clean_sql,
                    "explanation": explanation,
                    "result": result_rows,
                    "columns": columns,
                    "chart": chart_json,
                    "intent": intent,
                    "datasource_id": datasource_id,
                },
                operator_id=operator_id,  # 添加租户隔离
                username=(current_user or {}).get("username", ""),
            )

            return {
                "sql": clean_sql,
                "is_sql": True,
                "explanation": explanation,
                "session_id": session_id,  # 返回session_id给前端
                "message_id": assistant_msg_id,
                "result": result_rows,
                "columns": columns,
                "chart": chart_json,
                "intent": intent,
            }
        else:
            # No SQL found, treat as conversational response
            print(f"[DEBUG] {'未检测到SQL，作为对话回复处理' if user_lang == 'zh' else 'No SQL detected, treating as conversational response'}")
            
            # 保存对话历史（带租户隔离）
            conversation_history.add_message(
                session_id,
                "user",
                request.question,
                operator_id=operator_id,
                username=(current_user or {}).get("username", ""),
            )
            conversation_history.add_message(
                session_id,
                "assistant",
                raw_result,
                operator_id=operator_id,
                username=(current_user or {}).get("username", ""),
            )
            print(f"[DEBUG] 已保存对话历史到 session {session_id}, operator_id={operator_id}")
            
            return {
                "text": raw_result,
                "is_sql": False,
                "explanation": raw_result,
                "session_id": session_id  # 返回session_id给前端
            }

    except Exception as e:
        # 捕获所有异常并返回友好的错误信息
        error_msg = f"生成 SQL 失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v0/run_sql")
def run_sql(
    request: SqlRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        operator_id = _require_operator_id(current_user)
        print(f"Executing SQL: {request.sql}")  # Add logging to debug SQL errors
        print(f"DataSource ID: {request.datasource_id}")
        start_time = time.perf_counter()
        df, chart_json = run_query_and_chart(
            request.sql,
            request.question,
            operator_id=operator_id,
            datasource_id=request.datasource_id,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        get_analytics_manager().record_query(
            latency_ms=latency_ms,
            operator_id=operator_id,
            datasource_id=request.datasource_id,
            sql=request.sql,
        )

        # Convert DataFrame to list of dicts for JSON response
        return {
            "result": df.to_dict(orient='records'),
            "columns": df.columns.tolist(),
            "chart": chart_json
        }

    except pymysql.Error as e:
        # 数据库连接或执行错误
        error_msg = f"数据库错误: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        # 其他未预期的错误
        error_msg = f"执行失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v0/clear_session")
def clear_session(
    session_id: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """清除指定会话的对话历史（带租户隔离验证）"""
    try:
        operator_id = (current_user or {}).get("operator_id", "")
        success = conversation_history.clear_session(session_id, operator_id=operator_id)
        if not success:
            raise HTTPException(status_code=403, detail="无权删除此会话")
        print(f"[DEBUG] 已清除 session {session_id} 的历史记录, operator_id={operator_id}")
        return {"success": True, "message": "会话历史已清除"}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"清除会话失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v0/sessions")
def get_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    """获取当前用户的所有活跃会话（按 operator_id 过滤）"""
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    return conversation_history.get_sessions(operator_id=operator_id, username=username)

@app.get("/api/v0/history/{session_id}")
def get_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取指定会话的历史（带租户隔离验证）"""
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    history = conversation_history.get_history(session_id, operator_id=operator_id, username=username)
    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    pinned_ids = {item.get("message_id") for item in pinned_items if item.get("message_id")}
    for msg in history:
        msg["pinned"] = msg.get("id") in pinned_ids
    return history

@app.post("/api/v0/sessions/{session_id}/rename")
def rename_session(
    session_id: str,
    body: Dict = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """重命名会话（带租户隔离验证）"""
    operator_id = (current_user or {}).get("operator_id", "")
    new_title = body.get("title")
    username = (current_user or {}).get("username", "")
    if conversation_history.rename_session(session_id, new_title, operator_id=operator_id, username=username):
        return {"success": True, "message": "Session renamed"}
    raise HTTPException(status_code=404, detail="会话不存在或无权操作")

@app.delete("/api/v0/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """删除指定会话（带租户隔离验证）"""
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    success = conversation_history.clear_session(session_id, operator_id=operator_id, username=username)
    if not success:
        raise HTTPException(status_code=403, detail="无权删除此会话")
    return {"success": True, "message": "Session deleted"}

@app.post("/api/v0/feedback")
def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    print(f"收到反馈: {request.feedback_type} - {request.question}")
    operator_id = _require_operator_id(current_user)
    get_analytics_manager().record_feedback(request.feedback_type, operator_id=operator_id)
    if request.message_id:
        conversation_history.set_message_feedback(
            request.message_id,
            request.feedback_type,
            operator_id=operator_id,
        )

    # 如果是点踩，发送飞书通知
    if request.feedback_type == "down":
        send_lark_alert(request)

    return {"status": "success", "message": "Feedback received"}


def _refresh_pinned_item(pinned: Dict[str, Any], operator_id: str = "") -> Dict[str, Any]:
    sql = pinned.get("sql")
    if not sql:
        return pinned
    datasource_id = pinned.get("datasource_id")
    try:
        df, chart_json = run_query_and_chart(
            sql,
            pinned.get("question"),
            operator_id=operator_id,
            datasource_id=datasource_id,
        )
        result_rows = df.to_dict(orient="records")
        columns = df.columns.tolist()

        def _sanitize(value):
            try:
                import pandas as pd
                from decimal import Decimal
                if isinstance(value, Decimal):
                    return float(value)
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return value
            return value

        result_rows = _sanitize(result_rows)
        pinned["chart"] = chart_json
        pinned["result"] = result_rows
        pinned["columns"] = columns
        pinned["last_refreshed"] = datetime.now().isoformat()
        conversation_history.update_pinned(
            pinned.get("message_id"),
            pinned,
            operator_id=operator_id,
            username=pinned.get("username", ""),
        )

        insights_enabled = os.getenv("ANALYTICS_INSIGHT_ENABLED", "true").lower() == "true"
        if insights_enabled:
            get_insight_engine().generate_insights_from_result(
                result_rows=result_rows,
                columns=columns,
                title_hint=pinned.get("question") or "核心指标",
                sql=pinned.get("sql"),
                chart=pinned.get("chart"),
                operator_id=operator_id,
            )
    except Exception as exc:
        print(f"[Analytics] 刷新收藏图表失败: {exc}")
    return pinned


def _normalize_similarity_text(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[`'\"\(\)\[\]\{\}]+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def _build_pinned_similarity_text(item: Dict[str, Any]) -> str:
    sql = item.get("sql") or ""
    question = item.get("question") or item.get("content") or ""
    return f"{_normalize_similarity_text(sql)} | {_normalize_similarity_text(question)}".strip(" |")


def _similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

_PIN_EMBED_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_PIN_EMBED_CACHE_MAX = 300


def _get_zhipu_embedding_function():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2")
    api_base = os.getenv("ZHIPU_API_BASE")
    try:
        return ZhipuAIEmbeddingFunction(
            config={"api_key": api_key, "model_name": model_name, "api_base": api_base}
        )
    except Exception as exc:
        print(f"[Analytics] 初始化智谱 embedding 失败: {exc}")
        return None


def _get_text_embeddings(texts: List[str]) -> List[List[float]]:
    embedding_fn = _get_zhipu_embedding_function()
    if embedding_fn is None:
        raise RuntimeError("ZHIPU_API_KEY 未配置或 embedding 初始化失败")
    results: List[List[float]] = []
    pending = []
    pending_idx = []
    for idx, text in enumerate(texts):
        cached = _PIN_EMBED_CACHE.get(text)
        if cached is not None:
            results.append(cached)
        else:
            results.append([])
            pending.append(text)
            pending_idx.append(idx)
    if pending:
        embeddings = embedding_fn(pending)
        for text, emb, idx in zip(pending, embeddings, pending_idx):
            results[idx] = emb
            _PIN_EMBED_CACHE[text] = emb
        while len(_PIN_EMBED_CACHE) > _PIN_EMBED_CACHE_MAX:
            _PIN_EMBED_CACHE.popitem(last=False)
    for text in texts:
        if text in _PIN_EMBED_CACHE:
            _PIN_EMBED_CACHE.move_to_end(text)
    return results


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def _pinned_completeness_score(item: Dict[str, Any]) -> float:
    score = 0.0
    if item.get("chart"):
        score += 3.0
    if item.get("result") and item.get("columns"):
        score += 2.0
        try:
            score += min(len(item.get("result") or []), 20) / 20.0
        except Exception:
            pass
    if item.get("sql"):
        score += 1.0
    if item.get("question"):
        score += 0.5
    return score


def _pinned_sort_timestamp(item: Dict[str, Any]) -> str:
    return item.get("last_refreshed") or item.get("pinned_at") or ""


def _is_better_pinned(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    score_a = _pinned_completeness_score(a)
    score_b = _pinned_completeness_score(b)
    if score_a != score_b:
        return score_a > score_b
    return _pinned_sort_timestamp(a) >= _pinned_sort_timestamp(b)


def _dedupe_pinned_items(items: List[Dict[str, Any]], threshold: float, mode: str) -> List[Dict[str, Any]]:
    if not items:
        return items
    if mode == "ai":
        keys = [_build_pinned_similarity_text(item) for item in items]
        try:
            embeddings = _get_text_embeddings(keys)
        except Exception as exc:
            print(f"[Analytics] AI 去重失败，回退规则模式: {exc}")
            mode = "rule"
    if mode == "ai":
        groups = []
        for item, key, emb in zip(items, keys, embeddings):
            placed = False
            for group in groups:
                if _cosine_similarity(emb, group["emb"]) >= threshold:
                    best = group["best"]
                    if _is_better_pinned(item, best):
                        group["best"] = item
                        group["key"] = key
                        group["emb"] = emb
                    placed = True
                    break
            if not placed:
                groups.append({"key": key, "best": item, "emb": emb})
    else:
        groups = []
        for item in items:
            key = _build_pinned_similarity_text(item)
            placed = False
            for group in groups:
                if _similarity_ratio(key, group["key"]) >= threshold:
                    best = group["best"]
                    if _is_better_pinned(item, best):
                        group["best"] = item
                        group["key"] = key
                    placed = True
                    break
            if not placed:
                groups.append({"key": key, "best": item})
    deduped = [group["best"] for group in groups]
    deduped.sort(key=lambda x: x.get("position", 0))
    return deduped


def _refresh_all_pinned() -> None:
    operator_ids = conversation_history.list_operator_ids()
    for operator_id in operator_ids:
        usernames = conversation_history.list_usernames(operator_id=operator_id)
        if not usernames:
            usernames = [""]
        for username in usernames:
            if os.getenv("ANALYTICS_AUTO_PIN_ENABLED", "true").lower() == "true":
                _auto_pin_recent(operator_id, username=username)
            pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
            for item in pinned_items:
                _refresh_pinned_item(item, operator_id=operator_id)


def _auto_pin_recent(operator_id: str, username: str = "") -> None:
    if os.getenv("ANALYTICS_AUTO_PIN_ENABLED", "true").lower() != "true":
        return
    max_total = 6
    max_auto = int(os.getenv("ANALYTICS_AUTO_PIN_LIMIT", "3"))
    recent_days = int(os.getenv("ANALYTICS_AUTO_PIN_DAYS", "7"))
    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    if len(pinned_items) >= max_total:
        return
    pinned_ids = {item.get("message_id") for item in pinned_items}
    dedup_enabled = os.getenv("ANALYTICS_PIN_DEDUP_ENABLED", "true").lower() == "true"
    dedup_threshold = float(os.getenv("ANALYTICS_PIN_DEDUP_THRESHOLD", "0.86"))
    dedup_mode = os.getenv("ANALYTICS_PIN_DEDUP_MODE", "rule").lower()
    existing_keys = []
    for item in pinned_items:
        key = _build_pinned_similarity_text(item)
        if key:
            existing_keys.append(key)
    existing_embs: List[List[float]] = []
    if dedup_enabled and dedup_mode == "ai" and existing_keys:
        try:
            existing_embs = _get_text_embeddings(existing_keys)
        except Exception as exc:
            print(f"[Analytics] AI 去重初始化失败，回退规则模式: {exc}")
            dedup_mode = "rule"

    recent_messages = conversation_history.list_recent_messages(
        operator_id=operator_id,
        username=username,
        limit=200,
        since_days=recent_days,
    )
    candidates = []
    for msg in recent_messages:
        if msg.get("role") != "assistant":
            continue
        if not msg.get("chart"):
            continue
        if not msg.get("sql"):
            continue
        if msg.get("id") in pinned_ids:
            continue
        candidate_key = _build_pinned_similarity_text(msg)
        if candidate_key:
            if dedup_enabled and dedup_mode == "ai":
                try:
                    candidate_emb = _get_text_embeddings([candidate_key])[0]
                    if any(_cosine_similarity(candidate_emb, emb) >= dedup_threshold for emb in existing_embs):
                        continue
                except Exception:
                    if any(_similarity_ratio(candidate_key, key) >= dedup_threshold for key in existing_keys):
                        continue
            elif dedup_enabled:
                if any(_similarity_ratio(candidate_key, key) >= dedup_threshold for key in existing_keys):
                    continue
            else:
                if candidate_key in existing_keys:
                    continue
        candidates.append(msg)

    seen_keys = list(existing_keys)
    seen_embs = list(existing_embs)
    for msg in candidates[:max_auto]:
        if len(conversation_history.list_pinned(operator_id=operator_id, username=username)) >= max_total:
            break
        candidate_key = _build_pinned_similarity_text(msg)
        if candidate_key:
            if dedup_enabled and dedup_mode == "ai":
                try:
                    candidate_emb = _get_text_embeddings([candidate_key])[0]
                    if any(_cosine_similarity(candidate_emb, emb) >= dedup_threshold for emb in seen_embs):
                        continue
                except Exception:
                    if any(_similarity_ratio(candidate_key, key) >= dedup_threshold for key in seen_keys):
                        continue
            elif dedup_enabled:
                if any(_similarity_ratio(candidate_key, key) >= dedup_threshold for key in seen_keys):
                    continue
            else:
                if candidate_key in seen_keys:
                    continue
        try:
            conversation_history.pin_message(msg["id"], operator_id=operator_id, username=username)
            if candidate_key:
                seen_keys.append(candidate_key)
                if dedup_enabled and dedup_mode == "ai":
                    try:
                        seen_embs.append(_get_text_embeddings([candidate_key])[0])
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[Analytics] 自动收藏失败: {exc}")


@app.get("/api/v0/analytics/metrics")
def get_analytics_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    operator_id = _require_operator_id(current_user)
    metrics = get_analytics_manager().get_metrics(operator_id=operator_id)
    return metrics.to_dict()


@app.get("/api/v0/analytics/insights")
def get_analytics_insights(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    items = get_insight_engine().list_insights(operator_id=operator_id, limit=limit)
    return {"items": items}


@app.post("/api/v0/analytics/insights/clear")
def clear_analytics_insights(current_user: Dict[str, Any] = Depends(get_current_user)):
    operator_id = _require_operator_id(current_user)
    cleared = get_insight_engine().clear_insights(operator_id=operator_id)
    return {"success": True, "cleared": cleared}


@app.delete("/api/v0/analytics/insights/{insight_id}")
def delete_analytics_insight(
    insight_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    success = get_insight_engine().delete_insight(insight_id, operator_id=operator_id)
    if not success:
        raise HTTPException(status_code=404, detail="洞察不存在或无权删除")
    return {"success": True}


@app.get("/api/v0/analytics/pinned")
def get_pinned_charts(
    refresh: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    _auto_pin_recent(operator_id, username=username)
    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    if refresh:
        pinned_items = [_refresh_pinned_item(item, operator_id=operator_id) for item in pinned_items]
    if os.getenv("ANALYTICS_PIN_DEDUP_ENABLED", "true").lower() == "true":
        dedup_threshold = float(os.getenv("ANALYTICS_PIN_DEDUP_THRESHOLD", "0.86"))
        dedup_mode = os.getenv("ANALYTICS_PIN_DEDUP_MODE", "rule").lower()
        pinned_items = _dedupe_pinned_items(pinned_items, dedup_threshold, dedup_mode)
    return {"items": pinned_items}


@app.post("/api/v0/messages/{message_id}/pin")
def pin_message(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    try:
        payload = conversation_history.pin_message(message_id, operator_id=operator_id, username=username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "item": payload,
        "items": conversation_history.list_pinned(operator_id=operator_id, username=username),
    }


@app.delete("/api/v0/messages/{message_id}/pin")
def unpin_message(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    success = conversation_history.unpin_message(message_id, operator_id=operator_id, username=username)
    return {
        "success": success,
        "items": conversation_history.list_pinned(operator_id=operator_id, username=username),
    }


@app.post("/api/v0/analytics/pinned/{message_id}/refresh")
def refresh_pinned_chart(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    target = next((item for item in pinned_items if item.get("message_id") == message_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")
    refreshed = _refresh_pinned_item(target, operator_id=operator_id)
    return {"success": True, "item": refreshed}


@app.post("/api/v0/analytics/refresh")
def refresh_analytics(current_user: Dict[str, Any] = Depends(get_current_user)):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    _auto_pin_recent(operator_id, username=username)
    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    for item in pinned_items:
        _refresh_pinned_item(item, operator_id=operator_id)
    return {"success": True}


@app.post("/api/v0/analytics/clear")
def clear_analytics(current_user: Dict[str, Any] = Depends(get_current_user)):
    operator_id = _require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    cleared_pins = conversation_history.clear_pins(operator_id=operator_id, username=username)
    cleared_insights = get_insight_engine().clear_insights(operator_id=operator_id)
    cleared_events = get_analytics_manager().clear_events(operator_id=operator_id)
    return {
        "success": True,
        "cleared": {
            "pins": cleared_pins,
            "insights": cleared_insights,
            "events": cleared_events,
        },
    }

@app.post("/api/v0/generate_questions")
def generate_questions(current_user: Dict[str, Any] = Depends(get_current_user)):
    # Simple placeholder or call vn.generate_questions() if available
    return ["How many users are there?", "Show me the latest orders"]

@app.get("/")
def root():
    return {"message": "redtea chatBi MVP is running. Use /api/v0/ for API endpoints."}

# Simple HTML UI for testing
@app.get("/ui", response_class=FileResponse)
def ui():
    return FileResponse("templates/ui.html")


@app.on_event("startup")
def start_analytics_scheduler():
    enabled = os.getenv("ANALYTICS_SCHEDULER_ENABLED", "true").lower() == "true"
    if not enabled:
        return
    interval = int(os.getenv("ANALYTICS_REFRESH_INTERVAL", "3600"))
    start_insight_scheduler(_refresh_all_pinned, interval_seconds=interval)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
