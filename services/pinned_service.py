"""收藏图表刷新、去重、自动收藏。"""
import os
import re
import difflib
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict, Any, Callable

from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction

from services.label_service import recommend_chart_label

_PIN_EMBED_CACHE: OrderedDict[str, List[float]] = OrderedDict()
_PIN_EMBED_CACHE_MAX = 300


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
    dot = norm_a = norm_b = 0.0
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
    if _pinned_completeness_score(a) != _pinned_completeness_score(b):
        return _pinned_completeness_score(a) > _pinned_completeness_score(b)
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
                    if _is_better_pinned(item, group["best"]):
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
                    if _is_better_pinned(item, group["best"]):
                        group["best"] = item
                        group["key"] = key
                    placed = True
                    break
            if not placed:
                groups.append({"key": key, "best": item})
    deduped = [g["best"] for g in groups]
    deduped.sort(key=lambda x: x.get("position", 0))
    return deduped


def get_same_type_message_ids_to_replace(
    items: List[Dict[str, Any]], keep_message_id: str, threshold: float, mode: str
) -> List[str]:
    """
    找出与 keep_message_id 同类型（同一相似度组）的其它 pinned 的 message_id 列表，
    用于 pin 时「用最新覆盖原先」：unpin 这些 id，只保留 keep_message_id。
    若 keep_message_id 不在 items 中或同组仅一条，返回空列表。
    """
    if not items or not keep_message_id:
        return []
    target_item = next((x for x in items if x.get("message_id") == keep_message_id), None)
    if not target_item:
        return []
    keys = [_build_pinned_similarity_text(item) for item in items]
    if mode == "ai":
        try:
            embeddings = _get_text_embeddings(keys)
        except Exception:
            mode = "rule"
    # 构建分组：每组记录该组内所有 item（用于收集同组所有 message_id）
    if mode == "ai":
        groups: List[Dict[str, Any]] = []
        for item, key, emb in zip(items, keys, embeddings):
            placed = False
            for group in groups:
                if _cosine_similarity(emb, group["emb"]) >= threshold:
                    if _is_better_pinned(item, group["best"]):
                        group["best"] = item
                        group["key"] = key
                        group["emb"] = emb
                    group["all_items"].append(item)
                    placed = True
                    break
            if not placed:
                groups.append({"key": key, "best": item, "emb": emb, "all_items": [item]})
    else:
        groups = []
        for item in items:
            key = _build_pinned_similarity_text(item)
            placed = False
            for group in groups:
                if _similarity_ratio(key, group["key"]) >= threshold:
                    if _is_better_pinned(item, group["best"]):
                        group["best"] = item
                        group["key"] = key
                    group["all_items"].append(item)
                    placed = True
                    break
            if not placed:
                groups.append({"key": key, "best": item, "all_items": [item]})
    for group in groups:
        all_ids = [x.get("message_id") for x in group["all_items"] if x.get("message_id")]
        if keep_message_id in all_ids:
            return [mid for mid in all_ids if mid != keep_message_id]
    return []


def _sanitize_pinned_result(value):
    from decimal import Decimal
    import pandas as pd
    try:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _sanitize_pinned_result(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_pinned_result(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return value
    return value


def refresh_pinned_item(
    pinned: Dict[str, Any],
    operator_id: str,
    conversation_history,
    run_query_and_chart: Callable,
    get_insight_engine,
) -> Dict[str, Any]:
    """刷新单条收藏图表数据。"""
    sql = pinned.get("sql")
    if not sql:
        return pinned
    datasource_id = pinned.get("datasource_id")
    try:
        df, chart_json = run_query_and_chart(
            sql, pinned.get("question"), operator_id=operator_id, datasource_id=datasource_id
        )
        result_rows = _sanitize_pinned_result(df.to_dict(orient="records"))
        columns = df.columns.tolist()
        pinned["chart"] = chart_json
        pinned["result"] = result_rows
        pinned["columns"] = columns
        pinned["last_refreshed"] = datetime.now().isoformat()
        conversation_history.update_pinned(
            pinned.get("message_id"), pinned,
            operator_id=operator_id, username=pinned.get("username", ""),
        )
        user_settings = conversation_history.get_user_settings(operator_id=operator_id, username=pinned.get("username", ""))
        if user_settings.get("insight_enabled", True):
            get_insight_engine().generate_insights_from_result(
                result_rows=result_rows, columns=columns,
                title_hint=pinned.get("question") or "核心指标",
                sql=pinned.get("sql"), chart=pinned.get("chart"),
                operator_id=operator_id,
            )
    except Exception as exc:
        print(f"[Analytics] 刷新收藏图表失败: {exc}")
    return pinned


def refresh_all_pinned(conversation_history, run_query_and_chart: Callable, get_insight_engine) -> None:
    """刷新所有用户的收藏图表。"""
    operator_ids = conversation_history.list_operator_ids()
    for operator_id in operator_ids:
        usernames = conversation_history.list_usernames(operator_id=operator_id)
        if not usernames:
            usernames = [""]
        for username in usernames:
            if os.getenv("ANALYTICS_AUTO_PIN_ENABLED", "true").lower() == "true":
                auto_pin_recent(operator_id, username, conversation_history)
            pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
            for item in pinned_items:
                refresh_pinned_item(item, operator_id, conversation_history, run_query_and_chart, get_insight_engine)


def auto_pin_recent(
    operator_id: str,
    username: str,
    conversation_history,
) -> None:
    """根据用户设置自动收藏最近的有用图表。"""
    user_settings = conversation_history.get_user_settings(operator_id=operator_id, username=username)
    if not user_settings.get("auto_pin_enabled", True):
        return
    max_total = user_settings.get("max_pins", 6)
    max_auto = user_settings.get("auto_pin_limit", 3)
    recent_days = user_settings.get("auto_pin_days", 7)
    dedup_enabled = user_settings.get("dedup_enabled", True)
    dedup_threshold = user_settings.get("dedup_threshold", 0.70)
    dedup_mode = user_settings.get("dedup_mode", "rule").lower()
    require_helpful = user_settings.get("auto_pin_require_helpful", True)

    pinned_items = conversation_history.list_pinned(operator_id=operator_id, username=username)
    if len(pinned_items) >= max_total:
        return
    pinned_ids = {item.get("message_id") for item in pinned_items}

    existing_keys = [ _build_pinned_similarity_text(item) for item in pinned_items if _build_pinned_similarity_text(item) ]
    existing_embs: List[List[float]] = []
    if dedup_enabled and dedup_mode == "ai" and existing_keys:
        try:
            existing_embs = _get_text_embeddings(existing_keys)
        except Exception:
            dedup_mode = "rule"

    recent_messages = conversation_history.list_recent_messages(
        operator_id=operator_id, username=username, limit=200, since_days=recent_days,
    )
    candidates = []
    for msg in recent_messages:
        if msg.get("role") != "assistant" or not msg.get("chart") or not msg.get("sql"):
            continue
        if msg.get("id") in pinned_ids:
            continue
        if conversation_history.is_in_pin_blacklist(msg.get("id"), operator_id=operator_id, username=username):
            continue
        if require_helpful and msg.get("feedback_type") != "up":
            continue
        candidate_key = _build_pinned_similarity_text(msg)
        if candidate_key and dedup_enabled:
            if dedup_mode == "ai":
                try:
                    candidate_emb = _get_text_embeddings([candidate_key])[0]
                    if any(_cosine_similarity(candidate_emb, emb) >= dedup_threshold for emb in existing_embs):
                        continue
                except Exception:
                    if any(_similarity_ratio(candidate_key, k) >= dedup_threshold for k in existing_keys):
                        continue
            elif any(_similarity_ratio(candidate_key, k) >= dedup_threshold for k in existing_keys):
                continue
        elif candidate_key and candidate_key in existing_keys:
            continue
        candidates.append(msg)

    seen_keys = list(existing_keys)
    seen_embs = list(existing_embs)
    for msg in candidates[:max_auto]:
        if len(conversation_history.list_pinned(operator_id=operator_id, username=username)) >= max_total:
            break
        candidate_key = _build_pinned_similarity_text(msg)
        if candidate_key and dedup_enabled:
            if dedup_mode == "ai":
                try:
                    candidate_emb = _get_text_embeddings([candidate_key])[0]
                    if any(_cosine_similarity(candidate_emb, emb) >= dedup_threshold for emb in seen_embs):
                        continue
                except Exception:
                    if any(_similarity_ratio(candidate_key, k) >= dedup_threshold for k in seen_keys):
                        continue
            elif any(_similarity_ratio(candidate_key, k) >= dedup_threshold for k in seen_keys):
                continue
        elif candidate_key and candidate_key in seen_keys:
            continue
        try:
            payload = conversation_history.pin_message(msg["id"], operator_id=operator_id, username=username)
            # 自动 Pin 后智能选择一个标签（与手动 Pin 行为一致）
            try:
                label = recommend_chart_label(
                    question=payload.get("question") or "",
                    sql=payload.get("sql") or "",
                    columns=payload.get("columns"),
                    result=payload.get("result"),
                )
                if label:
                    label_dict = label.model_dump()
                    label_dict["updated_at"] = datetime.now().isoformat()
                    payload["label"] = label_dict
                    conversation_history.update_pinned(
                        msg["id"], payload, operator_id=operator_id, username=username
                    )
            except Exception as label_exc:
                print(f"[Analytics] 自动收藏后智能标签推荐失败: {label_exc}")
            if candidate_key:
                seen_keys.append(candidate_key)
                if dedup_enabled and dedup_mode == "ai":
                    try:
                        seen_embs.append(_get_text_embeddings([candidate_key])[0])
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[Analytics] 自动收藏失败: {exc}")
