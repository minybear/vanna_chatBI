"""
Analytics manager for system metrics and usage tracking.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional


class SimpleEmbeddingFunction:
    """Simple embedding function for metadata storage."""

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


@dataclass
class SystemMetrics:
    data_freshness: str
    ai_accuracy: float
    avg_latency_ms: int
    total_queries_today: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_freshness": self.data_freshness,
            "ai_accuracy": self.ai_accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "total_queries_today": self.total_queries_today,
        }


class AnalyticsManager:
    def __init__(self, chroma_client):
        self.collection = chroma_client.get_or_create_collection(
            name="metrics_events",
            embedding_function=SimpleEmbeddingFunction(),
        )

    def record_query(
        self,
        latency_ms: int,
        operator_id: str = "",
        datasource_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> None:
        payload = {
            "type": "query",
            "timestamp": datetime.now().isoformat(),
            "latency_ms": int(latency_ms) if latency_ms is not None else 0,
            "operator_id": operator_id or "",
            "datasource_id": datasource_id or "",
            "sql": sql or "",
        }
        self._add_event(payload, operator_id)

    def record_feedback(self, feedback_type: str, operator_id: str = "") -> None:
        payload = {
            "type": "feedback",
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "operator_id": operator_id or "",
        }
        self._add_event(payload, operator_id)

    def record_dashboard_refresh(self, operator_id: str = "") -> None:
        """记录看板图表刷新时间，用于洞察概述的「数据更新」展示。"""
        payload = {
            "type": "dashboard_refresh",
            "timestamp": datetime.now().isoformat(),
            "operator_id": operator_id or "",
        }
        self._add_event(payload, operator_id)

    def get_metrics(self, operator_id: str = "") -> SystemMetrics:
        events = self._get_events(operator_id)
        query_events = [e for e in events if e.get("type") == "query"]
        feedback_events = [e for e in events if e.get("type") == "feedback"]
        dashboard_refresh_events = [e for e in events if e.get("type") == "dashboard_refresh"]

        last_query_time = None
        last_dashboard_refresh_time = None
        latencies = []
        total_queries_today = 0
        today = datetime.now().date()

        for event in query_events:
            ts = self._parse_time(event.get("timestamp"))
            if ts:
                if last_query_time is None or ts > last_query_time:
                    last_query_time = ts
                if ts.date() == today:
                    total_queries_today += 1
            latency = event.get("latency_ms")
            if isinstance(latency, int):
                latencies.append(latency)

        for event in dashboard_refresh_events:
            ts = self._parse_time(event.get("timestamp"))
            if ts and (last_dashboard_refresh_time is None or ts > last_dashboard_refresh_time):
                last_dashboard_refresh_time = ts

        up_count = sum(1 for e in feedback_events if e.get("feedback_type") == "up")
        down_count = sum(1 for e in feedback_events if e.get("feedback_type") == "down")
        total_feedback = up_count + down_count
        ai_accuracy = round((up_count / total_feedback) * 100, 2) if total_feedback else 0.0
        avg_latency = int(median(latencies)) if latencies else 0

        # 数据更新：优先使用看板图表最后刷新时间，无则用最后查询时间
        freshness_ts = last_dashboard_refresh_time or last_query_time
        data_freshness = self._humanize_delta(freshness_ts) if freshness_ts else "暂无数据"

        return SystemMetrics(
            data_freshness=data_freshness,
            ai_accuracy=ai_accuracy,
            avg_latency_ms=avg_latency,
            total_queries_today=total_queries_today,
        )

    def _add_event(self, payload: Dict[str, Any], operator_id: str) -> None:
        event_id = f"{payload.get('type')}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.collection.add(
            ids=[event_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[
                {
                    "type": payload.get("type"),
                    "operator_id": operator_id or "",
                }
            ],
        )

    def _get_events(self, operator_id: str) -> List[Dict[str, Any]]:
        if operator_id:
            data = self.collection.get(where={"operator_id": operator_id})
        else:
            data = self.collection.get()
        documents = data.get("documents") if data else []
        events: List[Dict[str, Any]] = []
        for doc in documents or []:
            try:
                events.append(json.loads(doc))
            except Exception:
                continue
        return events

    def clear_events(self, operator_id: str = "") -> int:
        if operator_id:
            data = self.collection.get(where={"operator_id": operator_id})
        else:
            data = self.collection.get()
        ids = data.get("ids") if data else []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def _parse_time(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _humanize_delta(self, ts: datetime) -> str:
        now = datetime.now()
        delta = now - ts
        if delta < timedelta(minutes=1):
            return "刚刚"
        if delta < timedelta(hours=1):
            return f"{int(delta.total_seconds() // 60)}分钟前"
        if delta < timedelta(days=1):
            return f"{int(delta.total_seconds() // 3600)}小时前"
        return f"{int(delta.total_seconds() // 86400)}天前"


_analytics_manager: Optional[AnalyticsManager] = None


def init_analytics_manager(chroma_client) -> AnalyticsManager:
    global _analytics_manager
    _analytics_manager = AnalyticsManager(chroma_client)
    return _analytics_manager


def get_analytics_manager() -> AnalyticsManager:
    if _analytics_manager is None:
        raise RuntimeError("AnalyticsManager 未初始化")
    return _analytics_manager
