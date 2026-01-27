"""
Insight engine for proactive analytics.
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime
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
class Insight:
    insight_type: str
    title: str
    message: str
    severity: str
    category: str
    confidence: float
    created_at: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "data": self.data or {},
        }


class InsightEngine:
    def __init__(self, chroma_client):
        self.collection = chroma_client.get_or_create_collection(
            name="insights",
            embedding_function=SimpleEmbeddingFunction(),
        )

    def list_insights(self, operator_id: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        if operator_id:
            data = self.collection.get(where={"operator_id": operator_id})
        else:
            data = self.collection.get()
        documents = data.get("documents") if data else []
        insights: List[Dict[str, Any]] = []
        for doc in documents or []:
            try:
                insights.append(json.loads(doc))
            except Exception:
                continue
        insights.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return insights[:limit]

    def save_insight(
        self,
        insight_type: str,
        title: str,
        message: str,
        severity: str = "info",
        category: str = "trend",
        confidence: float = 0.5,
        data: Optional[Dict[str, Any]] = None,
        operator_id: str = "",
    ) -> None:
        data = data or {}
        llm_message = self._generate_llm_summary(title, message)
        if llm_message:
            data = {**data, "rule_message": message}
            message = llm_message
        
        # 去重逻辑: 检查最近24小时内是否有相同标题和类型的洞察
        existing_insight_id = self._find_duplicate_insight(
            title=title,
            insight_type=insight_type,
            operator_id=operator_id,
            hours=24
        )
        
        if existing_insight_id:
            # 更新现有洞察
            self._update_insight(
                insight_id=existing_insight_id,
                message=message,
                severity=severity,
                category=category,
                confidence=confidence,
                data=data,
            )
            return
        
        # 创建新洞察
        insight_id = f"{insight_type}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        payload = Insight(
            insight_type=insight_type,
            title=title,
            message=message,
            severity=severity,
            category=category,
            confidence=confidence,
            created_at=datetime.now().isoformat(),
            data=data,
        ).to_dict()
        payload["id"] = insight_id

        self.collection.add(
            ids=[insight_id],
            documents=[json.dumps(payload, ensure_ascii=False)],
            metadatas=[{"operator_id": operator_id or "", "insight_type": insight_type}],
        )

    def delete_insight(self, insight_id: str, operator_id: str = "") -> bool:
        if not insight_id:
            return False
        data = self.collection.get(ids=[insight_id])
        if not data or not data.get("ids"):
            return False
        meta = (data.get("metadatas") or [{}])[0]
        if operator_id:
            owner_id = meta.get("operator_id", "")
            if owner_id and owner_id != operator_id:
                return False
        self.collection.delete(ids=[insight_id])
        return True

    def clear_insights(self, operator_id: str = "") -> int:
        if operator_id:
            data = self.collection.get(where={"operator_id": operator_id})
        else:
            data = self.collection.get()
        ids = data.get("ids") if data else []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def _find_duplicate_insight(
        self,
        title: str,
        insight_type: str,
        operator_id: str = "",
        hours: int = 24
    ) -> Optional[str]:
        """查找指定时间窗口内是否存在相同标题和类型的洞察"""
        from datetime import timedelta
        
        if operator_id:
            data = self.collection.get(where={"operator_id": operator_id})
        else:
            data = self.collection.get()
        
        if not data or not data.get("documents"):
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for doc in data.get("documents", []):
            try:
                insight = json.loads(doc)
                # 检查标题、类型和时间
                if (insight.get("title") == title and 
                    insight.get("insight_type") == insight_type):
                    created_at = insight.get("created_at", "")
                    if created_at:
                        insight_time = datetime.fromisoformat(created_at)
                        if insight_time >= cutoff_time:
                            return insight.get("id")
            except Exception:
                continue
        
        return None
    
    def _update_insight(
        self,
        insight_id: str,
        message: str,
        severity: str,
        category: str,
        confidence: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新现有洞察"""
        try:
            existing_data = self.collection.get(ids=[insight_id])
            if not existing_data or not existing_data.get("documents"):
                return
            
            existing_doc = existing_data.get("documents", [])[0]
            insight = json.loads(existing_doc)
            
            # 更新字段
            insight["message"] = message
            insight["severity"] = severity
            insight["category"] = category
            insight["confidence"] = confidence
            insight["data"] = data or {}
            insight["updated_at"] = datetime.now().isoformat()
            
            # 删除旧记录
            self.collection.delete(ids=[insight_id])
            
            # 添加更新后的记录
            metadata = existing_data.get("metadatas", [{}])[0]
            self.collection.add(
                ids=[insight_id],
                documents=[json.dumps(insight, ensure_ascii=False)],
                metadatas=[metadata],
            )
        except Exception as exc:
            print(f"[InsightEngine] 更新洞察失败: {exc}")

    def _generate_llm_summary(self, title: str, message: str) -> Optional[str]:
        api_key = os.getenv("ZHIPU_API_KEY")
        model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4.5")
        if not api_key or not model:
            return None
        try:
            from zhipuai import ZhipuAI
        except Exception as exc:
            print(f"[InsightEngine] ZhipuAI SDK 未安装: {exc}")
            return None

        client = ZhipuAI(api_key=api_key)
        prompt = (
            "你是业务分析助手，请用一句简洁中文总结洞察，避免重复原句。\n"
            f"标题：{title}\n"
            f"洞察：{message}\n"
            "输出要求：不超过30字。"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=100,
                temperature=0.3,
                top_p=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content if response.choices else ""
            return content.strip() if content else None
        except Exception as exc:
            print(f"[InsightEngine] 生成洞察摘要失败: {exc}")
            return None

    def generate_insights_from_result(
        self,
        result_rows: List[Dict[str, Any]],
        columns: List[str],
        title_hint: str,
        sql: Optional[str] = None,
        chart: Optional[str] = None,
        operator_id: str = "",
    ) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []
        if not result_rows or not columns:
            return insights

        numeric_columns = []
        for col in columns:
            values = [row.get(col) for row in result_rows if isinstance(row.get(col), (int, float))]
            if len(values) >= max(2, len(result_rows) // 2):
                numeric_columns.append(col)

        if not numeric_columns:
            return insights

        target_col = numeric_columns[0]
        values = [row.get(target_col) for row in result_rows if isinstance(row.get(target_col), (int, float))]
        if len(values) < 3:
            return insights

        last_value = values[-1]
        previous_values = values[:-1]
        avg_value = sum(previous_values) / len(previous_values)
        if avg_value == 0:
            return insights

        change_ratio = (last_value - avg_value) / abs(avg_value)
        def _confidence_from_ratio(ratio: float) -> float:
            ratio = abs(ratio)
            if ratio >= 0.5:
                return 0.92
            if ratio >= 0.3:
                return 0.85
            if ratio >= 0.2:
                return 0.78
            return 0.68

        evidence = {
            "column": target_col,
            "last_value": last_value,
            "avg_value": avg_value,
            "change_ratio": change_ratio,
            "sql": sql,
            "chart": chart,
            "sample_rows": result_rows[:5],
        }

        if change_ratio <= -0.15:
            message = f"{title_hint} 出现明显下降，最近值较均值下降 {abs(change_ratio) * 100:.1f}%"
            self.save_insight(
                insight_type="anomaly",
                title=title_hint,
                message=message,
                severity="warning",
                category="anomaly",
                confidence=_confidence_from_ratio(change_ratio),
                data=evidence,
                operator_id=operator_id,
            )
            insights.append({"title": title_hint, "message": message, "severity": "warning"})
        elif change_ratio >= 0.15:
            message = f"{title_hint} 持续上升，最近值高于均值 {change_ratio * 100:.1f}%"
            self.save_insight(
                insight_type="trend_up",
                title=title_hint,
                message=message,
                severity="info",
                category="opportunity",
                confidence=_confidence_from_ratio(change_ratio),
                data=evidence,
                operator_id=operator_id,
            )
            insights.append({"title": title_hint, "message": message, "severity": "info"})

        if len(values) >= 3 and values[-3] < values[-2] < values[-1]:
            message = f"{title_hint} 最近三期呈现连续增长趋势"
            self.save_insight(
                insight_type="trend",
                title=title_hint,
                message=message,
                severity="info",
                category="trend",
                confidence=0.72,
                data=evidence,
                operator_id=operator_id,
            )
            insights.append({"title": title_hint, "message": message, "severity": "info"})

        return insights


_insight_engine: Optional[InsightEngine] = None


def init_insight_engine(chroma_client) -> InsightEngine:
    global _insight_engine
    _insight_engine = InsightEngine(chroma_client)
    return _insight_engine


def get_insight_engine() -> InsightEngine:
    if _insight_engine is None:
        raise RuntimeError("InsightEngine 未初始化")
    return _insight_engine
