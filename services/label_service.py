"""图表智能标签服务 - AI推荐与管理。"""
import os
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from core.schemas import (
    LabelType, AlertCategory, CelebrationCategory,
    ThresholdCondition, ThresholdConfig, ChartLabel
)


# 关键词映射表
ALERT_KEYWORDS = {
    AlertCategory.INVENTORY: [
        "库存", "存货", "stock", "inventory", "备货", "滞销", "缺货",
        "周转", "呆滞", "积压", "仓储", "出入库"
    ],
    AlertCategory.COST: [
        "成本", "费用", "cost", "expense", "支出", "开销", "预算",
        "亏损", "损失", "超支", "毛利", "利润率下降"
    ],
    AlertCategory.RISK: [
        "风险", "异常", "risk", "anomaly", "告警", "预警", "问题",
        "下滑", "下降", "减少", "流失", "退货", "投诉", "差评"
    ],
}

CELEBRATION_KEYWORDS = {
    CelebrationCategory.GMV: [
        "GMV", "销售额", "营收", "revenue", "sales", "销量", "订单金额",
        "成交额", "交易额", "收入"
    ],
    CelebrationCategory.GROWTH: [
        "增长", "growth", "上升", "提升", "增加", "同比", "环比",
        "新增", "拉新", "转化率", "复购"
    ],
    CelebrationCategory.PERFORMANCE: [
        "业绩", "performance", "达成", "完成", "目标", "KPI",
        "TOP", "排名", "冠军", "第一", "突破", "新高", "记录"
    ],
}

# SQL表名/字段名关键词映射
SQL_TABLE_KEYWORDS = {
    AlertCategory.INVENTORY: ["inventory", "stock", "warehouse", "sku"],
    AlertCategory.COST: ["cost", "expense", "budget", "fee"],
    AlertCategory.RISK: ["risk", "alert", "anomaly", "complaint"],
    CelebrationCategory.GMV: ["order", "sale", "revenue", "transaction"],
    CelebrationCategory.GROWTH: ["growth", "conversion", "retention"],
    CelebrationCategory.PERFORMANCE: ["performance", "target", "kpi", "achievement"],
}


class LabelRecommender:
    """AI标签推荐器"""

    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4.5")

    def recommend_label(
        self,
        question: str,
        sql: str = "",
        columns: Optional[List[str]] = None,
        result: Optional[List[Dict]] = None,
    ) -> Optional[ChartLabel]:
        """
        基于多维度分析推荐图表标签
        
        Args:
            question: 用户提问
            sql: 生成的SQL语句
            columns: 结果列名
            result: 查询结果数据
            
        Returns:
            推荐的ChartLabel，如果无法推荐则返回None
        """
        # 1. 基于规则的快速匹配
        rule_result = self._rule_based_recommend(question, sql, columns)
        if rule_result and rule_result[2] >= 0.7:  # 置信度足够高
            label_type, category, confidence = rule_result
            return self._build_label(label_type, category, confidence, result)

        # 2. 如果规则匹配置信度不够，尝试AI推荐
        if self.api_key:
            ai_result = self._ai_based_recommend(question, sql, columns)
            if ai_result:
                label_type, category, confidence = ai_result
                return self._build_label(label_type, category, confidence, result)

        # 3. 如果规则有结果但置信度不高，仍然返回
        if rule_result:
            label_type, category, confidence = rule_result
            return self._build_label(label_type, category, confidence, result)

        return None

    def _rule_based_recommend(
        self,
        question: str,
        sql: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Tuple[LabelType, str, float]]:
        """基于规则的标签推荐"""
        text = f"{question} {sql} {' '.join(columns or [])}".lower()
        
        alert_scores: Dict[str, float] = {}
        celebration_scores: Dict[str, float] = {}

        # 问题关键词匹配（权重高）
        for category, keywords in ALERT_KEYWORDS.items():
            score = sum(1.5 if kw.lower() in question.lower() else 0 for kw in keywords)
            score += sum(0.5 if kw.lower() in text else 0 for kw in keywords)
            if score > 0:
                alert_scores[category.value] = score

        for category, keywords in CELEBRATION_KEYWORDS.items():
            score = sum(1.5 if kw.lower() in question.lower() else 0 for kw in keywords)
            score += sum(0.5 if kw.lower() in text else 0 for kw in keywords)
            if score > 0:
                celebration_scores[category.value] = score

        # SQL表名匹配
        sql_lower = sql.lower() if sql else ""
        for category, keywords in SQL_TABLE_KEYWORDS.items():
            score = sum(0.8 if kw in sql_lower else 0 for kw in keywords)
            if score > 0:
                if isinstance(category, AlertCategory):
                    alert_scores[category.value] = alert_scores.get(category.value, 0) + score
                else:
                    celebration_scores[category.value] = celebration_scores.get(category.value, 0) + score

        # 选择得分最高的
        best_alert = max(alert_scores.items(), key=lambda x: x[1]) if alert_scores else (None, 0)
        best_celebration = max(celebration_scores.items(), key=lambda x: x[1]) if celebration_scores else (None, 0)

        if best_alert[1] > best_celebration[1] and best_alert[0]:
            confidence = min(best_alert[1] / 5.0, 1.0)  # 归一化到0-1
            return (LabelType.ALERT, best_alert[0], confidence)
        elif best_celebration[1] > 0 and best_celebration[0]:
            confidence = min(best_celebration[1] / 5.0, 1.0)
            return (LabelType.CELEBRATION, best_celebration[0], confidence)

        return None

    def _ai_based_recommend(
        self,
        question: str,
        sql: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Tuple[LabelType, str, float]]:
        """基于AI的标签推荐"""
        if not self.api_key:
            return None

        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=self.api_key)

            prompt = f"""你是一个数据分析专家，请分析以下查询并判断其业务类型。

用户问题: {question}
SQL语句: {sql}
数据列: {', '.join(columns or [])}

请判断这个查询属于以下哪种类型，返回JSON格式：

预警类 (alert):
- inventory: 库存相关（库存预警、缺货、滞销等）
- cost: 成本相关（费用、亏损、超支等）
- risk: 风险相关（异常、下滑、流失等）

喜报类 (celebration):
- gmv: GMV/销售额相关
- growth: 增长相关（同比增长、环比提升等）
- performance: 业绩相关（目标达成、排名、突破等）

返回格式（只返回JSON，不要其他内容）：
{{"type": "alert或celebration", "category": "具体分类", "confidence": 0.0-1.0}}

如果无法判断，返回：{{"type": null, "category": null, "confidence": 0}}
"""

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=100,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content.strip() if response.choices else ""
            
            # 解析JSON响应
            # 尝试提取JSON部分
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("type") and result.get("category"):
                    label_type = LabelType.ALERT if result["type"] == "alert" else LabelType.CELEBRATION
                    return (label_type, result["category"], result.get("confidence", 0.5))

        except Exception as exc:
            print(f"[LabelRecommender] AI推荐失败: {exc}")

        return None

    def _build_label(
        self,
        label_type: LabelType,
        category: str,
        confidence: float,
        result: Optional[List[Dict]] = None,
    ) -> ChartLabel:
        """构建标签对象"""
        # 根据标签类型推荐默认阈值
        threshold = self.recommend_threshold(label_type, category, result)

        return ChartLabel(
            type=label_type,
            category=category,
            ai_suggested=True,
            confidence=confidence,
            threshold=threshold,
            created_at=datetime.now().isoformat(),
        )

    def recommend_threshold(
        self,
        label_type: LabelType,
        category: str,
        result: Optional[List[Dict]] = None,
    ) -> ThresholdConfig:
        """推荐阈值配置"""
        # 预警类默认检测下降
        if label_type == LabelType.ALERT:
            condition = ThresholdCondition.DECREASE
            default_value = 15.0  # 下降15%触发预警
            
            # 根据分类调整
            if category == AlertCategory.INVENTORY.value:
                default_value = 20.0  # 库存下降20%
            elif category == AlertCategory.RISK.value:
                default_value = 10.0  # 风险类更敏感
        else:
            # 喜报类默认检测上升
            condition = ThresholdCondition.INCREASE
            default_value = 15.0  # 上升15%触发喜报
            
            if category == CelebrationCategory.GROWTH.value:
                default_value = 10.0  # 增长类更敏感
            elif category == CelebrationCategory.PERFORMANCE.value:
                default_value = 20.0  # 业绩达成要求高一些

        # 如果有历史数据，可以基于数据波动来调整阈值
        if result and len(result) >= 3:
            threshold_from_data = self._calculate_threshold_from_data(result)
            if threshold_from_data:
                default_value = threshold_from_data

        return ThresholdConfig(
            enabled=True,
            condition=condition,
            value=default_value,
            ai_recommended=True,
        )

    def _calculate_threshold_from_data(
        self,
        result: List[Dict],
    ) -> Optional[float]:
        """基于历史数据计算推荐阈值"""
        try:
            # 找到数值列
            numeric_values = []
            for row in result:
                for value in row.values():
                    if isinstance(value, (int, float)) and value > 0:
                        numeric_values.append(value)
                        break

            if len(numeric_values) < 3:
                return None

            # 计算变化率的标准差
            changes = []
            for i in range(1, len(numeric_values)):
                if numeric_values[i-1] != 0:
                    change = abs((numeric_values[i] - numeric_values[i-1]) / numeric_values[i-1]) * 100
                    changes.append(change)

            if changes:
                avg_change = sum(changes) / len(changes)
                # 阈值设为平均变化的1.5倍，最小10%，最大50%
                threshold = max(10.0, min(50.0, avg_change * 1.5))
                return round(threshold, 1)

        except Exception:
            pass

        return None


# 全局实例
_label_recommender: Optional[LabelRecommender] = None


def get_label_recommender() -> LabelRecommender:
    """获取标签推荐器实例"""
    global _label_recommender
    if _label_recommender is None:
        _label_recommender = LabelRecommender()
    return _label_recommender


def recommend_chart_label(
    question: str,
    sql: str = "",
    columns: Optional[List[str]] = None,
    result: Optional[List[Dict]] = None,
) -> Optional[ChartLabel]:
    """推荐图表标签的便捷函数"""
    return get_label_recommender().recommend_label(question, sql, columns, result)
