"""预警与喜报检测引擎。"""
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

from core.schemas import (
    LabelType, ThresholdCondition, ThresholdConfig, ChartLabel,
    AlertSeverity, AlertResult, CelebrationResult
)


class AlertCelebrationEngine:
    """预警与喜报检测引擎"""

    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4.5")

    def check_alert(
        self,
        pinned_item: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None,
    ) -> AlertResult:
        """
        检测预警条件
        
        Args:
            pinned_item: 看板图表数据
            historical_data: 历史数据（用于比较）
            
        Returns:
            AlertResult 预警检测结果
        """
        label_data = pinned_item.get("label")
        if not label_data:
            return AlertResult(triggered=False)

        # 解析标签
        label = self._parse_label(label_data)
        if not label or label.type != LabelType.ALERT:
            return AlertResult(triggered=False)

        # 检查阈值是否启用
        threshold = label.threshold
        if not threshold or not threshold.enabled:
            return AlertResult(triggered=False)

        # 获取当前数据
        result = pinned_item.get("result", [])
        columns = pinned_item.get("columns", [])
        if not result or not columns:
            return AlertResult(triggered=False)

        # 计算变化（库存类优先用 remaining 相关列，便于检测剩余量骤降）
        prefer_col = "remaining" if (label.category or "").lower() == "inventory" else None
        change_info = self._calculate_change(result, columns, historical_data, prefer_column_substring=prefer_col)
        if not change_info:
            return AlertResult(triggered=False)

        current_value, previous_value, change_ratio = change_info

        # 检查是否触发阈值
        triggered = self._check_threshold(change_ratio, threshold)
        if not triggered:
            return AlertResult(triggered=False)

        # 确定严重程度
        severity = self._determine_severity(change_ratio, threshold)

        # 生成预警消息
        question = pinned_item.get("question", "数据指标")
        title = f"{self._get_category_name(label.category)}预警：{question[:20]}"
        message = self._generate_alert_message(
            question, label.category, current_value, previous_value, change_ratio
        )
        suggestion = self._generate_suggestion(label.category, change_ratio)

        return AlertResult(
            triggered=True,
            severity=severity,
            title=title,
            message=message,
            metrics={
                "current": current_value,
                "previous": previous_value,
                "change_ratio": round(change_ratio * 100, 2),
                "threshold": threshold.value,
            },
            suggestion=suggestion,
            chart_id=pinned_item.get("message_id"),
        )

    def check_celebration(
        self,
        pinned_item: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None,
    ) -> CelebrationResult:
        """
        检测喜报条件
        
        Args:
            pinned_item: 看板图表数据
            historical_data: 历史数据
            
        Returns:
            CelebrationResult 喜报检测结果
        """
        label_data = pinned_item.get("label")
        if not label_data:
            return CelebrationResult(triggered=False)

        label = self._parse_label(label_data)
        if not label or label.type != LabelType.CELEBRATION:
            return CelebrationResult(triggered=False)

        threshold = label.threshold
        if not threshold or not threshold.enabled:
            return CelebrationResult(triggered=False)

        result = pinned_item.get("result", [])
        columns = pinned_item.get("columns", [])
        if not result or not columns:
            return CelebrationResult(triggered=False)

        change_info = self._calculate_change(result, columns, historical_data)
        if not change_info:
            return CelebrationResult(triggered=False)

        current_value, previous_value, change_ratio = change_info

        triggered = self._check_threshold(change_ratio, threshold)
        if not triggered:
            return CelebrationResult(triggered=False)

        question = pinned_item.get("question", "业务指标")
        title = f"{self._get_category_name(label.category)}喜报：{question[:20]}"
        achievement = f"{'环比' if not historical_data else ''}增长 {abs(change_ratio) * 100:.1f}%"
        highlights = self._generate_highlights(result, columns, change_ratio)

        return CelebrationResult(
            triggered=True,
            title=title,
            achievement=achievement,
            highlights=highlights,
            metrics={
                "current": current_value,
                "previous": previous_value,
                "change_ratio": round(change_ratio * 100, 2),
                "threshold": threshold.value,
            },
            chart_id=pinned_item.get("message_id"),
        )

    def batch_check(
        self,
        pinned_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量检测所有看板图表
        
        Returns:
            {
                "alerts": [AlertResult, ...],
                "celebrations": [CelebrationResult, ...],
            }
        """
        alerts = []
        celebrations = []

        for item in pinned_items:
            alert_result = self.check_alert(item)
            if alert_result.triggered:
                alerts.append(alert_result)

            celebration_result = self.check_celebration(item)
            if celebration_result.triggered:
                celebrations.append(celebration_result)

        return {
            "alerts": alerts,
            "celebrations": celebrations,
        }

    def _parse_label(self, label_data: Any) -> Optional[ChartLabel]:
        """解析标签数据"""
        if isinstance(label_data, ChartLabel):
            return label_data
        if isinstance(label_data, dict):
            try:
                return ChartLabel(**label_data)
            except Exception:
                return None
        return None

    def _calculate_change(
        self,
        result: List[Dict],
        columns: List[str],
        historical_data: Optional[List[Dict]] = None,
        prefer_column_substring: Optional[str] = None,
    ) -> Optional[tuple]:
        """计算数据变化。prefer_column_substring 用于库存类优先匹配 remaining 等列名。"""
        try:
            # 找到数值列（若指定优先关键字，先找名称包含该关键字的数值列）
            numeric_col = None
            if prefer_column_substring:
                key = prefer_column_substring.lower()
                for col in columns:
                    if key in col.lower():
                        sample_values = [row.get(col) for row in result[:5] if row.get(col) is not None]
                        if sample_values and all(isinstance(v, (int, float)) for v in sample_values):
                            numeric_col = col
                            break
            if not numeric_col:
                for col in columns:
                    sample_values = [row.get(col) for row in result[:5] if row.get(col) is not None]
                    if sample_values and all(isinstance(v, (int, float)) for v in sample_values):
                        numeric_col = col
                        break

            if not numeric_col:
                return None

            values = [row.get(numeric_col) for row in result if isinstance(row.get(numeric_col), (int, float))]
            if len(values) < 2:
                return None

            # 当前值取最后一个，前值取平均或倒数第二个
            current_value = values[-1]
            if historical_data:
                # 使用历史数据计算前值
                hist_values = [row.get(numeric_col) for row in historical_data 
                               if isinstance(row.get(numeric_col), (int, float))]
                previous_value = sum(hist_values) / len(hist_values) if hist_values else values[-2]
            else:
                # 使用当前数据的前N个值的平均
                previous_values = values[:-1]
                previous_value = sum(previous_values) / len(previous_values)

            if previous_value == 0:
                return None

            change_ratio = (current_value - previous_value) / abs(previous_value)
            return (current_value, previous_value, change_ratio)

        except Exception as e:
            print(f"[AlertEngine] 计算变化失败: {e}")
            return None

    def _check_threshold(
        self,
        change_ratio: float,
        threshold: ThresholdConfig,
    ) -> bool:
        """检查是否触发阈值"""
        threshold_value = threshold.value / 100.0  # 转换为小数

        if threshold.condition == ThresholdCondition.DECREASE:
            return change_ratio <= -threshold_value
        elif threshold.condition == ThresholdCondition.INCREASE:
            return change_ratio >= threshold_value
        elif threshold.condition == ThresholdCondition.ABOVE:
            return change_ratio >= threshold_value
        elif threshold.condition == ThresholdCondition.BELOW:
            return change_ratio <= -threshold_value

        return False

    def _determine_severity(
        self,
        change_ratio: float,
        threshold: ThresholdConfig,
    ) -> AlertSeverity:
        """确定预警严重程度"""
        abs_change = abs(change_ratio) * 100
        threshold_value = threshold.value

        if abs_change >= threshold_value * 2:
            return AlertSeverity.CRITICAL
        elif abs_change >= threshold_value * 1.5:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _get_category_name(self, category: str) -> str:
        """获取分类的中文名称"""
        category_names = {
            "inventory": "库存",
            "cost": "成本",
            "risk": "风险",
            "gmv": "销售额",
            "growth": "增长",
            "performance": "业绩",
        }
        return category_names.get(category, category)

    def _generate_alert_message(
        self,
        question: str,
        category: str,
        current: float,
        previous: float,
        change_ratio: float,
    ) -> str:
        """生成预警消息"""
        direction = "下降" if change_ratio < 0 else "上升"
        return (
            f"{question} 出现{direction}，"
            f"当前值 {current:.2f}，较之前均值 {previous:.2f} "
            f"{direction}了 {abs(change_ratio) * 100:.1f}%"
        )

    def _generate_suggestion(self, category: str, change_ratio: float) -> str:
        """生成建议"""
        suggestions = {
            "inventory": "建议检查库存周转情况，及时补货或调整采购计划",
            "cost": "建议审查成本结构，识别异常支出项目",
            "risk": "建议及时排查问题原因，制定应对措施",
        }
        return suggestions.get(category, "建议关注数据变化，及时采取措施")

    def _generate_highlights(
        self,
        result: List[Dict],
        columns: List[str],
        change_ratio: float,
    ) -> List[str]:
        """生成喜报亮点"""
        highlights = []
        highlights.append(f"数据较前期增长 {abs(change_ratio) * 100:.1f}%")

        # 尝试提取更多亮点
        if len(result) >= 3:
            highlights.append(f"数据呈现持续上升趋势")

        return highlights


# 全局实例
_alert_engine: Optional[AlertCelebrationEngine] = None


def get_alert_engine() -> AlertCelebrationEngine:
    """获取预警引擎实例"""
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertCelebrationEngine()
    return _alert_engine


def check_pinned_alerts(
    pinned_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """检测看板图表的预警和喜报"""
    return get_alert_engine().batch_check(pinned_items)
