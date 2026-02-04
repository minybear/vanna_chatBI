"""API 请求/响应模型。"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ============ 标签系统相关模型 ============

class LabelType(str, Enum):
    """标签主类型：预警类 / 喜报类"""
    ALERT = "alert"          # 预警类
    CELEBRATION = "celebration"  # 喜报类


class AlertCategory(str, Enum):
    """预警类标签子分类"""
    INVENTORY = "inventory"  # 库存预警
    COST = "cost"            # 成本预警
    RISK = "risk"            # 风险预警


class CelebrationCategory(str, Enum):
    """喜报类标签子分类"""
    GMV = "gmv"              # GMV/销售额
    GROWTH = "growth"        # 业务增长
    PERFORMANCE = "performance"  # 业绩达成


class ThresholdCondition(str, Enum):
    """阈值条件类型"""
    INCREASE = "increase"    # 上升触发
    DECREASE = "decrease"    # 下降触发
    ABOVE = "above"          # 高于阈值
    BELOW = "below"          # 低于阈值


class ThresholdConfig(BaseModel):
    """阈值配置"""
    enabled: bool = True
    condition: ThresholdCondition = ThresholdCondition.DECREASE
    value: float = 15.0      # 百分比阈值
    ai_recommended: bool = True  # 是否为AI推荐的阈值


class ChartLabel(BaseModel):
    """图表标签"""
    type: LabelType                     # 预警类 / 喜报类
    category: str                       # 具体分类
    ai_suggested: bool = True           # 是否为AI推荐
    confidence: float = 0.0             # AI推荐置信度
    threshold: Optional[ThresholdConfig] = None  # 阈值配置
    fatigue_cooldown_hours: Optional[float] = None  # 单图表疲劳冷却（小时），None=用全局，0=不限制，>0=该值
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class LabelUpdateRequest(BaseModel):
    """标签更新请求"""
    message_id: str
    label: ChartLabel


class ThresholdUpdateRequest(BaseModel):
    """阈值更新请求"""
    message_id: str
    threshold: ThresholdConfig


# ============ 飞书群管理相关模型 ============

class LarkGroup(BaseModel):
    """飞书群配置"""
    id: str
    name: str
    webhook_url: str
    description: Optional[str] = None
    group_type: Optional[str] = None  # alert / celebration / general
    chat_id: Optional[str] = None  # 群聊 ID，用于发送附件消息（应用机器人必填）
    created_at: Optional[str] = None
    operator_id: Optional[str] = None


class LarkGroupCreateRequest(BaseModel):
    """创建飞书群配置请求"""
    name: str
    webhook_url: str
    description: Optional[str] = None
    group_type: str = "general"
    chat_id: Optional[str] = None  # 群聊 ID，发送附件时必填


class SendReportRequest(BaseModel):
    """发送报告到飞书请求"""
    group_id: str
    report_content: str
    report_type: str = "insight"  # insight / alert / celebration
    chart_images: Optional[List[str]] = None  # base64 或 data URL 图片列表
    as_rich_text: bool = False  # True=以富文本消息发送（支持图片内嵌，需配置 chat_id 与应用凭证）


# ============ 预警/喜报相关模型 ============

class AlertSeverity(str, Enum):
    """预警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertResult(BaseModel):
    """预警检测结果"""
    triggered: bool = False
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""
    metrics: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    chart_id: Optional[str] = None


class CelebrationResult(BaseModel):
    """喜报检测结果"""
    triggered: bool = False
    title: str = ""
    achievement: str = ""
    highlights: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    chart_id: Optional[str] = None


# ============ 语音识别相关模型 ============

class SpeechRecognizeRequest(BaseModel):
    """语音识别请求（用于流式识别的元数据）"""
    format: str = "pcm"       # 音频格式：pcm, wav, mp3
    sample_rate: int = 16000  # 采样率
    language: str = "zh_cn"   # 语言


class SpeechRecognizeResponse(BaseModel):
    """语音识别响应"""
    success: bool
    text: str = ""
    confidence: float = 0.0
    error: Optional[str] = None


# ============ 原有模型 ============

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    datasource_id: Optional[str] = None
    kb_id: Optional[str] = None


class SqlRequest(BaseModel):
    sql: str
    question: str = None
    datasource_id: Optional[str] = None


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
