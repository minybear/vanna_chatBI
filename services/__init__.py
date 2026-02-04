# 业务服务：飞书通知、查询与图表、收藏刷新、标签、预警等

# 标签服务
from services.label_service import (
    LabelRecommender,
    get_label_recommender,
    recommend_chart_label,
)

# 预警喜报引擎
from services.alert_celebration_engine import (
    AlertCelebrationEngine,
    get_alert_engine,
    check_pinned_alerts,
)

# 飞书群管理
from services.lark_group_manager import (
    LarkGroupManager,
    LarkReportSender,
    init_lark_services,
    get_lark_group_manager,
    get_lark_report_sender,
)

# 语音识别服务
from services.speech_service import (
    SimpleSpeechService,
    get_speech_service,
    is_speech_service_configured,
)

__all__ = [
    # 标签服务
    "LabelRecommender",
    "get_label_recommender",
    "recommend_chart_label",
    # 预警喜报
    "AlertCelebrationEngine",
    "get_alert_engine",
    "check_pinned_alerts",
    # 飞书
    "LarkGroupManager",
    "LarkReportSender",
    "init_lark_services",
    "get_lark_group_manager",
    "get_lark_report_sender",
    # 语音
    "SimpleSpeechService",
    "get_speech_service",
    "is_speech_service_configured",
]
