# API 路由：认证、对话、分析、语音识别、杂项

from routers.auth_routes import router as auth_router
from routers.chat_routes import router as chat_router
from routers.analytics_routes import router as analytics_router
from routers.speech_routes import router as speech_router
from routers.misc_routes import router as misc_router, api_router

__all__ = [
    "auth_router",
    "chat_router", 
    "analytics_router",
    "speech_router",
    "misc_router",
    "api_router",
]
