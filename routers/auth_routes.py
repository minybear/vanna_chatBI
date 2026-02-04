"""认证与配置接口。"""
from fastapi import APIRouter, HTTPException

from core.config import AUTH_TOKEN_TTL_SECONDS
from core.auth import _authenticate_user, _issue_token, AUTH_ENABLED
from core.schemas import LoginRequest

router = APIRouter()


@router.post("/login")
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


@router.get("/config")
def get_config():
    import os
    raw_value = os.getenv("SHOW_SQL_DEBUG", "False")
    show_sql = raw_value.lower() == "true"
    
    # 语音输入开关
    speech_enabled = os.getenv("SPEECH_ENABLED", "false").lower() == "true"
    
    return {
        "show_sql": show_sql,
        "speech_enabled": speech_enabled,
    }
