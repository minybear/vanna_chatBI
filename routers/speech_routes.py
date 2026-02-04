"""语音识别 API 路由。"""
import os
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form

from core.auth import get_current_user, require_operator_id
from services.speech_service import get_speech_service, is_speech_service_configured

router = APIRouter()


@router.get("/speech/status")
def get_speech_service_status():
    """获取语音识别服务状态"""
    configured = is_speech_service_configured()
    return {
        "configured": configured,
        "provider": "xunfei" if configured else None,
        "message": "语音识别服务已配置" if configured else "语音识别服务未配置，请设置科大讯飞API密钥",
    }


@router.post("/speech/recognize")
async def recognize_speech(
    audio: UploadFile = File(...),
    format: str = Form(default="pcm"),
    sample_rate: int = Form(default=16000),
    language: str = Form(default="zh_cn"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    识别上传的音频文件
    
    Args:
        audio: 音频文件
        format: 音频格式 (pcm, wav, mp3)
        sample_rate: 采样率 (默认16000)
        language: 语言 (zh_cn, en_us)
    
    Returns:
        {"success": bool, "text": str, "confidence": float, "error": str}
    """
    require_operator_id(current_user)
    
    service = get_speech_service()
    if not service.is_configured():
        return {
            "success": False,
            "text": "",
            "error": "语音识别服务未配置，请联系管理员设置科大讯飞API密钥",
        }

    try:
        # 读取音频数据
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            return {
                "success": False,
                "text": "",
                "error": "音频文件为空",
            }

        # 限制文件大小 (最大10MB)
        max_size = 10 * 1024 * 1024
        if len(audio_data) > max_size:
            return {
                "success": False,
                "text": "",
                "error": "音频文件过大，请限制在10MB以内",
            }

        # 调用语音识别服务
        result = await service.recognize_async(
            audio_data=audio_data,
            audio_format=format,
            sample_rate=sample_rate,
            language=language,
        )

        return {
            "success": result.success,
            "text": result.text,
            "confidence": result.confidence,
            "error": result.error,
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"语音识别失败: {str(e)}",
        }


@router.post("/speech/recognize-base64")
async def recognize_speech_base64(
    request_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    识别 Base64 编码的音频数据
    
    Request Body:
        {
            "audio": "base64编码的音频数据",
            "format": "pcm",
            "sample_rate": 16000,
            "language": "zh_cn"
        }
    """
    import base64
    
    require_operator_id(current_user)
    
    service = get_speech_service()
    if not service.is_configured():
        return {
            "success": False,
            "text": "",
            "error": "语音识别服务未配置",
        }

    try:
        audio_base64 = request_data.get("audio", "")
        if not audio_base64:
            return {
                "success": False,
                "text": "",
                "error": "缺少音频数据",
            }

        # 解码 base64
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception:
            return {
                "success": False,
                "text": "",
                "error": "音频数据格式错误，请确保是有效的Base64编码",
            }

        format = request_data.get("format", "pcm")
        sample_rate = request_data.get("sample_rate", 16000)
        language = request_data.get("language", "zh_cn")

        result = await service.recognize_async(
            audio_data=audio_data,
            audio_format=format,
            sample_rate=sample_rate,
            language=language,
        )

        return {
            "success": result.success,
            "text": result.text,
            "confidence": result.confidence,
            "error": result.error,
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"语音识别失败: {str(e)}",
        }
