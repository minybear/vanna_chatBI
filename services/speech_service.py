"""科大讯飞语音识别服务。"""
import os
import json
import time
import base64
import hashlib
import hmac
import asyncio
from datetime import datetime
from urllib.parse import urlencode
from typing import Optional, Dict, Any

from core.schemas import SpeechRecognizeResponse


class XunfeiSpeechService:
    """科大讯飞语音识别服务"""

    # 讯飞语音识别API地址
    API_URL = "wss://iat-api.xfyun.cn/v2/iat"
    
    def __init__(
        self,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.app_id = app_id or os.getenv("XUNFEI_APP_ID")
        self.api_key = api_key or os.getenv("XUNFEI_API_KEY")
        self.api_secret = api_secret or os.getenv("XUNFEI_API_SECRET")

    def is_configured(self) -> bool:
        """检查是否已配置且启用"""
        # 检查是否启用语音功能
        speech_enabled = os.getenv("SPEECH_ENABLED", "false").lower() == "true"
        if not speech_enabled:
            return False
        return bool(self.app_id and self.api_key and self.api_secret)

    def _create_auth_url(self) -> str:
        """创建鉴权URL"""
        from datetime import timezone
        import urllib.parse

        # 生成RFC1123格式的时间戳
        now = datetime.now(timezone.utc)
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')

        # 拼接签名原文
        signature_origin = f"host: iat-api.xfyun.cn\ndate: {date}\nGET /v2/iat HTTP/1.1"

        # 使用hmac-sha256进行签名
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature = base64.b64encode(signature_sha).decode('utf-8')

        # 拼接authorization
        authorization_origin = (
            f'api_key="{self.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

        # 拼接url
        params = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn"
        }
        return f"{self.API_URL}?{urlencode(params)}"

    async def recognize_audio(
        self,
        audio_data: bytes,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        language: str = "zh_cn",
    ) -> SpeechRecognizeResponse:
        """
        识别音频数据
        
        Args:
            audio_data: 音频二进制数据
            audio_format: 音频格式 (pcm, wav, mp3)
            sample_rate: 采样率
            language: 语言 (zh_cn, en_us)
            
        Returns:
            SpeechRecognizeResponse
        """
        if not self.is_configured():
            return SpeechRecognizeResponse(
                success=False,
                error="科大讯飞语音服务未配置，请设置 XUNFEI_APP_ID, XUNFEI_API_KEY, XUNFEI_API_SECRET"
            )

        try:
            import websockets
        except ImportError:
            return SpeechRecognizeResponse(
                success=False,
                error="请安装 websockets 库: pip install websockets"
            )

        try:
            auth_url = self._create_auth_url()
            result_text = ""
            
            async with websockets.connect(auth_url) as ws:
                # 发送音频数据
                frame_size = 1280  # 每帧大小
                interval = 0.04   # 发送间隔

                # 构建业务参数
                common_args = {"app_id": self.app_id}
                business_args = {
                    "language": language,
                    "domain": "iat",
                    "accent": "mandarin" if language == "zh_cn" else "english",
                    "vad_eos": 3000,  # 静音检测时间
                    "dwa": "wpgs",    # 动态修正
                }
                
                # 发送第一帧
                first_frame = {
                    "common": common_args,
                    "business": business_args,
                    "data": {
                        "status": 0,  # 第一帧
                        "format": f"audio/L16;rate={sample_rate}",
                        "audio": base64.b64encode(audio_data[:frame_size]).decode('utf-8'),
                        "encoding": "raw" if audio_format == "pcm" else audio_format,
                    }
                }
                await ws.send(json.dumps(first_frame))
                
                # 发送中间帧
                index = frame_size
                while index < len(audio_data) - frame_size:
                    frame = {
                        "data": {
                            "status": 1,  # 中间帧
                            "format": f"audio/L16;rate={sample_rate}",
                            "audio": base64.b64encode(audio_data[index:index + frame_size]).decode('utf-8'),
                            "encoding": "raw" if audio_format == "pcm" else audio_format,
                        }
                    }
                    await ws.send(json.dumps(frame))
                    await asyncio.sleep(interval)
                    index += frame_size

                # 发送最后一帧
                last_frame = {
                    "data": {
                        "status": 2,  # 最后一帧
                        "format": f"audio/L16;rate={sample_rate}",
                        "audio": base64.b64encode(audio_data[index:]).decode('utf-8'),
                        "encoding": "raw" if audio_format == "pcm" else audio_format,
                    }
                }
                await ws.send(json.dumps(last_frame))

                # 接收识别结果
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=10)
                        result = json.loads(response)
                        
                        if result.get("code") != 0:
                            return SpeechRecognizeResponse(
                                success=False,
                                error=f"识别失败: {result.get('message', '未知错误')}"
                            )

                        data = result.get("data", {})
                        result_data = data.get("result", {})
                        
                        # 解析识别结果
                        ws_data = result_data.get("ws", [])
                        for ws_item in ws_data:
                            cw = ws_item.get("cw", [])
                            for cw_item in cw:
                                result_text += cw_item.get("w", "")

                        # 检查是否结束
                        if data.get("status") == 2:
                            break

                    except asyncio.TimeoutError:
                        break

            return SpeechRecognizeResponse(
                success=True,
                text=result_text.strip(),
                confidence=0.9,  # 讯飞不返回置信度，给一个默认值
            )

        except Exception as e:
            return SpeechRecognizeResponse(
                success=False,
                error=f"语音识别失败: {str(e)}"
            )

    def recognize_audio_sync(
        self,
        audio_data: bytes,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        language: str = "zh_cn",
    ) -> SpeechRecognizeResponse:
        """同步版本的语音识别"""
        return asyncio.run(self.recognize_audio(
            audio_data, audio_format, sample_rate, language
        ))


class SimpleSpeechService:
    """简化版语音识别服务（使用讯飞REST API）"""
    
    # 讯飞一句话识别API
    API_URL = "https://iat-api.xfyun.cn/v2/iat"
    
    def __init__(
        self,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.app_id = app_id or os.getenv("XUNFEI_APP_ID")
        self.api_key = api_key or os.getenv("XUNFEI_API_KEY")
        self.api_secret = api_secret or os.getenv("XUNFEI_API_SECRET")
        self._xunfei_service = XunfeiSpeechService(
            app_id=self.app_id,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return self._xunfei_service.is_configured()

    def recognize(
        self,
        audio_data: bytes,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        language: str = "zh_cn",
    ) -> SpeechRecognizeResponse:
        """识别音频（同步接口）"""
        return self._xunfei_service.recognize_audio_sync(
            audio_data, audio_format, sample_rate, language
        )

    async def recognize_async(
        self,
        audio_data: bytes,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        language: str = "zh_cn",
    ) -> SpeechRecognizeResponse:
        """识别音频（异步接口）"""
        return await self._xunfei_service.recognize_audio(
            audio_data, audio_format, sample_rate, language
        )


# 全局实例
_speech_service: Optional[SimpleSpeechService] = None


def get_speech_service() -> SimpleSpeechService:
    """获取语音识别服务实例"""
    global _speech_service
    if _speech_service is None:
        _speech_service = SimpleSpeechService()
    return _speech_service


def is_speech_service_configured() -> bool:
    """检查语音服务是否已配置"""
    return get_speech_service().is_configured()
