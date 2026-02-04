"""飞书群管理服务。"""
import base64
import io
import os
import json
import uuid
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any

from core.schemas import LarkGroup, AlertResult, CelebrationResult


class LarkGroupManager:
    """飞书群配置管理"""

    def __init__(self, chroma_client=None):
        self.chroma_client = chroma_client
        self.collection = None
        if chroma_client:
            from core.conversation_history import SimpleEmbeddingFunction
            self.collection = chroma_client.get_or_create_collection(
                name="lark_groups",
                embedding_function=SimpleEmbeddingFunction(),
            )
        
        # 从环境变量加载默认群配置
        self._default_groups = self._load_default_groups()

    def _load_default_groups(self) -> List[LarkGroup]:
        """从环境变量加载默认群配置"""
        groups = []
        
        # 兼容旧的单群配置
        default_webhook = os.getenv("LARK_WEBHOOK_URL")
        if default_webhook:
            groups.append(LarkGroup(
                id="default",
                name="默认群",
                webhook_url=default_webhook,
                description="默认飞书通知群",
                group_type="general",
            ))

        # 新的多群配置格式: LARK_GROUPS={"alert":"webhook1","celebration":"webhook2"}
        groups_json = os.getenv("LARK_GROUPS")
        if groups_json:
            try:
                groups_config = json.loads(groups_json)
                for name, webhook in groups_config.items():
                    groups.append(LarkGroup(
                        id=f"env-{name}",
                        name=name,
                        webhook_url=webhook,
                        group_type=name if name in ["alert", "celebration"] else "general",
                    ))
            except Exception as e:
                print(f"[LarkGroupManager] 解析LARK_GROUPS失败: {e}")

        return groups

    def list_groups(self, operator_id: str = "") -> List[Dict[str, Any]]:
        """获取用户可用的飞书群列表"""
        groups = []

        # 添加默认群
        for group in self._default_groups:
            groups.append(group.model_dump())

        # 从数据库加载用户配置的群
        if self.collection:
            try:
                if operator_id:
                    data = self.collection.get(where={"operator_id": operator_id})
                else:
                    data = self.collection.get()

                documents = data.get("documents") if data else []
                for doc in documents or []:
                    try:
                        group_data = json.loads(doc)
                        groups.append(group_data)
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LarkGroupManager] 加载群配置失败: {e}")

        return groups

    def get_group(self, group_id: str, operator_id: str = "") -> Optional[Dict[str, Any]]:
        """获取指定群配置"""
        # 检查默认群
        for group in self._default_groups:
            if group.id == group_id:
                return group.model_dump()

        # 从数据库查找
        if self.collection:
            try:
                data = self.collection.get(ids=[group_id])
                if data and data.get("ids"):
                    doc = (data.get("documents") or [None])[0]
                    if doc:
                        group_data = json.loads(doc)
                        # 检查权限
                        if operator_id and group_data.get("operator_id") and group_data.get("operator_id") != operator_id:
                            return None
                        return group_data
            except Exception:
                pass

        return None

    def add_group(
        self,
        name: str,
        webhook_url: str,
        description: str = "",
        group_type: str = "general",
        operator_id: str = "",
    ) -> Dict[str, Any]:
        """添加飞书群配置"""
        group_id = f"group-{uuid.uuid4().hex[:8]}"
        group = LarkGroup(
            id=group_id,
            name=name,
            webhook_url=webhook_url,
            description=description,
            group_type=group_type,
            created_at=datetime.now().isoformat(),
            operator_id=operator_id,
        )

        if self.collection:
            self.collection.upsert(
                ids=[group_id],
                documents=[json.dumps(group.model_dump(), ensure_ascii=False)],
                metadatas=[{"operator_id": operator_id, "group_type": group_type}],
            )

        return group.model_dump()

    def update_group(
        self,
        group_id: str,
        name: Optional[str] = None,
        webhook_url: Optional[str] = None,
        description: Optional[str] = None,
        group_type: Optional[str] = None,
        operator_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """更新飞书群配置"""
        existing = self.get_group(group_id, operator_id)
        if not existing:
            return None

        # 不能修改默认群
        if group_id.startswith("env-") or group_id == "default":
            return None

        if name is not None:
            existing["name"] = name
        if webhook_url is not None:
            existing["webhook_url"] = webhook_url
        if description is not None:
            existing["description"] = description
        if group_type is not None:
            existing["group_type"] = group_type

        if self.collection:
            self.collection.upsert(
                ids=[group_id],
                documents=[json.dumps(existing, ensure_ascii=False)],
                metadatas=[{"operator_id": operator_id, "group_type": existing.get("group_type", "general")}],
            )

        return existing

    def delete_group(self, group_id: str, operator_id: str = "") -> bool:
        """删除飞书群配置"""
        # 不能删除默认群
        if group_id.startswith("env-") or group_id == "default":
            return False

        existing = self.get_group(group_id, operator_id)
        if not existing:
            return False

        if self.collection:
            self.collection.delete(ids=[group_id])
            return True

        return False


class LarkReportSender:
    """飞书报告发送器"""

    def __init__(self, group_manager: LarkGroupManager):
        self.group_manager = group_manager
        self._tenant_access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

    def _get_tenant_access_token(self) -> Optional[str]:
        """获取飞书应用 tenant_access_token（需配置 LARK_APP_ID、LARK_APP_SECRET）"""
        app_id = os.getenv("LARK_APP_ID")
        app_secret = os.getenv("LARK_APP_SECRET")
        if not app_id or not app_secret:
            return None
        if self._tenant_access_token and self._token_expires_at and datetime.now().timestamp() < self._token_expires_at - 60:
            return self._tenant_access_token
        try:
            resp = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": app_id, "app_secret": app_secret},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("code") != 0:
                return None
            self._tenant_access_token = data.get("tenant_access_token")
            self._token_expires_at = datetime.now().timestamp() + data.get("expire", 7200)
            return self._tenant_access_token
        except Exception:
            return None

    def _upload_lark_image(self, image_data: str) -> Optional[str]:
        """
        上传图片到飞书，返回 image_key。image_data 可为 data URL（data:image/xxx;base64,...）或纯 base64。
        需要配置 LARK_APP_ID、LARK_APP_SECRET。
        """
        token = self._get_tenant_access_token()
        if not token:
            return None
        raw_b64 = image_data
        if raw_b64.startswith("data:"):
            idx = raw_b64.find("base64,")
            if idx != -1:
                raw_b64 = raw_b64[idx + 7:]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            print(f"[LarkReportSender] 图片 base64 解码失败: {e}")
            return None
        try:
            # 飞书上传图片：multipart/form-data，字段名 image，image_type=message
            resp = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/images",
                headers={"Authorization": f"Bearer {token}"},
                data={"image_type": "message"},
                files={"image": ("chart.png", io.BytesIO(raw_bytes), "image/png")},
                timeout=15,
            )
            body = resp.json() if resp.text else {}
            if resp.status_code != 200:
                print(f"[LarkReportSender] 上传图片 HTTP {resp.status_code}: {body}")
                return None
            if body.get("code") != 0:
                print(f"[LarkReportSender] 上传图片 API 错误 code={body.get('code')} msg={body.get('msg', body)}")
                return None
            # 响应格式: {"code":0,"data":{"image_key":"xxx"}}
            out = body.get("data") or body
            img_key = out.get("image_key") if isinstance(out, dict) else None
            if not img_key:
                print(f"[LarkReportSender] 上传图片响应无 image_key: {body}")
            return img_key
        except Exception as e:
            print(f"[LarkReportSender] 上传图片异常: {e}")
            return None

    def send_insight_report(
        self,
        group_id: str,
        report_content: str,
        chart_images: Optional[List[str]] = None,
        report_type: str = "insight",
        operator_id: str = "",
    ) -> Dict[str, Any]:
        """
        发送洞察报告到飞书群
        
        Args:
            group_id: 飞书群ID
            report_content: 报告内容（Markdown格式）
            chart_images: 图表截图（base64列表）
            report_type: 报告类型 (insight/alert/celebration)
            operator_id: 操作者ID
            
        Returns:
            {"success": bool, "error": str}
        """
        group = self.group_manager.get_group(group_id, operator_id)
        if not group:
            return {"success": False, "error": "飞书群配置不存在"}

        webhook_url = group.get("webhook_url")
        if not webhook_url:
            return {"success": False, "error": "Webhook URL未配置"}

        try:
            # 若有图表且配置了应用凭证，先上传图片，再发「报告正文卡片 + 每张图单独卡片」
            image_keys: List[str] = []
            if chart_images:
                for one in chart_images[:10]:
                    img_key = self._upload_lark_image(one)
                    if img_key:
                        image_keys.append(img_key)
            # 报告正文卡片（图表单独发；若无图表或上传失败，仅正文）
            card = self._build_report_card(
                report_content, report_type,
                chart_images=None,
                image_keys_for_card=image_keys if chart_images else None,
            )
            response = requests.post(
                webhook_url,
                json=card,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}
            result = response.json()
            if result.get("code") != 0 and result.get("StatusCode") != 0:
                return {"success": False, "error": result.get("msg", "发送失败")}
            # 每张图单独发一条卡片，便于展示
            for idx, img_key in enumerate(image_keys):
                chart_card = self._build_chart_only_card(img_key, idx + 1)
                r2 = requests.post(
                    webhook_url,
                    json=chart_card,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                if r2.status_code != 200:
                    print(f"[LarkReportSender] 图表卡片 {idx + 1} 发送失败 HTTP {r2.status_code}")
                else:
                    j = r2.json() if r2.text else {}
                    if j.get("code") != 0 and j.get("StatusCode") != 0:
                        print(f"[LarkReportSender] 图表卡片 {idx + 1} 发送失败: {j.get('msg', j)}")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def send_alert(
        self,
        group_id: str,
        alert: AlertResult,
        operator_id: str = "",
    ) -> Dict[str, Any]:
        """发送预警通知"""
        if not alert.triggered:
            return {"success": False, "error": "预警未触发"}

        group = self.group_manager.get_group(group_id, operator_id)
        if not group:
            # 尝试使用预警专用群
            groups = self.group_manager.list_groups(operator_id)
            alert_groups = [g for g in groups if g.get("group_type") == "alert"]
            if alert_groups:
                group = alert_groups[0]

        if not group:
            return {"success": False, "error": "未配置预警通知群"}

        card = self._build_alert_card(alert)
        
        try:
            response = requests.post(
                group["webhook_url"],
                json=card,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0 or result.get("StatusCode") == 0:
                    return {"success": True}
                return {"success": False, "error": result.get("msg", "发送失败")}

            return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def send_celebration(
        self,
        group_id: str,
        celebration: CelebrationResult,
        operator_id: str = "",
    ) -> Dict[str, Any]:
        """发送喜报通知"""
        if not celebration.triggered:
            return {"success": False, "error": "喜报未触发"}

        group = self.group_manager.get_group(group_id, operator_id)
        if not group:
            # 尝试使用喜报专用群
            groups = self.group_manager.list_groups(operator_id)
            celebration_groups = [g for g in groups if g.get("group_type") == "celebration"]
            if celebration_groups:
                group = celebration_groups[0]

        if not group:
            return {"success": False, "error": "未配置喜报通知群"}

        card = self._build_celebration_card(celebration)
        
        try:
            response = requests.post(
                group["webhook_url"],
                json=card,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0 or result.get("StatusCode") == 0:
                    return {"success": True}
                return {"success": False, "error": result.get("msg", "发送失败")}

            return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_chart_only_card(self, img_key: str, index: int) -> Dict[str, Any]:
        """构建仅含一张图表的卡片（用于飞书逐条发送图表）"""
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"📊 报告图表 {index}"},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "img",
                        "img_key": img_key,
                        "alt": {"tag": "plain_text", "content": f"图表{index}"},
                    },
                ],
            },
        }

    def _build_report_card(
        self,
        content: str,
        report_type: str,
        chart_images: Optional[List[str]] = None,
        image_keys_for_card: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """构建报告正文卡片。图表通过 image_keys_for_card 或单独消息发送，不在此卡片内上传。"""
        colors = {
            "insight": "blue",
            "alert": "red",
            "celebration": "green",
        }
        color = colors.get(report_type, "blue")
        titles = {
            "insight": "📊 AI 数据洞察报告",
            "alert": "🚨 数据预警通知",
            "celebration": "🎉 业务喜报",
        }
        title = titles.get(report_type, "AI 数据报告")

        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n... (内容过长，已截断)"

        elements: List[Dict[str, Any]] = [
            {"tag": "markdown", "content": content},
        ]

        # 图表由 send_insight_report 单独发卡片，此处仅在没有成功上传时提示下载
        if image_keys_for_card is not None and len(image_keys_for_card) == 0:
            elements.append({
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": "完整报告含图表请下载 HTML 查看。"},
                ],
            })

        elements.append({
            "tag": "note",
            "elements": [
                {"tag": "plain_text", "content": f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
            ],
        })

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color,
                },
                "elements": elements,
            },
        }

    def _build_alert_card(self, alert: AlertResult) -> Dict[str, Any]:
        """构建预警卡片"""
        severity_colors = {
            "info": "blue",
            "warning": "orange",
            "critical": "red",
        }
        color = severity_colors.get(alert.severity.value, "red")
        
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        icon = severity_icons.get(alert.severity.value, "🚨")

        metrics_text = ""
        if alert.metrics:
            metrics_text = f"\n\n**关键指标:**\n- 当前值: {alert.metrics.get('current', 'N/A')}\n- 前值: {alert.metrics.get('previous', 'N/A')}\n- 变化幅度: {alert.metrics.get('change_ratio', 0)}%"

        suggestion_text = ""
        if alert.suggestion:
            suggestion_text = f"\n\n**建议:** {alert.suggestion}"

        content = f"{alert.message}{metrics_text}{suggestion_text}"

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{icon} {alert.title}",
                    },
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content,
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"预警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ]
                    }
                ],
            }
        }

    def _build_celebration_card(self, celebration: CelebrationResult) -> Dict[str, Any]:
        """构建喜报卡片"""
        highlights_text = ""
        if celebration.highlights:
            highlights_text = "\n\n**亮点:**\n" + "\n".join(f"- {h}" for h in celebration.highlights)

        metrics_text = ""
        if celebration.metrics:
            metrics_text = f"\n\n**关键指标:**\n- 当前值: {celebration.metrics.get('current', 'N/A')}\n- 增长幅度: {celebration.metrics.get('change_ratio', 0)}%"

        content = f"🎊 {celebration.achievement}{highlights_text}{metrics_text}"

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"🎉 {celebration.title}",
                    },
                    "template": "green",
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content,
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"喜报时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ]
                    }
                ],
            }
        }


# 全局实例
_group_manager: Optional[LarkGroupManager] = None
_report_sender: Optional[LarkReportSender] = None


def init_lark_services(chroma_client=None):
    """初始化飞书服务"""
    global _group_manager, _report_sender
    _group_manager = LarkGroupManager(chroma_client)
    _report_sender = LarkReportSender(_group_manager)
    return _group_manager, _report_sender


def get_lark_group_manager() -> LarkGroupManager:
    """获取飞书群管理器"""
    global _group_manager
    if _group_manager is None:
        _group_manager = LarkGroupManager()
    return _group_manager


def get_lark_report_sender() -> LarkReportSender:
    """获取飞书报告发送器"""
    global _report_sender, _group_manager
    if _report_sender is None:
        if _group_manager is None:
            _group_manager = LarkGroupManager()
        _report_sender = LarkReportSender(_group_manager)
    return _report_sender
