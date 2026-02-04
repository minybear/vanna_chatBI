"""飞书点踩反馈通知。"""
import os
import requests
from datetime import datetime

from core.schemas import FeedbackRequest


def send_lark_alert(feedback: FeedbackRequest) -> None:
    """发送点踩反馈到飞书群。"""
    url = os.getenv("LARK_WEBHOOK_URL")
    if not url:
        print("=" * 60)
        print("❌ 错误: LARK_WEBHOOK_URL 环境变量未设置")
        print("解决方法: 在 .env 中添加 LARK_WEBHOOK_URL=你的webhook地址")
        print("=" * 60)
        return

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "🚨 用户反馈：回答不准确 (点踩)"},
                "template": "red"
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**用户提问:**\n{feedback.question}"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**生成的 SQL:**\n```sql\n{feedback.sql}\n```"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**解释:**\n{feedback.explanation or '无'}"}},
                {"tag": "hr"},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": f"反馈时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}]}
            ]
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 0:
                print("[飞书通知] ✅ 发送成功！")
            else:
                print(f"[飞书通知] ⚠️ 飞书返回错误: {result}")
        else:
            print(f"[飞书通知] ❌ HTTP错误: {resp.status_code}, 响应: {resp.text}")
    except requests.exceptions.Timeout:
        print("[飞书通知] ❌ 请求超时，请检查网络连接")
    except requests.exceptions.RequestException as e:
        print(f"[飞书通知] ❌ 网络错误: {e}")
    except Exception as e:
        print(f"[飞书通知] ❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
