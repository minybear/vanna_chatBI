"""环境变量与运行配置。"""
import os
from dotenv import load_dotenv

load_dotenv()

# === 认证与租户 ===
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
AUTH_USERS_JSON = os.getenv("AUTH_USERS_JSON", "")
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))
AUTH_TOKEN_SLIDING_ENABLED = os.getenv("AUTH_TOKEN_SLIDING_ENABLED", "true").lower() == "true"
AUTH_TOKEN_REFRESH_SECONDS = int(os.getenv("AUTH_TOKEN_REFRESH_SECONDS", "1800"))
JWT_SECRET = os.getenv("JWT_SECRET", "chatbi-default-secret-change-in-production")
JWT_ALGORITHM = "HS256"

TENANT_ISOLATION_ENABLED = os.getenv("ENABLE_TENANT_ISOLATION", "false").lower() == "true"
TENANT_COLUMN = os.getenv("TENANT_COLUMN", "operator_id")


def log_env_check():
    """启动时打印关键环境变量状态（调试用）。"""
    print("=" * 60)
    print("[ENV CHECK] 环境变量加载状态:")
    print(f"  - ZHIPU_API_KEY: {'已设置' if os.getenv('ZHIPU_API_KEY') else '❌ 未设置'}")
    print(f"  - DB_HOST: {'已设置' if os.getenv('DB_HOST') else '❌ 未设置'}")
    print(f"  - LARK_WEBHOOK_URL: {'已设置' if os.getenv('LARK_WEBHOOK_URL') else '未设置'}")
    if os.getenv("LARK_WEBHOOK_URL"):
        webhook_url = os.getenv("LARK_WEBHOOK_URL")
        print(f"  - Webhook URL (前40字符): {webhook_url[:40]}...")
    show_sql_raw = os.getenv("SHOW_SQL_DEBUG", "未设置")
    print(f"  - SHOW_SQL_DEBUG: '{show_sql_raw}' (类型: {type(show_sql_raw).__name__}, 长度: {len(show_sql_raw)})")
    print("=" * 60)
