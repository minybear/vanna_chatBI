"""
训练数据修复配置文件
请根据您的实际情况修改以下配置
"""

# 数据库名称（多数据库时需要确保所有表都在同一个数据库中，或者分别配置）
DATABASE_NAME = "your_database_name"  # ⚠️ 请修改为实际的数据库名

# 系统中使用的所有表名列表
# 脚本会检查这些表名是否使用了完全限定格式 (database.table)
TABLE_NAMES = [
    "control_cart",      # License消耗记录表
    "control_log",       # 控制日志表
    # 添加更多表名...
]

# 如果有多个数据库，可以使用字典映射
# TABLE_DATABASE_MAP = {
#     "control_cart": "database1",
#     "user_info": "database2",
# }

