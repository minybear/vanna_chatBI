"""
导入License使用情况相关的训练数据
确保Vanna能准确理解并生成正确的SQL查询
"""
import requests
import json

# API端点
API_BASE = "http://localhost:8000/api/v0"

# 训练数据列表
training_data = [
    {
        "type": "sql",
        "question": "最近一个月各license的使用情况",
        "content": """SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY stat_date, control_name
ORDER BY stat_date DESC, control_name;"""
    },
    {
        "type": "sql",
        "question": "最近一周各license的使用趋势",
        "content": """SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used,
  SUM(used_amount) AS daily_consumption
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY stat_date, control_name
ORDER BY stat_date DESC, control_name;"""
    },
    {
        "type": "sql",
        "question": "VOLTE_BUSINESS_LICENSE最近一个月的每日使用情况",
        "content": """SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used,
  SUM(used_amount) AS daily_consumption,
  MAX(control_value) AS total_capacity
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
  AND control_name = 'VOLTE_BUSINESS_LICENSE'
GROUP BY stat_date, control_name
ORDER BY stat_date DESC;"""
    },
    {
        "type": "sql",
        "question": "各license在最近一个月的平均使用情况",
        "content": """SELECT 
  control_name AS license_name,
  AVG(real_amount) AS avg_daily_used,
  MAX(real_amount) AS peak_daily_used,
  MIN(real_amount) AS lowest_daily_used,
  COUNT(DISTINCT DATE(create_time)) AS days_with_data
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY control_name
ORDER BY avg_daily_used DESC;"""
    },
    {
        "type": "sql",
        "question": "各license最近一个月的使用率",
        "content": """SELECT 
  control_name AS license_name,
  MAX(real_amount) AS current_used,
  MAX(control_value) AS total_capacity,
  ROUND(MAX(real_amount) / MAX(control_value) * 100, 2) AS usage_percentage
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY control_name
ORDER BY usage_percentage DESC;"""
    },
    {
        "type": "documentation",
        "question": "",
        "content": """## License字段使用说明

当用户询问license使用情况或消耗情况时：
- **real_amount**: 表示该license的实际使用总量（累计值），用于查询"使用情况"
- **used_amount**: 表示增量消耗（当天新增的使用量），用于分析"消耗趋势"
- **control_value**: license的总容量

**重要规则**：
1. 查询"各license的使用情况"时，应该返回时间序列数据（按日期分组）
2. SQL格式应为: SELECT DATE(create_time) AS stat_date, control_name, MAX(real_amount) ...
3. 使用 DATE(create_time) 而不是直接使用 create_time
4. 必须同时按 stat_date 和 control_name 分组
5. 这样的结果才能正确渲染为多条折线图（每个license一条线）"""
    }
]

def import_training_data():
    """导入训练数据到Vanna"""
    success_count = 0
    fail_count = 0
    
    print("="*60)
    print("开始导入License相关训练数据...")
    print("="*60)
    
    for i, data in enumerate(training_data, 1):
        try:
            response = requests.post(
                f"{API_BASE}/training_data",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ [{i}/{len(training_data)}] 成功: {data['type']} - {data['question'][:50] if data['question'] else '文档说明'}")
                success_count += 1
            else:
                print(f"❌ [{i}/{len(training_data)}] 失败: {response.status_code} - {response.text[:100]}")
                fail_count += 1
                
        except Exception as e:
            print(f"❌ [{i}/{len(training_data)}] 错误: {str(e)}")
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"导入完成！成功: {success_count}, 失败: {fail_count}")
    print("="*60)
    
    if success_count > 0:
        print("\n💡 提示：训练数据已导入，现在可以测试相同的问题，SQL应该会更准确。")

if __name__ == "__main__":
    import_training_data()






