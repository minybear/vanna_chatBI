-- License使用情况训练数据
-- 用于训练Vanna正确理解License相关的查询

-- ===== 示例1: 最近一个月各license的使用情况（按日期的时间序列） =====
-- 问题: 最近一个月各license的使用情况
-- SQL:
SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY stat_date, control_name
ORDER BY stat_date DESC, control_name;

-- 说明: 
-- 1. real_amount 表示当天该license的实际使用总量
-- 2. used_amount 表示当天的增量消耗，与查询使用情况无关
-- 3. 按日期和license名称分组，展示时间序列数据
-- 4. 结果适合生成多条折线图（每个license一条线）

-- ===== 示例2: 最近一周各license的使用趋势 =====
-- 问题: 最近一周各license的使用趋势
-- SQL:
SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used,
  SUM(used_amount) AS daily_consumption
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY stat_date, control_name
ORDER BY stat_date DESC, control_name;

-- ===== 示例3: 特定license的每日使用情况 =====
-- 问题: VOLTE_BUSINESS_LICENSE最近一个月的每日使用情况
-- SQL:
SELECT 
  DATE(create_time) AS stat_date,
  control_name AS license_name,
  MAX(real_amount) AS daily_total_used,
  SUM(used_amount) AS daily_consumption,
  MAX(control_value) AS total_capacity
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
  AND control_name = 'VOLTE_BUSINESS_LICENSE'
GROUP BY stat_date, control_name
ORDER BY stat_date DESC;

-- ===== 示例4: License使用汇总统计（不是时间序列） =====
-- 问题: 各license在最近一个月的平均使用情况
-- SQL:
SELECT 
  control_name AS license_name,
  AVG(real_amount) AS avg_daily_used,
  MAX(real_amount) AS peak_daily_used,
  MIN(real_amount) AS lowest_daily_used,
  COUNT(DISTINCT DATE(create_time)) AS days_with_data
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY control_name
ORDER BY avg_daily_used DESC;

-- ===== 示例5: License使用率分析 =====
-- 问题: 各license最近一个月的使用率
-- SQL:
SELECT 
  control_name AS license_name,
  MAX(real_amount) AS current_used,
  MAX(control_value) AS total_capacity,
  ROUND(MAX(real_amount) / MAX(control_value) * 100, 2) AS usage_percentage
FROM db_stat.control_cart
WHERE create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY control_name
ORDER BY usage_percentage DESC;

-- ===== 字段说明 =====
-- control_name: License类型名称
-- real_amount: 实际使用总量（累计值）
-- used_amount: 增量消耗（当天新增的使用量）
-- control_value: License总容量
-- create_time: 记录时间






