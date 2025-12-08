"""
为 CSV 中的 SQL 语料自动添加数据库前缀
用法: python fix_sql_table_names.py <csv_file>
"""

import csv
import re
import sys

# 表名到数据库的映射
TABLE_TO_DB = {
    # db_provision 数据库
    'tbl_watch_subscription': 'db_provision',
    'tbl_volte_service': 'db_provision',
    'tbl_vowifi_service': 'db_provision',
    'biz_watch_log': 'db_provision',
    
    # db_stat 数据库
    'auth_log': 'db_stat',
    'stat_min': 'db_stat',
    'stat_hour': 'db_stat',
    
    # db_mgmt 数据库 (如果有)
    # 可以根据需要添加
}

def add_db_prefix_to_sql(sql_content):
    """给 SQL 中的表名添加数据库前缀"""
    modified_sql = sql_content
    
    for table_name, db_name in TABLE_TO_DB.items():
        # 匹配模式：
        # FROM table_name
        # JOIN table_name
        # INTO table_name
        # UPDATE table_name
        # 但避免已经有前缀的情况 (db_name.table_name)
        
        # 使用 word boundary 确保完整匹配表名
        patterns = [
            (rf'\bFROM\s+`?{table_name}`?(?!\s*\.)', f'FROM {db_name}.{table_name}'),
            (rf'\bJOIN\s+`?{table_name}`?(?!\s*\.)', f'JOIN {db_name}.{table_name}'),
            (rf'\bINTO\s+`?{table_name}`?(?!\s*\.)', f'INTO {db_name}.{table_name}'),
            (rf'\bUPDATE\s+`?{table_name}`?(?!\s*\.)', f'UPDATE {db_name}.{table_name}'),
            # 处理反引号的情况
            (rf'\bFROM\s+{table_name}\b(?!\s*\.)', f'FROM {db_name}.{table_name}'),
            (rf'\bJOIN\s+{table_name}\b(?!\s*\.)', f'JOIN {db_name}.{table_name}'),
        ]
        
        for pattern, replacement in patterns:
            modified_sql = re.sub(pattern, replacement, modified_sql, flags=re.IGNORECASE)
    
    return modified_sql

def process_csv(input_file, output_file=None):
    """处理 CSV 文件"""
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    rows_modified = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        rows = []
        for row in reader:
            data_type = row.get('类型', '').strip().lower()
            
            if data_type == 'sql':
                original_content = row.get('内容', '')
                modified_content = add_db_prefix_to_sql(original_content)
                
                if original_content != modified_content:
                    row['内容'] = modified_content
                    rows_modified += 1
                    print(f"  [MODIFIED] {row.get('用户问题示例', '')[:50]}...")
            
            rows.append(row)
    
    # 写回文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n[SUMMARY]")
    print(f"  Modified SQL entries: {rows_modified}")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_sql_table_names.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    process_csv(csv_file)

