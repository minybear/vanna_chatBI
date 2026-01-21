"""
修复训练数据脚本 - 确保所有SQL和DDL使用完全限定的表名格式 (database.table)
"""
import os
import re
from dotenv import load_dotenv
from app import vn

# 加载环境变量
load_dotenv()

# 定义需要添加数据库前缀的表名列表（根据您的实际表名修改）
TABLE_NAMES = [
    'control_cart',
    'control_log', 
    'user_table',
    'license_info',
    # 添加更多表名...
]

# 从环境变量获取数据库名，如果没有设置，需要手动指定
DEFAULT_DATABASE = os.getenv('DB_NAME', 'your_database_name')  # 请修改为实际的数据库名

def needs_database_prefix(content, table_names):
    """
    检查内容是否需要添加数据库前缀
    返回: (需要修复, 涉及的表名列表)
    """
    needs_fix = False
    found_tables = []
    
    for table in table_names:
        # 检查是否存在不带数据库前缀的表名引用
        # 匹配 FROM table, JOIN table, UPDATE table, INSERT INTO table 等
        patterns = [
            rf'\bFROM\s+{table}\b',
            rf'\bJOIN\s+{table}\b',
            rf'\bINTO\s+{table}\b',
            rf'\bUPDATE\s+{table}\b',
            rf'\bTABLE\s+{table}\b',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # 检查是否已经有数据库前缀
                prefixed_pattern = rf'\b\w+\.{table}\b'
                if not re.search(prefixed_pattern, content, re.IGNORECASE):
                    needs_fix = True
                    if table not in found_tables:
                        found_tables.append(table)
    
    return needs_fix, found_tables

def add_database_prefix(content, table_names, database_name):
    """
    为SQL内容添加数据库前缀
    """
    fixed_content = content
    
    for table in table_names:
        # 替换 FROM table -> FROM database.table
        fixed_content = re.sub(
            rf'\bFROM\s+{table}\b',
            f'FROM {database_name}.{table}',
            fixed_content,
            flags=re.IGNORECASE
        )
        
        # 替换 JOIN table -> JOIN database.table
        fixed_content = re.sub(
            rf'\bJOIN\s+{table}\b',
            f'JOIN {database_name}.{table}',
            fixed_content,
            flags=re.IGNORECASE
        )
        
        # 替换 INTO table -> INTO database.table
        fixed_content = re.sub(
            rf'\bINTO\s+{table}\b',
            f'INTO {database_name}.{table}',
            fixed_content,
            flags=re.IGNORECASE
        )
        
        # 替换 UPDATE table -> UPDATE database.table
        fixed_content = re.sub(
            rf'\bUPDATE\s+{table}\b',
            f'UPDATE {database_name}.{table}',
            fixed_content,
            flags=re.IGNORECASE
        )
        
        # 替换 TABLE table -> TABLE database.table (for CREATE/ALTER)
        fixed_content = re.sub(
            rf'\bTABLE\s+{table}\b',
            f'TABLE {database_name}.{table}',
            fixed_content,
            flags=re.IGNORECASE
        )
    
    return fixed_content

def main():
    print("="*80)
    print("开始检查知识库训练数据...")
    print("="*80)
    
    # 获取所有训练数据
    try:
        df = vn.get_training_data()
    except Exception as e:
        print(f"❌ 获取训练数据失败: {e}")
        return
    
    if df is None or df.empty:
        print("✅ 知识库中没有训练数据")
        return
    
    print(f"\n📊 总共有 {len(df)} 条训练数据")
    print(f"   - SQL: {len(df[df['training_data_type'] == 'sql'])}")
    print(f"   - DDL: {len(df[df['training_data_type'] == 'ddl'])}")
    print(f"   - Documentation: {len(df[df['training_data_type'] == 'documentation'])}")
    
    # 统计需要修复的数据
    items_to_fix = []
    
    print("\n" + "="*80)
    print("检查中...")
    print("="*80 + "\n")
    
    for idx, row in df.iterrows():
        data_type = row['training_data_type']
        content = row['content']
        data_id = row['id']
        
        # 只检查 SQL 和 DDL 类型
        if data_type in ['sql', 'ddl'] and pd.notna(content):
            needs_fix, found_tables = needs_database_prefix(content, TABLE_NAMES)
            
            if needs_fix:
                items_to_fix.append({
                    'id': data_id,
                    'type': data_type,
                    'content': content,
                    'question': row.get('question', ''),
                    'tables': found_tables
                })
                
                print(f"⚠️  [{data_type.upper()}] 需要修复:")
                print(f"   ID: {data_id}")
                if data_type == 'sql' and pd.notna(row.get('question')):
                    print(f"   问题: {row['question']}")
                print(f"   涉及表: {', '.join(found_tables)}")
                print(f"   内容预览: {content[:100]}...")
                print()
    
    if not items_to_fix:
        print("✅ 所有训练数据都符合要求！")
        return
    
    print("="*80)
    print(f"⚠️  发现 {len(items_to_fix)} 条需要修复的数据")
    print("="*80 + "\n")
    
    # 询问用户是否继续修复
    response = input("是否要自动修复这些数据？(y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ 用户取消操作")
        return
    
    # 确认数据库名
    db_name = input(f"\n请输入数据库名 (默认: {DEFAULT_DATABASE}): ").strip()
    if not db_name:
        db_name = DEFAULT_DATABASE
    
    print(f"\n使用数据库名: {db_name}")
    print("\n" + "="*80)
    print("开始修复...")
    print("="*80 + "\n")
    
    success_count = 0
    fail_count = 0
    
    for item in items_to_fix:
        try:
            # 修复内容
            fixed_content = add_database_prefix(item['content'], TABLE_NAMES, db_name)
            
            print(f"🔧 修复 [{item['type'].upper()}] {item['id']}")
            print(f"   修复前: {item['content'][:80]}...")
            print(f"   修复后: {fixed_content[:80]}...")
            
            # 删除旧数据
            vn.remove_training_data(item['id'])
            
            # 添加修复后的数据
            if item['type'] == 'ddl':
                vn.train(ddl=fixed_content)
            elif item['type'] == 'sql':
                vn.train(question=item['question'], sql=fixed_content)
            
            print(f"   ✅ 成功\n")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 失败: {e}\n")
            fail_count += 1
    
    print("="*80)
    print("修复完成！")
    print("="*80)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 总计: {len(items_to_fix)}")

if __name__ == "__main__":
    import pandas as pd
    main()

