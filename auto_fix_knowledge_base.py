"""
自动检查和修复知识库中的训练数据
智能检测表名，自动修复为完全限定格式
"""
import os
import re
import pandas as pd
from dotenv import load_dotenv
from app import vn

# 加载环境变量
load_dotenv()

def extract_table_names(content):
    """
    从SQL/DDL内容中提取表名
    """
    tables = set()
    
    # 匹配 FROM/JOIN/INTO/UPDATE/TABLE 后面的表名
    patterns = [
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        r'\bTABLE\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # group(1) 是数据库前缀（可能为None），group(2) 是表名
            db_prefix = match.group(1)
            table_name = match.group(2)
            
            # 过滤掉SQL关键字
            sql_keywords = {'select', 'from', 'where', 'join', 'left', 'right', 
                           'inner', 'outer', 'on', 'and', 'or', 'not', 'exists'}
            if table_name.lower() not in sql_keywords:
                tables.add((db_prefix.rstrip('.') if db_prefix else None, table_name))
    
    return tables

def analyze_training_data(df):
    """
    分析训练数据，提取所有表名和数据库名
    """
    all_tables = {}  # {table_name: {'databases': set(), 'count': int, 'unqualified_count': int}}
    
    for idx, row in df.iterrows():
        data_type = row['training_data_type']
        content = str(row['content']) if pd.notna(row['content']) else ""
        
        if data_type in ['sql', 'ddl'] and content:
            tables = extract_table_names(content)
            
            for db_name, table_name in tables:
                if table_name not in all_tables:
                    all_tables[table_name] = {
                        'databases': set(),
                        'count': 0,
                        'unqualified_count': 0
                    }
                
                all_tables[table_name]['count'] += 1
                
                if db_name:
                    all_tables[table_name]['databases'].add(db_name)
                else:
                    all_tables[table_name]['unqualified_count'] += 1
    
    return all_tables

def check_needs_fix(content, table_names):
    """
    检查内容是否需要修复
    返回: (needs_fix, unqualified_tables)
    """
    unqualified = []
    
    for table in table_names:
        # 检查是否存在未限定的表名引用
        patterns = [
            rf'\bFROM\s+{table}\b(?!\s*[a-zA-Z_])',
            rf'\bJOIN\s+{table}\b(?!\s*[a-zA-Z_])',
            rf'\bINTO\s+{table}\b(?!\s*[a-zA-Z_])',
            rf'\bUPDATE\s+{table}\b(?!\s*[a-zA-Z_])',
            rf'\bTABLE\s+{table}\b(?!\s*[a-zA-Z_])',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # 确认不是已经有数据库前缀的
                qualified_pattern = rf'\b[a-zA-Z_][a-zA-Z0-9_]*\.{table}\b'
                if not re.search(qualified_pattern, content, re.IGNORECASE):
                    if table not in unqualified:
                        unqualified.append(table)
                    break
    
    return len(unqualified) > 0, unqualified

def fix_content(content, table_names, database_name):
    """
    修复内容，添加数据库前缀
    """
    fixed = content
    
    for table in table_names:
        # 替换各种SQL语句中的表名
        patterns_replacements = [
            (rf'\bFROM\s+{table}\b', f'FROM {database_name}.{table}'),
            (rf'\bJOIN\s+{table}\b', f'JOIN {database_name}.{table}'),
            (rf'\bINTO\s+{table}\b', f'INTO {database_name}.{table}'),
            (rf'\bUPDATE\s+{table}\b', f'UPDATE {database_name}.{table}'),
            (rf'\bTABLE\s+{table}\b', f'TABLE {database_name}.{table}'),
        ]
        
        for pattern, replacement in patterns_replacements:
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
    
    return fixed

def main():
    print("="*80)
    print("[INFO] 自动检查和修复知识库训练数据")
    print("="*80 + "\n")
    
    # 获取训练数据
    print("[INFO] 正在加载训练数据...")
    try:
        df = vn.get_training_data()
    except Exception as e:
        print(f"❌ 获取训练数据失败: {e}")
        return
    
    if df is None or df.empty:
        print("[INFO] 知识库中没有训练数据")
        return
    
    print(f"[OK] 已加载 {len(df)} 条训练数据")
    print(f"   - SQL: {len(df[df['training_data_type'] == 'sql'])} 条")
    print(f"   - DDL: {len(df[df['training_data_type'] == 'ddl'])} 条")
    print(f"   - Documentation: {len(df[df['training_data_type'] == 'documentation'])} 条")
    print()
    
    # 分析表名
    print("[INFO] 正在分析表名和数据库引用...")
    table_info = analyze_training_data(df)
    
    if not table_info:
        print("[INFO] 未检测到任何表引用")
        return
    
    print(f"\n[STAT] 检测到的表名统计:")
    print("-" * 80)
    
    all_table_names = []
    detected_databases = set()
    
    for table_name, info in sorted(table_info.items(), key=lambda x: x[1]['count'], reverse=True):
        all_table_names.append(table_name)
        status = "[OK]" if info['unqualified_count'] == 0 else "[WARN]"
        
        print(f"{status} {table_name}:")
        print(f"   引用次数: {info['count']}")
        print(f"   未限定次数: {info['unqualified_count']}")
        
        if info['databases']:
            detected_databases.update(info['databases'])
            print(f"   检测到的数据库: {', '.join(info['databases'])}")
        print()
    
    # 确定数据库名
    if detected_databases:
        if len(detected_databases) == 1:
            default_db = list(detected_databases)[0]
            print(f"[OK] 检测到数据库名: {default_db}")
        else:
            print(f"[WARN] 检测到多个数据库名: {', '.join(detected_databases)}")
            default_db = list(detected_databases)[0]
            print(f"   将使用: {default_db}")
    else:
        default_db = os.getenv('DB_NAME', 'your_database')
        print(f"[WARN] 未检测到数据库名，使用默认: {default_db}")
    
    print(f"\n{'='*80}")
    db_input = input(f"请确认数据库名 (直接回车使用 '{default_db}'): ").strip()
    database_name = db_input if db_input else default_db
    print(f"[OK] 使用数据库名: {database_name}")
    
    # 检查需要修复的数据
    print(f"\n{'='*80}")
    print("[INFO] 正在检查需要修复的数据...")
    print("="*80 + "\n")
    
    items_to_fix = []
    
    for idx, row in df.iterrows():
        data_type = row['training_data_type']
        content = str(row['content']) if pd.notna(row['content']) else ""
        
        if data_type in ['sql', 'ddl'] and content:
            needs_fix, unqualified = check_needs_fix(content, all_table_names)
            
            if needs_fix:
                items_to_fix.append({
                    'id': row['id'],
                    'type': data_type,
                    'content': content,
                    'question': row.get('question', ''),
                    'unqualified_tables': unqualified
                })
    
    if not items_to_fix:
        print("[OK] 所有数据都已正确使用完全限定表名！")
        return
    
    print(f"[WARN] 发现 {len(items_to_fix)} 条需要修复的数据:\n")
    
    for idx, item in enumerate(items_to_fix, 1):
        print(f"{idx}. [{item['type'].upper()}] {item['id']}")
        if item['type'] == 'sql' and item['question']:
            print(f"   问题: {item['question']}")
        print(f"   未限定的表: {', '.join(item['unqualified_tables'])}")
        print(f"   内容: {item['content'][:120]}...")
        print()
    
    # 询问是否修复
    print("="*80)
    response = input("是否立即修复这些数据？(y/n): ").strip().lower()
    
    if response != 'y':
        print("[CANCEL] 用户取消操作")
        return
    
    # 执行修复
    print(f"\n{'='*80}")
    print("[FIX] 开始修复...")
    print("="*80 + "\n")
    
    success_count = 0
    fail_count = 0
    
    for item in items_to_fix:
        try:
            # 修复内容
            fixed_content = fix_content(item['content'], item['unqualified_tables'], database_name)
            
            print(f"[FIX] 修复 [{item['type'].upper()}]")
            print(f"   ID: {item['id']}")
            if item['type'] == 'sql' and item['question']:
                print(f"   问题: {item['question']}")
            print(f"   涉及表: {', '.join(item['unqualified_tables'])}")
            print(f"   修复前: {item['content'][:100]}...")
            print(f"   修复后: {fixed_content[:100]}...")
            
            # 删除旧数据
            vn.remove_training_data(item['id'])
            
            # 添加修复后的数据
            if item['type'] == 'ddl':
                vn.train(ddl=fixed_content)
            elif item['type'] == 'sql':
                vn.train(question=item['question'], sql=fixed_content)
            
            print(f"   [OK] 成功\n")
            success_count += 1
            
        except Exception as e:
            print(f"   [ERROR] 失败: {e}\n")
            fail_count += 1
    
    # 总结
    print("="*80)
    print("[DONE] 修复完成！")
    print("="*80)
    print(f"[OK] 成功: {success_count} 条")
    if fail_count > 0:
        print(f"[ERROR] 失败: {fail_count} 条")
    print(f"[STAT] 总计: {len(items_to_fix)} 条")
    print()
    
    if success_count > 0:
        print("[NEXT] 接下来请:")
        print("   1. 重启应用（如果需要）")
        print("   2. 测试SQL生成功能")
        print("   3. 确认生成的SQL使用了完全限定表名")
        print()

if __name__ == "__main__":
    main()

