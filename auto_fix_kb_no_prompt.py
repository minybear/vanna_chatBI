"""
自动修复知识库 - 无交互版本
直接执行修复，使用检测到的数据库名
"""
import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from app import vn

# 加载环境变量
load_dotenv()

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def extract_table_names(content):
    """从SQL/DDL内容中提取表名"""
    tables = set()
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
            db_prefix = match.group(1)
            table_name = match.group(2)
            sql_keywords = {'select', 'from', 'where', 'join', 'left', 'right', 
                           'inner', 'outer', 'on', 'and', 'or', 'not', 'exists', 'current_timestamp'}
            if table_name.lower() not in sql_keywords:
                tables.add((db_prefix.rstrip('.') if db_prefix else None, table_name))
    
    return tables

def analyze_training_data(df):
    """分析训练数据，提取所有表名和数据库名"""
    all_tables = {}
    table_db_map = {}  # 记录每个表最常用的数据库
    
    for idx, row in df.iterrows():
        data_type = row['training_data_type']
        content = str(row['content']) if pd.notna(row['content']) else ""
        
        if data_type in ['sql', 'ddl'] and content:
            tables = extract_table_names(content)
            
            for db_name, table_name in tables:
                if table_name not in all_tables:
                    all_tables[table_name] = {
                        'databases': {},
                        'count': 0,
                        'unqualified_count': 0
                    }
                
                all_tables[table_name]['count'] += 1
                
                if db_name:
                    all_tables[table_name]['databases'][db_name] = \
                        all_tables[table_name]['databases'].get(db_name, 0) + 1
                else:
                    all_tables[table_name]['unqualified_count'] += 1
    
    # 为每个表确定最佳数据库
    for table_name, info in all_tables.items():
        if info['databases']:
            # 使用出现次数最多的数据库
            best_db = max(info['databases'].items(), key=lambda x: x[1])[0]
            table_db_map[table_name] = best_db
    
    return all_tables, table_db_map

def check_needs_fix(content, table_names):
    """检查内容是否需要修复 - 使用简单直接的方法"""
    unqualified = []
    
    for table in table_names:
        # 查找所有关键字+表名的组合
        keywords = ['FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE']
        
        for keyword in keywords:
            # 查找 "KEYWORD table" 的所有匹配
            pattern = rf'\b{keyword}\s+(\w+\.)?{table}\b'
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            
            for match in matches:
                # group(1) 是可选的 "database." 前缀
                db_prefix = match.group(1)
                
                # 如果没有前缀，说明是未限定的
                if not db_prefix:
                    if table not in unqualified:
                        unqualified.append(table)
                    break
            
            if table in unqualified:
                break
    
    return len(unqualified) > 0, unqualified

def fix_content(content, table_db_map):
    """修复内容，根据表名映射添加数据库前缀"""
    fixed = content
    
    for table, database in table_db_map.items():
        patterns_replacements = [
            (rf'\bFROM\s+{table}\b', f'FROM {database}.{table}'),
            (rf'\bJOIN\s+{table}\b', f'JOIN {database}.{table}'),
            (rf'\bINTO\s+{table}\b', f'INTO {database}.{table}'),
            (rf'\bUPDATE\s+{table}\b', f'UPDATE {database}.{table}'),
            (rf'\bTABLE\s+{table}\b', f'TABLE {database}.{table}'),
        ]
        
        for pattern, replacement in patterns_replacements:
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
    
    return fixed

def main():
    print("="*80)
    print("[自动修复] 知识库训练数据 - 无交互模式")
    print("="*80 + "\n")
    
    # 获取训练数据
    print("[步骤1] 加载训练数据...")
    try:
        df = vn.get_training_data()
    except Exception as e:
        print(f"[错误] 获取训练数据失败: {e}")
        return
    
    if df is None or df.empty:
        print("[信息] 知识库中没有训练数据")
        return
    
    print(f"[成功] 已加载 {len(df)} 条训练数据")
    print(f"  - SQL: {len(df[df['training_data_type'] == 'sql'])} 条")
    print(f"  - DDL: {len(df[df['training_data_type'] == 'ddl'])} 条")
    print(f"  - Documentation: {len(df[df['training_data_type'] == 'documentation'])} 条\n")
    
    # 分析表名
    print("[步骤2] 分析表名和数据库引用...")
    table_info, table_db_map = analyze_training_data(df)
    
    if not table_info:
        print("[信息] 未检测到任何表引用")
        return
    
    print(f"\n[统计] 检测到 {len(table_info)} 个表:\n")
    
    all_databases = set()
    for table_name, info in sorted(table_info.items(), key=lambda x: x[1]['count'], reverse=True):
        status = "[OK]" if info['unqualified_count'] == 0 else "[需修复]"
        print(f"{status} {table_name}: 引用{info['count']}次, 未限定{info['unqualified_count']}次", end="")
        
        if info['databases']:
            all_databases.update(info['databases'].keys())
            db_list = [f"{db}({count})" for db, count in info['databases'].items()]
            print(f", 数据库: {', '.join(db_list)}")
            if table_name in table_db_map:
                print(f"    -> 将使用: {table_db_map[table_name]}")
        else:
            print()
    
    print(f"\n[检测] 涉及数据库: {', '.join(all_databases)}")
    
    # 为没有数据库映射的表指定默认数据库
    default_db = list(all_databases)[0] if all_databases else 'db_provision'
    for table_name in table_info.keys():
        if table_name not in table_db_map:
            table_db_map[table_name] = default_db
            print(f"[默认] {table_name} -> {default_db}")
    print()
    
    # 检查需要修复的数据
    print("[步骤3] 检查需要修复的数据...")
    items_to_fix = []
    
    # 获取所有表名列表（不只是有数据库映射的）
    all_table_names = list(table_info.keys())
    
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
        print("[完成] 所有数据都已正确使用完全限定表名！")
        return
    
    print(f"\n[发现] {len(items_to_fix)} 条数据需要修复\n")
    
    # 显示前5条示例
    print("[示例] 需要修复的数据(前5条):")
    for idx, item in enumerate(items_to_fix[:5], 1):
        print(f"\n{idx}. [{item['type'].upper()}] {item['id']}")
        if item['type'] == 'sql' and item['question']:
            print(f"   问题: {item['question']}")
        print(f"   未限定表: {', '.join(item['unqualified_tables'])}")
        print(f"   内容: {item['content'][:100]}...")
    
    if len(items_to_fix) > 5:
        print(f"\n   ... 还有 {len(items_to_fix) - 5} 条\n")
    
    # 执行修复
    print("\n" + "="*80)
    print("[步骤4] 开始自动修复...")
    print("="*80 + "\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(items_to_fix, 1):
        try:
            # 为每个未限定的表确定数据库
            item_db_map = {table: table_db_map.get(table, list(all_databases)[0]) 
                          for table in item['unqualified_tables']}
            
            # 修复内容
            fixed_content = fix_content(item['content'], item_db_map)
            
            print(f"[{idx}/{len(items_to_fix)}] {item['id'][:20]}... ", end="")
            
            # 删除旧数据
            vn.remove_training_data(item['id'])
            
            # 添加修复后的数据
            if item['type'] == 'ddl':
                vn.train(ddl=fixed_content)
            elif item['type'] == 'sql':
                vn.train(question=item['question'], sql=fixed_content)
            
            print("[成功]")
            success_count += 1
            
        except Exception as e:
            print(f"[失败] {e}")
            fail_count += 1
    
    # 总结
    print("\n" + "="*80)
    print("[完成] 修复完成！")
    print("="*80)
    print(f"[成功] {success_count} 条")
    if fail_count > 0:
        print(f"[失败] {fail_count} 条")
    print(f"[总计] {len(items_to_fix)} 条")
    print()
    
    if success_count > 0:
        print("[下一步] 请:")
        print("  1. 测试SQL生成功能")
        print("  2. 确认生成的SQL使用了完全限定表名")
        print()

if __name__ == "__main__":
    main()

