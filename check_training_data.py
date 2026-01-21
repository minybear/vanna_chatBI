"""
检查训练数据脚本 - 只检查不修复，列出需要修复的项目
"""
import os
import re
import pandas as pd
from dotenv import load_dotenv
from app import vn

# 加载环境变量
load_dotenv()

try:
    from training_data_config import DATABASE_NAME, TABLE_NAMES
    print(f"✅ 已加载配置: 数据库={DATABASE_NAME}, 表数量={len(TABLE_NAMES)}")
except ImportError:
    print("⚠️  未找到 training_data_config.py，使用默认配置")
    DATABASE_NAME = "your_database_name"
    TABLE_NAMES = ["control_cart"]

def check_table_references(content):
    """
    检查内容中的表引用情况
    返回: {
        'has_unqualified': bool,  # 是否有未限定的表名
        'unqualified_tables': list,  # 未限定的表名
        'qualified_tables': list,  # 已限定的表名
    }
    """
    result = {
        'has_unqualified': False,
        'unqualified_tables': [],
        'qualified_tables': []
    }
    
    for table in TABLE_NAMES:
        # 查找所有表引用（FROM, JOIN, INTO, UPDATE, TABLE等）
        sql_keywords = ['FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE']
        
        for keyword in sql_keywords:
            # 检查未限定的表名: FROM table (没有database.前缀)
            unqualified_pattern = rf'\b{keyword}\s+{table}\b(?!\s*\.)'
            qualified_pattern = rf'\b{keyword}\s+\w+\.{table}\b'
            
            if re.search(unqualified_pattern, content, re.IGNORECASE):
                # 确认不是已经有前缀的
                if not re.search(qualified_pattern, content, re.IGNORECASE):
                    result['has_unqualified'] = True
                    if table not in result['unqualified_tables']:
                        result['unqualified_tables'].append(table)
            
            if re.search(qualified_pattern, content, re.IGNORECASE):
                if table not in result['qualified_tables']:
                    result['qualified_tables'].append(table)
    
    return result

def main():
    print("="*80)
    print("📊 训练数据检查报告")
    print("="*80)
    print(f"数据库名称: {DATABASE_NAME}")
    print(f"监控表数量: {len(TABLE_NAMES)}")
    print(f"表名列表: {', '.join(TABLE_NAMES)}")
    print("="*80 + "\n")
    
    # 获取所有训练数据
    try:
        df = vn.get_training_data()
    except Exception as e:
        print(f"❌ 获取训练数据失败: {e}")
        return
    
    if df is None or df.empty:
        print("📭 知识库中没有训练数据")
        return
    
    print(f"📚 知识库统计:")
    print(f"   总计: {len(df)} 条")
    print(f"   SQL: {len(df[df['training_data_type'] == 'sql'])} 条")
    print(f"   DDL: {len(df[df['training_data_type'] == 'ddl'])} 条")
    print(f"   Documentation: {len(df[df['training_data_type'] == 'documentation'])} 条")
    print()
    
    # 分类统计
    needs_fix = []
    already_good = []
    no_table_refs = []
    
    for idx, row in df.iterrows():
        data_type = row['training_data_type']
        content = str(row['content']) if pd.notna(row['content']) else ""
        data_id = row['id']
        
        # 只检查 SQL 和 DDL
        if data_type in ['sql', 'ddl'] and content:
            check_result = check_table_references(content)
            
            if check_result['has_unqualified']:
                needs_fix.append({
                    'id': data_id,
                    'type': data_type,
                    'content': content,
                    'question': row.get('question', ''),
                    'unqualified_tables': check_result['unqualified_tables'],
                    'qualified_tables': check_result['qualified_tables']
                })
            elif check_result['qualified_tables']:
                already_good.append({
                    'id': data_id,
                    'type': data_type,
                    'tables': check_result['qualified_tables']
                })
            else:
                no_table_refs.append({
                    'id': data_id,
                    'type': data_type
                })
    
    print("="*80)
    print("🔍 检查结果:")
    print("="*80 + "\n")
    
    print(f"✅ 已正确使用完全限定表名: {len(already_good)} 条")
    print(f"⚠️  需要修复（缺少数据库前缀）: {len(needs_fix)} 条")
    print(f"ℹ️  未引用监控表（可能是其他表或纯文档）: {len(no_table_refs)} 条")
    print()
    
    if needs_fix:
        print("="*80)
        print("⚠️  需要修复的数据明细:")
        print("="*80 + "\n")
        
        for idx, item in enumerate(needs_fix, 1):
            print(f"{idx}. [{item['type'].upper()}] {item['id']}")
            if item['type'] == 'sql' and item['question']:
                print(f"   问题: {item['question']}")
            print(f"   未限定的表: {', '.join(item['unqualified_tables'])}")
            if item['qualified_tables']:
                print(f"   已限定的表: {', '.join(item['qualified_tables'])}")
            print(f"   内容: {item['content'][:150]}...")
            print(f"   建议: 将 {', '.join(item['unqualified_tables'])} 改为 {', '.join([f'{DATABASE_NAME}.{t}' for t in item['unqualified_tables']])}")
            print()
    
    if already_good:
        print("="*80)
        print("✅ 已正确配置的数据示例:")
        print("="*80 + "\n")
        
        for idx, item in enumerate(already_good[:3], 1):  # 只显示前3个
            print(f"{idx}. [{item['type'].upper()}] {item['id']}")
            print(f"   使用的表: {', '.join(item['tables'])}")
            print()
    
    print("="*80)
    print("📋 总结:")
    print("="*80)
    if needs_fix:
        print(f"⚠️  发现 {len(needs_fix)} 条数据需要修复")
        print(f"💡 下一步: 运行 'python fix_training_data.py' 进行自动修复")
    else:
        print("✅ 所有数据都已正确配置！")
    print()

if __name__ == "__main__":
    main()

