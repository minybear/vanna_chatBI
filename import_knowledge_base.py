"""
知识库批量导入脚本
用法: python import_knowledge_base.py [--clear] <csv_file>

--clear: 先清空现有知识库再导入
"""

import csv
import sys
import os
from app import vn

def clear_all_training_data():
    """清空所有训练数据"""
    print("[INFO] Clearing all existing training data...")
    
    # 删除所有 collections
    try:
        vn.remove_collection("sql")
        print("  - Cleared SQL collection")
    except Exception as e:
        print(f"  - SQL collection clear failed (may not exist): {e}")
    
    try:
        vn.remove_collection("ddl")
        print("  - Cleared DDL collection")
    except Exception as e:
        print(f"  - DDL collection clear failed (may not exist): {e}")
    
    try:
        vn.remove_collection("documentation")
        print("  - Cleared Documentation collection")
    except Exception as e:
        print(f"  - Documentation collection clear failed (may not exist): {e}")
    
    print("[INFO] All training data cleared.\n")

def import_from_csv(csv_file):
    """从 CSV 文件导入知识库"""
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        return
    
    print(f"[INFO] Importing from: {csv_file}")
    
    stats = {"ddl": 0, "documentation": 0, "sql": 0, "skipped": 0, "error": 0}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader, start=1):
            try:
                data_type = row.get('类型', '').strip().lower()
                content = row.get('内容', '').strip()
                question = row.get('用户问题示例', '').strip()
                # tags = row.get('业务标签', '').strip()  # 可扩展用于过滤
                
                if not content:
                    print(f"  [SKIP] Row {i}: Empty content")
                    stats["skipped"] += 1
                    continue
                
                if data_type == 'ddl':
                    vn.train(ddl=content)
                    stats["ddl"] += 1
                    print(f"  [DDL] Row {i}: Added ({len(content)} chars)")
                    
                elif data_type == 'documentation':
                    vn.train(documentation=content)
                    stats["documentation"] += 1
                    print(f"  [DOC] Row {i}: Added ({len(content)} chars)")
                    
                elif data_type == 'sql':
                    if not question:
                        print(f"  [SKIP] Row {i}: SQL type requires a question")
                        stats["skipped"] += 1
                        continue
                    vn.train(question=question, sql=content)
                    stats["sql"] += 1
                    print(f"  [SQL] Row {i}: Added Q: {question[:50]}...")
                    
                else:
                    print(f"  [SKIP] Row {i}: Unknown type '{data_type}'")
                    stats["skipped"] += 1
                    
            except Exception as e:
                print(f"  [ERROR] Row {i}: {e}")
                stats["error"] += 1
    
    print("\n" + "="*50)
    print("[SUMMARY]")
    print(f"  DDL:           {stats['ddl']}")
    print(f"  Documentation: {stats['documentation']}")
    print(f"  SQL Pairs:     {stats['sql']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Errors:        {stats['error']}")
    print("="*50)
    print("\n[DONE] Import completed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_knowledge_base.py [--clear] <csv_file>")
        print("  --clear: Clear existing training data before import")
        sys.exit(1)
    
    clear_flag = "--clear" in sys.argv
    csv_file = [arg for arg in sys.argv[1:] if arg != "--clear"][0]
    
    if clear_flag:
        clear_all_training_data()
    
    import_from_csv(csv_file)

