import csv
import os
import sys
from app import vn

# Increase CSV field limit for large content
csv.field_size_limit(sys.maxsize)

CSV_FILE = 'ES_RAG_Knowledge_Base_v2.csv'

def import_knowledge_base():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] File not found: {CSV_FILE}")
        return

    print(f"[INFO] Starting import from {CSV_FILE}...")
    
    success_count = 0
    error_count = 0
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Skip empty rows
                if not row['类型'] or not row['内容']:
                    continue
                    
                data_type = row['类型'].strip().lower()
                content = row['内容'].strip()
                question = row.get('用户问题示例', '').strip()
                tag = row.get('业务标签', '').strip()
                
                try:
                    if data_type == 'ddl':
                        # For DDL, content is the DDL statement
                        vn.train(ddl=content)
                        print(f"  [DDL] Imported: {tag}")
                        
                    elif data_type == 'documentation':
                        # For documentation, content is the doc text
                        vn.train(documentation=content)
                        print(f"  [DOC] Imported: {tag}")
                        
                    elif data_type == 'sql':
                        # For SQL, we need both question and sql (content)
                        if not question:
                            print(f"  [WARN] Skipping SQL without question: {content[:30]}...")
                            continue
                        vn.train(question=question, sql=content)
                        print(f"  [SQL] Imported: {question}")
                        
                    else:
                        print(f"  [WARN] Unknown type: {data_type}")
                        continue
                        
                    success_count += 1
                    
                except Exception as e:
                    print(f"  [ERROR] Failed to import row: {e}")
                    error_count += 1

    except Exception as e:
        print(f"[FATAL] Error reading CSV: {e}")
        return

    print(f"\n[DONE] Import completed.")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")

if __name__ == "__main__":
    import_knowledge_base()

