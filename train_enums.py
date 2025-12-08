import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
DRY_RUN = False  # Set to False to actually train
CARDINALITY_THRESHOLD = 20  # Max unique values to consider as an enum
IGNORE_TABLES = ['auditlog', 'schema_migrations'] # Tables to skip
IGNORE_COLUMNS = ['id', 'created_at', 'updated_at', 'password'] # Columns to skip

# --- Vanna Setup (Copied from app.py) ---
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['api_base']
        )
        vanna_config = config.copy()
        if 'api_base' in vanna_config:
            del vanna_config['api_base']
        OpenAI_Chat.__init__(self, client=client, config=vanna_config)

config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_MODEL', 'glm-4'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': './chroma_db'
}

vn = MyVanna(config=config)

# --- Database Connection ---
def get_db_connection(db_name=None):
    db_config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'ssl': {
            'check_hostname': False,
            'verify_mode': False
        } if os.getenv('DB_SSL_ENABLED', 'True').lower() == 'true' else None
    }
    if db_name:
        db_config['database'] = db_name
    return pymysql.connect(**db_config)

def scan_and_train():
    print(f"[INFO] Starting Database Scan (DRY_RUN={DRY_RUN})...")

    target_dbs = os.getenv('TARGET_DATABASES', '').split(',')
    target_dbs = [db.strip() for db in target_dbs if db.strip()]

    if not target_dbs:
        print("[ERROR] No databases found in TARGET_DATABASES env var.")
        return

    for db_name in target_dbs:
        print(f"\n[DATABASE] Scanning database: {db_name}")
        try:
            conn = get_db_connection(db_name)
            cursor = conn.cursor()

            # 1. Get all tables
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                if table.lower() in IGNORE_TABLES:
                    continue

                # print(f"  Analyzing table: {table}")

                # 2. Get columns for the table
                cursor.execute(f"DESCRIBE `{table}`")
                columns = cursor.fetchall() # (Field, Type, Null, Key, Default, Extra)

                for col_info in columns:
                    col_name = col_info[0]
                    col_type = col_info[1].lower()

                    if col_name.lower() in IGNORE_COLUMNS:
                        continue

                    # Skip obviously non-enum types (text blobs, etc.)
                    if 'text' in col_type or 'blob' in col_type:
                        continue

                    try:
                        # 3. Check cardinality
                        query = f"SELECT DISTINCT `{col_name}` FROM `{table}` LIMIT {CARDINALITY_THRESHOLD + 1}"
                        cursor.execute(query)
                        values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]

                        if 0 < len(values) <= CARDINALITY_THRESHOLD:
                            # It's a candidate for enum!
                            values_str = ", ".join([f"'{v}'" for v in values])
                            # Include DB name in documentation to avoid ambiguity
                            doc_text = f"Database `{db_name}`, Table `{table}`, Column `{col_name}` contains values: [{values_str}]."

                            print(f"    [FOUND] {table}.{col_name} -> {values}")

                            if not DRY_RUN:
                                # print(f"       Training Vanna...")
                                vn.train(documentation=doc_text)
                        else:
                            pass

                    except Exception as e:
                        print(f"    [ERROR] analyzing {table}.{col_name}: {e}")

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to scan database {db_name}: {e}")

    print("\n[DONE] Scan Complete.")
    if DRY_RUN:
        print("[WARN] This was a DRY RUN. No data was sent to Vanna. Change DRY_RUN = False in the script to train.")

if __name__ == "__main__":
    scan_and_train()