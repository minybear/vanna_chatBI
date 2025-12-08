import os
import pymysql
from dotenv import load_dotenv
from app import vn  # Import the initialized Vanna instance

# Load environment variables
load_dotenv()

def get_db_connection(database=None):
    config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'ssl': {
            'check_hostname': False,
            'verify_mode': False # equivalent to CERT_NONE
        } if os.getenv('DB_SSL_ENABLED', 'True').lower() == 'true' else None
    }
    if database:
        config['database'] = database

    return pymysql.connect(**config)

def train_rag():
    target_dbs = os.getenv('TARGET_DATABASES', '').split(',')
    target_dbs = [db.strip() for db in target_dbs if db.strip()]

    if not target_dbs:
        print("ERROR: No databases specified in TARGET_DATABASES in .env file.")
        print("Please edit .env and add your database names (e.g., TARGET_DATABASES=crm,sales)")
        return

    print(f"Starting RAG training for databases: {target_dbs}")

    try:
        # Connect to get DDL
        # We connect without a specific DB first, then switch or query information_schema
        cnx = get_db_connection()
        cursor = cnx.cursor()

        for db_name in target_dbs:
            print(f"\n--- Processing Database: {db_name} ---")

            # Check if DB exists
            try:
                cursor.execute(f"USE {db_name}")
            except pymysql.Error as err:
                print(f"Skipping {db_name}: {err}")
                continue

            # Get all tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                print(f"  Training table: {table_name}...")

                # Get Create Table Statement
                try:
                    cursor.execute(f"SHOW CREATE TABLE {table_name}")
                    create_statement = cursor.fetchone()[1]

                    # CRITICAL: Modify DDL to include database name for cross-db context
                    # Replace "CREATE TABLE `table`" with "CREATE TABLE `db`.`table`"
                    # This helps the LLM understand which DB the table belongs to.

                    if f"CREATE TABLE `{table_name}`" in create_statement:
                        ddl_to_train = create_statement.replace(
                            f"CREATE TABLE `{table_name}`",
                            f"CREATE TABLE `{db_name}`.`{table_name}`"
                        )
                    elif f"CREATE TABLE {table_name}" in create_statement:
                         ddl_to_train = create_statement.replace(
                            f"CREATE TABLE {table_name}",
                            f"CREATE TABLE `{db_name}`.`{table_name}`"
                        )
                    else:
                        # Fallback: Prepend a comment or just use as is
                        ddl_to_train = f"-- Database: {db_name}\n{create_statement}"

                    # Send to Vanna
                    vn.train(ddl=ddl_to_train)

                    # OPTIONAL: Extract table comment to boost relevance
                    # Simple parsing to find COMMENT='...'
                    import re
                    match = re.search(r"COMMENT='([^']*)'", create_statement)
                    if match:
                        comment = match.group(1)
                        if comment:
                            doc = f"Table `{db_name}`.`{table_name}`: {comment}"
                            vn.train(documentation=doc)
                            print(f"    + Added doc: {doc}")

                except Exception as e:
                    print(f"  Error training table {table_name}: {e}")

        cursor.close()
        cnx.close()
        print("\nSUCCESS: RAG Training completed!")
        print("You can now run 'docker-compose up' to start the ChatBI server.")

    except pymysql.Error as err:
        print(f"Database Error: {err}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    train_rag()