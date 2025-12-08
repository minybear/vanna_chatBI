import os
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_dbs():
    # PyMySQL configuration
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

    try:
        cnx = pymysql.connect(**config)
        cursor = cnx.cursor()
        cursor.execute("SHOW DATABASES")

        databases = [db[0] for db in cursor.fetchall()]

        # Filter out system databases
        system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
        user_dbs = [db for db in databases if db not in system_dbs]

        print("--- FOUND DATABASES ---")
        print(",".join(user_dbs))
        print("--- END ---")

        cursor.close()
        cnx.close()

    except pymysql.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    list_dbs()