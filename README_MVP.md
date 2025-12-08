# Vanna ChatBI MVP

This is a Minimum Viable Product (MVP) for a ChatBI system using Vanna, MariaDB, and Zhipu AI (GLM-4).

## Prerequisites

*   Docker Desktop (Windows)
*   Python 3.9+ (for running the setup script locally)
*   MariaDB Database Access

## Setup Instructions

### 1. Configure Environment

Edit the `.env` file in this directory:

1.  **Database Name:** Fill in `TARGET_DATABASES` with the comma-separated list of databases you want to query (e.g., `TARGET_DATABASES=crm,sales`).
2.  **Credentials:** Verify `DB_USER`, `DB_PASSWORD`, and `DB_HOST`.
3.  **API Key:** Verify `ZHIPU_API_KEY`.

### 2. Train the AI (RAG Setup)

Before running the server, you need to "teach" the AI about your database structure.

Run the setup script locally:

```bash
pip install -r requirements.txt
python setup_rag.py
```

This script will:
*   Connect to your MariaDB.
*   Read the schema (DDL) for all tables in the specified `TARGET_DATABASES`.
*   Store the training data in a local `chroma_db` folder.

### 3. Start the Server

Run the application using Docker Compose:

```bash
docker-compose up --build
```

### 4. Access the Chat Interface

Open your browser and go to:

[http://localhost:8000/ui](http://localhost:8000/ui)

You can now chat with your database!

## Troubleshooting

*   **Database Connection:** If Docker cannot connect to MariaDB on your host, ensure `host.docker.internal` is working and your firewall allows the connection.
*   **SSL Errors:** If you encounter SSL issues, check the `DB_SSL_VERIFY` setting in `.env`.
