import os
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from fastapi import FastAPI, Response, Body
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from openai import OpenAI

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Initialize ChromaDB
        ChromaDB_VectorStore.__init__(self, config=config)

        # Initialize OpenAI Client for Zhipu AI
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['api_base']
        )

        # Remove api_base from config to avoid Vanna legacy check error
        vanna_config = config.copy()
        if 'api_base' in vanna_config:
            del vanna_config['api_base']

        # Initialize OpenAI_Chat with the client
        OpenAI_Chat.__init__(self, client=client, config=vanna_config)

    def run_sql(self, sql: str, **kwargs):
        # Custom SQL Runner to handle SSL and Multi-DB

        db_config = {
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
            cnx = pymysql.connect(**db_config)
            cursor = cnx.cursor()
            cursor.execute(sql)

            # Fetch column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch data
            data = cursor.fetchall()

            # Convert to pandas DataFrame
            df = pd.DataFrame(data, columns=columns)

            cursor.close()
            cnx.close()
            return df

        except pymysql.Error as err:
            print(f"SQL Error: {err}")
            raise err

# Configuration for Zhipu AI (GLM-4)
config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_MODEL', 'GLM-4.6'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': './chroma_db' # Path for ChromaDB storage
}

# Initialize Vanna
vn = MyVanna(config=config)

# Setup FastAPI
app = FastAPI(title="Vanna ChatBI MVP")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Manual API Routes Implementation ---

class QuestionRequest(BaseModel):
    question: str

class SqlRequest(BaseModel):
    sql: str

@app.post("/api/v0/generate_sql")
def generate_sql(request: QuestionRequest):
    sql = vn.generate_sql(question=request.question)
    return {"sql": sql}

@app.post("/api/v0/run_sql")
def run_sql(request: SqlRequest):
    df = vn.run_sql(sql=request.sql)
    # Convert DataFrame to list of dicts for JSON response
    return {"result": df.to_dict(orient='records'), "columns": df.columns.tolist()}

@app.post("/api/v0/generate_questions")
def generate_questions():
    # Simple placeholder or call vn.generate_questions() if available
    return ["How many users are there?", "Show me the latest orders"]

@app.get("/")
def root():
    return {"message": "Vanna ChatBI MVP is running. Use /api/v0/ for API endpoints."}

# Simple HTML UI for testing
@app.get("/ui", response_class=Response)
def ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vanna ChatBI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <style>
            body { background-color: #f8f9fa; height: 100vh; display: flex; flex-direction: column; }
            .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; max-width: 900px; margin: 0 auto; width: 100%; }
            .message { max-width: 85%; padding: 15px; border-radius: 15px; position: relative; animation: fadeIn 0.3s ease; }
            .message.user { align-self: flex-end; background-color: #0d6efd; color: white; border-bottom-right-radius: 5px; }
            .message.ai { align-self: flex-start; background-color: white; border: 1px solid #dee2e6; border-bottom-left-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .sql-block { background-color: #212529; color: #00ff9d; padding: 10px; border-radius: 5px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; font-size: 0.9em; }
            .table-responsive { margin-top: 10px; border-radius: 5px; overflow: hidden; border: 1px solid #dee2e6; }
            .table { margin-bottom: 0; font-size: 0.9rem; }
            .table th { background-color: #f1f3f5; }
            .input-area { background-color: white; padding: 20px; border-top: 1px solid #dee2e6; }
            .input-group { max-width: 900px; margin: 0 auto; box-shadow: 0 0 15px rgba(0,0,0,0.05); border-radius: 50px; overflow: hidden; }
            .form-control { border: none; padding: 15px 20px; font-size: 1rem; }
            .form-control:focus { box-shadow: none; }
            .btn-send { border-radius: 0; padding: 0 25px; }

            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            /* Loading Dots */
            .typing-indicator { display: inline-flex; gap: 5px; }
            .typing-indicator span { width: 8px; height: 8px; background-color: #adb5bd; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
            .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        </style>
    </head>
    <body>
        <!-- Header -->
        <nav class="navbar navbar-expand-lg navbar-light bg-white border-bottom shadow-sm">
            <div class="container-fluid" style="max-width: 900px;">
                <a class="navbar-brand fw-bold text-primary" href="#">
                    <i class="bi bi-database-fill-gear me-2"></i>Vanna ChatBI
                </a>
                <span class="badge bg-light text-dark border">MVP Environment</span>
            </div>
        </nav>

        <!-- Chat Area -->
        <div class="chat-container" id="chatBox">
            <div class="message ai">
                <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                Hello! I'm connected to your database. Ask me anything about your data!
                <div class="mt-2 text-muted small">Try: "Show me the top 5 users" or "Count orders by status"</div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="input-area">
            <div class="input-group">
                <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about your data..." onkeypress="if(event.key==='Enter') ask()">
                <button class="btn btn-primary btn-send" onclick="ask()">
                    <i class="bi bi-send-fill"></i>
                </button>
            </div>
            <div class="text-center mt-2 text-muted small">Powered by Vanna.ai & GLM-4</div>
        </div>

        <script>
            async function ask() {
                const input = document.getElementById('questionInput');
                const chatBox = document.getElementById('chatBox');
                const question = input.value.trim();
                if (!question) return;

                // Add User Message
                const userDiv = document.createElement('div');
                userDiv.className = 'message user';
                userDiv.innerHTML = `<div>${question}</div>`;
                chatBox.appendChild(userDiv);

                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Add Loading Indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message ai';
                loadingDiv.id = 'loadingMsg';
                loadingDiv.innerHTML = `
                    <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                    <div class="typing-indicator"><span></span><span></span><span></span></div>
                `;
                chatBox.appendChild(loadingDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    // 1. Generate SQL
                    const sqlRes = await fetch('/api/v0/generate_sql', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: question})
                    });
                    const sqlData = await sqlRes.json();

                    if (!sqlRes.ok) throw new Error(sqlData.detail || 'Failed to generate SQL');

                    const sql = sqlData.sql;

                    // Update Loading to show SQL
                    loadingDiv.innerHTML = `
                        <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                        <div>I generated this SQL for you:</div>
                        <div class="sql-block">${sql}</div>
                        <div class="typing-indicator mt-2"><span></span><span></span><span></span></div>
                        <div class="small text-muted mt-1">Running query...</div>
                    `;

                    // 2. Run SQL
                    const runRes = await fetch('/api/v0/run_sql', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({sql: sql})
                    });
                    const runData = await runRes.json();

                    if (!runRes.ok) throw new Error(runData.detail || 'Failed to run SQL');

                    // Render Table
                    let tableHtml = '<div class="table-responsive"><table class="table table-hover table-striped">';
                    tableHtml += '<thead><tr>';
                    runData.columns.forEach(col => tableHtml += `<th scope="col">${col}</th>`);
                    tableHtml += '</tr></thead><tbody>';

                    if (runData.result.length === 0) {
                        tableHtml += '<tr><td colspan="' + runData.columns.length + '" class="text-center text-muted">No results found</td></tr>';
                    } else {
                        runData.result.forEach(row => {
                            tableHtml += '<tr>';
                            runData.columns.forEach(col => tableHtml += `<td>${row[col]}</td>`);
                            tableHtml += '</tr>';
                        });
                    }
                    tableHtml += '</tbody></table></div>';

                    // Final Update
                    loadingDiv.innerHTML = `
                        <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                        <div>Here are the results:</div>
                        <div class="sql-block">${sql}</div>
                        ${tableHtml}
                    `;
                    loadingDiv.id = ''; // Remove ID so we don't select it next time

                } catch (e) {
                    loadingDiv.innerHTML = `
                        <div class="fw-bold mb-1 text-danger"><i class="bi bi-exclamation-triangle-fill me-2"></i>Error</div>
                        <div class="text-danger">${e.message}</div>
                    `;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
