import os
import re
import requests
from datetime import datetime
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Response, Body
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

# Load environment variables
load_dotenv()

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
zhipu_embedding = ZhipuAIEmbeddingFunction(config={'api_key': os.getenv('ZHIPU_API_KEY')})

config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_MODEL', 'GLM-4.6'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': './chroma_db', # Path for ChromaDB storage
    'embedding_function': zhipu_embedding
}

# Initialize Vanna
vn = MyVanna(config=config)

# Setup FastAPI
app = FastAPI(title="Vanna ChatBI MVP")

# Import and Include Knowledge Base Router
from knowledge_base_api import router as kb_router
app.include_router(kb_router, prefix="/api/v0", tags=["Knowledge Base"])

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
    question: str = None

class FeedbackRequest(BaseModel):
    question: str
    sql: str
    explanation: str = None
    feedback_type: str  # "up" (点赞) 或 "down" (点踩)
    comment: str = None

def send_lark_alert(feedback: FeedbackRequest):
    url = os.getenv("LARK_WEBHOOK_URL")
    if not url:
        print("Error: LARK_WEBHOOK_URL is not set.")
        return

    # 构造飞书卡片消息
    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "🚨 用户反馈：回答不准确 (点踩)"},
                "template": "red"
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**用户提问:**\n{feedback.question}"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**生成的 SQL:**\n```sql\n{feedback.sql}\n```"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**解释:**\n{feedback.explanation}"}},
                {"tag": "hr"},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": f"反馈时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}]}
            ]
        }
    }
    
    try:
        resp = requests.post(url, json=payload)
        print(f"Lark response: {resp.text}")
    except Exception as e:
        print(f"Failed to send Lark alert: {e}")

@app.get("/api/v0/config")
def get_config():
    # Controls whether the frontend displays the raw SQL code block
    # Default to False for production/business users
    show_sql = os.getenv('SHOW_SQL_DEBUG', 'False').lower() == 'true'
    return {"show_sql": show_sql}

@app.post("/api/v0/generate_sql")
def generate_sql(request: QuestionRequest):
    # 1. Generate SQL using Vanna
    # Allow LLM to see data for better SQL generation (e.g. finding distinct values)
    raw_result = vn.generate_sql(question=request.question, allow_llm_to_see_data=True)

    # 2. Clean and Validate SQL
    # Remove Markdown code blocks if present (e.g. ```sql ... ```)
    clean_result = re.sub(r'^```sql\s*', '', raw_result, flags=re.IGNORECASE)
    clean_result = re.sub(r'^```\s*', '', clean_result)
    clean_result = re.sub(r'\s*```$', '', clean_result)

    # Remove "intermediate_sql" marker if present
    clean_result = re.sub(r'^intermediate_sql\s*', '', clean_result, flags=re.IGNORECASE)
    
    # Remove SQL comments (lines starting with -- or #)
    lines = clean_result.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments
        line_no_comment = re.sub(r'\s*--.*$', '', line)
        line_no_comment = re.sub(r'\s*#.*$', '', line_no_comment)
        if line_no_comment.strip():
            cleaned_lines.append(line_no_comment)
    clean_result = '\n'.join(cleaned_lines)
    
    clean_result = clean_result.strip()

    # Heuristic check: Must start with SELECT, WITH, DESCRIBE, SHOW
    # Also check length to avoid short "I can't do that" responses being treated as SQL
    upper_result = clean_result.upper()
    valid_starts = ("SELECT", "WITH", "DESCRIBE", "SHOW")
    is_sql = any(upper_result.startswith(s) for s in valid_starts) and len(clean_result) > 10

    explanation = ""
    if is_sql:
        # 3. Generate Explanation (Human-in-the-loop / Business Friendly)
        try:
            # Use the existing client in vn to generate an explanation
            explanation_prompt = [
                {"role": "system", "content": "You are a helpful business intelligence assistant. Your goal is to explain data queries to non-technical users."},
                {"role": "user", "content": f"The user asked: '{request.question}'.\nWe generated this SQL: \n{clean_result}\n\nPlease provide a brief, clear explanation of what this query does. Focus on the business logic. Do not mention specific table names or SQL keywords. Keep it under 2 sentences."}
            ]

            response = vn.client.chat.completions.create(
                model=config['model'],
                messages=explanation_prompt,
                max_tokens=150,
                temperature=0.7
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating explanation: {e}")
            explanation = "Here is the data based on your request." # Fallback

    if is_sql:
        return {
            "sql": clean_result,
            "is_sql": True,
            "explanation": explanation
        }
    else:
        # It's a conversational response or error
        return {
            "text": raw_result,
            "is_sql": False,
            "explanation": raw_result # Use the text itself as explanation
        }

@app.post("/api/v0/run_sql")
def run_sql(request: SqlRequest):
    df = vn.run_sql(sql=request.sql)
    
    # 清理不符合 JSON 规范的浮点数值 (inf, -inf, nan)
    # 将它们替换为 None，避免 JSON 序列化错误
    import numpy as np
    df = df.replace([np.inf, -np.inf], None)  # 无穷大替换为 None
    df = df.replace({np.nan: None})  # NaN 替换为 None
    
    # --- 新增可视化逻辑 ---
    chart_json = None
    if request.question:
        try:
            # 1. 让 Vanna 生成绘图代码
            plotly_code = vn.generate_plotly_code(question=request.question, sql=request.sql, df=df)
            
            # 2. 执行代码获取 Figure 对象
            fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
            
            # 3. 转为 JSON 字符串传给前端
            if fig:
                chart_json = fig.to_json()
        except Exception as e:
            print(f"可视化生成失败: {e}")
    # --------------------

    # Convert DataFrame to list of dicts for JSON response
    return {
        "result": df.to_dict(orient='records'), 
        "columns": df.columns.tolist(),
        "chart": chart_json
    }

@app.post("/api/v0/feedback")
def submit_feedback(request: FeedbackRequest):
    print(f"收到反馈: {request.feedback_type} - {request.question}")
    
    # 如果是点踩，发送飞书通知
    if request.feedback_type == "down":
        send_lark_alert(request)
        
    return {"status": "success", "message": "Feedback received"}

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
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { background-color: #f8f9fa; height: 100vh; display: flex; flex-direction: column; }
            .main-container { flex: 1; overflow: hidden; display: flex; flex-direction: column; max-width: 1200px; margin: 0 auto; width: 100%; }
            .nav-tabs { padding: 0 20px; border-bottom: none; }
            .tab-content { flex: 1; overflow: hidden; display: flex; flex-direction: column; background: white; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 5px 5px; }
            .tab-pane { height: 100%; display: none; flex-direction: column; }
            .tab-pane.active { display: flex; }
            
            /* Chat Styles */
            .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 85%; padding: 15px; border-radius: 15px; position: relative; animation: fadeIn 0.3s ease; }
            .message.user { align-self: flex-end; background-color: #0d6efd; color: white; border-bottom-right-radius: 5px; }
            .message.ai { align-self: flex-start; background-color: #f8f9fa; border: 1px solid #dee2e6; border-bottom-left-radius: 5px; }
            .sql-block { background-color: #212529; color: #00ff9d; padding: 10px; border-radius: 5px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; font-size: 0.9em; display: none; }
            .explanation-block { background-color: #e9ecef; color: #495057; padding: 10px; border-radius: 5px; margin: 5px 0; font-style: italic; border-left: 4px solid #0d6efd; }
            
            /* Feedback Buttons */
            .feedback-actions { margin-top: 10px; display: flex; gap: 10px; opacity: 0.8; }
            .feedback-btn { border: none; background: none; color: #6c757d; cursor: pointer; padding: 2px 8px; transition: all 0.2s; }
            .feedback-btn:hover { color: #0d6efd; background-color: #e9ecef; border-radius: 4px; }
            .feedback-btn.active-up { color: #198754; font-weight: bold; }
            .feedback-btn.active-down { color: #dc3545; font-weight: bold; }

            /* KB Styles */
            .kb-container { flex: 1; overflow-y: auto; padding: 20px; }
            .type-badge { font-size: 0.8em; padding: 5px 10px; border-radius: 15px; }
            .type-ddl { background-color: #cfe2ff; color: #084298; }
            .type-doc { background-color: #d1e7dd; color: #0f5132; }
            .type-sql { background-color: #fff3cd; color: #664d03; }
            
            /* Table Optimization */
            #kbTable { table-layout: fixed; width: 100%; }
            #kbTable th:nth-child(1) { width: 100px; } /* Type */
            #kbTable th:nth-child(3) { width: 20%; }   /* Question */
            #kbTable th:nth-child(4) { width: 100px; text-align: center; } /* Actions */
            /* Content column takes remaining space */
            
            .content-cell {
                font-family: monospace; 
                font-size: 0.9em; 
                white-space: pre-wrap; 
                max-height: 100px; /* Reduced height */
                overflow-y: auto; 
                background: #f8f9fa;
                padding: 5px;
                border-radius: 4px;
                border: 1px solid #dee2e6;
            }

            .modal { background: rgba(0,0,0,0.5); }
            .modal-content { box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
            
            .input-area { background-color: white; padding: 20px; border-top: 1px solid #dee2e6; }
            .input-group { box-shadow: 0 0 15px rgba(0,0,0,0.05); border-radius: 50px; overflow: hidden; }
            .form-control { border: none; padding: 15px 20px; font-size: 1rem; }
            .form-control:focus { box-shadow: none; }
            .btn-send { border-radius: 0; padding: 0 25px; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .typing-indicator span { width: 8px; height: 8px; background-color: #adb5bd; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; display: inline-block; margin: 0 2px; }
            .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light bg-white border-bottom shadow-sm mb-3">
            <div class="container-fluid" style="max-width: 1200px;">
                <a class="navbar-brand fw-bold text-primary" href="#">
                    <i class="bi bi-database-fill-gear me-2"></i>Vanna ChatBI
                </a>
                <span class="badge bg-light text-dark border">MVP Environment</span>
            </div>
        </nav>

        <div class="main-container">
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="chat-tab" data-bs-toggle="tab" data-bs-target="#chat" type="button" role="tab">Chat</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="kb-tab" data-bs-toggle="tab" data-bs-target="#kb" type="button" role="tab" onclick="loadKnowledgeBase()">Knowledge Base</button>
                </li>
            </ul>
            
            <div class="tab-content" id="myTabContent">
                <!-- Chat Tab -->
                <div class="tab-pane fade show active" id="chat" role="tabpanel">
                    <div class="chat-container" id="chatBox">
                        <div class="message ai">
                            <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                            Hello! I'm connected to your database. Ask me anything about your data!
                        </div>
                    </div>
                    <div class="input-area">
                        <div class="input-group">
                            <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about your data..." onkeypress="if(event.key==='Enter') ask()">
                            <button class="btn btn-primary btn-send" onclick="ask()"><i class="bi bi-send-fill"></i></button>
                        </div>
                    </div>
                </div>
                
                <!-- KB Tab -->
                <div class="tab-pane fade" id="kb" role="tabpanel">
                    <div class="kb-container">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5>Training Data</h5>
                            <div>
                                <button class="btn btn-sm btn-success me-2" onclick="showAddModal()"><i class="bi bi-plus-lg me-1"></i>Add Data</button>
                                <button class="btn btn-sm btn-outline-primary" onclick="loadKnowledgeBase()"><i class="bi bi-arrow-clockwise me-1"></i>Refresh</button>
                            </div>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover" id="kbTable">
                                <thead>
                                    <tr>
                                        <th>Type</th>
                                        <th>Content / SQL</th>
                                        <th>Related Question</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr><td colspan="4" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Add Data Modal -->
        <div class="modal fade" id="addModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Add Training Data</h5>
                        <button type="button" class="btn-close" onclick="hideAddModal()"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Type</label>
                            <select class="form-select" id="addType" onchange="toggleAddFields()">
                                <option value="documentation">Documentation</option>
                                <option value="sql">SQL (QA Pair)</option>
                                <option value="ddl">DDL (Schema)</option>
                            </select>
                        </div>
                        <div class="mb-3" id="questionGroup" style="display:none;">
                            <label class="form-label">Question</label>
                            <input type="text" class="form-control" id="addQuestion" placeholder="e.g. How many users?">
                        </div>
                        <div class="mb-3">
                            <label class="form-label" id="contentLabel">Content</label>
                            <textarea class="form-control" id="addContent" rows="5"></textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" onclick="hideAddModal()">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="submitData()">Save</button>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let APP_CONFIG = { show_sql: false };
            let CHAT_HISTORY = {}; // Store message details for feedback

            window.onload = async () => {
                try {
                    const res = await fetch('/api/v0/config');
                    APP_CONFIG = await res.json();
                } catch (e) { console.error("Config error", e); }
            };

            async function ask() {
                const input = document.getElementById('questionInput');
                const chatBox = document.getElementById('chatBox');
                const question = input.value.trim();
                if (!question) return;

                appendMessage('user', question);
                input.value = '';
                
                const loadingId = 'loading-' + Date.now();
                appendLoading(loadingId);
                
                try {
                    const sqlRes = await fetch('/api/v0/generate_sql', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question})
                    });
                    const sqlData = await sqlRes.json();
                    
                    if (!sqlRes.ok) throw new Error(sqlData.detail || 'Failed to generate SQL');
                    
                    const loadingDiv = document.getElementById(loadingId);
                    
                    if (sqlData.is_sql) {
                        const explanation = sqlData.explanation || "Results:";
                        const displayStyle = APP_CONFIG.show_sql ? 'block' : 'none';
                        const msgId = 'msg-' + Date.now();

                        // Store for feedback
                        CHAT_HISTORY[msgId] = {
                            question: question,
                            sql: sqlData.sql,
                            explanation: explanation
                        };

                        const feedbackHtml = `
                            <div class="feedback-actions">
                                <button class="feedback-btn" onclick="submitFeedback('${msgId}', 'up', this)" title="回答准确">
                                    <i class="bi bi-hand-thumbs-up"></i> 赞
                                </button>
                                <button class="feedback-btn" onclick="submitFeedback('${msgId}', 'down', this)" title="回答有误">
                                    <i class="bi bi-hand-thumbs-down"></i> 踩
                                </button>
                            </div>
                        `;
                        
                        loadingDiv.innerHTML = `
                            <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                            <div class="explanation-block">${explanation}</div>
                            <div class="sql-block" style="display: ${displayStyle}">${sqlData.sql}</div>
                            <div class="typing-indicator mt-2"><span></span><span></span><span></span></div>
                            <div class="small text-muted mt-1">Running query...</div>
                        `;

                        const runRes = await fetch('/api/v0/run_sql', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({sql: sqlData.sql, question: question})
                        });
                        const runData = await runRes.json();
                        
                        if (!runRes.ok) throw new Error(runData.detail || 'Failed to run SQL');
                        
                        const columns = runData.columns || [];
                        const result = runData.result || [];
                        
                        let tableHtml = '<div class="table-responsive mt-2"><table class="table table-sm table-striped table-hover border">';
                        if (columns.length > 0) {
                            tableHtml += '<thead class="table-light"><tr>' + columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
                            if (result.length === 0) {
                                tableHtml += `<tr><td colspan="${columns.length}" class="text-center text-muted p-3">No results found</td></tr>`;
                            } else {
                                result.forEach(row => {
                                    tableHtml += '<tr>' + columns.map(c => `<td>${row[c] !== null ? row[c] : ''}</td>`).join('') + '</tr>';
                                });
                            }
                        } else {
                            tableHtml += '<tbody><tr><td class="text-center text-muted p-3">No data columns returned</td></tr>';
                        }
                        tableHtml += '</tbody></table></div>';
                        
                        loadingDiv.innerHTML = `
                            <div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div>
                            <div class="explanation-block">${explanation}</div>
                            <div class="sql-block" style="display: ${displayStyle}">${sqlData.sql}</div>
                            <div id="chart-container-${msgId}" style="width:100%; height:400px; display:none;"></div>
                            ${tableHtml}
                            ${feedbackHtml}
                        `;

                        if (runData.chart) {
                            try {
                                const chartContainer = document.getElementById(`chart-container-${msgId}`);
                                chartContainer.style.display = 'block';
                                const chartData = JSON.parse(runData.chart);
                                Plotly.newPlot(chartContainer, chartData.data, chartData.layout);
                            } catch (e) {
                                console.error("Chart render error", e);
                            }
                        }

                    } else {
                        loadingDiv.innerHTML = `<div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div><div>${sqlData.text}</div>`;
                    }
                    loadingDiv.id = '';
                } catch (e) {
                    const loadingDiv = document.getElementById(loadingId);
                    if(loadingDiv) {
                        loadingDiv.innerHTML = `<div class="fw-bold mb-1 text-danger">Error</div><div class="text-danger">${e.message}</div>`;
                        loadingDiv.id = '';
                    }
                }
            }

            function appendMessage(role, text) {
                const chatBox = document.getElementById('chatBox');
                const div = document.createElement('div');
                div.className = `message ${role}`;
                div.innerHTML = role === 'user' ? `<div>${text}</div>` : `<div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div><div>${text}</div>`;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function appendLoading(id) {
                const chatBox = document.getElementById('chatBox');
                const div = document.createElement('div');
                div.className = 'message ai';
                div.id = id;
                div.innerHTML = `<div class="fw-bold mb-1"><i class="bi bi-robot me-2"></i>AI Assistant</div><div class="typing-indicator"><span></span><span></span><span></span></div>`;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            let KB_DATA = []; // Store data for access by ID

            async function loadKnowledgeBase() {
                const tbody = document.querySelector('#kbTable tbody');
                tbody.innerHTML = '<tr><td colspan="4" class="text-center p-4"><div class="spinner-border text-primary" role="status"></div><div class="mt-2">Loading training data...</div></td></tr>';
                
                try {
                    const res = await fetch('/api/v0/get_training_data');
                    const data = await res.json();
                    KB_DATA = data || []; // Update global store
                    
                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No training data found.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = data.map(item => {
                        let badgeClass = 'type-doc';
                        let content = item.content || '';
                        if (item.training_data_type === 'ddl') badgeClass = 'type-ddl';
                        if (item.training_data_type === 'sql') badgeClass = 'type-sql';
                        
                        const displayContent = content.length > 300 ? content.substring(0, 300) + '...' : content;
                        // Use ID to find item in editData
                        return `
                        <tr>
                            <td class="align-middle"><span class="badge ${badgeClass} type-badge">${item.training_data_type}</span></td>
                            <td><div class="content-cell">${content}</div></td>
                            <td class="align-middle"><div class="text-truncate" title="${item.question || ''}">${item.question || '-'}</div></td>
                            <td class="align-middle text-center">
                                <button class="btn btn-sm btn-outline-primary me-1" onclick="editData('${item.id}')" title="Edit"><i class="bi bi-pencil"></i></button>
                                <button class="btn btn-sm btn-outline-danger" onclick="deleteData('${item.id}')" title="Delete"><i class="bi bi-trash"></i></button>
                            </td>
                        </tr>`;
                    }).join('');
                } catch (e) {
                    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Error loading data: ${e.message}</td></tr>`;
                }
            }

            // Modal Functions
            const addModal = new bootstrap.Modal(document.getElementById('addModal'));
            let currentEditId = null; // Track if we are editing
            
            function showAddModal() {
                currentEditId = null;
                document.querySelector('.modal-title').innerText = 'Add Training Data';
                document.getElementById('addType').value = 'documentation';
                document.getElementById('addType').disabled = false; 
                document.getElementById('addQuestion').value = '';
                document.getElementById('addContent').value = '';
                toggleAddFields();
                addModal.show();
            }

            function editData(id) {
                const item = KB_DATA.find(d => d.id === id);
                if (!item) {
                    alert('Error: Data not found');
                    return;
                }
                
                currentEditId = item.id;
                document.querySelector('.modal-title').innerText = 'Edit Training Data';
                
                let typeVal = 'documentation';
                if (item.training_data_type === 'ddl') typeVal = 'ddl';
                if (item.training_data_type === 'sql') typeVal = 'sql';
                
                document.getElementById('addType').value = typeVal;
                document.getElementById('addType').disabled = true; 
                
                document.getElementById('addQuestion').value = item.question || '';
                document.getElementById('addContent').value = item.content || '';
                
                toggleAddFields();
                addModal.show();
            }

            function hideAddModal() {
                addModal.hide();
                currentEditId = null;
            }

            function toggleAddFields() {
                const type = document.getElementById('addType').value;
                const qGroup = document.getElementById('questionGroup');
                const cLabel = document.getElementById('contentLabel');
                
                if (type === 'sql') {
                    qGroup.style.display = 'block';
                    cLabel.innerText = 'SQL Query';
                } else if (type === 'ddl') {
                    qGroup.style.display = 'none';
                    cLabel.innerText = 'DDL Statement';
                } else {
                    qGroup.style.display = 'none';
                    cLabel.innerText = 'Documentation Text';
                }
            }

            async function submitData() {
                const type = document.getElementById('addType').value;
                const content = document.getElementById('addContent').value.trim();
                const question = document.getElementById('addQuestion').value.trim();
                
                if (!content) {
                    alert('Content is required');
                    return;
                }
                if (type === 'sql' && !question) {
                    alert('Question is required for SQL type');
                    return;
                }

                try {
                    // If editing, delete old one first
                    if (currentEditId) {
                        const delRes = await fetch(`/api/v0/training_data/${currentEditId}`, {
                             method: 'DELETE'
                        });
                        if (!delRes.ok) {
                             throw new Error('Failed to update: could not remove old record');
                        }
                    }

                    // Add new record
                    const res = await fetch('/api/v0/training_data', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({type, content, question})
                    });
                    
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Failed to save data');
                    }
                    
                    hideAddModal();
                    loadKnowledgeBase(); // Reload table
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }

            async function submitFeedback(msgId, type, btnElement) {
                const data = CHAT_HISTORY[msgId];
                if (!data) return;

                const parent = btnElement.parentElement;
                const buttons = parent.querySelectorAll('.feedback-btn');
                buttons.forEach(b => {
                    b.classList.remove('active-up', 'active-down');
                    b.disabled = true; 
                });
                
                if (type === 'up') btnElement.classList.add('active-up');
                if (type === 'down') btnElement.classList.add('active-down');

                try {
                    await fetch('/api/v0/feedback', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            question: data.question,
                            sql: data.sql,
                            explanation: data.explanation,
                            feedback_type: type
                        })
                    });
                    
                    if (type === 'down') {
                        alert('感谢反馈！已自动通知管理员进行优化。');
                    }
                } catch (e) {
                    console.error('Feedback error', e);
                    buttons.forEach(b => b.disabled = false); 
                }
            }

            async function deleteData(id) {
                if (!confirm('Are you sure you want to delete this training data?')) return;
                
                try {
                    const res = await fetch(`/api/v0/training_data/${id}`, {
                        method: 'DELETE'
                    });
                    
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Failed to delete data');
                    }
                    
                    loadKnowledgeBase();
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
