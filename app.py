import os
import re
import requests
from datetime import datetime
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Response, Body, HTTPException
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
    try:
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

    except Exception as e:
        # 捕获所有异常并返回友好的错误信息
        error_msg = f"生成 SQL 失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v0/run_sql")
def run_sql(request: SqlRequest):
    try:
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

    except pymysql.Error as e:
        # 数据库连接或执行错误
        error_msg = f"数据库错误: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        # 其他未预期的错误
        error_msg = f"执行失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

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
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {
                --vanna-teal: #15a8a8;
                --vanna-navy: #023d60;
                --vanna-bg: #f8fafc;
                --vanna-surface: #ffffff;
                --vanna-border: #e2e8f0;
                --vanna-text: #0f172a;
                --vanna-text-dim: #64748b;
            }

            body {
                background-color: var(--vanna-bg);
                height: 100vh;
                display: flex;
                flex-direction: column;
                font-family: 'Space Grotesk', sans-serif;
                color: var(--vanna-text);
            }

            .navbar {
                background: rgba(255, 255, 255, 0.8) !important;
                backdrop-filter: blur(10px);
                border-bottom: 1px solid var(--vanna-border);
            }

            .main-container { flex: 1; overflow: hidden; display: flex; flex-direction: column; max-width: 1200px; margin: 0 auto; width: 100%; padding: 20px; }

            .card-container {
                background: var(--vanna-surface);
                border: 1px solid var(--vanna-border);
                border-radius: 24px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                height: 100%;
            }

            .nav-tabs { padding: 0 20px; border-bottom: 1px solid var(--vanna-border); background: var(--vanna-surface); }
            .nav-link { color: var(--vanna-text-dim); border: none; padding: 15px 20px; font-weight: 500; }
            .nav-link.active { color: var(--vanna-teal); border-bottom: 2px solid var(--vanna-teal); background: transparent; }
            .nav-link:hover { color: var(--vanna-teal); }

            .tab-content { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
            .tab-pane { height: 100%; display: none; flex-direction: column; }
            .tab-pane.active { display: flex; }

            /* Chat Styles */
            .chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 24px; }

            .message-wrapper { display: flex; gap: 12px; max-width: 85%; animation: fadeIn 0.3s ease; }
            .message-wrapper.user { align-self: flex-end; flex-direction: row-reverse; }
            .message-wrapper.ai { align-self: flex-start; }

            .avatar {
                width: 36px; height: 36px;
                border-radius: 10px;
                display: flex; align-items: center; justify-content: center;
                flex-shrink: 0;
                font-size: 18px;
            }
            .avatar.ai { background: rgba(21, 168, 168, 0.1); color: var(--vanna-teal); }
            .avatar.user { background: var(--vanna-navy); color: white; }

            .message-bubble {
                padding: 16px 20px;
                border-radius: 18px;
                position: relative;
                line-height: 1.6;
                font-size: 15px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }

            .message-wrapper.user .message-bubble {
                background: linear-gradient(135deg, #15a8a8 0%, #0e8a8a 100%);
                color: white;
                border-top-right-radius: 4px;
            }

            .message-wrapper.ai .message-bubble {
                background: var(--vanna-surface);
                border: 1px solid var(--vanna-border);
                color: var(--vanna-text);
                border-top-left-radius: 4px;
            }

            /* Data Card Style for Charts/SQL */
            .data-card {
                margin-top: 12px;
                border: 1px solid var(--vanna-border);
                border-radius: 12px;
                overflow: visible; /* 改为visible防止图表被裁剪 */
                background: #fff;
            }

            .data-card-header {
                padding: 12px 16px;
                background: #f8fafc;
                border-bottom: 1px solid var(--vanna-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .data-card-title { font-size: 13px; font-weight: 600; color: var(--vanna-text-dim); text-transform: uppercase; letter-spacing: 0.5px; }

            .sql-block {
                background-color: #1e293b;
                color: #e2e8f0;
                padding: 16px;
                font-family: 'Space Mono', monospace;
                font-size: 13px;
                margin: 0;
                overflow-x: auto;
            }

            .explanation-text { margin-bottom: 8px; }

            /* Feedback Buttons */
            .feedback-actions { margin-top: 8px; display: flex; gap: 8px; opacity: 0.6; transition: opacity 0.2s; }
            .message-wrapper:hover .feedback-actions { opacity: 1; }
            .feedback-btn { border: none; background: none; color: var(--vanna-text-dim); cursor: pointer; padding: 4px 8px; font-size: 13px; border-radius: 6px; }
            .feedback-btn:hover { background: #f1f5f9; color: var(--vanna-teal); }
            .feedback-btn.active-up { color: #10b981; background: rgba(16, 185, 129, 0.1); }
            .feedback-btn.active-down { color: #ef4444; background: rgba(239, 68, 68, 0.1); }

            /* Input Area */
            .input-area { padding: 24px; background: var(--vanna-surface); border-top: 1px solid var(--vanna-border); }
            .input-group {
                background: var(--vanna-bg);
                border: 1px solid var(--vanna-border);
                border-radius: 16px;
                padding: 6px;
                transition: all 0.2s;
            }
            .input-group:focus-within {
                border-color: var(--vanna-teal);
                box-shadow: 0 0 0 3px rgba(21, 168, 168, 0.1);
                background: #fff;
            }
            .form-control { border: none; background: transparent; padding: 12px 16px; font-size: 15px; }
            .form-control:focus { background: transparent; box-shadow: none; }
            .btn-send {
                width: 40px; height: 40px;
                border-radius: 12px;
                background: var(--vanna-teal);
                border: none;
                color: white;
                display: flex; align-items: center; justify-content: center;
                transition: transform 0.2s;
            }
            .btn-send:hover { background: #0e8a8a; transform: scale(1.05); }

            /* KB Styles */
            .kb-container { flex: 1; overflow-y: auto; padding: 24px; }
            .type-badge { font-size: 11px; padding: 4px 8px; border-radius: 6px; font-weight: 600; text-transform: uppercase; }
            .type-ddl { background: rgba(2, 61, 96, 0.1); color: var(--vanna-navy); }
            .type-doc { background: rgba(21, 168, 168, 0.1); color: var(--vanna-teal); }
            .type-sql { background: rgba(245, 158, 11, 0.1); color: #d97706; }

            .content-cell {
                font-family: 'Space Mono', monospace;
                font-size: 12px;
                background: var(--vanna-bg);
                padding: 8px 12px;
                border-radius: 8px;
                border: 1px solid var(--vanna-border);
                max-height: 100px;
                overflow-y: auto;
            }

            /* Chart container specific styles */
            .data-card [id^="chart-container-"] {
                overflow: visible !important;
            }
            
            /* Plotly specific overrides */
            .js-plotly-plot .plotly .modebar {
                top: 10px !important;
                right: 10px !important;
            }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .typing-indicator span { width: 6px; height: 6px; background-color: var(--vanna-text-dim); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; display: inline-block; margin: 0 2px; }
            .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light fixed-top">
            <div class="container-fluid" style="max-width: 1200px;">
                <a class="navbar-brand fw-bold" href="#" style="color: var(--vanna-navy);">
                    <i class="bi bi-stars me-2" style="color: var(--vanna-teal);"></i>Vanna ChatBI
                </a>
                <span class="badge bg-light text-dark border rounded-pill px-3">MVP Environment</span>
            </div>
        </nav>

        <div class="main-container" style="margin-top: 60px;">
            <div class="card-container">
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
                            <div class="message-wrapper ai">
                                <div class="avatar ai"><i class="bi bi-robot"></i></div>
                                <div class="message-bubble">
                                    Hello! I'm connected to your database. Ask me anything about your data!
                                </div>
                            </div>
                        </div>
                        <div class="input-area">
                            <div class="input-group">
                                <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about your data..." onkeypress="if(event.key==='Enter') ask()">
                                <button class="btn btn-send" onclick="ask()"><i class="bi bi-send-fill"></i></button>
                            </div>
                        </div>
                    </div>

                    <!-- KB Tab -->
                    <div class="tab-pane fade" id="kb" role="tabpanel">
                        <div class="kb-container">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <h5 class="mb-0 fw-bold">Training Data</h5>
                                <div>
                                    <button class="btn btn-sm btn-success me-2 rounded-pill px-3" onclick="showAddModal()"><i class="bi bi-plus-lg me-1"></i>Add Data</button>
                                    <button class="btn btn-sm btn-outline-secondary rounded-pill px-3" onclick="loadKnowledgeBase()"><i class="bi bi-arrow-clockwise me-1"></i>Refresh</button>
                                </div>
                            </div>
                            <div class="table-responsive">
                                <table class="table table-hover align-middle" id="kbTable">
                                    <thead class="table-light">
                                        <tr>
                                            <th style="width: 100px">Type</th>
                                            <th>Content / SQL</th>
                                            <th style="width: 25%">Related Question</th>
                                            <th style="width: 100px" class="text-center">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td colspan="4" class="text-center p-5 text-muted">Loading...</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Add Data Modal -->
        <div class="modal fade" id="addModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content border-0 shadow-lg" style="border-radius: 20px;">
                    <div class="modal-header border-bottom-0 pb-0">
                        <h5 class="modal-title fw-bold">Add Training Data</h5>
                        <button type="button" class="btn-close" onclick="hideAddModal()"></button>
                    </div>
                    <div class="modal-body pt-4">
                        <div class="mb-3">
                            <label class="form-label small fw-bold text-muted">DATA TYPE</label>
                            <select class="form-select" id="addType" onchange="toggleAddFields()">
                                <option value="documentation">Documentation</option>
                                <option value="sql">SQL (QA Pair)</option>
                                <option value="ddl">DDL (Schema)</option>
                            </select>
                        </div>
                        <div class="mb-3" id="questionGroup" style="display:none;">
                            <label class="form-label small fw-bold text-muted">QUESTION</label>
                            <input type="text" class="form-control bg-light" id="addQuestion" placeholder="e.g. How many users?">
                        </div>
                        <div class="mb-3">
                            <label class="form-label small fw-bold text-muted" id="contentLabel">CONTENT</label>
                            <textarea class="form-control bg-light" id="addContent" rows="6" style="font-family: monospace; font-size: 13px;"></textarea>
                        </div>
                    </div>
                    <div class="modal-footer border-top-0">
                        <button type="button" class="btn btn-light rounded-pill px-4" onclick="hideAddModal()">Cancel</button>
                        <button type="button" class="btn btn-primary rounded-pill px-4" style="background: var(--vanna-teal); border: none;" onclick="submitData()">Save Data</button>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let APP_CONFIG = { show_sql: false };
            let CHAT_HISTORY = {};

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

                    if (!sqlRes.ok) {
                        const errorData = await sqlRes.json();
                        throw new Error(errorData.detail || 'SQL Generation Failed');
                    }

                    const sqlData = await sqlRes.json();
                    const loadingDiv = document.getElementById(loadingId);

                    if (sqlData.is_sql) {
                        const explanation = sqlData.explanation || "Here are the results:";
                        const displayStyle = APP_CONFIG.show_sql ? 'block' : 'none';
                        const msgId = 'msg-' + Date.now();

                        CHAT_HISTORY[msgId] = {
                            question: question,
                            sql: sqlData.sql,
                            explanation: explanation
                        };

                        // Initial SQL State
                        loadingDiv.innerHTML = `
                            <div class="avatar ai"><i class="bi bi-robot"></i></div>
                            <div class="message-bubble" style="width: 100%">
                                <div class="explanation-text">${explanation}</div>
                                <div class="data-card" style="display: ${displayStyle}">
                                    <div class="data-card-header">
                                        <span class="data-card-title">Generated SQL</span>
                                    </div>
                                    <pre class="sql-block">${sqlData.sql}</pre>
                                </div>
                                <div class="d-flex align-items-center mt-3 text-muted small">
                                    <div class="spinner-border spinner-border-sm me-2 text-primary" role="status"></div>
                                    Running query on database...
                                </div>
                            </div>
                        `;

                        const runRes = await fetch('/api/v0/run_sql', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({sql: sqlData.sql, question: question})
                        });

                        if (!runRes.ok) {
                            const errorData = await runRes.json();
                            throw new Error(errorData.detail || 'SQL Execution Failed');
                        }

                        const runData = await runRes.json();
                        const columns = runData.columns || [];
                        const result = runData.result || [];

                        // Generate Table HTML
                        let tableHtml = '<div class="table-responsive" style="max-height: 400px; overflow-y: auto; overflow-x: auto;"><table class="table table-sm table-hover mb-0" style="font-size: 13px; white-space: nowrap;">';
                        if (columns.length > 0) {
                            tableHtml += '<thead class="table-light sticky-top"><tr>' + columns.map(c => `<th style="padding: 10px 12px;">${c}</th>`).join('') + '</tr></thead><tbody>';
                            if (result.length === 0) {
                                tableHtml += `<tr><td colspan="${columns.length}" class="text-center text-muted p-3">No results found</td></tr>`;
                            } else {
                                result.forEach(row => {
                                    tableHtml += '<tr>' + columns.map(c => `<td style="padding: 8px 12px;">${row[c] !== null ? row[c] : ''}</td>`).join('') + '</tr>';
                                });
                            }
                        } else {
                            tableHtml += '<tbody><tr><td class="text-center text-muted p-3">No data columns returned</td></tr>';
                        }
                        tableHtml += '</tbody></table></div>';

                        // Final Content
                        loadingDiv.innerHTML = `
                            <div class="avatar ai"><i class="bi bi-robot"></i></div>
                            <div class="message-bubble" style="width: 100%">
                                <div class="explanation-text">${explanation}</div>

                                <!-- SQL Card -->
                                <div class="data-card mb-3" style="display: ${displayStyle}">
                                    <div class="data-card-header">
                                        <span class="data-card-title">Generated SQL</span>
                                    </div>
                                    <pre class="sql-block">${sqlData.sql}</pre>
                                </div>

                                <!-- Chart Card -->
                                <div id="chart-card-${msgId}" class="data-card mb-3" style="display:none;">
                                    <div class="data-card-header">
                                        <span class="data-card-title">Visualization</span>
                                    </div>
                                    <div id="chart-container-${msgId}" style="width:100%; min-height:500px; padding:20px;"></div>
                                </div>

                                <!-- Data Table Card -->
                                <div class="data-card">
                                    <div class="data-card-header">
                                        <span class="data-card-title">Result Data</span>
                                        <span class="badge bg-light text-dark border">${result.length} rows</span>
                                    </div>
                                    ${tableHtml}
                                </div>

                                <div class="feedback-actions">
                                    <button class="feedback-btn" onclick="submitFeedback('${msgId}', 'up', this)">
                                        <i class="bi bi-hand-thumbs-up"></i> Helpful
                                    </button>
                                    <button class="feedback-btn" onclick="submitFeedback('${msgId}', 'down', this)">
                                        <i class="bi bi-hand-thumbs-down"></i> Not Helpful
                                    </button>
                                </div>
                            </div>
                        `;

                        if (runData.chart) {
                            try {
                                const chartCard = document.getElementById(`chart-card-${msgId}`);
                                const chartContainer = document.getElementById(`chart-container-${msgId}`);
                                chartCard.style.display = 'block';

                                const chartData = JSON.parse(runData.chart);

                                // Apply Vanna Theme to Plotly (Tech/Modern Look)
                                const layout = chartData.layout || {};
                                layout.paper_bgcolor = 'rgba(0,0,0,0)';
                                layout.plot_bgcolor = 'rgba(0,0,0,0)';
                                layout.font = { family: 'Space Grotesk, sans-serif', color: '#64748b', size: 12 };
                                layout.colorway = ['#15a8a8', '#023d60', '#fe5d26', '#bf1363', '#7c3aed'];
                                layout.autosize = true;
                                // 增大边距以防止标签被截断,使用automargin自动调整
                                layout.margin = { l: 80, r: 40, t: 60, b: 120, pad: 10 };

                                // Axis styling for "Tech" feel (Clean & Crisp)
                                if (!layout.xaxis) layout.xaxis = {};
                                layout.xaxis.automargin = true;
                                layout.xaxis.showgrid = true;
                                layout.xaxis.gridcolor = 'rgba(0,0,0,0.05)';
                                layout.xaxis.linecolor = '#e2e8f0';
                                layout.xaxis.tickfont = { family: 'Space Mono, monospace', size: 10 };
                                layout.xaxis.tickangle = -45; // 倾斜标签避免重叠
                                layout.xaxis.tickmode = 'auto'; // 自动选择刻度
                                layout.xaxis.nticks = 15; // 限制刻度数量

                                if (!layout.yaxis) layout.yaxis = {};
                                layout.yaxis.automargin = true;
                                layout.yaxis.showgrid = true;
                                layout.yaxis.gridcolor = 'rgba(0,0,0,0.05)';
                                layout.yaxis.zeroline = false;
                                layout.yaxis.tickfont = { family: 'Space Mono, monospace', size: 10 };

                                // Hover label styling (Dark tooltip for contrast)
                                layout.hoverlabel = {
                                    bgcolor: '#1e293b',
                                    bordercolor: 'transparent',
                                    font: { family: 'Space Grotesk, sans-serif', color: '#ffffff' }
                                };

                                // 动态计算容器高度
                                const dataLength = chartData.data && chartData.data[0] ? 
                                    (chartData.data[0].x ? chartData.data[0].x.length : chartData.data[0].y ? chartData.data[0].y.length : 10) : 10;
                                const minHeight = Math.max(500, dataLength * 25 + 150); // 根据数据点数量动态调整
                                chartContainer.style.height = minHeight + 'px';

                                Plotly.newPlot(chartContainer, chartData.data, layout, {
                                    displayModeBar: 'hover',
                                    responsive: true,
                                    displaylogo: false,
                                    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
                                }).then(() => {
                                    // 绘制完成后调整大小以确保完全显示
                                    window.dispatchEvent(new Event('resize'));
                                });
                            } catch (e) {
                                console.error("Chart render error", e);
                            }
                        }

                    } else {
                        // Conversational Response
                        loadingDiv.innerHTML = `
                            <div class="avatar ai"><i class="bi bi-robot"></i></div>
                            <div class="message-bubble">
                                ${sqlData.text}
                            </div>
                        `;
                    }
                    loadingDiv.id = '';
                } catch (e) {
                    const loadingDiv = document.getElementById(loadingId);
                    if(loadingDiv) {
                        loadingDiv.innerHTML = `
                            <div class="avatar ai"><i class="bi bi-exclamation-triangle-fill text-danger"></i></div>
                            <div class="message-bubble text-danger border-danger bg-light">
                                <strong>Error:</strong> ${e.message}
                            </div>
                        `;
                        loadingDiv.id = '';
                    }
                }
            }

            function appendMessage(role, text) {
                const chatBox = document.getElementById('chatBox');
                const div = document.createElement('div');
                div.className = `message-wrapper ${role}`;

                if (role === 'user') {
                    div.innerHTML = `
                        <div class="avatar user"><i class="bi bi-person-fill"></i></div>
                        <div class="message-bubble">${text}</div>
                    `;
                } else {
                    div.innerHTML = `
                        <div class="avatar ai"><i class="bi bi-robot"></i></div>
                        <div class="message-bubble">${text}</div>
                    `;
                }

                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function appendLoading(id) {
                const chatBox = document.getElementById('chatBox');
                const div = document.createElement('div');
                div.className = 'message-wrapper ai';
                div.id = id;
                div.innerHTML = `
                    <div class="avatar ai"><i class="bi bi-robot"></i></div>
                    <div class="message-bubble">
                        <div class="typing-indicator"><span></span><span></span><span></span></div>
                    </div>
                `;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            let KB_DATA = [];

            async function loadKnowledgeBase() {
                const tbody = document.querySelector('#kbTable tbody');
                tbody.innerHTML = '<tr><td colspan="4" class="text-center p-5"><div class="spinner-border text-primary" role="status"></div><div class="mt-2 text-muted small">Loading training data...</div></td></tr>';

                try {
                    const res = await fetch('/api/v0/get_training_data');
                    const data = await res.json();
                    KB_DATA = data || [];

                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted p-5">No training data found.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = data.map(item => {
                        let badgeClass = 'type-doc';
                        let content = item.content || '';
                        if (item.training_data_type === 'ddl') badgeClass = 'type-ddl';
                        if (item.training_data_type === 'sql') badgeClass = 'type-sql';

                        const displayContent = content.length > 300 ? content.substring(0, 300) + '...' : content;

                        return `
                        <tr>
                            <td><span class="badge ${badgeClass} type-badge">${item.training_data_type}</span></td>
                            <td><div class="content-cell">${content}</div></td>
                            <td><div class="text-truncate" style="max-width: 200px;" title="${item.question || ''}">${item.question || '-'}</div></td>
                            <td class="text-center">
                                <button class="btn btn-sm btn-light border me-1" onclick="editData('${item.id}')" title="Edit"><i class="bi bi-pencil"></i></button>
                                <button class="btn btn-sm btn-light border text-danger" onclick="deleteData('${item.id}')" title="Delete"><i class="bi bi-trash"></i></button>
                            </td>
                        </tr>`;
                    }).join('');
                } catch (e) {
                    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-danger p-4">Error loading data: ${e.message}</td></tr>`;
                }
            }

            // Modal Functions
            const addModal = new bootstrap.Modal(document.getElementById('addModal'));
            let currentEditId = null;

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
                    cLabel.innerText = 'SQL QUERY';
                } else if (type === 'ddl') {
                    qGroup.style.display = 'none';
                    cLabel.innerText = 'DDL STATEMENT';
                } else {
                    qGroup.style.display = 'none';
                    cLabel.innerText = 'DOCUMENTATION TEXT';
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
                    if (currentEditId) {
                        const delRes = await fetch(`/api/v0/training_data/${currentEditId}`, {
                             method: 'DELETE'
                        });
                        if (!delRes.ok) throw new Error('Failed to update');
                    }

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
                    loadKnowledgeBase();
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
                        alert('Thanks for your feedback! We will improve the model.');
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
