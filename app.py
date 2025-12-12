import os
import re
import time
import requests
from datetime import datetime
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Response, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

    def detect_language(self, text: str) -> str:
        """
        检测文本的主要语言(中文或英文)
        返回: 'zh' (中文) 或 'en' (英文)
        """
        # 统计中文字符数量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字母数量
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        # 如果中文字符占比超过30%,判定为中文
        total_chars = len(text.strip())
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3 or chinese_chars > english_chars:
            return 'zh'
        else:
            return 'en'

    def generate_sql_optimized(self, question: str, allow_llm_to_see_data=False, **kwargs):
        """
        Optimized version of generate_sql that returns both SQL and Explanation in a single pass.
        """
        # 0. 检测用户问题的语言
        user_lang = self.detect_language(question)
        print(f"[DEBUG] 检测到用户语言: {'中文' if user_lang == 'zh' else '英文'}")
        
        # 1. Retrieve Context
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        # 2. Construct Prompt (根据语言动态生成)
        # #region agent log
        t_start_prompt = time.time()
        # #endregion
        
        # 获取当前时间信息
        from datetime import datetime
        current_time = datetime.now()
        current_date_str = current_time.strftime('%Y-%m-%d')
        current_datetime_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        current_weekday_en = current_time.strftime('%A')
        
        # 中文星期映射
        weekday_map = {
            'Monday': '星期一',
            'Tuesday': '星期二',
            'Wednesday': '星期三',
            'Thursday': '星期四',
            'Friday': '星期五',
            'Saturday': '星期六',
            'Sunday': '星期日'
        }
        current_weekday_zh = weekday_map.get(current_weekday_en, current_weekday_en)
        
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        if initial_prompt is None:
            if user_lang == 'zh':
                # 中文提示词 - 添加当前时间信息
                time_context = (
                    f"===当前时间信息===\n"
                    f"当前日期: {current_date_str}\n"
                    f"当前时间: {current_datetime_str}\n"
                    f"星期: {current_weekday_zh}\n"
                    f"注意: 在生成涉及时间的SQL查询时，请使用上述当前时间作为参考基准。\n\n"
                )
                initial_prompt = (
                    time_context
                    + f"你是一个 {self.dialect} 数据库专家。"
                    + "请帮助生成 SQL 查询来回答用户的问题。"
                    + "你的回复应该仅基于给定的上下文，并遵循响应指南和格式说明。"
                )
            else:
                # 英文提示词 - 添加当前时间信息
                time_context = (
                    f"===Current Time Information===\n"
                    f"Current Date: {current_date_str}\n"
                    f"Current Time: {current_datetime_str}\n"
                    f"Day of Week: {current_weekday_en}\n"
                    f"Note: When generating SQL queries involving time, use the above current time as the reference point.\n\n"
                )
                initial_prompt = (
                    time_context
                    + f"You are a {self.dialect} expert. "
                    + "Please help to generate a SQL query to answer the question. "
                    + "Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
                )

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        # Custom Guidelines for Single-Pass Explanation (根据语言切换)
        if user_lang == 'zh':
            initial_prompt += (
                "===响应指南 \n"
                "1. 如果提供的上下文足够，请生成有效的 SQL 查询。\n"
                "2. **重要**: 你必须用中文提供简短的解释（1-2句话），说明查询的作用。\n"
                "3. 按照以下格式响应:\n"
                "   解释: [你的中文解释]\n"
                "   ```sql\n   [你的SQL查询]\n   ```\n"
                "4. 如果提供的上下文几乎足够，但需要了解特定列中的特定字符串，请生成一个中间SQL查询来查找该列中的不同字符串。在查询前添加注释 intermediate_sql。\n"
                "5. 如果提供的上下文不足，请用中文解释为什么无法生成。\n"
                f"6. 确保输出的 SQL 符合 {self.dialect} 标准且可执行。\n"
                "7. **MySQL关键提示**: 不要使用带格式字符串的 `UNIX_TIMESTAMP`（例如 `UNIX_TIMESTAMP(col, '%Y-%m-%d')` 是无效的）。`UNIX_TIMESTAMP()` 只接受 0 或 1 个参数。要格式化时间戳，请使用 `DATE_FORMAT(FROM_UNIXTIME(timestamp_col/1000), '%Y-%m-%d')`（如果是毫秒）或 `DATE_FORMAT(FROM_UNIXTIME(timestamp_col), '%Y-%m-%d')`（如果是秒）。\n"
                "8. **语言要求**: 所有解释和说明文字必须使用中文。\n"
                "9. **多数据库架构要求**: 本系统使用多数据库架构，SQL中的所有表名必须使用完全限定格式 `数据库名.表名`（例如: `SELECT * FROM database_name.table_name`）。绝不能省略数据库名，否则会导致执行错误。\n"
            )
        else:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. If the provided context is sufficient, please generate a valid SQL query. \n"
                "2. **IMPORTANT**: You must provide a brief explanation (1-2 sentences) in English of what the query does. \n"
                "3. Format your response as follows:\n"
                "   Explanation: [Your explanation here in English]\n"
                "   ```sql\n   [Your SQL here]\n   ```\n"
                "4. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
                "5. If the provided context is insufficient, please explain in English why it can't be generated. \n"
                f"6. Ensure that the output SQL is {self.dialect}-compliant and executable. \n"
                "7. **CRITICAL for MySQL**: DO NOT use `UNIX_TIMESTAMP` with a format string (e.g., `UNIX_TIMESTAMP(col, '%Y-%m-%d')` is INVALID). `UNIX_TIMESTAMP()` only accepts 0 or 1 argument. To format a timestamp, use `DATE_FORMAT(FROM_UNIXTIME(timestamp_col/1000), '%Y-%m-%d')` (if ms) or `DATE_FORMAT(FROM_UNIXTIME(timestamp_col), '%Y-%m-%d')` (if seconds). \n"
                "8. **Language requirement**: All explanations must be in English. \n"
                "9. **Multi-Database Architecture Requirement**: This system uses a multi-database architecture. ALL table names in SQL queries MUST use the fully qualified format `database_name.table_name` (e.g., `SELECT * FROM database_name.table_name`). Never omit the database name or the query will fail. \n"
            )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        # 3. Submit Prompt
        llm_response = self.submit_prompt(message_log, **kwargs)

        # 4. Handle Intermediate SQL (Simplified for optimization)
        if "intermediate_sql" in llm_response and allow_llm_to_see_data:
            # If intermediate SQL is needed, we might have to do the standard flow or just return.
            # For now, let's use the standard extract_sql to get the query, run it, and re-prompt.
            # This is the slow path, but necessary for correctness if hit.
            intermediate_sql = self.extract_sql(llm_response)
            try:
                df = self.run_sql(intermediate_sql)
                # Re-prompt with data (根据语言切换提示文本)
                message_log.append(self.assistant_message(llm_response))
                if user_lang == 'zh':
                    message_log.append(self.user_message(
                        f"以下是中间SQL查询 {intermediate_sql} 的结果（pandas DataFrame格式）: \n"
                        + df.to_markdown()
                    ))
                else:
                    message_log.append(self.user_message(
                        f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n"
                        + df.to_markdown()
                    ))
                llm_response = self.submit_prompt(message_log, **kwargs)
            except Exception as e:
                if user_lang == 'zh':
                    return f"执行中间SQL时出错: {e}"
                else:
                    return f"Error running intermediate SQL: {e}"

        return llm_response

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
    'embedding_function': zhipu_embedding,
    'dialect': 'MySQL' # 显式指定 MySQL 方言，防止 LLM 生成不兼容的 SQL 函数 (如 UNIX_TIMESTAMP 多参数)
}

# Initialize Vanna
vn = MyVanna(config=config)

# Setup FastAPI
app = FastAPI(title="Vanna ChatBI MVP")

# Mount static files (for serving images)
app.mount("/img", StaticFiles(directory="img"), name="img")

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
        # 0. 检测用户问题的语言
        user_lang = vn.detect_language(request.question)
        
        # 1. Generate SQL using Optimized Vanna Method (Single Pass for SQL + Explanation)
        raw_result = vn.generate_sql_optimized(question=request.question, allow_llm_to_see_data=True)
        
        # 【调试日志】记录LLM原始输出,方便排查问题
        print(f"\n{'='*60}")
        print(f"[DEBUG] 用户问题: {request.question}")
        print(f"[DEBUG] 检测语言: {'中文' if user_lang == 'zh' else '英文'}")
        print(f"[DEBUG] LLM原始回复:\n{raw_result}")
        print(f"{'='*60}\n")

        # 2. Parse Result to extract SQL and Explanation
        # Expected format: "Explanation: ... ```sql ... ```"
        
        # 尝试多种SQL提取模式(增强鲁棒性)
        sql_match = None
        explanation = ""
        
        # 模式1: 标准的 ```sql ... ```
        sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        
        # 模式2: 通用的 ``` ... ```
        if not sql_match:
            sql_match = re.search(r"```\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        
        # 模式3: 检测SELECT/INSERT/UPDATE/DELETE等SQL关键字(无代码块包裹的情况)
        if not sql_match:
            sql_keywords_pattern = r'(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s+.*?(?:;|$)'
            potential_sql = re.search(sql_keywords_pattern, raw_result, re.DOTALL | re.IGNORECASE)
            if potential_sql:
                print(f"[DEBUG] {'检测到无代码块包裹的SQL，尝试提取...' if user_lang == 'zh' else 'Detected SQL without code blocks, attempting extraction...'}")
                
                # 提取从SQL关键字开始到分号或字符串末尾的内容
                sql_start = potential_sql.start()
                
                # 尝试找到SQL结束位置(分号或两个连续换行)
                sql_end_match = re.search(r';|\n\n', raw_result[sql_start:])
                if sql_end_match:
                    sql_end = sql_start + sql_end_match.end()
                else:
                    sql_end = len(raw_result)
                
                extracted_sql = raw_result[sql_start:sql_end].strip()
                
                # 只清理开头的注释行（以 -- 或 /* 开头的行），保留SQL语句本身
                lines = extracted_sql.split('\n')
                sql_start_index = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # 跳过空行和注释行
                    if stripped and not stripped.startswith('--') and not stripped.startswith('/*'):
                        sql_start_index = i
                        break
                extracted_sql = '\n'.join(lines[sql_start_index:]).strip()
                
                # 创建一个模拟的match对象
                class MockMatch:
                    def __init__(self, text):
                        self._text = text
                    def group(self, n):
                        return self._text
                
                sql_match = MockMatch(extracted_sql)
                
                # 提取SQL之前的内容作为解释
                explanation = raw_result[:sql_start].strip()

        if sql_match:
            clean_sql = sql_match.group(1).strip()
            
            # Remove "intermediate_sql" marker if present
            clean_sql = re.sub(r'^intermediate_sql\s*', '', clean_sql, flags=re.IGNORECASE)
            # 清理SQL中的注释行(-- 开头),但保留内联注释
            clean_sql_lines = []
            for line in clean_sql.split('\n'):
                stripped = line.strip()
                # 保留非注释行,或者是有代码的注释行(内联注释)
                if not stripped.startswith('--') or ' -- ' in line:
                    clean_sql_lines.append(line)
            clean_sql = '\n'.join(clean_sql_lines).strip()
            
            # Extract explanation (everything before the SQL block usually)
            # Or look for "Explanation:" or "解释:" prefix
            if not explanation:  # 如果之前没有提取到explanation
                # 尝试匹配中文"解释:"
                explanation_match = re.search(r"解释[:：]\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE)
                if not explanation_match:
                    # 尝试匹配英文"Explanation:"
                    explanation_match = re.search(r"Explanation:\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE)
                
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    # Fallback: Use text before code block
                    parts = raw_result.split("```")
                    if len(parts) > 0:
                        explanation = parts[0].replace("Explanation:", "").replace("解释:", "").replace("解释：", "").strip()
            
            if not explanation:
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == 'zh' else "Here is the SQL query for your request."
            
            # 确保explanation中不包含SQL代码
            if "SELECT" in explanation.upper() or "FROM" in explanation.upper():
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == 'zh' else "Here is the SQL query for your request."

            print(f"[DEBUG] {'成功提取SQL' if user_lang == 'zh' else 'Successfully extracted SQL'}: {clean_sql[:100]}...")
            print(f"[DEBUG] {'解释' if user_lang == 'zh' else 'Explanation'}: {explanation[:100]}...")
            
            return {
                "sql": clean_sql,
                "is_sql": True,
                "explanation": explanation
            }
        else:
            # No SQL found, treat as conversational response
            print(f"[DEBUG] {'未检测到SQL，作为对话回复处理' if user_lang == 'zh' else 'No SQL detected, treating as conversational response'}")
            return {
                "text": raw_result,
                "is_sql": False,
                "explanation": raw_result
            }

    except Exception as e:
        # 捕获所有异常并返回友好的错误信息
        error_msg = f"生成 SQL 失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v0/run_sql")
def run_sql(request: SqlRequest):
    try:
        print(f"Executing SQL: {request.sql}")  # Add logging to debug SQL errors
        df = vn.run_sql(sql=request.sql)

        # 清理不符合 JSON 规范的浮点数值 (inf, -inf, nan)
        # 将它们替换为 None，避免 JSON 序列化错误
        import numpy as np
        df = df.replace([np.inf, -np.inf], None)  # 无穷大替换为 None
        df = df.replace({np.nan: None})  # NaN 替换为 None

        # --- 新增可视化逻辑 ---
        chart_json = None
        # 性能优化：前端已实现智能图表生成 (ECharts)，后端不再生成 Plotly 代码，以节省 LLM 调用时间和 Token
        # if request.question:
        #     try:
        #         # 1. 让 Vanna 生成绘图代码
        #         plotly_code = vn.generate_plotly_code(question=request.question, sql=request.sql, df=df)
        #
        #         # 2. 执行代码获取 Figure 对象
        #         fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
        #
        #         # 3. 转为 JSON 字符串传给前端
        #         if fig:
        #             chart_json = fig.to_json()
        #     except Exception as e:
        #         print(f"可视化生成失败: {e}")
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
        <!-- 替换为 ECharts - 更现代化的图表库 -->
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <!-- 备用: 保留Plotly以防兼容性问题 -->
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
                overflow: hidden;
            }
            .avatar.ai { 
                background: rgba(21, 168, 168, 0.1); 
                color: var(--vanna-teal);
                padding: 0;
            }
            .avatar.ai img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
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
            
            .data-card-header .btn-group {
                opacity: 0.7;
                transition: opacity 0.2s;
            }
            
            .data-card-header:hover .btn-group {
                opacity: 1;
            }
            
            .data-card-header .btn {
                border-color: var(--vanna-border);
                color: var(--vanna-text-dim);
            }
            
            .data-card-header .btn:hover {
                border-color: var(--vanna-teal);
                color: var(--vanna-teal);
                background: rgba(21, 168, 168, 0.05);
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
                                <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                                    <button class="btn btn-sm btn-danger me-2 rounded-pill px-3" onclick="batchDelete()" id="batchDeleteBtn" style="display:none;"><i class="bi bi-trash me-1"></i>Batch Delete</button>
                                    <button class="btn btn-sm btn-outline-secondary rounded-pill px-3" onclick="loadKnowledgeBase()"><i class="bi bi-arrow-clockwise me-1"></i>Refresh</button>
                                </div>
                            </div>
                            <div class="table-responsive">
                                <table class="table table-hover align-middle" id="kbTable">
                                    <thead class="table-light">
                                        <tr>
                                            <th style="width: 40px"><input class="form-check-input" type="checkbox" id="selectAllKb" onclick="toggleSelectAll()"></th>
                                            <th style="width: 100px">Type</th>
                                            <th>Content / SQL</th>
                                            <th style="width: 25%">Related Question</th>
                                            <th style="width: 100px" class="text-center">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td colspan="5" class="text-center p-5 text-muted">Loading...</td></tr>
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
            let CHART_INSTANCES = {}; // 存储ECharts实例

            window.onload = async () => {
                try {
                    const res = await fetch('/api/v0/config');
                    APP_CONFIG = await res.json();
                } catch (e) { console.error("Config error", e); }
            };

            // ECharts 渲染函数 - 科技感主题
            function renderECharts(container, chartJsonStr, resultData, columns, msgId) {
                console.log('开始渲染ECharts', {resultData, columns});
                
                // 如果没有数据,直接使用Plotly
                if (!resultData || resultData.length === 0 || !columns || columns.length === 0) {
                    console.log('数据为空,使用Plotly渲染');
                    throw new Error('No data for ECharts');
                }
                
                // 智能检测图表类型和数据列
                const echartsOption = smartGenerateChart(resultData, columns);
                
                // 动态计算容器高度
                // 优化：使用固定高度，避免数据量大时图表过高，依靠 dataZoom 进行缩放
                container.style.height = '400px';

                // 初始化 ECharts
                const chart = echarts.init(container, null, {
                    renderer: 'canvas',
                    useDirtyRect: false
                });
                
                chart.setOption(echartsOption);
                console.log('ECharts渲染成功', echartsOption);
                
                // 存储实例用于导出
                CHART_INSTANCES[msgId] = chart;
                
                // 响应式
                window.addEventListener('resize', () => {
                    chart.resize();
                });
                
                return chart;
            }
            
            // 智能图表类型检测
            // 检测是否为长格式时间序列数据 (例如: date, category, value)
            function isLongFormatTimeSeries(resultData, columns) {
                if (columns.length !== 3) return null;
                
                const [col1, col2, col3] = columns;
                const col1Lower = col1.toLowerCase();
                const col2Lower = col2.toLowerCase();
                
                // 第一列是日期/时间
                const isDateColumn = col1Lower.match(/date|time|day|month|year|dt|stat|hour/);
                // 第二列是分类/类型
                const isCategoryColumn = col2Lower.match(/type|category|name|kind|group|measure/);
                
                if (isDateColumn && isCategoryColumn) {
                    // 检查第一列的值是否看起来像日期
                    const firstVal = resultData[0]?.[col1];
                    if (firstVal) {
                        const valStr = firstVal.toString();
                        // 支持多种日期格式: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD
                        if (valStr.match(/^\d{8}$/) || 
                            valStr.match(/^\d{4}[-/]\d{1,2}[-/]\d{1,2}/) ||
                            valStr.match(/^\d{4}[-/]\d{1,2}/)) {
                            return {dateCol: col1, categoryCol: col2, valueCol: col3};
                        }
                    }
                }
                
                return null;
            }
            
            // 转换长格式为宽格式 (date, category, value -> date, cat1, cat2, ...)
            function transformLongToWide(resultData, dateCol, categoryCol, valueCol) {
                // 1. 获取所有唯一的分类
                const categories = [...new Set(resultData.map(row => row[categoryCol]))];
                
                // 2. 按日期分组
                const dateGroups = {};
                resultData.forEach(row => {
                    const date = row[dateCol];
                    if (!dateGroups[date]) {
                        dateGroups[date] = {};
                    }
                    dateGroups[date][row[categoryCol]] = row[valueCol];
                });
                
                // 3. 转换为宽格式
                const wideData = Object.keys(dateGroups).map(date => {
                    const row = {[dateCol]: date};
                    categories.forEach(cat => {
                        row[cat] = dateGroups[date][cat] || 0;
                    });
                    return row;
                });
                
                // 4. 按日期排序
                wideData.sort((a, b) => {
                    const dateA = a[dateCol].toString();
                    const dateB = b[dateCol].toString();
                    return dateA.localeCompare(dateB);
                });
                
                return {
                    data: wideData,
                    columns: [dateCol, ...categories]
                };
            }
            
            // 格式化日期 (20251110 -> 2025-11-10)
            function formatDate(dateValue) {
                const dateStr = dateValue.toString();
                
                // YYYYMMDD -> YYYY-MM-DD
                if (dateStr.match(/^\d{8}$/)) {
                    return dateStr.substring(0, 4) + '-' + 
                           dateStr.substring(4, 6) + '-' + 
                           dateStr.substring(6, 8);
                }
                
                // YYYYMM -> YYYY-MM
                if (dateStr.match(/^\d{6}$/)) {
                    return dateStr.substring(0, 4) + '-' + dateStr.substring(4, 6);
                }
                
                return dateStr;
            }
            
            // 检测列的数据类型（数值列 vs 文本列）
            function detectColumnTypes(resultData, columns) {
                const columnTypes = {};
                
                columns.forEach(col => {
                    // 检查该列的所有值
                    let numericCount = 0;
                    let totalCount = 0;
                    
                    resultData.forEach(row => {
                        const val = row[col];
                        if (val !== null && val !== undefined && val !== '') {
                            totalCount++;
                            // 检查是否为数字
                            if (!isNaN(Number(val)) && val !== true && val !== false) {
                                numericCount++;
                            }
                        }
                    });
                    
                    // 如果超过80%的值是数字，则认为是数值列
                    columnTypes[col] = (totalCount > 0 && numericCount / totalCount >= 0.8) ? 'numeric' : 'text';
                });
                
                return columnTypes;
            }

            function detectChartType(resultData, columns, columnTypes = null) {
                const xColumn = columns[0];
                
                // 如果没有提供列类型，自动检测
                if (!columnTypes) {
                    columnTypes = detectColumnTypes(resultData, columns);
                }
                
                // 只考虑数值列作为Y轴
                const yColumns = columns.slice(1).filter(col => columnTypes[col] === 'numeric');
                
                // 规则0: 检查是否为长格式时间序列
                const longFormat = isLongFormatTimeSeries(resultData, columns);
                if (longFormat) {
                    return 'line-long-format';
                }
                
                // 规则1: 检查列名中的关键词
                const columnNames = columns.join(' ').toLowerCase();
                
                // 趋势类关键词 (折线图)
                if (columnNames.match(/trend|趋势|time|时间|date|日期|month|月|year|年|day|天|hour|小时/)) {
                    // 检查X轴是否为日期或时间序列
                    const firstVal = resultData[0]?.[xColumn];
                    if (firstVal && (firstVal.toString().match(/\d{4}[-/]\d{1,2}[-/]\d{1,2}/) || 
                                     firstVal.toString().match(/\d{4}[-/]\d{1,2}/) ||
                                     firstVal.toString().match(/\d{8}/) ||
                                     firstVal.toString().match(/\d{1,2}:\d{2}/))) {
                        return 'line';
                    }
                }
                
                // 占比类关键词 (饼图)
                if (columnNames.match(/percent|百分比|占比|比例|rate|率|proportion|份额|distribution|分布/) &&
                    yColumns.length === 1 && resultData.length <= 10) {
                    return 'pie';
                }
                
                // 规则2: 根据数据特征判断
                if (yColumns.length === 1) {
                    const yData = resultData.map(row => Number(row[yColumns[0]]) || 0);
                    const total = yData.reduce((a, b) => a + b, 0);
                    
                    // 如果所有值加起来接近100,可能是百分比 → 饼图
                    if (total >= 95 && total <= 105 && resultData.length <= 8) {
                        return 'pie';
                    }
                    
                    // 如果数据点很多(>15),且有明显趋势 → 折线图
                    if (resultData.length > 15) {
                        return 'line';
                    }
                }
                
                // 规则3: 多系列数据 → 折线图(方便对比)
                if (yColumns.length > 1 && resultData.length > 5) {
                    return 'line';
                }
                
                // 默认: 柱状图(最常用)
                return 'bar';
            }
            
            // 智能生成图表配置
            function smartGenerateChart(resultData, columns) {
                // 先检测列类型
                const columnTypes = detectColumnTypes(resultData, columns);
                console.log('列类型检测结果:', columnTypes);
                
                // 检测最佳图表类型（传入列类型信息）
                const chartType = detectChartType(resultData, columns, columnTypes);
                console.log('智能选择图表类型:', chartType);
                
                // 处理长格式时间序列数据
                if (chartType === 'line-long-format') {
                    const longFormat = isLongFormatTimeSeries(resultData, columns);
                    if (longFormat) {
                        console.log('检测到长格式时间序列数据，正在转换...', longFormat);
                        const transformed = transformLongToWide(
                            resultData, 
                            longFormat.dateCol, 
                            longFormat.categoryCol, 
                            longFormat.valueCol
                        );
                        
                        console.log('转换后的数据:', transformed);
                        
                        // 格式化日期
                        const formattedData = transformed.data.map(row => {
                            const newRow = {...row};
                            newRow[longFormat.dateCol] = formatDate(row[longFormat.dateCol]);
                            return newRow;
                        });
                        
                        // 使用转换后的宽格式数据生成折线图
                        const xColumn = transformed.columns[0];
                        const yColumns = transformed.columns.slice(1);
                        const xAxisData = formattedData.map(row => row[xColumn]);
                        
                        return generateLineChart(formattedData, transformed.columns, xColumn, yColumns, xAxisData);
                    }
                }
                
                // 假设第一列是X轴(标签/类别)
                const xColumn = columns[0];
                
                // 只选择数值列作为Y轴系列（排除文本列）
                let yColumns = columns.slice(1).filter(col => columnTypes[col] === 'numeric');
                
                console.log('X轴列:', xColumn, '| Y轴数值列:', yColumns);
                
                // 如果没有数值列，使用所有列（回退逻辑）
                if (yColumns.length === 0) {
                    console.warn('未检测到数值列，使用所有列');
                    yColumns = columns.slice(1);
                }
                
                // 提取X轴数据
                const xAxisData = resultData.map(row => {
                    let val = row[xColumn];
                    // 尝试格式化日期
                    if (val && val.toString().match(/^\d{8}$/)) {
                        val = formatDate(val);
                    }
                    // 处理长标签
                    if (val && typeof val === 'string' && val.length > 20) {
                        return val.substring(0, 20) + '...';
                    }
                    return val !== null ? String(val) : '';
                });
                
                // 根据图表类型生成不同配置
                if (chartType === 'pie') {
                    return generatePieChart(resultData, columns, xColumn, yColumns);
                } else if (chartType === 'line') {
                    return generateLineChart(resultData, columns, xColumn, yColumns, xAxisData);
                } else {
                    return generateBarChart(resultData, columns, xColumn, yColumns, xAxisData);
                }
            }
            
            // 生成柱状图配置
            function generateBarChart(resultData, columns, xColumn, yColumns, xAxisData) {
                // 提取Y轴数据系列
                const series = yColumns.map((col, idx) => {
                    const data = resultData.map(row => {
                        const val = row[col];
                        return val !== null && val !== undefined ? Number(val) : 0;
                    });
                    
                    // 多系列时使用不同颜色
                    const colors = [
                        ['#15a8a8', '#0e8a8a'],
                        ['#00d4ff', '#0099cc'],
                        ['#fe5d26', '#cc4a1e'],
                        ['#7c3aed', '#5b21b6'],
                        ['#bf1363', '#8b0a46']
                    ];
                    const colorPair = colors[idx % colors.length];
                    
                    return {
                        name: col,
                        type: 'bar',
                        data: data,
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: colorPair[0] },
                                { offset: 1, color: colorPair[1] }
                            ]),
                            borderRadius: [8, 8, 0, 0]
                        },
                        emphasis: {
                            itemStyle: {
                                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                    { offset: 0, color: '#00d4ff' },
                                    { offset: 1, color: '#15a8a8' }
                                ])
                            }
                        },
                        animationDuration: 1000,
                        animationEasing: 'elasticOut',
                        label: {
                            show: resultData.length < 20,
                            position: 'top',
                            color: '#64748b',
                            fontSize: 11
                        }
                    };
                });
                
                // 计算Y轴数据范围
                let allValues = [];
                series.forEach(s => {
                    allValues = allValues.concat(s.data);
                });
                const minVal = Math.min(...allValues);
                const maxVal = Math.max(...allValues);
                
                // 计算合理的Y轴范围（添加10%的padding）
                const range = maxVal - minVal;
                const padding = range * 0.1;
                let yMin = minVal - padding;
                let yMax = maxVal + padding;
                
                // 如果所有值都是正数，从0开始显示
                if (minVal >= 0) {
                    yMin = 0;
                }
                
                // 如果所有值都是负数，最大值设为0
                if (maxVal <= 0) {
                    yMax = 0;
                }
                
                // 科技感配色方案
                const techColors = ['#15a8a8', '#00d4ff', '#fe5d26', '#7c3aed', '#bf1363', '#fbbf24'];
                
                // 图例配置(多系列时显示)
                const legendConfig = yColumns.length > 1 ? {
                    show: true,
                    top: 10,
                    left: 'center',
                    textStyle: {
                        color: '#64748b',
                        fontSize: 12
                    },
                    itemGap: 20
                } : { show: false };
                
                return {
                    backgroundColor: 'transparent',
                    color: techColors,
                    textStyle: {
                        fontFamily: 'Space Grotesk, sans-serif',
                        color: '#64748b'
                    },
                    title: {
                        text: '',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#0f172a',
                            fontSize: 16,
                            fontWeight: 600
                        }
                    },
                    legend: legendConfig,
                    tooltip: {
                        trigger: 'axis',
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        borderColor: 'transparent',
                        textStyle: {
                            color: '#ffffff',
                            fontFamily: 'Space Grotesk, sans-serif'
                        },
                        axisPointer: {
                            type: 'shadow',
                            shadowStyle: {
                                color: 'rgba(21, 168, 168, 0.1)'
                            }
                        }
                    },
                    grid: {
                        left: '80px',
                        right: '40px',
                        top: yColumns.length > 1 ? '80px' : '60px',
                        bottom: '100px',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: xAxisData,
                        axisLine: {
                            lineStyle: {
                                color: '#e2e8f0'
                            }
                        },
                        axisLabel: {
                            color: '#64748b',
                            fontSize: 11,
                            fontFamily: 'Space Mono, monospace',
                            rotate: resultData.length > 10 ? 45 : 0,
                            interval: 0
                        },
                        splitLine: {
                            show: false
                        }
                    },
                    yAxis: {
                        type: 'value',
                        min: yMin,
                        max: yMax,
                        axisLine: {
                            show: false
                        },
                        axisLabel: {
                            color: '#64748b',
                            fontSize: 11,
                            fontFamily: 'Space Mono, monospace'
                        },
                        splitLine: {
                            lineStyle: {
                                color: 'rgba(0,0,0,0.05)',
                                type: 'dashed'
                            }
                        }
                    },
                    series: series,
                    toolbox: {
                        show: resultData.length > 10,
                        feature: {
                            dataZoom: {
                                show: true,
                                title: {
                                    zoom: '区域缩放',
                                    back: '还原'
                                }
                            },
                            restore: {
                                show: true,
                                title: '还原'
                            }
                        },
                        iconStyle: {
                            borderColor: '#64748b'
                        },
                        emphasis: {
                            iconStyle: {
                                borderColor: '#15a8a8'
                            }
                        },
                        right: 20,
                        top: 10
                    }
                };
            }
            
            // 生成折线图配置
            function generateLineChart(resultData, columns, xColumn, yColumns, xAxisData) {
                const techColors = ['#15a8a8', '#00d4ff', '#fe5d26', '#7c3aed', '#bf1363'];
                
                const series = yColumns.map((col, idx) => {
                    const data = resultData.map(row => {
                        const val = row[col];
                        return val !== null && val !== undefined ? Number(val) : 0;
                    });
                    
                    const color = techColors[idx % techColors.length];
                    
                    return {
                        name: col,
                        type: 'line',
                        data: data,
                        smooth: true,
                        lineStyle: {
                            color: color,
                            width: 3
                        },
                        itemStyle: {
                            color: color,
                            borderWidth: 2,
                            borderColor: '#fff'
                        },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: color + '40' },
                                { offset: 1, color: color + '08' }
                            ])
                        },
                        emphasis: {
                            focus: 'series',
                            itemStyle: {
                                borderWidth: 3,
                                shadowBlur: 10,
                                shadowColor: color
                            }
                        },
                        animationDuration: 1500,
                        animationEasing: 'cubicOut',
                        symbol: 'circle',
                        symbolSize: 8,
                        showSymbol: resultData.length <= 30
                    };
                });
                
                // 计算Y轴数据范围
                let allValues = [];
                series.forEach(s => {
                    allValues = allValues.concat(s.data);
                });
                const minVal = Math.min(...allValues);
                const maxVal = Math.max(...allValues);
                
                // 计算合理的Y轴范围（添加10%的padding）
                const range = maxVal - minVal;
                const padding = range * 0.1;
                let yMin = minVal - padding;
                let yMax = maxVal + padding;
                
                // 如果所有值都是正数，从0开始显示
                if (minVal >= 0) {
                    yMin = 0;
                }
                
                // 如果所有值都是负数，最大值设为0
                if (maxVal <= 0) {
                    yMax = 0;
                }
                
                return {
                    backgroundColor: 'transparent',
                    color: techColors,
                    textStyle: {
                        fontFamily: 'Space Grotesk, sans-serif',
                        color: '#64748b'
                    },
                    title: {
                        text: '',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#0f172a',
                            fontSize: 16,
                            fontWeight: 600
                        }
                    },
                    legend: {
                        show: yColumns.length > 1,
                        top: 10,
                        left: 'center',
                        textStyle: {
                            color: '#64748b',
                            fontSize: 12
                        }
                    },
                    tooltip: {
                        trigger: 'axis',
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        borderColor: 'transparent',
                        textStyle: {
                            color: '#ffffff',
                            fontFamily: 'Space Grotesk, sans-serif'
                        },
                        axisPointer: {
                            type: 'line',
                            lineStyle: {
                                color: 'rgba(21, 168, 168, 0.3)',
                                width: 2,
                                type: 'dashed'
                            }
                        }
                    },
                    grid: {
                        left: '80px',
                        right: '40px',
                        top: yColumns.length > 1 ? '80px' : '60px',
                        bottom: '100px',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: xAxisData,
                        boundaryGap: false,
                        axisLine: {
                            lineStyle: {
                                color: '#e2e8f0'
                            }
                        },
                        axisLabel: {
                            color: '#64748b',
                            fontSize: 11,
                            fontFamily: 'Space Mono, monospace',
                            rotate: resultData.length > 15 ? 45 : 0,
                            interval: 'auto'
                        },
                        splitLine: {
                            show: false
                        }
                    },
                    yAxis: {
                        type: 'value',
                        min: yMin,
                        max: yMax,
                        axisLine: {
                            show: false
                        },
                        axisLabel: {
                            color: '#64748b',
                            fontSize: 11,
                            fontFamily: 'Space Mono, monospace'
                        },
                        splitLine: {
                            lineStyle: {
                                color: 'rgba(0,0,0,0.05)',
                                type: 'dashed'
                            }
                        }
                    },
                    series: series,
                    toolbox: {
                        show: resultData.length > 10,
                        feature: {
                            dataZoom: {
                                show: true,
                                title: {
                                    zoom: '区域缩放',
                                    back: '还原'
                                }
                            },
                            restore: {
                                show: true,
                                title: '还原'
                            }
                        },
                        iconStyle: {
                            borderColor: '#64748b'
                        },
                        emphasis: {
                            iconStyle: {
                                borderColor: '#15a8a8'
                            }
                        },
                        right: 20,
                        top: 10
                    }
                };
            }
            
            // 生成饼图配置
            function generatePieChart(resultData, columns, xColumn, yColumns) {
                const techColors = ['#15a8a8', '#00d4ff', '#fe5d26', '#7c3aed', '#bf1363', '#fbbf24', '#10b981', '#f59e0b'];
                
                // 饼图数据格式
                const pieData = resultData.map((row, idx) => ({
                    name: String(row[xColumn]),
                    value: Number(row[yColumns[0]]) || 0,
                    itemStyle: {
                        color: techColors[idx % techColors.length],
                        borderRadius: 10,
                        borderColor: '#fff',
                        borderWidth: 2
                    }
                }));
                
                return {
                    backgroundColor: 'transparent',
                    color: techColors,
                    textStyle: {
                        fontFamily: 'Space Grotesk, sans-serif',
                        color: '#64748b'
                    },
                    title: {
                        text: '',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            color: '#0f172a',
                            fontSize: 16,
                            fontWeight: 600
                        }
                    },
                    tooltip: {
                        trigger: 'item',
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        borderColor: 'transparent',
                        textStyle: {
                            color: '#ffffff',
                            fontFamily: 'Space Grotesk, sans-serif'
                        },
                        formatter: '{b}: {c} ({d}%)'
                    },
                    legend: {
                        show: true,
                        orient: 'vertical',
                        right: 20,
                        top: 'center',
                        textStyle: {
                            color: '#64748b',
                            fontSize: 12
                        },
                        formatter: function(name) {
                            if (name.length > 15) {
                                return name.substring(0, 15) + '...';
                            }
                            return name;
                        }
                    },
                    series: [{
                        name: yColumns[0],
                        type: 'pie',
                        radius: ['45%', '75%'],
                        center: ['40%', '50%'],
                        data: pieData,
                        label: {
                            show: true,
                            color: '#64748b',
                            fontFamily: 'Space Grotesk, sans-serif',
                            fontSize: 12,
                            formatter: '{d}%'
                        },
                        labelLine: {
                            show: true,
                            length: 15,
                            length2: 10,
                            lineStyle: {
                                color: '#e2e8f0'
                            }
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 15,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(21, 168, 168, 0.5)'
                            },
                            label: {
                                show: true,
                                fontSize: 14,
                                fontWeight: 'bold'
                            }
                        },
                        animationDuration: 2000,
                        animationEasing: 'elasticOut'
                    }]
                };
            }


            // Plotly 回退渲染函数
            function renderPlotly(container, chartJsonStr) {
                const chartData = JSON.parse(chartJsonStr);
                const layout = chartData.layout || {};
                
                layout.paper_bgcolor = 'rgba(0,0,0,0)';
                layout.plot_bgcolor = 'rgba(0,0,0,0)';
                layout.font = { family: 'Space Grotesk, sans-serif', color: '#64748b', size: 12 };
                layout.colorway = ['#15a8a8', '#023d60', '#fe5d26', '#bf1363', '#7c3aed'];
                layout.autosize = true;
                layout.margin = { l: 80, r: 40, t: 60, b: 120, pad: 10 };

                if (!layout.xaxis) layout.xaxis = {};
                layout.xaxis.automargin = true;
                layout.xaxis.showgrid = true;
                layout.xaxis.gridcolor = 'rgba(0,0,0,0.05)';
                layout.xaxis.linecolor = '#e2e8f0';
                layout.xaxis.tickfont = { family: 'Space Mono, monospace', size: 10 };
                layout.xaxis.tickangle = -45;

                if (!layout.yaxis) layout.yaxis = {};
                layout.yaxis.automargin = true;
                layout.yaxis.showgrid = true;
                layout.yaxis.gridcolor = 'rgba(0,0,0,0.05)';
                layout.yaxis.zeroline = false;
                layout.yaxis.tickfont = { family: 'Space Mono, monospace', size: 10 };

                Plotly.newPlot(container, chartData.data, layout, {
                    displayModeBar: 'hover',
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
                });
            }

            // 导出图表
            function downloadChart(msgId, format) {
                const chart = CHART_INSTANCES[msgId];
                if (chart) {
                    const url = chart.getDataURL({
                        type: format || 'png',
                        pixelRatio: 2,
                        backgroundColor: '#fff'
                    });
                    const link = document.createElement('a');
                    link.download = `chart-${msgId}.${format}`;
                    link.href = url;
                    link.click();
                } else {
                    alert('图表实例未找到');
                }
            }

            // 全屏切换
            function toggleFullscreen(elementId) {
                const element = document.getElementById(elementId);
                if (!document.fullscreenElement) {
                    element.requestFullscreen().catch(err => {
                        console.error('全屏失败:', err);
                    });
                } else {
                    document.exitFullscreen();
                }
            }

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
                            <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                            <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                                        <span class="data-card-title"><i class="bi bi-bar-chart-fill me-2"></i>Visualization</span>
                                        <div class="btn-group btn-group-sm" role="group">
                                            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="downloadChart('${msgId}', 'png')" title="导出PNG">
                                                <i class="bi bi-download"></i>
                                            </button>
                                            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="toggleFullscreen('chart-container-${msgId}')" title="全屏">
                                                <i class="bi bi-arrows-fullscreen"></i>
                                            </button>
                                        </div>
                                    </div>
                                    <div id="chart-container-${msgId}" style="width:100%; min-height:400px; padding:20px;"></div>
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

                        // 性能优化：后端不再返回 chart 数据，由前端直接根据 data 生成图表
                        // if (runData.chart) {
                        if (result && result.length > 0 && columns && columns.length > 0) {
                            try {
                                const chartCard = document.getElementById(`chart-card-${msgId}`);
                                const chartContainer = document.getElementById(`chart-container-${msgId}`);
                                chartCard.style.display = 'block';

                                // 尝试使用ECharts渲染
                                try {
                                    renderECharts(chartContainer, null, result, columns, msgId);
                                } catch (echartsError) {
                                    console.warn("ECharts render failed:", echartsError);
                                    chartCard.style.display = 'none'; // 渲染失败则隐藏
                                }
                            } catch (e) {
                                console.error("Chart render error", e);
                            }
                        }

                    } else {
                        // Conversational Response
                        loadingDiv.innerHTML = `
                            <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                            <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-exclamation-triangle-fill text-danger&quot;></i>'"></div>
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
                        <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                    <div class="avatar ai"><img src="/img/Attached_image.png" alt="AI Avatar" onerror="this.style.display='none';this.parentElement.innerHTML='<i class=&quot;bi bi-robot&quot;></i>'"></div>
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
                tbody.innerHTML = '<tr><td colspan="5" class="text-center p-5"><div class="spinner-border text-primary" role="status"></div><div class="mt-2 text-muted small">Loading training data...</div></td></tr>';

                try {
                    const res = await fetch('/api/v0/get_training_data');
                    const data = await res.json();
                    KB_DATA = data || [];

                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted p-5">No training data found.</td></tr>';
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
                            <td><input class="form-check-input kb-checkbox" type="checkbox" value="${item.id}" onchange="updateBatchDeleteBtn()"></td>
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
                    tbody.innerHTML = `<tr><td colspan="5" class="text-center text-danger p-4">Error loading data: ${e.message}</td></tr>`;
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

            function toggleSelectAll() {
                const selectAll = document.getElementById('selectAllKb');
                const checkboxes = document.querySelectorAll('.kb-checkbox');
                checkboxes.forEach(cb => cb.checked = selectAll.checked);
                updateBatchDeleteBtn();
            }

            function updateBatchDeleteBtn() {
                const checkboxes = document.querySelectorAll('.kb-checkbox:checked');
                const btn = document.getElementById('batchDeleteBtn');
                if (checkboxes.length > 0) {
                    btn.style.display = 'inline-block';
                    btn.innerHTML = `<i class="bi bi-trash me-1"></i>Delete (${checkboxes.length})`;
                } else {
                    btn.style.display = 'none';
                }
            }

            async function batchDelete() {
                const checkboxes = document.querySelectorAll('.kb-checkbox:checked');
                if (checkboxes.length === 0) return;

                if (!confirm(`Are you sure you want to delete ${checkboxes.length} items?`)) return;

                const ids = Array.from(checkboxes).map(cb => cb.value);

                try {
                    const res = await fetch('/api/v0/training_data/batch_delete', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ids: ids})
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Failed to batch delete');
                    }
                    
                    const result = await res.json();
                    if (result.errors && result.errors.length > 0) {
                        alert(`Deleted ${result.deleted_count} items. Errors:\n${result.errors.join('\n')}`);
                    }

                    loadKnowledgeBase();
                    document.getElementById('selectAllKb').checked = false;
                    document.getElementById('batchDeleteBtn').style.display = 'none';
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
