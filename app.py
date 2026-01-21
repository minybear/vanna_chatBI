import os
import re
import time
import json
import uuid
import requests
from datetime import datetime
from decimal import Decimal
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Response, Body, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pymysql
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional, List, Dict

# Load environment variables
load_dotenv()

# 对话历史管理（ChromaDB 持久化存储）
class SimpleEmbeddingFunction:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def name(self) -> str:
        return "simple"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in input]

    def _embed_text(self, text: str) -> List[float]:
        import hashlib
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]


class ChromaConversationHistory:
    """使用 ChromaDB 持久化保存会话与消息"""
    def __init__(self, chroma_client, max_history_per_session=10):
        self.max_history = max_history_per_session
        self.embedding_function = SimpleEmbeddingFunction()
        self.messages = chroma_client.get_or_create_collection(
            name="chat_messages",
            embedding_function=self.embedding_function,
        )
        self.sessions = chroma_client.get_or_create_collection(
            name="chat_sessions",
            embedding_function=self.embedding_function,
        )

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _get_session_record(self, session_id: str) -> Optional[Dict]:
        data = self.sessions.get(ids=[session_id])
        if data and data.get("ids"):
            meta = (data.get("metadatas") or [{}])[0]
            title = (data.get("documents") or [None])[0] or meta.get("title")
            return {
                "title": title,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "message_count": meta.get("message_count", 0),
            }
        return None

    def _upsert_session(self, session_id: str, title: str, created_at: str, updated_at: str, message_count: int):
        self.sessions.upsert(
            ids=[session_id],
            documents=[title],
            metadatas=[{
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": message_count,
            }],
        )

    def add_message(self, session_id: str, role: str, content: str, data: Optional[Dict] = None):
        """添加一条消息到会话历史"""
        now = self._now()
        def _sanitize_for_json(value):
            try:
                import pandas as pd
                from decimal import Decimal
                
                # 处理 Decimal 类型
                if isinstance(value, Decimal):
                    return float(value)
                
                # 处理 pandas Timestamp
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            
            # 递归处理字典
            if isinstance(value, dict):
                return {k: _sanitize_for_json(v) for k, v in value.items()}
            
            # 递归处理列表
            if isinstance(value, list):
                return [_sanitize_for_json(v) for v in value]
            
            # 处理其他日期时间类型
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return value
            
            return value

        payload = {
            "role": role,
            "content": content,
            "timestamp": now,
        }
        if data:
            payload.update(_sanitize_for_json(data))

        msg_id = f"{session_id}-{uuid.uuid4().hex}"
        self.messages.add(
            documents=[json.dumps(payload, ensure_ascii=False)],
            ids=[msg_id],
            metadatas=[{
                "session_id": session_id,
                "timestamp": now,
                "role": role,
            }],
        )

        session_record = self._get_session_record(session_id)
        if session_record is None:
            created_at = now
            message_count = 0
            title = f"Session {session_id[:8]}"
        else:
            created_at = session_record.get("created_at") or now
            message_count = session_record.get("message_count", 0)
            title = session_record.get("title") or f"Session {session_id[:8]}"

        message_count += 1
        if role == "user" and message_count == 1:
            title = content[:30] + ("..." if len(content) > 30 else "")

        self._upsert_session(session_id, title, created_at, now, message_count)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """获取指定会话的历史记录"""
        data = self.messages.get(where={"session_id": session_id})
        if not data or not data.get("ids"):
            return []

        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        messages = []
        for msg_id, doc, meta in zip(ids, documents, metadatas):
            try:
                msg = json.loads(doc) if isinstance(doc, str) else {"content": doc}
            except Exception:
                msg = {"content": doc}
            msg.setdefault("role", meta.get("role"))
            msg.setdefault("timestamp", meta.get("timestamp"))
            msg["id"] = msg_id
            messages.append(msg)

        messages.sort(key=lambda x: x.get("timestamp") or "")
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def get_context_history(self, session_id: str) -> List[Dict]:
        return self.get_history(session_id, limit=self.max_history)

    def clear_session(self, session_id: str):
        """清除指定会话的历史"""
        self.messages.delete(where={"session_id": session_id})
        self.sessions.delete(ids=[session_id])

    def get_sessions(self) -> List[Dict]:
        """Get list of active sessions with metadata"""
        data = self.sessions.get()
        if not data or not data.get("ids"):
            return []

        sessions_list = []
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        for session_id, doc, meta in zip(ids, documents, metadatas):
            title = doc or meta.get("title") or f"Session {session_id}"
            sessions_list.append({
                "id": session_id,
                "title": title,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at", meta.get("created_at")),
                "message_count": meta.get("message_count", 0),
            })

        return sorted(sessions_list, key=lambda x: x.get("updated_at") or "", reverse=True)

    def rename_session(self, session_id: str, new_title: str):
        session_record = self._get_session_record(session_id)
        if session_record is None:
            return False
        created_at = session_record.get("created_at") or self._now()
        updated_at = self._now()
        message_count = session_record.get("message_count", 0)
        title = new_title or session_record.get("title") or f"Session {session_id}"
        self._upsert_session(session_id, title, created_at, updated_at, message_count)
        return True

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

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        重写父类方法，支持相似度阈值过滤
        ChromaDB使用L2距离，距离越小越相似
        """
        # 获取原始检索结果（包含距离信息）
        raw_results = self.sql_collection.query(
            query_texts=[question],
            n_results=self.n_results_sql,
        )
        
        # 相似度阈值：L2距离小于此值才保留（可通过config配置）
        distance_threshold = kwargs.get('sql_distance_threshold', 
                                       getattr(self, 'sql_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=True)
    
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """重写父类方法，支持相似度阈值过滤"""
        raw_results = self.ddl_collection.query(
            query_texts=[question],
            n_results=self.n_results_ddl,
        )
        
        distance_threshold = kwargs.get('ddl_distance_threshold', 
                                       getattr(self, 'ddl_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=False)
    
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """重写父类方法，支持相似度阈值过滤"""
        raw_results = self.documentation_collection.query(
            query_texts=[question],
            n_results=self.n_results_documentation,
        )
        
        distance_threshold = kwargs.get('doc_distance_threshold', 
                                       getattr(self, 'doc_distance_threshold', 1.5))
        
        return self._filter_by_distance(raw_results, distance_threshold, parse_json=False)
    
    def _filter_by_distance(self, query_results, threshold: float, parse_json: bool = False) -> list:
        """
        根据L2距离阈值过滤结果
        
        Args:
            query_results: ChromaDB查询结果
            threshold: 距离阈值，小于此值才保留（L2距离越小越相似）
            parse_json: 是否需要解析JSON（SQL类型需要）
        
        Returns:
            过滤后的文档列表
        """
        if query_results is None or 'documents' not in query_results:
            return []
        
        documents = query_results.get('documents', [[]])[0]
        distances = query_results.get('distances', [[]])[0]
        
        # 过滤：只保留距离小于阈值的结果
        filtered_docs = []
        for doc, dist in zip(documents, distances):
            if dist <= threshold:
                if parse_json:
                    try:
                        filtered_docs.append(json.loads(doc))
                    except:
                        filtered_docs.append(doc)
                else:
                    filtered_docs.append(doc)
            else:
                print(f"[DEBUG] 过滤掉相似度低的结果 (距离={dist:.3f} > 阈值={threshold})")
        
        print(f"[DEBUG] 相似度过滤: {len(documents)}条 -> {len(filtered_docs)}条 (阈值={threshold})")
        return filtered_docs

    def detect_intent(self, text: str) -> Dict[str, str]:
        """
        简单意图识别（规则优先）
        返回: intent, preferred_chart, time_grain, agg_hint
        """
        q = (text or "").lower()
        intent = "detail"
        preferred_chart = "table"
        time_grain = ""
        agg_hint = ""

        # 关键词优先级调整：明细优先级最高
        detail_keywords = ["明细", "详情", "列表", "详细信息", "明细数据", "数据明细"]
        trend_keywords = ["趋势", "变化", "走势", "曲线", "按日", "按月", "按周", "按小时", "同比", "环比"]
        distribution_keywords = ["分布", "占比", "比例", "构成", "份额"]
        compare_keywords = ["对比", "比较", "差异"]
        ranking_keywords = ["top", "排行", "排名", "前几"]

        # 【优先级1】明细数据 - 必须优先判断
        if any(k in q for k in detail_keywords):
            intent = "detail"
            preferred_chart = "table"
        # 【优先级2】排行数据
        elif any(k in q for k in ranking_keywords):
            intent = "ranking"
            preferred_chart = "bar"
        # 【优先级3】趋势分析
        elif any(k in q for k in trend_keywords):
            intent = "trend"
            preferred_chart = "line"
            agg_hint = "sum"
        # 【优先级4】分布占比
        elif any(k in q for k in distribution_keywords):
            intent = "distribution"
            preferred_chart = "pie"
        # 【优先级5】对比分析
        elif any(k in q for k in compare_keywords):
            intent = "comparison"
            preferred_chart = "bar"

        if "按日" in q or "每日" in q or "日" in q:
            time_grain = "day"
        elif "按月" in q or "每月" in q or "月" in q:
            time_grain = "month"
        elif "按周" in q or "每周" in q or "周" in q:
            time_grain = "week"
        elif "按小时" in q or "每小时" in q or "小时" in q:
            time_grain = "hour"

        return {
            "intent": intent,
            "preferred_chart": preferred_chart,
            "time_grain": time_grain,
            "agg_hint": agg_hint,
        }

    def _recommend_chart_type(self, df: pd.DataFrame, question: str) -> str:
        """
        基于数据特征和问题内容，智能推荐图表类型
        """
        row_count = len(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        question_lower = question.lower()
        
        # 【优先级1】明细数据场景 - 不画图，返回 None
        # 强匹配：如果问题中包含"明细"或"详情"，一定不画图
        strong_detail_keywords = ['明细', '详情', '列表', '详细信息', 'detail', 'list', 'records']
        if any(kw in question_lower for kw in strong_detail_keywords):
            print(f"[DEBUG] 检测到明细关键词，不生成图表")
            return None  # 明细数据不应该画图
        
        # 【优先级2】单值场景
        if row_count == 1 and len(numeric_cols) == 1:
            return "Indicator (go.Indicator) - 单一指标卡片"
        
        # 【优先级3】排行场景（优先于趋势判断）
        if any(kw in question_lower for kw in ['top', 'rank', '排名', '排行', '前', '最']):
            return "Bar Chart (px.bar) - 横向柱状图（水平）"
        
        # 【优先级4】占比/分布场景
        if any(kw in question_lower for kw in ['占比', '比例', '分布', 'proportion', 'distribution', '构成']):
            if len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() <= 8:
                return "Pie Chart (px.pie) - 饼图"
            else:
                return "Bar Chart (px.bar) - 柱状图"
        
        # 【优先级5】时间序列场景（趋势分析）
        trend_keywords = ['趋势', '变化', '走势', '曲线', 'trend', '按日', '按月', '按周']
        if datetime_cols or any(kw in question_lower for kw in trend_keywords):
            return "Line Chart (px.line) - 时间序列折线图"
        
        # 【优先级6】对比场景
        if any(kw in question_lower for kw in ['对比', '比较', 'compare', 'vs']):
            return "Bar Chart (px.bar) - 分组柱状图"
        
        # 【默认】根据数据结构推荐
        if len(numeric_cols) >= 2:
            return "Scatter Plot (px.scatter) - 散点图"
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            return "Bar Chart (px.bar) - 柱状图"
        else:
            return None  # 无合适图表，返回表格

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df: pd.DataFrame = None, chart_recommendation: str = None, **kwargs
    ) -> str:
        """
        优化版可视化代码生成，传递真实数据样本给LLM
        
        Args:
            question: 用户原始问题
            sql: 生成的SQL查询
            df: 查询结果DataFrame（真实数据）
            chart_recommendation: 外部传入的图表推荐（如果为None则表示不需要图表）
            **kwargs: 其他参数
        """
        if df is None:
            raise ValueError("DataFrame cannot be None for visualization generation")
        
        # 【关键】如果外部传入 chart_recommendation=None，说明是明细数据，直接返回 None
        if chart_recommendation is None:
            print(f"[DEBUG] 明细数据场景（外部传入），不生成可视化代码")
            return None
        
        # 检测用户问题的语言
        user_lang = self.detect_language(question) if question else 'zh'
        
        # 1. 数据分析
        row_count = len(df)
        col_count = len(df.columns)
        
        # 识别数值列、分类列、时间列
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 数据样本（前5行，转为JSON格式）
        sample_data = df.head(min(5, row_count)).to_dict('records')
        
        # 数据统计信息
        stats_info = []
        for col in numeric_cols[:5]:  # 最多展示5个数值列的统计
            try:
                stats_info.append(
                    f"  - {col}: 最小值={df[col].min()}, 最大值={df[col].max()}, "
                    f"平均值={df[col].mean():.2f}, 唯一值数={df[col].nunique()}"
                )
            except Exception:
                pass
        
        # 2. 如果没有传入推荐，则调用内部推荐逻辑
        if not chart_recommendation:
            chart_recommendation = self._recommend_chart_type(df, question or "")
            # 【关键】如果推荐结果是 None，说明是明细数据，不应该生成图表
            if chart_recommendation is None:
                print(f"[DEBUG] 明细数据场景（内部检测），不生成可视化代码")
                return None  # 返回 None，不生成图表
        
        # 3. 构建提示词
        if user_lang == 'zh':
            system_prompt = f"""你是一位数据可视化专家，精通 Plotly 图表库。

===用户问题===
{question or '数据可视化'}

===SQL查询===
{sql or 'N/A'}

===数据集信息===
- 数据维度: {row_count} 行 × {col_count} 列
- 数值列: {', '.join(numeric_cols) if numeric_cols else '无'}
- 分类列: {', '.join(categorical_cols) if categorical_cols else '无'}
- 时间列: {', '.join(datetime_cols) if datetime_cols else '无'}

===数据类型===
{df.dtypes.to_string()}

===数据样本（前{min(5, row_count)}行真实数据）===
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

===数值列统计===
{chr(10).join(stats_info) if stats_info else '无数值列'}

===推荐图表类型===
{chart_recommendation}
"""
            
            user_prompt = """请为这个数据集生成专业的 Plotly 可视化代码，要求：

**图表选择原则**：
1. 时间序列数据 → 使用 px.line()（折线图），X轴为时间字段
2. 分类占比数据 → 使用 px.pie() 或 px.bar()（饼图/柱状图）
3. 多维度对比 → 使用 px.bar()（分组柱状图）
4. 单一指标 → 使用 go.Indicator()（指示器卡片）
5. 排名数据 → 使用 px.bar()（横向柱状图）

**代码质量要求**：
1. 必须正确识别 X轴 和 Y轴 字段（参考上面的数据样本）
2. 为图表设置有意义的中文标题（基于用户问题）
3. 设置清晰的轴标签（使用 labels 参数）
4. 如果有时间字段，确保正确识别并作为X轴
5. 优化布局（font 字体大小、hovermode 悬停模式）
6. 对于时间类数据，使用 pd.to_datetime() 转换（如果需要）
7. 对于大数据集（>100行），考虑适当聚合

**输出格式**：
- 只输出可执行的 Python 代码，不要任何文字解释
- 代码中 DataFrame 变量名必须是 'df'
- 图表对象变量名必须是 'fig'
- 不要包含 fig.show()
- 必须 import 所需的库（import plotly.express as px 或 import plotly.graph_objects as go）

**示例代码结构**：
```python
import plotly.express as px

# 如果需要预处理时间字段
# df['date_col'] = pd.to_datetime(df['date_col'], format='%Y%m%d')

# 生成图表
fig = px.line(
    df, 
    x='时间字段名',  # 替换为实际字段
    y='数值字段名',  # 替换为实际字段
    title='用户友好的中文标题',
    labels={'时间字段名': '时间', '数值字段名': '数值'}
)

# 优化布局
fig.update_layout(
    font=dict(size=14),
    hovermode='x unified'
)
```

请立即生成代码（只输出代码，不要任何解释）：
"""
        else:  # 英文版
            system_prompt = f"""You are a data visualization expert proficient in Plotly.

===User Question===
{question or 'Data Visualization'}

===SQL Query===
{sql or 'N/A'}

===Dataset Information===
- Dimensions: {row_count} rows × {col_count} columns
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
- Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'None'}

===Data Types===
{df.dtypes.to_string()}

===Data Sample (first {min(5, row_count)} rows)===
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

===Numeric Statistics===
{chr(10).join(stats_info) if stats_info else 'No numeric columns'}

===Recommended Chart Type===
{chart_recommendation}
"""
            
            user_prompt = """Generate professional Plotly visualization code for this dataset:

**Chart Selection Rules**:
1. Time series → px.line() with time on X-axis
2. Category proportions → px.pie() or px.bar()
3. Multi-dimensional comparison → px.bar() (grouped)
4. Single metric → go.Indicator()
5. Rankings → px.bar() (horizontal)

**Code Quality Requirements**:
1. Correctly identify X and Y axis fields (refer to data sample above)
2. Set meaningful chart title
3. Add clear axis labels (use labels parameter)
4. If datetime column exists, use it as X-axis
5. Optimize layout (font size, hovermode)
6. Convert datetime if needed using pd.to_datetime()
7. Consider aggregation for large datasets (>100 rows)

**Output Format**:
- Python code only, no explanations
- DataFrame variable name: 'df'
- Figure variable name: 'fig'
- Do not include fig.show()
- Must import required libraries (import plotly.express as px or import plotly.graph_objects as go)

Generate code now (code only, no explanations):
"""
        
        message_log = [
            self.system_message(system_prompt),
            self.user_message(user_prompt)
        ]
        
        # 提交给LLM生成代码
        plotly_code = self.submit_prompt(message_log, **kwargs)
        
        # 提取并清理代码
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def validate_trend_sql(self, sql: str) -> bool:
        """
        趋势类SQL最低校验: 时间字段 + 聚合 + group by + order by
        """
        if not sql:
            return False
        sql_upper = sql.upper()
        has_group_by = "GROUP BY" in sql_upper
        has_order_by = "ORDER BY" in sql_upper
        has_agg = any(k in sql_upper for k in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("])
        has_time_col = any(k in sql_upper for k in [
            "DATE", "TIME", "DAY", "MONTH", "YEAR", "HOUR", "WEEK",
            "STAT_DATE", "COLLECT_TIME", "DT"
        ])
        return has_group_by and has_order_by and has_agg and has_time_col

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

    def generate_sql_optimized(self, question: str, allow_llm_to_see_data=False, conversation_history: Optional[List[Dict]] = None, intent: Optional[Dict] = None, **kwargs):
        """
        Optimized version of generate_sql that returns both SQL and Explanation in a single pass.
        支持对话历史上下文。
        
        Args:
            question: 用户问题
            allow_llm_to_see_data: 是否允许LLM查看数据
            conversation_history: 对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 其他参数
        """
        # 0. 检测用户问题的语言
        user_lang = self.detect_language(question)
        print(f"[DEBUG] 检测到用户语言: {'中文' if user_lang == 'zh' else '英文'}")
        
        # 1. Retrieve Context
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        # 2. Construct Prompt (根据语言动态生成)
        
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

        # RAG优化：降低token限制，避免prompt过长
        # DDL部分限制在3000 tokens（约3-5个表结构）
        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=3000)

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        # Documentation部分限制在5000 tokens（约5-8条业务说明）
        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=5000)

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
                "10. **严禁参数化查询占位符**: SQL查询必须是完整可执行的语句，不要使用参数占位符如 `?` 或 `:param`。所有值必须直接嵌入SQL中（字符串用单引号包裹）。\n"
                "11. **对话上下文**: 如果用户的问题中出现\"11月\"、\"上个月\"、\"本月\"等相对时间表述，请结合之前的对话历史和当前时间信息来理解用户的意图。\n"
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
                "10. **NO Parameterized Queries**: The SQL must be a complete executable statement. DO NOT use parameter placeholders like `?` or `:param`. All values must be embedded directly in the SQL (strings wrapped in single quotes). \n"
                "11. **Conversation Context**: If the user mentions relative time expressions like \"last month\" or \"this month\", use the conversation history and current time information to understand their intent. \n"
            )

        # 意图增强提示（趋势类强约束）
        if intent and intent.get("intent") == "trend":
            if user_lang == 'zh':
                initial_prompt += (
                    "12. **趋势类SQL强制要求**: 必须包含时间维度字段（如日期/时间列），并进行聚合统计（如 SUM/COUNT/AVG），"
                    "且必须包含 `GROUP BY 时间字段` 与 `ORDER BY 时间字段`，按时间升序排列。\n"
                )
            else:
                initial_prompt += (
                    "12. **Trend SQL requirements**: Must include a time dimension column, an aggregate metric (SUM/COUNT/AVG), "
                    "and include `GROUP BY time` plus `ORDER BY time` in ascending order.\n"
                )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        # 添加对话历史到消息日志（在示例之后，当前问题之前）
        if conversation_history and len(conversation_history) > 0:
            print(f"[DEBUG] 添加 {len(conversation_history)} 条对话历史到上下文")
            for msg in conversation_history:
                if msg['role'] == 'user':
                    message_log.append(self.user_message(msg['content']))
                elif msg['role'] == 'assistant':
                    message_log.append(self.assistant_message(msg['content']))

        message_log.append(self.user_message(question))

        # 【调试日志】输出提示词与消息日志，便于排查
        print(f"\n{'='*60}")
        print("[DEBUG] Prompt (system):")
        print(initial_prompt)
        print(f"[DEBUG] message_log size: {len(message_log)}")
        for i, msg in enumerate(message_log):
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            print(f"[DEBUG] message_log[{i}] role={role}")
            print(content)
        print(f"{'='*60}\n")

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
zhipu_embedding = ZhipuAIEmbeddingFunction(config={
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'api_base': os.getenv('ZHIPU_API_BASE')  # Pass api_base to embedding function
})

config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_MODEL', 'GLM-4.7'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': './chroma_db', # Path for ChromaDB storage
    'embedding_function': zhipu_embedding,
    'dialect': 'MySQL', # 显式指定 MySQL 方言，防止 LLM 生成不兼容的 SQL 函数 (如 UNIX_TIMESTAMP 多参数)
    # RAG优化：限制检索数量，避免把整个知识库塞进prompt
    'n_results_ddl': 3,           # DDL只检索最相关的3条表结构
    'n_results_documentation': 5,  # 文档检索5条最相关的业务说明
    'n_results_sql': 5,            # SQL示例检索5条最相似的问答对
}

# Initialize Vanna
vn = MyVanna(config=config)

# 设置相似度阈值（L2距离，越小越相似）
# 调优建议：
# - 1.0: 非常严格，只保留高度相关的结果
# - 1.5: 适中（推荐起始值）
# - 2.0: 宽松，保留更多候选结果
vn.sql_distance_threshold = float(os.getenv('RAG_SQL_THRESHOLD', '1.5'))
vn.ddl_distance_threshold = float(os.getenv('RAG_DDL_THRESHOLD', '1.5'))
vn.doc_distance_threshold = float(os.getenv('RAG_DOC_THRESHOLD', '1.5'))

# 初始化对话历史管理器（使用 ChromaDB 持久化）
conversation_history = ChromaConversationHistory(
    chroma_client=vn.chroma_client,
    max_history_per_session=10
)

def run_query_and_chart(sql: str, question: Optional[str] = None):
    df = vn.run_sql(sql=sql)

    # 清理不符合 JSON 规范的数据类型
    import numpy as np
    from decimal import Decimal
    
    # 1. 将 Decimal 类型转换为 float
    for col in df.columns:
        if df[col].dtype == 'object':  # 可能包含 Decimal
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    
    # 2. 清理 inf, -inf, nan
    df = df.replace([np.inf, -np.inf], None)
    df = df.replace({np.nan: None})
    
    # 3. 智能识别并转换日期列（针对字符串类型的日期字段）
    potential_date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'dt', 'day', 'month'])]
    for col in potential_date_cols:
        if df[col].dtype == 'object':  # 字符串类型
            try:
                # 先检查数据格式
                sample_value = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                print(f"[DEBUG] 列 '{col}' 的样本值: {sample_value}")
                
                # 尝试多种日期格式
                converted = False
                
                # 格式1: yyyyMMdd (8位数字字符串，如 20251224)
                if len(sample_value) == 8 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
                    print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (格式: yyyyMMdd)")
                    converted = True
                
                # 格式2: yyyyMMddHH (10位数字字符串)
                elif len(sample_value) == 10 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d%H', errors='coerce')
                    print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (格式: yyyyMMddHH)")
                    converted = True
                
                # 格式3: ISO 格式或其他常见格式
                if not converted:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        print(f"[DEBUG] 成功将列 '{col}' 转换为 datetime 类型 (自动识别格式)")
                    else:
                        print(f"[DEBUG] 列 '{col}' 转换后全部为 NaT，可能格式不支持")
                        
            except Exception as e:
                print(f"[DEBUG] 列 '{col}' 无法转换为 datetime: {e}")
    
    # 4. 先进行意图检测，判断是否需要生成图表
    chart_json = None
    if question:
        try:
            # 意图检测
            intent = vn.detect_intent(question)
            print(f"[DEBUG] 问题意图: {intent}")
            
            # 如果意图是明细数据，直接跳过图表生成
            if intent.get('intent') == 'detail':
                print(f"[DEBUG] 明细数据场景，跳过图表生成")
            else:
                # 打印 DataFrame 信息用于调试
                print(f"\n[DEBUG] ===== DataFrame 信息 =====")
                print(f"[DEBUG] 列名: {df.columns.tolist()}")
                print(f"[DEBUG] 数据类型:\n{df.dtypes}")
                print(f"[DEBUG] 数据预览:\n{df.head()}")
                print(f"[DEBUG] 数据值:\n{df.to_dict('records')}")
                print(f"[DEBUG] ========================\n")
                
                # 先推荐图表类型
                chart_recommendation = vn._recommend_chart_type(df, question)
                
                # 如果推荐为 None（明细数据），不生成图表
                if chart_recommendation is None:
                    print(f"[DEBUG] 图表推荐为 None，不生成图表")
                else:
                    plotly_code = vn.generate_plotly_code(
                        question=question, 
                        sql=sql, 
                        df=df,
                        chart_recommendation=chart_recommendation
                    )
                    
                    # 如果返回 None，说明是明细数据，不生成图表
                    if plotly_code is None:
                        print(f"[DEBUG] 明细数据不生成图表，只返回表格")
                    else:
                        print(f"[DEBUG] 生成的 Plotly 代码:\n{plotly_code}\n")
                        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
                        if fig:
                            chart_json = fig.to_json()
        except Exception as e:
            print(f"可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 将时间类型转换为字符串（用于 JSON 序列化）
    try:
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            df[col] = df[col].apply(lambda v: v.isoformat() if pd.notna(v) and hasattr(v, "isoformat") else v)
    except Exception as e:
        print(f"[DEBUG] 时间列序列化处理失败: {e}")

    return df, chart_json

# Setup FastAPI
app = FastAPI(title="redtea chatBi MVP")

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
    session_id: Optional[str] = None  # 支持会话ID以跟踪对话历史

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
        print(f"\n{'='*60}")
        print(f"[DEBUG] 接收到问题: {request.question}")
        print(f"[DEBUG] 入参 session_id: {request.session_id}")
        print(f"{'='*60}\n")
        # 0. 获取或生成 session_id
        import uuid
        session_id = request.session_id or str(uuid.uuid4())
        
        # 0.1 检测用户问题的语言
        user_lang = vn.detect_language(request.question)

        # 0.2 意图识别（用于趋势类SQL约束）
        intent = vn.detect_intent(request.question)
        
        # 0.3 获取该会话的对话历史
        history = conversation_history.get_context_history(session_id)
        print(f"[DEBUG] Session ID: {session_id}, 历史记录数: {len(history)}")
        print(f"[DEBUG] Intent: {intent}")
        
        # 1. Generate SQL using Optimized Vanna Method (Single Pass for SQL + Explanation)
        # 将对话历史传递给 generate_sql_optimized
        raw_result = vn.generate_sql_optimized(
            question=request.question, 
            allow_llm_to_see_data=True,
            conversation_history=history,
            intent=intent
        )
        
        # 【调试日志】记录LLM原始输出,方便排查问题
        print(f"\n{'='*60}")
        print(f"[DEBUG] 用户问题: {request.question}")
        print(f"[DEBUG] Session ID: {session_id}")
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

            # 趋势类SQL校验，不通过则尝试重试一次
            if intent.get("intent") == "trend" and not vn.validate_trend_sql(clean_sql):
                print("[DEBUG] Trend SQL validation failed, retrying with stronger constraints...")
                retry_prompt = request.question + "（请严格输出趋势类SQL：时间维度+聚合+按时间分组排序）"
                raw_retry = vn.generate_sql_optimized(
                    question=retry_prompt,
                    allow_llm_to_see_data=True,
                    conversation_history=history,
                    intent=intent
                )
                retry_match = re.search(r"```sql\s*(.*?)\s*```", raw_retry, re.DOTALL | re.IGNORECASE)
                if retry_match:
                    retry_sql = retry_match.group(1).strip()
                    retry_sql = re.sub(r'^intermediate_sql\s*', '', retry_sql, flags=re.IGNORECASE).strip()
                    if vn.validate_trend_sql(retry_sql):
                        clean_sql = retry_sql
                        print("[DEBUG] Trend SQL validation passed after retry.")
            
            # 保存对话历史
            conversation_history.add_message(session_id, "user", request.question)
            print(f"[DEBUG] 已保存对话历史到 session {session_id}")

            try:
                df, chart_json = run_query_and_chart(clean_sql, request.question)
            except pymysql.Error as e:
                error_msg = f"数据库错误: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            result_rows = df.to_dict(orient='records')
            columns = df.columns.tolist()

            conversation_history.add_message(
                session_id,
                "assistant",
                f"{explanation}\n\nSQL: {clean_sql}",
                data={
                    "question": request.question,
                    "sql": clean_sql,
                    "explanation": explanation,
                    "result": result_rows,
                    "columns": columns,
                    "chart": chart_json,
                    "intent": intent,
                }
            )

            return {
                "sql": clean_sql,
                "is_sql": True,
                "explanation": explanation,
                "session_id": session_id,  # 返回session_id给前端
                "result": result_rows,
                "columns": columns,
                "chart": chart_json,
                "intent": intent,
            }
        else:
            # No SQL found, treat as conversational response
            print(f"[DEBUG] {'未检测到SQL，作为对话回复处理' if user_lang == 'zh' else 'No SQL detected, treating as conversational response'}")
            
            # 保存对话历史
            conversation_history.add_message(session_id, "user", request.question)
            conversation_history.add_message(session_id, "assistant", raw_result)
            print(f"[DEBUG] 已保存对话历史到 session {session_id}")
            
            return {
                "text": raw_result,
                "is_sql": False,
                "explanation": raw_result,
                "session_id": session_id  # 返回session_id给前端
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
        df, chart_json = run_query_and_chart(request.sql, request.question)

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

@app.post("/api/v0/clear_session")
def clear_session(session_id: str = Body(..., embed=True)):
    """清除指定会话的对话历史"""
    try:
        conversation_history.clear_session(session_id)
        print(f"[DEBUG] 已清除 session {session_id} 的历史记录")
        return {"success": True, "message": "会话历史已清除"}
    except Exception as e:
        error_msg = f"清除会话失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v0/sessions")
def get_sessions():
    """获取所有活跃会话"""
    return conversation_history.get_sessions()

@app.get("/api/v0/history/{session_id}")
def get_history(session_id: str):
    """获取指定会话的历史"""
    return conversation_history.get_history(session_id)

@app.post("/api/v0/sessions/{session_id}/rename")
def rename_session(session_id: str, body: Dict = Body(...)):
    """重命名会话"""
    new_title = body.get("title")
    if conversation_history.rename_session(session_id, new_title):
        return {"success": True, "message": "Session renamed"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/api/v0/sessions/{session_id}")
def delete_session(session_id: str):
    """删除指定会话"""
    conversation_history.clear_session(session_id)
    return {"success": True, "message": "Session deleted"}

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
    return {"message": "redtea chatBi MVP is running. Use /api/v0/ for API endpoints."}

# Simple HTML UI for testing
@app.get("/ui", response_class=FileResponse)
def ui():
    return FileResponse("templates/ui.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
