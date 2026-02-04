import os
import re
import time
import json
import uuid
from datetime import datetime
from decimal import Decimal
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
from fastapi import FastAPI, Body, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pymysql
import pandas as pd
from openai import OpenAI
from typing import Optional, List, Dict, Any

# 核心配置与认证（拆到 core）
from core.config import log_env_check
from core.auth import get_current_user, transform_sql_with_tenant, require_operator_id as _require_operator_id
from core.schemas import QuestionRequest, SqlRequest, LoginRequest, FeedbackRequest
from core.conversation_history import ChromaConversationHistory

# 业务服务（拆到 services）
from services.lark_notification import send_lark_alert
from services import query_chart as query_chart_service
from services import pinned_service as pinned_service_module

# 数据源和知识库管理
from datasource_manager import init_datasource_manager, get_datasource_manager
from knowledge_base_manager import init_kb_manager, get_kb_manager
from analytics_manager import init_analytics_manager, get_analytics_manager
from insight_engine import init_insight_engine, get_insight_engine
from insight_scheduler import start_insight_scheduler, init_alert_scheduler

log_env_check()

# ChromaConversationHistory 已迁至 core.conversation_history，在 vn 创建后实例化


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Initialize ChromaDB
        ChromaDB_VectorStore.__init__(self, config=config)

        # Initialize OpenAI Client for Zhipu AI
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['api_base']
        )

        self.sql_model = os.getenv("OPENROUTER_SQL_MODEL", "openai/gpt-5.1")
        self.sql_client = self._build_openrouter_client()

        # Remove api_base from config to avoid Vanna legacy check error
        vanna_config = config.copy()
        if 'api_base' in vanna_config:
            del vanna_config['api_base']

        # Initialize OpenAI_Chat with the client
        OpenAI_Chat.__init__(self, client=client, config=vanna_config)

    def _build_openrouter_client(self) -> Optional[OpenAI]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        referer = os.getenv("OPENROUTER_REFERER")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name

        if headers:
            return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        return OpenAI(api_key=api_key, base_url=base_url)

    def _submit_sql_prompt(self, prompt, **kwargs) -> str:
        if self.sql_client is None:
            return self.submit_prompt(prompt, **kwargs)

        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        model = kwargs.get("model", self.sql_model)
        response = self.sql_client.chat.completions.create(
            model=model,
            messages=prompt,
            stop=None,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

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

    def detect_intent(self, text: str, use_ai: bool = None) -> Dict[str, str]:
        """
        智能意图识别（支持AI模型和规则两种模式）
        返回: intent, preferred_chart, time_grain, agg_hint
        
        Args:
            text: 用户问题文本
            use_ai: 是否使用AI模型进行意图识别（None时根据环境变量决定）
        """
        # 如果未指定use_ai，则从环境变量读取配置
        if use_ai is None:
            use_ai = os.getenv("USE_AI_INTENT_DETECTION", "false").lower() == "true"
        
        # 如果启用AI意图识别，优先使用AI模型
        if use_ai:
            try:
                ai_result = self._detect_intent_with_ai(text)
                if ai_result:
                    print(f"[DEBUG] AI意图识别结果: {ai_result}")
                    return ai_result
                else:
                    print(f"[DEBUG] AI意图识别失败，回退到规则模式")
            except Exception as e:
                print(f"[DEBUG] AI意图识别异常: {e}，回退到规则模式")
        
        # 回退到规则模式
        return self._detect_intent_with_rules(text)
    
    def _detect_intent_with_ai(self, text: str) -> Optional[Dict[str, str]]:
        """
        使用AI模型进行意图识别
        """
        api_key = os.getenv("ZHIPU_API_KEY")
        model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4-Flash")
        
        if not api_key or not model:
            return None
        
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            print("[DEBUG] ZhipuAI SDK未安装，无法使用AI意图识别")
            return None
        
        client = ZhipuAI(api_key=api_key)
        
        # 构建意图识别prompt
        prompt = f"""你是一个数据分析意图识别专家。请分析用户的问题，判断用户的查询意图。

用户问题：{text}

请按照以下JSON格式返回结果（只返回JSON，不要其他内容）：
{{
    "intent": "意图类型（trend/distribution/ranking/comparison/detail/aggregation）",
    "preferred_chart": "推荐图表类型（line/bar/pie/table/scatter）",
    "time_grain": "时间粒度（day/month/week/hour/''）",
    "agg_hint": "聚合提示（sum/avg/count/''）",
    "confidence": 0.95,
    "reasoning": "简短说明识别理由"
}}

意图类型说明：
- trend: 趋势分析（包含增长率、变化、走势、同比、环比等）
- distribution: 分布占比（包含占比、比例、构成、份额等）
- ranking: 排行榜（包含top、排名、排行、前几等）
- comparison: 对比分析（包含对比、比较、差异等）
- detail: 明细数据（明确要求查看明细、详情、列表等，且不含其他分析意图）
- aggregation: 聚合统计（总数、平均值、计数等单一指标）

图表类型说明：
- line: 折线图（适用于趋势分析）
- bar: 柱状图（适用于排行、对比）
- pie: 饼图（适用于占比分布）
- table: 表格（适用于明细数据）
- scatter: 散点图（适用于相关性分析）

重要规则：
1. 如果问题中同时包含"明细"和"趋势/增长率"等词，应优先判断为trend而非detail
2. "增长率"、"增长"、"下降"、"上升"等都属于趋势分析
3. 只有明确要求查看明细数据且不含任何分析意图时才判断为detail"""

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=500,
                temperature=0.1,  # 低温度以获得更稳定的结果
                top_p=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            
            content = response.choices[0].message.content if response.choices else ""
            if not content:
                return None
            
            # 解析JSON响应
            import json
            # 尝试提取JSON（可能被markdown包裹）
            content = content.strip()
            if content.startswith("```"):
                # 移除markdown代码块
                lines = content.split("\n")
                content = "\n".join([l for l in lines if not l.strip().startswith("```")])
            
            result = json.loads(content)
            
            # 验证必需字段
            if not all(k in result for k in ["intent", "preferred_chart"]):
                print(f"[DEBUG] AI返回结果缺少必需字段: {result}")
                return None
            
            # 补充默认值
            return {
                "intent": result.get("intent", "detail"),
                "preferred_chart": result.get("preferred_chart", "table"),
                "time_grain": result.get("time_grain", ""),
                "agg_hint": result.get("agg_hint", ""),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
            }
            
        except Exception as e:
            print(f"[DEBUG] AI意图识别失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_intent_with_rules(self, text: str) -> Dict[str, str]:
        """
        基于规则的意图识别（原有逻辑）
        """
        q = (text or "").lower()
        intent = "detail"
        preferred_chart = "table"
        time_grain = ""
        agg_hint = ""

        # 关键词优先级调整：明细优先级最高
        detail_keywords = ["明细", "详情", "列表", "详细信息", "明细数据", "数据明细"]
        # 增加更多趋势关键词，包括增长率、下降、上升等
        trend_keywords = ["趋势", "变化", "走势", "曲线", "按日", "按月", "按周", "按小时", "同比", "环比", 
                         "增长率", "增长", "下降", "上升", "增速", "降幅", "涨幅"]
        distribution_keywords = ["分布", "占比", "比例", "构成", "份额"]
        compare_keywords = ["对比", "比较", "差异"]
        ranking_keywords = ["top", "排行", "排名", "前几"]

        # 【优先级1】明细数据 - 必须优先判断（且不与其他意图冲突）
        # 只有明确提到明细关键词且不包含趋势、分布等关键词时才判定为明细
        has_detail_keyword = any(k in q for k in detail_keywords)
        has_trend_keyword = any(k in q for k in trend_keywords)
        has_distribution_keyword = any(k in q for k in distribution_keywords)
        has_compare_keyword = any(k in q for k in compare_keywords)
        has_ranking_keyword = any(k in q for k in ranking_keywords)
        
        if has_detail_keyword and not (has_trend_keyword or has_distribution_keyword or has_compare_keyword or has_ranking_keyword):
            intent = "detail"
            preferred_chart = "table"
        # 【优先级2】排行数据
        elif has_ranking_keyword:
            intent = "ranking"
            preferred_chart = "bar"
        # 【优先级3】趋势分析（包括增长率、变化等）
        elif has_trend_keyword:
            intent = "trend"
            preferred_chart = "line"
            agg_hint = "sum"
        # 【优先级4】分布占比
        elif has_distribution_keyword:
            intent = "distribution"
            preferred_chart = "pie"
        # 【优先级5】对比分析
        elif has_compare_keyword:
            intent = "comparison"
            preferred_chart = "bar"

        # 时间粒度检测（更精确的匹配，避免误判）
        if "按日" in q or "每日" in q:
            time_grain = "day"
        elif "按月" in q or "每月" in q:
            time_grain = "month"
        elif "按周" in q or "每周" in q:
            time_grain = "week"
        elif "按小时" in q or "每小时" in q:
            time_grain = "hour"

        return {
            "intent": intent,
            "preferred_chart": preferred_chart,
            "time_grain": time_grain,
            "agg_hint": agg_hint,
        }

    def _infer_chart_intent_with_ai(
        self, question: str, df: pd.DataFrame, columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        根据问题、结果列名与数据特征，用 AI 推断图表类型及用列（更精确，尤其当用户未说「占比」但数据适合饼图时）。
        需设置环境变量 CHART_INTENT_USE_AI=true 且 ZHIPU_API_KEY 可用。返回含 preferred_chart、chart_label_col、chart_value_col 等。
        """
        api_key = os.getenv("ZHIPU_API_KEY")
        model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4-Flash")
        if not api_key or not model:
            return None
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            return None
        row_count = len(df)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        sample = df.head(2).to_dict(orient="records") if row_count > 0 else []
        prompt = f"""根据用户问题与查询结果列信息，推断最适合的图表类型及用列。只返回 JSON，不要其他文字。

用户问题：{question}
结果列名：{columns}
行数：{row_count}
数值列：{numeric_cols}
前两行示例：{sample}

请按以下 JSON 格式返回（只返回 JSON）：
{{
  "preferred_chart": "line|bar|pie|scatter|table",
  "chart_label_col": "用于饼图扇区名/柱状图X轴的列名（从上面列名中选）",
  "chart_value_col": "用于饼图数值/柱状图Y轴的列名（从上面列名中选）",
  "chart_x_col": "折线图X轴列名（若有时间列可填）",
  "reasoning": "简短理由"
}}

规则：若结果为分类+占比/数量且行数较少（如<=15），preferred_chart 应为 pie，chart_label_col 为类别列，chart_value_col 为占比或数量列。"""
        try:
            client = ZhipuAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return None
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))
            import json
            data = json.loads(content)
            preferred = (data.get("preferred_chart") or "").lower()
            if preferred not in ("line", "bar", "pie", "scatter", "table"):
                return None
            result = {"preferred_chart": preferred}
            if data.get("chart_label_col") and data["chart_label_col"] in columns:
                result["chart_label_col"] = data["chart_label_col"]
            if data.get("chart_value_col") and data["chart_value_col"] in columns:
                result["chart_value_col"] = data["chart_value_col"]
            if data.get("chart_x_col") and data["chart_x_col"] in columns:
                result["chart_x_col"] = data["chart_x_col"]
            return result
        except Exception as e:
            print(f"[DEBUG] AI 图表意图推断失败: {e}")
            return None

    def adjust_intent_for_result(
        self, question: str, df: pd.DataFrame, columns: List[str], intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据执行结果列名、数据类型与问题，统一推荐图表类型及用列（折线/柱状/饼图/散点）。
        返回更新后的 intent，可含 preferred_chart、chart_x_col、chart_label_col、chart_value_col、chart_value_cols、chart_y_col。
        """
        out = dict(intent)
        row_count = len(df)
        if row_count == 0:
            return out
        # 可选：使用 AI 根据结果列+问题推断图表类型（更精确，如数据适合饼图但用户未说「占比」）
        if os.getenv("CHART_INTENT_USE_AI", "false").lower() == "true":
            ai_intent = self._infer_chart_intent_with_ai(question, df, columns)
            if ai_intent and ai_intent.get("preferred_chart"):
                out.update(ai_intent)
                print(f"[DEBUG] 图表意图已由 AI 推断: {ai_intent}")
                return out
        col_lower = {c: (c or "").lower() for c in columns}
        q_lower = (question or "").lower()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # 时间列（折线 X 轴），避免 "stat" 匹配到 status_percentage
        time_keywords = ["date", "time", "_dt", "day", "month", "year", "stat_date", "stat_day", "create_at", "update_at"]
        time_col = None
        for c in columns:
            low = col_lower[c]
            if any(kw in low for kw in time_keywords):
                time_col = c
                break
            if low in ("dt", "date", "time", "day", "month", "year"):
                time_col = c
                break
        # 描述/类别列（柱状 X 轴、饼图扇区名）：按列名语义推断，不写死列名，AI 生成任意列名只要含这些关键词即可
        label_keywords = [
            "name", "desc", "描述", "名称", "类型", "类别", "label", "标题", "说明", "text", "标题",
            "status_name", "status_desc", "_name", "_desc", "type_name", "category",
        ]
        label_col = None
        for c in columns:
            if any(kw in col_lower[c] for kw in label_keywords):
                label_col = c
                break
        # 数值列：占比类 → 饼图用；计数/金额 → 柱状/折线用。同样按语义关键词推断，不写死列名
        percent_keywords = ["percentage", "percent", "占比", "比例", "率", "份额", "pct", "ratio", "比例值", "percent_of_total", "percent_of"]
        count_keywords = ["count", "cnt", "数量", "笔数", "amount", "sum", "total", "value", "值", "status_count", "biz_count", "num", "数量"]
        value_col = None
        for c in columns:
            if any(kw in col_lower[c] for kw in percent_keywords):
                value_col = c
                break
        if value_col is None:
            for c in columns:
                if c in numeric_cols and any(kw in col_lower[c] for kw in count_keywords):
                    value_col = c
                    break
        if value_col is None and numeric_cols:
            value_col = numeric_cols[0]
        # 数值列用于 Y 轴：按语义关键词推断（不写死列名），排除纯 status/code/id 列
        value_like_kw = ["count", "cnt", "amount", "sum", "total", "value", "数量", "笔数", "金额", "值", "percentage", "percent", "占比", "比例", "pct", "num"]
        status_like_kw = ["status", "code", "id", "状态", "编码"]
        def _is_value_like(col: str) -> bool:
            low = col_lower.get(col, "")
            return any(kw in low for kw in value_like_kw)
        def _is_status_like(col: str) -> bool:
            low = col_lower.get(col, "")
            if any(kw in low for kw in value_like_kw):
                return False
            return any(kw in low for kw in status_like_kw) or (low.endswith("_id") or low.endswith("id"))
        value_cols = [c for c in numeric_cols if _is_value_like(c)]
        if not value_cols:
            value_cols = [c for c in numeric_cols if not _is_status_like(c)]
        if not value_cols:
            value_cols = numeric_cols[:5]
        if value_col and value_col in numeric_cols:
            if value_col not in value_cols:
                value_cols = [value_col] + value_cols[:4]
            else:
                value_cols = [value_col] + [c for c in value_cols if c != value_col][:4]
        else:
            value_cols = value_cols[:5]
        # 问题意图关键词
        trend_kw = ["趋势", "变化", "走势", "增长率", "增长", "下降", "上升", "同比", "环比", "trend"]
        dist_kw = ["占比", "比例", "分布", "构成", "份额", "percent", "proportion", "distribution"]
        rank_kw = ["top", "排行", "排名", "前几", "排名"]
        comp_kw = ["对比", "比较", "差异"]
        has_trend = any(kw in q_lower for kw in trend_kw)
        has_distribution = any(kw in q_lower for kw in dist_kw)
        has_ranking = any(kw in q_lower for kw in rank_kw)
        has_comparison = any(kw in q_lower for kw in comp_kw)
        has_percentage_col = any(any(kw in col_lower[c] for kw in percent_keywords) for c in columns)
        # 1) 占比/分布 → 饼图
        if (has_distribution or has_percentage_col) and row_count <= 15 and value_col and (label_col or columns[0] != value_col):
            out["preferred_chart"] = "pie"
            out["chart_label_col"] = label_col if label_col else columns[0]
            out["chart_value_col"] = value_col
            return out
        # 2) 趋势 + 有时间列 → 折线图
        if (has_trend or time_col) and time_col and value_cols:
            out["preferred_chart"] = "line"
            out["chart_x_col"] = time_col
            out["chart_value_cols"] = value_cols[:5]
            if len(value_cols) == 1:
                out["chart_value_col"] = value_cols[0]
            return out
        # 3) 排行/对比 或 有类别列无时间 → 柱状图
        if (has_ranking or has_comparison or label_col) and value_cols:
            out["preferred_chart"] = "bar"
            out["chart_label_col"] = label_col if label_col else columns[0]
            out["chart_value_cols"] = value_cols[:5]
            if len(value_cols) == 1:
                out["chart_value_col"] = value_cols[0]
            return out
        # 4) 仅两个数值列且问题像相关性 → 散点图（可选）
        if len(numeric_cols) >= 2 and ("相关" in q_lower or "correlation" in q_lower):
            out["preferred_chart"] = "scatter"
            out["chart_x_col"] = numeric_cols[0]
            out["chart_y_col"] = numeric_cols[1]
            return out
        # 5) 默认：有时间用折线，否则柱状
        if time_col and value_cols:
            out["preferred_chart"] = "line"
            out["chart_x_col"] = time_col
            out["chart_value_cols"] = value_cols[:5]
            if value_cols:
                out["chart_value_col"] = value_cols[0]
        else:
            out["preferred_chart"] = "bar"
            out["chart_label_col"] = label_col if label_col else columns[0]
            out["chart_value_cols"] = value_cols[:5]
            if value_cols:
                out["chart_value_col"] = value_cols[0]
        return out

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
        # 强匹配：只有明确包含明细关键词且不包含趋势、分布等关键词时才判定为明细
        strong_detail_keywords = ['明细', '详情', '列表', '详细信息', 'detail', 'list', 'records']
        trend_keywords_check = ['趋势', '变化', '走势', '增长率', '增长', '下降', '上升', '增速', '降幅', '涨幅', '同比', '环比']
        distribution_keywords_check = ['占比', '比例', '分布', 'proportion', 'distribution', '构成']
        
        has_detail = any(kw in question_lower for kw in strong_detail_keywords)
        has_trend = any(kw in question_lower for kw in trend_keywords_check)
        has_distribution = any(kw in question_lower for kw in distribution_keywords_check)
        
        # 只有明确是明细且不是趋势/分布时才不画图
        if has_detail and not (has_trend or has_distribution):
            print(f"[DEBUG] 检测到明细关键词且无趋势/分布关键词，不生成图表")
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
        # 时间列：除 datetime64 外，也按列名识别（因 query_chart 可能已把日期转成字符串）
        time_like_cols = [
            c for c in df.columns
            if any(kw in (c or "").lower() for kw in ["date", "time", "dt", "day", "month", "stat"])
        ]
        has_time = len(datetime_cols) > 0 or len(time_like_cols) > 0
        trend_keywords = ['趋势', '变化', '走势', '曲线', 'trend', '按日', '按月', '按周', 
                         '增长率', '增长', '下降', '上升', '增速', '降幅', '涨幅', '同比', '环比']
        if has_time and len(numeric_cols) > 0:
            return "Line Chart (px.line) - 时间序列折线图"
        if any(kw in question_lower for kw in trend_keywords):
            return "Line Chart (px.line) - 时间序列折线图"
        
        # 【优先级6】对比场景
        if any(kw in question_lower for kw in ['对比', '比较', 'compare', 'vs']):
            return "Bar Chart (px.bar) - 分组柱状图"
        
        # 【默认】根据数据结构推荐：有「时间名列+分类+数值」时优先折线图（多线）
        if has_time and categorical_cols and numeric_cols:
            return "Line Chart (px.line) - 时间序列折线图"
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

        # 2.5 数据含「时间+分类+数值」时，必须生成多序列折线图（每种类型/状态一条线）
        # 不依赖 chart_recommendation 是否为 line：有 stat_date/date+状态/类型+数值就多线
        time_like_cols = [
            c for c in df.columns
            if any(kw in (c or "").lower() for kw in ["date", "time", "dt", "day", "month", "stat"])
        ]
        has_time_col = len(datetime_cols) > 0 or len(time_like_cols) > 0
        is_line_chart = "line" in (chart_recommendation or "").lower()
        # 条件：推荐折线图 或 数据结构为「时间+分类+数值」→ 一律多线
        question_text = (question or "").lower()
        if (is_line_chart or (has_time_col and categorical_cols and numeric_cols)) and categorical_cols and numeric_cols:
            time_col = (datetime_cols[0] if datetime_cols else (time_like_cols[0] if time_like_cols else ""))

            if time_col:
                # 优先用状态/类型描述列作为分线维度，便于「不同类型或状态用多条线展示」
                cat_col = ""
                for preferred in ["status_desc", "provisioning_status", "status", "license_name", "name", "type", "category"]:
                    if preferred in df.columns:
                        cat_col = preferred
                        break
                if not cat_col:
                    for c in categorical_cols:
                        if c in time_like_cols:
                            continue
                        if any(kw in c.lower() for kw in ["status", "类型", "状态", "desc", "name", "type", "category"]):
                            cat_col = c
                            break
                if not cat_col:
                    # 不用时间列当分类，取第一个非时间列的分类列
                    for c in categorical_cols:
                        if c not in time_like_cols:
                            cat_col = c
                            break
                if not cat_col:
                    cat_col = categorical_cols[0]

                if "license_total_capacity" in df.columns and ("容量" in question_text or "capacity" in question_text):
                    value_col = "license_total_capacity"
                elif "daily_total_used" in df.columns:
                    value_col = "daily_total_used"
                else:
                    preferred_metrics = [c for c in numeric_cols if any(k in c.lower() for k in ["used", "count", "total", "amount"])]
                    value_col = preferred_metrics[0] if preferred_metrics else numeric_cols[0]
                title = (question or "趋势分析").replace("'", "").strip() or "趋势分析"
                return (
                    "import pandas as pd\n"
                    "import plotly.express as px\n\n"
                    f"df['{time_col}'] = pd.to_datetime(df['{time_col}'], errors='coerce')\n"
                    "df = df.dropna(subset=["
                    f"'{time_col}', '{cat_col}', '{value_col}'"
                    "])\n"
                    f"fig = px.line(\n"
                    f"    df,\n"
                    f"    x='{time_col}',\n"
                    f"    y='{value_col}',\n"
                    f"    color='{cat_col}',\n"
                    f"    title='{title}',\n"
                    f"    labels={{'{time_col}': '日期', '{value_col}': '数值', '{cat_col}': '类型'}},\n"
                    f")\n"
                    "fig.update_layout(font=dict(size=14), hovermode='x unified')\n"
                )
        
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
2. **多条线规则（必须遵守）**：若数据中存在类型/状态/分类等分类型字段（如 status_desc、provisioning_status、类型、状态），折线图必须使用 color=分类型列，使每种类型或状态单独一条线，不得将所有数据合并为一条线。
3. 分类占比数据 → 使用 px.pie() 或 px.bar()（饼图/柱状图）
4. 多维度对比 → 使用 px.bar()（分组柱状图）
5. 单一指标 → 使用 go.Indicator()（指示器卡片）
6. 排名数据 → 使用 px.bar()（横向柱状图）

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
2. **Multi-line rule (mandatory)**: If the data has categorical columns (e.g. type, status, status_desc, provisioning_status), use px.line(..., color=categorical_column) so each category is a separate line; never plot all data as a single line.
3. Category proportions → px.pie() or px.bar()
4. Multi-dimensional comparison → px.bar() (grouped)
5. Single metric → go.Indicator()
6. Rankings → px.bar() (horizontal)

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
        
        # 提交给LLM生成代码（优先使用 OpenRouter）
        if self.sql_client is not None:
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = os.getenv("OPENROUTER_SQL_MODEL", self.sql_model)
            plotly_code = self._submit_sql_prompt(message_log, **call_kwargs)
        else:
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

    def generate_sql_optimized(
        self,
        question: str,
        allow_llm_to_see_data=False,
        conversation_history: Optional[List[Dict]] = None,
        intent: Optional[Dict] = None,
        operator_id: Optional[str] = None,
        **kwargs,
    ):
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

        # 去重：避免提示词重复导致长度膨胀与响应变慢
        def _dedupe_strings(values: List[str]) -> List[str]:
            seen = set()
            result = []
            for item in values:
                key = (item or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        def _dedupe_question_sql(values: List[Dict]) -> List[Dict]:
            seen = set()
            result = []
            for item in values or []:
                if not isinstance(item, dict):
                    continue
                q = (item.get("question") or "").strip()
                s = (item.get("sql") or "").strip()
                key = (q, s)
                if q and s and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        ddl_list = _dedupe_strings(ddl_list or [])
        doc_list = _dedupe_strings(doc_list or [])
        question_sql_list = _dedupe_question_sql(question_sql_list or [])

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
            static_doc_key = self.static_documentation.strip()
            if static_doc_key and static_doc_key not in {d.strip() for d in doc_list}:
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
                "8. **MariaDB兼容**: 不要使用窗口函数（如 LAG/LEAD/OVER）。需要前一日/环比/同比时，用子查询或自连接实现。\n"
                "9. **语言要求**: 所有解释和说明文字必须使用中文。\n"
                "10. **多数据库架构要求**: 本系统使用多数据库架构，SQL中的所有表名必须使用完全限定格式 `数据库名.表名`（例如: `SELECT * FROM database_name.table_name`）。绝不能省略数据库名，否则会导致执行错误。\n"
                "11. **严禁参数化查询占位符**: SQL查询必须是完整可执行的语句，不要使用参数占位符如 `?` 或 `:param`。所有值必须直接嵌入SQL中（字符串用单引号包裹）。\n"
                "12. **对话上下文**: 如果用户的问题中出现\"11月\"、\"上个月\"、\"本月\"等相对时间表述，请结合之前的对话历史和当前时间信息来理解用户的意图。\n"
                "13. **结果列命名与图表识别**: 为便于系统自动选择合适图表，SELECT 结果列名建议包含语义标识。占比/分布类查询：类别列别名建议含 name/desc/label/名称/描述/类型/status_name 等其一，数值列别名建议含 percent/percentage/count/占比/比例/数量/status_count/status_percent 等其一（例如 AS status_name, AS status_count, AS status_percent）。趋势类：结果中应包含日期/时间列且列名含 date/time/day/month 等。排行/对比类：建议含类别名与数量/金额列。\n"
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
                "8. **MariaDB compatibility**: Do NOT use window functions (e.g., LAG/LEAD/OVER). For previous-day or period-over-period comparisons, use subqueries or self-joins. \n"
                "9. **Language requirement**: All explanations must be in English. \n"
                "10. **Multi-Database Architecture Requirement**: This system uses a multi-database architecture. ALL table names in SQL queries MUST use the fully qualified format `database_name.table_name` (e.g., `SELECT * FROM database_name.table_name`). Never omit the database name or the query will fail. \n"
                "11. **NO Parameterized Queries**: The SQL must be a complete executable statement. DO NOT use parameter placeholders like `?` or `:param`. All values must be embedded directly in the SQL (strings wrapped in single quotes). \n"
                "12. **Conversation Context**: If the user mentions relative time expressions like \"last month\" or \"this month\", use the conversation history and current time information to understand their intent. \n"
                "13. **Result column naming for chart detection**: Use SELECT aliases that help the system pick the right chart. For proportion/distribution queries: use a category column alias containing name/desc/label/status_name and a value column containing percent/percentage/count/status_count/status_percent (e.g. AS status_name, AS status_percent). For time series: include a date/time column. For rankings: include label and count/amount columns. \n"
            )

        # 意图增强提示（趋势类强约束）
        if intent and intent.get("intent") == "trend":
            if user_lang == 'zh':
                initial_prompt += (
                    "14. **趋势类SQL强制要求**: 必须包含时间维度字段（如日期/时间列），并进行聚合统计（如 SUM/COUNT/AVG），"
                    "且必须包含 `GROUP BY 时间字段` 与 `ORDER BY 时间字段`，按时间升序排列。\n"
                )
            else:
                initial_prompt += (
                    "14. **Trend SQL requirements**: Must include a time dimension column, an aggregate metric (SUM/COUNT/AVG), "
                    "and include `GROUP BY time` plus `ORDER BY time` in ascending order.\n"
                )
        # 占比/分布类：强化结果列命名，便于系统识别饼图
        if intent and intent.get("intent") == "distribution":
            if user_lang == 'zh':
                initial_prompt += (
                    "14. **占比/分布类SQL结果列命名**: 本查询将用于饼图展示。SELECT 中请使用别名："
                    "类别列别名须包含 name/desc/label/名称/类型/status_name 等其一（如 AS status_name），"
                    "数值列别名须包含 percent/percentage/count/占比/数量/status_count/status_percent 等其一（如 AS status_count, AS status_percent）。\n"
                )
            else:
                initial_prompt += (
                    "14. **Distribution/Pie chart SQL column aliases**: This result will be used for a pie chart. "
                    "Use SELECT aliases: category column alias must contain name/desc/label/status_name, "
                    "value column alias must contain percent/percentage/count/status_count/status_percent (e.g. AS status_name, AS status_percent).\n"
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

        # 3. Submit Prompt (SQL生成使用专用模型)
        call_kwargs = dict(kwargs)
        if self.sql_client is not None:
            call_kwargs["model"] = os.getenv("OPENROUTER_SQL_MODEL", self.sql_model)
            llm_response = self._submit_sql_prompt(message_log, **call_kwargs)
        else:
            call_kwargs["model"] = os.getenv("ZHIPU_MODEL", "GLM-4.7")
            llm_response = self.submit_prompt(message_log, **call_kwargs)

        # 4. Handle Intermediate SQL (Simplified for optimization)
        if "intermediate_sql" in llm_response and allow_llm_to_see_data:
            # If intermediate SQL is needed, we might have to do the standard flow or just return.
            # For now, let's use the standard extract_sql to get the query, run it, and re-prompt.
            # This is the slow path, but necessary for correctness if hit.
            intermediate_sql = self.extract_sql(llm_response)
            try:
                df = self.run_sql(intermediate_sql, operator_id=operator_id)
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
                if self.sql_client is not None:
                    llm_response = self._submit_sql_prompt(message_log, **call_kwargs)
                else:
                    llm_response = self.submit_prompt(message_log, **call_kwargs)
            except Exception as e:
                if user_lang == 'zh':
                    return f"执行中间SQL时出错: {e}"
                else:
                    return f"Error running intermediate SQL: {e}"

        return llm_response

    def run_sql(self, sql: str, **kwargs):
        # Custom SQL Runner to handle SSL and Multi-DB
        operator_id = kwargs.get("operator_id")
        datasource_id = kwargs.get("datasource_id")  # 支持指定数据源
        
        try:
            sql = transform_sql_with_tenant(sql, operator_id)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        # 尝试使用数据源管理器获取连接配置
        try:
            ds_manager = get_datasource_manager()
            db_config = ds_manager.get_connection_config(datasource_id)
        except Exception as e:
            print(f"[run_sql] 使用数据源管理器失败，降级到环境变量配置: {e}")
            # 降级到环境变量配置
            db_config = {
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'ssl': {
                    'check_hostname': False,
                    'verify_mode': False  # equivalent to CERT_NONE
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

base_dir = os.path.dirname(os.path.abspath(__file__))
default_chroma_path = os.path.join(base_dir, "chroma_db")
config = {
    'api_key': os.getenv('ZHIPU_API_KEY'),
    'model': os.getenv('ZHIPU_YULIAO_MODEL', 'GLM-4.5'),
    'api_base': os.getenv('ZHIPU_API_BASE'),
    'path': os.getenv('CHROMA_DB_PATH', default_chroma_path),  # Path for ChromaDB storage
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

# 初始化数据源管理器
print("[INIT] 初始化数据源管理器...")
ds_manager = init_datasource_manager(vn.chroma_client)

# 初始化知识库管理器
print("[INIT] 初始化知识库管理器...")
kb_manager = init_kb_manager(vn.chroma_client, zhipu_embedding)
print("[INIT] 知识库管理器初始化完成")

# 初始化系统指标与洞察引擎
print("[INIT] 初始化 Analytics 管理器...")
analytics_manager = init_analytics_manager(vn.chroma_client)
print("[INIT] 初始化 Insight 引擎...")
insight_engine = init_insight_engine(vn.chroma_client)

# 初始化飞书服务
print("[INIT] 初始化飞书群管理服务...")
from services.lark_group_manager import init_lark_services
lark_group_manager, lark_report_sender = init_lark_services(vn.chroma_client)

def run_query_and_chart(
    sql: str,
    question: Optional[str] = None,
    operator_id: Optional[str] = None,
    datasource_id: Optional[str] = None,
):
    """委托给 services.query_chart 执行 SQL 并生成图表。"""
    return query_chart_service.run_query_and_chart(vn, sql, question, operator_id, datasource_id)


# Setup FastAPI
app = FastAPI(title="redtea chatBi MVP")

# Mount static files (for serving images)
app.mount("/img", StaticFiles(directory="img"), name="img")

# Mount static files (for serving videos and other static assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import and Include Knowledge Base Router
from knowledge_base_api import router as kb_router
app.include_router(
    kb_router,
    prefix="/api/v0",
    tags=["Knowledge Base"],
    dependencies=[Depends(get_current_user)],
)

# Import and Include DataSource Router
from datasource_api import router as ds_router
app.include_router(
    ds_router,
    prefix="/api/v0",
    tags=["DataSource"],
    dependencies=[Depends(get_current_user)],
)

# 认证与配置（拆到 routers）
from routers.auth_routes import router as auth_router
app.include_router(auth_router, prefix="/api/v0", tags=["Auth"])

# 根路径、UI、generate_questions（拆到 routers）
from routers.misc_routes import router as misc_router, api_router
app.include_router(misc_router)
app.include_router(api_router, prefix="/api/v0", tags=["Misc"])

# 对话与 SQL（拆到 routers.chat_routes）
from routers.chat_routes import router as chat_router
app.include_router(chat_router, prefix="/api/v0", tags=["Chat"])

# 分析看板、设置、收藏（拆到 routers.analytics_routes）
from routers.analytics_routes import router as analytics_router
app.include_router(analytics_router, prefix="/api/v0", tags=["Analytics"])

# 语音识别（拆到 routers.speech_routes）
from routers.speech_routes import router as speech_router
app.include_router(speech_router, prefix="/api/v0", tags=["Speech"])

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Auth-Token", "X-Auth-Expires-In"],
)

# --- Chat / Analytics 路由已迁至 routers.chat_routes / routers.analytics_routes ---
# 以下仅保留供 startup 调度器使用的 _refresh_all_pinned 及薄封装

def _refresh_pinned_item(pinned: Dict[str, Any], operator_id: str = "") -> Dict[str, Any]:
    return pinned_service_module.refresh_pinned_item(
        pinned, operator_id, conversation_history, run_query_and_chart, get_insight_engine
    )


def _refresh_all_pinned() -> None:
    pinned_service_module.refresh_all_pinned(conversation_history, run_query_and_chart, get_insight_engine)


def _auto_pin_recent(operator_id: str, username: str = "") -> None:
    pinned_service_module.auto_pin_recent(operator_id, username, conversation_history)


def _dedupe_pinned_items(items: List[Dict[str, Any]], threshold: float, mode: str) -> List[Dict[str, Any]]:
    return pinned_service_module._dedupe_pinned_items(items, threshold, mode)


# generate_questions、root、ui 已迁至 routers.misc_routes


def _check_all_alerts() -> None:
    """检测所有用户的看板预警和喜报（含疲劳度：同一图表同类型在冷却期内不重复发送）"""
    from services.alert_celebration_engine import check_pinned_alerts
    from services.lark_group_manager import get_lark_report_sender
    from services.alert_fatigue import (
        should_send,
        record_sent,
        get_effective_cooldown_hours,
    )

    try:
        operator_ids = conversation_history.list_operator_ids()
        for operator_id in operator_ids:
            usernames = conversation_history.list_usernames(operator_id=operator_id)
            if not usernames:
                usernames = [""]
            for username in usernames:
                pinned_items = conversation_history.list_pinned(
                    operator_id=operator_id, username=username
                )
                if not pinned_items:
                    continue

                user_settings = conversation_history.get_user_settings(
                    operator_id=operator_id, username=username
                )
                chart_to_item = {item.get("message_id"): item for item in pinned_items}

                result = check_pinned_alerts(pinned_items)
                alerts = result.get("alerts", [])
                celebrations = result.get("celebrations", [])

                sender = get_lark_report_sender()
                for alert in alerts:
                    chart_id = getattr(alert, "chart_id", None) or (alert.dict().get("chart_id") if hasattr(alert, "dict") else None)
                    label = (chart_to_item.get(chart_id) or {}).get("label") if chart_id else None
                    if isinstance(label, dict):
                        pass
                    elif hasattr(label, "model_dump"):
                        label = label.model_dump()
                    elif hasattr(label, "dict"):
                        label = label.dict()
                    else:
                        label = None
                    cooldown = get_effective_cooldown_hours(user_settings, label)
                    if not should_send(operator_id, chart_id or "", "alert", cooldown):
                        continue
                    try:
                        sender.send_alert("default", alert, operator_id)
                        record_sent(operator_id, chart_id or "", "alert")
                    except Exception as e:
                        print(f"[AlertScheduler] 发送预警失败: {e}")

                for celebration in celebrations:
                    chart_id = getattr(celebration, "chart_id", None) or (celebration.dict().get("chart_id") if hasattr(celebration, "dict") else None)
                    label = (chart_to_item.get(chart_id) or {}).get("label") if chart_id else None
                    if hasattr(label, "model_dump"):
                        label = label.model_dump()
                    elif hasattr(label, "dict"):
                        label = label.dict()
                    elif not isinstance(label, dict):
                        label = None
                    cooldown = get_effective_cooldown_hours(user_settings, label)
                    if not should_send(operator_id, chart_id or "", "celebration", cooldown):
                        continue
                    try:
                        sender.send_celebration("default", celebration, operator_id)
                        record_sent(operator_id, chart_id or "", "celebration")
                    except Exception as e:
                        print(f"[AlertScheduler] 发送喜报失败: {e}")

    except Exception as exc:
        print(f"[AlertScheduler] 检测预警失败: {exc}")


@app.on_event("startup")
def startup():
    # 供 routers 通过 request.app.state 获取
    app.state.vn = vn
    app.state.conversation_history = conversation_history
    app.state.run_query_and_chart = run_query_and_chart

    enabled = os.getenv("ANALYTICS_SCHEDULER_ENABLED", "true").lower() == "true"
    if not enabled:
        return
    
    # 启动洞察刷新调度器
    interval = int(os.getenv("ANALYTICS_REFRESH_INTERVAL", "3600"))
    start_insight_scheduler(_refresh_all_pinned, interval_seconds=interval)
    
    # 启动预警检测调度器
    alert_enabled = os.getenv("ALERT_SCHEDULER_ENABLED", "true").lower() == "true"
    if alert_enabled:
        init_alert_scheduler(_check_all_alerts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
