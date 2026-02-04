"""
generate_sql 接口的完整业务逻辑，供 routers.chat_routes 调用。
依赖 vn、conversation_history、run_query_and_chart 由调用方注入。
"""
import re
import time
import uuid
from typing import Dict, Any

import pymysql
from fastapi import HTTPException

from core.auth import transform_sql_with_tenant, require_operator_id
from core.schemas import QuestionRequest
from analytics_manager import get_analytics_manager


def handle_generate_sql(
    vn,
    conversation_history,
    run_query_and_chart,
    request: QuestionRequest,
    current_user: Dict[str, Any],
):
    """执行 generate_sql 的完整流程，返回响应字典或抛出 HTTPException。"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        datasource_id = request.datasource_id
        kb_id = request.kb_id

        user_lang = vn.detect_language(request.question)
        intent = vn.detect_intent(request.question)
        operator_id = require_operator_id(current_user)
        username = (current_user or {}).get("username", "")
        history = conversation_history.get_context_history(
            session_id, operator_id=operator_id, username=username
        )

        raw_result = vn.generate_sql_optimized(
            question=request.question,
            allow_llm_to_see_data=True,
            conversation_history=history,
            intent=intent,
            operator_id=operator_id,
        )

        sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        if not sql_match:
            sql_match = re.search(r"```\s*(.*?)\s*```", raw_result, re.DOTALL | re.IGNORECASE)
        if not sql_match:
            sql_keywords_pattern = r'(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s+.*?(?:;|$)'
            potential_sql = re.search(sql_keywords_pattern, raw_result, re.DOTALL | re.IGNORECASE)
            if potential_sql:
                sql_start = potential_sql.start()
                sql_end_match = re.search(r';|\n\n', raw_result[sql_start:])
                sql_end = sql_start + sql_end_match.end() if sql_end_match else len(raw_result)
                extracted_sql = raw_result[sql_start:sql_end].strip()
                lines = extracted_sql.split("\n")
                sql_start_index = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("--") and not stripped.startswith("/*"):
                        sql_start_index = i
                        break
                extracted_sql = "\n".join(lines[sql_start_index:]).strip()

                class MockMatch:
                    def __init__(self, text):
                        self._text = text
                    def group(self, n):
                        return self._text
                sql_match = MockMatch(extracted_sql)
                explanation = raw_result[:sql_start].strip()
            else:
                explanation = ""
        else:
            explanation = ""

        if sql_match:
            clean_sql = sql_match.group(1).strip()
            clean_sql = re.sub(r"^intermediate_sql\s*", "", clean_sql, flags=re.IGNORECASE)
            clean_sql_lines = [
                line for line in clean_sql.split("\n")
                if not line.strip().startswith("--") or " -- " in line
            ]
            clean_sql = "\n".join(clean_sql_lines).strip()

            if not explanation:
                explanation_match = re.search(
                    r"解释[:：]\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE
                )
                if not explanation_match:
                    explanation_match = re.search(
                        r"Explanation:\s*(.*?)(?=```)", raw_result, re.DOTALL | re.IGNORECASE
                    )
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    parts = raw_result.split("```")
                    if parts:
                        explanation = (
                            parts[0]
                            .replace("Explanation:", "")
                            .replace("解释:", "")
                            .replace("解释：", "")
                            .strip()
                        )
            if not explanation:
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == "zh" else "Here is the SQL query for your request."
            if "SELECT" in explanation.upper() or "FROM" in explanation.upper():
                explanation = "根据您的问题，我生成了以下SQL查询。" if user_lang == "zh" else "Here is the SQL query for your request."

            if intent.get("intent") == "trend" and not vn.validate_trend_sql(clean_sql):
                retry_prompt = request.question + "（请严格输出趋势类SQL：时间维度+聚合+按时间分组排序）"
                raw_retry = vn.generate_sql_optimized(
                    question=retry_prompt,
                    allow_llm_to_see_data=True,
                    conversation_history=history,
                    intent=intent,
                    operator_id=operator_id,
                )
                retry_match = re.search(r"```sql\s*(.*?)\s*```", raw_retry, re.DOTALL | re.IGNORECASE)
                if retry_match:
                    retry_sql = retry_match.group(1).strip()
                    retry_sql = re.sub(r"^intermediate_sql\s*", "", retry_sql, flags=re.IGNORECASE).strip()
                    if vn.validate_trend_sql(retry_sql):
                        clean_sql = retry_sql

            conversation_history.add_message(
                session_id, "user", request.question,
                operator_id=operator_id, username=username,
            )

            try:
                clean_sql = transform_sql_with_tenant(clean_sql, operator_id)
                start_time = time.perf_counter()
                df, chart_json = run_query_and_chart(
                    clean_sql, request.question,
                    operator_id=operator_id, datasource_id=datasource_id,
                )
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                get_analytics_manager().record_query(
                    latency_ms=latency_ms,
                    operator_id=operator_id,
                    datasource_id=datasource_id,
                    sql=clean_sql,
                )
            except pymysql.Error as e:
                raise HTTPException(status_code=400, detail=f"数据库错误: {str(e)}")

            result_rows = df.to_dict(orient="records")
            columns = df.columns.tolist()
            # 根据结果列名+问题覆盖意图，推荐饼图及用列（如 status_name + percentage）
            intent = vn.adjust_intent_for_result(request.question, df, columns, intent)
            assistant_msg_id = conversation_history.add_message(
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
                    "datasource_id": datasource_id,
                },
                operator_id=operator_id,
                username=username,
            )
            return {
                "sql": clean_sql,
                "is_sql": True,
                "explanation": explanation,
                "session_id": session_id,
                "message_id": assistant_msg_id,
                "result": result_rows,
                "columns": columns,
                "chart": chart_json,
                "intent": intent,
            }
        else:
            conversation_history.add_message(
                session_id, "user", request.question,
                operator_id=operator_id, username=username,
            )
            conversation_history.add_message(
                session_id, "assistant", raw_result,
                operator_id=operator_id, username=username,
            )
            return {
                "text": raw_result,
                "is_sql": False,
                "explanation": raw_result,
                "session_id": session_id,
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成 SQL 失败: {str(e)}")
