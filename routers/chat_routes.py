"""对话与 SQL：generate_sql、run_sql、sessions、history、feedback。"""
import time
from typing import Dict, Any
import pymysql
from fastapi import APIRouter, Body, HTTPException, Depends

from core.auth import get_current_user, require_operator_id
from core.schemas import QuestionRequest, SqlRequest, FeedbackRequest
from services.generate_sql_handler import handle_generate_sql
from services.lark_notification import send_lark_alert
from services import pinned_service as pinned_service_module
from analytics_manager import get_analytics_manager
from routers.deps import get_vn, get_conversation_history, get_run_query_and_chart

router = APIRouter()


@router.post("/generate_sql")
def generate_sql(
    request: QuestionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    vn=Depends(get_vn),
    conversation_history=Depends(get_conversation_history),
    run_query_and_chart=Depends(get_run_query_and_chart),
):
    return handle_generate_sql(vn, conversation_history, run_query_and_chart, request, current_user)


@router.post("/run_sql")
def run_sql(
    request: SqlRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    run_query_and_chart=Depends(get_run_query_and_chart),
):
    try:
        operator_id = require_operator_id(current_user)
        start_time = time.perf_counter()
        df, chart_json = run_query_and_chart(
            request.sql,
            request.question,
            operator_id=operator_id,
            datasource_id=request.datasource_id,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        get_analytics_manager().record_query(
            latency_ms=latency_ms,
            operator_id=operator_id,
            datasource_id=request.datasource_id,
            sql=request.sql,
        )
        return {
            "result": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "chart": chart_json,
        }
    except pymysql.Error as e:
        raise HTTPException(status_code=400, detail=f"数据库错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行失败: {str(e)}")


@router.post("/clear_session")
def clear_session(
    session_id: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = (current_user or {}).get("operator_id", "")
    success = conversation_history.clear_session(session_id, operator_id=operator_id)
    if not success:
        raise HTTPException(status_code=403, detail="无权删除此会话")
    return {"success": True, "message": "会话历史已清除"}


@router.get("/sessions")
def get_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    return conversation_history.get_sessions(operator_id=operator_id, username=username)


@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    history = conversation_history.get_history(
        session_id, operator_id=operator_id, username=username
    )
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    pinned_ids = {item.get("message_id") for item in pinned_items if item.get("message_id")}
    for msg in history:
        msg["pinned"] = msg.get("id") in pinned_ids
    return history


@router.post("/sessions/{session_id}/rename")
def rename_session(
    session_id: str,
    body: Dict = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = (current_user or {}).get("operator_id", "")
    new_title = body.get("title")
    username = (current_user or {}).get("username", "")
    if conversation_history.rename_session(
        session_id, new_title, operator_id=operator_id, username=username
    ):
        return {"success": True, "message": "Session renamed"}
    raise HTTPException(status_code=404, detail="会话不存在或无权操作")


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = (current_user or {}).get("operator_id", "")
    username = (current_user or {}).get("username", "")
    success = conversation_history.clear_session(
        session_id, operator_id=operator_id, username=username
    )
    if not success:
        raise HTTPException(status_code=403, detail="无权删除此会话")
    return {"success": True, "message": "Session deleted"}


@router.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    get_analytics_manager().record_feedback(request.feedback_type, operator_id=operator_id)
    if request.message_id:
        conversation_history.set_message_feedback(
            request.message_id, request.feedback_type, operator_id=operator_id
        )
    if request.feedback_type == "down":
        send_lark_alert(request)
    # 点赞(Helpful)后立即执行一次自动收藏，无需用户再打开看板才触发
    if request.feedback_type == "up":
        try:
            pinned_service_module.auto_pin_recent(operator_id, username, conversation_history)
        except Exception as e:
            # 不因自动收藏失败影响反馈响应
            print(f"[feedback] auto_pin_recent 失败: {e}")
    return {"status": "success", "message": "Feedback received"}
