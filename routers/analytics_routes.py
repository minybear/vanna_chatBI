"""分析看板、洞察、用户设置、收藏图表、标签、预警喜报、飞书群。"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Body, HTTPException, Depends

from core.auth import get_current_user, require_operator_id
from core.schemas import (
    ChartLabel, LabelUpdateRequest, ThresholdUpdateRequest,
    LarkGroupCreateRequest, SendReportRequest,
)
from routers.deps import get_conversation_history, get_run_query_and_chart
from analytics_manager import get_analytics_manager
from insight_engine import get_insight_engine
from services import pinned_service as pinned_service_module
from services.label_service import get_label_recommender, recommend_chart_label
from services.alert_celebration_engine import get_alert_engine, check_pinned_alerts
from services.lark_group_manager import get_lark_group_manager, get_lark_report_sender

router = APIRouter()


def _refresh_pinned_item(item: Dict, operator_id: str, conversation_history, run_query_and_chart):
    return pinned_service_module.refresh_pinned_item(
        item, operator_id, conversation_history, run_query_and_chart, get_insight_engine
    )


@router.get("/analytics/metrics")
def get_analytics_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = require_operator_id(current_user)
    return get_analytics_manager().get_metrics(operator_id=operator_id).to_dict()


@router.get("/analytics/insights")
def get_analytics_insights(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    user_settings = conversation_history.get_user_settings(
        operator_id=operator_id, username=username
    )
    if not user_settings.get("insight_enabled", True):
        return {"items": [], "disabled": True}
    items = get_insight_engine().list_insights(operator_id=operator_id, limit=limit)
    return {"items": items}


@router.post("/analytics/insights/clear")
def clear_analytics_insights(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = require_operator_id(current_user)
    cleared = get_insight_engine().clear_insights(operator_id=operator_id)
    return {"success": True, "cleared": cleared}


@router.delete("/analytics/insights/{insight_id}")
def delete_analytics_insight(
    insight_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = require_operator_id(current_user)
    success = get_insight_engine().delete_insight(insight_id, operator_id=operator_id)
    if not success:
        raise HTTPException(status_code=404, detail="洞察不存在或无权删除")
    return {"success": True}


@router.post("/analytics/insight-report")
def generate_insight_report(
    request_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    operator_id = require_operator_id(current_user)
    dashboard_data = request_data.get("dashboard_data", [])
    if not dashboard_data:
        return {"success": False, "error": "没有看板数据"}
    api_key = os.getenv("ZHIPU_API_KEY")
    model = os.getenv("ZHIPU_YULIAO_MODEL", "GLM-4.5")
    if not api_key:
        return {"success": False, "error": "AI服务未配置，请联系管理员"}
    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key)
        data_summary = []
        for i, item in enumerate(dashboard_data[:6], 1):
            question = item.get("question", f"图表{i}")
            columns = item.get("columns", [])
            result = item.get("result", [])
            summary = f"### 图表{i}: {question}\n"
            summary += f"- 数据列: {', '.join(columns[:5])}\n"
            summary += f"- 数据行数: {len(result)}\n"
            if result and columns:
                for col in columns[:3]:
                    values = [
                        row.get(col) for row in result
                        if isinstance(row.get(col), (int, float))
                    ]
                    if values:
                        summary += f"- {col}: 最小={min(values):.2f}, 最大={max(values):.2f}, 平均={sum(values)/len(values):.2f}\n"
                if result:
                    summary += f"- 最新数据: {result[-1]}\n"
            data_summary.append(summary)
        report_date = datetime.now().strftime("%Y年%m月%d日")
        # 明确每个图表的标题，用于要求 AI 在核心指标分析中按此格式输出小节
        chart_headings = [
            f"#### 图表{i}: {item.get('question', f'图表{i}')}"
            for i, item in enumerate(dashboard_data[:6], 1)
        ]
        chart_headings_text = "\n".join(chart_headings)
        prompt = f"""你是一位资深的数据分析师，请根据以下看板数据生成一份专业的AI洞察报告。

## 看板数据摘要

{chr(10).join(data_summary)}

## 报告要求

请生成一份结构清晰、内容专业的中文洞察报告，包含以下部分：

1. **报告概述** - 简要说明本次分析的数据范围和时间
2. **核心指标分析** - 对每个图表的关键指标进行深入分析。**必须**为每个图表单独设一个小节，小节标题**必须**使用以下格式（不可改写法）：
{chart_headings_text}
即：每个小节以四级标题 "#### 图表N: 对应图表名称" 开头，N 为 1、2、3…，名称与上面看板数据摘要中的「图表N: xxx」一致，然后在下方写该图表的分析内容。
3. **趋势洞察** - 识别数据中的趋势、模式和异常
4. **业务建议** - 基于数据分析给出可操作的业务建议
5. **风险提示** - 指出需要关注的潜在风险点
6. **总结** - 简要总结关键发现

报告格式要求：
- 使用Markdown格式
- 语言简洁专业
- 数据引用准确
- 建议具体可行
- 核心指标分析部分必须包含上述 {len(dashboard_data[:6])} 个小节标题，以便图表能正确对应到分析内容下

报告日期: {report_date}
"""
        response = client.chat.completions.create(
            model=model,
            max_tokens=3000,
            temperature=0.5,
            top_p=0.8,
            messages=[{"role": "user", "content": prompt}],
        )
        report_content = response.choices[0].message.content if response.choices else ""
        if not report_content:
            return {"success": False, "error": "AI生成报告失败"}
        final_report = f"""# AI 数据洞察报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**数据来源**: ChatBI 看板  
**分析图表数**: {len(dashboard_data)}

---

{report_content}

---

*本报告由 RedTea ChatBI AI 自动生成，仅供参考。*
"""
        return {
            "success": True,
            "report": final_report,
            "title": f"AI洞察报告_{datetime.now().strftime('%Y%m%d')}",
        }
    except ImportError:
        return {"success": False, "error": "AI SDK未安装，请联系管理员"}
    except Exception as exc:
        print(f"[ERROR] 生成AI洞察报告失败: {exc}")
        return {"success": False, "error": f"报告生成失败: {str(exc)}"}


@router.get("/settings")
def get_settings(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    settings = conversation_history.get_user_settings(
        operator_id=operator_id, username=username
    )
    return {"settings": settings}


@router.put("/settings")
def update_settings(
    settings_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    settings = conversation_history.save_user_settings(
        settings_data, operator_id=operator_id, username=username
    )
    return {"success": True, "settings": settings}


@router.post("/settings/reset")
def reset_settings(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    default_settings = conversation_history.get_default_settings()
    settings = conversation_history.save_user_settings(
        default_settings, operator_id=operator_id, username=username
    )
    return {"success": True, "settings": settings}


@router.get("/analytics/pinned")
def get_pinned_charts(
    refresh: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
    run_query_and_chart=Depends(get_run_query_and_chart),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    user_settings = conversation_history.get_user_settings(
        operator_id=operator_id, username=username
    )
    try:
        pinned_service_module.auto_pin_recent(operator_id, username, conversation_history)
    except Exception as e:
        print(f"[ERROR get_pinned_charts] auto_pin_recent 失败: {e}")
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    if refresh:
        refreshed_items = []
        for item in pinned_items:
            try:
                refreshed_items.append(
                    _refresh_pinned_item(item, operator_id, conversation_history, run_query_and_chart)
                )
            except Exception as e:
                print(f"[ERROR get_pinned_charts] refresh_pinned_item 失败: {e}")
                refreshed_items.append(item)
        pinned_items = refreshed_items
    if user_settings.get("dedup_enabled", True):
        try:
            dedup_threshold = user_settings.get("dedup_threshold", 0.70)
            dedup_mode = user_settings.get("dedup_mode", "rule").lower()
            pinned_items = pinned_service_module._dedupe_pinned_items(
                pinned_items, dedup_threshold, dedup_mode
            )
        except Exception as e:
            print(f"[ERROR get_pinned_charts] _dedupe_pinned_items 失败: {e}")
    return {"items": pinned_items}


@router.post("/messages/{message_id}/pin")
def pin_message(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    try:
        payload = conversation_history.pin_message(
            message_id, operator_id=operator_id, username=username
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Pin 到看板后智能选择一个标签（用于洞察时的预警/喜报区分）
    try:
        label = recommend_chart_label(
            question=payload.get("question") or "",
            sql=payload.get("sql") or "",
            columns=payload.get("columns"),
            result=payload.get("result"),
        )
        if label:
            label_dict = label.model_dump()
            label_dict["updated_at"] = datetime.now().isoformat()
            payload["label"] = label_dict
            conversation_history.update_pinned(
                message_id, payload, operator_id=operator_id, username=username
            )
    except Exception as e:
        print(f"[pin_message] 智能标签推荐失败: {e}")

    # 用最新覆盖原先：若开启去重，则移除与当前 pin 同类型的旧 pin，避免看板出现「叠图层」需删两次
    try:
        user_settings = conversation_history.get_user_settings(
            operator_id=operator_id, username=username
        )
        if user_settings.get("dedup_enabled", True):
            dedup_threshold = user_settings.get("dedup_threshold", 0.70)
            dedup_mode = (user_settings.get("dedup_mode") or "rule").lower()
            pinned_items = conversation_history.list_pinned(
                operator_id=operator_id, username=username
            )
            to_replace = pinned_service_module.get_same_type_message_ids_to_replace(
                pinned_items, message_id, dedup_threshold, dedup_mode
            )
            for old_id in to_replace:
                if old_id:
                    conversation_history.unpin_message(
                        old_id, operator_id=operator_id, username=username, add_to_blacklist=False
                    )
    except Exception as e:
        print(f"[pin_message] 同类型覆盖去重失败: {e}")

    return {
        "success": True,
        "item": payload,
        "items": conversation_history.list_pinned(
            operator_id=operator_id, username=username
        ),
    }


@router.delete("/messages/{message_id}/pin")
def unpin_message(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    success = conversation_history.unpin_message(
        message_id, operator_id=operator_id, username=username
    )
    return {
        "success": success,
        "items": conversation_history.list_pinned(
            operator_id=operator_id, username=username
        ),
    }


@router.delete("/analytics/pinned")
def clear_user_pins(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    cleared_count = 0
    for item in pinned_items:
        msg_id = item.get("message_id")
        if msg_id:
            conversation_history.unpin_message(
                msg_id, operator_id=operator_id, username=username
            )
            cleared_count += 1
    return {
        "success": True,
        "cleared": cleared_count,
        "items": conversation_history.list_pinned(
            operator_id=operator_id, username=username
        ),
    }


@router.post("/analytics/pinned/{message_id}/refresh")
def refresh_pinned_chart(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
    run_query_and_chart=Depends(get_run_query_and_chart),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    target = next(
        (item for item in pinned_items if item.get("message_id") == message_id),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")
    refreshed = _refresh_pinned_item(
        target, operator_id, conversation_history, run_query_and_chart
    )
    get_analytics_manager().record_dashboard_refresh(operator_id=operator_id)
    return {"success": True, "item": refreshed}


@router.post("/analytics/refresh")
def refresh_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
    run_query_and_chart=Depends(get_run_query_and_chart),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    pinned_service_module.auto_pin_recent(operator_id, username, conversation_history)
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    for item in pinned_items:
        _refresh_pinned_item(item, operator_id, conversation_history, run_query_and_chart)
    get_analytics_manager().record_dashboard_refresh(operator_id=operator_id)
    return {"success": True}


@router.post("/analytics/clear")
def clear_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    cleared_pins = conversation_history.clear_pins(
        operator_id=operator_id, username=username
    )
    cleared_insights = get_insight_engine().clear_insights(operator_id=operator_id)
    cleared_events = get_analytics_manager().clear_events(operator_id=operator_id)
    return {
        "success": True,
        "cleared": {
            "pins": cleared_pins,
            "insights": cleared_insights,
            "events": cleared_events,
        },
    }


@router.delete("/analytics/pinned/all")
def clear_all_pins(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    require_operator_id(current_user)
    try:
        all_data = conversation_history.pins.get()
    except Exception as e:
        if "Chroma" in type(e).__name__ or "chroma" in str(type(e)).lower():
            print(f"[ERROR clear_all_pins] ChromaDB 内部错误: {e}")
        return {
            "success": False,
            "cleared": 0,
            "message": "ChromaDB 暂时不可用，请稍后重试或检查 chroma_db 数据目录",
        }
    all_ids = all_data.get("ids") if all_data else []
    if not all_ids:
        return {"success": True, "cleared": 0, "message": "没有数据需要清理"}
    try:
        conversation_history.pins.delete(ids=all_ids)
    except Exception as e:
        if "Chroma" in type(e).__name__ or "chroma" in str(type(e)).lower():
            print(f"[ERROR clear_all_pins] ChromaDB delete 失败: {e}")
        return {"success": False, "cleared": 0, "message": "删除失败，ChromaDB 可能损坏"}
    return {
        "success": True,
        "cleared": len(all_ids),
        "message": f"已清理 {len(all_ids)} 条 pinned 数据",
    }


# ============ 图表标签 API ============

@router.get("/analytics/labels/categories")
def get_label_categories():
    """获取标签分类体系"""
    return {
        "alert": {
            "name": "预警类",
            "description": "触发智能预警的图表类型",
            "categories": [
                {"id": "inventory", "name": "库存预警", "description": "库存变化、缺货、滞销等"},
                {"id": "cost", "name": "成本预警", "description": "费用异常、超支、亏损等"},
                {"id": "risk", "name": "风险预警", "description": "业务风险、异常指标等"},
            ],
        },
        "celebration": {
            "name": "喜报类",
            "description": "触发智能喜报的图表类型",
            "categories": [
                {"id": "gmv", "name": "销售额", "description": "GMV、营收、订单金额等"},
                {"id": "growth", "name": "业务增长", "description": "增长率、转化率提升等"},
                {"id": "performance", "name": "业绩达成", "description": "KPI完成、目标突破等"},
            ],
        },
    }


@router.post("/analytics/pinned/{message_id}/label/recommend")
def recommend_label_for_pinned(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    """为收藏图表推荐标签"""
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    target = next(
        (item for item in pinned_items if item.get("message_id") == message_id),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")

    label = recommend_chart_label(
        question=target.get("question", ""),
        sql=target.get("sql", ""),
        columns=target.get("columns", []),
        result=target.get("result", []),
    )

    if label:
        return {"success": True, "label": label.model_dump()}
    return {"success": False, "error": "无法推荐标签，请手动设置"}


@router.put("/analytics/pinned/{message_id}/label")
def update_pinned_label(
    message_id: str,
    label_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    """更新收藏图表的标签"""
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    target = next(
        (item for item in pinned_items if item.get("message_id") == message_id),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")

    # 验证标签数据
    try:
        label = ChartLabel(**label_data)
        label.ai_suggested = False  # 用户手动设置
        label.updated_at = datetime.now().isoformat()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"标签数据格式错误: {e}")

    # 更新 pinned item
    target["label"] = label.model_dump()
    conversation_history.update_pinned(
        message_id, target, operator_id=operator_id, username=username
    )

    return {"success": True, "item": target}


@router.delete("/analytics/pinned/{message_id}/label")
def delete_pinned_label(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    """删除收藏图表的标签"""
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    target = next(
        (item for item in pinned_items if item.get("message_id") == message_id),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")

    if "label" in target:
        del target["label"]
        conversation_history.update_pinned(
            message_id, target, operator_id=operator_id, username=username
        )

    return {"success": True, "item": target}


# ============ 预警与喜报 API ============

@router.post("/analytics/alerts/check")
def check_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    """检测所有收藏图表的预警和喜报"""
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )

    result = check_pinned_alerts(pinned_items)
    
    return {
        "success": True,
        "alerts": [a.model_dump() for a in result["alerts"]],
        "celebrations": [c.model_dump() for c in result["celebrations"]],
        "checked_count": len(pinned_items),
    }


@router.post("/analytics/pinned/{message_id}/check-alert")
def check_single_alert(
    message_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_history=Depends(get_conversation_history),
):
    """检测单个图表的预警/喜报"""
    operator_id = require_operator_id(current_user)
    username = (current_user or {}).get("username", "")
    
    pinned_items = conversation_history.list_pinned(
        operator_id=operator_id, username=username
    )
    target = next(
        (item for item in pinned_items if item.get("message_id") == message_id),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="收藏图表不存在")

    engine = get_alert_engine()
    alert = engine.check_alert(target)
    celebration = engine.check_celebration(target)

    return {
        "success": True,
        "alert": alert.model_dump() if alert.triggered else None,
        "celebration": celebration.model_dump() if celebration.triggered else None,
    }


# ============ 飞书群管理 API ============

@router.get("/lark/groups")
def list_lark_groups(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取飞书群列表"""
    operator_id = require_operator_id(current_user)
    groups = get_lark_group_manager().list_groups(operator_id)
    return {"success": True, "groups": groups}


@router.post("/lark/groups")
def create_lark_group(
    request: LarkGroupCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """创建飞书群配置"""
    operator_id = require_operator_id(current_user)
    group = get_lark_group_manager().add_group(
        name=request.name,
        webhook_url=request.webhook_url,
        description=request.description or "",
        group_type=request.group_type,
        operator_id=operator_id,
    )
    return {"success": True, "group": group}


@router.put("/lark/groups/{group_id}")
def update_lark_group(
    group_id: str,
    update_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """更新飞书群配置"""
    operator_id = require_operator_id(current_user)
    group = get_lark_group_manager().update_group(
        group_id,
        name=update_data.get("name"),
        webhook_url=update_data.get("webhook_url"),
        description=update_data.get("description"),
        group_type=update_data.get("group_type"),
        operator_id=operator_id,
    )
    if not group:
        raise HTTPException(status_code=404, detail="飞书群配置不存在或无权修改")
    return {"success": True, "group": group}


@router.delete("/lark/groups/{group_id}")
def delete_lark_group(
    group_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """删除飞书群配置"""
    operator_id = require_operator_id(current_user)
    success = get_lark_group_manager().delete_group(group_id, operator_id)
    if not success:
        raise HTTPException(status_code=404, detail="飞书群配置不存在或无权删除")
    return {"success": True}


@router.post("/lark/send-report")
def send_report_to_lark(
    request: SendReportRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """发送洞察报告到飞书群"""
    operator_id = require_operator_id(current_user)
    result = get_lark_report_sender().send_insight_report(
        group_id=request.group_id,
        report_content=request.report_content,
        chart_images=request.chart_images,
        report_type=request.report_type,
        operator_id=operator_id,
    )
    return result


@router.post("/lark/send-alert")
def send_alert_to_lark(
    alert_data: Dict[str, Any] = Body(...),
    group_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """发送预警通知到飞书群"""
    from core.schemas import AlertResult, AlertSeverity
    
    operator_id = require_operator_id(current_user)
    
    try:
        # 构建 AlertResult
        alert = AlertResult(
            triggered=True,
            severity=AlertSeverity(alert_data.get("severity", "warning")),
            title=alert_data.get("title", "数据预警"),
            message=alert_data.get("message", ""),
            metrics=alert_data.get("metrics"),
            suggestion=alert_data.get("suggestion"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预警数据格式错误: {e}")

    result = get_lark_report_sender().send_alert(
        group_id=group_id or "default",
        alert=alert,
        operator_id=operator_id,
    )
    return result


@router.post("/lark/send-celebration")
def send_celebration_to_lark(
    celebration_data: Dict[str, Any] = Body(...),
    group_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """发送喜报通知到飞书群"""
    from core.schemas import CelebrationResult
    
    operator_id = require_operator_id(current_user)
    
    try:
        celebration = CelebrationResult(
            triggered=True,
            title=celebration_data.get("title", "业务喜报"),
            achievement=celebration_data.get("achievement", ""),
            highlights=celebration_data.get("highlights"),
            metrics=celebration_data.get("metrics"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"喜报数据格式错误: {e}")

    result = get_lark_report_sender().send_celebration(
        group_id=group_id or "default",
        celebration=celebration,
        operator_id=operator_id,
    )
    return result
