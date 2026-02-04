"""路由依赖：从 app.state 获取 vn、conversation_history、run_query_and_chart。"""
from fastapi import Request


def get_vn(request: Request):
    return getattr(request.app.state, "vn", None)


def get_conversation_history(request: Request):
    return getattr(request.app.state, "conversation_history", None)


def get_run_query_and_chart(request: Request):
    return getattr(request.app.state, "run_query_and_chart", None)
