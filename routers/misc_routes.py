"""根路径、UI 页面、generate_questions。"""
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from core.auth import get_current_user

router = APIRouter()


@router.get("/", include_in_schema=False)
def root():
    return {"message": "redtea chatBi MVP is running. Use /api/v0/ for API endpoints."}


@router.get("/ui", response_class=FileResponse, include_in_schema=False)
def ui():
    return FileResponse("templates/ui.html")


# 以下挂载到 prefix=/api/v0 下
api_router = APIRouter()


@api_router.post("/generate_questions")
def generate_questions(current_user: dict = Depends(get_current_user)):
    return ["How many users are there?", "Show me the latest orders"]
