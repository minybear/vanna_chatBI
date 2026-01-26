"""
知识库管理 API 路由
提供知识库的CRUD、训练数据管理、文件上传和异步训练等接口
"""
import os
import shutil
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from knowledge_base_manager import get_kb_manager, KnowledgeBase, TrainingTask
from training_worker import get_training_worker

router = APIRouter()

# 延迟导入 vn，避免循环导入
_vn = None

def get_vn():
    """延迟获取 vn 实例，避免循环导入"""
    global _vn
    if _vn is None:
        from app import vn
        _vn = vn
    return _vn


# ==================== 原有接口（向后兼容，操作默认知识库）====================

class TrainingDataRequest(BaseModel):
    id: Optional[str] = None
    type: str  # 'ddl', 'documentation', 'sql'
    content: str
    question: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    ids: List[str]


@router.get("/get_training_data")
def get_training_data():
    """获取默认知识库的训练数据（向后兼容）"""
    vn = get_vn()
    df = vn.get_training_data()
    if df is None or df.empty:
        return []
    return df.to_dict(orient='records')


@router.post("/training_data")
def add_training_data(data: TrainingDataRequest):
    """添加训练数据到默认知识库（向后兼容）"""
    try:
        vn = get_vn()
        if data.type == 'ddl':
            vn.train(ddl=data.content)
        elif data.type == 'documentation':
            vn.train(documentation=data.content)
        elif data.type == 'sql':
            if not data.question:
                raise HTTPException(status_code=400, detail="SQL training requires a question")
            vn.train(question=data.question, sql=data.content)
        else:
            raise HTTPException(status_code=400, detail="Invalid training type")
        return {"status": "success", "message": "Training data added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/training_data/{id}")
def delete_training_data(id: str):
    """从默认知识库删除训练数据（向后兼容）"""
    try:
        vn = get_vn()
        success = vn.remove_training_data(id=id)
        if success:
            return {"status": "success", "message": "Training data removed"}
        else:
            raise HTTPException(status_code=404, detail="Training data not found or failed to remove")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training_data/batch_delete")
def batch_delete_training_data(request: BatchDeleteRequest):
    """批量删除默认知识库的训练数据（向后兼容）"""
    try:
        vn = get_vn()
        deleted_count = 0
        errors = []
        for id in request.ids:
            try:
                success = vn.remove_training_data(id=id)
                if success:
                    deleted_count += 1
                else:
                    errors.append(f"Failed to remove {id}")
            except Exception as e:
                errors.append(f"Error removing {id}: {str(e)}")

        return {
            "status": "success",
            "message": f"Deleted {deleted_count} items",
            "deleted_count": deleted_count,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 多知识库管理接口 ====================

class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., description="知识库名称")
    description: str = Field(default="", description="知识库描述")
    datasource_ids: List[str] = Field(default=[], description="关联的数据源ID列表")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    datasource_ids: Optional[List[str]] = None


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    id: str
    name: str
    description: str
    datasource_ids: List[str]
    collection_prefix: str
    status: str
    ddl_count: int
    sql_count: int
    doc_count: int
    created_at: str
    updated_at: str


class KBTrainingDataRequest(BaseModel):
    """知识库训练数据请求"""
    type: str  # 'ddl', 'documentation', 'sql'
    content: str
    question: Optional[str] = None


class TrainingTaskResponse(BaseModel):
    """训练任务响应"""
    id: str
    kb_id: str
    file_name: str
    total_records: int
    processed_records: int
    success_count: int
    fail_count: int
    status: str
    error_message: Optional[str]
    created_at: str
    updated_at: str


def _kb_to_response(kb: KnowledgeBase) -> Dict[str, Any]:
    """将KnowledgeBase转换为响应字典"""
    return kb.to_dict()


def _task_to_response(task: TrainingTask) -> Dict[str, Any]:
    """将TrainingTask转换为响应字典"""
    return task.to_dict()


@router.get("/knowledge_bases", response_model=List[KnowledgeBaseResponse])
def list_knowledge_bases():
    """获取所有知识库列表"""
    try:
        manager = get_kb_manager()
        kbs = manager.list_all()
        return [_kb_to_response(kb) for kb in kbs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge_bases/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(kb_id: str):
    """获取单个知识库详情"""
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return _kb_to_response(kb)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge_bases", response_model=KnowledgeBaseResponse)
def create_knowledge_base(data: KnowledgeBaseCreate):
    """创建新知识库"""
    try:
        manager = get_kb_manager()
        kb = manager.create(
            name=data.name,
            description=data.description,
            datasource_ids=data.datasource_ids
        )
        return _kb_to_response(kb)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge_bases/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(kb_id: str, data: KnowledgeBaseUpdate):
    """更新知识库"""
    try:
        manager = get_kb_manager()

        # 构建更新字典（只包含非None的字段）
        update_data = {}
        if data.name is not None:
            update_data['name'] = data.name
        if data.description is not None:
            update_data['description'] = data.description
        if data.datasource_ids is not None:
            update_data['datasource_ids'] = data.datasource_ids

        kb = manager.update(kb_id, **update_data)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return _kb_to_response(kb)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge_bases/{kb_id}")
def delete_knowledge_base(kb_id: str):
    """删除知识库"""
    try:
        manager = get_kb_manager()
        success = manager.delete(kb_id)
        if not success:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return {"status": "success", "message": "知识库已删除"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 知识库训练数据管理 ====================

@router.get("/knowledge_bases/{kb_id}/training_data")
def get_kb_training_data(kb_id: str):
    """获取指定知识库的训练数据"""
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        df = manager.get_training_data(kb_id)
        if df is None or df.empty:
            return []
        return df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge_bases/{kb_id}/training_data")
def add_kb_training_data(kb_id: str, data: KBTrainingDataRequest):
    """添加训练数据到指定知识库"""
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if data.type == 'sql' and not data.question:
            raise HTTPException(status_code=400, detail="SQL类型必须提供question")

        data_id = manager.add_training_data(
            kb_id=kb_id,
            data_type=data.type,
            content=data.content,
            question=data.question
        )
        return {"status": "success", "message": "Training data added", "id": data_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge_bases/{kb_id}/training_data/{data_id}")
def delete_kb_training_data(kb_id: str, data_id: str):
    """从指定知识库删除训练数据"""
    try:
        manager = get_kb_manager()
        success = manager.remove_training_data(kb_id, data_id)
        if success:
            return {"status": "success", "message": "Training data removed"}
        else:
            raise HTTPException(status_code=404, detail="Training data not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文件上传和异步训练 ====================

# 创建上传目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/knowledge_bases/{kb_id}/upload")
async def upload_training_file(kb_id: str, file: UploadFile = File(...)):
    """
    上传训练语料文件到指定知识库
    支持 CSV 格式，列名：类型/type, 内容/content, 用户问题示例/question
    """
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 验证文件类型
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="只支持CSV文件")

        # 保存文件
        file_path = os.path.join(UPLOAD_DIR, f"{kb_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "message": "文件上传成功",
            "file_path": file_path,
            "file_name": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge_bases/{kb_id}/train", response_model=TrainingTaskResponse)
async def start_training(kb_id: str, file_path: str = None, background_tasks: BackgroundTasks = None):
    """
    启动异步训练任务
    可以指定file_path（上传后返回的路径），或者使用最近上传的文件
    """
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 如果没有指定文件路径，查找最近上传的文件
        if not file_path:
            files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(f"{kb_id}_")]
            if not files:
                raise HTTPException(status_code=400, detail="没有找到上传的训练文件")
            # 按修改时间排序，取最新的
            files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x)), reverse=True)
            file_path = os.path.join(UPLOAD_DIR, files[0])

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="训练文件不存在")

        # 启动异步训练
        worker = get_training_worker(manager)
        task = worker.start_training(kb_id, file_path)

        return _task_to_response(task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge_bases/{kb_id}/training_status", response_model=List[TrainingTaskResponse])
def get_training_status(kb_id: str):
    """获取知识库的所有训练任务状态"""
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        tasks = manager.get_tasks_by_kb(kb_id)
        return [_task_to_response(task) for task in tasks]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training_tasks/{task_id}", response_model=TrainingTaskResponse)
def get_training_task(task_id: str):
    """获取单个训练任务状态"""
    try:
        manager = get_kb_manager()
        task = manager.get_training_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        return _task_to_response(task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training_tasks/{task_id}/cancel")
def cancel_training_task(task_id: str):
    """取消训练任务"""
    try:
        manager = get_kb_manager()
        worker = get_training_worker(manager)

        success = worker.cancel_training(task_id)
        if success:
            return {"status": "success", "message": "训练任务已取消"}
        else:
            return {"status": "info", "message": "任务不在运行中或已完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 批量导入语料 ====================

class CorpusItem(BaseModel):
    """语料项"""
    id: Optional[str] = None
    type: str  # ddl/sql/documentation
    content: str
    question: Optional[str] = None
    tags: Optional[str] = None


class ImportCorpusRequest(BaseModel):
    """批量导入语料请求"""
    corpus: List[CorpusItem]


class ImportCorpusResponse(BaseModel):
    """批量导入语料响应"""
    status: str
    imported_count: int
    success_count: int
    fail_count: int
    errors: List[Dict[str, Any]] = []


@router.post("/knowledge_bases/{kb_id}/import_corpus", response_model=ImportCorpusResponse)
def import_corpus(kb_id: str, request: ImportCorpusRequest):
    """
    批量导入语料到知识库
    
    支持在线生成的语料直接导入，无需下载CSV文件
    """
    try:
        manager = get_kb_manager()
        kb = manager.get(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        imported_count = len(request.corpus)
        success_count = 0
        fail_count = 0
        errors = []
        
        for idx, item in enumerate(request.corpus):
            try:
                # 验证数据
                if not item.content:
                    errors.append({"index": idx, "error": "内容不能为空"})
                    fail_count += 1
                    continue
                
                if item.type == 'sql' and not item.question:
                    errors.append({"index": idx, "error": "SQL类型必须提供question"})
                    fail_count += 1
                    continue
                
                # 添加训练数据
                manager.add_training_data(
                    kb_id=kb_id,
                    data_type=item.type,
                    content=item.content,
                    question=item.question
                )
                success_count += 1
                
            except Exception as e:
                errors.append({"index": idx, "error": str(e)})
                fail_count += 1
        
        return ImportCorpusResponse(
            status="success",
            imported_count=imported_count,
            success_count=success_count,
            fail_count=fail_count,
            errors=errors
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
