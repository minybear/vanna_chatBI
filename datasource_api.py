"""
数据源管理 API 路由
提供数据源的CRUD、测试连接、激活等接口
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from datasource_manager import get_datasource_manager, DataSource
from corpus_generator import CorpusGenerator

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


class DataSourceCreate(BaseModel):
    """创建数据源请求"""
    name: str = Field(..., max_length=20, description="数据源名称（20字以内）")
    host: str = Field(..., description="数据库主机")
    port: int = Field(default=3306, description="数据库端口")
    user: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    databases: List[str] = Field(default=[], description="选中的数据库列表")
    default_database: Optional[str] = Field(None, description="默认数据库")
    ssl_enabled: bool = Field(default=True, description="是否启用SSL")


class DataSourceUpdate(BaseModel):
    """更新数据源请求"""
    name: Optional[str] = Field(None, max_length=20)
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    databases: Optional[List[str]] = None
    default_database: Optional[str] = None
    ssl_enabled: Optional[bool] = None


class DataSourceResponse(BaseModel):
    """数据源响应"""
    id: str
    name: str
    host: str
    port: int
    user: str
    password: str = "******"
    databases: List[str] = []
    default_database: Optional[str] = None
    database: Optional[str] = None  # 向后兼容
    ssl_enabled: bool
    status: str
    created_at: str
    updated_at: str


class TestConnectionResponse(BaseModel):
    """测试连接响应"""
    success: bool
    message: str
    databases: List[str]


def _ds_to_response(ds: DataSource) -> Dict[str, Any]:
    """将DataSource转换为响应字典"""
    return ds.to_dict(include_password=False)


@router.get("/datasources", response_model=List[DataSourceResponse])
def list_datasources():
    """获取所有数据源列表"""
    try:
        manager = get_datasource_manager()
        datasources = manager.list_all()
        return [_ds_to_response(ds) for ds in datasources]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasources/{ds_id}", response_model=DataSourceResponse)
def get_datasource(ds_id: str):
    """获取单个数据源详情"""
    try:
        manager = get_datasource_manager()
        ds = manager.get(ds_id)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        return _ds_to_response(ds)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasources", response_model=DataSourceResponse)
def create_datasource(data: DataSourceCreate):
    """创建新数据源"""
    try:
        manager = get_datasource_manager()
        ds = manager.create(
            name=data.name,
            host=data.host,
            port=data.port,
            user=data.user,
            password=data.password,
            databases=data.databases,
            default_database=data.default_database,
            ssl_enabled=data.ssl_enabled
        )
        return _ds_to_response(ds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/datasources/{ds_id}", response_model=DataSourceResponse)
def update_datasource(ds_id: str, data: DataSourceUpdate):
    """更新数据源"""
    try:
        manager = get_datasource_manager()
        
        # 构建更新字典（只包含非None的字段）
        update_data = {}
        if data.name is not None:
            update_data['name'] = data.name
        if data.host is not None:
            update_data['host'] = data.host
        if data.port is not None:
            update_data['port'] = data.port
        if data.user is not None:
            update_data['user'] = data.user
        if data.password is not None:
            update_data['password'] = data.password
        if data.databases is not None:
            update_data['databases'] = data.databases
        if data.default_database is not None:
            update_data['default_database'] = data.default_database
        if data.ssl_enabled is not None:
            update_data['ssl_enabled'] = data.ssl_enabled
        
        ds = manager.update(ds_id, **update_data)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        return _ds_to_response(ds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasources/{ds_id}")
def delete_datasource(ds_id: str):
    """删除数据源"""
    try:
        manager = get_datasource_manager()
        success = manager.delete(ds_id)
        if not success:
            raise HTTPException(status_code=404, detail="数据源不存在")
        return {"status": "success", "message": "数据源已删除"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasources/{ds_id}/test", response_model=TestConnectionResponse)
def test_connection(ds_id: str):
    """测试数据源连接"""
    try:
        manager = get_datasource_manager()
        result = manager.test_connection(ds_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasources/{ds_id}/activate")
def activate_datasource(ds_id: str):
    """激活数据源（设为当前使用的数据源）"""
    try:
        manager = get_datasource_manager()
        success = manager.activate(ds_id)
        if not success:
            raise HTTPException(status_code=400, detail="激活失败，请检查数据源连接")
        return {"status": "success", "message": "数据源已激活"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasources/active/current")
def get_active_datasource():
    """获取当前激活的数据源"""
    try:
        manager = get_datasource_manager()
        ds = manager.get_active_datasource()
        if not ds:
            return {"id": None, "name": None, "message": "没有激活的数据源"}
        return _ds_to_response(ds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasources/{ds_id}/tables")
def get_datasource_tables(ds_id: str, database: Optional[str] = None):
    """
    获取数据源下的表列表
    
    Args:
        ds_id: 数据源ID
        database: 指定数据库名，如果不提供则返回所有选中数据库的表
    """
    try:
        manager = get_datasource_manager()
        ds = manager.get(ds_id)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        if database:
            # 返回指定数据库的表
            tables = manager.get_tables(ds_id, database)
            return {
                "database": database,
                "tables": tables
            }
        else:
            # 返回所有选中数据库的表
            all_tables = manager.get_all_tables_by_datasource(ds_id)
            return {
                "databases": list(all_tables.keys()),
                "tables_by_database": all_tables
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasources/{ds_id}/tables/{database}/{table}/ddl")
def get_table_ddl(ds_id: str, database: str, table: str):
    """获取表的DDL语句"""
    try:
        manager = get_datasource_manager()
        ds = manager.get(ds_id)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        ddl = manager.get_table_ddl(ds_id, database, table)
        return {
            "database": database,
            "table": table,
            "ddl": ddl
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GenerateCorpusRequest(BaseModel):
    """生成语料请求"""
    databases: Optional[List[str]] = Field(None, description="要处理的数据库列表，不提供则使用数据源配置的databases")
    generate_documentation: bool = Field(True, description="是否生成字段文档")
    generate_sql_examples: bool = Field(True, description="是否生成SQL示例")


class CorpusItem(BaseModel):
    """语料项"""
    id: str
    type: str  # ddl/sql/documentation
    content: str
    question: Optional[str] = None
    tags: Optional[str] = None


class GenerateCorpusResponse(BaseModel):
    """生成语料响应"""
    datasource_name: str
    table_count: int
    corpus: List[CorpusItem]


@router.post("/datasources/{ds_id}/generate_corpus", response_model=GenerateCorpusResponse)
def generate_corpus(ds_id: str, request: GenerateCorpusRequest = GenerateCorpusRequest()):
    """
    从数据源生成语料
    
    1. 提取所有表的DDL
    2. 使用AI生成字段说明文档
    3. 使用AI生成SQL查询示例
    """
    try:
        print(f"[API] 开始生成语料，数据源ID: {ds_id}")
        vn = get_vn()
        generator = CorpusGenerator(vn)
        
        result = generator.generate_corpus_from_datasource(
            datasource_id=ds_id,
            databases=request.databases,
            generate_documentation=request.generate_documentation,
            generate_sql_examples=request.generate_sql_examples
        )
        
        print(f"[API] 语料生成完成: 表数量={result['table_count']}, 语料数量={len(result['corpus'])}")
        return result
    except ValueError as e:
        print(f"[API] 生成语料失败 (ValueError): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API] 生成语料失败 (Exception): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成语料失败: {str(e)}")
