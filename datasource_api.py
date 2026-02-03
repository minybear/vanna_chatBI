"""
数据源管理 API 路由
提供数据源的CRUD、测试连接、激活等接口
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import queue
import threading

from datasource_manager import get_datasource_manager, DataSource
from corpus_generator import CorpusGenerator
from field_metadata import (
    get_field_metadata_manager, 
    FieldMetadata, 
    DDLQualityReport,
    FieldQuality,
    MetadataSource
)
from corpus_cache import cache_corpus_ready_data

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


# ============== DDL 质量分析相关 API ==============

class AnalyzeDDLQualityRequest(BaseModel):
    """分析 DDL 质量请求"""
    databases: Optional[List[str]] = Field(None, description="要分析的数据库列表")
    tables: Optional[Dict[str, List[str]]] = Field(None, description="要分析的表，格式: {database: [table1, table2]}")
    use_ai_inference: bool = Field(False, description="是否使用AI推断字段含义（较慢但更准确）")


class FieldMetadataItem(BaseModel):
    """字段元数据项"""
    database: str
    table: str
    column: str
    data_type: str
    ddl_comment: Optional[str] = None
    ai_inferred: Optional[str] = None
    human_defined: Optional[str] = None
    enum_values: Optional[Dict[str, str]] = None
    confidence: float = 0.0
    quality: str
    needs_review: bool = False


class TableQualityInfo(BaseModel):
    """表质量信息"""
    database: str
    table: str
    ddl_hash: str
    total_fields: int
    high_quality_count: int
    needs_confirm_count: int
    unknown_count: int
    fields: List[FieldMetadataItem]


class DDLQualityReportResponse(BaseModel):
    """DDL 质量分析报告响应"""
    datasource_id: str
    databases: List[str]
    total_tables: int
    total_fields: int
    high_quality_fields: int
    needs_confirm_fields: int
    unknown_fields: int
    tables: List[TableQualityInfo]
    # 分类列表（用于前端快速展示）
    high_quality_list: List[Dict[str, Any]]
    needs_confirm_list: List[Dict[str, Any]]
    unknown_list: List[Dict[str, Any]]


@router.post("/datasources/{ds_id}/analyze_ddl_quality", response_model=DDLQualityReportResponse)
def analyze_ddl_quality(ds_id: str, request: AnalyzeDDLQualityRequest = AnalyzeDDLQualityRequest()):
    """
    分析数据源的 DDL 质量
    
    返回字段质量分析报告，包括：
    - 高质量字段（有清晰注释）
    - 需确认字段（AI已推断，待确认）
    - 未知字段（无法推断，需手动补充）
    """
    try:
        print(f"[API] 开始分析DDL质量，数据源ID: {ds_id}")
        
        manager = get_datasource_manager()
        ds = manager.get(ds_id)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        meta_manager = get_field_metadata_manager()
        
        # 确定要分析的数据库和表
        tables_to_analyze = []
        target_databases = []
        
        if request.tables:
            # 如果指定了表，只分析这些表
            target_databases = list(request.tables.keys())
            for db_name, table_names in request.tables.items():
                for table_name in table_names:
                    tables_to_analyze.append((db_name, table_name))
        else:
            # 否则分析所有配置的数据库下的所有表
            target_databases = request.databases or ds.databases
            if not target_databases:
                raise HTTPException(status_code=400, detail="数据源未配置数据库")
            
            for db_name in target_databases:
                tables = manager.get_tables(ds_id, db_name)
                for table in tables:
                    table_name = table['name'] if isinstance(table, dict) else table
                    tables_to_analyze.append((db_name, table_name))
        
        print(f"[API] 待分析表数量: {len(tables_to_analyze)}")
        
        # 并发获取 DDL
        ddl_map = {}  # {(db, table): ddl}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_table = {
                executor.submit(manager.get_table_ddl, ds_id, db, table): (db, table)
                for db, table in tables_to_analyze
            }
            for future in as_completed(future_to_table):
                db, table = future_to_table[future]
                try:
                    ddl = future.result()
                    if ddl:
                        ddl_map[(db, table)] = ddl
                except Exception as e:
                    print(f"[API] 获取DDL失败 {db}.{table}: {e}")
        
        print(f"[API] 获取到 {len(ddl_map)} 个表的DDL")
        
        # 分析每个表的字段质量
        table_quality_list = []
        high_quality_list = []
        needs_confirm_list = []
        unknown_list = []
        
        total_fields = 0
        total_high = 0
        total_confirm = 0
        total_unknown = 0
        
        # 存储 TableMetadata 对象，用于生成增强 DDL
        table_meta_map = {}  # {(db, table): TableMetadata}
        
        for (db_name, table_name), ddl in ddl_map.items():
            table_meta = meta_manager.analyze_table_quality(db_name, table_name, ddl)
            table_meta_map[(db_name, table_name)] = table_meta
            
            # 转换为响应格式
            fields_response = []
            for field in table_meta.fields:
                field_item = {
                    "database": field.database,
                    "table": field.table,
                    "column": field.column,
                    "data_type": field.data_type,
                    "ddl_comment": field.ddl_comment,
                    "ai_inferred": field.ai_inferred,
                    "human_defined": field.human_defined,
                    "enum_values": field.enum_values,
                    "confidence": field.confidence,
                    "quality": field.quality,
                    "needs_review": field.needs_review
                }
                fields_response.append(field_item)
                
                # 分类到不同列表
                if field.quality == FieldQuality.HIGH.value:
                    high_quality_list.append(field_item)
                elif field.quality == FieldQuality.NEEDS_CONFIRM.value:
                    needs_confirm_list.append(field_item)
                else:
                    unknown_list.append(field_item)
            
            table_quality_list.append({
                "database": db_name,
                "table": table_name,
                "ddl_hash": table_meta.ddl_hash,
                "total_fields": table_meta.total_fields,
                "high_quality_count": table_meta.high_quality_count,
                "needs_confirm_count": table_meta.needs_confirm_count,
                "unknown_count": table_meta.unknown_count,
                "fields": fields_response
            })
            
            total_fields += table_meta.total_fields
            total_high += table_meta.high_quality_count
            total_confirm += table_meta.needs_confirm_count
            total_unknown += table_meta.unknown_count
        
        print(f"[API] DDL质量分析完成: 总字段={total_fields}, 高质量={total_high}, 需确认={total_confirm}, 未知={total_unknown}")
        
        # ========== 生成增强 DDL 并缓存，供语料生成使用 ==========
        enhanced_ddls = {}  # {(db, table): enhanced_ddl}
        field_metadata = {}  # {(db, table): {column: description}}
        
        for (db_name, table_name), table_meta in table_meta_map.items():
            # 生成增强后的 DDL（带完整字段注释）
            enhanced_ddl = meta_manager.generate_enhanced_ddl(table_meta)
            enhanced_ddls[(db_name, table_name)] = enhanced_ddl
            
            # 收集字段描述映射
            field_desc_map = {}
            for field in table_meta.fields:
                field_desc_map[field.column] = field.get_description()
            field_metadata[(db_name, table_name)] = field_desc_map
        
        # 缓存供语料生成使用
        cache_corpus_ready_data(
            datasource_id=ds_id,
            enhanced_ddls=enhanced_ddls,
            field_metadata=field_metadata
        )
        print(f"[API] 增强DDL已缓存，共 {len(enhanced_ddls)} 个表")
        # ========================================================
        
        return {
            "datasource_id": ds_id,
            "databases": target_databases,
            "total_tables": len(table_quality_list),
            "total_fields": total_fields,
            "high_quality_fields": total_high,
            "needs_confirm_fields": total_confirm,
            "unknown_fields": total_unknown,
            "tables": table_quality_list,
            "high_quality_list": high_quality_list,
            "needs_confirm_list": needs_confirm_list,
            "unknown_list": unknown_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] 分析DDL质量失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析DDL质量失败: {str(e)}")


class SaveFieldMetadataRequest(BaseModel):
    """保存字段元数据请求"""
    fields: List[FieldMetadataItem]


@router.post("/datasources/{ds_id}/field_metadata/batch_save")
def batch_save_field_metadata(ds_id: str, request: SaveFieldMetadataRequest):
    """
    批量保存字段元数据
    
    用于保存用户确认/修改的字段描述
    """
    try:
        print(f"[API] 批量保存字段元数据，数据源ID: {ds_id}, 字段数: {len(request.fields)}")
        
        meta_manager = get_field_metadata_manager()
        
        metadata_list = []
        for field in request.fields:
            meta = FieldMetadata(
                database=field.database,
                table=field.table,
                column=field.column,
                data_type=field.data_type,
                ddl_comment=field.ddl_comment,
                ai_inferred=field.ai_inferred,
                human_defined=field.human_defined,
                enum_values=field.enum_values,
                confidence=field.confidence,
                quality=field.quality,
                source=MetadataSource.HUMAN.value if field.human_defined else MetadataSource.AI_INFERRED.value,
                needs_review=False  # 已确认，不需要审核
            )
            metadata_list.append(meta)
        
        meta_manager.batch_save(metadata_list, ds_id)
        
        print(f"[API] 字段元数据保存成功")
        return {
            "status": "success",
            "message": f"已保存 {len(metadata_list)} 个字段的元数据",
            "saved_count": len(metadata_list)
        }
        
    except Exception as e:
        print(f"[API] 保存字段元数据失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"保存字段元数据失败: {str(e)}")


@router.get("/datasources/{ds_id}/field_metadata/{database}/{table}")
def get_table_field_metadata(ds_id: str, database: str, table: str):
    """
    获取表的字段元数据
    """
    try:
        meta_manager = get_field_metadata_manager()
        
        # 获取该表所有字段的元数据
        field_map = meta_manager.get_field_description_map(database, table)
        
        return {
            "database": database,
            "table": table,
            "fields": field_map
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取字段元数据失败: {str(e)}")


# ============== 增强版语料生成 API ==============

class GenerateCorpusEnhancedRequest(BaseModel):
    """增强版生成语料请求"""
    databases: Optional[List[str]] = Field(None, description="要处理的数据库列表")
    tables: Optional[Dict[str, List[str]]] = Field(None, description="要处理的表，格式: {database: [table1, table2]}")
    generate_documentation: bool = Field(True, description="是否生成字段文档")
    generate_sql_examples: bool = Field(True, description="是否生成SQL示例")
    use_metadata_enhancement: bool = Field(True, description="是否使用字段元数据增强")
    skip_quality_check: bool = Field(False, description="是否跳过质量检查直接生成")


@router.post("/datasources/{ds_id}/generate_corpus_enhanced")
def generate_corpus_enhanced(ds_id: str, request: GenerateCorpusEnhancedRequest):
    """
    增强版语料生成
    
    1. 先分析 DDL 质量
    2. 使用元数据增强 DDL
    3. 生成高质量语料
    """
    try:
        print(f"[API] 开始增强版语料生成，数据源ID: {ds_id}")
        
        manager = get_datasource_manager()
        ds = manager.get(ds_id)
        if not ds:
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        meta_manager = get_field_metadata_manager()
        
        # 确定要处理的数据库和表
        target_databases = request.databases or ds.databases
        if not target_databases:
            raise HTTPException(status_code=400, detail="数据源未配置数据库")
        
        # 如果不跳过质量检查，先分析质量
        if not request.skip_quality_check:
            # 分析 DDL 质量
            analyze_request = AnalyzeDDLQualityRequest(
                databases=target_databases,
                tables=request.tables
            )
            quality_report = analyze_ddl_quality(ds_id, analyze_request)
            
            # 如果有需要确认或未知的字段，返回质量报告让用户确认
            if quality_report['needs_confirm_fields'] > 0 or quality_report['unknown_fields'] > 0:
                return {
                    "status": "needs_review",
                    "message": f"发现 {quality_report['needs_confirm_fields']} 个字段需要确认，{quality_report['unknown_fields']} 个字段需要手动补充",
                    "quality_report": quality_report
                }
        
        # 开始生成语料
        vn = get_vn()
        generator = CorpusGenerator(vn)
        
        result = generator.generate_corpus_from_datasource(
            datasource_id=ds_id,
            databases=target_databases,
            tables=request.tables,
            generate_documentation=request.generate_documentation,
            generate_sql_examples=request.generate_sql_examples,
            use_metadata=request.use_metadata_enhancement
        )
        
        return {
            "status": "success",
            "message": f"语料生成完成，共 {result['table_count']} 个表，{len(result['corpus'])} 条语料",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] 增强版语料生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成语料失败: {str(e)}")


# ============== SSE 流式语料生成 API ==============

class GenerateCorpusStreamRequest(BaseModel):
    """流式语料生成请求"""
    databases: Optional[List[str]] = Field(None, description="要处理的数据库列表")
    tables: Optional[Dict[str, List[str]]] = Field(None, description="要处理的表")
    generate_documentation: bool = Field(True, description="是否生成字段文档")
    generate_sql_examples: bool = Field(True, description="是否生成SQL示例")


def _sse_event(event_type: str, data: dict) -> str:
    """生成 SSE 事件格式"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/datasources/{ds_id}/generate_corpus_stream")
def generate_corpus_stream(ds_id: str, request: GenerateCorpusStreamRequest = GenerateCorpusStreamRequest()):
    """
    流式语料生成 (Server-Sent Events)
    
    实时返回生成进度和结果
    """
    def generate() -> Generator[str, None, None]:
        try:
            # 发送开始事件
            yield _sse_event("start", {
                "message": "开始语料生成",
                "datasource_id": ds_id,
                "timestamp": time.time()
            })
            
            manager = get_datasource_manager()
            ds = manager.get(ds_id)
            if not ds:
                yield _sse_event("error", {"message": "数据源不存在"})
                return
            
            # 检查缓存
            from corpus_cache import get_corpus_ready_data
            cached = get_corpus_ready_data(ds_id)
            if cached:
                yield _sse_event("progress", {
                    "stage": "cache",
                    "message": f"使用缓存的增强DDL，共 {cached.get_table_count()} 个表",
                    "table_count": cached.get_table_count()
                })
            else:
                yield _sse_event("progress", {
                    "stage": "no_cache",
                    "message": "未找到缓存，将重新获取DDL"
                })
            
            # 开始生成
            vn = get_vn()
            generator = CorpusGenerator(vn)
            
            # 计算批次信息
            target_databases = request.databases or ds.databases
            batch_size = generator._batch_size
            
            yield _sse_event("progress", {
                "stage": "init",
                "message": f"初始化完成，批处理大小: {batch_size}",
                "batch_size": batch_size,
                "databases": target_databases
            })
            
            # 执行生成
            start_time = time.time()
            result = generator.generate_corpus_from_datasource(
                datasource_id=ds_id,
                databases=target_databases,
                tables=request.tables,
                generate_documentation=request.generate_documentation,
                generate_sql_examples=request.generate_sql_examples,
                use_metadata=True
            )
            elapsed = time.time() - start_time
            
            # 发送完成事件
            yield _sse_event("complete", {
                "message": "语料生成完成",
                "table_count": result["table_count"],
                "corpus_count": len(result["corpus"]),
                "elapsed_seconds": round(elapsed, 1),
                "result": result
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse_event("error", {
                "message": str(e),
                "type": type(e).__name__
            })
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
