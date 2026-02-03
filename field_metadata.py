"""
字段元数据管理模块
管理数据库字段的描述信息，支持多来源（DDL注释、AI推断、人工定义）
"""
import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class MetadataSource(str, Enum):
    """元数据来源"""
    DDL = "ddl"           # 来自 DDL COMMENT
    AI_INFERRED = "ai"    # AI 推断
    HUMAN = "human"       # 人工定义


class FieldQuality(str, Enum):
    """字段质量等级"""
    HIGH = "high"           # 高质量：有清晰的注释
    NEEDS_CONFIRM = "needs_confirm"  # 需确认：AI 已推断，待确认
    UNKNOWN = "unknown"     # 未知：无法推断


@dataclass
class FieldMetadata:
    """字段元数据"""
    database: str
    table: str
    column: str
    data_type: str
    
    # 三层来源的描述
    ddl_comment: Optional[str] = None       # 来自 DDL
    ai_inferred: Optional[str] = None       # AI 推断
    human_defined: Optional[str] = None     # 人工定义（最高优先）
    
    # 枚举值（适用于 status/type 等字段）
    enum_values: Optional[Dict[str, str]] = None  # {"0": "禁用", "1": "启用"}
    
    # 元信息
    confidence: float = 0.0      # AI 推断置信度 (0-1)
    quality: str = FieldQuality.UNKNOWN.value
    source: str = MetadataSource.DDL.value   # 当前使用的来源
    needs_review: bool = False   # 是否需要人工审核
    
    # 时间戳
    created_at: str = ""
    updated_at: str = ""
    updated_by: Optional[str] = None  # 人工修改者
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def get_description(self) -> str:
        """获取最终描述（按优先级）"""
        return (
            self.human_defined or
            self.ai_inferred or
            self.ddl_comment or
            f"字段 {self.column}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldMetadata':
        """从字典创建"""
        # 过滤掉不存在的字段
        valid_fields = {
            'database', 'table', 'column', 'data_type',
            'ddl_comment', 'ai_inferred', 'human_defined',
            'enum_values', 'confidence', 'quality', 'source',
            'needs_review', 'created_at', 'updated_at', 'updated_by'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class TableMetadata:
    """表元数据"""
    database: str
    table: str
    ddl: str
    ddl_hash: str  # DDL 的 hash，用于检测变化
    fields: List[FieldMetadata] = field(default_factory=list)
    
    # 统计信息
    total_fields: int = 0
    high_quality_count: int = 0
    needs_confirm_count: int = 0
    unknown_count: int = 0
    
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'database': self.database,
            'table': self.table,
            'ddl': self.ddl,
            'ddl_hash': self.ddl_hash,
            'fields': [f.to_dict() for f in self.fields],
            'total_fields': self.total_fields,
            'high_quality_count': self.high_quality_count,
            'needs_confirm_count': self.needs_confirm_count,
            'unknown_count': self.unknown_count,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


@dataclass
class DDLQualityReport:
    """DDL 质量分析报告"""
    datasource_id: str
    databases: List[str]
    
    # 统计
    total_tables: int = 0
    total_fields: int = 0
    high_quality_fields: int = 0
    needs_confirm_fields: int = 0
    unknown_fields: int = 0
    
    # 详细信息
    tables: List[TableMetadata] = field(default_factory=list)
    
    # 分类后的字段列表（用于前端展示）
    high_quality_list: List[Dict] = field(default_factory=list)
    needs_confirm_list: List[Dict] = field(default_factory=list)
    unknown_list: List[Dict] = field(default_factory=list)
    
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'datasource_id': self.datasource_id,
            'databases': self.databases,
            'total_tables': self.total_tables,
            'total_fields': self.total_fields,
            'high_quality_fields': self.high_quality_fields,
            'needs_confirm_fields': self.needs_confirm_fields,
            'unknown_fields': self.unknown_fields,
            'tables': [t.to_dict() for t in self.tables],
            'high_quality_list': self.high_quality_list,
            'needs_confirm_list': self.needs_confirm_list,
            'unknown_list': self.unknown_list,
            'created_at': self.created_at
        }


class FieldMetadataManager:
    """字段元数据管理器"""
    
    # 常见字段名映射规则
    FIELD_NAME_PATTERNS = {
        # ID 类
        r'^id$': '主键ID',
        r'_id$': 'ID',
        r'^uuid$': '唯一标识符',
        
        # 用户相关
        r'^user_?id$': '用户ID',
        r'^usr_?id$': '用户ID',
        r'^user_?name$': '用户名',
        r'^nick_?name$': '昵称',
        r'^real_?name$': '真实姓名',
        r'^email$': '电子邮箱',
        r'^phone$': '手机号码',
        r'^mobile$': '手机号码',
        
        # 时间相关
        r'^created?_?(at|time|date)?$': '创建时间',
        r'^crt_?t[im]?$': '创建时间',
        r'^updated?_?(at|time|date)?$': '更新时间',
        r'^upd_?t[im]?$': '更新时间',
        r'^deleted?_?(at|time|date)?$': '删除时间',
        r'^start_?(time|date)?$': '开始时间',
        r'^end_?(time|date)?$': '结束时间',
        r'^expire[ds]?_?(at|time|date)?$': '过期时间',
        
        # 状态相关
        r'^status$': '状态',
        r'^state$': '状态',
        r'^is_': '是否',
        r'^has_': '是否有',
        r'^enabled?$': '是否启用',
        r'^disabled?$': '是否禁用',
        r'^deleted?$': '是否删除',
        r'^active$': '是否激活',
        r'^visible$': '是否可见',
        r'^locked$': '是否锁定',
        
        # 数量/金额相关
        r'^count$': '数量',
        r'_count$': '数量',
        r'^num$': '数量',
        r'_num$': '数量',
        r'^amount$': '金额',
        r'^amt$': '金额',
        r'^price$': '价格',
        r'^total$': '总计',
        r'^balance$': '余额',
        r'^fee$': '费用',
        
        # 描述相关
        r'^desc(ription)?$': '描述',
        r'^remark$': '备注',
        r'^note$': '备注',
        r'^comment$': '注释',
        r'^memo$': '备忘',
        
        # 排序相关
        r'^sort$': '排序值',
        r'^order$': '排序值',
        r'^seq(uence)?$': '序号',
        r'^priority$': '优先级',
        r'^weight$': '权重',
        
        # 其他常见
        r'^type$': '类型',
        r'^category$': '分类',
        r'^level$': '级别',
        r'^version$': '版本',
        r'^code$': '编码',
        r'^name$': '名称',
        r'^title$': '标题',
        r'^content$': '内容',
        r'^url$': 'URL地址',
        r'^path$': '路径',
        r'^ip$': 'IP地址',
        r'^ext_?data$': '扩展数据',
        r'^extra$': '额外信息',
        r'^meta$': '元信息',
        r'^config$': '配置',
        r'^setting$': '设置',
    }
    
    # 可能是枚举的字段名
    ENUM_FIELD_PATTERNS = [
        r'^status$', r'^state$', r'^type$', r'^level$',
        r'^category$', r'^flag$', r'^mode$', r'^role$',
        r'_status$', r'_state$', r'_type$', r'_level$',
    ]
    
    def __init__(self, storage_path: str = "./field_metadata"):
        """
        初始化字段元数据管理器
        
        Args:
            storage_path: 元数据存储路径
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._cache: Dict[str, FieldMetadata] = {}
        self._load_cache()
    
    def _get_storage_file(self, datasource_id: str) -> str:
        """获取存储文件路径"""
        return os.path.join(self.storage_path, f"{datasource_id}.json")
    
    def _load_cache(self):
        """加载所有缓存"""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for key, value in data.items():
                            self._cache[key] = FieldMetadata.from_dict(value)
                except Exception as e:
                    print(f"[FieldMetadataManager] 加载缓存失败 {filename}: {e}")
    
    def _save_to_file(self, datasource_id: str):
        """保存到文件"""
        filepath = self._get_storage_file(datasource_id)
        
        # 筛选该数据源的元数据
        data = {}
        for key, meta in self._cache.items():
            if key.startswith(f"{datasource_id}:"):
                data[key] = meta.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _make_key(self, database: str, table: str, column: str) -> str:
        """生成缓存key"""
        return f"{database}:{table}:{column}"
    
    def _compute_ddl_hash(self, ddl: str) -> str:
        """计算 DDL 的 hash"""
        # 移除空白字符后计算 hash，避免格式差异导致的 hash 变化
        normalized = re.sub(r'\s+', ' ', ddl.strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, database: str, table: str, column: str) -> Optional[FieldMetadata]:
        """获取字段元数据"""
        key = self._make_key(database, table, column)
        return self._cache.get(key)
    
    def save(self, metadata: FieldMetadata, datasource_id: str = "default"):
        """保存字段元数据"""
        key = self._make_key(metadata.database, metadata.table, metadata.column)
        metadata.updated_at = datetime.now().isoformat()
        self._cache[key] = metadata
        self._save_to_file(datasource_id)
    
    def batch_save(self, metadata_list: List[FieldMetadata], datasource_id: str = "default"):
        """批量保存字段元数据"""
        for meta in metadata_list:
            key = self._make_key(meta.database, meta.table, meta.column)
            meta.updated_at = datetime.now().isoformat()
            self._cache[key] = meta
        self._save_to_file(datasource_id)
    
    def infer_field_description(self, column: str, data_type: str, ddl_comment: Optional[str] = None) -> Tuple[str, float]:
        """
        推断字段描述
        
        Args:
            column: 字段名
            data_type: 数据类型
            ddl_comment: DDL 中的注释
            
        Returns:
            (描述, 置信度)
        """
        # 如果有 DDL 注释且长度 >= 2，直接使用
        if ddl_comment and len(ddl_comment.strip()) >= 2:
            return ddl_comment.strip(), 1.0
        
        column_lower = column.lower()
        
        # 尝试匹配规则
        for pattern, description in self.FIELD_NAME_PATTERNS.items():
            if re.match(pattern, column_lower, re.IGNORECASE):
                return description, 0.8
        
        # 尝试拆分命名（下划线分隔）
        parts = column_lower.split('_')
        if len(parts) > 1:
            # 尝试匹配每个部分
            descriptions = []
            for part in parts:
                for pattern, desc in self.FIELD_NAME_PATTERNS.items():
                    if re.match(pattern, part, re.IGNORECASE):
                        descriptions.append(desc)
                        break
            if descriptions:
                return ''.join(descriptions), 0.6
        
        # 无法推断
        return None, 0.0
    
    def is_likely_enum_field(self, column: str, data_type: str) -> bool:
        """判断是否可能是枚举字段"""
        column_lower = column.lower()
        data_type_lower = data_type.lower()
        
        # 类型检查：通常是 INT, TINYINT, SMALLINT
        is_int_type = any(t in data_type_lower for t in ['int', 'tinyint', 'smallint'])
        
        # 名称检查
        for pattern in self.ENUM_FIELD_PATTERNS:
            if re.search(pattern, column_lower, re.IGNORECASE):
                return True
        
        return False
    
    def parse_ddl_fields(self, ddl: str) -> List[Dict[str, Any]]:
        """
        解析 DDL 中的字段信息
        
        Args:
            ddl: CREATE TABLE 语句
            
        Returns:
            字段信息列表 [{column, data_type, comment}, ...]
        """
        fields = []
        
        # 方法1: 按行解析，更可靠
        # 先提取括号内的字段定义部分
        create_match = re.search(r'CREATE\s+TABLE[^(]*\((.*)\)', ddl, re.IGNORECASE | re.DOTALL)
        if not create_match:
            return fields
        
        fields_section = create_match.group(1)
        
        # 按行分割，处理每个字段定义
        # 需要处理逗号分隔，但注意括号内的逗号（如 decimal(10,2)）
        lines = []
        current_line = ""
        paren_depth = 0
        
        for char in fields_section:
            if char == '(':
                paren_depth += 1
                current_line += char
            elif char == ')':
                paren_depth -= 1
                current_line += char
            elif char == ',' and paren_depth == 0:
                lines.append(current_line.strip())
                current_line = ""
            else:
                current_line += char
        
        if current_line.strip():
            lines.append(current_line.strip())
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过主键、索引等约束定义
            upper_line = line.upper()
            if any(upper_line.startswith(kw) for kw in ['PRIMARY', 'KEY', 'INDEX', 'UNIQUE', 'CONSTRAINT', 'FOREIGN']):
                continue
            
            # 匹配字段定义: `column_name` TYPE ... COMMENT '...'
            # 支持带反引号和不带反引号的字段名
            field_match = re.match(r'`?(\w+)`?\s+(\w+(?:\([^)]+\))?)', line)
            if not field_match:
                continue
            
            column = field_match.group(1)
            data_type = field_match.group(2)
            
            # 单独提取 COMMENT（更可靠）
            comment = None
            comment_match = re.search(r"COMMENT\s+['\"](.+?)['\"]", line, re.IGNORECASE)
            if comment_match:
                comment = comment_match.group(1)
            
            fields.append({
                'column': column,
                'data_type': data_type,
                'comment': comment
            })
        
        return fields
    
    def analyze_table_quality(
        self, 
        database: str, 
        table: str, 
        ddl: str,
        use_cached: bool = True
    ) -> TableMetadata:
        """
        分析单个表的字段质量
        
        Args:
            database: 数据库名
            table: 表名
            ddl: DDL 语句
            use_cached: 是否使用缓存的元数据
            
        Returns:
            TableMetadata
        """
        ddl_hash = self._compute_ddl_hash(ddl)
        fields_info = self.parse_ddl_fields(ddl)
        
        field_metadata_list = []
        high_count = 0
        confirm_count = 0
        unknown_count = 0
        
        for field_info in fields_info:
            column = field_info['column']
            data_type = field_info['data_type']
            ddl_comment = field_info.get('comment')
            
            # 检查是否有缓存的人工定义
            cached = self.get(database, table, column) if use_cached else None
            
            # 推断描述
            inferred_desc, confidence = self.infer_field_description(
                column, data_type, ddl_comment
            )
            
            # 确定质量等级
            if cached and cached.human_defined:
                # 有人工定义，高质量
                quality = FieldQuality.HIGH.value
                source = MetadataSource.HUMAN.value
                high_count += 1
            elif ddl_comment and len(ddl_comment.strip()) >= 3:
                # DDL 注释足够详细，高质量
                quality = FieldQuality.HIGH.value
                source = MetadataSource.DDL.value
                high_count += 1
            elif inferred_desc and confidence >= 0.6:
                # AI/规则推断成功，需确认
                quality = FieldQuality.NEEDS_CONFIRM.value
                source = MetadataSource.AI_INFERRED.value
                confirm_count += 1
            else:
                # 无法推断
                quality = FieldQuality.UNKNOWN.value
                source = MetadataSource.DDL.value
                unknown_count += 1
            
            # 检查是否可能是枚举字段
            is_enum = self.is_likely_enum_field(column, data_type)
            
            field_meta = FieldMetadata(
                database=database,
                table=table,
                column=column,
                data_type=data_type,
                ddl_comment=ddl_comment,
                ai_inferred=inferred_desc if confidence > 0 else None,
                human_defined=cached.human_defined if cached else None,
                enum_values=cached.enum_values if cached else None,
                confidence=confidence,
                quality=quality,
                source=source,
                needs_review=(quality != FieldQuality.HIGH.value) or is_enum
            )
            
            field_metadata_list.append(field_meta)
        
        return TableMetadata(
            database=database,
            table=table,
            ddl=ddl,
            ddl_hash=ddl_hash,
            fields=field_metadata_list,
            total_fields=len(field_metadata_list),
            high_quality_count=high_count,
            needs_confirm_count=confirm_count,
            unknown_count=unknown_count
        )
    
    def generate_enhanced_ddl(self, table_meta: TableMetadata) -> str:
        """
        生成增强后的 DDL（添加字段描述注释）
        
        Args:
            table_meta: 表元数据
            
        Returns:
            带有增强注释的 DDL
        """
        enhanced_ddl = table_meta.ddl
        
        for field in table_meta.fields:
            description = field.get_description()
            
            # 如果原 DDL 中没有 COMMENT，添加注释
            if not field.ddl_comment and description:
                # 查找字段定义并添加 COMMENT
                pattern = rf"`{field.column}`\s+{re.escape(field.data_type)}"
                replacement = f"`{field.column}` {field.data_type} COMMENT '{description}'"
                enhanced_ddl = re.sub(pattern, replacement, enhanced_ddl, flags=re.IGNORECASE)
        
        return enhanced_ddl
    
    def get_field_description_map(self, database: str, table: str) -> Dict[str, str]:
        """
        获取表的字段描述映射
        
        Args:
            database: 数据库名
            table: 表名
            
        Returns:
            {column_name: description}
        """
        result = {}
        prefix = f"{database}:{table}:"
        
        for key, meta in self._cache.items():
            if key.startswith(prefix):
                result[meta.column] = meta.get_description()
        
        return result


# 全局单例
_field_metadata_manager: Optional[FieldMetadataManager] = None


def get_field_metadata_manager() -> FieldMetadataManager:
    """获取字段元数据管理器单例"""
    global _field_metadata_manager
    if _field_metadata_manager is None:
        storage_path = os.getenv('FIELD_METADATA_PATH', './field_metadata')
        _field_metadata_manager = FieldMetadataManager(storage_path)
    return _field_metadata_manager
