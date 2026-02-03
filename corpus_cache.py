"""
语料生成缓存模块

存储 DDL 质量分析结果，供语料生成直接使用
避免重复获取 DDL 和分析字段元数据
"""
import time
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class CorpusReadyData:
    """语料生成就绪数据"""
    datasource_id: str
    # 增强后的 DDL: {(database, table): enhanced_ddl}
    enhanced_ddls: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # 字段描述映射: {(database, table): {column: description}}
    field_metadata: Dict[Tuple[str, str], Dict[str, str]] = field(default_factory=dict)
    # 表列表: {database: [table_name, ...]}
    tables_by_db: Dict[str, list] = field(default_factory=dict)
    # 时间戳
    created_at: float = 0.0
    expires_at: float = 0.0
    
    def is_valid(self) -> bool:
        """检查数据是否有效（未过期）"""
        return time.time() < self.expires_at
    
    def get_table_count(self) -> int:
        """获取表数量"""
        return len(self.enhanced_ddls)


# 全局缓存
_corpus_ready_cache: Dict[str, CorpusReadyData] = {}

# 缓存过期时间（秒）- 默认5分钟
_CACHE_TTL = 300


def set_cache_ttl(ttl_seconds: int):
    """设置缓存过期时间"""
    global _CACHE_TTL
    _CACHE_TTL = ttl_seconds


def cache_corpus_ready_data(
    datasource_id: str,
    enhanced_ddls: Dict[Tuple[str, str], str],
    field_metadata: Dict[Tuple[str, str], Dict[str, str]],
    tables_by_db: Dict[str, list] = None
):
    """
    缓存质量分析结果，供语料生成使用
    
    Args:
        datasource_id: 数据源ID
        enhanced_ddls: 增强后的DDL字典 {(database, table): enhanced_ddl}
        field_metadata: 字段描述映射 {(database, table): {column: description}}
        tables_by_db: 按数据库分组的表列表
    """
    now = time.time()
    
    # 构建 tables_by_db
    if tables_by_db is None:
        tables_by_db = {}
        for (db, table) in enhanced_ddls.keys():
            if db not in tables_by_db:
                tables_by_db[db] = []
            if table not in tables_by_db[db]:
                tables_by_db[db].append(table)
    
    cache_data = CorpusReadyData(
        datasource_id=datasource_id,
        enhanced_ddls=enhanced_ddls,
        field_metadata=field_metadata,
        tables_by_db=tables_by_db,
        created_at=now,
        expires_at=now + _CACHE_TTL
    )
    
    _corpus_ready_cache[datasource_id] = cache_data
    
    print(f"[CorpusCache] 缓存就绪数据: datasource={datasource_id}, "
          f"tables={cache_data.get_table_count()}, ttl={_CACHE_TTL}s")


def get_corpus_ready_data(datasource_id: str) -> Optional[CorpusReadyData]:
    """
    获取缓存的语料就绪数据
    
    Args:
        datasource_id: 数据源ID
        
    Returns:
        CorpusReadyData 或 None（未命中或已过期）
    """
    cached = _corpus_ready_cache.get(datasource_id)
    
    if cached is None:
        print(f"[CorpusCache] 未命中: datasource={datasource_id}")
        return None
    
    if not cached.is_valid():
        # 已过期，清除
        del _corpus_ready_cache[datasource_id]
        print(f"[CorpusCache] 已过期: datasource={datasource_id}")
        return None
    
    remaining = int(cached.expires_at - time.time())
    print(f"[CorpusCache] 命中缓存: datasource={datasource_id}, "
          f"tables={cached.get_table_count()}, remaining={remaining}s")
    
    return cached


def clear_cache(datasource_id: str = None):
    """
    清除缓存
    
    Args:
        datasource_id: 指定数据源ID，None则清除所有
    """
    if datasource_id:
        if datasource_id in _corpus_ready_cache:
            del _corpus_ready_cache[datasource_id]
            print(f"[CorpusCache] 清除缓存: datasource={datasource_id}")
    else:
        _corpus_ready_cache.clear()
        print(f"[CorpusCache] 清除所有缓存")


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计信息"""
    now = time.time()
    stats = {
        "total_cached": len(_corpus_ready_cache),
        "cache_ttl": _CACHE_TTL,
        "entries": []
    }
    
    for ds_id, data in _corpus_ready_cache.items():
        remaining = max(0, int(data.expires_at - now))
        stats["entries"].append({
            "datasource_id": ds_id,
            "table_count": data.get_table_count(),
            "created_at": data.created_at,
            "remaining_seconds": remaining,
            "is_valid": data.is_valid()
        })
    
    return stats
