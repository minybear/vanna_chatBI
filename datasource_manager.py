"""
数据源管理模块
支持多数据源的CRUD、连接池管理、加密存储
"""
import os
import json
import uuid
import pymysql
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from utils.encryption import encrypt_password, decrypt_password


@dataclass
class DataSource:
    """数据源配置数据类"""
    id: str
    name: str
    host: str
    port: int
    user: str
    password: str  # 加密存储
    databases: List[str] = None  # 选中的数据库列表（支持多个）
    default_database: Optional[str] = None  # 默认数据库（执行SQL时使用）
    ssl_enabled: bool = True
    status: str = "inactive"  # active/inactive/error
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if self.databases is None:
            self.databases = []
    
    @property
    def database(self) -> Optional[str]:
        """向后兼容：返回默认数据库或第一个选中的数据库"""
        return self.default_database or (self.databases[0] if self.databases else None)
    
    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """转换为字典，默认不包含密码"""
        data = asdict(self)
        if not include_password:
            data['password'] = '******'
        # 添加向后兼容的 database 字段
        data['database'] = self.database
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """从字典创建实例"""
        # 向后兼容：如果只有 database 字段，转换为 databases 列表
        databases = data.get('databases', [])
        if not databases and data.get('database'):
            databases = [data.get('database')]
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            host=data.get('host', ''),
            port=int(data.get('port', 3306)),
            user=data.get('user', ''),
            password=data.get('password', ''),
            databases=databases,
            default_database=data.get('default_database'),
            ssl_enabled=data.get('ssl_enabled', True),
            status=data.get('status', 'inactive'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', '')
        )


class SimpleEmbeddingFunction:
    """简单的嵌入函数，用于元数据存储"""
    def __init__(self, dim: int = 8):
        self.dim = dim

    def name(self) -> str:
        return "simple"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in input]

    def _embed_text(self, text: str) -> List[float]:
        import hashlib
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]


class DataSourceManager:
    """
    数据源管理器
    负责数据源的CRUD操作、连接池管理、持久化存储
    """
    
    def __init__(self, chroma_client):
        """
        初始化数据源管理器
        
        Args:
            chroma_client: ChromaDB 客户端实例
        """
        self.chroma_client = chroma_client
        self.embedding_function = SimpleEmbeddingFunction()
        
        # 创建或获取数据源集合
        self.datasources_collection = chroma_client.get_or_create_collection(
            name="datasources",
            embedding_function=self.embedding_function,
        )
        
        # 连接池：存储活跃的数据库连接配置
        self._connection_pools: Dict[str, Dict[str, Any]] = {}
        
        # 当前激活的数据源ID
        self._active_datasource_id: Optional[str] = None
        
        # 默认数据源（从环境变量加载）
        self._default_datasource = self._load_default_datasource()
    
    def _load_default_datasource(self) -> Optional[DataSource]:
        """从环境变量加载默认数据源配置"""
        host = os.getenv('DB_HOST')
        if not host:
            return None
        
        # 支持多个数据库：DB_DATABASES=db1,db2,db3 或单个 DB_NAME=db1
        db_databases_str = os.getenv('DB_DATABASES', '')
        db_name = os.getenv('DB_NAME', '')
        
        if db_databases_str:
            databases = [db.strip() for db in db_databases_str.split(',') if db.strip()]
        elif db_name:
            databases = [db_name]
        else:
            databases = []
        
        return DataSource(
            id='default',
            name='默认数据源 (ENV)',
            host=host,
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),  # 环境变量中的密码不加密存储
            databases=databases,
            default_database=db_name or (databases[0] if databases else None),
            ssl_enabled=os.getenv('DB_SSL_ENABLED', 'True').lower() == 'true',
            status='active',
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def _now(self) -> str:
        """获取当前时间ISO格式"""
        return datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"ds-{uuid.uuid4().hex[:12]}"
    
    def create(self, name: str, host: str, port: int, user: str, password: str,
               databases: Optional[List[str]] = None, default_database: Optional[str] = None,
               ssl_enabled: bool = True) -> DataSource:
        """
        创建新数据源
        
        Args:
            name: 数据源名称（20字以内）
            host: 数据库主机
            port: 数据库端口
            user: 用户名
            password: 密码（明文，会被加密存储）
            databases: 选中的数据库列表
            default_database: 默认数据库
            ssl_enabled: 是否启用SSL
            
        Returns:
            创建的数据源对象
            
        Raises:
            ValueError: 名称过长或重复
        """
        # 验证名称长度
        if len(name) > 20:
            raise ValueError("数据源名称不能超过20个字符")
        
        # 检查名称是否重复
        existing = self.get_by_name(name)
        if existing:
            raise ValueError(f"数据源名称 '{name}' 已存在")
        
        now = self._now()
        ds = DataSource(
            id=self._generate_id(),
            name=name,
            host=host,
            port=port,
            user=user,
            password=encrypt_password(password),  # 加密存储
            databases=databases or [],
            default_database=default_database,
            ssl_enabled=ssl_enabled,
            status='inactive',
            created_at=now,
            updated_at=now
        )
        
        # 存储到 ChromaDB
        self._save_to_chromadb(ds)
        
        print(f"[DataSource] 创建数据源: {ds.name} ({ds.id})")
        return ds
    
    def _save_to_chromadb(self, ds: DataSource):
        """保存数据源到 ChromaDB"""
        doc = json.dumps(ds.to_dict(include_password=True), ensure_ascii=False)
        
        # 检查是否已存在
        existing = self.datasources_collection.get(ids=[ds.id])
        if existing and existing.get('ids'):
            # 更新
            self.datasources_collection.update(
                ids=[ds.id],
                documents=[doc],
                metadatas=[{
                    'name': ds.name,
                    'host': ds.host,
                    'status': ds.status,
                    'updated_at': ds.updated_at
                }]
            )
        else:
            # 新增
            self.datasources_collection.add(
                ids=[ds.id],
                documents=[doc],
                metadatas=[{
                    'name': ds.name,
                    'host': ds.host,
                    'status': ds.status,
                    'updated_at': ds.updated_at
                }]
            )
    
    def get(self, ds_id: str) -> Optional[DataSource]:
        """
        根据ID获取数据源
        
        Args:
            ds_id: 数据源ID
            
        Returns:
            数据源对象，不存在返回None
        """
        # 特殊处理默认数据源
        if ds_id == 'default' and self._default_datasource:
            return self._default_datasource
        
        data = self.datasources_collection.get(ids=[ds_id])
        if not data or not data.get('ids'):
            return None
        
        doc = data.get('documents', [None])[0]
        if not doc:
            return None
        
        try:
            ds_dict = json.loads(doc)
            return DataSource.from_dict(ds_dict)
        except Exception as e:
            print(f"[DataSource] 解析数据源失败: {e}")
            return None
    
    def get_by_name(self, name: str) -> Optional[DataSource]:
        """根据名称获取数据源"""
        data = self.datasources_collection.get(where={"name": name})
        if not data or not data.get('ids'):
            return None
        
        doc = data.get('documents', [None])[0]
        if not doc:
            return None
        
        try:
            ds_dict = json.loads(doc)
            return DataSource.from_dict(ds_dict)
        except Exception:
            return None
    
    def list_all(self) -> List[DataSource]:
        """
        获取所有数据源列表
        
        Returns:
            数据源列表（按更新时间倒序）
        """
        datasources = []
        
        # 添加默认数据源（如果存在）
        if self._default_datasource:
            datasources.append(self._default_datasource)
        
        # 从 ChromaDB 获取所有数据源
        data = self.datasources_collection.get()
        if data and data.get('ids'):
            for doc in data.get('documents', []):
                try:
                    ds_dict = json.loads(doc)
                    datasources.append(DataSource.from_dict(ds_dict))
                except Exception as e:
                    print(f"[DataSource] 解析数据源失败: {e}")
        
        # 按更新时间倒序排序
        datasources.sort(key=lambda x: x.updated_at or '', reverse=True)
        return datasources
    
    def update(self, ds_id: str, **kwargs) -> Optional[DataSource]:
        """
        更新数据源
        
        Args:
            ds_id: 数据源ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的数据源对象
        """
        if ds_id == 'default':
            raise ValueError("默认数据源不能修改")
        
        ds = self.get(ds_id)
        if not ds:
            return None
        
        # 更新字段
        if 'name' in kwargs:
            if len(kwargs['name']) > 20:
                raise ValueError("数据源名称不能超过20个字符")
            ds.name = kwargs['name']
        
        if 'host' in kwargs:
            ds.host = kwargs['host']
        if 'port' in kwargs:
            ds.port = int(kwargs['port'])
        if 'user' in kwargs:
            ds.user = kwargs['user']
        if 'password' in kwargs and kwargs['password']:
            ds.password = encrypt_password(kwargs['password'])
        if 'databases' in kwargs:
            ds.databases = kwargs['databases'] if kwargs['databases'] else []
        if 'default_database' in kwargs:
            ds.default_database = kwargs['default_database']
        if 'ssl_enabled' in kwargs:
            ds.ssl_enabled = kwargs['ssl_enabled']
        
        ds.updated_at = self._now()
        
        # 保存更新
        self._save_to_chromadb(ds)
        
        # 如果是活跃数据源，更新连接池
        if ds_id in self._connection_pools:
            self._update_connection_pool(ds)
        
        print(f"[DataSource] 更新数据源: {ds.name} ({ds.id})")
        return ds
    
    def delete(self, ds_id: str) -> bool:
        """
        删除数据源
        
        Args:
            ds_id: 数据源ID
            
        Returns:
            是否删除成功
        """
        if ds_id == 'default':
            raise ValueError("默认数据源不能删除")
        
        # 从连接池移除
        if ds_id in self._connection_pools:
            del self._connection_pools[ds_id]
        
        # 如果是当前激活的数据源，切换到默认
        if self._active_datasource_id == ds_id:
            self._active_datasource_id = None
        
        # 从 ChromaDB 删除
        self.datasources_collection.delete(ids=[ds_id])
        
        print(f"[DataSource] 删除数据源: {ds_id}")
        return True
    
    def test_connection(self, ds_id: str) -> Dict[str, Any]:
        """
        测试数据源连接
        
        Args:
            ds_id: 数据源ID
            
        Returns:
            测试结果 {success: bool, message: str, databases: List[str]}
        """
        ds = self.get(ds_id)
        if not ds:
            return {'success': False, 'message': '数据源不存在', 'databases': []}
        
        try:
            # 获取解密后的密码
            password = ds.password
            if ds.id != 'default':
                password = decrypt_password(ds.password)
            
            config = {
                'host': ds.host,
                'port': ds.port,
                'user': ds.user,
                'password': password,
            }
            
            if ds.ssl_enabled:
                config['ssl'] = {
                    'check_hostname': False,
                    'verify_mode': False
                }
            
            cnx = pymysql.connect(**config)
            cursor = cnx.cursor()
            
            # 获取数据库列表
            cursor.execute("SHOW DATABASES")
            databases = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            cnx.close()
            
            # 更新状态为 active
            if ds.id != 'default':
                ds.status = 'active'
                ds.updated_at = self._now()
                self._save_to_chromadb(ds)
            
            return {
                'success': True,
                'message': '连接成功',
                'databases': databases
            }
            
        except pymysql.Error as e:
            # 更新状态为 error
            if ds.id != 'default':
                ds.status = 'error'
                ds.updated_at = self._now()
                self._save_to_chromadb(ds)
            
            return {
                'success': False,
                'message': f'连接失败: {str(e)}',
                'databases': []
            }
    
    def activate(self, ds_id: str) -> bool:
        """
        激活数据源（设为当前使用的数据源）
        
        Args:
            ds_id: 数据源ID
            
        Returns:
            是否激活成功
        """
        ds = self.get(ds_id)
        if not ds:
            return False
        
        # 测试连接
        result = self.test_connection(ds_id)
        if not result['success']:
            return False
        
        self._active_datasource_id = ds_id
        self._update_connection_pool(ds)
        
        print(f"[DataSource] 激活数据源: {ds.name} ({ds.id})")
        return True
    
    def _update_connection_pool(self, ds: DataSource):
        """更新连接池配置"""
        password = ds.password
        if ds.id != 'default':
            password = decrypt_password(ds.password)
        
        self._connection_pools[ds.id] = {
            'host': ds.host,
            'port': ds.port,
            'user': ds.user,
            'password': password,
            'databases': ds.databases,
            'default_database': ds.default_database,
            'ssl_enabled': ds.ssl_enabled
        }
    
    def get_active_datasource(self) -> Optional[DataSource]:
        """获取当前激活的数据源"""
        if self._active_datasource_id:
            return self.get(self._active_datasource_id)
        return self._default_datasource
    
    def get_connection_config(self, ds_id: Optional[str] = None, database: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据库连接配置
        
        Args:
            ds_id: 数据源ID，如果不提供则使用当前激活的数据源
            database: 指定要连接的数据库，如果不提供则使用默认数据库
            
        Returns:
            数据库连接配置字典
        """
        target_id = ds_id or self._active_datasource_id or 'default'
        
        # 检查连接池缓存
        if target_id in self._connection_pools:
            config = self._connection_pools[target_id].copy()
        else:
            ds = self.get(target_id)
            if not ds:
                ds = self._default_datasource
            
            if not ds:
                raise ValueError("没有可用的数据源配置")
            
            password = ds.password
            if ds.id != 'default':
                password = decrypt_password(ds.password)
            
            config = {
                'host': ds.host,
                'port': ds.port,
                'user': ds.user,
                'password': password,
                'databases': ds.databases,
                'default_database': ds.default_database,
                'ssl_enabled': ds.ssl_enabled
            }
        
        # 确定要使用的数据库
        target_database = database or config.get('default_database')
        if not target_database and config.get('databases'):
            target_database = config['databases'][0]
        
        # 构建 pymysql 连接配置
        db_config = {
            'host': config['host'],
            'port': config['port'],
            'user': config['user'],
            'password': config['password'],
            # 添加超时设置，避免连接卡住
            'connect_timeout': 10,  # 连接超时10秒
            'read_timeout': 30,      # 读取超时30秒
            'write_timeout': 30,      # 写入超时30秒
        }
        
        if target_database:
            db_config['database'] = target_database
        
        if config.get('ssl_enabled'):
            db_config['ssl'] = {
                'check_hostname': False,
                'verify_mode': False
            }
        
        return db_config
    
    @contextmanager
    def get_connection(self, ds_id: Optional[str] = None, database: Optional[str] = None):
        """
        获取数据库连接（上下文管理器）
        
        Args:
            ds_id: 数据源ID
            database: 指定要连接的数据库
            
        Yields:
            pymysql 连接对象
        """
        config = self.get_connection_config(ds_id, database)
        cnx = pymysql.connect(**config)
        try:
            yield cnx
        finally:
            cnx.close()
    
    def get_tables(self, ds_id: str, database: str) -> List[Dict[str, Any]]:
        """
        获取指定数据库的所有表信息
        
        Args:
            ds_id: 数据源ID
            database: 数据库名
            
        Returns:
            表信息列表 [{name, type, rows, comment}, ...]
        """
        try:
            print(f"[DataSourceManager] 获取数据库 {database} 的表列表 (数据源: {ds_id})")
            with self.get_connection(ds_id, database) as cnx:
                cursor = cnx.cursor()
                
                # 确保使用正确的数据库（如果连接时没有指定）
                try:
                    cursor.execute(f"USE `{database}`")
                except Exception as e:
                    print(f"[DataSourceManager] USE数据库失败: {e}")
                
                # 先尝试使用 information_schema
                try:
                    cursor.execute("""
                        SELECT 
                            TABLE_NAME as name,
                            TABLE_TYPE as type,
                            TABLE_ROWS as rows,
                            TABLE_COMMENT as comment
                        FROM information_schema.TABLES 
                        WHERE TABLE_SCHEMA = %s
                        ORDER BY TABLE_NAME
                    """, (database,))
                    
                    columns = [desc[0] for desc in cursor.description]
                    tables = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    cursor.close()
                    print(f"[DataSourceManager] 使用information_schema找到 {len(tables)} 个表")
                    return tables
                except Exception as e:
                    print(f"[DataSourceManager] information_schema查询失败，尝试SHOW TABLES: {e}")
                    cursor.close()
                    
                    # 重新获取连接并尝试 SHOW TABLES
                    with self.get_connection(ds_id, database) as cnx2:
                        cursor2 = cnx2.cursor()
                        try:
                            cursor2.execute(f"SHOW TABLES FROM `{database}`")
                        except Exception as e2:
                            print(f"[DataSourceManager] SHOW TABLES FROM失败，尝试USE后SHOW TABLES: {e2}")
                            cursor2.execute(f"USE `{database}`")
                            cursor2.execute("SHOW TABLES")
                        
                        tables = []
                        for row in cursor2.fetchall():
                            table_name = row[0] if isinstance(row, tuple) else row
                            tables.append({
                                'name': table_name,
                                'type': 'BASE TABLE',
                                'rows': 0,
                                'comment': ''
                            })
                        cursor2.close()
                        print(f"[DataSourceManager] 使用SHOW TABLES找到 {len(tables)} 个表")
                        return tables
        except Exception as e:
            print(f"[DataSourceManager] 获取表列表失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_table_ddl(self, ds_id: str, database: str, table: str, max_retries: int = 2) -> str:
        """
        获取表的DDL语句（带超时和重试机制）
        
        Args:
            ds_id: 数据源ID
            database: 数据库名
            table: 表名
            max_retries: 最大重试次数，默认2次
            
        Returns:
            CREATE TABLE DDL语句
        """
        for attempt in range(max_retries + 1):
            try:
                retry_msg = f" (重试 {attempt}/{max_retries})" if attempt > 0 else ""
                print(f"[DataSourceManager] 获取表DDL: {database}.{table}{retry_msg}")
                
                with self.get_connection(ds_id, database) as cnx:
                    cursor = None
                    try:
                        cursor = cnx.cursor()
                        
                        # 尝试使用完整表名（连接配置中已设置read_timeout=30秒）
                        try:
                            cursor.execute(f"SHOW CREATE TABLE `{database}`.`{table}`")
                        except Exception as e1:
                            error_msg = str(e1).lower()
                            if "doesn't exist" in error_msg or "不存在" in error_msg or "table" in error_msg and "exist" in error_msg:
                                print(f"[DataSourceManager] 表不存在: {database}.{table}")
                                return ""
                            
                            print(f"[DataSourceManager] 使用完整表名失败，尝试不指定数据库: {e1}")
                            # 如果失败，尝试不指定数据库（因为连接时已经指定了）
                            try:
                                cursor.execute(f"SHOW CREATE TABLE `{table}`")
                            except Exception as e2:
                                error_msg2 = str(e2).lower()
                                if "doesn't exist" in error_msg2 or "不存在" in error_msg2:
                                    print(f"[DataSourceManager] 表不存在: {database}.{table}")
                                    return ""
                                
                                # 检查是否是超时错误
                                if "timeout" in error_msg2 or "timed out" in error_msg2 or "read timeout" in error_msg2:
                                    if attempt < max_retries:
                                        print(f"[DataSourceManager] 查询超时，将重试 ({attempt+1}/{max_retries})")
                                        import time
                                        time.sleep(2)  # 超时后等待2秒再重试
                                        continue
                                    else:
                                        print(f"[DataSourceManager] 查询超时（已重试{max_retries}次），跳过: {database}.{table}")
                                        return ""
                                
                                if attempt < max_retries:
                                    print(f"[DataSourceManager] 获取DDL失败，将重试: {e2}")
                                    import time
                                    time.sleep(1)
                                    continue
                                else:
                                    print(f"[DataSourceManager] 获取DDL失败（已重试{max_retries}次）: {e2}")
                                    return ""
                        
                        result = cursor.fetchone()
                        if result and len(result) > 1:
                            ddl = result[1]
                            print(f"[DataSourceManager] 成功获取DDL: {database}.{table} (长度: {len(ddl)})")
                            return ddl
                        else:
                            print(f"[DataSourceManager] 警告: DDL结果为空: {database}.{table}")
                            return ""
                            
                    finally:
                        if cursor:
                            try:
                                cursor.close()
                            except:
                                pass
                        
            except Exception as e:
                error_msg = str(e).lower()
                # 检查是否是超时错误
                if "timeout" in error_msg or "timed out" in error_msg or "read timeout" in error_msg:
                    if attempt < max_retries:
                        print(f"[DataSourceManager] 连接/查询超时，将重试 ({attempt+1}/{max_retries}): {e}")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        print(f"[DataSourceManager] 连接/查询超时（已重试{max_retries}次），跳过: {database}.{table}")
                        return ""
                
                if attempt < max_retries:
                    print(f"[DataSourceManager] 获取DDL异常，将重试 ({attempt+1}/{max_retries}): {e}")
                    import time
                    time.sleep(1)
                    continue
                else:
                    print(f"[DataSourceManager] 获取DDL异常（已重试{max_retries}次）: {e}")
                    import traceback
                    traceback.print_exc()
                    return ""
        
        return ""
    
    def get_all_tables_by_datasource(self, ds_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取数据源下所有选中数据库的表信息
        
        Args:
            ds_id: 数据源ID
            
        Returns:
            {database_name: [table_info, ...], ...}
        """
        ds = self.get(ds_id)
        if not ds:
            return {}
        
        result = {}
        for db in ds.databases:
            try:
                result[db] = self.get_tables(ds_id, db)
            except Exception as e:
                print(f"[DataSource] 获取 {db} 表列表失败: {e}")
                result[db] = []
        
        return result
    
    def initialize_all(self):
        """
        初始化所有数据源连接
        在服务启动时调用
        """
        print("[DataSource] 开始初始化所有数据源...")
        
        datasources = self.list_all()
        success_count = 0
        
        for ds in datasources:
            result = self.test_connection(ds.id)
            if result['success']:
                self._update_connection_pool(ds)
                success_count += 1
                print(f"  ✓ {ds.name} ({ds.host})")
            else:
                print(f"  ✗ {ds.name} ({ds.host}): {result['message']}")
        
        print(f"[DataSource] 初始化完成: {success_count}/{len(datasources)} 个数据源可用")
        
        # 如果没有激活的数据源，激活第一个可用的
        if not self._active_datasource_id and success_count > 0:
            for ds in datasources:
                if ds.status == 'active' or ds.id == 'default':
                    self._active_datasource_id = ds.id
                    break


# 全局数据源管理器实例
_datasource_manager: Optional[DataSourceManager] = None


def get_datasource_manager(chroma_client=None) -> DataSourceManager:
    """获取全局数据源管理器实例"""
    global _datasource_manager
    if _datasource_manager is None:
        if chroma_client is None:
            raise ValueError("首次调用必须提供 chroma_client")
        _datasource_manager = DataSourceManager(chroma_client)
    return _datasource_manager


def init_datasource_manager(chroma_client) -> DataSourceManager:
    """初始化全局数据源管理器"""
    global _datasource_manager
    _datasource_manager = DataSourceManager(chroma_client)
    _datasource_manager.initialize_all()
    return _datasource_manager
