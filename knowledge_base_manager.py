"""
知识库管理模块
支持多知识库的CRUD、ChromaDB collection隔离、训练数据管理
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class KnowledgeBase:
    """知识库配置数据类"""
    id: str
    name: str
    description: str
    datasource_ids: List[str]  # 关联的数据源ID列表（支持多个）
    collection_prefix: str  # ChromaDB collection前缀
    status: str  # ready/training/error
    ddl_count: int = 0
    sql_count: int = 0
    doc_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBase':
        """从字典创建实例"""
        # 兼容旧数据：datasource_id (单个) -> datasource_ids (列表)
        datasource_ids = data.get('datasource_ids', [])
        if not datasource_ids and data.get('datasource_id'):
            datasource_ids = [data.get('datasource_id')]
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            datasource_ids=datasource_ids,
            collection_prefix=data.get('collection_prefix', ''),
            status=data.get('status', 'ready'),
            ddl_count=data.get('ddl_count', 0),
            sql_count=data.get('sql_count', 0),
            doc_count=data.get('doc_count', 0),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', '')
        )


@dataclass
class TrainingTask:
    """训练任务数据类"""
    id: str
    kb_id: str
    file_name: str
    total_records: int
    processed_records: int
    success_count: int
    fail_count: int
    status: str  # pending/processing/completed/failed
    error_message: Optional[str]
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingTask':
        """从字典创建实例"""
        return cls(
            id=data.get('id', ''),
            kb_id=data.get('kb_id', ''),
            file_name=data.get('file_name', ''),
            total_records=data.get('total_records', 0),
            processed_records=data.get('processed_records', 0),
            success_count=data.get('success_count', 0),
            fail_count=data.get('fail_count', 0),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message'),
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


class KnowledgeBaseManager:
    """
    知识库管理器
    负责知识库的CRUD操作、collection隔离、训练数据管理
    """
    
    # 默认知识库ID
    DEFAULT_KB_ID = "default"
    DEFAULT_KB_NAME = "默认知识库"
    DEFAULT_COLLECTION_PREFIX = ""  # 空前缀，使用原始collection名
    
    def __init__(self, chroma_client, embedding_function=None):
        """
        初始化知识库管理器
        
        Args:
            chroma_client: ChromaDB 客户端实例
            embedding_function: 用于训练数据的嵌入函数
        """
        self.chroma_client = chroma_client
        self.training_embedding_function = embedding_function
        self.meta_embedding_function = SimpleEmbeddingFunction()
        
        # 创建或获取知识库元数据集合
        self.kb_collection = chroma_client.get_or_create_collection(
            name="knowledge_bases",
            embedding_function=self.meta_embedding_function,
        )
        
        # 创建或获取训练任务集合
        self.tasks_collection = chroma_client.get_or_create_collection(
            name="training_tasks",
            embedding_function=self.meta_embedding_function,
        )
        
        # 知识库collection缓存
        self._kb_collections: Dict[str, Dict[str, Any]] = {}
        
        # 确保默认知识库存在
        self._ensure_default_kb()
    
    def _ensure_default_kb(self):
        """确保默认知识库存在"""
        existing = self.get(self.DEFAULT_KB_ID)
        if not existing:
            # 创建默认知识库记录
            now = self._now()
            default_kb = KnowledgeBase(
                id=self.DEFAULT_KB_ID,
                name=self.DEFAULT_KB_NAME,
                description="系统默认知识库，使用原始的sql/ddl/documentation集合",
                datasource_ids=[],  # 默认知识库不关联特定数据源
                collection_prefix=self.DEFAULT_COLLECTION_PREFIX,
                status='ready',
                created_at=now,
                updated_at=now
            )
            self._save_to_chromadb(default_kb)
            print(f"[KnowledgeBase] 创建默认知识库")
    
    def _now(self) -> str:
        """获取当前时间ISO格式"""
        return datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"kb-{uuid.uuid4().hex[:12]}"
    
    def _generate_task_id(self) -> str:
        """生成训练任务ID"""
        return f"task-{uuid.uuid4().hex[:12]}"
    
    def _get_collection_name(self, kb: KnowledgeBase, collection_type: str) -> str:
        """
        获取知识库的collection名称
        
        Args:
            kb: 知识库对象
            collection_type: sql/ddl/documentation
            
        Returns:
            collection名称
        """
        if kb.collection_prefix:
            return f"{kb.collection_prefix}{collection_type}"
        return collection_type
    
    def create(self, name: str, description: str = "", datasource_ids: Optional[List[str]] = None) -> KnowledgeBase:
        """
        创建新知识库
        
        Args:
            name: 知识库名称
            description: 描述
            datasource_ids: 关联的数据源ID列表
            
        Returns:
            创建的知识库对象
            
        Raises:
            ValueError: 名称重复
        """
        # 检查名称是否重复
        existing = self.get_by_name(name)
        if existing:
            raise ValueError(f"知识库名称 '{name}' 已存在")
        
        kb_id = self._generate_id()
        # 生成collection前缀（使用ID确保唯一性）
        collection_prefix = f"kb_{kb_id.replace('-', '_')}_"
        
        now = self._now()
        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            datasource_ids=datasource_ids or [],
            collection_prefix=collection_prefix,
            status='ready',
            created_at=now,
            updated_at=now
        )
        
        # 存储到 ChromaDB
        self._save_to_chromadb(kb)
        
        # 创建对应的training data collections
        self._get_or_create_kb_collections(kb)
        
        print(f"[KnowledgeBase] 创建知识库: {kb.name} ({kb.id})")
        return kb
    
    def _save_to_chromadb(self, kb: KnowledgeBase):
        """保存知识库配置到 ChromaDB"""
        doc = json.dumps(kb.to_dict(), ensure_ascii=False)
        
        # 检查是否已存在
        existing = self.kb_collection.get(ids=[kb.id])
        # 将 datasource_ids 列表转为 JSON 字符串存储在 metadata 中
        ds_ids_str = json.dumps(kb.datasource_ids) if kb.datasource_ids else '[]'
        
        if existing and existing.get('ids'):
            # 更新
            self.kb_collection.update(
                ids=[kb.id],
                documents=[doc],
                metadatas=[{
                    'name': kb.name,
                    'status': kb.status,
                    'datasource_ids': ds_ids_str,
                    'updated_at': kb.updated_at
                }]
            )
        else:
            # 新增
            self.kb_collection.add(
                ids=[kb.id],
                documents=[doc],
                metadatas=[{
                    'name': kb.name,
                    'status': kb.status,
                    'datasource_ids': ds_ids_str,
                    'updated_at': kb.updated_at
                }]
            )
    
    def _get_or_create_kb_collections(self, kb: KnowledgeBase) -> Dict[str, Any]:
        """
        获取或创建知识库的training data collections
        
        Args:
            kb: 知识库对象
            
        Returns:
            包含sql/ddl/documentation collection的字典
        """
        if kb.id in self._kb_collections:
            return self._kb_collections[kb.id]
        
        collections = {}
        for ctype in ['sql', 'ddl', 'documentation']:
            cname = self._get_collection_name(kb, ctype)
            collections[ctype] = self.chroma_client.get_or_create_collection(
                name=cname,
                embedding_function=self.training_embedding_function,
            )
        
        self._kb_collections[kb.id] = collections
        return collections
    
    def get(self, kb_id: str) -> Optional[KnowledgeBase]:
        """
        根据ID获取知识库
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            知识库对象，不存在返回None
        """
        data = self.kb_collection.get(ids=[kb_id])
        if not data or not data.get('ids'):
            return None
        
        doc = data.get('documents', [None])[0]
        if not doc:
            return None
        
        try:
            kb_dict = json.loads(doc)
            kb = KnowledgeBase.from_dict(kb_dict)
            # 更新统计信息
            self._update_kb_stats(kb)
            return kb
        except Exception as e:
            print(f"[KnowledgeBase] 解析知识库失败: {e}")
            return None
    
    def _update_kb_stats(self, kb: KnowledgeBase):
        """更新知识库的统计信息"""
        try:
            collections = self._get_or_create_kb_collections(kb)
            kb.sql_count = collections['sql'].count()
            kb.ddl_count = collections['ddl'].count()
            kb.doc_count = collections['documentation'].count()
        except Exception as e:
            print(f"[KnowledgeBase] 更新统计信息失败: {e}")
    
    def get_by_name(self, name: str) -> Optional[KnowledgeBase]:
        """根据名称获取知识库"""
        data = self.kb_collection.get(where={"name": name})
        if not data or not data.get('ids'):
            return None
        
        doc = data.get('documents', [None])[0]
        if not doc:
            return None
        
        try:
            kb_dict = json.loads(doc)
            return KnowledgeBase.from_dict(kb_dict)
        except Exception:
            return None
    
    def list_all(self) -> List[KnowledgeBase]:
        """
        获取所有知识库列表
        
        Returns:
            知识库列表（按更新时间倒序）
        """
        knowledge_bases = []
        
        # 从 ChromaDB 获取所有知识库
        data = self.kb_collection.get()
        if data and data.get('ids'):
            for doc in data.get('documents', []):
                try:
                    kb_dict = json.loads(doc)
                    kb = KnowledgeBase.from_dict(kb_dict)
                    self._update_kb_stats(kb)
                    knowledge_bases.append(kb)
                except Exception as e:
                    print(f"[KnowledgeBase] 解析知识库失败: {e}")
        
        # 按更新时间倒序排序
        knowledge_bases.sort(key=lambda x: x.updated_at or '', reverse=True)
        return knowledge_bases
    
    def update(self, kb_id: str, **kwargs) -> Optional[KnowledgeBase]:
        """
        更新知识库
        
        Args:
            kb_id: 知识库ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的知识库对象
        """
        kb = self.get(kb_id)
        if not kb:
            return None
        
        # 更新字段
        if 'name' in kwargs:
            # 检查名称是否重复
            existing = self.get_by_name(kwargs['name'])
            if existing and existing.id != kb_id:
                raise ValueError(f"知识库名称 '{kwargs['name']}' 已存在")
            kb.name = kwargs['name']
        
        if 'description' in kwargs:
            kb.description = kwargs['description']
        if 'datasource_ids' in kwargs:
            kb.datasource_ids = kwargs['datasource_ids'] or []
        if 'status' in kwargs:
            kb.status = kwargs['status']
        
        kb.updated_at = self._now()
        
        # 保存更新
        self._save_to_chromadb(kb)
        
        print(f"[KnowledgeBase] 更新知识库: {kb.name} ({kb.id})")
        return kb
    
    def delete(self, kb_id: str) -> bool:
        """
        删除知识库
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            是否删除成功
        """
        if kb_id == self.DEFAULT_KB_ID:
            raise ValueError("默认知识库不能删除")
        
        kb = self.get(kb_id)
        if not kb:
            return False
        
        # 删除对应的training data collections
        for ctype in ['sql', 'ddl', 'documentation']:
            cname = self._get_collection_name(kb, ctype)
            try:
                self.chroma_client.delete_collection(name=cname)
            except Exception as e:
                print(f"[KnowledgeBase] 删除collection {cname} 失败: {e}")
        
        # 从缓存移除
        if kb_id in self._kb_collections:
            del self._kb_collections[kb_id]
        
        # 从元数据collection删除
        self.kb_collection.delete(ids=[kb_id])
        
        print(f"[KnowledgeBase] 删除知识库: {kb.name} ({kb_id})")
        return True
    
    def get_training_data(self, kb_id: str) -> pd.DataFrame:
        """
        获取知识库的训练数据
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            训练数据DataFrame
        """
        kb = self.get(kb_id)
        if not kb:
            return pd.DataFrame()
        
        collections = self._get_or_create_kb_collections(kb)
        
        df = pd.DataFrame()
        
        # SQL数据
        sql_data = collections['sql'].get()
        if sql_data and sql_data.get('ids'):
            documents = []
            for doc in sql_data['documents']:
                try:
                    documents.append(json.loads(doc))
                except:
                    documents.append({'question': '', 'sql': doc})
            
            df_sql = pd.DataFrame({
                'id': sql_data['ids'],
                'question': [doc.get('question', '') for doc in documents],
                'content': [doc.get('sql', '') for doc in documents],
                'training_data_type': 'sql'
            })
            df = pd.concat([df, df_sql])
        
        # DDL数据
        ddl_data = collections['ddl'].get()
        if ddl_data and ddl_data.get('ids'):
            df_ddl = pd.DataFrame({
                'id': ddl_data['ids'],
                'question': [None] * len(ddl_data['ids']),
                'content': ddl_data['documents'],
                'training_data_type': 'ddl'
            })
            df = pd.concat([df, df_ddl])
        
        # Documentation数据
        doc_data = collections['documentation'].get()
        if doc_data and doc_data.get('ids'):
            df_doc = pd.DataFrame({
                'id': doc_data['ids'],
                'question': [None] * len(doc_data['ids']),
                'content': doc_data['documents'],
                'training_data_type': 'documentation'
            })
            df = pd.concat([df, df_doc])
        
        return df
    
    def add_training_data(self, kb_id: str, data_type: str, content: str, 
                          question: Optional[str] = None) -> str:
        """
        添加训练数据到知识库
        
        Args:
            kb_id: 知识库ID
            data_type: 数据类型 (sql/ddl/documentation)
            content: 内容
            question: 问题（仅sql类型需要）
            
        Returns:
            训练数据ID
        """
        kb = self.get(kb_id)
        if not kb:
            raise ValueError(f"知识库 {kb_id} 不存在")
        
        collections = self._get_or_create_kb_collections(kb)
        
        # 生成ID（与内容一一对应，重复上传相同语料会得到相同 ID）
        from vanna.legacy.utils import deterministic_uuid
        
        if data_type == 'sql':
            if not question:
                raise ValueError("SQL类型必须提供question")
            doc = json.dumps({'question': question, 'sql': content}, ensure_ascii=False)
            doc_id = deterministic_uuid(doc) + "-sql"
            coll = collections['sql']
        elif data_type == 'ddl':
            doc_id = deterministic_uuid(content) + "-ddl"
            coll = collections['ddl']
        elif data_type == 'documentation':
            doc_id = deterministic_uuid(content) + "-doc"
            coll = collections['documentation']
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        # 已存在则跳过，避免重复调用 embedding 和写入（去重）
        existing = coll.get(ids=[doc_id])
        if existing and existing.get('ids'):
            return doc_id
        
        if data_type == 'sql':
            coll.add(documents=[doc], ids=[doc_id])
        else:
            coll.add(documents=[content], ids=[doc_id])
        
        # 更新知识库统计
        kb.updated_at = self._now()
        self._update_kb_stats(kb)
        self._save_to_chromadb(kb)
        
        return doc_id
    
    def remove_training_data(self, kb_id: str, data_id: str) -> bool:
        """
        从知识库删除训练数据
        
        Args:
            kb_id: 知识库ID
            data_id: 训练数据ID
            
        Returns:
            是否删除成功
        """
        kb = self.get(kb_id)
        if not kb:
            return False
        
        collections = self._get_or_create_kb_collections(kb)
        
        if data_id.endswith("-sql"):
            collections['sql'].delete(ids=[data_id])
        elif data_id.endswith("-ddl"):
            collections['ddl'].delete(ids=[data_id])
        elif data_id.endswith("-doc"):
            collections['documentation'].delete(ids=[data_id])
        else:
            return False
        
        # 更新知识库统计
        kb.updated_at = self._now()
        self._update_kb_stats(kb)
        self._save_to_chromadb(kb)
        
        return True
    
    # 训练任务管理
    
    def create_training_task(self, kb_id: str, file_name: str, total_records: int) -> TrainingTask:
        """创建训练任务"""
        now = self._now()
        task = TrainingTask(
            id=self._generate_task_id(),
            kb_id=kb_id,
            file_name=file_name,
            total_records=total_records,
            processed_records=0,
            success_count=0,
            fail_count=0,
            status='pending',
            error_message=None,
            created_at=now,
            updated_at=now
        )
        
        self._save_task(task)
        return task
    
    def _save_task(self, task: TrainingTask):
        """保存训练任务"""
        doc = json.dumps(task.to_dict(), ensure_ascii=False)
        
        existing = self.tasks_collection.get(ids=[task.id])
        if existing and existing.get('ids'):
            self.tasks_collection.update(
                ids=[task.id],
                documents=[doc],
                metadatas=[{
                    'kb_id': task.kb_id,
                    'status': task.status,
                    'updated_at': task.updated_at
                }]
            )
        else:
            self.tasks_collection.add(
                ids=[task.id],
                documents=[doc],
                metadatas=[{
                    'kb_id': task.kb_id,
                    'status': task.status,
                    'updated_at': task.updated_at
                }]
            )
    
    def get_training_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务"""
        data = self.tasks_collection.get(ids=[task_id])
        if not data or not data.get('ids'):
            return None
        
        doc = data.get('documents', [None])[0]
        if not doc:
            return None
        
        try:
            return TrainingTask.from_dict(json.loads(doc))
        except Exception:
            return None
    
    def get_tasks_by_kb(self, kb_id: str) -> List[TrainingTask]:
        """获取知识库的所有训练任务"""
        data = self.tasks_collection.get(where={"kb_id": kb_id})
        if not data or not data.get('ids'):
            return []
        
        tasks = []
        for doc in data.get('documents', []):
            try:
                tasks.append(TrainingTask.from_dict(json.loads(doc)))
            except Exception:
                pass
        
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks
    
    def update_training_task(self, task_id: str, **kwargs) -> Optional[TrainingTask]:
        """更新训练任务"""
        task = self.get_training_task(task_id)
        if not task:
            return None
        
        if 'processed_records' in kwargs:
            task.processed_records = kwargs['processed_records']
        if 'success_count' in kwargs:
            task.success_count = kwargs['success_count']
        if 'fail_count' in kwargs:
            task.fail_count = kwargs['fail_count']
        if 'status' in kwargs:
            task.status = kwargs['status']
        if 'error_message' in kwargs:
            task.error_message = kwargs['error_message']
        
        task.updated_at = self._now()
        self._save_task(task)
        return task
    
    def delete_training_task(self, task_id: str) -> bool:
        """
        从 ChromaDB 删除训练任务记录，并将对应知识库状态从 training 恢复为 ready。
        
        Returns:
            是否删除成功（任务存在则 True）
        """
        task = self.get_training_task(task_id)
        if not task:
            return False
        self.tasks_collection.delete(ids=[task_id])
        kb = self.get(task.kb_id)
        if kb and kb.status == "training":
            self.update(task.kb_id, status="ready")
        return True
    
    def get_kb_collections(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """
        获取知识库的ChromaDB collections
        用于直接访问训练数据进行RAG检索
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            包含sql/ddl/documentation collection的字典
        """
        kb = self.get(kb_id)
        if not kb:
            return None
        return self._get_or_create_kb_collections(kb)


# 全局知识库管理器实例
_kb_manager: Optional[KnowledgeBaseManager] = None


def get_kb_manager(chroma_client=None, embedding_function=None) -> KnowledgeBaseManager:
    """获取全局知识库管理器实例"""
    global _kb_manager
    if _kb_manager is None:
        if chroma_client is None:
            raise ValueError("首次调用必须提供 chroma_client")
        _kb_manager = KnowledgeBaseManager(chroma_client, embedding_function)
    return _kb_manager


def init_kb_manager(chroma_client, embedding_function=None) -> KnowledgeBaseManager:
    """初始化全局知识库管理器"""
    global _kb_manager
    _kb_manager = KnowledgeBaseManager(chroma_client, embedding_function)
    return _kb_manager
