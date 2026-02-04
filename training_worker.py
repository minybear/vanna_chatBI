"""
异步训练任务处理器
支持后台训练、进度追踪、错误处理
"""
import os
import csv
import asyncio
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from knowledge_base_manager import (
    KnowledgeBaseManager,
    TrainingTask,
    get_kb_manager
)


def _read_training_csv(file_path: str) -> pd.DataFrame:
    """
    读取训练用 CSV，兼容多行内容、引号内逗号、以及「业务标签」列未加引号含逗号的情况。
    若某行被解析为超过 4 列，将第 5 列及之后合并回第 4 列（业务标签）。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows: List[List[str]] = []
        for row in reader:
            if len(row) > 4:
                # 业务标签列未加引号导致被拆成多列，合并回第 4 列
                row = row[:3] + [",".join(row[3:])]
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["类型", "内容", "用户问题示例", "业务标签"])
    return pd.DataFrame(rows[1:], columns=rows[0])


class TrainingWorker:
    """
    异步训练任务处理器
    使用线程池执行训练任务，支持进度回调
    """
    
    # 单条语料训练超时（秒），防止 embedding API 无响应导致任务一直阻塞
    ROW_TRAINING_TIMEOUT = 120

    def __init__(self, kb_manager: KnowledgeBaseManager, max_workers: int = 2):
        """
        初始化训练处理器
        
        Args:
            kb_manager: 知识库管理器实例
            max_workers: 最大并发训练任务数
        """
        self.kb_manager = kb_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._timeout_executor = ThreadPoolExecutor(max_workers=4)  # 用于带超时的单条训练
        self._running_tasks: Dict[str, bool] = {}  # task_id -> is_running
    
    def start_training(self, kb_id: str, file_path: str, 
                       progress_callback: Optional[Callable] = None) -> TrainingTask:
        """
        启动异步训练任务
        
        Args:
            kb_id: 知识库ID
            file_path: CSV文件路径
            progress_callback: 进度回调函数 (task_id, processed, total, status)
            
        Returns:
            创建的训练任务
        """
        # 读取文件获取记录数（兼容多行、引号内逗号、业务标签未引号含逗号）
        try:
            df = _read_training_csv(file_path)
            total_records = len(df)
        except Exception as e:
            raise ValueError(f"无法读取文件: {e}")
        
        # 创建训练任务
        file_name = os.path.basename(file_path)
        task = self.kb_manager.create_training_task(kb_id, file_name, total_records)
        
        # 更新知识库状态为训练中
        self.kb_manager.update(kb_id, status='training')
        
        # 提交到线程池执行
        self._running_tasks[task.id] = True
        self.executor.submit(
            self._execute_training,
            task.id,
            kb_id,
            file_path,
            progress_callback
        )
        
        return task
    
    def _execute_training(self, task_id: str, kb_id: str, file_path: str,
                          progress_callback: Optional[Callable] = None):
        """
        执行训练任务（在线程池中运行）
        
        Args:
            task_id: 任务ID
            kb_id: 知识库ID
            file_path: CSV文件路径
            progress_callback: 进度回调函数
        """
        try:
            # 更新任务状态为处理中
            self.kb_manager.update_training_task(task_id, status='processing')
            
            # 读取CSV文件（兼容多行、引号内逗号、业务标签未引号含逗号）
            df = _read_training_csv(file_path)
            total_records = len(df)
            
            # 显示CSV列名
            print(f"[Training] CSV列名: {df.columns.tolist()}")
            
            processed = 0
            success = 0
            fail = 0
            cancelled = False
            
            for idx, row in df.iterrows():
                # 检查是否被取消
                if not self._running_tasks.get(task_id, False):
                    self.kb_manager.update_training_task(
                        task_id,
                        status='cancelled',
                        error_message='任务已取消'
                    )
                    self.kb_manager.update(kb_id, status='ready')
                    cancelled = True
                    break
                
                try:
                    # 解析训练数据
                    data_type = self._get_data_type(row)
                    content = self._get_content(row)
                    question = self._get_question(row)
                    
                    if not content:
                        print(f"[Training] 跳过空内容记录: {idx+1}")
                        fail += 1
                        processed += 1
                        continue
                    
                    # 单条训练带超时，防止 embedding API 无响应导致任务一直阻塞
                    future = self._timeout_executor.submit(
                        self.kb_manager.add_training_data,
                        kb_id=kb_id,
                        data_type=data_type,
                        content=content,
                        question=question,
                    )
                    try:
                        future.result(timeout=self.ROW_TRAINING_TIMEOUT)
                    except FuturesTimeoutError:
                        fail += 1
                        processed += 1
                        print(f"[Training] [{idx+1}/{total_records}] 单条训练超时({self.ROW_TRAINING_TIMEOUT}s)，已跳过")
                        self.kb_manager.update_training_task(
                            task_id,
                            processed_records=processed,
                            success_count=success,
                            fail_count=fail,
                        )
                        continue
                    
                    success += 1
                    print(f"[Training] [{idx+1}/{total_records}] 训练成功: {data_type}")
                    
                except Exception as e:
                    fail += 1
                    print(f"[Training] [{idx+1}/{total_records}] 训练失败: {e}")
                
                processed += 1
                
                # 更新进度
                self.kb_manager.update_training_task(
                    task_id,
                    processed_records=processed,
                    success_count=success,
                    fail_count=fail
                )
                
                # 调用进度回调
                if progress_callback:
                    try:
                        progress_callback(task_id, processed, total_records, 'processing')
                    except Exception:
                        pass
            
            if cancelled:
                print(f"[Training] 任务已取消: {task_id}")
                return
            
            # 训练完成
            final_status = 'completed' if fail == 0 else 'completed'
            error_msg = None if fail == 0 else f'部分记录训练失败: {fail}/{total_records}'
            
            self.kb_manager.update_training_task(
                task_id,
                status=final_status,
                error_message=error_msg
            )
            
            # 更新知识库状态
            self.kb_manager.update(kb_id, status='ready')
            
            print(f"[Training] 训练完成: 成功 {success}, 失败 {fail}, 总计 {total_records}")
            
            if progress_callback:
                try:
                    progress_callback(task_id, processed, total_records, final_status)
                except Exception:
                    pass
            
        except Exception as e:
            error_msg = f"训练出错: {str(e)}"
            print(f"[Training] {error_msg}")
            traceback.print_exc()
            
            self.kb_manager.update_training_task(
                task_id,
                status='failed',
                error_message=error_msg
            )
            
            # 更新知识库状态
            self.kb_manager.update(kb_id, status='error')
            
            if progress_callback:
                try:
                    progress_callback(task_id, 0, 0, 'failed')
                except Exception:
                    pass
        
        finally:
            # 清理任务状态
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
            
            # 删除临时文件
            try:
                if os.path.exists(file_path) and '/tmp/' in file_path or '\\tmp\\' in file_path:
                    os.remove(file_path)
            except Exception:
                pass
    
    def _get_data_type(self, row: pd.Series) -> str:
        """从行数据中提取数据类型"""
        # 尝试多种列名
        for col in ['类型', 'type', 'data_type', 'training_data_type']:
            if col in row and pd.notna(row[col]):
                dtype = str(row[col]).lower()
                if dtype in ['ddl', 'sql', 'documentation', 'doc']:
                    if dtype == 'doc':
                        return 'documentation'
                    return dtype
        
        # 默认为documentation
        return 'documentation'
    
    def _get_content(self, row: pd.Series) -> str:
        """从行数据中提取内容"""
        for col in ['内容', 'content', 'sql', 'ddl', 'documentation']:
            if col in row and pd.notna(row[col]):
                return str(row[col]).strip()
        return ''
    
    def _get_question(self, row: pd.Series) -> Optional[str]:
        """从行数据中提取问题"""
        for col in ['用户问题示例', 'question', '问题', 'user_question']:
            if col in row and pd.notna(row[col]):
                return str(row[col]).strip()
        return None
    
    def cancel_training(self, task_id: str) -> bool:
        """
        取消训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否取消成功
        """
        if task_id in self._running_tasks:
            self._running_tasks[task_id] = False
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务状态"""
        return self.kb_manager.get_training_task(task_id)
    
    def is_task_running(self, task_id: str) -> bool:
        """检查任务是否正在运行"""
        return self._running_tasks.get(task_id, False)
    
    def shutdown(self):
        """关闭训练处理器"""
        # 取消所有运行中的任务
        for task_id in list(self._running_tasks.keys()):
            self._running_tasks[task_id] = False
        
        # 关闭线程池
        self.executor.shutdown(wait=False)
        self._timeout_executor.shutdown(wait=False)


# 异步训练函数（供FastAPI BackgroundTasks使用）
async def train_knowledge_base_async(kb_id: str, file_path: str,
                                     kb_manager: KnowledgeBaseManager) -> str:
    """
    异步训练知识库（协程版本）
    
    Args:
        kb_id: 知识库ID
        file_path: CSV文件路径
        kb_manager: 知识库管理器
        
    Returns:
        任务ID
    """
    worker = TrainingWorker(kb_manager)
    task = worker.start_training(kb_id, file_path)
    return task.id


# 全局训练处理器实例
_training_worker: Optional[TrainingWorker] = None


def get_training_worker(kb_manager: KnowledgeBaseManager = None) -> TrainingWorker:
    """获取全局训练处理器实例"""
    global _training_worker
    if _training_worker is None:
        if kb_manager is None:
            kb_manager = get_kb_manager()
        _training_worker = TrainingWorker(kb_manager)
    return _training_worker


def init_training_worker(kb_manager: KnowledgeBaseManager) -> TrainingWorker:
    """初始化全局训练处理器"""
    global _training_worker
    _training_worker = TrainingWorker(kb_manager)
    return _training_worker
