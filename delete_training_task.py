#!/usr/bin/env python3
"""
检查并删除 ChromaDB 中的训练任务记录，并恢复对应知识库状态。

用法:
  python delete_training_task.py task-4cf063f95a39   # 删除指定任务
  python delete_training_task.py                      # 列出默认知识库的训练任务
  python delete_training_task.py --kb default --delete-running  # 删除默认库下所有进行中任务

环境变量: CHROMA_DB_PATH  默认项目下 chroma_db
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def get_chroma_path():
    return os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH).strip() or DEFAULT_CHROMA_PATH


def main():
    parser = argparse.ArgumentParser(description="检查/删除训练任务")
    parser.add_argument("task_id", nargs="?", help="要删除的任务 ID，例如 task-4cf063f95a39")
    parser.add_argument("--kb", default="default", help="知识库 ID，默认 default")
    parser.add_argument("--delete-running", action="store_true", help="删除该知识库下所有 processing/pending 任务")
    parser.add_argument("--path", default=None, help="ChromaDB 目录（覆盖 CHROMA_DB_PATH）")
    args = parser.parse_args()

    path = args.path or get_chroma_path()
    if not os.path.isdir(path):
        print(f"[错误] 目录不存在: {path}")
        sys.exit(1)

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("[错误] 请先安装 chromadb: pip install chromadb")
        sys.exit(1)

    try:
        from knowledge_base_manager import (
            KnowledgeBaseManager,
            SimpleEmbeddingFunction,
        )
    except ImportError as e:
        print(f"[错误] 导入失败: {e}")
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )
    manager = KnowledgeBaseManager(client, SimpleEmbeddingFunction())

    if args.task_id:
        # 检查并删除指定任务
        task_id = args.task_id.strip()
        task = manager.get_training_task(task_id)
        if not task:
            print(f"[检查] 未找到任务: {task_id}")
            sys.exit(0)
        print(f"[检查] 任务存在: {task_id}")
        print(f"  kb_id={task.kb_id}, status={task.status}, processed={task.processed_records}/{task.total_records}")
        manager.delete_training_task(task_id)
        print(f"[删除] 已删除任务 {task_id}，并已将知识库 {task.kb_id} 状态恢复为 ready")
        return

    if args.delete_running:
        tasks = manager.get_tasks_by_kb(args.kb)
        running = [t for t in tasks if t.status in ("processing", "pending")]
        if not running:
            print(f"[检查] 知识库 {args.kb} 下没有进行中的任务")
            return
        for t in running:
            manager.delete_training_task(t.id)
            print(f"[删除] 已删除任务 {t.id} (status={t.status})，知识库已恢复为 ready")
        return

    # 仅列出该知识库的训练任务
    tasks = manager.get_tasks_by_kb(args.kb)
    if not tasks:
        print(f"[检查] 知识库 {args.kb} 下没有训练任务记录")
        return
    print(f"[检查] 知识库 {args.kb} 的训练任务（共 {len(tasks)} 条）:")
    for t in tasks:
        print(f"  {t.id}  status={t.status}  processed={t.processed_records}/{t.total_records}  file={t.file_name}")
    print("\n删除指定任务: python delete_training_task.py <task_id>")
    print("删除所有进行中任务: python delete_training_task.py --kb default --delete-running")


if __name__ == "__main__":
    main()
