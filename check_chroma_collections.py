#!/usr/bin/env python3
"""
检测并可选清理 ChromaDB 中异常的 collection。

用法:
  python check_chroma_collections.py              # 仅检测，列出每个 collection 状态
  python check_chroma_collections.py --clean      # 删除检测到异常的 collection（会提示确认）
  python check_chroma_collections.py --clean-all  # 不确认，直接删除所有异常 collection

环境变量:
  CHROMA_DB_PATH  持久化目录，默认项目下的 chroma_db
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def get_chroma_path():
    return os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH).strip() or DEFAULT_CHROMA_PATH


def main():
    parser = argparse.ArgumentParser(description="检测并可选清理 ChromaDB 异常 collection")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="删除检测到异常的 collection（会逐项确认）",
    )
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="删除所有异常 collection，不确认",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="ChromaDB 持久化目录（覆盖 CHROMA_DB_PATH）",
    )
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
        client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as e:
        print(f"[错误] 无法连接 ChromaDB ({path}): {e}")
        sys.exit(1)

    collections = client.list_collections()
    if not collections:
        print(f"[信息] 目录下没有 collection: {path}")
        return

    results = []
    for coll in collections:
        name = coll.name
        try:
            # 触发一次读，异常 collection 会在这里报错（如 compaction/metadata 错误）
            data = coll.get(limit=1)
            count = len(data.get("ids") or [])
            # 尝试获取总数：部分版本支持 peek/count
            try:
                total = coll.count()
            except Exception:
                total = None
            status = "ok"
            msg = str(total) if total is not None else f"可读(limit=1 返回 {count} 条)"
        except Exception as e:
            status = "error"
            msg = str(e)
        results.append((name, status, msg))

    print(f"ChromaDB 路径: {path}")
    print(f"共 {len(results)} 个 collection:\n")
    for name, status, msg in results:
        icon = "✓" if status == "ok" else "✗"
        print(f"  {icon} {name}: {status} - {msg}")

    errors = [r for r in results if r[1] == "error"]
    if not errors:
        print("\n[结果] 所有 collection 检测正常。")
        return

    print(f"\n[结果] 发现 {len(errors)} 个异常 collection: {[r[0] for r in errors]}")

    if not (args.clean or args.clean_all):
        print("如需删除异常 collection 后由应用自动重建空库，请使用: --clean 或 --clean-all")
        return

    for name, status, msg in errors:
        if args.clean_all:
            do_delete = True
        else:
            try:
                ans = input(f"是否删除异常 collection [{name}]? (y/N): ").strip().lower()
                do_delete = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                do_delete = False
        if do_delete:
            try:
                client.delete_collection(name=name)
                print(f"  已删除: {name}")
            except Exception as e:
                print(f"  删除失败 [{name}]: {e}")
    print("完成。")


if __name__ == "__main__":
    main()
