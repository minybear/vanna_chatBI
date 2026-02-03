#!/usr/bin/env python3
"""删除 pinned_charts collection：先尝试 API，若 DB 损坏则尝试删对应 segment 目录或提示清空 chroma_db。"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_DB_PATH", os.path.join(BASE, "chroma_db")))


def main():
    # 1) 尝试通过 Chroma API 删除
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        client.delete_collection("pinned_charts")
        print("已通过 API 删除 collection: pinned_charts")
        return
    except Exception as e:
        err_msg = str(e).lower()
        if "malformed" in err_msg or "disk image" in err_msg:
            print("ChromaDB 底层 SQLite 已损坏，无法通过 API 删除。尝试从 SQLite 查出 pinned_charts 的 segment 并删除对应目录...")
        else:
            print(f"API 删除失败: {e}")
            sys.exit(1)

    # 2) 从 SQLite 查询 pinned_charts 的 segment id（可能因损坏而失败）
    sqlite_path = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not os.path.isfile(sqlite_path):
        print("未找到 chroma.sqlite3，无法继续。")
        sys.exit(1)

    try:
        import sqlite3
        conn = sqlite3.connect(sqlite_path)
        cur = conn.execute(
            "select s.id, c.name from segments s join collections c on s.collection=c.id where c.name='pinned_charts'"
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"查询 SQLite 失败（数据库可能已损坏）: {e}")
        print("建议：若可接受丢失当前 Chroma 全部数据，可手动删除整个 chroma_db 目录，重启应用后会重建空库。")
        sys.exit(1)

    if not rows:
        print("未在 SQLite 中找到 pinned_charts 的 segment 记录。")
        print("建议：若仍报错，可手动删除整个 chroma_db 目录后重启应用。")
        return

    # 3) 删除对应 UUID 目录
    for seg_id, cname in rows:
        seg_dir = os.path.join(CHROMA_PATH, seg_id)
        if os.path.isdir(seg_dir):
            import shutil
            shutil.rmtree(seg_dir)
            print(f"已删除 segment 目录: {seg_id} (pinned_charts)")
        else:
            print(f"segment 目录不存在（可能已删）: {seg_dir}")

    print("完成。请重启应用；若仍出现 pinned 相关错误，需考虑删除整个 chroma_db 并重建。")


if __name__ == "__main__":
    main()
