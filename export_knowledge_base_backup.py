#!/usr/bin/env python3
"""
导出所有知识库语料到 JSON 备份文件（仅读取 KB 相关 collection，不触碰 pins/会话）。

用法:
  python export_knowledge_base_backup.py [--output backup.json]

备份文件格式可直接用于重建 ChromaDB 后通过「导入语料」或 restore_knowledge_base_backup.py 恢复。
"""
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def get_chroma_path():
    return os.path.abspath(os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH).strip() or DEFAULT_CHROMA_PATH)


def main():
    parser = argparse.ArgumentParser(description="导出知识库语料到 JSON 备份")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径，默认 knowledge_base_backup_YYYYMMDD_HHMMSS.json")
    parser.add_argument("--path", default=None, help="ChromaDB 目录（覆盖 CHROMA_DB_PATH）")
    args = parser.parse_args()

    path = args.path or get_chroma_path()
    if not os.path.isdir(path):
        print(f"[错误] 目录不存在: {path}")
        return 1

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("[错误] 请先安装: pip install chromadb")
        return 1

    # 只连接 Chroma，不 get_or_create 任何可能损坏的 collection
    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    existing = {c.name for c in client.list_collections()}

    if "knowledge_bases" not in existing:
        print("[信息] 未找到 knowledge_bases collection，可能尚未创建任何知识库。")
        out = args.output or f"knowledge_base_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"knowledge_bases": [], "exported_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        print(f"已写入空备份: {out}")
        return 0

    kb_coll = client.get_collection(name="knowledge_bases")
    kb_data = kb_coll.get()
    if not kb_data or not kb_data.get("ids"):
        print("[信息] 知识库列表为空。")
        out = args.output or f"knowledge_base_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"knowledge_bases": [], "exported_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        print(f"已写入: {out}")
        return 0

    knowledge_bases_backup = []
    for idx, doc in enumerate(kb_data.get("documents") or []):
        try:
            kb = json.loads(doc) if isinstance(doc, str) else doc
        except Exception:
            continue
        kb_id = kb.get("id") or kb_data["ids"][idx]
        name = kb.get("name", "")
        collection_prefix = kb.get("collection_prefix") or ""
        # 与 knowledge_base_manager 命名一致
        sql_name = f"{collection_prefix}sql" if collection_prefix else "sql"
        ddl_name = f"{collection_prefix}ddl" if collection_prefix else "ddl"
        doc_name = f"{collection_prefix}documentation" if collection_prefix else "documentation"

        corpus = []
        for ctype, cname in [("sql", sql_name), ("ddl", ddl_name), ("documentation", doc_name)]:
            if cname not in existing:
                continue
            try:
                coll = client.get_collection(name=cname)
                data = coll.get()
            except Exception as e:
                print(f"[警告] 读取 {cname} 失败: {e}")
                continue
            if not data or not data.get("ids"):
                continue
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            for i, (doc_id, doc_content) in enumerate(zip(ids, docs)):
                if ctype == "sql":
                    try:
                        obj = json.loads(doc_content) if isinstance(doc_content, str) else {}
                        corpus.append({
                            "type": "sql",
                            "content": obj.get("sql", doc_content),
                            "question": obj.get("question") or "",
                        })
                    except Exception:
                        corpus.append({"type": "sql", "content": doc_content or "", "question": ""})
                else:
                    corpus.append({
                        "type": ctype,
                        "content": doc_content if isinstance(doc_content, str) else (doc_content or ""),
                        "question": None,
                    })

        knowledge_bases_backup.append({
            "id": kb_id,
            "name": name,
            "description": kb.get("description", ""),
            "corpus": corpus,
        })
        print(f"  已导出知识库: {name} ({kb_id}), 语料数: {len(corpus)}")

    out = args.output or f"knowledge_base_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "knowledge_bases": knowledge_bases_backup,
        "exported_at": datetime.now().isoformat(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n备份已写入: {out}")
    total = sum(len(kb["corpus"]) for kb in knowledge_bases_backup)
    print(f"共 {len(knowledge_bases_backup)} 个知识库, {total} 条语料。")
    return 0


if __name__ == "__main__":
    exit(main())
