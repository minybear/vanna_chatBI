#!/usr/bin/env python3
"""
从 export_knowledge_base_backup.py 生成的 JSON 恢复知识库语料到 ChromaDB。

使用前请先停止应用并删除 chroma_db 目录；然后直接运行本脚本即可（会创建新库并写入语料），无需先启动应用。

用法:
  python restore_knowledge_base_backup.py backup.json [--dry-run]
"""
import os
import json
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="从备份 JSON 恢复知识库语料")
    parser.add_argument("backup_file", help="备份 JSON 文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将恢复的内容，不写入")
    args = parser.parse_args()

    if not os.path.isfile(args.backup_file):
        print(f"[错误] 文件不存在: {args.backup_file}")
        return 1

    with open(args.backup_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    knowledge_bases = data.get("knowledge_bases", [])
    if not knowledge_bases:
        print("[信息] 备份中无知识库数据。")
        return 0

    if args.dry_run:
        for kb in knowledge_bases:
            print(f"  {kb.get('name', kb.get('id'))}: {len(kb.get('corpus', []))} 条语料")
        return 0

    # 延迟导入，确保 dotenv 已加载
    try:
        from knowledge_base_manager import get_kb_manager, init_kb_manager
        from app import vn
        from vanna.legacy.ZhipuAI.ZhipuAI_embeddings import ZhipuAIEmbeddingFunction
        import os as _os
        zhipu_embedding = ZhipuAIEmbeddingFunction(config={
            'api_key': _os.getenv('ZHIPU_API_KEY'),
            'api_base': _os.getenv('ZHIPU_API_BASE'),
        })
        init_kb_manager(vn.chroma_client, zhipu_embedding)
        manager = get_kb_manager()
    except Exception as e:
        print(f"[错误] 初始化失败（请确保 ChromaDB 已重建并可在当前环境连接）: {e}")
        return 1

    for kb in knowledge_bases:
        kb_id = kb.get("id", "")
        name = kb.get("name", "")
        description = kb.get("description", "")
        corpus = kb.get("corpus", [])
        if not corpus:
            print(f"  跳过（无语料）: {name}")
            continue

        is_default = kb_id == "default"
        if not is_default:
            existing = manager.get(kb_id)
            if not existing:
                existing = manager.get_by_name(name)
            if not existing:
                try:
                    created = manager.create(name=name, description=description or "", datasource_ids=[])
                    kb_id = created.id
                except Exception as e:
                    print(f"[错误] 创建知识库 {name} 失败: {e}")
                    continue
            else:
                kb_id = existing.id

        success = 0
        for item in corpus:
            try:
                ctype = item.get("type") or "documentation"
                content = (item.get("content") or "").strip()
                question = item.get("question")
                if not content:
                    continue
                manager.add_training_data(
                    kb_id=kb_id,
                    data_type=ctype,
                    content=content,
                    question=question if ctype == "sql" else None,
                )
                success += 1
            except Exception as e:
                print(f"    跳过一条 ({ctype}): {e}")
        print(f"  已恢复: {name}, 成功 {success}/{len(corpus)} 条")

    print("恢复完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
