from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import build_rag_service, crawl_to_disk, index_documents, ingest_pages_to_disk
from .settings import RAW_PAGES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="西电官网数据 RAG 智能检索系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="采集官方公开页面")
    crawl_parser.add_argument("--max-pages", type=int, default=200)
    crawl_parser.add_argument("--max-depth", type=int, default=2)

    ingest_parser = subparsers.add_parser("ingest-pages", help="从原文 HTML 快照生成结构化文档")
    ingest_parser.add_argument("--pages-dir", type=Path, default=RAW_PAGES_DIR)

    subparsers.add_parser("index", help="构建本地向量索引")

    ask_parser = subparsers.add_parser("ask", help="命令行问答")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--category", default=None)
    ask_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "crawl":
        count = crawl_to_disk(max_pages=args.max_pages, max_depth=args.max_depth)
        print(f"已采集并保存 {count} 篇文档")
    elif args.command == "ingest-pages":
        count = ingest_pages_to_disk(pages_dir=args.pages_dir)
        print(f"已从原文页面导入 {count} 篇文档")
    elif args.command == "index":
        doc_count, chunk_count = index_documents()
        print(f"已索引 {doc_count} 篇文档，生成 {chunk_count} 个文本切片")
    elif args.command == "ask":
        service = build_rag_service()
        result = service.ask(args.question, top_k=args.top_k, category=args.category)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
