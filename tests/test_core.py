from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import xidian_rag.crawler as crawler_module
from xidian_rag.categorizer import categorize
from xidian_rag.chunking import make_chunks, split_text
from xidian_rag.crawler import make_doc_id, raw_page_path, save_raw_page_snapshot
from xidian_rag.embeddings import HashEmbeddingProvider
from xidian_rag.models import Document
from xidian_rag.pipeline import ingest_pages_to_disk
from xidian_rag.rag import REFUSAL, RagService
from xidian_rag.settings import Settings
from xidian_rag.vector_store import LocalVectorStore


class CoreTests(unittest.TestCase):
    def test_split_text_keeps_short_text(self) -> None:
        self.assertEqual(split_text("西电 推免 政策", max_chars=20), ["西电 推免 政策"])

    def test_categorize_prefers_keyword_match(self) -> None:
        category = categorize(
            "关于推荐免试研究生工作的通知",
            "通知",
            {"保研": ["推免", "推荐免试"], "竞赛": ["比赛"]},
        )
        self.assertEqual(category, "保研")

    def test_raw_page_snapshot_keeps_original_bytes(self) -> None:
        url = "https://gr.xidian.edu.cn/example"
        payload = b"<html><body> raw \xe8\xa5\xbf\xe7\x94\xb5 </body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = crawler_module.RAW_PAGES_DIR
            crawler_module.RAW_PAGES_DIR = Path(tmp)
            try:
                save_raw_page_snapshot(url, payload)
                path = raw_page_path(url)
                self.assertEqual(path.name, f"{make_doc_id(url)}.html")
                self.assertEqual(path.read_bytes(), payload)
            finally:
                crawler_module.RAW_PAGES_DIR = original_dir

    def test_ingest_pages_generates_documents_from_html(self) -> None:
        repeated_text = "学校发布推荐免试研究生工作通知，说明推免资格、综合成绩和申请流程。"
        html = f"""
        <html>
          <head><title>推荐免试研究生工作通知</title></head>
          <body><article>{repeated_text * 4}</article></body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            pages_dir = Path(tmp) / "pages"
            documents_path = Path(tmp) / "documents.jsonl"
            pages_dir.mkdir()
            (pages_dir / "notice.html").write_text(html, encoding="utf-8")

            count = ingest_pages_to_disk(pages_dir=pages_dir, documents_path=documents_path)

            self.assertEqual(count, 1)
            document = Document.from_dict(json.loads(documents_path.read_text(encoding="utf-8")))
            self.assertEqual(document.title, "推荐免试研究生工作通知")
            self.assertIn("推免资格", document.content)
            self.assertEqual(document.source_site, "本地原文")

    def test_ingest_pages_skips_short_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pages_dir = Path(tmp) / "pages"
            documents_path = Path(tmp) / "documents.jsonl"
            pages_dir.mkdir()
            (pages_dir / "empty.html").write_text("<html><title>空页面</title><body>太短</body></html>", encoding="utf-8")

            count = ingest_pages_to_disk(pages_dir=pages_dir, documents_path=documents_path)

            self.assertEqual(count, 0)
            self.assertEqual(documents_path.read_text(encoding="utf-8"), "")

    def test_rag_refuses_empty_store(self) -> None:
        settings = Settings(None, "https://api.openai.com/v1", "chat", "embed", False, False, "local", 1, 0)
        store = LocalVectorStore(path=Path(tempfile.gettempdir()) / "missing-xidian-rag-index.json")
        provider = HashEmbeddingProvider()
        service = RagService(store, provider, settings)
        result = service.ask("保研政策有哪些要求")
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.answer, REFUSAL)

    def test_local_vector_search_finds_related_chunk(self) -> None:
        document = Document(
            doc_id="doc1",
            title="推荐免试研究生工作通知",
            url="https://gr.xidian.edu.cn/example",
            source_site="研究生院",
            category="保研",
            publish_date="2026-05-01",
            content="学校发布推荐免试研究生工作通知，说明推免资格、综合成绩和申请流程。",
            crawl_time="2026-05-13T00:00:00Z",
            checksum="abc",
        )
        chunks = make_chunks([document], max_chars=80)
        provider = HashEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalVectorStore(path=Path(tmp) / "index.json")
            store.build(chunks, provider)
            hits = store.search("保研 推免 资格", provider, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.category, "保研")


if __name__ == "__main__":
    unittest.main()
