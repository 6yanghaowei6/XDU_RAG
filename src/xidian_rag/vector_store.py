from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .categorizer import keyword_score
from .embeddings import EmbeddingProvider
from .io import ensure_parent
from .models import Chunk, SearchHit
from .settings import CHROMA_DIR, INDEX_PATH, Settings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


class LocalVectorStore:
    def __init__(self, path: Path = INDEX_PATH) -> None:
        self.path = path
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def build(self, chunks: list[Chunk], provider: EmbeddingProvider, batch_size: int = 64) -> None:
        self.chunks = chunks
        self.vectors = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.vectors.extend(provider.embed([chunk.text for chunk in batch]))
        self.save()

    def save(self) -> None:
        ensure_parent(self.path)
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self.vectors,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            self.chunks = []
            self.vectors = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        self.vectors = payload.get("vectors", [])

    def search(
        self,
        query: str,
        provider: EmbeddingProvider,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchHit]:
        if not self.chunks:
            self.load()
        if not self.chunks:
            return []
        query_vector = provider.embed([query])[0]
        hits: list[SearchHit] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if category and category != "全部" and chunk.category != category:
                continue
            vector_score = cosine_similarity(query_vector, vector)
            lexical_score = keyword_score(query, f"{chunk.title} {chunk.text}")
            score = vector_score * 0.78 + lexical_score * 0.22
            hits.append(SearchHit(chunk=chunk, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def stats(self) -> dict[str, int]:
        if not self.chunks:
            self.load()
        categories: dict[str, int] = {}
        for chunk in self.chunks:
            categories[chunk.category] = categories.get(chunk.category, 0) + 1
        return {"chunks": len(self.chunks), **{f"category:{key}": value for key, value in categories.items()}}


class ChromaVectorStore:
    def __init__(self, path: Path = CHROMA_DIR, collection_name: str = "xidian_official_pages") -> None:
        self.path = path
        self.collection_name = collection_name

    def _collection(self):
        import chromadb

        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.path))
        return client.get_or_create_collection(name=self.collection_name)

    def build(self, chunks: list[Chunk], provider: EmbeddingProvider, batch_size: int = 64) -> None:
        collection = self._collection()
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = provider.embed([chunk.text for chunk in batch])
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[
                    {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "url": chunk.url,
                        "source_site": chunk.source_site,
                        "category": chunk.category,
                        "publish_date": chunk.publish_date or "",
                    }
                    for chunk in batch
                ],
            )

    def load(self) -> None:
        self._collection()

    def search(
        self,
        query: str,
        provider: EmbeddingProvider,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchHit]:
        collection = self._collection()
        where = None
        if category and category != "全部":
            where = {"category": category}
        result = collection.query(
            query_embeddings=provider.embed([query]),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=metadata["doc_id"],
                text=text,
                title=metadata["title"],
                url=metadata["url"],
                source_site=metadata["source_site"],
                category=metadata["category"],
                publish_date=metadata.get("publish_date") or None,
            )
            score = 1.0 / (1.0 + float(distance))
            score = score * 0.78 + keyword_score(query, f"{chunk.title} {chunk.text}") * 0.22
            hits.append(SearchHit(chunk=chunk, score=score))
        return hits

    def stats(self) -> dict[str, int]:
        collection = self._collection()
        data = collection.get(include=["metadatas"])
        categories: dict[str, int] = {}
        for metadata in data.get("metadatas", []):
            category = metadata.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1
        return {"chunks": len(data.get("ids", [])), **{f"category:{key}": value for key, value in categories.items()}}


def build_vector_store(settings: Settings):
    if settings.vector_store == "chroma":
        return ChromaVectorStore()
    return LocalVectorStore()
