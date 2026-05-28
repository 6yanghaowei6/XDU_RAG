from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import build_rag_service
from .settings import load_settings, load_sources
from .vector_store import build_vector_store

app = FastAPI(title="西电官网数据 RAG 智能检索系统", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="学生自然语言问题")
    category: str | None = Field(default=None, description="可选分类：政策、保研、生活、竞赛、教务、学工")
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sources")
def sources() -> dict:
    allowed_domains, seed_sources, category_keywords = load_sources()
    return {
        "allowed_domains": allowed_domains,
        "seed_sources": [asdict(source) for source in seed_sources],
        "categories": list(category_keywords.keys()),
    }


@app.get("/stats")
def stats() -> dict:
    store = build_vector_store(load_settings())
    return store.stats()


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    service = build_rag_service()
    answer = service.ask(request.question, top_k=request.top_k, category=request.category)
    return answer.to_dict()
