from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SourceConfig:
    url: str
    source_site: str
    category: str
    may_require_vpn: bool = False


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    url: str
    source_site: str
    category: str
    publish_date: str | None
    content: str
    crawl_time: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Document":
        return cls(**value)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    url: str
    source_site: str
    category: str
    publish_date: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class Citation:
    title: str
    url: str
    source_site: str
    category: str
    publish_date: str | None
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    answer: str
    citations: list[Citation]
    has_evidence: bool
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "has_evidence": self.has_evidence,
            "mode": self.mode,
        }
