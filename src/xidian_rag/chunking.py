from __future__ import annotations

from .models import Chunk, Document


def split_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        window = clean[start:end]
        split_at = max(window.rfind("。"), window.rfind("；"), window.rfind("！"), window.rfind("？"))
        if split_at > max_chars * 0.45 and end < len(clean):
            end = start + split_at + 1
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks(documents: list[Document], max_chars: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.content, max_chars=max_chars, overlap=overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}",
                    doc_id=document.doc_id,
                    text=text,
                    title=document.title,
                    url=document.url,
                    source_site=document.source_site,
                    category=document.category,
                    publish_date=document.publish_date,
                )
            )
    return chunks
