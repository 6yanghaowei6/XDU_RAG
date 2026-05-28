from __future__ import annotations

from .embeddings import EmbeddingProvider
from .models import Answer, Citation, SearchHit
from .settings import Settings
from .vector_store import LocalVectorStore

REFUSAL = "未在已收录官方资料中找到依据。请更换问法，或先更新知识库后再查询。"


def make_snippet(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def hits_to_citations(hits: list[SearchHit]) -> list[Citation]:
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for hit in hits:
        chunk = hit.chunk
        key = f"{chunk.url}#{chunk.chunk_id}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        citations.append(
            Citation(
                title=chunk.title,
                url=chunk.url,
                source_site=chunk.source_site,
                category=chunk.category,
                publish_date=chunk.publish_date,
                snippet=make_snippet(chunk.text),
                score=round(hit.score, 4),
            )
        )
    return citations


class OpenAICompatibleChatClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for API chat")
        self.settings = settings

    def complete(self, question: str, hits: list[SearchHit]) -> str:
        import requests

        context = "\n\n".join(
            f"[{index}] 标题：{hit.chunk.title}\n来源：{hit.chunk.url}\n"
            f"发布时间：{hit.chunk.publish_date or '未知'}\n片段：{hit.chunk.text}"
            for index, hit in enumerate(hits, start=1)
        )
        prompt = (
            "你是西安电子科技大学官网资料检索助手。只能根据给定资料回答，"
            "不得编造资料外信息。回答要简洁，并在关键结论后标注引用编号。\n\n"
            f"问题：{question}\n\n资料：\n{context}"
        )
        response = requests.post(
            f"{self.settings.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.openai_chat_model,
                "messages": [
                    {"role": "system", "content": "你只根据检索资料作答，缺少依据时必须说明无法确认。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


class RagService:
    def __init__(self, store: LocalVectorStore, provider: EmbeddingProvider, settings: Settings) -> None:
        self.store = store
        self.provider = provider
        self.settings = settings

    def ask(self, question: str, top_k: int = 5, category: str | None = None) -> Answer:
        hits = self.store.search(question, self.provider, top_k=top_k, category=category)
        reliable_hits = [hit for hit in hits if hit.score >= 0.08]
        if not reliable_hits:
            return Answer(answer=REFUSAL, citations=[], has_evidence=False, mode="refusal")

        citations = hits_to_citations(reliable_hits)
        if self.settings.use_api_chat and self.settings.openai_api_key:
            try:
                answer = OpenAICompatibleChatClient(self.settings).complete(question, reliable_hits)
                return Answer(answer=answer, citations=citations, has_evidence=True, mode="api")
            except Exception as exc:
                fallback = self._extractive_answer(question, reliable_hits)
                fallback += f"\n\n提示：API 生成失败，已使用抽取式回答。原因：{exc}"
                return Answer(answer=fallback, citations=citations, has_evidence=True, mode="extractive-fallback")

        return Answer(
            answer=self._extractive_answer(question, reliable_hits),
            citations=citations,
            has_evidence=True,
            mode="extractive",
        )

    def _extractive_answer(self, question: str, hits: list[SearchHit]) -> str:
        lines = ["根据已收录的西电官方资料，找到以下相关信息："]
        for index, hit in enumerate(hits[:3], start=1):
            date = hit.chunk.publish_date or "发布时间未知"
            lines.append(f"{index}. {hit.chunk.title}（{hit.chunk.source_site}，{date}）：{make_snippet(hit.chunk.text, 180)}")
        lines.append("请以引用链接中的原文为最终依据。")
        return "\n".join(lines)
