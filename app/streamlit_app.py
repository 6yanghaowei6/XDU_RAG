from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from xidian_rag.pipeline import build_rag_service
from xidian_rag.settings import load_settings, load_sources
from xidian_rag.vector_store import build_vector_store


st.set_page_config(page_title="西电官网 RAG 检索", page_icon="XD", layout="wide")

st.title("西电官网数据 RAG 智能检索")
st.caption("面向政策、保研、生活安排和比赛通知的官方资料问答。答案以已收录官网内容为依据。")

allowed_domains, _sources, category_keywords = load_sources()
categories = ["全部", *category_keywords.keys()]

with st.sidebar:
    st.header("检索设置")
    category = st.selectbox("资料分类", categories, index=0)
    top_k = st.slider("引用数量", min_value=1, max_value=10, value=5)
    stats = build_vector_store(load_settings()).stats()
    st.metric("已索引切片", stats.get("chunks", 0))
    st.divider()
    st.write("官方来源白名单")
    for domain in allowed_domains:
        st.caption(domain)

question = st.text_input("请输入你的问题", placeholder="例如：保研政策有哪些要求？近期有哪些竞赛通知？")

if st.button("检索", type="primary", use_container_width=False) and question.strip():
    with st.spinner("正在检索官方资料..."):
        service = build_rag_service()
        result = service.ask(
            question.strip(),
            top_k=top_k,
            category=None if category == "全部" else category,
        )

    st.subheader("回答")
    if result.has_evidence:
        st.success(result.answer)
    else:
        st.warning(result.answer)

    if result.citations:
        st.subheader("引用来源")
        for index, citation in enumerate(result.citations, start=1):
            with st.container(border=True):
                st.markdown(f"**[{index}] [{citation.title}]({citation.url})**")
                st.caption(
                    f"{citation.source_site} · {citation.category} · "
                    f"{citation.publish_date or '发布时间未知'} · score={citation.score}"
                )
                st.write(citation.snippet)
else:
    st.info("先构建知识库后开始提问：`xidian-rag crawl --max-pages 120`，然后 `xidian-rag index`。")
