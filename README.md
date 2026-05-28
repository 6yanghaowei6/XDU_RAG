# 西电官网数据 RAG 智能检索系统

面向西电学生的首版 RAG 智能检索系统。系统采集西电公开官网页面，构建本地知识库，并通过自然语言问答返回带来源引用的答案。

## 功能范围

- 官方站点白名单采集：西电主站、综合信息网、信息公开网、教务处、研究生院、学生工作部。
- 聚焦内容：学校官方政策、推免/保研政策、生活安排、比赛通知。
- 检索能力：向量检索为主，关键词分类和文本匹配辅助。
- 回答约束：只基于检索命中的官方资料作答；没有依据时拒答。
- 界面：FastAPI 后端接口 + Streamlit 问答前端。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

没有 API Key 时系统会使用本地哈希向量检索和抽取式回答，适合先验证流程。配置 OpenAI-compatible API 后会启用模型生成答案。

默认向量存储为 `VECTOR_STORE=local`，无需额外服务；需要使用 Chroma 时，执行 `pip install -r requirements-chroma.txt`，把 `.env` 改为 `VECTOR_STORE=chroma` 后重新执行 `xidian-rag index`。

## 构建知识库

先小规模抓取公开页面：

```powershell
xidian-rag crawl --max-pages 120
xidian-rag index
```

也可以手动导入原文 HTML：把网页原文放入 `data/knowledge_base/pages/`，再执行：

```powershell
xidian-rag ingest-pages
xidian-rag index
```

也可以先不抓取，直接用 `ask` 验证空库拒答逻辑：

```powershell
xidian-rag ask "保研政策有哪些要求"
```

## 启动服务

```powershell
uvicorn xidian_rag.api:app --reload --host 127.0.0.1 --port 8000
```

接口示例：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question":"近期有哪些竞赛通知？","top_k":5}'
```

启动前端：

```powershell
streamlit run app/streamlit_app.py
```

## 项目结构

```text
app/streamlit_app.py        Streamlit 前端
config/sources.json         官方数据源白名单
src/xidian_rag/             核心包
tests/                      基础单元测试
data/knowledge_base/pages/  抓取到的网页原文快照（未清洗、未抽取）
data/documents.jsonl        抽取后的结构化文档
data/chunks.jsonl           处理后的文本切片
data/index/                 本地向量索引
data/crawl_failures.jsonl   抓取失败日志
```

`data/knowledge_base/` 只用于保存抓取到的原文。系统会在成功请求页面后，把原始响应内容按 `doc_id.html` 写入
`data/knowledge_base/pages/`；也可以手动把 `.html` 原文放入该目录，通过 `xidian-rag ingest-pages` 生成
`data/documents.jsonl`。后续的正文抽取结果、切片和索引仍写入 `data/` 下的处理产物文件。

## API 配置

`.env` 支持以下配置：

```text
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
USE_API_EMBEDDINGS=false
USE_API_CHAT=false
VECTOR_STORE=local
```

`USE_API_CHAT=false` 时，系统会用检索片段生成简短抽取式答案；`USE_API_EMBEDDINGS=false` 时，系统会用本地哈希向量，避免演示环境依赖外部模型。

## 数据合规

首版只采集公开、官方、可访问页面，不绕过登录、VPN 或权限限制。`notice.xidian.edu.cn` 若无法访问，会记录失败原因，不做规避。
