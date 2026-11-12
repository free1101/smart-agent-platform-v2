# 代码文档 · Smart Agent Platform

本文档面向要读代码或改代码的人。产品视角的介绍、功能清单与截图在 [README.md](../README.md)。

---

## 目录

- [1. 分层架构](#1-分层架构)
- [2. 目录结构与模块职责](#2-目录结构与模块职责)
- [3. 一次对话请求的完整数据流](#3-一次对话请求的完整数据流)
- [4. 关键设计决策](#4-关键设计决策)
- [5. 配置项](#5-配置项)
- [6. HTTP API](#6-http-api)
- [7. 数据模型](#7-数据模型)
- [8. 检索链路](#8-检索链路)
- [9. 摄取链路](#9-摄取链路)
- [10. 测试矩阵](#10-测试矩阵)
- [11. 部署](#11-部署)
- [12. 扩展点](#12-扩展点)

---

## 1. 分层架构

```
┌─────────────────────────────────────────────────────────┐
│ 接入层                                                    │
│   web/index.html（演示前端，原生 JS + 手工解析 SSE）        │
│   dashboard/（Streamlit 运营控制台，6 页）                  │
├─────────────────────────────────────────────────────────┤
│ API 层  app/api/                                         │
│   chat.py  /api/chat · /api/chat/stream · /kb · /profile  │
│   admin.py /api/admin/*（摄取 / 索引 / 链路 / 总览）        │
│   eval.py  /api/eval/*（评测集 / 触发评测）                 │
│   deps.py       require_api_key（503 / 401 / 放行）        │
│   ratelimit.py  进程内 60s 滑动窗口（429 + Retry-After）    │
├─────────────────────────────────────────────────────────┤
│ Agent 层  app/agents/（LangGraph 状态图）                  │
│   supervisor → consultant / profile / react / support    │
│   上下文工程 context.py（超 8 轮折叠摘要）                  │
│   检查点 SqliteSaver → data/langgraph_state.sqlite        │
├─────────────────────────────────────────────────────────┤
│ 能力层                                                    │
│   rag/          BM25 + Chroma + 融合 + 重排               │
│   ingestion/    PDF/Word/MD 解析 + 图片描述 + 双层去重      │
│   mcp/          7 个 MCP 工具（预约 / 天气 / 知识）         │
│   services/     知识库 / 画像 / 长期记忆                   │
│   observability/ 链路追踪 + 评测指标                       │
├─────────────────────────────────────────────────────────┤
│ 数据层  app/db/（SQLAlchemy 2.0 + SQLite，10 张表）          │
│   Chroma 向量库（按 tenant 一个 collection）                │
└─────────────────────────────────────────────────────────┘
```

## 2. 目录结构与模块职责

### 2.1 `app/`（服务端）

| 路径 | 职责 |
|---|---|
| `app/main.py` | FastAPI 入口。模块导入时 `init_db()` 建表，挂载 3 个 router，`GET /` 与 `GET /api/health` 免鉴权，`/web` 挂静态前端 |
| `config/settings.py` | `pydantic_settings.BaseSettings`，读 `.env`，导出单例 `settings`。`has_llm_key` 会把 `sk-xxx` / `your*` 这类占位符判为未配置 |
| `api/deps.py` | `require_api_key()`：未配置 `API_KEY` 且未显式放行 → 503；key 缺失或错误 → 401；用 `hmac.compare_digest` 比较。支持 `X-API-Key` 与 `Authorization: Bearer/Token` |
| `api/ratelimit.py` | `check_rate_limit()`：进程内 60 秒滑动窗口 `deque`，桶键为 `api_key[:16] @ client_ip`，超限返回 429 + `Retry-After`；`RATE_LIMIT_PER_MINUTE=0` 关闭 |
| `api/routes/chat.py` | 对话、租户、知识库查询、画像、工单 |
| `api/routes/admin.py` | 文档摄取、索引重建、摄取记录、链路查询、总览指标、集合统计 |
| `api/routes/eval.py` | 评测集信息与评测触发（子进程调 `scripts/run_eval.py`，超时 120s） |
| `agents/graph.py` | `build_agent_graph()` 组装状态图；`_default_checkpointer()` 建 `SqliteSaver`，失败降级 `MemorySaver` |
| `agents/state.py` | `AgentState`（TypedDict）：`messages, user_input, session_id, tenant_id, intent, intent_reason, awaiting_booking, retrieved_docs, booking, reply, sources, handoff_reason, profile, user_key, summary, context` |
| `agents/intent_router.py` | `IntentRouter.classify()`：强信号规则 → LLM 分类 → 规则兜底；`is_explicit_handoff()` 识别"转人工" |
| `agents/consultant.py` | `consultant_node()`：调检索 → `build_consult_prompt()` 拼提示词 → 回答 + 来源 |
| `agents/business.py` | `business_node()` 规则状态机：`extract_booking_info()` 抽要素 → MCP 建单 → 写行为事件与长期记忆；含改期 / 取消分支 |
| `agents/react.py` | `react_node()`：`LLM.bind_tools()` 后自主循环调工具，`MAX_TOOL_STEPS=5`；无 key 或 `awaiting_booking=True` 时委托 `business_node()` |
| `agents/context.py` | 上下文工程：`MAX_ROUNDS=8` 触发压缩，`MAX_RECENT_ROUNDS=4` 保留原轮次，`prepare_history()` 返回 `(摘要块, 最近历史)` |
| `agents/profile.py` | `profile_node()`：读画像 + 行为，生成个性化回答与回访邀约 |
| `agents/support.py` | `support_node()`：转人工兜底 |
| `rag/base.py` | 抽象基类：`VectorHit` / `BaseVectorStore` / `BaseReranker` / `BaseSplitter`，以及 `EmbeddingIdentityMismatch` |
| `rag/factory.py` | `create_vector_store()` / `create_reranker()` / `create_splitter()`，按配置实例化 |
| `rag/vector_store.py` | `ChromaVectorStore`（按租户派生 collection 名）+ `InMemoryVectorStore`；`check_identity()` 比对 embedding 模型与维度，不匹配抛异常 |
| `rag/hybrid_retriever.py` | `HybridRetriever`：`build()` / `append_docs()` / `remove_docs()` / `rebuild_docs()` / `search()`，内部按 `bm25` → `dense` → `fuse` 三阶段记录 |
| `rag/fusion.py` | `rrf_fuse()` / `normalize_scores()` / `weighted_fuse()` 两种融合 |
| `rag/rerankers/` | `RuleReranker`（关键词/分类/长度规则）、`CrossEncoderReranker`（bge-reranker）、`LLMReranker`（listwise） |
| `rag/text_splitter.py` | `split_text()` 中文分块（句读 + 长度窗口 + overlap）、`tokenize()` 分词（停用词 + bigram） |
| `ingestion/pipeline.py` | `IngestionPipeline.run()` / `run_directory()`，阶段：`load` → `caption` → `split` → `embed_upsert` |
| `ingestion/loaders.py` | `PdfLoader`（按页保序，图片转占位符）、`DocxLoader`、`TextLoader`、`split_images()`、`fill_image_captions()`、`content_hash()` |
| `ingestion/dedup.py` | `file_hash()`（整文件 sha256）、`text_hash()`（块级指纹）、`find_success_record()`、`existing_content_hashes()` |
| `ingestion/image_captioner.py` | `ImageCaptioner.caption()` 调视觉模型生成图片描述 |
| `observability/trace.py` | `TraceContext` / `trace_scope(kind=...)` / `record_stage()`，`MAX_STAGES=50`、payload 截断，落 `trace_runs` + `trace_stages` |
| `observability/evaluation/metrics.py` | `hit_at_k()` / `mrr_at_k()` / `recall_at_k()` / `context_precision_at_k()` / `evaluate_case()` / `aggregate()` |
| `observability/evaluation/runner.py` | `EvaluationRunner`：跑 golden set、LLM 判定、Ragas 适配、写 `eval_runs` |
| `services/knowledge_service.py` | `import_seed()` / `add_doc()` / `search()` / `stats()` / `rebuild_index()` / `list_docs()`，按租户缓存检索器 |
| `services/profile_service.py` | `record_behavior()` / `get_profile()` / `should_return_reminder()` / `generate_return_message()` |
| `services/memory_service.py` | `remember()` / `recall()` / `format_recall()`：关键词重叠 ×2 + importance/10，再乘时间衰减（`RECALL_DAYS=90`），单用户上限 100 条 |
| `llm/provider.py` | `create_chat_model()` / `create_embeddings()`：统一走 OpenAI 兼容接口；缺 key 直接 `RuntimeError`，不返回假模型 |
| `mcp/tools.py` | 7 个工具：`create_booking` / `query_booking` / `cancel_booking` / `get_weather` / `query_knowledge_hub` / `list_collections` / `get_document_summary` |
| `mcp/client.py` | `get_mcp_tools()` 通过 `langchain-mcp-adapters` 接入，`mcp_enabled()` 判断可用性 |
| `db/models.py` | 10 张 ORM 表，见[第 7 节](#7-数据模型) |

### 2.2 `dashboard/`（运营控制台）

| 文件 | 页面 | 核心功能 |
|---|---|---|
| `dashboard/app.py` | 入口 | `st.navigation` 组织六页 + 侧边栏租户选择器 |
| `dashboard/common.py` | 公共层 | 直接复用 `app/api/routes/admin.py` 的纯函数，保证面板口径与接口一致 |
| `pages/overview.py` | 总览 | 指标卡、近七日查询量、延迟分位、最近失败摄取、最近异常查询 |
| `pages/data_browser.py` | 数据浏览 | 分类 / 关键词筛选、分页、单条详情 |
| `pages/ingestion_manager.py` | 摄取管理 | 上传文件、开关图片描述、查看记录、重建索引 |
| `pages/ingestion_traces.py` | 摄取追踪 | 阶段瀑布 + 阶段明细 |
| `pages/query_traces.py` | 查询追踪 | 阶段时间线、召回与融合明细、命中知识 |
| `pages/evaluation_panel.py` | 评估面板 | 跑评测、指标展示、未命中下钻、历史回归 |

### 2.3 `scripts/`

| 脚本 | 用途 |
|---|---|
| `run_web.py` | 启动 Web 演示（建表 + 导入种子 + uvicorn） |
| `run_dashboard.py` | 启动 Streamlit 控制台 |
| `import_knowledge_seed.py` | 导入种子知识，`--replace` 全量重建 |
| `ingest_docs.py` | 命令行文档摄取 |
| `rebuild_index.py` | 按租户重建向量索引 |
| `run_eval.py` | 跑评测，输出 `data/eval_report.json` |
| `demo_chat.py` | 命令行对话 |
| `start_mcp_server.py` / `test_mcp_client.py` | MCP 服务端启动与协议验证 |
| `smoke_dashboard.py` | 启动 Streamlit 用 Playwright 逐页截图自检，有异常退出码 1 |

## 3. 一次对话请求的完整数据流

以 `POST /api/chat/stream` 为例：

```
1. deps.require_api_key        校验 key（未配置→503，缺失/错误→401）
2. ratelimit.check_rate_limit  滑动窗口计数（超限→429）
3. chat.py 组装 AgentState     messages + session_id + tenant_id + user_key
4. context.prepare_history()   超 8 轮 → 折叠摘要，取「摘要 + 最近 4 轮」
5. graph 入口 supervisor
   └ intent_router.classify()  规则强信号 → LLM → 规则兜底
        ├─ consultant  → knowledge_service.search() → 拼提示词 → LLM → reply + sources
        ├─ react       → LLM.bind_tools() → 循环调 MCP 工具（≤5 步）→ reply
        ├─ profile     → profile_service.get_profile() + list_behaviors() → reply
        └─ support     → 返回人工渠道
6. business 分支额外动作
   ├ mcp.create_booking()      写 booking_tickets
   ├ record_behavior()         写 user_behaviors
   └ memory.remember()         写 memory_entries
7. SqliteSaver 落检查点        thread_id = session_id
8. 返回                        ChatResponse{reply, intent, intent_reason, sources, booking, session_id, trace_id}
   （流式路径改为 SSE 事件：meta → node* → delta* → done）
```

关键点：

- **状态只有一份**：所有节点都读写同一个 `AgentState`，节点之间不互传私有变量。
- **上下文工程不删消息**：`prepare_history()` 只在组装提示词时替换，`state["messages"]` 保持全量——因为 `business_node` 的跨轮补全依赖完整历史。
- **长期记忆走 `context` 字段**：召回结果注入 `state["context"]`，不写进对话历史，避免污染多轮语境。

## 4. 关键设计决策

### 4.1 为什么用 LangGraph 状态图，而不是单 Agent 循环

单 Agent 把意图判断、检索、回答、工具调用揉在一个循环里：难以扩展（加一种意图要改核心提示词）、难以排查（不知道错在哪一步）、难以单测（每个分支都要构造完整会话）。状态图把每个职责做成独立节点，节点可单独调用、可单独测试、可在前端暴露进度。

### 4.2 混合检索为什么优于纯向量

- BM25 擅长带明确词面的问题（"多少钱""几点关门"），向量擅长近义表达（"想歇会儿"→"足底按摩"）。
- 两种分数尺度不同，直接加会失真。本项目提供两条融合路径：
  - `rrf_fuse()`：只用排名，鲁棒但会抹平分数差；
  - `weighted_fuse()`：先 `min-max` 归一化再按权重加权，保留分差。
- 当前默认 `FUSION_STRATEGY=rrf`、`RRF_K=60`，可通过配置切到加权融合做 A/B。

### 4.3 ReAct 与规则流程的共存策略

`react_node()` 不是无条件启用，四条分流规则：

| 条件 | 走向 |
|---|---|
| 有真实 key 且不在补全中 | `react_node`，模型自主决定调哪些工具 |
| 无 key / key 是占位符 | 委托 `business_node`，行为与纯规则版一致 |
| `awaiting_booking=True`（正在逐项收字段） | 走规则状态机，一问一答更稳定 |
| 模型不支持 function calling 或调用抛错 | 回退 `business_node`，业务不中断 |

`MAX_TOOL_STEPS=5` 防止模型陷入工具调用死循环。

### 4.4 无伪降级

`create_chat_model()` 与 `create_embeddings()` 在缺 key 时直接抛错，各节点捕获后返回明确提示。代码里不存在"用模板文本假装模型回复"或"用哈希向量假装 embedding"的分支。这样演示效果与线上行为一致，任何异常都能定位到真实原因。

### 4.5 用户行为的不可变事件流

业务节点完成预约/取消/改期时**只写行为事件**（`user_behaviors`，只追加不修改），画像表 `user_profiles` 由聚合实时计算并缓存。好处是：画像的计算策略可以随时改，改完重算即可，不需要迁移历史数据。

### 4.6 三层记忆的分工

| 机制 | 作用域 | 存放 | 解决什么 |
|---|---|---|---|
| LangGraph checkpointer | 单会话（`thread_id`） | `data/langgraph_state.sqlite` | 指代追问、跨轮补全、改期取消 |
| 上下文工程 `context.py` | 单会话内进入模型的那一段 | `state["summary"]` | token 膨胀与注意力稀释 |
| 长期记忆 `memory_service.py` | 跨会话（`user_key`） | `memory_entries` 表 | 换会话仍记得老客户 |

这三者容易混，所以刻意用不同的存储位置和不同的作用域键分开，避免出现"删历史导致预约补全失效"这类耦合问题。

### 4.7 鉴权默认拒绝

`require_api_key()` 的三态语义：未配置 key 且未设 `ALLOW_UNAUTHENTICATED=true` → 503；配置了但请求没带/带错 → 401；匹配 → 放行。默认值选择"拒绝"而非"放行"，避免部署时忘配 key 导致接口裸奔。

### 4.8 限流为什么不用 Redis

单机演示与中小规模部署下，进程内 `deque` 滑动窗口足够，且不引入外部依赖。代价是多副本部署时计数不共享——这个取舍已写在 `ratelimit.py` 的注释里，需要横向扩展时替换实现即可。

## 5. 配置项

全部通过 `.env` 覆盖，定义在 `app/config/settings.py`。

### 模型

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` | `qwen-plus` | 对话模型 |
| `LLM_API_KEY` / `LLM_BASE_URL` | — | OpenAI 兼容端点凭据 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 向量模型 |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | — | 留空则复用 LLM 的 |
| `VISION_MODEL` | `qwen-vl-max` | 文档图片描述模型 |
| `VISION_ENABLED` | `True` | 关闭后摄取跳过图片描述 |

### 检索

| 变量 | 默认 | 说明 |
|---|---|---|
| `RETRIEVER_TOP_K` | `5` | 融合后返回条数 |
| `RETRIEVER_CANDIDATES` | `20` | 每路召回候选数 |
| `RETRIEVER_WEIGHT_BM25` / `RETRIEVER_WEIGHT_VECTOR` | `0.4` / `0.6` | 加权融合权重 |
| `FUSION_STRATEGY` | `rrf` | `rrf` 或 `weighted` |
| `RRF_K` | `60` | RRF 平滑常数 |
| `RERANK_TOP_K` | `3` | 重排后保留条数 |
| `RERANKER_PROVIDER` | `rule` | `rule` / `cross_encoder` / `llm` / `none` |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-base` | 交叉编码器重排模型 |

### 存储

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/smart_agent.db` | 业务库（换 PostgreSQL 改这一行） |
| `VECTOR_STORE` | `chroma` | `chroma` 或 `memory` |
| `CHROMA_DIR` | `data/chroma` | 持久化目录 |
| `CHROMA_BATCH_SIZE` | `64` | 批量写入大小 |

### 摄取

| 变量 | 默认 |
|---|---|
| `INGEST_MAX_MB` | `20` |
| `INGEST_CHUNK_SIZE` / `INGEST_CHUNK_OVERLAP` | `200` / `40` |
| `CAPTION_MAX_IMAGES` | `3` |

### 访问控制

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_KEY` | 空 | 为空且未放行 → 受保护接口 503 |
| `ALLOW_UNAUTHENTICATED` | `False` | 本地开发可置 `true` |
| `RATE_LIMIT_PER_MINUTE` | `120` | `0` 关闭限流 |

> 上下文工程与长期记忆的参数是模块内常量，不在 settings 里：`context.py` 的 `MAX_ROUNDS=8` / `MAX_RECENT_ROUNDS=4`，`memory_service.py` 的 `MAX_MEMORIES_PER_USER=100` / `RECALL_DAYS=90`。

## 6. HTTP API

除 `GET /`、`GET /api/health`、`/web/*` 外，所有 `/api/**` 端点都带 `Depends(require_api_key)` 与 `Depends(check_rate_limit)`。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 首页信息（含 LLM 与向量库状态） |
| GET | `/api/health` | 健康检查 |
| GET | `/api/tenants` | 租户列表 |
| POST | `/api/chat` | 同步对话 |
| POST | `/api/chat/stream` | SSE 流式对话，事件 `meta` / `node` / `delta` / `done` / `error` |
| GET | `/api/chat/tickets` | 最近工单 |
| POST | `/api/kb/search` | 检索调试（返回结果 + trace_id） |
| GET | `/api/kb/stats` | 知识库统计 |
| GET | `/api/kb/docs` | 知识分页列表 |
| GET | `/api/profile` | 画像 + 最近行为 + 回访建议 |
| GET | `/api/eval/dataset` | 评测集信息 |
| POST | `/api/eval/run` | 触发评测 |
| POST | `/api/admin/ingest` | 上传并摄取（后缀白名单，超限 413） |
| POST | `/api/admin/ingest/rebuild` | 重建租户索引 |
| GET | `/api/admin/ingestions` | 摄取记录 |
| GET | `/api/admin/traces` | 链路列表（按 kind / tenant / status 过滤） |
| GET | `/api/admin/traces/{trace_id}` | 链路详情（阶段按 seq 展开） |
| GET | `/api/admin/overview` | 总览指标 |
| GET | `/api/admin/collections` | 租户与集合统计 |

`ChatRequest`：`{message, session_id?, tenant_id?, user_key?}`
`ChatResponse`：`{reply, intent, intent_reason, sources, booking, session_id, trace_id}`

## 7. 数据模型

`app/db/models.py`，10 张表：

| 表 | 关键字段 | 说明 |
|---|---|---|
| `tenants` | `id, name(unique), industry, description` | 租户 |
| `knowledge_docs` | `tenant_id, content, category, keywords(JSON), embedding(JSON), source, source_hash, content_hash, chunk_index, doc_group, page, is_active` | 知识块，双 hash 支撑去重 |
| `booking_tickets` | `tenant_id, session_id, customer_name, service_type, appointment_time, status, notes` | 工单 |
| `user_behaviors` | `tenant_id, session_id, action_type, action_data(JSON)` | 不可变行为事件 |
| `user_profiles` | `tenant_id, session_id, favorite_service, favorite_time_period, favorite_duration, total_appointments, last_appointment_at` | 聚合画像 |
| `memory_entries` | `tenant_id, user_key, session_id, memory_type, content, importance` | 跨会话长期记忆 |
| `ingestion_records` | `tenant_id, filename, file_type, source_hash, status, chunks, images, elapsed_ms, error, trace_id` | 摄取记录 |
| `trace_runs` | `trace_id(unique), kind, tenant_id, query, status, total_ms, error` | 链路主表 |
| `trace_stages` | `trace_id, seq, name, elapsed_ms, payload(JSON)` | 链路阶段 |
| `eval_runs` | `tenant_id, total, top_k, top1_accuracy, top3_accuracy, metrics(JSON), config(JSON)` | 评测历史 |

## 8. 检索链路

```
query
 ├─ tokenize()                    中文分词：停用词过滤 + bigram
 ├─ BM25 召回 top N               N = RETRIEVER_CANDIDATES
 ├─ 向量召回 top N                Chroma query（metadata 过滤 tenant_id）
 ├─ fuse()                        rrf_fuse 或 normalize + weighted_fuse
 ├─ rerank()                      rule / cross_encoder / llm
 └─ 返回 top_k                    附 scores 与 trace 阶段记录
```

两个容易被忽略的实现细节：

- **embedding 身份校验**：`vector_store.check_identity()` 在打开 collection 时比对 `embedding_identity.json` 里的模型名与维度。换了 embedding 模型却复用旧索引，会得到语义错乱但"看起来能跑"的结果——所以这里选择直接抛 `EmbeddingIdentityMismatch`，要求重建索引。
- **trace 三阶段落库**：`bm25` / `dense` / `fuse` 各自 `record_stage()`，payload 有长度上限（`MAX_STR=300`、`MAX_LIST=20`），避免链路表膨胀。

## 9. 摄取链路

```
文件
 ├─ load            按类型选 Loader；PDF 按页保序，图片转占位符
 ├─ caption         ImageCaptioner 逐图生成描述，回填到占位符原位
 ├─ split           split_text() 分块 + overlap
 ├─ dedup           文件级 sha256 命中 → skipped；块级 text_hash 去重
 └─ embed_upsert    写 knowledge_docs + 写向量索引
```

幂等语义：

- 同一份文件（`sha256` 相同）成功导入过，再跑直接 `skipped`；
- 不同文件里的重复段落按块级指纹去重，不会在检索结果里重复出现；
- `--force` 跳过文件级判断，但仍做块级去重；
- 单张图片描述失败只跳过该图，不阻断整篇文档。

## 10. 测试矩阵

`python -m pytest -m "not e2e"`

| 文件 | 覆盖 |
|---|---|
| `test_vector_store.py` | 向量库增删改查、metadata 过滤、Chroma 持久化、embedding 身份校验（模型不符 / 维度不符 / 文件缺失 / 文件损坏） |
| `test_retriever.py` | 分块、分词、混合检索 Top-1、关键词命中、空查询、重排 |
| `test_fusion.py` | RRF 名次映射、权重生效、空输入 |
| `test_ingestion.py` | hash 忽略空白、Markdown 解析、图片占位替换、幂等跳过、force 块级去重、目录摄取、不支持扩展名 |
| `test_agents.py` | 三类意图分类 + 预约要素抽取 |
| `test_memory.py` | 多轮补全 / 改期 / 取消、线程隔离、`awaiting_booking` 跨 LLM 交接存活、ReAct 短答案续问 |
| `test_long_term_memory.py` | 上下文压缩阈值、召回按 user + tenant 隔离、跨会话召回注入、预约写长期记忆 |
| `test_profile.py` | 画像意图、行为记录与聚合、空画像、回访消息、时段推断 |
| `test_eval_metrics.py` | 五个指标口径正确性 + 缺标注时降级（不臆造分数） |
| `test_api_auth.py` | 503 / 401 / 三种 header 写法 / 429 / 按 IP 分桶 / 限流可关闭 |
| `test_trace.py` | 无 trace 时静默、阶段收集、payload 截断、阶段数上限、异常标记并重抛 |

## 11. 部署

```bash
docker compose up --build
# app       8000  uvicorn app.main:app
# mcp       9000  python -m app.mcp.tools
# dashboard 8501  streamlit run dashboard/app.py
```

三个服务共用一个镜像（`Dockerfile`，python:3.11-slim，非 root `appuser`），靠 `command` 区分角色。`app` 与 `dashboard` 挂同一个 `./data` 卷，保证控制台看到的是同一份库。

上线前必查：

1. `.env` 里配好 `API_KEY`，否则受保护接口返回 503；
2. `data/chroma/embedding_identity.json` 与当前 `EMBEDDING_MODEL` 一致，否则需要 `python scripts/rebuild_index.py --tenant <name>`；
3. `LLM_API_KEY` 与 `EMBEDDING_API_KEY` 都是真实值（占位符会被 `has_llm_key` 判为未配置）。

## 12. 扩展点

| 想做的事 | 改哪里 |
|---|---|
| 接入新行业 | 调 `import_knowledge_seed.py` 或 `ingest_docs.py` 指定新 `tenant`，不动代码 |
| 加一种意图 | 在 `intent_router.py` 加规则 → 在 `graph.py` 加节点与条件边 |
| 换向量库 | 实现 `BaseVectorStore` 接口 → 在 `rag/factory.py` 的 `create_vector_store()` 注册 |
| 换重排策略 | 实现 `BaseReranker` → 在 `rerankers/__init__.py` 注册 → 改 `RERANKER_PROVIDER` |
| 换分块策略 | 实现 `BaseSplitter` → 改工厂配置 |
| 加 MCP 工具 | 在 `app/mcp/tools.py` 用 `@mcp.tool()` 声明，业务节点会自动出现在工具清单里 |
| 换数据库 | 改 `DATABASE_URL`；ORM 层已是 SQLAlchemy 2.0 标准写法 |
| 加控制台页面 | 在 `dashboard/pages/` 加文件，`st.navigation` 自动发现 |
