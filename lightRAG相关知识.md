# LightRAG 知识图谱构建与检索机制问答整理

## 问题一：这个项目是如何抽取文档进行知识图谱构建的，输入数据是什么形式，输出形式是什么形式，是直接写提示词让模型进行抽取吗，抽取后有没有做校验

### 总览
LightRAG 的实体/关系抽取核心在 `lightrag/operate.py` 的 `extract_entities()`（`operate.py:3246`），提示词模板集中在 `lightrag/prompt.py`。整个流程是：**文档解析 → 分块(chunking) → 直接用提示词调用 LLM 抽取 → 多层规则化校验/清洗 → 合并入图谱**。

### 1. 输入数据形式
抽取阶段接收的不是原始文件，而是**已经分好块的文本片段**，schema 是 `TextChunkSchema`（`lightrag/base.py:72`）：

```python
{"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int, "file_path": str}
```

在此之前，原始文档（PDF/DOCX/TXT/Markdown 等）先经过统一解析层（`lightrag/parser/`，支持 `legacy`/`native`/`mineru`/`docling` 引擎）解析为 LightRAG 自有的中间格式 `LightRAG Document`（含表格、图片、公式等多模态内容和章节元数据），再通过 `chunker/` 中的多种切分策略（Token-size、Recursive、Vector 语义、Paragraph 语义）切成块。多模态块还会带 `sidecar`（`{"type": "drawing"/"table"/"equation", "id": ...}`）字段。

### 2. 是否直接写提示词让模型抽取
是的，**直接用人工编写的提示词模板调用 LLM**，没有用 function-calling/tool-use 之类的结构化抽取框架。`extract_entities()` 从 `PROMPTS` 字典里取模板拼出 system prompt + user prompt（`operate.py:3357-3388`），然后通过 `use_llm_func_with_cache` 调用配置的抽取角色 LLM（`global_config["role_llm_funcs"]["extract"]`，带缓存）。

支持两种提示词风格（`prompt.py`）：
- **文本分隔符模式**（默认）：`entity_extraction_system_prompt` / `entity_extraction_user_prompt`，要求模型输出用 `<|#|>` 分隔的元组行，例如 `entity<|#|>实体名<|#|>类型<|#|>描述`，`relation<|#|>源<|#|>目标<|#|>关键词<|#|>描述`，并以 `<|COMPLETE|>` 结束。
- **JSON 结构化模式**（`entity_extraction_use_json=True` 时）：`entity_extraction_json_*` 系列模板，要求模型输出 `{"entities": [...], "relationships": [...]}`，并通过 `response_format={"type": "json_object"}` 让 LLM provider 走 JSON 模式。

提示词中还会把 `entity_types_guidance`（实体类型指引）、`max_total_records`/`max_entity_records`（数量上限）、`language`（输出语言）等参数动态注入。

此外还有一轮 **gleaning（二次补抽取）**：当 `entity_extract_max_gleaning > 0` 时，把首次抽取的对话历史回放给模型并附加 `entity_continue_extraction_*` 提示词，让模型补充遗漏或修正格式错误的记录（`operate.py:3442-3513`），再按描述长度选优合并。

### 3. 输出形式
- 文本模式：分隔符元组文本流（见上），结束符 `<|COMPLETE|>`。
- JSON 模式：`{"entities": [{"name","type","description"}...], "relationships": [{"source","target","keywords","description"}...]}`。

两种模式最终都被解析成统一的中间结构 `(maybe_nodes, maybe_edges)`：`maybe_nodes: dict[entity_name -> [node_dict...]]`，`maybe_edges: dict[(src,tgt) -> [edge_dict...]]`，再交给 `merge_nodes_and_edges()` 合并写入图存储（`upsert_node`/`upsert_edge`）和向量库（entity/relation VDB）。

### 4. 抽取后是否做校验 — 做了，而且是多层规则化校验
并**没有**用 JSON Schema/Pydantic 之类的 schema 校验库，而是自定义的多层防御式清洗逻辑：

**(a) 格式纠错层**（`_process_extraction_result`, `operate.py:1211`）
- 检查 `<|COMPLETE|>` 是否存在、按多重分隔符切分记录
- `fix_tuple_delimiter_corruption` 修复 LLM 输出中分隔符被破坏的情况
- `_normalize_text_extraction_record_attributes` 修复"关系行误用 entity 前缀"这类已知错误模式
- JSON 模式下用 `json_repair.loads` 容错解析略微畸形的 JSON（`_process_json_extraction_result`, `operate.py:647`）

**(b) 字段级校验/清洗**（`_handle_single_entity_extraction` / `_handle_single_relationship_extraction`, `operate.py:435/522`）
- 字段数量必须严格匹配（entity 4 个、relation 5 个），否则丢弃并 warning
- `sanitize_and_normalize_extracted_text` + `normalize_extracted_info`：清 HTML 标签、全角转半角、去除中英文间多余空格、去引号等
- entity_name/description 不能为空；entity_type 不能含 `'()<>|/\` 等非法字符，逗号分隔的类型取第一个非空 token，统一转小写
- 关系：source==target 的自环关系被丢弃；weight 用 `is_float_regex` 校验，非法则回退默认值 1.0

**(c) 长度/容量限制**
- `_truncate_entity_identifier`：实体名同时按字符数和 UTF-8 字节数截断（兼容 Milvus 等向量库 VARCHAR 字节限制）
- `entity_extract_max_records`/`max_entity_records`：限制单次响应的记录条数

**(d) 合并阶段的去重与一致性处理**（`_merge_nodes_then_upsert`/`_merge_edges_then_upsert`）
- 按描述内容去重、按 `source_id`/`file_path` 合并多块来源、entity_type 取出现频次最高者
- 当合并后描述过多/过长时，触发 `_handle_entity_relation_summary` 做 **map-reduce 式 LLM 摘要**（再次写提示词调用 LLM 压缩描述，带缓存）
- `apply_source_ids_limit` 对超量来源做 FIFO/KEEP 策略截断

简言之：**抽取本身是纯提示词驱动的自由文本/JSON 生成**，但抽取结果会经过相当工程化的"解析容错 → 字段规则校验 → 截断/去重/合并 → （必要时）二次 LLM 摘要"的多层流水线，而不是简单地信任模型输出直接入库。

---

## 问题二：抽取之后存到数据库中结构化展示这一步怎么做的，在用户提问时又是如何检索的，是多路召回吗

### 一、抽取结果如何结构化存储

LightRAG 用 4 类可插拔存储后端持久化抽取结果（`lightrag/base.py`，实现见 `lightrag/kg/`，支持 Neo4j/PostgreSQL/NetworkX/MongoDB/Memgraph 等）：

#### 1. 图存储（`chunk_entity_relation_graph`）— 核心结构化数据
合并阶段 `_merge_nodes_then_upsert` / `_merge_edges_then_upsert`（`operate.py:1926/2255`）把抽取并清洗后的实体/关系写成图节点和边：

- **实体节点**：
  ```python
  {entity_id, entity_type, description, source_id(多个chunk_id用GRAPH_FIELD_SEP拼接),
   file_path, created_at, truncate}
  ```
- **关系边**（无向图，src/tgt 按字符串排序保证一致性）：
  ```python
  {src_id, tgt_id, weight, keywords, description, source_id, file_path, created_at, truncate}
  ```
同名实体/同一对实体间的关系会被合并：描述去重拼接，过多/过长时触发 LLM map-reduce 摘要（`_handle_entity_relation_summary`），`entity_type` 取出现频次最高者，`source_id`/`file_path` 做来源汇总和数量截断。

#### 2. 向量库 — 三个独立的 VDB
- `entities_vdb`：embedding 内容是 `f"{entity_name}\n{description}"`，payload 含 `entity_name/entity_type/content/source_id/file_path`
- `relationships_vdb`：embedding 内容是 `f"{keywords}\t{src_id}\n{tgt_id}\n{description}"`（`operate.py:2798`），payload 含 `src_id/tgt_id/keywords/description/weight/...`
- `chunks_vdb`：原始文本块的向量索引（用于 naive/mix 检索）

#### 3. KV 存储
- `full_docs`/`text_chunks`：原文与分块内容
- `full_entities`/`full_relations`：每个文档抽取出的实体名/关系对清单
- `entity_chunks`/`relation_chunks`：维护"某实体/关系"对应的**完整** chunk_id 列表（用于增量重建 `rebuild_knowledge_from_chunks`，不受截断策略影响）
- `llm_response_cache`：抽取/查询 LLM 调用的响应缓存

#### 4. 文档状态存储（`doc_status`）
跟踪每个文档的处理阶段（PENDING/PROCESSING/PROCESSED/FAILED）。

简言之：实体/关系既写入**图数据库**（结构化关系/属性，供图遍历），又各自写入**专属向量库**（供语义检索），文本块本身也单独建了向量索引——三套索引并存，互为补充。

### 二、查询时如何检索 — 是的，是多路召回

检索逻辑在 `kg_query` → `_build_query_context` → `_perform_kg_search`（`operate.py:3673/4899/4202`），是一个清晰的 4 阶段、多路并行的召回+融合流水线：

#### 阶段 0：查询关键词拆分
先用 LLM 把用户问题拆成 **高层关键词（hl_keywords，主题/概念）** 和 **低层关键词（ll_keywords，具体实体）**（`get_keywords_from_query` → `extract_keywords_only`），分别驱动后面两条图检索路径。

#### 阶段 1：多路并行召回（`_perform_kg_search`）
按 `mode` 不同，触发不同组合：

| 路径 | 触发条件 | 做法 |
|---|---|---|
| **Local（实体路）** | local/hybrid/mix | 用 ll_keywords 向量检索 `entities_vdb` → 取回种子实体 → 通过图存储查它们的一跳关系（`_find_most_related_edges_from_entities`），按 (度数, 权重) 排序 |
| **Global（关系路）** | global/hybrid/mix | 用 hl_keywords 向量检索 `relationships_vdb` → 取回种子关系 → 反查关联的实体（`_find_most_related_entities_from_relationships`），保持向量相似度顺序 |
| **Naive/向量路** | mix（且配置了 chunks_vdb） | 直接用原始 query 向量检索 `chunks_vdb`，拿到原文文本块 |

三路召回的 embedding 是**批量一次性计算**的（query/ll/hl 合批），减少往返。

local 和 global 两路结果再做 **round-robin 交替合并 + 去重**得到 `final_entities`/`final_relations`（`operate.py:4343-4397`）。

#### 阶段 2：Token 截断
按 token 预算对实体/关系列表做截断排序（`_apply_token_truncation`）。

#### 阶段 3：文本块的二次多路召回与融合（`_merge_all_chunks`）
最终送给 LLM 的文本块同样来自三路，再次 round-robin 融合去重：
1. **vector_chunks**：阶段1的直接向量检索结果（"Naive 模式"贡献）
2. **entity_chunks**：从命中实体回溯其来源 chunk（`_find_related_text_unit_from_entities`），可配置两种选取算法：
   - `WEIGHT`：按实体出现频次做加权轮询（`pick_by_weighted_polling`）
   - `VECTOR`：按 query 向量相似度排序选取（`pick_by_vector_similarity`），失败会自动降级回 WEIGHT
3. **relation_chunks**：从命中关系回溯其来源 chunk（`_find_related_text_unit_from_relations`，并与 entity_chunks 去重）

#### 阶段 4：（可选）重排序 + 最终上下文拼装
若 `enable_rerank=True` 且配置了 rerank 模型（Cohere/Jina/阿里/通用 API，见 `lightrag/rerank.py`），会调用 `apply_rerank_if_enabled`（`utils.py:3278`）对融合后的文本块做语义重排，再按 `chunk_top_k` 截断。最后 `_build_context_str` 把实体表、关系表、文本块拼成结构化上下文字符串，连同 `rag_response` 系统提示词一起送给 LLM 生成回答。

#### 小结
检索是**实体向量召回（local）+ 关系向量召回（global）+ 原文块向量召回（naive，仅 mix 模式）** 三条独立路径并行执行，再各自沿图谱扩展出关联实体/关系/文本块，最后通过 round-robin 去重融合 + 可选 rerank 统一裁剪——是相当典型的多路召回 + 融合排序架构（这也正是 `mix` 模式被官方推荐配合 reranker 使用的原因）。

---

## 问题三：构建知识图谱之后的效果是比原始的RAG效果好吗

是的——根据项目自己的评测结果（`docs/Reproduce.md`），在 Agriculture/CS/Legal/Mix 四个领域数据集上，LightRAG（图谱增强）相比 NaiveRAG、RQ-RAG、HyDE 等基线在 Comprehensiveness（全面性）、Diversity（多样性）、Empowerment（有效赋能）三项指标上全面胜出，例如 Legal 领域 Overall 胜率高达 84.8% vs 15.2%。

不过这类提升主要体现在**需要跨文档、多跳关联的高层次/综合性问题**上——图谱把分散在不同 chunk 里的实体关系显式连接起来，所以"总结全局主题""比较多个实体""推导隐含关系"这类问题答得更全面、更有条理。但代价也很明显：
- 索引阶段要多跑一轮（甚至多轮 gleaning）LLM 抽取 + 摘要合并，构建成本和时间显著高于纯向量化
- 对于**简单事实型检索**（"某段话里提到了什么数值"），原始的 chunk 向量召回往往已经足够准确，图谱带来的增益有限，甚至可能因为额外的实体/关系噪声拉低精确度

所以更准确的说法是："图谱化通常能让复杂、综合性问题的回答质量明显优于纯向量 RAG，但对简单检索型问题的边际收益较小、成本却更高"——这也是为什么 LightRAG 默认推荐 `mix` 模式（图谱召回 + 原始向量召回融合），而不是完全替代朴素检索。
