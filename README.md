# claude-session-rag

跨 session 语义记忆检索框架，适用于需要跨 session 长期记忆的 Claude Code 项目。把历史对话建向量索引，每次新消息触发时自动召回相关记忆注入上下文。

## 架构

```
用户消息
  │
  ├── 关键词提取（jieba）
  │
  ├── BM25 搜索（session_archive.md）
  ├── 向量搜索（LanceDB cosine）
  └── SQL LIKE 精确匹配（专有名词兜底）
          │
          RRF 融合排序
          │
      注入 Claude 上下文
```

数据源支持两种：
- `session_archive.md`：结构化对话摘要，主数据源
- `.jsonl` 原始对话文件：Claude Code 项目目录下的完整对话记录

## 安装

```bash
pip install lancedb openai pyarrow tiktoken jieba rank-bm25
```

## 配置

在项目根目录或 `/root/.env` 里设置：

```bash
EMBEDDING_API_KEY=sk-...          # embedding API key（支持 SiliconFlow / OpenAI）
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
LANCE_DB_PATH=./memory_db          # LanceDB 存储路径（可选）
JSONL_DIR=/path/to/.claude/projects/-root  # JSONL 数据源目录（可选）
```

推荐用 [SiliconFlow](https://siliconflow.cn) 免费 API，`BAAI/bge-m3` 模型，1024 维。

## 建索引

### 默认（session_archive + JSONL 全量）

```bash
python build_index.py
```

### 指定路径

```bash
python build_index.py /path/to/session_archive.md /path/to/jsonl_dir
```

- 第一个参数：`session_archive.md` 路径
- 第二个参数：包含 `.jsonl` 文件的目录（Claude Code 项目目录）

### 分块策略

使用 **tiktoken** 按 token 切块：
- 块大小：400 token
- 重叠：50 token（相邻块之间）
- 编码：`cl100k_base`

`session_archive.md` 先按 `---` 分段，每段再用 tiktoken 切块。`.jsonl` 提取 `role=assistant` 的纯文本（跳过 `tool_use` / `thinking` 块），同样走 tiktoken。

增量更新：已处理的 chunk 自动跳过，只 embed 新内容。

### JSONL 数据源

Claude Code 的对话历史存在 `.claude/projects/<project-id>/*.jsonl`，`build_index.py` 会：

1. 扫描目录下所有 `.jsonl` 文件
2. 提取 `type=assistant`、`message.role=assistant` 的条目
3. 跳过 `tool_use`、`tool_result`、`thinking` 类型的 content block
4. 对提取出的纯文本做 tiktoken 切块，写入同一个 LanceDB 表

```bash
# 指定 JSONL 目录
JSONL_DIR=/root/.claude/projects/-root python build_index.py
```

## 检索

### 启动检索服务

```bash
python search_server.py
# 默认监听 127.0.0.1:15200
```

### 接口

```bash
# 混合检索（推荐）
curl -G --data-urlencode "q=你的查询" --data-urlencode "top_k=5" \
  http://127.0.0.1:15200/hybrid

# 向量检索
curl -G --data-urlencode "q=你的查询" http://127.0.0.1:15200/vector
```

**注意**：中文查询必须 URL encode，直接拼 URL 会被截断。

### 混合检索策略

三层融合，解决不同类型查询的召回问题：

1. **BM25**：关键词频率匹配，适合精确词汇
2. **向量搜索**：语义相似度，适合模糊表达
3. **SQL LIKE 精确匹配**：`text LIKE '%关键词%'`，专为中文专有名词兜底
   - FTS（tantivy）对中文使用 ASCII 分词，中文词无法命中
   - 向量相似度对专有名词（人名、地名、品牌）往往低于 threshold
   - LIKE 命中的结果强制 score=0.9，排在最前

结果用 RRF（Reciprocal Rank Fusion）排序后取 top_k。

## 接入 hook

参考 `hooks/` 目录：

- `recall_keywords.py`：从用户消息提取搜索关键词
- `notice_filter.py`：过滤召回结果，判断是否需要注入
- `breath_search.example.py`：与 OB breath 工具并行搜索的示例

## 文件结构

```
.
├── build_index.py        # 建索引（session_archive + JSONL）
├── search_server.py      # HTTP 检索服务入口
├── server.py             # HTTP handler
├── search.py             # 命令行测试检索
├── rag/
│   └── hybrid.py         # 混合检索逻辑（BM25 + 向量 + LIKE）
├── index/
│   ├── vector.py         # LanceDB 连接 + 向量搜索
│   └── bm25.py           # BM25 搜索
├── hooks/                # Claude Code hook 集成示例
└── utils/
    └── config.py         # 配置读取
```

## 费用参考

SiliconFlow `BAAI/bge-m3`：免费（截至 2026-08）

全量建索引（5000+ chunk）约 1-2 分钟，之后增量更新只处理新增内容。

每次检索：1 次 embedding API 调用（约 100ms）+ 本地 LanceDB 查询。
