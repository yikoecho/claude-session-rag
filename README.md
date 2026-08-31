# claude-session-rag

Claude Code 的语义记忆召回系统。将你的对话历史建立索引，在每条新消息发出时，通过 `UserPromptSubmit` hook 把相关的历史上下文注入到提示词里。

这不是一个通用 RAG 框架。它专门为 Claude Code CLI 设计：对话是长期运行的 session，主要文档是 Claude Code 自动写入的 `.jsonl` 对话记录，辅以手写的 session 摘要，目标是让 Claude 在不撑爆上下文窗口的前提下，记住几周前发生的事。

> **安全提示：** 搜索服务器没有身份验证，只绑定 `127.0.0.1`，不要暴露 15200 端口。`.env`、`session_archive.md`、`*.jsonl`、`memory_db/`、`eval_baseline/` 这些都不能进版本库——它们包含真实的对话内容。`.gitignore` 已经覆盖了以上所有路径，但 `eval_baseline/` 特别容易被遗漏，因为它是后加的目录。

## 架构

```
*.jsonl（必须）                  session_archive.md（可选）
   │  Claude Code 自动写入           人工整理的 session 摘要
   └──────────────┬──────────────────────────┘
                  ↓ build_index.py
     LanceDB（bge-m3 向量） + 内存 BM25
                  ↓ server.py（HTTP 在 127.0.0.1:15200）
     UserPromptSubmit hook（memory_recall.sh）
                  ↓
     additionalContext 注入到 Claude Code 提示词
```

**两路数据源互补，不分主次：**

- **`JSONL_DIR` 下的 `*.jsonl`**：Claude Code 自动写入的原始对话记录，覆盖所有消息，是技术细节、实现语言的主要来源。`build_index.py` 从中提取用户侧的 `<channel>` 标签内容和 assistant 侧的 `tool_use` 回复，滑动窗口切块后进 LanceDB 向量索引。**注意：** 从 Claude Code 2.1.183 起，若某轮没有可见 text 输出，系统会注入一句终端旁白（如"已回复用户"），这些旁白存在 `text` 块里而非 `tool_use` 块，`build_index.py` 会自动跳过。

- **`session_archive.md`（可选）**：人工整理或 Stop hook 自动生成的摘要文件，为 BM25 增加一层语义压缩，适合情感类、关系类的召回——这类内容通常没有可搜的专有名词，靠摘要语义抽象才能命中。技术类召回主要靠 JSONL，不受影响。

**实体提取：** `enrich_entities.py` 读取 `session_index.jsonl`（每行一个条目，含 `key`、`text`、`date` 字段），调用 LLM 提取命名实体，写回 `entities` 字段。这是 BM25 实体路径的基础——没有它，人名、技术术语、项目名等低频词的关键词召回基本不可用。

## 环境要求

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep)（`rg`）在 PATH 中（hook 需要）
- 兼容 OpenAI 客户端的 Embedding API（测试使用 [SiliconFlow](https://siliconflow.cn) + `BAAI/bge-m3`）
- OpenRouter API key（可选，用于 Haiku 召回过滤和实体提取）

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
git clone https://github.com/yikoecho/claude-session-rag.git
cd claude-session-rag
cp config.example.env .env
# 编辑 .env，至少设置 EMBEDDING_API_KEY 和 JSONL_DIR
```

建索引：

```bash
python build_index.py
# 或：python build_index.py /path/to/session_archive.md /path/to/jsonl_dir
# session_archive.md 是可选的，不传或传空路径跳过
```

启动搜索服务器：

```bash
# 入口说明：
# search_server.py  —— 推荐，完整入口（初始化 LanceDB、FTS、recall agent，然后启动 HTTP 服务）
# server.py         —— HTTP handler + 路由定义，由 search_server.py 导入；也可以单独运行
# search.py         —— 命令行检索工具，不启动服务，直接输出检索结果
python search_server.py
```

测试：

```bash
curl "http://127.0.0.1:15200/hybrid?q=你的查询&top_k=3"
```

配置 Claude Code hook，在 `.claude/settings.json` 里加：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash /path/to/claude-session-rag/hooks/memory_recall.sh"
          }
        ]
      }
    ]
  }
}
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/hybrid` | RRF 混合搜索（向量 + BM25 + 关键词精确匹配），返回 JSON |
| GET | `/bm25` | 纯 BM25 搜索，返回纯文本 |
| GET | `/reload_bm25` | 重建内存 BM25 索引（archive 更新后调用） |
| POST | `/recall` | Haiku 过滤器——对候选结果做三档判定（见下文） |

### `/recall` 三档判定

`/recall` 是这套管道最关键的部分。它把混合搜索的候选结果送给小模型（默认 `claude-haiku-4-5`），返回以下四种 verdict：

- **`sufficient`**：候选片段能直接回答问题，注入到 `additionalContext`。
- **`partial`**：找到了相关内容但置信度低，加前缀 `以下内容与问题相关但可能未直接回答，仅供参考：` 注入，让 Claude 知道这不是确定的答案。
- **`none`**：没找到相关内容，注入 `[archive_status] 档案中未找到与该问题相关的记录。` 让 Claude 知道这个沉默是主动判断，不是管道故障。
- **`unfiltered`**：`RECALL_ENABLED=False` 或 LLM 调用失败时的降级态，候选结果原样注入并加标记。

**`none` 是防止幻觉的关键。** 没有它，"没找到"和"根本没搜"对下游来说看起来一样，Claude 可能用脑补来填补空白。需要 `OPENROUTER_API_KEY`（或等价的 `LLM_BACKEND` 配置）；无 LLM 时降级到 `unfiltered`。

## 配置

所有配置通过环境变量（或 `.env` 文件）设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_API_KEY` | — | **必须**，Embedding API key |
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | Embedding API 地址 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型名 |
| `LANCE_DB_PATH` | `<repo>/memory_db` | LanceDB 存储路径 |
| `JSONL_DIR` | `~/.claude/projects/-root` | **必须**，原始 `.jsonl` 文件目录 |
| `ARCHIVE_FILE` | `~/.claude/session_archive.md` | 可选，BM25 用的摘要文件 |
| `JSONL_INDEX_FILE` | `~/.claude/session_index.jsonl` | BM25 实体路径用的索引文件 |
| `BM25_ENTITY_MIN_SCORE` | `7.0` | 实体 BM25 命中的最低分数阈值 |
| `SEARCH_PORT` | `15200` | 搜索服务器端口 |
| `LLM_BACKEND` | 自动检测 | `none` \| `siliconflow` \| `ollama` \| `api` \| `openrouter` |
| `OPENROUTER_API_KEY` | — | 可选，启用 Haiku 召回过滤 |
| `RECALL_MODEL` | `anthropic/claude-haiku-4-5` | 召回过滤模型 |

完整列表见 `config.example.env`。

### `BM25_ENTITY_MIN_SCORE`

实体 BM25 路径为高置信度的实体命中保留 2 个结果槽位，分数达到阈值才能占用。默认值 `7.0` 是在作者的语料（约 780 条实体，中文对话）上调出来的，你可能需要根据自己的数据重新调整：

- 看到不相关的实体结果 → 提高阈值
- 已知专有名词没出现 → 降低阈值
- 用 `eval_rag.py --diff` 量化改动效果

## LLM 后端

| 模式 | 费用 | 说明 |
|------|------|------|
| `none` | 免费 | 零配置，纯向量+BM25，无 LLM |
| `siliconflow` | 免费 | 远程 API，可复用 `EMBEDDING_API_KEY` |
| `ollama` | 免费 | 本地推理，运行 `ollama pull qwen2.5:3b` |
| `api` | 付费/免费 | 任何 OpenAI 兼容端点（付费：OpenRouter、Anthropic API；免费：Groq、Cerebras、OpenRouter 免费模型如 `meta-llama/llama-3.1-8b-instruct:free`）|
| `openrouter` | 付费/免费 | `api` 的别名，向后兼容保留 |

自动检测：设置了 `OPENROUTER_API_KEY` 则使用 `openrouter`，否则 `none`。

## Eval 框架

`eval_recall.py` 是这套系统的核心评估工具，直接调用真实的 `memory_recall.sh` hook（`FORCE_RECALL=1` 绕过 notice_filter），从 `/tmp/recall_eval_result.json` 读取结构化的 verdict/reason/候选数。

**重要：eval 必须走真实 hook，不能重新实现管道逻辑。** 两套实现只要不是同一份代码就会持续分叉，而且分叉是隐蔽的。

查询集分两组，必须分开报指标：

- **recall 组**（有记录的事）：测命中率。期望 `sufficient` 或 `partial`。
- **no_record 组**（确定没有记录的事）：测 none 率。期望 `none`。两组混报会互相掩盖问题。

示例查询集见 `eval_baseline/eval_queries.example.json`（替换成你的真实内容，不要提交进版本库）。

当前基线（在作者语料上，基于真实 hook 路径）：
- recall 组命中率：96%（25条）
- no_record 组 none 率：7/7，编造率 0%
- 测量条件：search_server 健康（无端口冲突，`ss -tlnp | grep 15200` 确认）、`FORCE_RECALL=1` 直调 hook、三次连跑无波动

**关于 eval 环境的教训：** 三次连跑结果一致才是基线，单次读数只是读数。这套系统在部署期间踩到了三次 eval 环境污染（索引路径写错、eval 脚本逻辑分叉、server 进程崩溃循环），每次都产生了一个看起来合理的数字（88%、84%），但地基是坏的。**仪器比被测系统更容易坏。** 任何 eval 跑分前，先确认 search_server 进程健康（`systemctl status search_server`），再确认端口没被孤儿进程占用（`ss -tlnp | grep 15200`）。

**关于三次连跑结果完全一致：** 这可能意味着 25 条 query 的候选质量都在 Haiku 的决策边界之外（好事），也可能意味着 server 端或 agent 层有缓存命中。如果未来某次改动让分数变化，但三次仍然完全一致，就需要检查是否有缓存在起作用，避免误把缓存当基线。

## 调试与可观测性

**三条已知的静默失败模式（会让"检索出错"看起来像"没找到"）：**

1. **hybrid_candidates 解析失败**：`/hybrid` 返回的 keyword 路径 score 是 `null`，`round(None, 3)` 抛 TypeError，整个 Python 块 fallback 到 `[]`，hybrid 路径消失。会在 `memory_recall_error.log` 里输出 `[WARN][B] hybrid_candidates 解析失败`。

2. **recall_agent JSON parse 失败**：Haiku 偶尔在 reason 字段里插入真实换行符，`json.loads` 报 `Expecting ',' delimiter`，fallback 到 `unfiltered` 把所有候选原样注入。已加换行折叠预处理修复。

3. **notice_filter SKIP vs 检索无结果**：两者对下游都是"没有注入"，但原因完全不同。SKIP 是门控拦截，无结果是检索层找不到。`none` verdict 的 `[archive_status]` 注入就是为了区分这两种情况——让 Claude 知道"我搜了，没有"而不是"我没搜"。

**日志文件：**
- `memory_recall_error.log`：hook 的中间层日志，包含各路候选数、top 分、timing、WARN/ERROR
- `server.log`：search server 日志，包含 recall_agent 的判断结果

## Hook 说明

**`hooks/memory_recall.sh`**：主 hook。每条用户消息触发：门控（`notice_filter.py`）→ 提取关键词（`recall_keywords.py`）→ 三路搜索（BM25 关键词路径 + hybrid + jsonl grep）→ Haiku 过滤 → 结果注入 `additionalContext`。

**`hooks/recall_keywords.py`**：用 jieba 对提示词分词、去停用词、输出关键词串。可选 LLM 查询改写（`QUERY_REWRITE_ENABLED=true`），默认关闭（每条消息多一个 LLM 往返，延迟明显）。

**`hooks/stop_archive.sh`**：session 结束时自动把摘要追加到 `session_archive.md`。需要 `RECALL_API_KEY`，安装为 `Stop` hook。

**`hooks/notice_filter.py`**：消息门控规则。短消息（≤6字）直接 SKIP；7-15字需要检测到信号词（技术术语、情感词等）才继续；长消息直接通过。

## Limitations

**向量分数不能用来判断相关性。** bge-m3 对几乎任何中文输入都给出 ~0.5 的余弦相似度——索引里不存在的编造词是 0.50，真实命中是 0.54。差距不足以设阈值。空召回判断只能靠 BM25：命中给正常分数，零命中就是零命中。

**RRF 融合分只反映排名，不反映匹配质量。** 分数是 `1/(60+rank)`，top-1 恒等于 0.0164，前几名全挤在 0.016–0.033。这个区间是 k=60 时 RRF 的全部值域，不管切块参数怎么调都不会变——用它评估检索质量是错的。指标应该是命中率和 MRR，不是分数分布。

**chunk 不能切太碎。** 按单条消息切块会让平均长度掉到 80 字符左右，短文本的 embedding 语义信号很弱，容易被表面词汇主导。合并相邻消息成滑动窗口（当前默认 5 条/步长 3）之后，检索质量有明显改善。合并时注意断点：跨 session 不合并、时间间隔超 30 分钟不合并、exclude_ranges 跳过区间后不缝合。

**词汇鸿沟真实存在。** 用户用生活语言提问（"那个排队的问题"），索引里存的是实现语言（函数名、错误信息、配置项），BM25 匹配不上，向量也桥不过去。aliases 字段是兜底——只在主路零命中时调用，不进主搜索池。根本解法是写入时就生成 aliases（"你以后会怎么称呼这件事"），历史记录可用 `enrich_entities.py` 批量回填。

## 已知问题

**中文停用词：** 不加停用词列表，高频功能词（一起、然后、过）IDF 接近零但仍作为 BM25 token，导致通用口语片段分数虚高。`hooks/recall_keywords.py` 里的 jieba 停用词表处理了查询端分词；索引端用同样的列表。

**RRF 单路惩罚：** RRF 分数是 `1/(60+rank)`，只出现在一条路径上的结果永远输给两路都有的结果，即使单路排名很高。低频专有名词 BM25 分高但向量匹配弱，可能从最终 top-k 里消失。实体保留槽位机制是补偿手段，但需要 `BM25_ENTITY_MIN_SCORE` 调对。

**自然语言指代词命中率：** "方案一"、"那个部署脚本"、"上次说的那个组件"这类指代型 query 命中率低。索引记录的是实现语言（组件名、技术细节），而用户事后的自然说法是生活语言（"方案一"、"那个搜索的配置"）。根本解法是写入时同时生成 aliases 字段（"你以后会怎么称呼这件事"），并对历史记录批量回填。查询侧改写对纯指代词无效——LLM 在没有上下文时无法把"方案一"扩展成正确的实指。

**跨查询分数不可比：** BM25 分数不能跨查询比较。单词查询的绝对分数永远低于三词查询，不要用同一个 `BM25_ENTITY_MIN_SCORE` 阈值横向对比不同类型的查询。

## Session Archive 格式

用 `---` 分隔的 Markdown，每段有 `### YYYY-MM-DD` 时间戳：

```markdown
## Session: 2026-06-15 to 2026-06-17

### 2026-06-15 23:45 CST

这段对话发生了什么...

---

### 2026-06-16 14:30 CST

另一天的摘要...

---
```

## License

MIT
