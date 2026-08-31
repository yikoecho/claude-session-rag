# CLAUDE.md

给 Claude 看的安装与排障指南。用户大概率是把这个仓库链接直接丢给你，然后说"帮我装上"。按下面的顺序做，不要跳步。

README 面向人类，讲的是设计意图；这份文件讲的是实际操作时会卡在哪里。两份内容有冲突时，以这份为准。

---

## 这个项目是什么

Claude Code 的语义记忆召回系统。索引用户的历史对话，在每条新消息发出时通过 `UserPromptSubmit` hook 把相关历史注入提示词，让 Claude 记住几周前的事而不撑爆上下文。

**不是通用 RAG 框架。** 它绑定 Claude Code CLI 的 `.jsonl` 对话记录格式，换个环境不能直接用。

---

## 安装顺序

### 1. 克隆并装依赖

```bash
git clone https://github.com/yikoecho/claude-session-rag.git
cd claude-session-rag
pip install -r requirements.txt
```

**坑：** Ubuntu 23.04+ / Debian 12+ 会报 `externally-managed-environment`。两个解法，推荐后者：

```bash
pip install -r requirements.txt --break-system-packages   # 快，但污染系统 Python
# 或
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

用了 venv 的话，后面 hook 里调 Python 的地方要写 venv 里的绝对路径，否则 hook 跑起来找不到包。

### 2. 确认 ripgrep 存在

```bash
rg --version || echo "需要先装 ripgrep"
```

hook 依赖 `rg`，没有会静默失败。

### 3. 配置 .env

```bash
cp config.example.env .env
```

**必填两项，你替用户填不了第一项：**

- `EMBEDDING_API_KEY` — 让用户自己去拿。默认用 SiliconFlow（https://siliconflow.cn）的 `BAAI/bge-m3`，注册后在控制台创建。任何 OpenAI 兼容的 embedding 端点都行，改 `EMBEDDING_BASE_URL` 即可。
- `JSONL_DIR` — 指向 `~/.claude/projects/` 下对应项目的子目录。用这条找：

```bash
ls -dt ~/.claude/projects/*/ | head -5
```

按修改时间排序，最上面那个通常就是当前项目。目录名是路径转义过的，确认一下再填。

`OPENROUTER_API_KEY` 可选，用于 Haiku 召回过滤和实体提取。没有的话系统降级到 `unfiltered` 档，能跑但会把未经筛选的候选全塞给 Claude。

### 4. 建索引

```bash
python build_index.py
```

第一次跑会调用大量 embedding API。几万条消息大概几分钟、几百次批量调用。**跑之前提醒用户这会产生 API 费用。**

如果用户有手写的 session 摘要：

```bash
python build_index.py /path/to/session_archive.md /path/to/jsonl_dir
```

摘要是可选的。没有摘要系统能跑，但情感类、关系类的召回会明显弱——那类内容通常没有可搜的专有名词，靠摘要的语义压缩才能召回。技术类召回主要靠 jsonl 原文，不受影响。

### 5. 启动搜索服务

```bash
nohup python search_server.py > /tmp/rag_server.log 2>&1 &
```

**坑：** `search_server.py` 是阻塞进程。直接前台跑会把你的会话挂住，之后什么都做不了。一定要放后台。

生产环境建议写 systemd unit（`Restart=always`），崩了自动起。

三个入口的区别：
- `search_server.py` — 完整入口，初始化 LanceDB + FTS + recall agent 后启动 HTTP。**用这个。**
- `server.py` — HTTP handler 和路由定义，被上面那个导入。
- `search.py` — 命令行检索工具，不启服务，调试用。

### 6. 验证

```bash
curl "http://127.0.0.1:15200/hybrid?q=测试&top_k=3"
```

返回 `{"results": [...]}` 格式、每项有 `text` 和 `source` 字段就算通了。`results` 是空数组说明索引是空的，回到第 4 步看输出。连接被拒说明服务没起来，看 `/tmp/rag_server.log`。

### 7. 配 hook

在 `.claude/settings.json` 里加：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash /ABSOLUTE/PATH/TO/claude-session-rag/hooks/memory_recall.sh"
          }
        ]
      }
    ]
  }
}
```

**必须是绝对路径**，用 `pwd` 的输出替换，不要留 `/path/to/`。

`hooks/memory_recall.sh.example` 需要复制成 `memory_recall.sh` 再改里面的路径。

---

## 数据源：最容易搞错的地方

这一节比其他所有内容都重要。装的时候不影响，但索引质量完全取决于它。

Claude Code 的 `.jsonl` 里，一条 assistant 消息的 `content` 数组可能包含 `text` 块和 `tool_use` 块。**如果用户通过插件（Telegram、Discord 等）收发消息，实际说出去的话在 `tool_use` 块里，`text` 块里装的是另一回事。**

Claude Code 从 2.1.183 起有个行为：某轮回复如果没有可见的 text 输出（比如只用了消息插件），系统会注入一句提示要求产生可见输出。于是模型会在发完插件消息后，额外写一句简短的终端旁白应付掉这个注入——类似"已回复用户'好的'"。

**这些旁白就是 `text` 块的内容。它们不是对话，是 bug 的副产物，索引它们等于索引一堆碎碎念。**

同理，用户侧的消息如果经由插件进来，实际文字包在 `<channel>` 标签里，`user` 轮次的其余部分是各种系统注入（compact 摘要、命令输出、位置触发等），占比可能超过 80%。

所以提取逻辑应该是：

- assistant 侧：取 `tool_use` 块里的实际回复内容，跳过 `text` 块
- user 侧：从 `<channel>` 标签里提取，剔除系统注入
- 两侧都要，只索引一侧会造成大面积漏检

**如果用户不用消息插件、直接在终端里对话**，那 `text` 块就是真实内容，按常规提取即可。装之前先确认用户是哪种用法。

判断方法：随便打开一个 jsonl，看 assistant 轮次里 `text` 块和 `tool_use` 块的比例。如果绝大多数轮次是纯 `tool_use`、且 `text` 块普遍很短（几十字），那就是插件场景。

---

## 检索行为的几个特性

装完之后用户可能会问，先说清楚免得他们调错方向。

**向量分数不能用来判断相关性。** bge-m3 对几乎任何中文输入都给出 0.5 左右的余弦相似度——库里完全没有的编造词是 0.50，真实命中是 0.54。这个差距不足以设阈值。**空召回判断只能靠 BM25**：命中给正常分数，没命中就是没命中，这才是可用的判据。

**RRF 融合分只反映排名，不反映质量。** 分数是 `1/(60+rank)`，top-1 恒等于 0.0164，前几名全挤在 0.016–0.033。想调阈值就得让接口返回融合前的原始分。

**chunk 不能切太碎。** 按单条消息切块会让平均长度掉到 80 字符左右，短文本的 embedding 语义信号很弱，容易被表面词汇主导。合并相邻消息成窗口（3–5 条、300–500 字符）之后，检索质量有明显改善。合并时注意断点：跨 session 不合并、时间间隔过大不合并。

**词汇鸿沟是真实存在的。** 用户用生活语言提问（"听歌的板块"），索引里存的是实现语言（"song-card display:none"），BM25 匹配不上，向量也桥不过去。`enrich_entities.py` 生成的 `aliases` 字段就是为这个准备的——它只在主路零命中时作为兜底调用，不进主搜索池，避免挤占候选位。

---

## 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `externally-managed-environment` | 系统 Python 保护 | `--break-system-packages` 或建 venv |
| 建索引后搜什么都返回空 | `JSONL_DIR` 指错了 | `ls $JSONL_DIR/*.jsonl` 确认有文件 |
| 连接被拒 | 服务没起来 | 看 `/tmp/rag_server.log` |
| 会话挂住不动 | `search_server.py` 前台跑了 | 杀掉，加 `nohup ... &` |
| hook 配了但没注入 | 路径不是绝对路径，或 `rg` 不在 PATH | 检查 settings.json，跑 `rg --version` |
| 搜出来全是碎碎念 | 提取了 `text` 块而非 `tool_use` | 见上面「数据源」一节 |
| 明明聊过却搜不到 | 内容不在索引里，不一定是检索问题 | 先 `rg` 原文确认内容存在，再查索引 |
| 每次都返回固定条数、从不空手 | 没有阈值，`top_k` 恒定 | 正常行为，空召回判断走 BM25 零命中分支 |

---

## 排障时的原则

这套系统里，"改了"和"生效了"是两件事，中间断过很多次。给用户下结论前先跑命令确认：

- 不要凭代码推断运行时状态。`grep` 到某个字段存在，不代表它有数据；某个函数在，不代表它被调用。
- LanceDB 表结构直接查：`table.to_pandas().columns`。字段可能只存在于 jsonl 侧而不在向量库里。
- "索引里没有" 和 "检索不到" 要分开验。先 `rg` 原文，再查索引，最后才怀疑排序。
- 时间戳是 UTC。用户说的时间通常是本地时区，换算错八小时会框到完全不同的内容，而且错得很隐蔽。

---

## 安全

- 搜索服务无鉴权，只绑 `127.0.0.1`，**不要暴露 15200 端口**。
- 这些路径包含真实对话内容，绝不能进版本库：`.env`、`session_archive.md`、`*.jsonl`、`memory_db/`、`eval_baseline/`。

- **`eval_baseline/` 的 ignore 规则要用白名单写法。** 按扩展名逐个排除（`eval_baseline/*.json`、`eval_baseline/*.jsonl`）只挡得住已知类型，以后往里放个 `.csv` 或 `.txt` 就会漏出去。改成整个目录排除再放行示例文件：

  ```gitignore
  eval_baseline/
  !eval_baseline/*.example.json
  ```

  `exclude_ranges.json` 之类的运行时配置同理——推示例文件，真实那份进 ignore。
- 帮用户提交代码前跑一遍 `git status`，确认没有数据文件被 add。
