<h1 align="center">🔮 Future-Agent</h1>

<p align="center">
  <strong>Privacy-first personal knowledge enhancer — connect your vault to the world.</strong><br>
  <strong>隐私优先的个人知识增强助手 —— 连接你的知识库与外部世界。</strong>
</p>

<p align="center">
  <a href="https://github.com/yourname/future-agent/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Local--First-100%25%20Offline-green" alt="Local-First">
</p>

<p align="center">
  <a href="#features">核心功能</a> •
  <a href="#quickstart">快速开始</a> •
  <a href="#usage">使用指南</a> •
  <a href="#self-rag">Self-RAG v1.5</a> •
  <a href="#architecture">架构设计</a> •
  <a href="#privacy">隐私保护</a> •
  <a href="#contributing">贡献指南</a>
</p>

---

## 📖 简介

Future-Agent 读取你的本地笔记库（Obsidian Vault 或纯 Markdown 文件夹），将其转化为可语义检索的向量网络。每天自动从 arXiv 获取最新论文，基于你当前的知识状态推荐最相关的文献，最终生成可直接粘贴到 Obsidian Daily Notes 的 Markdown 日报。

**适合谁用？**

- 🧑‍💻 正在技术转型的工程师，需要持续跟踪前沿论文
- 📝 管理个人知识库的研究者或独立开发者
- ✍️ 希望建立「输入—消化—输出」闭环的技术写作者

---

<a id="features"></a>
## ✨ 核心功能

| 功能 | 说明 | 状态 |
|------|------|:--:|
| **知识库 Ingestion** | 解析 Obsidian Vault 或纯 Markdown 文件夹，提取笔记内容、标签、WikiLinks、Frontmatter | ✅ |
| **向量化索引** | 使用 SBERT 将笔记分块嵌入，存入本地 ChromaDB，建立可语义检索的知识网络 | ✅ |
| **arXiv 智能监控** | 按指定领域（cs.CL / cs.LG / cs.IR 等）批量获取最新论文元数据与摘要 | ✅ |
| **个性化推荐** | 基于笔记库语义状态，从当日 arXiv 论文中推荐 Top-K 最相关文献 | ✅ |
| **多样性约束** | 自动平衡「强相关延续」与「边界拓展」，防止信息茧房 | ✅ |
| **日报生成** | 生成 Markdown 格式推荐报告，可直接粘贴至 Obsidian Daily Notes | ✅ |
| **增量更新** | 基于 Chunk ID 实现增量索引，仅处理新增或修改的笔记 | ✅ |
| **Self-RAG 批判式推荐** | v1.5 新增：通过 LLM 对每篇候选论文做相关性批判（IsRel）、事实校验（IsSup）与效用评估（IsUse），过滤噪声推荐 | ✅ |

---

<a id="quickstart"></a>
## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/zhouyang1989/Future-Agent.git
cd Future-Agent
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. 配置项目

```bash
cp config.yaml config.local.yaml
```

编辑 `config.local.yaml`，填入你的真实路径：

```yaml
knowledge_source:
  reader_type: "obsidian"
  obsidian:
    vault_path: "/Users/你的用户名/Library/Mobile Documents/iCloud~md~obsidian/Documents/SecondBrain"
    ignore_folders: ["Daily", "Personal", "Archive"]
```

> ⚠️ `config.local.yaml` 已被 `.gitignore` 保护，**永远不会被提交到 Git**，确保你的隐私路径不会泄露。

### 4. 使用演示模式体验（无需真实笔记库）

```bash
# 1. 将演示知识库摄入向量库
python src/cli.py ingest --source ./demo-vault/ --reader markdown

# 2. 抓取最近 7 天 arXiv 论文
python src/cli.py fetch --days 30 --max-results 500

# 3. 生成今日推荐日报
python src/cli.py recommend --top-k 5 --diversity boundary_mix

# 4. 查看系统状态
python src/cli.py status
```

---

<a id="usage"></a>
## 📚 使用指南

### CLI 命令总览

```bash
# 查看帮助
python src/cli.py --help

# 子命令帮助
python src/cli.py ingest --help
python src/cli.py fetch --help
python src/cli.py recommend --help
```

### `ingest` —— 知识库摄入

将本地笔记库解析、分块、嵌入并索引到 ChromaDB。

```bash
# 从配置文件读取路径并摄入 Obsidian Vault
python src/cli.py ingest

# 摄入指定路径的 Obsidian Vault
python src/cli.py ingest --source /path/to/your/vault --reader obsidian

# 摄入纯 Markdown 文件夹
python src/cli.py ingest --source ./demo-vault/ --reader markdown

# 强制全量重建索引（默认是增量更新）
python src/cli.py ingest --force
```

**增量更新机制**：系统通过 Chunk ID 检查向量库中是否已存在，仅对新增或变更的文本块重新嵌入，避免重复计算。

### `fetch` —— arXiv 论文抓取

按配置的分类和日期范围批量获取论文。

```bash
# 抓取最近 30 天论文（默认）
python src/cli.py fetch

# 抓取最近 3 天
python src/cli.py fetch --days 3

# 限制抓取数量
python src/cli.py fetch --max-results 50

# 仅缓存原始数据，跳过向量化（用于预缓存）
python src/cli.py fetch --no-index
```

### `recommend` —— 生成推荐日报

基于当前知识库状态，从已抓取的 arXiv 论文中推荐最相关的文献。

```bash
# 生成日报并输出到指定目录
python src/cli.py recommend --output ./data/reports/

# 同时写入 Obsidian Inbox（需在配置中指定 inbox_path）
python src/cli.py recommend --to-obsidian

# 调整推荐数量
python src/cli.py recommend --top-k 5

# 切换多样性策略
python src/cli.py recommend --diversity boundary_mix
# 可选：strong_only（保守）/ boundary_mix（平衡）/ cross_domain（激进跨界）
```

---

<a id="self-rag"></a>
## 🧠 Self-RAG 批判式推荐（v1.5）

v1.5 在 v1.0 向量检索的基础上，引入 LLM 驱动的三阶段批判链，对候选论文做逐层过滤，显著降低推荐噪声。

### 工作流程

```
v1.0 Matcher 粗排（本地向量，零 LLM 成本）
  ↓
[Retrieve Decision] —— 判断今日候选池是否值得推荐
  ↓
[IsRel] —— 逐篇批判：论文是否真正与用户研究相关？
  ↓
[生成推荐理由] —— 基于用户笔记上下文，生成中文推荐语
  ↓
[IsSup] —— 逐句校验推荐语是否被论文摘要事实支持
[IsUse] —— 评估论文对用户当前研究的效用分值（1-5 分）
  ↓
输出通过批判链的最终推荐日报
```

### 启用方式

```bash
# 基础用法（balanced 模式，推荐数量 5 篇）
python src/cli.py recommend --self-rag

# 指定模式与推荐数量
python src/cli.py recommend --self-rag --mode strict --top-k 5
```

### 推理模式说明

| 模式 | IsSup 保留条件 | 适用场景 |
|------|--------------|----------|
| `strict` | Fully Supported，或 Partially Supported 且效用 ≥ 4 分 | 精读，要求推荐理由每句话都有摘要依据 |
| `balanced`（默认）| Fully 或 Partially Supported | 日常使用，兼顾数量与质量 |
| `creative` | Fully / Partially / No Support | 头脑风暴，允许合理推断 |
| `fast` | 跳过 IsSup 校验 | 降低 API 调用成本，快速出结果 |

### 配置 LLM

在 `config.local.yaml` 中填入 LLM 配置（v1.0 无需此步骤）：

```yaml
self_rag:
  enabled: true
  llm:
    provider: "kimi"           # 可选：kimi / openai / custom（本地 vLLM）
    model: "moonshot-v1-8k"
    api_key: "your-api-key"    # 或留空，从环境变量 KIMI_API_KEY 读取
```

> Self-RAG 仅在推荐阶段向 LLM 传输论文摘要片段与笔记摘录，**不会上传完整笔记库**。支持本地 vLLM / Ollama 部署实现 100% 离线运行。

### 报告输出示例

生成的 Markdown 报告结构如下：

```markdown
# 📚 Future-Agent Daily Report —— 2026-05-08

## 你的知识状态快照
- 当前笔记库：42 篇技术笔记，覆盖 6 个主题
- 最近关注：RAG、向量检索、端侧部署

---

## 🔥 今日推荐（Top 5）

### 1. [论文标题] —— 强相关 ⭐
**作者**：XXX et al. | **arXiv**: 2505.01234 | **分类**: cs.CL
**关联笔记**: [[05_rag]]、[[03_hnsw]]
**推荐理由**: 这篇论文提出了一种新的检索粒度控制方法，与你笔记中关于 HNSW 多层路由的讨论直接相关。

### 2. [论文标题] —— 边界拓展 🌱
...
```

### `status` —— 查看系统状态

```bash
python src/cli.py status
```

输出当前配置概览、向量库统计（笔记 / 论文数量）以及运行时目录检查。

---

<a id="architecture"></a>
## 🏗️ 架构设计

### 分层架构

Future-Agent 采用严格的分层架构，每层只依赖下层：

```
CLI (cli.py)
  ↓
Engine (matcher.py / self_rag/) + Generators (report_generator.py)
  ↓
Storage (vector_store.py) + Embeddings (embedder.py)
  ↓
Processors (chunker.py)
  ↓
Readers (base.py / obsidian.py / markdown.py) + Fetchers (arxiv_fetcher.py)
```

### 目录结构

```text
future-agent/
├── demo-vault/                 # 演示知识库（6 篇脱敏学习笔记，可直接体验）
│   ├── 01_word2vec.md
│   ├── 02_SBERT.md
│   ├── 03_HNSW.md
│   ├── 04_RAG.md
│   ├── 05_Chromadb.md
│   └── 06_langchain.md
│
├── data/                       # 运行时数据（已被 .gitignore 保护，永不提交）
│   ├── raw/                    # arXiv 原始缓存（JSON）
│   ├── processed/              # 嵌入缓存（.npy + .json）
│   ├── chroma/                 # ChromaDB 本地持久化
│   └── reports/                # 生成的 Markdown 日报
│
├── src/                        # 核心源码
│   ├── readers/                # 知识库读取层（解耦设计，支持多源扩展）
│   │   ├── base.py             # BaseReader 抽象接口 + Document 数据类
│   │   ├── obsidian.py         # ObsidianReader：解析 WikiLinks + Frontmatter
│   │   ├── markdown.py         # MarkdownFolderReader：纯 Markdown 文件夹
│   │   └── __init__.py         # Reader 工厂（READER_REGISTRY + get_reader）
│   ├── fetchers/               # 外部数据获取层
│   │   └── arxiv_fetcher.py    # ArxivFetcher + Paper 数据类 + 本地缓存
│   ├── processors/             # 文本处理层
│   │   └── chunker.py          # Chunker：标题感知分块 + 重叠保留
│   ├── embeddings/             # 向量化层
│   │   └── embedder.py         # SBERT 封装 + 批量编码 + 本地缓存
│   ├── storage/                # 存储层
│   │   └── vector_store.py     # VectorStore：ChromaDB 双集合管理
│   ├── engine/                 # 核心推荐引擎
│   │   ├── matcher.py          # Matcher：相似度计算 + 多样性约束（v1.0）
│   │   └── self_rag/           # Self-RAG 批判式推荐引擎（v1.5）
│   │       ├── self_rag_recommender.py  # 主入口：三阶段批判流程
│   │       ├── passage_critique.py      # IsRel：相关性批判
│   │       ├── generation_validator.py  # IsSup / IsUse：事实校验与效用评估
│   │       ├── llm_wrapper.py           # LLM 客户端（Kimi / OpenAI / vLLM）
│   │       └── base.py                  # 基础类型（Relevance, SupportLevel）
│   ├── generators/             # 报告生成层
│   │   └── report_generator.py # ReportGenerator：Jinja2 → Markdown
│   ├── config_loader.py        # Config 类（支持点号分隔嵌套键访问）
│   └── cli.py                  # Typer CLI 统一入口
│
├── templates/                  # Jinja2 模板
│   └── daily_report.md.j2      # 日报模板（项目同时内置默认模板）
│
├── notebooks/                  # 实验与探索
│   └── exploration.ipynb
│
├── tests/                      # 单元测试（待补充）
│   ├── test_chunker.py
│   └── test_readers.py
│
├── config.yaml                 # 主配置文件模板
├── requirements.txt            # Python 依赖列表
├── .gitignore                  # Git 忽略规则
└── README.md                   # 本文件
```

### 技术栈

| 层级 | 技术选型 | 选型理由 |
|------|----------|----------|
| **CLI 框架** | [Typer](https://typer.tiangolo.com/) | 类型安全，自动生成帮助文档 |
| **终端 UI** | [Rich](https://rich.readthedocs.io/) | 进度条、表格、面板美化 |
| **笔记解析** | 自定义 Reader（正则 + PyYAML） | 深度解析 Obsidian 特有语法（WikiLinks、Frontmatter、标签） |
| **文本嵌入** | [sentence-transformers](https://www.sbert.net/) (`paraphrase-multilingual-MiniLM-L12-v2`) | 轻量、多语言、支持 MPS（Apple Silicon） |
| **向量检索** | [ChromaDB](https://www.trychroma.com/) + HNSW | 本地持久化、零运维 |
| **数据获取** | [arxiv](https://pypi.org/project/arxiv/) 官方库 | 内置速率限制与自动重试 |
| **报告生成** | [Jinja2](https://jinja.palletsprojects.com/) | 模板化 Markdown 输出 |
| **配置管理** | PyYAML | 支持嵌套键访问（如 `knowledge_source.obsidian.vault_path`） |

### 核心设计模式

1. **抽象基类 + 工厂模式**：`BaseReader` 定义统一接口，`READER_REGISTRY` 实现工厂注册，便于扩展新笔记源（如 Notion、Logseq）。
2. **数据类驱动**：所有跨层传输的数据对象均为 `@dataclass`：`Document`、`Chunk`、`Paper`、`Recommendation`。
3. **懒加载（Lazy Loading）**：`Embedder.model` 属性延迟加载 SBERT，避免导入时耗时。
4. **配置注入**：所有模块的构造函数均接收 `config: Optional[dict]`，从 `Config.raw` 字典读取配置项，保持模块可独立测试。

### 多级缓存策略

- **arXiv 缓存**：`ArxivFetcher` 按查询语句 MD5 缓存结果到 `data/raw/arxiv/*.json`，避免重复请求 API。
- **嵌入缓存**：`Embedder` 按 chunk 内容 + 模型名计算 MD5，缓存 `.npy` 向量文件到 `data/processed/embed_cache/`。
- **增量索引**：`VectorStore.upsert_notes()` 检查 ID 是否已存在，仅新增/修改的笔记重新嵌入；论文库则每日全量重建。

---

<a id="privacy"></a>
## 🔒 隐私保护

Future-Agent 采用 **本地优先（Local-First）** 架构：

- ✅ **笔记内容永不离开本机**：所有嵌入计算均在本地 SBERT 完成
- ✅ **无需注册与登录**：零账号体系，零云端同步
- ✅ **向量库本地持久化**：ChromaDB 数据仅存于 `./data/chroma/`
- ✅ **配置隔离**：`config.local.yaml` 被 `.gitignore` 严格保护，避免路径等隐私信息泄露
- ✅ **只读模式**：系统仅读取你的笔记库，不会修改任何已有文件（仅生成新的推荐报告）

> **v1.0 核心功能完全不调用外部 LLM API**，所有计算本地完成，无需任何 API Key 即可运行完整流程。

---

<a id="contributing"></a>
## 🤝 贡献指南

Future-Agent 目前处于早期验证阶段，非常欢迎社区贡献！

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 Future-Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<p align="center">
  <sub>Built with ❤️ by someone who believes knowledge should find you, not the other way around.</sub>
</p>
