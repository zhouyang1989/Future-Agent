# LangChain 大模型应用开发框架学习文档

## 1. 引言

2022年底ChatGPT横空出世，大语言模型（Large Language Models, LLMs）以前所未有的能力席卷全球。然而，将LLM的能力真正落地为可用的应用程序，开发者面临的挑战远不止调用一个模型API那么简单。复杂的提示词工程、多轮对话的上下文管理、外部知识库的接入与检索、多步骤任务的编排与调度、工具链的精准调用、生产环境的流式传输与监控——这些系统级需求催生了对**LLM应用开发框架**的迫切需求。

在这一背景之下，LangChain应运而生。由Harrison Chase于2022年10月发布的LangChain，旨在成为“大模型应用开发的操作系统”——通过一套标准化、模块化的组件，将LLM、外部数据源、计算工具和组织逻辑无缝连接，让开发者能够以最小的摩擦构建复杂的AI应用。经过三年的快速迭代和社区共建，LangChain已从早期的实验性工具演变为AI应用开发领域最具影响力的开源框架之一，截至2025年全球开发者社区已贡献了超过500个扩展工具。2025年10月，LangChain正式发布v1.0版本，标志着该框架进入生产就绪的“工程化”阶段。

本文档旨在为读者提供一份全面、结构化的LangChain技术学习指南。我们将从框架的设计理念与核心架构入手，深入到各关键组件（Models、Prompts、Chains、Memory、Tools、Agents）的详细解析，剖析LangChain表达式语言（LCEL）的革命性组合范式，介绍模型输入输出标准化的最新进展，演示如何构建RAG（检索增强生成）应用，并涵盖生产部署、生态工具链及与其他框架的横向对比。全文以中文撰写，关键学术术语保留英文。


## 2. LangChain框架概述

### 2.1 设计理念与核心价值

LangChain的设计哲学根植于三个核心理念：**模块化**（Modularity）、**可组合性**（Composability）和**可扩展性**（Extensibility）。它将复杂的LLM应用开发过程分解为多个独立的、功能单一的组件，每个组件负责特定的职责，开发者可以根据需要灵活地选择和组合这些组件，像搭积木一样构建完整的应用。

LangChain的核心价值可以归纳为三个层面：

- **抽象层标准化**：提供统一的接口封装不同大语言模型（OpenAI、Anthropic、Google、HuggingFace、国产开源模型等）的调用方式，开发者无需关注底层API差异，实现模型无关（Model-Agnostic）的开发。
- **组件化架构**：将提示词管理、记忆管理、工具调用、推理规划等复杂能力拆解为可复用的独立模块，通过链式编排形成完整的应用流程。
- **生态整合力**：支持与主流云服务商的对象存储、向量数据库、消息队列等基础设施无缝对接，并提供LangSmith（可观测性监控）和LangServe（部署服务）等配套工具，覆盖开发、测试、部署、监控的全生命周期。

LangChain可应用于多种场景，包括：智能客服系统的上下文记忆管理、金融分析报告的自动生成、工业设备的故障诊断推理链、企业文档的智能问答系统、知识图谱驱动的信息检索等。相较于直接调用大模型API，LangChain框架可使开发效率提升3-5倍。

### 2.2 LangChain v1.0：里程碑式成熟发布

2025年10月20日，LangChain团队正式发布LangChain 1.0与LangGraph 1.0，这是两大框架的首个主要稳定版本，标志着AI Agent开发正式进入“工程化”阶段。

**v1.0的核心革新**包括：

| 特性 | 说明 |
|------|------|
| **`create_agent` 抽象** | 提供声明式配置替代传统的手工编码，开发者只需定义工具链、记忆机制和规划策略即可快速构建Agent |
| **Middleware 中间件系统** | 在Agent执行循环的每一步提供细粒度控制，内置人工介入、摘要生成、PII（个人身份信息）脱敏等中间件，同时支持自定义中间件 |
| **Standard Content Blocks** | 跨模型提供商的标准输出规范，统一处理推理轨迹、引用标记和服务端工具调用，实现真正的模型无关开发 |
| **精简化的包结构** | 主包体积缩减58%，遗留功能移至 `langchain-classic` 子包，保持向后兼容的同时大幅优化依赖管理 |
| **LangGraph运行时支撑** | Agent底层由LangGraph提供可靠的持久化运行时，支持自动状态快照、原子性操作和长流程恢复 |

v1.0版本明确承诺：直到2.0版本之前**不会有破坏性变更**，确保了企业级应用的长期稳定性。

### 2.3 整体架构概览

LangChain的架构可以分为四层：

```
┌─────────────────────────────────────────┐
│              应用层 (Applications)         │
│     RAG问答 │ 智能客服 │ 文本摘要 │ 代码助手  │
├─────────────────────────────────────────┤
│          核心抽象层 (Core Abstractions)     │
│    Models │ Prompts │ Chains │ LCEL      │
├─────────────────────────────────────────┤
│         高级能力层 (Advanced Capabilities)   │
│   Memory │ Retrieval │ Agents │ Tools     │
├─────────────────────────────────────────┤
│       基础设施与生态层 (Infrastructure)      │
│  LangServe │ LangSmith │ LangGraph │ Hub  │
└─────────────────────────────────────────┘
```

**核心抽象层**提供与LLM交互的基础构建块：Models抽象统一了各类模型的调用接口，Prompts提供模板化的提示词管理，Chains将多个步骤串联为完整的处理流水线，而LCEL（LangChain表达式语言）则作为贯穿始终的声明式组合“语言”。

**高级能力层**在核心抽象之上构建更复杂的系统能力：Memory维护对话的上下文和历史状态，Retrieval实现外部知识库的语义检索，Tools定义可供LLM调用的外部功能接口，Agents通过推理-行动循环自主决策并使用工具。

**基础设施层**提供将原型推向生产所需的工程化支持：LangServe一键部署API服务，LangSmith全程追踪监控和评估，LangGraph支撑复杂工作流编排和持久化状态管理。


## 3. 核心组件详解

LangChain将LLM应用开发中的最常见需求抽象为六大核心模块：**Models**（模型层）、**Prompts**（提示管理）、**Chains**（处理链）、**Memory**（记忆系统）、**Tools**（工具接口）和**Agents**（智能代理）。这六大模块构成了LangChain能力体系的基本骨架。

### 3.1 Models（模型层）

Models组件是LangChain构建AI应用的基础，负责封装与各种语言模型的交互，为上层组件提供统一的模型调用能力。它主要涵盖三类模型：

- **LLMs（Large Language Models）** ：接收文本字符串输入，返回文本字符串输出的纯文本完成模型，如GPT-3.5、GPT-4、Claude等。
- **Chat Models（对话模型）** ：以消息序列（Messages）作为输入、返回聊天消息作为输出的对话模型，如GPT-4o、Claude 3.5 Sonnet等。消息类型包括系统消息（SystemMessage）、用户消息（HumanMessage）、AI消息（AIMessage）等。
- **Embedding Models（嵌入模型）** ：将文本映射为稠密向量的嵌入模型，用于语义相似度比较和向量检索。

LangChain通过统一的接口屏蔽了不同提供商的API差异。一个典型的模型调用示例：

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

messages = [
    SystemMessage(content="你是一位专业的技术顾问。"),
    HumanMessage(content="请简要解释什么是微服务架构。")
]
response = model.invoke(messages)
```

v1.0版本进一步强化了模型集成能力，通过**Standard Content Blocks**实现了跨提供商（OpenAI、Anthropic、Google等）的统一内容规范，包括对推理追踪（Reasoning Traces）、引用标记（Citations）和服务端工具调用的标准化处理。

### 3.2 Prompts（提示管理）

Prompts组件负责构建和优化模型的输入。它通过模板系统（PromptTemplate）实现结构化的提示词管理，包含变量插值、示例选择（Few-Shot示例）和格式控制等功能，使提示工程更加系统化和可维护。

**核心功能**：

- **PromptTemplate**：最基础的类型，将用户输入的变量插入预定义的模板。
- **ChatPromptTemplate**：专门用于对话模型的提示模板，支持多角色消息序列的构建。
- **FewShotPromptTemplate**：从示例集中动态选择相关示例，构建Few-Shot学习的提示词。
- **PipelinePromptTemplate**：支持多个PromptTemplate的组合与复用。

一个典型的提示模板示例：

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，请基于以下参考资料回答用户问题。"),
    ("human", "参考资料：\n{documents}\n\n问题：{question}")
])

prompt = template.invoke({
    "role": "金融合规专家",
    "documents": "法规第3章：所有金融交易必须记录在案...",
    "question": "这笔交易需要哪些合规文件？"
})
```

### 3.3 Chains（处理链）

Chains（处理链）是LangChain框架中最具代表性的核心概念。它将多个组件连接形成完整的工作流程——可以像一条流水线，将数据在多个步骤之间逐步传递：经过Prompts格式化、Models推理、输出解析（Output Parser）等组件。

传统的定义方式包括LLMChain（最基本的单独调用链）和SequentialChain（按顺序将多个链组合）。然而，从2025年起推荐的写法是使用**LCEL（LangChain表达式语言）** ，它采用简洁、函数式的流水线语法：

```python
chain = prompt | model | output_parser
```

这种写法的本质是：**前一步的输出作为后一步的输入**，通过管道操作符 `|` 将多个组件串联。

一个完整的多步骤流水线——先总结再翻译：

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

summarize_prompt = PromptTemplate.from_template(
    "请总结以下文本：\n\n{text}"
)
translate_prompt = PromptTemplate.from_template(
    "将以下内容翻译为西班牙语：\n\n{summary}"
)

summarize_chain = summarize_prompt | model
translate_chain = translate_prompt | model

full_chain = (
    summarize_chain
    | (lambda x: {"summary": x.content})
    | translate_chain
)

result = full_chain.invoke({
    "text": "气候变化正在导致海洋快速升温，这将对全球生态系统产生深远影响..."
})
```

### 3.4 Memory（记忆系统）

Memory组件负责维护应用的状态和历史信息。由于LLM本身是无状态的——每次API调用都不知道之前的对话内容，记忆系统的作用就是将历史交互注入到新的请求中，从而支持连贯的多轮对话。

LangChain提供多层次的记忆类型：

| 记忆类型 | 用途 | 代表实现 |
|---------|------|---------|
| **短期记忆** | 保存当前会话的对话历史，记录用户-助手之间的多轮交互 | `ConversationBufferMemory` |
| **长期记忆** | 结合向量数据库存储和检索历史关键信息 | `VectorStoreRetrieverMemory` |
| **摘要记忆** | 对过长对话历史自动生成摘要，控制上下文长度 | `ConversationSummaryMemory` |
| **工具记忆** | 记录外部API的调用历史和返回结果 | 自定义ToolMemory |

一个简单的短期记忆使用示例：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("我的订单号是12345")
memory.chat_memory.add_ai_message("好的，已查找到您的订单，目前正在配送中。")

# 后续调用时，记忆中的历史消息会自动注入到Prompt中
```

在v1.0版本中，记忆系统与LangGraph的持久化状态管理深度集成，支持自动状态快照（每步执行生成可恢复的检查点）和多级存储策略（内存缓存 + 对象存储的分层设计），使系统在服务中断后能够从断点恢复。

### 3.5 Tools（工具接口）

Tools组件使LLM能够与外部世界交互，将语言模型的推理能力扩展到调用API、查询数据库、执行代码、读写文件等实际操作。LangChain通过统一的Tool抽象，让LLM能够“意识到”有哪些工具可用，并自主决定何时使用以及如何使用这些工具。

核心要素：

- **工具名称（Name）** ：唯一标识符，供LLM识别和选择工具。
- **工具描述（Description）** ：自然语言描述工具的功能、适用场景和参数含义，帮助LLM理解工具能力。
- **输入Schema（Args Schema）** ：通过JSON Schema或Pydantic模型定义工具的输入参数格式。
- **执行函数（Function）** ：实际执行工具逻辑的Python函数。

一个自定义工具的定义示例：

```python
from langchain.tools import tool

@tool
def get_stock_price(code: str) -> str:
    """查询指定股票代码的最新价格。输入参数code为股票代码，如'AAPL'。"""
    # 实际实现中调用股票API
    return f"股票{code}当前价格为150.25美元"

tools = [get_stock_price]
```

LangChain预置了丰富的工具集，涵盖：Google搜索、Wikipedia查询、计算器、Python REPL、API请求、数据库SQL查询、文件系统操作等。v1.0版本的工具注册机制进一步简化为声明式配置。

### 3.6 Agents（智能代理）

Agents是LangChain最强大的核心概念，也是v1.0版本重点强化的方向。Agent的核心机制是：**让LLM作为“大脑”自主决策和调度工具**。它使用LLM的推理能力来分析用户请求、制定行动计划、选择合适的工具执行操作，并根据工具返回的结果重新评估，直到认为可以给出最终答案为止。

**Agent的核心执行循环（ReAct模式）** ：

```
用户输入 → 推理分析 → 判断是否需要工具
    ├── 需要工具 → 选择工具 → 执行工具 → 获取结果 → 回到推理阶段
    └── 不需要工具 → 生成最终答案 → 返回
```

在LangChain 1.0中，Agent的构建方式已从早期的 `initialize_agent` 演进为更简洁的 `create_agent` 抽象：

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_stock_price, search_tool, calculator],
    prompt="你是一位金融分析助手。请基于用户问题，选择合适的工具获取数据并给出分析结论。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "对比苹果和微软的当前股价，哪家更高？高出多少？"}]
})
```

v1.0版本引入的**Middleware中间件系统**为Agent提供了更精细的执行控制：

- **Human-in-the-loop中间件**：在关键决策点暂停，等待人工审批。
- **Summarization中间件**：当对话历史过长时自动生成摘要，防止上下文溢出。
- **PII Redaction中间件**：在发送给LLM之前自动脱敏个人身份信息。
- **自定义中间件**：开发者可以编写中间件，在Agent执行的任意点介入自定义逻辑。


## 4. LangChain表达式语言（LCEL）
### “搭积木式”的声明式编程范式

### 4.1 为什么需要LCEL？

早期LangChain（v0.1时代）采用的是 `LLMChain`、`SequentialChain` 等类来组合组件。这种面向对象的Chain定义方式虽然直观，但存在三个核心痛点：**配置繁琐**（需要手动管理各步骤之间的输入输出键名对应）、**不够灵活**（难以插入自定义处理逻辑或条件分支）、**难以从原型平滑过渡到生产**（无法自然支持流式传输、异步调用和批量处理）。

LCEL（LangChain Expression Language）的诞生正是为了解决这些问题。它提供了一种**声明式的、基于管道的**方法来组合链，使得构建复杂、生产级的任务链变得异常简单和直观。

### 4.2 Runnable协议与管道操作符

LCEL的核心是一个被称为 **Runnable协议** 的简单而强大的概念。任何实现了Runnable接口的组件都具有以下能力：

- **`invoke`**：标准同步调用。
- **`ainvoke`**：异步调用，支持asyncio。
- **`stream`**：流式返回结果，实现逐token输出。
- **`batch`**：批量处理多个输入。
- **`astream_log`**：实时获取中间步骤的日志。

管道操作符 `|` 是LCEL的语法核心。`a | b` 的含义是：**创建一个新的Runnable，它首先执行a，然后将a的输出作为b的输入**。

### 4.3 组合模式

LCEL支持多种组合方式：

**1. 顺序组合（Sequential）** ：
```python
chain = prompt | model | output_parser
```

**2. 并行组合（Parallel）** ：将输入同时传递给多个子链，收集所有结果。
```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain
)
# 三条子链并行执行，返回包含summary、keywords、sentiment的字典
```

**3. 条件分支（Conditional）** ：基于输入或中间结果动态选择不同的执行路径。
```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: "紧急" in x["text"], urgent_chain),     # 条件1：紧急处理
    (lambda x: "技术" in x["text"], technical_chain),   # 条件2：技术问题
    default_chain                                       # 默认路径
)
```

**4. 回退策略（Fallback）** ：主链失败时自动切换到备用方案。
```python
chain = primary_model | parser
chain_with_fallback = chain.with_fallbacks([backup_chain])
```

### 4.4 LCEL的核心优势

| 优势 | 说明 |
|------|------|
| **一致的接口** | 所有组件都遵循相同的Runnable协议，任何可以使用单个组件的地方都可以使用组合后的链 |
| **声明式语法** | 使用 `|` 操作符直观表达数据流动，代码简洁易读 |
| **自动获得生产特性** | 链自动继承流式（stream）、异步（async）、批量（batch）能力，只需改变调用方法 |
| **中间结果可观测** | 支持 `astream_log` 和LangSmith追踪，实时查看每个步骤的输入输出 |
| **可组合性极强** | 可将任何Runnable组合为更复杂的Runnable，构建任意复杂度的处理流水线 |

在LangChain v1.0中，LCEL已从可选特性成为框架的**核心基石**，贯穿于Models、Prompts、Chains、Tools和Agents等所有组件的组合之中。


## 5. 模型输入输出标准化
### v1.0的重大改进

### 5.1 Standard Content Blocks

v1.0版本最受关注的改进之一是Standard Content Blocks——一种**跨模型提供商的统一内容规范**。其核心目标是解决不同LLM提供商输出格式不一致的痛点。

在传统模式下，切换到不同模型（例如从OpenAI迁移到Anthropic）时，由于各家的消息格式、工具调用标记和结构化输出方式各异，导致应用代码需要大幅修改。Standard Content Blocks通过定义统一的数据结构解决了这一问题。除传统的纯文本消息外，该规范还定义了标准化格式来处理推理轨迹（例如o1系列“思考过程”的可见性）、引用标记（支持模型在回答中包含可追溯的引用来源）以及服务端工具调用（如Web搜索、代码解释器等无需客户端编码即可触发的远程能力）。

实际效果：某电商平台的实践表明，该规范使模型切换成本降低65%，跨团队协作效率提升40%。

### 5.2 结构化输出生成

v1.0版本将结构化输出（Structured Output）直接集成到Agent的主执行循环中，而非像传统方式那样通过额外的LLM调用来解析输出。

这一改进带来两个直接收益：**降低延迟**（消除了额外的LLM调用步骤）、**降低Token成本**（生成与解析合并在单次调用中完成）。开发者可以通过Tool Calling或Provider Native两种策略精细控制结构化输出的生成方式：

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class StockAnalysis(BaseModel):
    symbol: str = Field(description="股票代码")
    price: float = Field(description="当前股价")
    recommendation: str = Field(description="投资建议：买入/持有/卖出")
    confidence: float = Field(description="建议置信度0-1")

parser = PydanticOutputParser(pydantic_object=StockAnalysis)

chain = prompt | model | parser
# 自动获得Pydantic模型的类型安全和验证功能
```


## 6. 检索增强生成（RAG）

RAG（Retrieval-Augmented Generation）是LangChain最重要的应用范式之一。它通过将外部知识库的精准信息动态注入生成过程，形成“检索-增强-生成”的闭环，有效解决了LLM的知识时效性缺陷和幻觉问题。

### 6.1 RAG的数据处理流水线

LangChain为RAG提供了完整的数据处理流水线：

**第一步：文档加载（Document Loaders）** 。LangChain支持超过15种文档格式的加载，包括PDF（`PyPDFLoader`）、网页（`WebBaseLoader`）、数据库（`SQLDatabase`）、Markdown、CSV等。

**第二步：文本分割（Text Splitters）** 。使用语义分割策略将长文档切割为适合LLM处理的小块。最常用的是`RecursiveCharacterTextSplitter`，按段落、换行、句号、空格的优先级递归拆分：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 每块500个字符
    chunk_overlap=50,       # 相邻块重叠50字符，保持语义连贯
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

**第三步：向量嵌入（Embedding）** 。使用嵌入模型将文本块转化为稠密向量。推荐使用高质量的嵌入模型，如`text-embedding-3-small`（OpenAI）或`bge-small-en-v1.5`（开源）。

**第四步：向量存储（Vector Store）** 。LangChain支持多种向量数据库，包括Chroma、FAISS、Pinecone、Weaviate、Milvus等。将文档向量存入向量数据库后，即可在查询时执行语义相似度搜索：

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### 6.2 混合检索策略

为提高召回率，LangChain支持混合检索——同时使用稀疏检索（BM25，基于关键词匹配）和稠密检索（基于语义相似度）：

```python
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 关键词匹配30%，语义相似度70%
)
```

混合检索在中文场景下尤为重要：BM25可以精确匹配专业术语和实体名称，而语义向量则捕捉同义表达和概念关联。

### 6.3 构建完整的RAG链

使用LCEL将所有组件串联为完整的RAG问答链：

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

template = """基于以下参考资料回答用户问题。仅使用参考资料中的信息，如果找不到答案就说“找不到相关信息”。

参考资料：
{context}

用户问题：{question}
回答："""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

answer = rag_chain.invoke("法国的首都是什么？")
```

### 6.4 LangChain中的三种RAG模式

LangChain实践中提供了三种层级的RAG实现方式：

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| **极简版** | 使用内置的离线Embedding模型和内存存储，零配置开箱即用 | 快速原型验证 |
| **标准版** | 自定义文档加载器、分块策略、向量数据库和检索器 | 中小型生产应用 |
| **进阶版** | 混合检索、多路召回、重排序（Reranker）、查询转换（Query Transformation） | 企业级高精度场景 |


## 7. 环境搭建与快速入门

### 7.1 安装与环境配置

LangChain以Python为主要开发语言，提供完整的pip包管理：

```bash
# 安装核心包（v1.0+）
pip install langchain

# 安装常用集成包
pip install langchain-openai     # OpenAI集成
pip install langchain-community  # 社区集成
pip install langchain-chroma     # Chroma向量数据库集成

# 环境变量配置（.env文件）
OPENAI_API_KEY=sk-xxxxx
LANGCHAIN_API_KEY=ls_xxxxx       # LangSmith追踪
```

**版本要求**：LangChain 1.0要求Python 3.10及以上版本，Python 3.9因2025年10月EOL（终止支持）已被放弃。

### 7.2 第一个LangChain程序

最简5分钟上手示例：

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 1. 初始化模型
model = ChatOpenAI(model="gpt-4o-mini")

# 2. 构建消息
messages = [
    SystemMessage(content="你是一位友好的AI助手。"),
    HumanMessage(content="用一句话介绍什么是LangChain。")
]

# 3. 调用模型
response = model.invoke(messages)
print(response.content)
```

输出示例：“LangChain是一个用于构建由大语言模型驱动的应用程序的模块化框架。”

### 7.3 第一个LCEL链

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("请用{language}写一首关于{topic}的五言绝句。")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

poem = chain.invoke({"language": "中文", "topic": "春天"})
print(poem)
```

### 7.4 第一个Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def calculate(expression: str) -> str:
    """计算数学表达式的结果。例如：'2+3*4'。"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。输入参数city为城市名称。"""
    return f"{city}今天晴朗，气温22°C"

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[calculate, get_weather],
    prompt="你是一位生活助手。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天多少度？适合出门吗？"}]
})
```


## 8. 生产部署与生态工具链

### 8.1 LangServe：一键API部署

LangServe是一个开源库，可以将任何LangChain链或Agent一键部署为生产级REST API（基于FastAPI构建）：

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="RAG知识问答API")
add_routes(app, rag_chain, path="/qa")

# 启动：uvicorn app:app --host 0.0.0.0 --port 8000
```

部署后自动获得：`/qa/invoke`（标准调用）、`/qa/stream`（流式输出）、`/qa/batch`（批量处理）等端点，以及内置的Swagger文档和Playground测试界面。

### 8.2 LangSmith：全链路可观测性平台

LangSmith是LangChain官方提供的可观测性和评估平台：

- **追踪（Tracing）** ：记录每次链调用的完整链路，支持查看每步的输入输出、延迟和Token消耗。
- **监控（Monitoring）** ：实时追踪生产环境中的关键指标（延迟、错误率、Token用量）。
- **评估（Evaluation）** ：支持自动化评估数据集（Dataset）运行、人工标注反馈和A/B实验对比。
- **Playground**：在线调试提示词和链配置。

2026年，LangSmith进一步推出了**Fleet（原Agent Builder）** ，增加了Agent身份管理、共享与权限控制功能，支持企业内安全地管理Agent集群。

### 8.3 LangGraph：复杂工作流编排引擎

LangGraph是LangChain的底层运行时引擎，专注于处理复杂的、有状态的多步骤工作流。它解决了传统线性Chain难以处理的场景：

- **持久化状态管理**：自动快照每步执行状态，支持从断点恢复。
- **动态分支与循环**：基于运行时条件实现复杂流程编排。
- **人工介入点（Human-in-the-loop）** ：支持在关键节点等待人工审批。
- **插件系统**：支持自定义节点类型与执行策略。

### 8.4 工具链全景

```
LangChain开发工作流
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ LangSmith│    │LangChain│    │ LangServe│    │ LangSmith │
│  (调试)  │───→│ (构建)  │───→│  (部署)  │───→│  (监控)   │
└─────────┘    └─────────┘    └──────────┘    └──────────┘
     ↑                                              │
     └────────────── 反馈循环 ──────────────────────┘
```


## 9. 最佳实践与性能优化

### 9.1 提示词工程最佳实践

- **角色定义清晰**：在SystemMessage中明确角色身份和职责边界（如“你是金融合规专家，回答必须引用法律法规原文”）。
- **输出格式约束**：使用Output Parser强制结构化输出，确保下游解析可靠。
- **示例驱动（Few-Shot）** ：复杂任务应提供2-3个正确示例，引导LLM的输出格式和质量。
- **减少无关信息**：精简Prompt中的上下文，仅保留与当前问题直接相关的内容。

### 9.2 性能优化策略

**1. 流式输出降低首Token延迟**

```python
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)
```

流式输出使用户在LLM完成全部推理前即可看到部分结果，显著改善交互体验。

**2. 异步调用提升并发**

```python
import asyncio
results = await chain.abatch([{"question": q} for q in questions])
```

**3. 合理的Chunk策略**：根据场景选择合适的分块大小，Q&A场景建议500字符，摘要场景可更大。

**4. 模型选择与缓存**：简单任务使用小型模型（如gpt-4o-mini），复杂推理才启用大模型。对相同的输入启用结果缓存。

**5. 并行执行**：对互不依赖的子任务使用并行链，缩短总耗时。


## 10. LangChain与其他框架的横向对比

| 维度 | LangChain | LlamaIndex | Haystack | Dify |
|------|-----------|-----------|----------|------|
| **定位** | 通用LLM应用框架 | 数据索引与检索 | NLP管道框架 | 低代码LLM平台 |
| **核心优势** | Agent编排、生态最丰富 | 数据处理、索引结构强大 | 检索管道灵活 | 可视化拖拽构建 |
| **适用场景** | Agent、RAG、复杂工作流 | 大规模文档检索、数据分析 | 搜索系统、FAQ匹配 | 零代码快速原型 |
| **学习曲线** | 中等 | 较高 | 中等 | 低 |
| **开源协议** | MIT | MIT | Apache 2.0 | Apache 2.0 |
| **社区活跃度** | 最高（15万+ GitHub Star） | 高 | 中 | 高 |

**选型建议**：
- **构建Agent、RAG系统、复杂多步骤工作流**：首选LangChain，其生态和社区最为完善。
- **大规模文档索引和结构化数据检索**：LlamaIndex的索引结构和检索算法更为丰富。
- **快速搭建可交互的AI应用前台**：Dify提供极低的入门门槛。
- **定制化搜索管道**：Haystack的管道抽象在搜索系统上更为灵活。


## 11. 总结与展望

LangChain以系统化的方式解决了LLM应用开发中“从模型能力到可用的应用程序”的关键鸿沟。它将看似简单的“调API”扩展为一套完整的工程化方法——通过Models和Prompts标准交互接口，通过Chains和LCEL编排复杂流程，通过Memory维护会话状态，通过Tools和Agents赋予模型使用外部世界的能力。

**LangChain的核心贡献可以归纳为三个方面**：

1. **架构贡献**：建立了LLM应用开发的组件化范式。将Agent开发从“手工编码”提升为“工程化构建”，通过`create_agent`抽象、Middleware中间件系统和Standard Content Blocks等创新，为生产级Agent应用奠定了稳定的技术基础。

2. **组合性贡献**：LCEL（LangChain表达式语言）以声明式的管道语法和统一的Runnable协议，彻底改变了AI工作流的构建方式。从简单的“Prompt → Model → Output”到包含数百步骤的复杂Agent，LCEL都提供了统一的、可生产的组合范式。

3. **生态贡献**：LangSmith（可观测性）、LangServe（部署）和LangGraph（运行时）构成了完整的工具链，覆盖了从原型开发到生产部署、从监控追踪到持续优化的全生命周期。这使得LangChain不仅是开发框架，更是一个**完整的AI应用工程平台**。

展望未来，LangChain 1.0明确了继续向**智能体工程平台**演进的方向：在多模态Agent、自治Agent和低代码/无代码Agent构建器方面持续发力。对每一位从事LLM应用开发的工程师和研究者而言，掌握LangChain不仅是掌握一个工具，更是掌握一套与大语言模型协作的工程化思维和方法论。


## 附录A：LangChain版本变迁简史

| 时间 | 里程碑 | 关键变化 |
|------|--------|---------|
| 2022年10月 | LangChain首次发布 | 提供LLMChain、Agent等基础抽象，比ChatGPT早约1-2个月推出 |
| 2023年 | 快速增长期 | 社区爆炸式增长，集成数百种工具和模型，成为LLM应用开发首选框架|
| 2024年 | LCEL全面推广 | LCEL成为链组合的推荐方式，Runnable协议统一了各组件接口|
| 2025年10月 | **LangChain 1.0发布** | `create_agent`抽象、Middleware系统、Standard Content Blocks、底层迁移至LangGraph运行时|
| 2026年 | 1.0稳定迭代 | LangSmith Fleet、Sandboxes、Deep Agents等持续生态扩展|


## 附录B：核心API速查表

| API | 用途 | 示例 |
|-----|------|------|
| `model.invoke(msg)` | 模型同步调用 | `model.invoke("你好")` |
| `model.ainvoke(msg)` | 模型异步调用 | `await model.ainvoke("你好")` |
| `model.stream(msg)` | 模型流式输出 | `for chunk in model.stream(...)` |
| `chain.invoke(input)` | 链同步执行 | `chain.invoke({"q": "test"})` |
| `chain.batch(inputs)` | 链批量执行 | `chain.batch([{"q": "1"}, {"q": "2"}])` |
| `chain.astream_log(input)` | 实时日志流 | `async for event in chain.astream_log(...)` |
| `agent.invoke(...)` | Agent执行 | `agent.invoke({"messages": [...]})` |
| `add_routes(app, chain, path)` | 部署REST API | `add_routes(app, chain, "/qa")` |


## 参考文献

1. LangChain Official Documentation. LangChain v1.0 Overview. [docs.langchain.com](https://docs.langchain.com). 2025.

2. The LangChain Team. LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones. [langchain.com/blog](https://www.langchain.com/blog). 2025-10-22.

3. The LangChain Team. LangChain 1.0 now generally available. [changelog.langchain.com](https://changelog.langchain.com). 2025-10-22.

4. 《LangChain大模型开发实践》. 清华大学出版社. ISBN: 9787302692287. 2025.

5. 百度开发者平台. LangChain从入门到实战：2025年AI开发者的核心框架指南. 2026-04-25.

6. 腾讯云开发者社区. LangChain 核心组件剖析：Models、Prompts、Chains 详解. 2025-11-28.

7. 腾讯云开发者社区. 构建AI智能体：LangChain智能体. 2025-11-24.

8. Kanaries Docs. 详解：什么是 LangChain？如何使用 LangChain Chains？ 2025.

9. 阿里云开发者社区. LangChain LCEL深度解析：基于Runnable协议的声明式编程新范式. 2025-11-25.

10. 百度云. 基于LangChain构建高效RAG应用：从原理到实践的全链路解析. 2025-11-06.

11. 百度开发者平台. LangChain 1.0 正式发布：构建生产级智能体框架的关键突破. 2026-04-30.

12. The LangChain Team. March 2026: LangChain Newsletter. [langchain.com](https://www.langchain.com). 2026-04-01.

13. 36氪. 模型不再是关键？LangChain创始人：真正决定Agent上限的是运行框架. 2026-03-13.

