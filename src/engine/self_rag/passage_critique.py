"""
IsRel Token 实现：对检索召回的 passage 做相关性批判。
"""

import json
import re
from typing import Any, Optional

from .base import LLMClient, Relevance


class PassageCritiqueEngine:
    """
    相关性批判引擎。

    对应 Self-RAG 的 IsRel token：模型对检索器召回的每个 passage
    独立判断其是否与用户当前研究真正相关，过滤噪声。
    """

    PROMPT_TEMPLATE = """你是 Future-Agent 的相关性批判模块。你的任务不是判断"这篇论文和用户是否都在做 AI"，而是判断"这篇论文的方法论能否直接迁移到用户当前的技术栈，解决同类工程痛点"。

    【通用判断原则】
    相关性 ≠ 关键词重叠。请严格区分以下两种情况：

    1. 方法论同构（应判 Relevant）：两篇论文研究的是同一类工程问题的不同解法。例如：
       - 用户研究"按需检索填补信息缺口"（Self-RAG）
       - 论文研究"主动信息检索优化上下文"（Active Information Seeking）
       - 两者都是"信息缺失时如何获取外部知识来改进生成"，方法论可直接迁移。

    2. 仅关键词共享（应判 Irrelevant）：两篇论文都在 NLP/AI 领域，但技术方向完全不同。例如：
       - 用户研究"句子语义嵌入"（SBERT）
       - 论文研究"句法成分解析"（Constituent Parsing）
       - 两者都涉及预训练模型，但一个做语义向量，一个做句法树，方法无法迁移。

    3. 知识缺口填补（应判 Relevant）：用户笔记中反复提及但未解决的痛点，论文提供了新的解决思路。

    用户当前研究状态：
    {note}

    候选论文：
    标题：{title}
    摘要：{abstract}

    【示例 1 —— Relevant（方法论同构）】
    用户笔记：正在研究 Self-RAG 的按需检索机制，解决 LLM 幻觉和事实性错误。
    论文：Context Training with Active Information Seeking
    判断：Relevant。理由：用户研究"信息缺失时按需检索"，论文研究"主动检索优化上下文"，两者是同一类问题的不同解法，方法论高度同构，可直接迁移。

    【示例 2 —— Irrelevant（仅关键词共享）】
    用户笔记：正在研究 SBERT 句子嵌入和语义向量检索。
    论文：Exploiting Pre-trained Transformers for Sequence-to-Sequence Constituent Parsing
    判断：Irrelevant。理由：用户核心技术是"语义向量表示"，论文核心技术是"句法结构分析"。两者虽都涉及预训练模型，但方法论完全不同，无迁移价值。

    【示例 3 —— Irrelevant（方向无关）】
    用户笔记：正在研究 RAG 检索增强生成，通过外部检索提升生成质量。
    论文：Inference-Time Machine Unlearning via Gated Activation Redirection
    判断：Irrelevant。理由：用户研究的是"检索增强生成"，论文研究的是"删除训练数据影响以保护隐私"。两者目标完全不同，不存在方法迁移路径。

    【示例 4 —— Relevant（知识缺口填补）】
    用户笔记：正在研究 LLM 的幻觉问题和事实准确性，关注测试时优化。
    论文：Query-Conditioned Test-Time Self-Training for Large Language Models
    判断：Relevant。理由：用户核心痛点是"LLM 事实性错误"，论文直接研究"测试时自训练纠正模型误解"，精准填补知识缺口。

    现在请判断这篇论文。你必须在 reason 中明确回答三个问题：
    1. 用户笔记的核心技术方向是什么？
    2. 这篇论文的核心技术方向是什么？
    3. 两者是"方法论同构"、"知识缺口填补"、还是"仅关键词共享/方向不同"？

    输出要求（必须严格用英文 JSON）：
    {{"relevance": "Relevant|Irrelevant", "reason": "用户研究XX，论文研究YY。两者是方法论同构/知识缺口填补/仅关键词共享，因此..."}}
    """

    def __init__(self, llm_client: LLMClient, config: Optional[dict] = None):
        self.llm = llm_client
        self.config = config or {}

    def critique(self, note_context: str, paper: Any) -> Relevance:
        """
        对单篇论文做相关性批判。

        Args:
            note_context: 用户笔记上下文（可包含多条 chunk 拼接）。
            paper: 具有 .title 和 .abstract 属性的论文对象（或 dict）。

        Returns:
            Relevance.RELEVANT 或 IRRELEVANT
        """
        if hasattr(paper, "get"):
            title = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "")
        else:
            title = getattr(paper, "title", "Unknown")
            abstract = getattr(paper, "abstract", "")

        prompt = self.PROMPT_TEMPLATE.format(
            note=note_context[:1500],
            title=title,
            abstract=abstract[:2000]
        )

        raw = self.llm.complete(prompt, temperature=0.0, max_tokens=128)
        return self._parse(raw)

    def critique_batch(self, note_context: str, papers: list) -> list:
        """
        批量批判（顺序调用，后续可改为并行）。
        返回过滤后的 Relevant papers。
        """
        results = []
        for p in papers:
            rel = self.critique(note_context, p)
            if rel == Relevance.RELEVANT:
                results.append(p)
        return results

    def _parse(self, raw: str) -> Relevance:
        try:
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                val = obj.get("relevance", "").strip()
                return Relevance(val)
        except (json.JSONDecodeError, ValueError):
            pass

        raw_up = raw.upper()
        if "RELEVANT" in raw_up and "IRRELEVANT" not in raw_up:
            return Relevance.RELEVANT
        return Relevance.IRRELEVANT