"""
IsSup / IsUse Token 实现：对生成内容做事实校验与效用评估。
"""

import json
import re
from typing import Any, List, Optional, Tuple

from .base import LLMClient, SupportLevel


class GenerationValidator:
    """
    生成校验引擎。

    对应 Self-RAG 的 IsSup（事实支持度）与 IsUse（效用评分）：
    在 Segment 生成后，批判其是否被检索 passage 支持，以及整体是否有用。
    """

    PROMPT_TEMPLATE = """你是 Future-Agent 的事实校验与效用评估模块。你必须逐句拆解推荐语中的每个断言，并在摘要中找到对应依据。不允许输出模糊的中间答案。

    生成的推荐语：
    {generation}

    来源论文摘要：
    {passages}

    【步骤 1：主张拆解】
    将推荐语拆分为独立断言（每个句号或逗号分隔的完整观点）。对每个断言，在摘要中找到最直接对应的文字，并逐条写出分析（格式：「主张 → 摘要依据 → 判断类型」）。

    【步骤 2：依据标注规则】
    - direct_quote：摘要中有几乎相同的描述，或明确提及了该事实。
    - reasonable_inference：摘要描述了背景和问题，推荐语据此做出合理延伸，但摘要未直接陈述该结论。推断必须与摘要逻辑一致，不能矛盾。
    - unsupported：摘要中完全找不到依据。
    - contradiction：推荐语与摘要内容存在事实冲突。

    特别注意：
    - 如果推荐语中出现"论文提出了一种..."、"核心贡献在于..."、"作者发现..."等断言，必须在摘要中找到明确的"We propose..."、"Our contribution is..."、"We show that..."或同等强度的直接声明。如果摘要只描述了背景、问题或现有方法局限，而没有明确声明本文的新方法，则该断言必须判为 unsupported 或 reasonable_inference，绝不能判为 direct_quote。
    - 如果推荐语将论文的某个"背景/动机"描述为论文本身的"贡献/方法"，必须判为 unsupported。

    【步骤 3：支持度判定】
    根据步骤 1 的逐条分析，得出整体支持度：
    - Fully Supported：推荐语中的核心方法论断言（如"提出了XX方法"、"使用YY技术"、"在ZZ数据集上实验"）全部能在摘要中找到 direct_quote。允许存在少量合理的价值推断（如"对...有重要意义"），但这些推断必须与摘要逻辑一致，不能脑补。
    - Partially Supported：核心方法论断言有 direct_quote 支撑，但推荐语中包含明显的推断延伸（如将论文的"背景动机"扩展为"核心贡献"、或对未来影响的展望）。推断不与摘要矛盾，但也不是摘要直接陈述的。
    - No Support：核心方法论断言（如"提出了XX方法"）在摘要中找不到 direct_quote，或推荐语把摘要中完全没有提及的内容包装为论文贡献。
    - Contradictory：推荐语与摘要存在事实冲突（如摘要说"需要外部数据"，推荐语说"无需外部数据"）。

    特别注意：
    - "评价性语句"（如"具有重要意义"、"值得注意"、"提供了新思路"）本身不需要 direct_quote，但前提是它所评价的**对象**（即论文的具体方法/发现）必须有 direct_quote 支撑。
    - 如果推荐语通篇只有评价没有事实，或事实与摘要不符，判 No Support。

    【步骤 4：效用评估（1-5 分）】
    utility 评分与 support 评分独立。即使推荐语的推断部分较多（Partially Supported），只要论文本身对用户研究有价值，就应该给高分。

    - 5 分：论文精准命中用户当前核心痛点，能直接指导下一步工程决策。
    - 4 分：与用户技术栈有直接借鉴意义。
    - 3 分：有参考价值，属于外围知识。
    - 2 分：几乎无帮助，属于常识。
    - 1 分：完全无关或误导。

    评分时必须回答：如果用户今天只能读一篇论文，这篇是不是优先选择？不要给安全中间分。

    【输出格式要求】
    完成以上步骤 1-4 的分析后，在所有分析文字的最后，单独输出一行严格的 JSON 对象（不加 markdown 包裹，直接输出花括号，不得换行）：
    {{"support": "Fully Supported|Partially Supported|No Support|Contradictory", "utility": 1-5, "reason": "一句话中文解释，指出最严重的问题或最大的价值点"}}

    警告：如果你跳过步骤 1-3 的主张逐条分析，直接输出 JSON，系统会丢弃这条推荐。请先完成完整分析，再在末尾输出 JSON。
    """

    def __init__(self, llm_client: LLMClient, config: Optional[dict] = None):
        self.llm = llm_client
        self.config = config or {}
        self.is_sup_weight = self.config.get("self_rag", {}).get("is_sup_weight", 1.0)
        self.is_use_weight = self.config.get("self_rag", {}).get("is_use_weight", 1.0)

    def validate(self, generation: str, passages: List[Any]) -> Tuple[SupportLevel, int]:
        """
        对生成文本做事实校验与效用评分。

        Args:
            generation: 模型生成的推荐语或摘要。
            passages: 支撑该生成的论文列表（需有 .abstract）。

        Returns:
            (support_level, utility_score)
        """
        passages_text = "\n\n".join(
            f"Paper {i + 1}: {getattr(p, 'abstract', str(p))[:800]}"
            for i, p in enumerate(passages)
        )

        prompt = self.PROMPT_TEMPLATE.format(
            generation=generation[:1500],
            passages=passages_text
        )

        raw = self.llm.complete(prompt, temperature=0.0, max_tokens=1024)
        return self._parse(raw)

    def compute_segment_score(self,
                              generation_prob: float,
                              support_level: SupportLevel,
                              utility_score: int) -> float:
        """
        计算 Segment-level Beam Search 的综合得分。

        Score = log P(generation) + w_sup * P(IsSup=ideal) + w_use * P(IsUse=ideal)

        在 prompt-based 阶段，我们用启发式映射代替精确概率：
        - Fully Supported -> 1.0, Partially -> 0.5, No -> 0.0, Contradictory -> -1.0
        - Utility 1-5 线性映射到 0.0-1.0
        """
        support_map = {
            SupportLevel.FULLY: 1.0,
            SupportLevel.PARTIALLY: 0.5,
            SupportLevel.NO: 0.0,
            SupportLevel.CONTRADICTORY: -1.0,
        }
        s_score = support_map.get(support_level, 0.0)
        u_score = (utility_score - 1) / 4.0 if utility_score else 0.5

        # 简单加权；后续可引入 LLM logprob
        return generation_prob + self.is_sup_weight * s_score + self.is_use_weight * u_score

    def _parse(self, raw: str) -> Tuple[SupportLevel, int]:
        try:
            # 找所有无嵌套的 {…}，取最后一个（推理链末尾的结论 JSON）
            matches = re.findall(r'\{[^{}]*\}', raw, re.DOTALL)
            if matches:
                obj = json.loads(matches[-1])
                support = SupportLevel(obj.get("support", "No Support"))
                utility = int(obj.get("utility", 3))
                utility = max(1, min(5, utility))
                return support, utility
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback：字符串匹配兜底，避免解析失败时全部降为 NO
        raw_up = raw.upper()
        if "FULLY SUPPORTED" in raw_up:
            return SupportLevel.FULLY, 4
        if "PARTIALLY SUPPORTED" in raw_up:
            return SupportLevel.PARTIALLY, 3
        if "CONTRADICTORY" in raw_up:
            return SupportLevel.CONTRADICTORY, 1
        return SupportLevel.NO, 2