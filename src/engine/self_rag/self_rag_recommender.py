"""
self_rag_recommender.py
v1.5 Self-RAG 批判式推荐引擎（重构版）。

新流程：
    1. v1.0 Matcher 粗排（本地向量召回 Top-K）
    2. Retrieve Decision：LLM 判断今日是否有值得推荐的
    3. IsRel 批判：逐篇判断论文与用户研究的相关性
    4. 分段生成推荐理由 + IsSup 校验
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.config_loader import resolve_config_path
from src.engine.matcher import Matcher, Recommendation
from src.engine.self_rag import (
    PassageCritiqueEngine,
    Relevance,
    SupportLevel,
)
from src.engine.self_rag.generation_validator import GenerationValidator
from src.engine.self_rag.llm_wrapper import KimiClient, OpenAICompatibleClient
from src.generators.report_generator import ReportGenerator

logger = logging.getLogger("future-agent.self_rag_recommender")

# ------------------------------------------------------------------------------
# mode 对 IsSup 结果的控制：哪些 support_level 可以被保留
# ------------------------------------------------------------------------------
_MODE_SUP_FILTER = {
    # strict：FULLY 无条件保留；PARTIALLY 需 utility >= 4
    "strict": {SupportLevel.FULLY, SupportLevel.PARTIALLY},
    "balanced": {SupportLevel.FULLY, SupportLevel.PARTIALLY},
    "creative": {SupportLevel.FULLY, SupportLevel.PARTIALLY, SupportLevel.NO},
    "fast": None,  # fast 模式跳过 IsSup 校验，直接全部保留
}


class NoRecommendationError(Exception):
    """Self-RAG 未生成任何有效推荐时抛出。"""
    pass


def _create_llm_client(config: dict):
    """
    根据配置创建 LLM Client。
    支持 kimi / openai / custom（OpenAI-compatible，如本地 vLLM）。
    """
    llm_cfg = config.get("self_rag", {}).get("llm", {})
    provider = llm_cfg.get("provider", "kimi")
    model = llm_cfg.get("model", "moonshot-v1-8k")

    # 优先读取配置文件中的 api_key，若未配置则读取对应环境变量
    api_key = llm_cfg.get("api_key")
    if not api_key:
        if provider == "kimi":
            api_key = os.getenv("KIMI_API_KEY")
        elif provider in ("openai", "custom"):
            api_key = os.getenv("OPENAI_API_KEY")

    if provider == "kimi":
        if not api_key:
            raise ValueError(
                "使用 Kimi 模型需要提供 API Key。请在 config.local.yaml 中配置 "
                "self_rag.llm.api_key，或在环境变量中设置 KIMI_API_KEY。"
            )
        return KimiClient(model=model, api_key=api_key)
    elif provider == "openai":
        if not api_key:
            raise ValueError(
                "使用 OpenAI 模型需要提供 API Key。请在 config.local.yaml 中配置 "
                "self_rag.llm.api_key，或在环境变量中设置 OPENAI_API_KEY。"
            )
        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=None,
        )
    elif provider == "custom":
        if not api_key:
            raise ValueError(
                "使用 custom 模型需要提供 API Key。请在 config.local.yaml 中配置 "
                "self_rag.llm.api_key。"
            )
        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=llm_cfg.get("base_url"),
        )
    else:
        raise ValueError(f"未知的 LLM provider: {provider}")


class SelfRAGRecommendEngine:
    """v1.5 Self-RAG 批判式推荐引擎（重构版）。"""

    def __init__(
        self,
        config: dict,
        mode: str,
        beam_size: int,
        top_k: int,
    ) -> None:
        self.config = config
        self.mode = mode
        self.beam_size = beam_size  # 保留参数以兼容 CLI，但不再使用
        self.top_k = top_k

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_from_candidates(candidates: List[Recommendation]) -> str:
        """
        从 Matcher 粗排结果中聚合用户上下文。
        提取关联笔记的标题+chunk，按相似度去重后拼接。
        """
        seen_notes = set()
        parts = []
        for rec in sorted(candidates, key=lambda x: x.similarity, reverse=True):
            note_key = rec.matched_note
            if note_key in seen_notes:
                continue
            seen_notes.add(note_key)
            parts.append(f"《{rec.matched_note}》\n{rec.matched_chunk}")
            if len(parts) >= 5:
                break
        return "\n\n---\n\n".join(parts)[:2000]

    def _has_valuable_candidates(
        self,
        llm_client,
        user_context: str,
        candidates: List[Recommendation],
    ) -> bool:
        """
        Step 2: Retrieve Decision。
        基于用户笔记和粗排候选，判断今日是否有值得推荐的论文。
        """
        candidate_summary = "\n\n".join(
            f"{i + 1}. 《{rec.title}》\n"
            f"   相似度: {rec.similarity:.3f} | 关联笔记: 《{rec.matched_note}》\n"
            f"   摘要: {rec.abstract[:200]}..."
            for i, rec in enumerate(candidates[: self.top_k])
        )

        prompt = f"""你是 Future-Agent 的推荐决策模块。基于用户笔记和召回的候选论文，判断今日是否有值得向用户推荐的论文。

用户笔记上下文：
{user_context[:1500]}

召回的候选论文（按相似度排序，共 {len(candidates)} 篇）：
{candidate_summary}

判断规则：
- Yes：候选论文中至少有一篇对用户当前研究有直接帮助、填补知识缺口或提供有意义的新视角。
- No：候选论文整体与用户当前研究关联较弱，或内容过于泛泛，今日不值得推荐。

重要：你必须用英文输出 JSON：
{{"recommend": "Yes|No", "reason": "简短的中文解释"}}
"""

        raw = llm_client.complete(prompt, temperature=0.0, max_tokens=128)
        logger.info(f"Retrieve Decision 原始输出: {raw[:100]}...")

        try:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                val = obj.get("recommend", "").strip().upper()
                reason = obj.get("reason", "")
                logger.info(f"Retrieve Decision: {val} | 原因: {reason}")
                return val == "YES"
        except (json.JSONDecodeError, ValueError):
            pass

        # 字符级 fallback
        has_yes = "YES" in raw.upper()
        logger.info(f"Retrieve Decision (fallback): {'Yes' if has_yes else 'No'}")
        return has_yes

    def _generate_explanation(
        self,
        llm_client,
        user_context: str,
        rec: Recommendation,
    ) -> str:
        """为单篇论文生成推荐理由。"""
        prompt = f"""你是 Future-Agent 的推荐生成模块。请基于用户笔记和候选论文，生成一条简洁的中文推荐语。

用户笔记上下文：
{user_context[:1000]}

候选论文：
标题：《{rec.title}》
摘要：{rec.abstract[:500]}

要求：
1. 说明论文与用户研究方向的关联
2. 提炼核心贡献或价值（1-2 句话）
3. 控制在 150 字以内
4. 直接输出推荐语正文，不要加标题、编号或 markdown 格式
5. 在末尾用括号标注来源，如（来源：《论文标题》）

推荐语："""
        return llm_client.complete(prompt, temperature=0.3, max_tokens=512)

    def _is_sup_filter_pass(self, support_level: SupportLevel, utility: int) -> bool:
        """根据当前 mode 判断 IsSup 结果是否可保留。"""
        allowed = _MODE_SUP_FILTER.get(self.mode)
        if allowed is None:
            # fast 模式：跳过校验，全部通过
            return True
        if support_level not in allowed:
            return False
        # strict 模式额外约束：PARTIALLY 需 utility >= 4
        if self.mode == "strict" and support_level == SupportLevel.PARTIALLY:
            return utility >= 4
        return True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: Path,
        date_str: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[List[Recommendation], Path]:
        """
        执行重构后的 v1.5 Self-RAG 推荐流程。

        Returns:
            (Recommendation 列表, 报告文件路径)
        """
        # Step 1: v1.0 Matcher 粗排（本地向量计算，零 LLM 成本）
        logger.info("🎯 Step 1: Matcher 粗排...")
        matcher = Matcher(self.config)
        candidates = matcher.recommend(top_k=self.top_k * 3, strategy="boundary_mix") # 扩大召回范围3倍
        if not candidates:
            raise NoRecommendationError("粗排未召回任何候选论文，请先执行 ingest 和 fetch")

        logger.info(f"粗排召回 {len(candidates)} 篇候选论文")

        # Step 2: 从粗排结果聚合用户上下文
        user_context = self._build_context_from_candidates(candidates)

        # Step 3: Retrieve Decision（LLM 判断今日是否有值得推荐的）
        logger.info("🧠 Step 2: Retrieve Decision...")
        llm_client = _create_llm_client(self.config)

        if not self._has_valuable_candidates(llm_client, user_context, candidates):
            raise NoRecommendationError("Retrieve Decision 判断今日无值得推荐的内容")

        # Step 4: IsRel 批判（逐篇判断相关性）
        logger.info("🔍 Step 3: IsRel 批判...")
        critique_engine = PassageCritiqueEngine(llm_client, self.config)
        relevant_candidates = []
        for i, rec in enumerate(candidates):
            rel = critique_engine.critique(user_context, rec)

            logger.info(f"IsRel for {rec.title}: {rel}")

            if progress_callback:
                progress_callback("isrel", i + 1, len(candidates))
            time.sleep(0.8)  # 降低瞬时请求密度，避免 429
            if rel == Relevance.RELEVANT:
                relevant_candidates.append(rec)
                logger.debug(f"IsRel: ✅ {rec.title[:40]}...")
            else:
                logger.debug(f"IsRel: ❌ {rec.title[:40]}...")

        if not relevant_candidates:
            raise NoRecommendationError("IsRel 批判后无相关论文")

        logger.info(f"IsRel 保留 {len(relevant_candidates)} 篇")

        # Step 5: 分段生成推荐理由 + IsSup 校验
        logger.info("✍️ Step 4: 生成推荐理由 + IsSup 校验...")
        validator = GenerationValidator(llm_client, self.config)
        final_recommendations = []

        for i, rec in enumerate(relevant_candidates):
            # 生成推荐理由
            explanation = self._generate_explanation(llm_client, user_context, rec)
            rec.explanation = explanation
            if progress_callback:
                progress_callback("generate", i + 1, len(relevant_candidates))
            time.sleep(0.8)  # 降低瞬时请求密度，避免 429

            # IsSup 校验（fast 模式跳过）
            if self.mode == "fast":
                rec.metadata["support_level"] = "Skipped"
                rec.metadata["utility_score"] = "-"
                final_recommendations.append(rec)
                continue

            support, utility = validator.validate(explanation, [rec])

            logger.info(f"IsSup for {rec.title}: {support}, utility={utility}")

            rec.metadata["support_level"] = support.value
            rec.metadata["utility_score"] = utility

            if self._is_sup_filter_pass(support, utility):
                final_recommendations.append(rec)
                logger.debug(
                    f"IsSup: ✅ {rec.title[:40]}... | {support.value} | 效用:{utility}"
                )
            else:
                logger.debug(
                    f"IsSup: 🚫 过滤 {rec.title[:40]}... | {support.value} | 效用:{utility}"
                )

        if not final_recommendations:
            raise NoRecommendationError(
                f"IsSup 校验后（mode={self.mode}）无有效推荐"
            )

        logger.info(f"最终推荐: {len(final_recommendations)} 篇")

        # Step 6: 生成报告
        generator = ReportGenerator(self.config)
        report_path = generator.generate(
            recommendations=final_recommendations,
            output_dir=output_dir,
            date_str=date_str,
            strategy=f"self_rag_{self.mode}",
        )

        return final_recommendations, report_path
