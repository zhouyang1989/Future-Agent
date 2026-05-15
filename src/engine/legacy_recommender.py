"""
legacy_recommender.py
v1.0 纯本地向量匹配推荐引擎封装。
"""
import logging
from pathlib import Path
from typing import List, Tuple

from src.engine.matcher import Matcher, Recommendation
from src.generators.report_generator import ReportGenerator

logger = logging.getLogger("future-agent.legacy_recommender")


class LegacyRecommendEngine:
    """v1.0 传统推荐引擎：固定检索 + 相似度匹配 + 多样性约束。"""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(
        self,
        top_k: int,
        diversity: str,
        output_dir: Path,
        date_str: str,
    ) -> Tuple[List[Recommendation], Path]:
        """
        执行 v1.0 推荐流程。

        Args:
            top_k: 最终推荐数量
            diversity: 多样性策略（strong_only / boundary_mix / cross_domain）
            output_dir: 报告输出目录
            date_str: 日期字符串，用于文件名

        Returns:
            (Recommendation 列表, 报告文件路径)
        """
        matcher = Matcher(self.config)
        recommendations = matcher.recommend(top_k=top_k, strategy=diversity)

        generator = ReportGenerator(self.config)
        report_path = generator.generate(
            recommendations=recommendations,
            output_dir=output_dir,
            date_str=date_str,
            strategy=diversity,
        )
        return recommendations, report_path
