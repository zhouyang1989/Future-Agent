"""
matcher.py
核心推荐引擎。

v1.0 实现多样性约束推荐：
    1. 用笔记库所有 chunk 作为查询，检索论文库候选
    2. 按相似度分桶（强相关 / 边界拓展 / 跨界）
    3. 根据策略从各桶采样，防止信息茧房
    4. 生成模板化推荐理由
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.embeddings.embedder import Embedder
from src.storage.vector_store import VectorStore

logger = logging.getLogger("future-agent.matcher")

# 预定义技术领域的高区分度术语（轻量词典，v1.1 可扩展）
DOMAIN_FINGERPRINTS = {
    "nlp_embedding": {
        "terms": {
            "word2vec", "skip-gram", "cbow", "embedding", "bert", "transformer",
            "token", "vocabulary", "semantic", "cosine", "analogy", "vector",
            "distributed representation", "huffman", "softmax", "negative sampling",
            "sentence", "paraphrase", "sbert", "attention", "llm", "rag",
        },
        "arxiv_cats": {"cs.CL", "cs.LG", "cs.AI"},
    },
    "vector_retrieval": {
        "terms": {
            "hnsw", "ann", "nearest neighbor", "faiss", "chromadb", "milvus",
            "index", "quantization", "ivf", "graph", "navigable", "routing",
            "similarity search", "approximate", "distance", "cosine", "l2",
        },
        "arxiv_cats": {"cs.IR", "cs.DB", "cs.LG"},
    },
    "cv_vision": {
        "terms": {
            "yolo", "detection", "segmentation", "opencv", "cnn", "resnet",
            "image", "pixel", "bbox", "keypoint", "gesture", "tracking",
            "3d vision", "depth", "pose", "camera", "calibration",
        },
        "arxiv_cats": {"cs.CV"},
    },
    "ml_theory": {
        "terms": {
            "online learning", "regret bound", "oracle inequality", "convergence",
            "distribution shift", "non-stationary", "covariate", "bandit", "mab",
            "rmse", "mae", "forecasting", "time series", "electricity", "load",
        },
        "arxiv_cats": {"stat.ML", "cs.LG", "eess.SP"},
    },
}


@dataclass
class Recommendation:
    """单条推荐结果。"""
    paper_id: str
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: List[str]
    pdf_url: str
    published: str

    similarity: float
    matched_note: str
    matched_chunk: str
    explanation: str

    diversity_bucket: str  # "strong" / "boundary" / "cross"
    metadata: dict = field(default_factory=dict)


class Matcher:
    """
    推荐匹配引擎。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.store = VectorStore(config)
        self.embedder = Embedder(config)

        # 多样性阈值
        self.threshold_high: float = self.config.get(
            "recommendation.threshold_high", 0.70
        )
        self.threshold_mid: float = self.config.get(
            "recommendation.threshold_mid", 0.55
        )
        # 每个笔记块查询返回的论文数
        self.query_top_k: int = self.config.get(
            "recommendation.query_top_k", 5
        )

    def recommend(
        self,
        top_k: int = 5,
        strategy: str = "boundary_mix",
    ) -> List[Recommendation]:
        """
        生成推荐列表。

        Args:
            top_k: 最终推荐数量
            strategy: strong_only / boundary_mix / cross_domain

        Returns:
            Recommendation 列表
        """
        logger.info(f"🎯 推荐策略: {strategy}, 目标数量: {top_k}")

        # 1. 读取笔记库全部 chunk
        notes_data = self._get_all_notes()
        if not notes_data:
            logger.warning("⚠️ 笔记库为空，请先执行 ingest")
            return []

        logger.info(f"📚 笔记库: {len(notes_data)} 个 chunk")

        # 2. 用每个笔记 chunk 查询论文库，收集候选
        candidates = self._collect_candidates(notes_data)
        if not candidates:
            logger.warning("⚠️ 论文库为空，请先执行 fetch")
            return []

        logger.info(f"🔍 原始候选池: {len(candidates)} 条（含重复）")

        # 3. 按 paper_id 去重，保留最佳匹配记录
        unique_candidates = self._deduplicate_candidates(candidates)
        logger.info(f"🧹 去重后候选: {len(unique_candidates)} 篇")

        # 4. 按相似度分桶
        buckets = self._bucket_candidates(unique_candidates)

        # 5. 按策略采样
        selected = self._select_by_strategy(buckets, top_k, strategy)

        # 跨界论文需要额外缓冲，避免为凑数而推荐弱相关内容
        # 如果 threshold_mid=0.50，则 cross 论文必须 ≥ 0.55 才能保留
        cross_quality_buffer = 0.05

        quality_selected = []
        for item in selected:
            sim = item["similarity"]

            # strong (≥0.70) 和 boundary (≥0.50) 无条件保留
            if sim >= self.threshold_mid:
                quality_selected.append(item)
            # cross 桶论文必须高于 mid 阈值 + 缓冲带
            elif sim >= self.threshold_mid + cross_quality_buffer:
                quality_selected.append(item)
            else:
                logger.debug(
                    f"🚫 质量守门过滤: {item['metadata'].get('title', '')[:40]}... "
                    f"sim={sim:.3f}"
                )

        if not quality_selected:
            logger.info("⚠️ 今日论文池与笔记库关联度较低，无高质量推荐")
            return []

        # 6. 生成解释并包装为 Recommendation
        recommendations = self._generate_explanations(quality_selected)

        # 7. 最终排序：强相关 > 边界 > 跨界
        bucket_order = {"strong": 0, "boundary": 1, "cross": 2}
        recommendations.sort(
            key=lambda x: (bucket_order[x.diversity_bucket], -x.similarity)
        )

        # 9. 日志统计
        strong_c = sum(1 for r in recommendations if r.diversity_bucket == 'strong')
        boundary_c = sum(1 for r in recommendations if r.diversity_bucket == 'boundary')
        cross_c = sum(1 for r in recommendations if r.diversity_bucket == 'cross')

        logger.info(
            f"✅ 最终推荐: {len(recommendations)} 篇 "
            f"(强相关 {strong_c} | 边界 {boundary_c} | 跨界 {cross_c})"
        )

        if len(recommendations) < top_k:
            logger.info(
                f"ℹ️ 宁缺毋滥生效: 目标上限 {top_k} 篇，实际输出 {len(recommendations)} 篇"
            )

        return recommendations

    # -------------------------------------------------------------------------
    # 内部流程
    # -------------------------------------------------------------------------

    def _get_all_notes(self) -> List[Dict]:
        """从 notes 集合获取所有 chunk 的文本、嵌入和元数据。"""
        result = self.store.notes_collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        if not result or not result.get("ids"):
            return []

        notes = []
        for i in range(len(result["ids"])):
            notes.append({
                "id": result["ids"][i],
                "text": result["documents"][i],
                "embedding": np.array(result["embeddings"][i]),
                "metadata": result["metadatas"][i],
            })
        return notes

    def _collect_candidates(self, notes_data: List[Dict]) -> List[Dict]:
        """
        用每个笔记 chunk 查询论文库。
        返回扁平候选列表，含匹配来源信息。
        """
        candidates: List[Dict] = []

        for note in notes_data:
            embedding = note["embedding"].reshape(1, -1)
            results = self.store.papers_collection.query(
                query_embeddings=embedding.tolist(),
                n_results=self.query_top_k,
                include=["metadatas", "documents", "distances"],
            )

            if not results or not results["ids"] or not results["ids"][0]:
                continue

            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]

            for i in range(len(ids)):
                # ChromaDB cosine distance = 1 - cosine_similarity
                similarity = 1.0 - distances[i]

                candidates.append({
                    "paper_id": ids[i],
                    "similarity": float(similarity),
                    "metadata": metadatas[i],
                    "document": documents[i],
                    "matched_note_title": note["metadata"].get(
                        "source_doc_title", note["id"]
                    ),
                    "matched_chunk_text": note["text"][:200],
                })

        return candidates

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        按 paper_id 去重，保留相似度最高的匹配。
        同时聚合所有匹配来源。
        """
        best: Dict[str, Dict] = {}
        sources: Dict[str, List[Dict]] = {}

        for c in candidates:
            pid = c["paper_id"]

            if pid not in sources:
                sources[pid] = []
            sources[pid].append({
                "note_title": c["matched_note_title"],
                "chunk_text": c["matched_chunk_text"],
                "similarity": c["similarity"],
            })

            if pid not in best or c["similarity"] > best[pid]["similarity"]:
                best[pid] = c.copy()

        # 将来源合并到 best
        for pid in best:
            sources[pid].sort(key=lambda x: x["similarity"], reverse=True)
            best[pid]["all_sources"] = sources[pid]
            best[pid]["matched_note_title"] = sources[pid][0]["note_title"]
            best[pid]["matched_chunk_text"] = sources[pid][0]["chunk_text"]

        return list(best.values())

    def _bucket_candidates(self, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """按相似度分桶。"""
        buckets = {"strong": [], "boundary": [], "cross": []}

        for c in candidates:
            sim = c["similarity"]
            if sim >= self.threshold_high:
                buckets["strong"].append(c)
            elif sim >= self.threshold_mid:
                buckets["boundary"].append(c)
            else:
                buckets["cross"].append(c)

        for key in buckets:
            buckets[key].sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(
            f"📊 分桶: 强相关 {len(buckets['strong'])} | "
            f"边界 {len(buckets['boundary'])} | "
            f"跨界 {len(buckets['cross'])}"
        )
        return buckets

    def _compute_domain_score(self, note_text: str, paper_text: str, paper_cats: List[str]) -> float:
        """
        计算笔记与论文的领域一致性分数。
        基于内容术语重叠 + arXiv 分类偏好，完全不依赖用户标签。

        返回: 0.3 ~ 1.0 的惩罚/奖励系数
        """
        note_lower = note_text.lower()
        paper_lower = paper_text.lower()

        # 1. 检查笔记内容命中了哪个领域指纹
        note_domain_hits: Dict[str, int] = {}
        for domain, info in DOMAIN_FINGERPRINTS.items():
            hit_count = sum(1 for term in info["terms"] if term in note_lower)
            if hit_count > 0:
                note_domain_hits[domain] = hit_count

        if not note_domain_hits:
            # 笔记内容无法识别领域，保守处理（不惩罚也不奖励）
            return 1.0

        # 2. 取笔记最可能的领域（命中术语最多的）
        dominant_domain = max(note_domain_hits, key=note_domain_hits.get)

        # 3. 检查论文是否属于该领域的友好分类
        domain_info = DOMAIN_FINGERPRINTS[dominant_domain]
        paper_cats_set = set(paper_cats)

        # 论文分类与笔记领域匹配？
        cat_match = bool(paper_cats_set & domain_info["arxiv_cats"])

        # 4. 检查论文内容是否也包含该领域术语（双向验证）
        paper_domain_hit = sum(
            1 for term in domain_info["terms"] if term in paper_lower
        )

        # 5. 计算最终系数
        if cat_match and paper_domain_hit >= 2:
            # 强匹配：分类对 + 论文内容也有领域术语
            return 1.0
        elif cat_match or paper_domain_hit >= 1:
            # 中等匹配
            return 0.85
        else:
            # 笔记是 NLP，论文是电力预测：严重错配，大幅降权
            # 但保留一点空间，防止过度过滤
            return 0.45

    def _select_by_strategy(
        self,
        buckets: Dict[str, List[Dict]],
        top_k: int,
        strategy: str,
    ) -> List[Dict]:
        """按策略从各桶采样，数量不足时从强相关桶补足。"""
        strong = buckets["strong"]
        boundary = buckets["boundary"]
        cross = buckets["cross"]
        selected: List[Dict] = []
        seen: set = set()

        def _take(source: List[Dict], n: int):
            """从 source 取 n 条不重复的候选。"""
            taken = []
            for item in source:
                if item["paper_id"] not in seen:
                    seen.add(item["paper_id"])
                    taken.append(item)
                    if len(taken) >= n:
                        break
            return taken

        if strategy == "strong_only":
            selected = _take(strong, top_k)

        elif strategy == "boundary_mix":
            # 强相关 2 + 边界 2 + 跨界 1（默认 5 篇时）
            n_strong = max(1, top_k // 2 + top_k % 2)
            n_boundary = max(1, top_k // 3)
            n_cross = top_k - n_strong - n_boundary

            selected.extend(_take(strong, n_strong))
            selected.extend(_take(boundary, n_boundary))
            selected.extend(_take(cross, n_cross))

        elif strategy == "cross_domain":
            # 跨界优先，但保留 1 篇强相关作为锚点
            n_cross = min(top_k - 1, len(cross)) if cross else 0
            n_boundary = min(top_k - n_cross - 1, len(boundary)) if boundary else 0
            n_strong = top_k - n_cross - n_boundary

            selected.extend(_take(cross, n_cross))
            selected.extend(_take(boundary, n_boundary))
            selected.extend(_take(strong, n_strong))

        else:
            selected = _take(strong, top_k)

        # 数量不足时，用强相关补足
        if len(selected) < top_k:
            for item in strong:
                if item["paper_id"] not in seen:
                    seen.add(item["paper_id"])
                    selected.append(item)
                    if len(selected) >= top_k:
                        break

        return selected

    def _generate_explanations(self, selected: List[Dict]) -> List[Recommendation]:
        """包装为 Recommendation 对象，生成模板化推荐理由。"""
        recommendations = []

        for item in selected:
            meta = item["metadata"]
            sim = item["similarity"]

            # 确定桶
            if sim >= self.threshold_high:
                bucket = "strong"
            elif sim >= self.threshold_mid:
                bucket = "boundary"
            else:
                bucket = "cross"

            note_title = item["matched_note_title"]
            explanation = self._build_explanation(
                bucket=bucket,
                note_title=note_title,
                similarity=sim,
                categories=meta.get("categories", []),
            )

            rec = Recommendation(
                paper_id=item["paper_id"],
                arxiv_id=meta.get("arxiv_id", item["paper_id"].split(":")[0]),
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", "Unknown"),
                abstract=meta.get("abstract", "")[:500],
                categories=meta.get("categories", []),
                pdf_url=meta.get("pdf_url", ""),
                published=meta.get("published", ""),
                similarity=sim,
                matched_note=note_title,
                matched_chunk=item["matched_chunk_text"],
                explanation=explanation,
                diversity_bucket=bucket,
                metadata=meta,
            )
            recommendations.append(rec)

        return recommendations

    def _build_explanation(
        self,
        bucket: str,
        note_title: str,
        similarity: float,
        categories: List[str],
    ) -> str:
        """v1.0 模板化推荐理由，v2.0 可接入 LLM 生成。"""
        cat_str = ", ".join(categories[:2]) if categories else "相关领域"

        if bucket == "strong":
            return (
                f"这篇论文与你笔记《{note_title}》高度相关（相似度 {similarity:.2f}），"
                f"可能延续或深化了你已掌握的知识体系，建议优先阅读。"
            )
        elif bucket == "boundary":
            return (
                f"这篇论文与《{note_title}》部分相关（相似度 {similarity:.2f}），"
                f"但引入了 {cat_str} 的新方法或视角，适合作为知识边界的拓展。"
            )
        else:
            return (
                f"这篇论文与你当前笔记库关联较弱（相似度 {similarity:.2f}），"
                f"属于较少关注的 {cat_str} 领域，作为跨界阅读可能带来新启发。"
            )