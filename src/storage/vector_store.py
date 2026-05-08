"""
vector_store.py
向量存储模块。
v1.0 使用 ChromaDB 本地持久化，管理两个集合：
    - notes: 用户笔记的嵌入
    - papers: arXiv 论文摘要的嵌入
特性：
    - 增量 upsert（基于 doc_id + chunk_index）
    - 元数据过滤（按标签、分类检索）
    - 双集合联合查询（笔记 vs 论文匹配）
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

from src.processors.chunker import Chunk

logger = logging.getLogger("future-agent.vector_store")


class VectorStore:
    """
    ChromaDB 向量存储封装。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.persist_dir = Path(
            self.config.get("vector_store.persist_dir", "./data/chroma")
        )
        self.distance_fn = self.config.get("vector_store.distance_fn", "cosine")
        self.notes_collection_name = self.config.get(
            "vector_store.notes_collection", "my_notes"
        )
        self.papers_collection_name = self.config.get(
            "vector_store.papers_collection", "arxiv_papers"
        )

        # 初始化客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建集合
        self.notes_collection = self._get_or_create_collection(
            self.notes_collection_name
        )
        self.papers_collection = self._get_or_create_collection(
            self.papers_collection_name
        )

        logger.info(f"💾 向量库已连接: {self.persist_dir}")

    def _get_or_create_collection(self, name: str):
        """获取或创建集合，指定距离函数。"""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.distance_fn},
        )

    # -------------------------------------------------------------------------
    # Notes 集合操作
    # -------------------------------------------------------------------------

    def upsert_notes(
        self, chunks: List[Chunk], embeddings: np.ndarray, force: bool = False
    ) -> None:
        """
        批量插入/更新笔记向量。
        使用 doc_id + chunk_index 作为唯一 ID，支持增量更新。
        """
        if not chunks:
            return

        ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 如果非 force 模式，先检查哪些 ID 已存在，避免重复写入
        if not force:
            existing = self.notes_collection.get(ids=ids, include=[])
            existing_ids = set(existing["ids"]) if existing else set()
            if existing_ids:
                logger.info(f"🔄 跳过 {len(existing_ids)} 个已存在的向量")

        self.notes_collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info(f"✅ 笔记向量已存储: {len(ids)} 个")

    def query_notes(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        在笔记集合中查询最相似的向量。

        Args:
            query_embeddings: (n, dim) 查询向量
            top_k: 返回数量
            where: ChromaDB 元数据过滤条件，如 {"tags": {"$contains": "RAG"}}

        Returns:
            查询结果列表，每项包含 id, distance, metadata, document
        """
        results = self.notes_collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        return self._format_results(results)

    # -------------------------------------------------------------------------
    # Papers 集合操作
    # -------------------------------------------------------------------------

    def upsert_papers(
        self, chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        """
        批量插入论文向量。
        论文通常全文重建，不做增量检查（arXiv 论文每天更新，直接覆盖）。
        """
        if not chunks:
            return

        ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.papers_collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info(f"✅ 论文向量已存储: {len(ids)} 个")

    def query_papers(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """在论文集合中查询最相似的向量。"""
        results = self.papers_collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return self._format_results(results)

    # -------------------------------------------------------------------------
    # 跨集合操作（核心：笔记 → 论文匹配）
    # -------------------------------------------------------------------------

    def match_notes_to_papers(
        self,
        note_chunk: Chunk,
        note_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        核心匹配逻辑：用一个笔记块去论文库中找最相关的论文。
        返回论文的元数据 + 相似度分数 + 关联文本。
        """
        results = self.papers_collection.query(
            query_embeddings=[note_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        return self._format_results(results)

    def stats(self) -> Dict[str, int]:
        """返回两个集合的统计信息。"""
        notes_count = self.notes_collection.count()
        papers_count = self.papers_collection.count()
        return {
            "notes_count": notes_count,
            "papers_count": papers_count,
        }

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_results(raw_results: Dict) -> List[Dict]:
        """
        将 ChromaDB 的嵌套列表结果格式化为扁平列表。
        """
        formatted = []
        ids = raw_results.get("ids", [[]])
        distances = raw_results.get("distances", [[]])
        metadatas = raw_results.get("metadatas", [[]])
        documents = raw_results.get("documents", [[]])

        # ChromaDB 返回的是 batch 嵌套结构，取第一个 batch
        batch_ids = ids[0] if ids else []
        batch_distances = distances[0] if distances else []
        batch_metadatas = metadatas[0] if metadatas else []
        batch_documents = documents[0] if documents else []

        for i in range(len(batch_ids)):
            formatted.append(
                {
                    "id": batch_ids[i],
                    "distance": batch_distances[i] if i < len(batch_distances) else None,
                    "metadata": batch_metadatas[i] if i < len(batch_metadatas) else {},
                    "document": batch_documents[i] if i < len(batch_documents) else "",
                }
            )

        return formatted