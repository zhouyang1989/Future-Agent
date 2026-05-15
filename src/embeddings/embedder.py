"""
embedder.py
文本嵌入模块。
v1.0 使用 SBERT 本地模型，支持 MPS（Apple Silicon）加速。
特性：
    - 批量编码 + 进度显示
    - 嵌入结果本地缓存（避免重复计算未变更的 chunk）
    - 自动检测设备（mps / cuda / cpu）
"""
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config_loader import project_root
from src.processors.chunker import Chunk

logger = logging.getLogger("future-agent.embedder")


class Embedder:
    """
    SBERT 嵌入器封装。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.model_name: str = self.config.get(
            "embedding.model_name",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.device: str = self._auto_select_device()
        self.normalize: bool = self.config.get("embedding.normalize", True)
        self.batch_size: int = self.config.get("embedding.batch_size", 32)

        # 缓存配置
        self.cache_dir = project_root() / "data" / "processed" / "embed_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 延迟加载模型（避免初始化时耗时）
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型。"""
        if self._model is None:
            logger.info(f"🔄 加载嵌入模型: {self.model_name}")
            logger.info(f"🖥️  计算设备: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, chunks: List[Chunk], show_progress: bool = True) -> np.ndarray:
        """
        对 Chunk 列表进行批量嵌入。

        Args:
            chunks: Chunk 对象列表

        Returns:
            embeddings: (n_chunks, dim) 的 numpy 数组
        """
        if not chunks:
            return np.array([])

        texts = [c.text for c in chunks]
        logger.info(f"🔢 待嵌入文本块: {len(texts)} 个")

        # 尝试从缓存加载
        cached_embeddings = self._load_cache(chunks)
        if cached_embeddings is not None:
            logger.info("💾 命中嵌入缓存，跳过计算")
            return cached_embeddings

        # 批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        logger.info(f"✅ 嵌入完成: shape={embeddings.shape}, dim={embeddings.shape[1]}")

        # 写入缓存
        self._save_cache(chunks, embeddings)

        return embeddings

    def _auto_select_device(self) -> str:
        """
        自动选择最优计算设备。
        优先级: mps (Apple Silicon) > cuda > cpu
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # -------------------------------------------------------------------------
    # 缓存机制（基于 chunk 内容的 hash）
    # -------------------------------------------------------------------------

    def _compute_cache_key(self, chunks: List[Chunk]) -> str:
        """基于 chunk 文本内容 + 模型名称 计算缓存 key。"""
        content = "".join(c.text for c in chunks)
        content += self.model_name
        content += str(self.normalize)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_cache(self, chunks: List[Chunk]) -> Optional[np.ndarray]:
        """尝试加载缓存的嵌入向量。"""
        key = self._compute_cache_key(chunks)
        cache_file = self.cache_dir / f"{key}.npy"
        meta_file = self.cache_dir / f"{key}.json"

        if cache_file.exists() and meta_file.exists():
            try:
                embeddings = np.load(cache_file)
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                if meta.get("count") == len(chunks):
                    return embeddings
            except Exception:
                pass
        return None

    def _save_cache(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """保存嵌入向量到本地缓存。"""
        key = self._compute_cache_key(chunks)
        cache_file = self.cache_dir / f"{key}.npy"
        meta_file = self.cache_dir / f"{key}.json"

        np.save(cache_file, embeddings)
        with open(meta_file, "w") as f:
            json.dump({"count": len(chunks), "model": self.model_name}, f)