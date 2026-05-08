"""
arxiv_fetcher.py
arXiv 论文获取模块。
使用 arxiv 官方 Python 库（2.x API），支持：
    - 多分类 OR 查询 + 提交日期范围过滤
    - 本地 JSON 缓存（避免重复请求）
    - 速率限制与自动重试
    - 转换为 Chunk 对象供嵌入使用
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import arxiv
except ImportError:
    arxiv = None

from src.processors.chunker import Chunk

logger = logging.getLogger("future-agent.fetchers")


@dataclass
class Paper:
    """arXiv 论文标准化数据对象。"""
    arxiv_id: str           # 短 ID，如 2505.01234
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime     # naive datetime（已去除 tzinfo）
    updated: datetime
    pdf_url: str
    entry_id: str
    doi: Optional[str] = None


class ArxivFetcher:
    """
    arXiv 论文获取器。
    v1.0 只获取元数据（标题+摘要），不下载 PDF 全文。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if arxiv is None:
            raise ImportError(
                "未安装 arxiv 库。请执行: pip install arxiv>=2.0"
            )

        self.config = config or {}
        self.categories: List[str] = self.config.get(
            "categories", ["cs.CL", "cs.LG", "cs.IR"]
        )
        self.cache_dir: Path = Path(
            self.config.get("cache_dir", "./data/raw/arxiv")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # arXiv 客户端：内置速率限制（3秒间隔）+ 重试
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )

    def fetch(
        self,
        days: int = 7,
        max_results: int = 100,
        categories: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        从 arXiv 获取最近 N 天的论文。

        Args:
            days: 最近多少天（基于提交日期 submittedDate）
            max_results: 最大返回数量
            categories: 覆盖默认分类列表

        Returns:
            Paper 对象列表，按提交日期从新到旧排序
        """
        target_cats = categories or self.categories
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # 构建 arXiv 高级查询语法
        cat_query = " OR ".join([f"cat:{c}" for c in target_cats])
        start_str = cutoff_date.strftime("%Y%m%d")
        end_str = datetime.utcnow().strftime("%Y%m%d")
        date_query = f"submittedDate:[{start_str} TO {end_str}]"

        query = f"({cat_query}) AND {date_query}"

        # 缓存检查
        cache_key = hashlib.md5(
            f"{query}_{max_results}".encode("utf-8")
        ).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            logger.info(f"💾 命中 arXiv 缓存: {cache_file.name}")
            return self._load_from_cache(cache_file)

        logger.info(f"🌐 请求 arXiv API")
        logger.info(f"   查询: {query}")
        logger.info(f"   上限: {max_results}")

        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            max_results=max_results,
        )

        papers: List[Paper] = []
        try:
            for result in self.client.results(search):
                published = result.published.replace(tzinfo=None)

                # 客户端二次过滤（处理时区/边界偏差）
                if published < cutoff_date:
                    continue

                papers.append(
                    Paper(
                        arxiv_id=self._extract_short_id(result),
                        title=result.title,
                        authors=[str(a) for a in result.authors],
                        abstract=result.summary,
                        categories=result.categories,
                        published=published,
                        updated=result.updated.replace(tzinfo=None),
                        pdf_url=getattr(result, "pdf_url", ""),
                        entry_id=result.entry_id,
                        doi=getattr(result, "doi", None),
                    )
                )
        except Exception as e:
            logger.error(f"❌ arXiv API 请求失败: {e}")
            raise

        logger.info(f"📥 成功获取 {len(papers)} 篇论文")
        self._save_to_cache(cache_file, papers)
        return papers

    def to_chunks(self, papers: List[Paper]) -> List[Chunk]:
        """
        将 Paper 列表转换为 Chunk 列表。
        每篇论文生成 1 个 Chunk（标题 + 分类 + 摘要），供嵌入使用。
        论文摘要通常 1000-3000 字符，在 chunk_size=512 范围内，
        因此单篇论文不再二次切分。
        """
        chunks: List[Chunk] = []
        for paper in papers:
            text = (
                f"Title: {paper.title}\n"
                f"Categories: {', '.join(paper.categories)}\n"
                f"Abstract: {paper.abstract}"
            )

            chunks.append(
                Chunk(
                    text=text,
                    doc_id=paper.arxiv_id,
                    chunk_index=0,
                    metadata={
                        "title": paper.title,
                        "authors": self._format_authors(paper.authors),
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                        "published": paper.published.isoformat(),
                        "pdf_url": paper.pdf_url,
                        "entry_id": paper.entry_id,
                        "doi": paper.doi,
                        "source_type": "arxiv",
                    },
                )
            )
        return chunks

    # -------------------------------------------------------------------------
    # 内部工具方法
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_short_id(result) -> str:
        """从 entry_id 提取短 ID，兼容不同 arxiv 库版本。"""
        if hasattr(result, "get_short_id"):
            return result.get_short_id()
        # 回退：从 URL 提取，entry_id 形如 http://arxiv.org/abs/2505.01234
        return result.entry_id.split("/")[-1]

    @staticmethod
    def _format_authors(authors: List[str]) -> str:
        """格式化作者列表：前 3 人 + et al.。"""
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + " et al."

    def _save_to_cache(self, path: Path, papers: List[Paper]) -> None:
        """序列化 Paper 列表到本地 JSON。"""
        data = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "categories": p.categories,
                "published": p.published.isoformat(),
                "updated": p.updated.isoformat(),
                "pdf_url": p.pdf_url,
                "entry_id": p.entry_id,
                "doi": p.doi,
            }
            for p in papers
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 缓存已保存: {path.name}")

    def _load_from_cache(self, path: Path) -> List[Paper]:
        """从本地 JSON 反序列化 Paper 列表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = []
        for p in data:
            papers.append(
                Paper(
                    arxiv_id=p["arxiv_id"],
                    title=p["title"],
                    authors=p["authors"],
                    abstract=p["abstract"],
                    categories=p["categories"],
                    published=datetime.fromisoformat(p["published"]),
                    updated=datetime.fromisoformat(p["updated"]),
                    pdf_url=p["pdf_url"],
                    entry_id=p["entry_id"],
                    doi=p.get("doi"),
                )
            )
        logger.info(f"📂 从缓存加载 {len(papers)} 篇论文")
        return papers