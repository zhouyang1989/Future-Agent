"""
fetchers/__init__.py
外部数据获取层。
"""
from src.fetchers.arxiv_fetcher import ArxivFetcher, Paper

__all__ = ["ArxivFetcher", "Paper"]