"""
readers/__init__.py
读取器工厂，根据配置返回对应的 Reader 实例。
"""
from typing import Optional

from src.readers.base import BaseReader
from src.readers.markdown import MarkdownFolderReader
from src.readers.obsidian import ObsidianReader


READER_REGISTRY = {
    "obsidian": ObsidianReader,
    "markdown": MarkdownFolderReader,
    # v1.5 扩展：notion, zotero, logseq...
}


def get_reader(reader_type: str, config: Optional[dict] = None) -> BaseReader:
    """
    工厂函数：根据类型字符串返回对应的 Reader 实例。

    Args:
        reader_type: 读取器类型标识，如 'obsidian', 'markdown'
        config: 该读取器对应的配置子树

    Returns:
        BaseReader 实例

    Raises:
        ValueError: 如果 reader_type 未注册
    """
    reader_cls = READER_REGISTRY.get(reader_type)
    if not reader_cls:
        raise ValueError(
            f"未知的读取器类型: {reader_type}。可选: {list(READER_REGISTRY.keys())}"
        )
    return reader_cls(config=config)