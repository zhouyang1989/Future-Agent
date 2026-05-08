"""
markdown.py
纯 Markdown 文件夹读取器。
不解析 WikiLinks 和 Frontmatter（或仅做基础解析），适用于非 Obsidian 的 Markdown 知识库。
"""
from pathlib import Path
from typing import Iterator, Optional

from src.readers.base import BaseReader, Document
from src.readers.obsidian import ObsidianReader


class MarkdownFolderReader(BaseReader):
    """
    通用 Markdown 文件夹读取器。
    功能子集：支持 Frontmatter 和基础标签提取，但不解析 WikiLinks（因为非 Obsidian 环境通常不用）。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.ignore_folders: set = set(self.config.get("ignore_folders", []))
        self._obsidian_helper = ObsidianReader(config={
            "ignore_folders": list(self.ignore_folders),
            "extract_frontmatter": True,
            "extract_wikilinks": False,  # Markdown 文件夹默认不解析 WikiLinks
        })

    @property
    def reader_type(self) -> str:
        return "markdown"

    def read(self, path: str) -> Iterator[Document]:
        folder_path = Path(path).expanduser().resolve()
        if not folder_path.exists():
            raise FileNotFoundError(f"Markdown 文件夹不存在: {folder_path}")

        # 复用 ObsidianReader 的文件收集逻辑，但只传文件夹路径
        md_files = self._obsidian_helper._collect_md_files(folder_path)

        for file_path in md_files:
            doc = self._obsidian_helper._parse_file(file_path, folder_path)
            if doc:
                # 覆盖 reader_type 标识
                doc.metadata["reader_type"] = self.reader_type
                yield doc