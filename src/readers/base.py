"""
base.py
知识库读取器的抽象基类与数据模型。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class Document:
    """
    标准化文档对象，所有 Reader 的输出统一为此格式。

    Attributes:
        content: 文档正文（已去除 frontmatter、特殊语法等）
        metadata: 附加元数据
        doc_id: 全局唯一标识（建议用相对于知识库根目录的路径）
    """
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        # 确保 metadata 中始终包含 source 和 title
        if "source" not in self.metadata:
            self.metadata["source"] = self.doc_id
        if "title" not in self.metadata:
            self.metadata["title"] = self.doc_id


class BaseReader(ABC):
    """
    所有知识库读取器必须实现的接口。

    设计意图：
        - 解耦：CLI 和引擎层不依赖具体笔记软件
        - 扩展：未来增加 NotionReader / LogseqReader 时，只需实现此接口
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Args:
            config: 该读取器对应的配置子树（如 config["knowledge_source"]["obsidian"]）
        """
        self.config = config or {}

    @abstractmethod
    def read(self, path: str) -> Iterator[Document]:
        """
        读取指定路径的知识库，产出标准化 Document 对象。

        Args:
            path: 知识库根目录（Obsidian Vault 路径或 Markdown 文件夹路径）

        Yields:
            Document: 包含正文、元数据、doc_id 的标准化文档
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def reader_type(self) -> str:
        """返回读取器类型标识，如 'obsidian' / 'markdown' / 'notion'"""
        raise NotImplementedError