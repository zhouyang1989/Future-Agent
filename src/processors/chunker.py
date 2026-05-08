"""
chunker.py
文本分块处理器。
v1.0 策略：标题感知分块（Heading-aware Chunking）+ 长度兜底。
    - 优先按 Markdown 标题（## / ###）切分，保持语义完整性
    - 如果单块超过 max_chunk_size，再按句子边界二次切分
    - 块之间保留 overlap，避免上下文断裂
"""
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from src.readers.base import Document


@dataclass
class Chunk:
    """
    标准化文本块对象。
    """
    text: str
    doc_id: str                    # 来源文档 ID
    chunk_index: int               # 在文档中的块序号
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.doc_id}:{self.chunk_index}, {preview}...)"


class Chunker:
    """
    分块器。v1.0 默认使用 heading-aware 策略。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.chunk_size: int = self.config.get("chunk_size", 512)
        self.chunk_overlap: int = self.config.get("chunk_overlap", 128)
        self.min_chunk_length: int = self.config.get("min_chunk_length", 50)
        self.strategy: str = self.config.get("chunk_strategy", "heading_aware")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        对一批文档执行分块。
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self._chunk_single(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks

    @staticmethod
    def _build_chunk_metadata(doc: Document) -> dict:
        """构建 chunk 元数据，过滤掉空列表（ChromaDB 不允许空列表）。"""
        metadata: dict = {
            "source_doc_title": doc.metadata.get("title", doc.doc_id),
        }
        tags = doc.metadata.get("tags", [])
        if tags:
            metadata["tags"] = tags
        links = doc.metadata.get("links", [])
        if links:
            metadata["links"] = links
        return metadata

    def _chunk_single(self, doc: Document) -> List[Chunk]:
        """
        对单个文档分块。
        流程：
            1. 按 Markdown 标题切分为初始块
            2. 对超长块按句子边界二次切分
            3. 应用重叠
        """
        text = doc.content.strip()
        if not text:
            return []

        # 第一步：按标题切分
        sections = self._split_by_headings(text)

        chunks: List[Chunk] = []
        chunk_idx = 0

        for section_text in sections:
            section_text = section_text.strip()
            if len(section_text) < self.min_chunk_length:
                continue

            # 如果单节超过 chunk_size，按句子切分
            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_by_sentences(
                    section_text, max_len=self.chunk_size, overlap=self.chunk_overlap
                )
                # 提取标题行，用于给后续子块保留上下文
                heading_line = ""
                lines = section_text.split("\n")
                if lines and lines[0].startswith("#"):
                    heading_line = lines[0].strip()

                for idx, sub_text in enumerate(sub_chunks):
                    if len(sub_text) >= self.min_chunk_length:
                        # 非首个子块保留标题上下文
                        if idx > 0 and heading_line:
                            sub_text = heading_line + "\n" + sub_text
                        chunks.append(
                            Chunk(
                                text=sub_text,
                                doc_id=doc.doc_id,
                                chunk_index=chunk_idx,
                                metadata=self._build_chunk_metadata(doc),
                            )
                        )
                        chunk_idx += 1
            else:
                chunks.append(
                    Chunk(
                        text=section_text,
                        doc_id=doc.doc_id,
                        chunk_index=chunk_idx,
                        metadata=self._build_chunk_metadata(doc),
                    )
                )
                chunk_idx += 1

        return chunks

    # -------------------------------------------------------------------------
    # 分块策略实现
    # -------------------------------------------------------------------------

    @staticmethod
    def _split_by_headings(text: str) -> List[str]:
        """
        按 Markdown 标题（## / ### / ####）切分。
        保留标题与其后续内容在一起。
        """
        # 匹配行首的 ## 标题（支持中文/英文标题）
        pattern = r'(?:^|\n)(#{2,4}\s+.+?)(?=\n#{2,4}\s+|\Z)'
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))

        if not matches:
            # 没有二级及以上标题，整篇作为一个 section
            return [text]

        sections: List[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        # 处理第一个标题之前的内容（如果有的话）
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble and len(preamble) > 20:  # 忽略过短的前言
                sections.insert(0, preamble)

        return sections

    def _split_by_sentences(
        self, text: str, max_len: int, overlap: int
    ) -> List[str]:
        """
        按句子边界切分长文本。中文-aware：
            - 中文句子边界：。！？；…
            - 英文句子边界：. ! ? ;
            - 保留重叠：后一块开头包含前一块末尾 overlap 长度
        """
        # 先按句子边界拆分
        sentence_pattern = r'(?<=[。！？；.!?;…])\s+'
        sentences = re.split(sentence_pattern, text)

        # 清理空句
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # 退化情况：按字符硬切
            return self._hard_split(text, max_len, overlap)

        chunks: List[str] = []
        current_chunk = ""

        for sent in sentences:
            # 如果单句就超过 max_len，先对单句硬切
            if len(sent) > max_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self._hard_split(sent, max_len, overlap))
                continue

            # 尝试加入当前块
            if len(current_chunk) + len(sent) + 1 <= max_len:
                current_chunk += sent + " "
            else:
                # 当前块已满，保存
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 新块以重叠方式开始：包含前一块末尾 overlap 长度
                prev_tail = current_chunk.strip()[-overlap:] if len(current_chunk) > overlap else current_chunk.strip()
                current_chunk = prev_tail + " " + sent + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def _hard_split(text: str, max_len: int, overlap: int) -> List[str]:
        """
        退化策略：按字符数硬切，保留重叠。
        用于超长单句或无法找到句子边界的情况。
        """
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap
        return chunks