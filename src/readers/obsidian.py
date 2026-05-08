"""
obsidian.py
Obsidian Vault 专用读取器（v1.1 修复版）。
解析特性：
    - YAML Frontmatter
    - WikiLinks（排除图片嵌入）
    - 属性标签提取（如 **所属领域**：[[xxx]]）
    - 中文-aware 字数统计
"""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import yaml

from src.readers.base import BaseReader, Document


class ObsidianReader(BaseReader):
    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.ignore_folders: Set[str] = set(
            self.config.get("ignore_folders", [])
        )
        self.extract_frontmatter: bool = self.config.get(
            "extract_frontmatter", True
        )
        self.extract_wikilinks: bool = self.config.get(
            "extract_wikilinks", True
        )

    @property
    def reader_type(self) -> str:
        return "obsidian"

    def read(self, path: str) -> Iterator[Document]:
        vault_path = Path(path).expanduser().resolve()
        if not vault_path.exists():
            raise FileNotFoundError(f"Obsidian Vault 路径不存在: {vault_path}")
        if not vault_path.is_dir():
            raise NotADirectoryError(f"路径不是目录: {vault_path}")

        md_files = self._collect_md_files(vault_path)

        for file_path in md_files:
            doc = self._parse_file(file_path, vault_path)
            if doc:
                yield doc

    def _collect_md_files(self, vault_path: Path) -> List[Path]:
        md_files: List[Path] = []
        for root, dirs, files in os.walk(vault_path):
            rel_root = Path(root).relative_to(vault_path)
            if any(part in self.ignore_folders for part in rel_root.parts):
                dirs[:] = []
                continue
            for f in files:
                if f.endswith(".md"):
                    md_files.append(Path(root) / f)
        md_files.sort()
        return md_files

    def _parse_file(self, file_path: Path, vault_path: Path) -> Optional[Document]:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = file_path.read_text(encoding="gbk")

        # 1. 分离 Frontmatter
        frontmatter: dict = {}
        body = raw_text
        if self.extract_frontmatter and raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except yaml.YAMLError:
                    body = raw_text

        # 2. 提取 WikiLinks（排除图片嵌入）
        links: List[str] = []
        embeds: List[str] = []
        if self.extract_wikilinks:
            links, embeds = self._extract_wikilinks_and_embeds(body)

        # 3. 提取标签（#tag 语法）
        tags: List[str] = self._extract_tags(body)

        # 4. 从属性文本提取"领域/分类"（如 **所属领域**：[[Embedding]]）
        categories: List[str] = self._extract_categories(body)

        # 合并：把 categories 也并入 tags（如果你希望它们出现在标签里）
        # 或者单独保留在 metadata["categories"] 中
        all_tags = list(dict.fromkeys(tags + categories))  # 去重，保持顺序

        # 5. 构建 doc_id 和标题
        rel_path = file_path.relative_to(vault_path)
        doc_id = str(rel_path.with_suffix(""))
        title = frontmatter.get("title", self._clean_title(rel_path.stem))

        # 6. 元数据
        stat = file_path.stat()
        metadata = {
            "source": str(file_path),
            "vault_path": str(vault_path),
            "title": title,
            "tags": all_tags,
            "categories": categories,      # 单独保留，如 ["Embedding"]
            "links": links,
            "embeds": embeds,              # 图片/附件嵌入
            "frontmatter": frontmatter,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "word_count": self._count_words(body),  # 中文-aware
            "char_count": len(body),       # 纯字符数（含标点）
        }

        return Document(
            content=body,
            metadata=metadata,
            doc_id=doc_id,
        )

    # -------------------------------------------------------------------------
    # 解析工具方法（v1.1 修复）
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_wikilinks_and_embeds(text: str) -> Tuple[List[str], List[str]]:
        """
        区分 WikiLinks [[...]] 和 Embeds ![[...]]。
        返回: (links, embeds)
        """
        # 匹配 ![[...]]（图片/附件嵌入）
        embed_pattern = r"!\[\[(.*?)\]\]"
        embed_matches = re.findall(embed_pattern, text)
        embeds = [m.strip() for m in embed_matches if m.strip()]
        embeds = list(dict.fromkeys(embeds))  # 去重

        # 匹配 [[...]]，但排除前面有 ! 的情况（即 ![[...]]）
        # 使用负向回顾后发 (?<!!) 确保前面不是 !
        link_pattern = r"(?<!!)\[\[(.*?)\]\]"
        link_matches = re.findall(link_pattern, text)

        links: List[str] = []
        seen: Set[str] = set()
        for match in link_matches:
            link_target = match.split("|")[0].strip()
            if link_target and link_target not in seen:
                seen.add(link_target)
                links.append(link_target)

        return links, embeds

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        """
        提取 #tag / #tag/subtag。
        排除代码块、纯数字、URL 锚点。
        """
        text_no_code = re.sub(r"```[\s\S]*?```", "", text)
        text_no_code = re.sub(r"`[^`]+`", "", text_no_code)

        pattern = r"#([a-zA-Z0-9_\-/]+)"
        matches = re.findall(pattern, text_no_code)

        results: List[str] = []
        seen: Set[str] = set()
        for tag in matches:
            if tag.isdigit():
                continue
            if tag not in seen:
                seen.add(tag)
                results.append(tag)
        return results

    @staticmethod
    def _extract_categories(text: str) -> List[str]:
        """
        从属性行提取分类信息，例如：
            **所属领域**：[[Embedding]]
            **领域**：[[NLP]]
            **分类**：#cs.CL
        返回 WikiLink 目标或标签值。
        """
        categories: List[str] = []
        seen: Set[str] = set()

        # 匹配 **xxx**：[[yyy]] 或 **xxx**: [[yyy]]
        pattern = r"\*\*[^*]+\*\*[：:]\s*\[\[(.*?)\]\]"
        matches = re.findall(pattern, text)
        for m in matches:
            val = m.split("|")[0].strip()
            if val and val not in seen:
                seen.add(val)
                categories.append(val)

        # 匹配 **xxx**：#yyy
        pattern2 = r"\*\*[^*]+\*\*[：:]\s*#([a-zA-Z0-9_\-/]+)"
        matches2 = re.findall(pattern2, text)
        for m in matches2:
            if m not in seen:
                seen.add(m)
                categories.append(m)

        return categories

    @staticmethod
    def _count_words(text: str) -> int:
        """
        中文-aware 字数统计：
        - 中文字符：每个字算 1 词
        - 英文单词：按空格分词
        - 数字串：按空格分词
        """
        # 中文字符数
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # 移除中文字符后，按空格统计英文/数字词
        text_no_chinese = re.sub(r"[\u4e00-\u9fff]", " ", text)
        other_words = len(text_no_chinese.split())
        return chinese_chars + other_words

    @staticmethod
    def _clean_title(stem: str) -> str:
        cleaned = re.sub(r"^\d+[\.\-_]\s*", "", stem)
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        return cleaned.strip()