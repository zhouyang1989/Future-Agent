"""
Self-RAG 基础数据结构与类型定义。
"""

from enum import Enum
from typing import Protocol


class Relevance(str, Enum):
    """相关性批判令牌"""
    RELEVANT = "Relevant"
    IRRELEVANT = "Irrelevant"


class SupportLevel(str, Enum):
    """事实支持度批判令牌"""
    FULLY = "Fully Supported"
    PARTIALLY = "Partially Supported"
    NO = "No Support"
    CONTRADICTORY = "Contradictory"


class LLMClient(Protocol):
    """
    LLM 调用抽象协议。v1.5 prompt-based 阶段通过外部 LLM API
   （OpenAI / Claude / 本地 vLLM）实现 Reflection Tokens。
    """

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        ...
