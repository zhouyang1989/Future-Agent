"""
Self-RAG 推理引擎：从固定检索管道升级为主动认知 Agent。
"""

from .base import LLMClient, Relevance, SupportLevel
from .generation_validator import GenerationValidator
from .llm_wrapper import KimiClient, OpenAICompatibleClient
from .passage_critique import PassageCritiqueEngine

__all__ = [
    "KimiClient",
    "LLMClient",
    "OpenAICompatibleClient",
    "Relevance",
    "SupportLevel",
    "GenerationValidator",
    "PassageCritiqueEngine",
]
