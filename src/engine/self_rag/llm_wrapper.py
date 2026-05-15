"""
LLMClient 实现：支持 OpenAI、Kimi 及任何 OpenAI-compatible API。
"""

import logging
import os
import time
from typing import Optional

from .base import LLMClient

logger = logging.getLogger("future-agent.llm_wrapper")


class OpenAICompatibleClient(LLMClient):
    """
    通用 OpenAI-compatible 客户端。
    覆盖 OpenAI、Kimi、DeepSeek、vLLM 等所有兼容接口。
    内置指数退避重试，应对 429 Rate Limit 和服务端过载。
    """

    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # 秒

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
    ):
        import openai
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        import openai

        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except openai.RateLimitError as e:
                last_err = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"⏳ 遇到 Rate Limit (429)，{delay:.1f} 秒后重试 (第 {attempt + 1}/{self.MAX_RETRIES} 次)...")
                time.sleep(delay)
            except openai.APIStatusError as e:
                last_err = e
                if getattr(e, "status_code", None) in {503, 502, 504}:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    logger.warning(f"⏳ 服务端过载 ({e.status_code})，{delay:.1f} 秒后重试 (第 {attempt + 1}/{self.MAX_RETRIES} 次)...")
                    time.sleep(delay)
                else:
                    raise

        # 重试耗尽
        logger.error(f"❌ LLM 请求在 {self.MAX_RETRIES} 次重试后仍然失败")
        raise last_err


class KimiClient(OpenAICompatibleClient):
    """
    Kimi (Moonshot) 专用客户端。
    默认模型：moonshot-v1-8k / moonshot-v1-32k / moonshot-v1-128k
    """

    def __init__(
            self,
            model: str = "moonshot-v1-8k",
            api_key: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )