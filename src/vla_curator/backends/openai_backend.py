"""
OpenAI GPT model backend.

Requires: pip install 'vla-data-curator[openai]'
Environment variable: OPENAI_API_KEY

Supports all OpenAI vision models that accept the ``image_url`` content type:
  gpt-4o            — best overall, multimodal
  gpt-4o-mini       — faster, cheaper, multimodal
  gpt-4-turbo       — strong reasoning, supports vision
  o1, o1-mini       — reasoning models (text-only currently)

Usage
-----
from vla_curator.backends.openai_backend import OpenAIBackend
from vla_curator.config import OpenAIConfig

cfg = OpenAIConfig(model_name="gpt-4o", api_key_env_var="OPENAI_API_KEY")
backend = OpenAIBackend(cfg)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import OpenAIConfig
from .base import GenerationResult, ModelBackend, Prompt, PromptImage, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIBackend(ModelBackend):
    """
    Model backend wrapping the ``openai`` Python SDK (v1+).

    Works with the standard OpenAI API and any OpenAI-compatible endpoint
    (Azure OpenAI, Together, Groq, etc.) via ``config.base_url``.
    """

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai is required for the OpenAI backend. "
                "Install with: pip install 'vla-data-curator[openai]'"
            ) from e

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{self.config.api_key_env_var}' is not set."
            )

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # We handle retries ourselves via tenacity
        )
        return self._client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: Prompt) -> list:
        """Build the ``messages`` list for the Chat Completions API."""
        messages = []

        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})

        # User message with interleaved text + images
        content: list = []
        for img in prompt.images:
            b64 = img.to_base64(fmt="JPEG")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": self.config.detail,
                },
            })

        content.append({"type": "text", "text": prompt.text})
        messages.append({"role": "user", "content": content})
        return messages

    # ------------------------------------------------------------------
    # generate (with retry)
    # ------------------------------------------------------------------

    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        if prompt.is_multimodal() and not self.supports_multimodal:
            raise NotImplementedError(
                f"Model {self.config.model_name!r} does not support images."
            )

        try:
            import openai
        except ImportError:
            raise ImportError("openai package required.")

        retryable = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )

        @retry(
            retry=retry_if_exception_type(retryable),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=self.config.retry_base_delay,
                max=self.config.retry_max_delay,
            ),
            reraise=True,
        )
        def _call() -> GenerationResult:
            client = self._get_client()
            messages = self._build_messages(prompt)

            logger.debug("OpenAI request: model=%s, images=%d, text_len=%d",
                         self.config.model_name, len(prompt.images), len(prompt.text))

            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return GenerationResult(
                text=text,
                model=response.model,
                finish_reason=finish_reason,
                usage=usage,
                metadata={**prompt.metadata, "provider": "openai"},
            )

        return _call()

    def generate_batch(self, prompts: List[Prompt], **kwargs) -> List[GenerationResult]:
        """Sequential calls with rate-limit pacing."""
        results = []
        rpm = self.config.requests_per_minute
        delay = (60.0 / rpm) if rpm else 0.0

        for i, p in enumerate(prompts):
            logger.debug("OpenAI batch: %d/%d", i + 1, len(prompts))
            results.append(self.generate(p, **kwargs))
            if delay > 0 and i < len(prompts) - 1:
                time.sleep(delay)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def supports_multimodal(self) -> bool:
        """Vision is supported for gpt-4o*, gpt-4-turbo, and gpt-4-vision models."""
        vision_prefixes = ("gpt-4o", "gpt-4-turbo", "gpt-4-vision")
        return any(self.config.model_name.startswith(p) for p in vision_prefixes)
