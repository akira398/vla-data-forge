"""
Google Gemini model backend.

Requires: pip install 'vla-data-curator[gemini]'
Environment variable: GOOGLE_API_KEY

Supported models (as of mid-2025):
  gemini-1.5-pro        — best reasoning, large context, multimodal
  gemini-1.5-flash      — faster, cheaper, still multimodal
  gemini-2.0-flash-exp  — experimental next-gen
  gemini-1.0-pro        — legacy text-only

Usage
-----
from vla_curator.backends.gemini import GeminiBackend
from vla_curator.config import GeminiConfig

cfg = GeminiConfig(model_name="gemini-1.5-pro", api_key_env_var="GOOGLE_API_KEY")
backend = GeminiBackend(cfg)
result = backend.generate(Prompt(text="Describe this robot action.", images=[...]))
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

from ..config import GeminiConfig
from .base import GenerationResult, ModelBackend, Prompt, PromptImage, TokenUsage

logger = logging.getLogger(__name__)

# Exceptions that warrant a retry
_RETRYABLE: tuple[type[Exception], ...] = ()  # populated lazily below


def _get_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Lazily import Gemini SDK exceptions for retry logic."""
    try:
        import google.api_core.exceptions as gae
        return (gae.ResourceExhausted, gae.ServiceUnavailable, gae.DeadlineExceeded)
    except ImportError:
        return (Exception,)


class GeminiBackend(ModelBackend):
    """
    Model backend wrapping the ``google-generativeai`` SDK.

    The client is initialised lazily on first use so that importing this module
    does not fail if the SDK is not installed (only a warning is issued).
    """

    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        self._client: Optional[Any] = None
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy SDK initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> Any:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for the Gemini backend. "
                "Install with: pip install 'vla-data-curator[gemini]'"
            ) from e

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{self.config.api_key_env_var}' is not set. "
                "Export your Google API key before running."
            )
        genai.configure(api_key=api_key)

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        safety_settings = self.config.safety_settings or {}

        model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings or None,
        )
        self._model = model
        return model

    @property
    def _sdk_model(self) -> Any:
        if self._model is None:
            self._init_client()
        return self._model

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_content(self, prompt: Prompt) -> list:
        """Build the ``contents`` list expected by the Gemini SDK."""
        parts: list = []

        if prompt.system_prompt:
            # Gemini 1.5+ supports system_instruction separately, but we
            # prepend it to the user turn for compatibility with 1.0.
            parts.append(prompt.system_prompt + "\n\n")

        # Interleave images and text
        for img in prompt.images:
            pil = img.to_pil()
            parts.append(pil)

        parts.append(prompt.text)
        return parts

    # ------------------------------------------------------------------
    # generate (with retry)
    # ------------------------------------------------------------------

    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        retryable = _get_retryable_exceptions()

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
            model = self._sdk_model
            content = self._build_content(prompt)

            logger.debug("Gemini request: %d images, text length=%d",
                         len(prompt.images), len(prompt.text))

            response = model.generate_content(content)

            finish_reason = "stop"
            try:
                finish_reason = response.candidates[0].finish_reason.name.lower()
            except (AttributeError, IndexError):
                pass

            text = ""
            try:
                text = response.text
            except ValueError:
                # Response blocked by safety filters
                text = "[BLOCKED]"
                finish_reason = "safety"
                logger.warning("Gemini response blocked by safety filters.")

            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = TokenUsage(
                    prompt_tokens=getattr(um, "prompt_token_count", 0),
                    completion_tokens=getattr(um, "candidates_token_count", 0),
                    total_tokens=getattr(um, "total_token_count", 0),
                )

            return GenerationResult(
                text=text,
                model=self.config.model_name,
                finish_reason=finish_reason,
                usage=usage,
                metadata={**prompt.metadata, "provider": "gemini"},
            )

        return _call()

    def generate_batch(self, prompts: List[Prompt], **kwargs) -> List[GenerationResult]:
        """
        Sequential calls with a small inter-request delay for rate limiting.
        Gemini does not have a batch endpoint as of 2025.
        """
        results = []
        rpm = self.config.requests_per_minute
        delay = (60.0 / rpm) if rpm else 0.0

        for i, p in enumerate(prompts):
            logger.debug("Batch progress: %d/%d", i + 1, len(prompts))
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
        return "gemini"

    @property
    def supports_multimodal(self) -> bool:
        return True
