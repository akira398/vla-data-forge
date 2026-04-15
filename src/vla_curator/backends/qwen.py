"""
Qwen / Qwen-VL model backend.

Two operating modes (set via ``QwenConfig.mode``):

  api    — DashScope API (Alibaba Cloud).  Requires DASHSCOPE_API_KEY.
           Uses the OpenAI-compatible endpoint at
           https://dashscope.aliyuncs.com/compatible-mode/v1
           Supported models: qwen-vl-max, qwen-vl-plus, qwen2-vl-72b-instruct

  local  — HuggingFace transformers (Qwen-VL / Qwen2-VL checkpoints).
           Requires: pip install 'vla-data-curator[qwen-local]'
           Set ``local_model_path`` to a local checkpoint directory or
           use a HuggingFace repo ID (downloaded automatically).

Requires (API mode):  pip install 'vla-data-curator[qwen-api]'
Requires (local):     pip install 'vla-data-curator[qwen-local]'
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

from ..config import QwenConfig
from .base import GenerationResult, ModelBackend, Prompt, TokenUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API mode (DashScope via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


class _QwenAPIMode:
    """Internal helper that calls DashScope via the openai-compatible API."""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package required for Qwen API mode. "
                "Install with: pip install 'vla-data-curator[qwen-api]'"
            ) from e

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{self.config.api_key_env_var}' is not set."
            )

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self.DASHSCOPE_BASE_URL,
            timeout=self.config.timeout,
            max_retries=0,
        )
        return self._client

    def call(self, prompt: Prompt) -> GenerationResult:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for Qwen API mode.")

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
        def _inner() -> GenerationResult:
            client = self._get_client()
            messages = []

            if prompt.system_prompt:
                messages.append({"role": "system", "content": prompt.system_prompt})

            content: list = []
            for img in prompt.images:
                b64 = img.to_base64(fmt="JPEG")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            content.append({"type": "text", "text": prompt.text})
            messages.append({"role": "user", "content": content})

            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            return GenerationResult(
                text=choice.message.content or "",
                model=self.config.model_name,
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
                metadata={**prompt.metadata, "provider": "qwen", "mode": "api"},
            )

        return _inner()


# ---------------------------------------------------------------------------
# Local mode (HuggingFace transformers)
# ---------------------------------------------------------------------------


class _QwenLocalMode:
    """Internal helper that runs a Qwen-VL model locally via transformers."""

    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers, torch, and accelerate are required for local Qwen. "
                "Install with: pip install 'vla-data-curator[qwen-local]'"
            ) from e

        model_id = str(self.config.local_model_path) if self.config.local_model_path else self.config.model_name
        logger.info("Loading Qwen model from %s …", model_id)

        # Qwen2-VL uses AutoProcessor; Qwen-VL uses AutoTokenizer
        try:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.config.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        logger.info("Qwen model loaded.")

    def call(self, prompt: Prompt) -> GenerationResult:
        if self._model is None:
            self._load()

        import torch

        # Build query in Qwen-VL format
        query_parts = []
        for img in prompt.images:
            pil = img.to_pil()
            query_parts.append({"image": pil})
        query_parts.append({"text": prompt.text})

        # Try Qwen2-VL processor path
        if self._processor is not None:
            inputs = self._processor(
                text=[prompt.text],
                images=[img.to_pil() for img in prompt.images] if prompt.images else None,
                return_tensors="pt",
            ).to(self.config.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature or None,
                )
            text = self._processor.decode(output_ids[0], skip_special_tokens=True)
        else:
            # Qwen-VL tokenizer path
            tokenizer = self._tokenizer
            full_text = (prompt.system_prompt + "\n" if prompt.system_prompt else "") + prompt.text
            inputs = tokenizer(full_text, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                )
            text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return GenerationResult(
            text=text,
            model=self.config.model_name,
            finish_reason="stop",
            metadata={**prompt.metadata, "provider": "qwen", "mode": "local"},
        )


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------


class QwenBackend(ModelBackend):
    """
    Qwen model backend supporting both API (DashScope) and local inference.

    Select mode via ``QwenConfig.mode``.
    """

    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        if config.mode == "api":
            self._impl = _QwenAPIMode(config)
        else:
            self._impl = _QwenLocalMode(config)

    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        return self._impl.call(prompt)

    def generate_batch(self, prompts: List[Prompt], **kwargs) -> List[GenerationResult]:
        results = []
        rpm = self.config.requests_per_minute
        delay = (60.0 / rpm) if (rpm and self.config.mode == "api") else 0.0

        for i, p in enumerate(prompts):
            results.append(self.generate(p, **kwargs))
            if delay > 0 and i < len(prompts) - 1:
                time.sleep(delay)
        return results

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def provider(self) -> str:
        return "qwen"

    @property
    def supports_multimodal(self) -> bool:
        return True
