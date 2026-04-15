"""
Abstract model backend interface.

All model providers (Gemini, OpenAI, Qwen, …) implement this interface.
The generation pipeline only talks to ``ModelBackend`` — it never imports a
provider-specific SDK directly.  This makes swapping backends a one-line
config change.

Design
------
- ``Prompt`` is the single input type: text + optional images + optional system prompt.
  Multimodal content is represented as a list of ``PromptImage`` objects.
- ``GenerationResult`` is the single output type: text + metadata.
- ``generate_batch`` has a default sequential implementation.  Providers that
  support true batch inference (e.g. OpenAI Batch API) should override it.
- Rate limiting and retry logic are the backend's responsibility, not the
  pipeline's.  Each backend wires up a ``RateLimiter`` and ``RetryWithBackoff``
  from ``utils/rate_limiter.py``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Prompt primitives
# ---------------------------------------------------------------------------


@dataclass
class PromptImage:
    """
    An image to include in a prompt.

    Holds exactly one of: PIL Image, numpy array, file path, or raw bytes.
    Use ``to_pil()`` or ``to_bytes(fmt)`` to convert to whatever the backend
    SDK expects.
    """

    image: Optional[Any] = None          # PIL.Image.Image
    array: Optional[np.ndarray] = None   # (H, W, 3) uint8 numpy
    path: Optional[str] = None           # Absolute path to image file
    raw_bytes: Optional[bytes] = None    # Already-encoded JPEG/PNG bytes
    mime_type: str = "image/jpeg"

    @classmethod
    def from_numpy(cls, arr: np.ndarray, fmt: str = "JPEG") -> "PromptImage":
        arr = np.asarray(arr, dtype=np.uint8)
        buf = BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(arr).save(buf, format=fmt)
        return cls(raw_bytes=buf.getvalue(), mime_type=f"image/{fmt.lower()}")

    @classmethod
    def from_path(cls, path: str) -> "PromptImage":
        return cls(path=path)

    def to_pil(self) -> Any:
        """Return a PIL.Image.Image regardless of internal representation."""
        from PIL import Image as PILImage
        if self.image is not None:
            return self.image
        if self.array is not None:
            return PILImage.fromarray(self.array.astype(np.uint8))
        if self.path is not None:
            return PILImage.open(self.path).convert("RGB")
        if self.raw_bytes is not None:
            return PILImage.open(BytesIO(self.raw_bytes)).convert("RGB")
        raise ValueError("PromptImage has no image data.")

    def to_bytes(self, fmt: str = "JPEG") -> bytes:
        """Return encoded bytes."""
        if self.raw_bytes is not None:
            return self.raw_bytes
        buf = BytesIO()
        self.to_pil().save(buf, format=fmt)
        return buf.getvalue()

    def to_base64(self, fmt: str = "JPEG") -> str:
        """Return base64-encoded string (for OpenAI-style APIs)."""
        import base64
        return base64.b64encode(self.to_bytes(fmt)).decode()


@dataclass
class Prompt:
    """
    A single request to a model backend.

    ``images`` is an ordered list — backends preserve the order when
    constructing their API payload.  For models that do not support images,
    the backend raises ``NotImplementedError`` if ``images`` is non-empty.
    """

    text: str
    images: List[PromptImage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata forwarded verbatim to the result for tracing."""

    def is_multimodal(self) -> bool:
        return len(self.images) > 0


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationResult:
    """Normalised output from any model backend."""

    text: str
    """The model's response text."""
    model: str
    """Identifier of the model that produced this response."""
    finish_reason: str = "stop"
    usage: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Carries through ``prompt.metadata`` plus backend-specific extras
    (e.g. safety ratings from Gemini, logprobs from OpenAI).
    """

    def is_complete(self) -> bool:
        return self.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class ModelBackend(abc.ABC):
    """
    Abstract base class for all model provider backends.

    Implementing a new backend requires:
    1. Subclass ``ModelBackend``.
    2. Implement ``generate`` (and optionally ``generate_batch``).
    3. Set ``model_name``, ``provider``, ``supports_multimodal`` properties.
    4. Register the class in ``BackendRegistry``.

    The backend is responsible for:
    - Authentication (reading API keys from environment)
    - Request construction
    - Rate limiting and retry logic
    - Mapping the provider's response format to ``GenerationResult``
    """

    @abc.abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        """
        Submit a single prompt and return a result.

        Should raise ``RuntimeError`` on non-retryable failures and let
        tenacity handle retryable ones (HTTP 429, 503, timeouts).
        """
        ...

    def generate_batch(self, prompts: List[Prompt], **kwargs) -> List[GenerationResult]:
        """
        Submit multiple prompts.

        Default: sequential calls.  Override for providers with native batch
        endpoints (e.g. OpenAI Batch API).
        """
        return [self.generate(p, **kwargs) for p in prompts]

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Full model identifier, e.g. 'gemini-1.5-pro', 'gpt-4o'."""
        ...

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Short provider name, e.g. 'gemini', 'openai', 'qwen'."""
        ...

    @property
    @abc.abstractmethod
    def supports_multimodal(self) -> bool:
        """Whether this backend accepts images in prompts."""
        ...

    def health_check(self) -> bool:
        """
        Verify that the backend is reachable and credentials are valid.
        Returns True on success, False on failure.
        Default: send a trivial prompt and catch exceptions.
        """
        try:
            result = self.generate(Prompt(text="Hello."))
            return bool(result.text)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
