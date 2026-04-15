"""
Backend registry — maps provider names to backend classes.

Keeping the registry here (rather than in __init__.py) avoids circular
imports and makes it trivial to add new backends without touching existing
code.

Usage
-----
from vla_curator.backends.registry import create_backend
from vla_curator.config import GeminiConfig

cfg = GeminiConfig()
backend = create_backend(cfg)   # Returns GeminiBackend instance
"""

from __future__ import annotations

from typing import Dict, Type

from ..config import AnyBackendConfig, GeminiConfig, OpenAIConfig, QwenConfig
from .base import ModelBackend


class BackendRegistry:
    """
    Registry mapping provider name strings to ``ModelBackend`` subclasses.

    Adding a new provider
    ---------------------
    1. Write the backend class in a new file under ``backends/``.
    2. Call ``BackendRegistry.register("my_provider", MyBackend)`` in that file
       (at module level, after the class definition).
    3. Import the new module in ``backends/__init__.py`` so the registration
       runs at import time.
    """

    _registry: Dict[str, Type[ModelBackend]] = {}

    @classmethod
    def register(cls, provider: str, backend_cls: Type[ModelBackend]) -> None:
        cls._registry[provider] = backend_cls

    @classmethod
    def get(cls, provider: str) -> Type[ModelBackend]:
        if provider not in cls._registry:
            raise KeyError(
                f"Unknown provider {provider!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[provider]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ---------------------------------------------------------------------------
# Register built-in backends
# ---------------------------------------------------------------------------

def _register_backends() -> None:
    from .gemini import GeminiBackend
    from .openai_backend import OpenAIBackend
    from .qwen import QwenBackend

    BackendRegistry.register("gemini", GeminiBackend)
    BackendRegistry.register("openai", OpenAIBackend)
    BackendRegistry.register("qwen", QwenBackend)


_register_backends()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(config: AnyBackendConfig) -> ModelBackend:
    """
    Instantiate the correct ``ModelBackend`` subclass for a given config.

    Example
    -------
    from vla_curator.config import load_backend_config
    cfg = load_backend_config("configs/backends/gemini.yaml")
    backend = create_backend(cfg)
    """
    backend_cls = BackendRegistry.get(config.provider)
    return backend_cls(config)
