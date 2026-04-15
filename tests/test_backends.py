"""
Tests for model backend interface and parsing.

All model calls are mocked — no real API keys required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vla_curator.backends.base import GenerationResult, ModelBackend, Prompt, PromptImage
from vla_curator.backends.registry import BackendRegistry, create_backend
from vla_curator.config import GeminiConfig, OpenAIConfig, QwenConfig


class TestPromptImage:
    def test_from_numpy_to_pil(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        pi = PromptImage.from_numpy(arr)
        pil = pi.to_pil()
        assert pil.size == (64, 64)

    def test_to_base64(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        pi = PromptImage.from_numpy(arr)
        b64 = pi.to_base64()
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_to_bytes(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        pi = PromptImage.from_numpy(arr)
        raw = pi.to_bytes()
        assert isinstance(raw, bytes)
        assert len(raw) > 0


class TestPrompt:
    def test_is_multimodal_with_images(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        p = Prompt(text="test", images=[PromptImage.from_numpy(arr)])
        assert p.is_multimodal()

    def test_not_multimodal_without_images(self):
        p = Prompt(text="test")
        assert not p.is_multimodal()


class TestGenerationResult:
    def test_is_complete(self):
        r = GenerationResult(text="hello", model="test", finish_reason="stop")
        assert r.is_complete()

    def test_not_complete_if_length(self):
        r = GenerationResult(text="truncated", model="test", finish_reason="length")
        assert not r.is_complete()


class ConcreteBackend(ModelBackend):
    """Minimal concrete implementation for testing the ABC."""

    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        return GenerationResult(text="test response", model="test-model")

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def provider(self) -> str:
        return "test"

    @property
    def supports_multimodal(self) -> bool:
        return True


class TestModelBackendABC:
    def test_concrete_implementation(self):
        backend = ConcreteBackend()
        result = backend.generate(Prompt(text="hello"))
        assert result.text == "test response"

    def test_generate_batch_uses_generate(self):
        backend = ConcreteBackend()
        prompts = [Prompt(text=f"p{i}") for i in range(3)]
        results = backend.generate_batch(prompts)
        assert len(results) == 3
        assert all(r.text == "test response" for r in results)

    def test_repr(self):
        backend = ConcreteBackend()
        assert "test-model" in repr(backend)


class TestBackendRegistry:
    def test_available_backends(self):
        available = BackendRegistry.available()
        assert "gemini" in available
        assert "openai" in available
        assert "qwen" in available

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError):
            BackendRegistry.get("nonexistent_provider")

    def test_create_backend_gemini(self):
        cfg = GeminiConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = create_backend(cfg)
        assert backend.provider == "gemini"
        assert backend.model_name == "gemini-1.5-pro"

    def test_create_backend_openai(self):
        cfg = OpenAIConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = create_backend(cfg)
        assert backend.provider == "openai"

    def test_create_backend_qwen(self):
        cfg = QwenConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = create_backend(cfg)
        assert backend.provider == "qwen"


class TestGeminiBackend:
    """Test GeminiBackend request construction without calling the API."""

    def test_build_content_text_only(self):
        from vla_curator.backends.gemini import GeminiBackend
        cfg = GeminiConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = GeminiBackend(cfg)
        prompt = Prompt(text="hello world")
        content = backend._build_content(prompt)
        assert "hello world" in content

    def test_build_content_with_system(self):
        from vla_curator.backends.gemini import GeminiBackend
        cfg = GeminiConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = GeminiBackend(cfg)
        prompt = Prompt(text="hello", system_prompt="You are a robot expert.")
        content = backend._build_content(prompt)
        assert any("robot expert" in str(c) for c in content)


class TestOpenAIBackend:
    """Test OpenAIBackend message construction."""

    def test_build_messages_text_only(self):
        from vla_curator.backends.openai_backend import OpenAIBackend
        cfg = OpenAIConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = OpenAIBackend(cfg)
        prompt = Prompt(text="describe this")
        msgs = backend._build_messages(prompt)
        assert msgs[-1]["role"] == "user"
        assert any("describe this" in str(c) for c in msgs[-1]["content"])

    def test_build_messages_with_system(self):
        from vla_curator.backends.openai_backend import OpenAIBackend
        cfg = OpenAIConfig(api_key_env_var="NONEXISTENT_KEY")
        backend = OpenAIBackend(cfg)
        prompt = Prompt(text="hello", system_prompt="You are helpful.")
        msgs = backend._build_messages(prompt)
        assert msgs[0]["role"] == "system"
        assert "helpful" in msgs[0]["content"]

    def test_supports_multimodal_gpt4o(self):
        from vla_curator.backends.openai_backend import OpenAIBackend
        cfg = OpenAIConfig(model_name="gpt-4o", api_key_env_var="NONEXISTENT_KEY")
        backend = OpenAIBackend(cfg)
        assert backend.supports_multimodal

    def test_not_multimodal_gpt35(self):
        from vla_curator.backends.openai_backend import OpenAIBackend
        cfg = OpenAIConfig(model_name="gpt-3.5-turbo", api_key_env_var="NONEXISTENT_KEY")
        backend = OpenAIBackend(cfg)
        assert not backend.supports_multimodal
