"""
Pydantic-based configuration schemas for all pipeline components.

We use Pydantic v2 BaseModel for configs (not dataclasses) because:
  - YAML/dict → typed object with validation at load time
  - `.model_dump()` for clean serialisation alongside experiment results
  - Field defaults + validators keep configs self-documenting

YAML files in configs/ hold actual values; these classes define the schema
and serve as the single source of truth for what fields exist.

Loading a config
----------------
    from vla_curator.config import GenerationConfig, load_config
    cfg = load_config("configs/generation/default.yaml", GenerationConfig)

All path-valued fields use pathlib.Path so callers get type-safe path ops.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Common fields shared across all dataset sources."""

    name: str
    split: str = "train"
    max_episodes: Optional[int] = None
    """Cap the number of episodes loaded.  Useful for quick experiments."""
    shuffle: bool = False
    shuffle_seed: int = 42
    image_size: tuple[int, int] = (256, 256)
    """(H, W) target size for all images.  None = no resizing."""
    cache_dir: Optional[Path] = None
    """Local cache for HuggingFace or TFDS downloads."""


class ECoTDatasetConfig(DatasetConfig):
    """Config for the Embodied-CoT / embodied_features_bridge HF dataset."""

    name: str = "embodied_features_bridge"
    hf_repo: str = "Embodied-CoT/embodied_features_bridge"
    """HuggingFace Hub dataset repository."""
    reasoning_columns: list[str] = Field(
        default=[
            "task_reasoning",
            "subtask_reasoning",
            "move_reasoning",
            "gripper_reasoning",
            "attribute_reasoning",
            "spatial_reasoning",
        ]
    )
    require_reasoning: bool = False
    """If True, skip episodes where no reasoning traces are present."""


class BridgeV2DatasetConfig(DatasetConfig):
    """
    Config for Bridge v2 data.

    Recommended setup (OpenVLA download)
    -------------------------------------
    After running the OpenVLA wget + mv commands you have:

        <BASE_DIR>/bridge_orig/1.0.0/dataset_info.json
                                      *.tfrecord.gz ...

    Set:
        source: tfds
        local_path: <BASE_DIR>/bridge_orig   # the renamed folder, NOT the parent

    The reader will auto-discover the version subdirectory (1.0.0/) inside it
    and call ``tfds.builder_from_directory`` — no network access required.
    """

    name: str = "bridge_v2"
    source: Literal["tfds", "hdf5", "rlds"] = "tfds"
    """
    tfds  — local TFDS-format dataset (bridge_orig layout from OpenVLA).
            Requires tensorflow + tensorflow-datasets.
    hdf5  — per-episode HDF5 files (older Bridge v2 distribution).
    rlds  — alias for tfds; kept for backward compatibility.
    """
    local_path: Optional[Path] = None
    """
    Path to the dataset root folder.

    For ``source=tfds``:  path to the ``bridge_orig`` folder
                          (the one containing the ``1.0.0/`` version directory).
    For ``source=hdf5``:  path to a directory tree of .hdf5 / .h5 files.
    """
    tfds_name: str = "bridge_dataset"
    """
    Only used as a fallback when ``local_path`` is None and the dataset must
    be found in the default TFDS cache (rare).  The OpenVLA workflow always
    sets ``local_path``, so this field is effectively unused in that case.
    """
    include_secondary_camera: bool = True

    @model_validator(mode="after")
    def check_local_path(self) -> "BridgeV2DatasetConfig":
        if self.source in ("hdf5", "rlds") and self.local_path is None:
            raise ValueError(
                f"local_path must be set when source='{self.source}'"
            )
        return self


# ---------------------------------------------------------------------------
# Backend (model provider) configs
# ---------------------------------------------------------------------------


class BackendConfig(BaseModel):
    """Common fields for all model provider backends."""

    provider: Literal["gemini", "openai", "qwen"] = "gemini"
    model_name: str
    api_key_env_var: str
    """Name of the environment variable that holds the API key."""
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 5
    retry_base_delay: float = 1.0
    """Initial back-off delay in seconds (doubles each retry)."""
    retry_max_delay: float = 60.0
    requests_per_minute: Optional[int] = None
    """Rate limit. None = no limit enforced by the client."""
    timeout: float = 120.0
    """Per-request timeout in seconds."""

    @field_validator("api_key_env_var")
    @classmethod
    def key_env_exists_warning(cls, v: str) -> str:
        if not os.environ.get(v):
            import warnings
            warnings.warn(
                f"Environment variable '{v}' is not set. "
                "API calls will fail until it is exported.",
                stacklevel=2,
            )
        return v


class GeminiConfig(BackendConfig):
    provider: Literal["gemini"] = "gemini"
    model_name: str = "gemini-1.5-pro"
    api_key_env_var: str = "GOOGLE_API_KEY"
    safety_settings: Dict[str, str] = Field(default_factory=dict)
    """Gemini-specific safety threshold overrides."""


class OpenAIConfig(BackendConfig):
    provider: Literal["openai"] = "openai"
    model_name: str = "gpt-4o"
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    """Override for Azure OpenAI or compatible endpoints."""
    detail: Literal["auto", "low", "high"] = "auto"
    """Image detail level for vision requests."""


class QwenConfig(BackendConfig):
    """
    Qwen / Qwen-VL backend config.

    ``mode`` controls whether to call the DashScope API or run a local
    HuggingFace checkpoint.  Local mode requires the ``qwen-local`` extras.
    """

    provider: Literal["qwen"] = "qwen"
    model_name: str = "qwen-vl-max"
    api_key_env_var: str = "DASHSCOPE_API_KEY"
    mode: Literal["api", "local"] = "api"
    local_model_path: Optional[Path] = None
    """Path to a local Qwen-VL checkpoint (required when mode='local')."""
    device: str = "cuda"

    @model_validator(mode="after")
    def check_local_path(self) -> "QwenConfig":
        if self.mode == "local" and self.local_model_path is None:
            raise ValueError("local_model_path must be set when mode='local'")
        return self


AnyBackendConfig = Union[GeminiConfig, OpenAIConfig, QwenConfig]


# ---------------------------------------------------------------------------
# Generation pipeline config
# ---------------------------------------------------------------------------


class FrameSamplingConfig(BaseModel):
    """Controls which frames are sent to the model for annotation."""

    strategy: Literal["uniform", "keyframe", "all"] = "uniform"
    """
    uniform   — sample N evenly-spaced frames from the episode.
    keyframe  — sample frames at action transitions (gripper open/close, large Δ).
    all       — send every frame (expensive; use only for short episodes).
    """
    num_frames: int = 8
    """Number of frames to sample per episode (for 'uniform' strategy)."""
    keyframe_threshold: float = 0.05
    """Minimum L∞ norm of action delta to count as a keyframe transition."""


class GenerationConfig(BaseModel):
    """Full config for the reasoning-trace generation pipeline."""

    dataset: ECoTDatasetConfig = Field(default_factory=ECoTDatasetConfig)
    backend: AnyBackendConfig = Field(default_factory=GeminiConfig)
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    output_dir: Path = Path("outputs/generation")
    batch_size: int = 1
    """Number of episodes to process in one scheduler batch."""
    resume: bool = True
    """Skip episodes already present in output_dir."""
    dry_run: bool = False
    """Build prompts but do not call the model. For testing prompt logic."""
    save_raw_responses: bool = True
    """Store the raw model response alongside the parsed trace."""
    num_workers: int = 1
    """Parallel episode workers (each worker sends sequential API requests)."""


# ---------------------------------------------------------------------------
# Curation / interleaving config
# ---------------------------------------------------------------------------


class CurationConfig(BaseModel):
    """Config for the ECoT + Bridge v2 interleaving pipeline."""

    ecot: ECoTDatasetConfig = Field(default_factory=ECoTDatasetConfig)
    bridge: BridgeV2DatasetConfig = Field(default_factory=BridgeV2DatasetConfig)
    alignment_strategy: str = "nearest"
    """One of AlignmentStrategy enum values: 'exact', 'nearest', 'broadcast'."""
    output_dir: Path = Path("outputs/curated")
    export_format: Literal["jsonl", "hdf5"] = "jsonl"
    schema_version: str = "1.0"
    validate_output: bool = True
    """Run structural validation on each episode before writing."""
    save_images: bool = False
    """
    Whether to embed images as base64 in the JSONL output.
    False (default) = save images to disk and store paths.
    Disk-path mode is strongly preferred for large-scale runs.
    """
    image_output_dir: Optional[Path] = None
    """Where to save frame images when save_images=False. Defaults to output_dir/images."""

    @model_validator(mode="after")
    def set_image_dir(self) -> "CurationConfig":
        if self.image_output_dir is None:
            self.image_output_dir = self.output_dir / "images"
        return self


# ---------------------------------------------------------------------------
# Loader utility
# ---------------------------------------------------------------------------


_CONFIG_CLASS_MAP: Dict[str, type] = {
    "generation": GenerationConfig,
    "curation": CurationConfig,
    "ecot_dataset": ECoTDatasetConfig,
    "bridge_dataset": BridgeV2DatasetConfig,
    "gemini": GeminiConfig,
    "openai": OpenAIConfig,
    "qwen": QwenConfig,
}


def load_config(path: Union[str, Path], config_cls: type) -> Any:
    """
    Load a YAML config file and parse it into a typed Pydantic model.

    Example
    -------
    cfg = load_config("configs/generation/default.yaml", GenerationConfig)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_cls(**data)


def load_backend_config(path: Union[str, Path]) -> AnyBackendConfig:
    """
    Load a backend config, automatically selecting the right subclass
    based on the ``provider`` field in the YAML.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    provider = data.get("provider", "gemini")
    cls_map = {"gemini": GeminiConfig, "openai": OpenAIConfig, "qwen": QwenConfig}
    if provider not in cls_map:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(cls_map)}")
    return cls_map[provider](**data)
