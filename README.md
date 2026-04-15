# VLA Data Curator

A research-grade Python framework for curating and preprocessing datasets for
**Vision-Language-Action (VLA)** model training, with a focus on
reasoning-aware embodied datasets.

The primary workflow: load robot manipulation demonstrations from
[Embodied-CoT](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge)
and [Bridge v2](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/),
annotate episodes with structured reasoning traces using a pluggable model
backend (Gemini, GPT-4o, Qwen-VL), and produce merged interleaved episodes
ready for downstream VLA training.

---

## Repository Structure

```
vla-data-curator/
├── configs/
│   ├── datasets/          # embodied_cot.yaml, bridge_v2.yaml
│   ├── backends/          # gemini.yaml, openai.yaml, qwen.yaml
│   ├── generation/        # default.yaml (generation pipeline)
│   └── curation/          # interleaved.yaml (merge pipeline)
│
├── src/vla_curator/
│   ├── schemas/           # Canonical data types (dataclasses + Pydantic configs)
│   │   ├── base.py            RobotAction, NumpyArrayMixin
│   │   ├── embodied_cot.py    ReasoningTrace, ECoTStep, ECoTEpisode
│   │   ├── bridge_v2.py       BridgeObservation, BridgeStep, BridgeEpisode
│   │   ├── interleaved.py     EnrichedObservation, AlignedStep, InterleavedEpisode
│   │   └── modalities.py      DepthMap, SceneGraph, ModalityRegistry
│   │
│   ├── datasets/          # Dataset readers (abstract + concrete)
│   │   ├── base.py            DatasetReader ABC
│   │   ├── embodied_cot.py    ECoTDatasetReader (HuggingFace)
│   │   └── bridge_v2.py       BridgeV2DatasetReader (TFDS / HDF5 / RLDS)
│   │
│   ├── backends/          # Model provider abstraction
│   │   ├── base.py            ModelBackend ABC, Prompt, GenerationResult
│   │   ├── gemini.py          Google Gemini (google-generativeai SDK)
│   │   ├── openai_backend.py  OpenAI GPT-4o and vision models
│   │   ├── qwen.py            Qwen-VL (DashScope API or local HF)
│   │   └── registry.py        BackendRegistry, create_backend()
│   │
│   ├── generation/        # Reasoning-trace generation pipeline
│   │   ├── prompt_builder.py  ECoTPromptBuilder, frame sampling strategies
│   │   ├── response_parser.py ReasoningTraceParser (JSON extraction + fallbacks)
│   │   ├── trace_postprocessor.py  Cleaning, propagation, coverage stats
│   │   └── pipeline.py        GenerationPipeline (orchestrator + resume)
│   │
│   ├── curation/          # Interleaving + output pipeline
│   │   ├── interleaver.py     EpisodeInterleaver (ECoT × Bridge v2)
│   │   ├── validator.py       DatasetValidator, ValidationResult
│   │   └── export.py          JSONLExporter, HDF5Exporter (placeholder)
│   │
│   ├── visualization/
│   │   ├── frame_viewer.py    FrameViewer — frame grids, reasoning overlays, GIF
│   │   └── trajectory_viewer.py  TrajectoryViewer — action plots, coverage heatmaps
│   │
│   ├── utils/
│   │   ├── io.py              save_jsonl, load_jsonl, save_image, etc.
│   │   ├── logging.py         setup_logging, get_logger
│   │   └── rate_limiter.py    RateLimiter (token bucket), RetryWithBackoff
│   │
│   └── config.py          # Pydantic configs: DatasetConfig, BackendConfig, etc.
│
├── scripts/               # CLI entry points
│   ├── visualize_ecot.py      Inspect episodes with frame grids / reasoning traces
│   ├── generate_traces.py     Run reasoning-trace generation pipeline
│   ├── curate_interleaved.py  Merge ECoT + Bridge v2
│   └── validate_dataset.py    Validate curated JSONL output
│
├── tests/
│   ├── conftest.py            Shared fixtures (synthetic in-memory data)
│   ├── test_schemas.py        Schema unit tests
│   ├── test_backends.py       Backend interface + prompt construction tests
│   ├── test_generation.py     Prompt builder, response parser, postprocessor
│   └── test_curation.py       Interleaver, validator, exporter
│
├── notebooks/
│   └── explore_ecot.ipynb     Interactive exploration notebook
│
├── pyproject.toml
└── README.md
```

---

## Installation

### 1. Create a conda environment

```bash
conda create -n vla-forge python=3.11 -y
conda activate vla-forge
```

### 2. Clone and install

**Minimum (core + ECoT loading + visualisation):**

```bash
git clone https://github.com/akira398/vla-data-forge
cd vla-data-forge
pip install -e ".[viz]"
```

**With model backends:**

```bash
# Gemini
pip install -e ".[gemini]"
export GOOGLE_API_KEY="your-key"

# OpenAI
pip install -e ".[openai]"
export OPENAI_API_KEY="your-key"

# Qwen via DashScope API
pip install -e ".[qwen-api]"
export DASHSCOPE_API_KEY="your-key"

# Qwen local inference (requires GPU)
pip install -e ".[qwen-local]"
```

**Bridge v2 via tensorflow-datasets:**

```bash
pip install -e ".[bridge]"
```

**Everything:**

```bash
pip install -e ".[all]"
```

---

## Quick Start

### 1. Inspect Embodied-CoT episodes

```bash
# Show 3 episode grids in a matplotlib window
python scripts/visualize_ecot.py --max-episodes 3

# Save PNG grids and trajectory plots
python scripts/visualize_ecot.py --max-episodes 5 --mode summary --save-dir outputs/viz

# Show full reasoning trace for step 4 of episode 0
python scripts/visualize_ecot.py --episode-idx 0 --step-idx 4 --mode step
```

### 2. Generate reasoning traces

```bash
# With Gemini (default)
python scripts/generate_traces.py --max-episodes 10

# With GPT-4o
python scripts/generate_traces.py \
    --backend-config configs/backends/openai.yaml \
    --max-episodes 10

# Dry run (builds prompts but skips API calls)
python scripts/generate_traces.py --dry-run --max-episodes 5
```

### 3. Curate interleaved dataset

```bash
python scripts/curate_interleaved.py --max-episodes 100 --alignment nearest

# With Bridge v2 from local HDF5
python scripts/curate_interleaved.py \
    --bridge-source hdf5 \
    --bridge-path /data/bridge_v2 \
    --output-dir outputs/curated
```

### 4. Validate output

```bash
python scripts/validate_dataset.py outputs/curated/episodes.jsonl
python scripts/validate_dataset.py outputs/curated/episodes.jsonl \
    --min-reasoning 0.5 --report outputs/validation_report.json
```

---

## Dataset Assumptions

### Embodied-CoT (`Embodied-CoT/embodied_features_bridge`)

- Distributed via HuggingFace Datasets.
- Each row is an episode with a `steps` column.
- Each step contains: `observation/image_0` (uint8), `action` (float32 × 7),
  `language_instruction`, and optional reasoning fields under `reasoning/`.
- Episode IDs correspond to Bridge v2 source file paths.

If the actual column layout differs from the above, update `COLUMN_MAP` in
`src/vla_curator/datasets/embodied_cot.py`.

### Bridge v2

- Supports three loading modes: `tfds` (tensorflow-datasets), `hdf5`, `rlds`.
- Actions are 7-DoF: `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`.
- Two RGB cameras: `image_0` (primary) and `image_1` (secondary).

---

## Canonical Data Schema

### `InterleavedEpisode` (training-ready output)

```python
@dataclass
class InterleavedEpisode:
    episode_id:          str
    task_description:    str
    steps:               List[AlignedStep]
    alignment_metadata:  AlignmentMetadata
    provenance:          DataProvenance
    schema_version:      str = "1.0"

@dataclass
class AlignedStep:
    step_index:           int
    observation:          EnrichedObservation
    action:               np.ndarray          # (7,) float32
    reasoning:            Optional[ReasoningTrace]
    is_first / is_last:   bool
    alignment_confidence: float               # 1.0=direct, 0.7=propagated

@dataclass
class EnrichedObservation:
    image:               Optional[np.ndarray]  # (H,W,3) uint8
    state:               Optional[np.ndarray]  # (7,) float32
    depth_map:           DepthMap              # always present, check .valid
    scene_graph:         SceneGraph            # always present, check .valid
    extra_modalities:    Dict[str, Any]

@dataclass
class ReasoningTrace:
    task_reasoning:      str   # What is the overall task?
    subtask_reasoning:   str   # What subtask is active?
    move_reasoning:      str   # What arm motion and why?
    gripper_reasoning:   str   # Open or close? Why?
    attribute_reasoning: str   # Relevant object attributes
    spatial_reasoning:   str   # Spatial relationships
```

### Alignment strategies

| Strategy | Description |
|----------|-------------|
| `exact` | Only steps with a direct VLM annotation get reasoning |
| `nearest` | Propagate from nearest annotated step (default) |
| `broadcast` | Copy single episode-level trace to all steps |

---

## Supported Model Backends

| Provider | Config class | Env var | Notes |
|----------|-------------|---------|-------|
| Google Gemini | `GeminiConfig` | `GOOGLE_API_KEY` | Best multimodal reasoning |
| OpenAI GPT-4o | `OpenAIConfig` | `OPENAI_API_KEY` | Strong, widely available |
| Qwen-VL (API) | `QwenConfig(mode="api")` | `DASHSCOPE_API_KEY` | DashScope endpoint |
| Qwen-VL (local) | `QwenConfig(mode="local")` | — | Requires GPU |

### Adding a new backend

```python
# 1. Create src/vla_curator/backends/my_provider.py
from vla_curator.backends.base import ModelBackend, Prompt, GenerationResult

class MyProviderBackend(ModelBackend):
    def generate(self, prompt: Prompt, **kwargs) -> GenerationResult:
        # ... call your API ...
        return GenerationResult(text=response_text, model=self.model_name)

    @property
    def model_name(self) -> str: return "my-model-v1"

    @property
    def provider(self) -> str: return "my_provider"

    @property
    def supports_multimodal(self) -> bool: return True

# 2. Register it
from vla_curator.backends.registry import BackendRegistry
BackendRegistry.register("my_provider", MyProviderBackend)

# 3. Create a config class and add to AnyBackendConfig union in config.py
```

---

## Python API

```python
from vla_curator.config import ECoTDatasetConfig, GeminiConfig, GenerationConfig
from vla_curator.datasets import ECoTDatasetReader
from vla_curator.backends.registry import create_backend
from vla_curator.generation import GenerationPipeline
from vla_curator.visualization import FrameViewer, TrajectoryViewer

# Load a few episodes
cfg = ECoTDatasetConfig(max_episodes=5)
reader = ECoTDatasetReader(cfg)

for episode in reader:
    print(episode)
    # ECoTEpisode(id='...', steps=50, reasoning_coverage=40%)

# Visualize
viewer = FrameViewer()
first_ep = reader.take(1)[0]
viewer.show_episode(first_ep, max_frames=8)

# Generate traces
gen_cfg = GenerationConfig(
    dataset=cfg,
    backend=GeminiConfig(),
    output_dir="outputs/generated",
    dry_run=True,   # Don't call the API
)
backend = create_backend(gen_cfg.backend)
pipeline = GenerationPipeline(gen_cfg, backend, reader)
pipeline.run()
```

---

## Configuration System

All configs are Pydantic models and can be loaded from YAML:

```python
from vla_curator.config import load_config, GenerationConfig

cfg = load_config("configs/generation/default.yaml", GenerationConfig)
cfg.backend.model_name = "gemini-1.5-flash"   # Override at runtime
```

---

## Running Tests

```bash
# All tests (no API keys, no GPU needed)
pytest

# With coverage
pytest --cov=vla_curator --cov-report=html

# Specific module
pytest tests/test_schemas.py -v
pytest tests/test_generation.py -v
```

---

## Extending with New Modalities

To add a new modality (e.g. optical flow):

1. Add a dataclass to `schemas/modalities.py`:

   ```python
   @dataclass
   class OpticalFlow:
       valid: bool = False
       data: Optional[np.ndarray] = None   # (H, W, 2) float32
   ```

2. Add an `optical_flow: OpticalFlow` field to `EnrichedObservation` in
   `schemas/interleaved.py`.

3. Implement `OpticalFlowExtractor(ModalityExtractor)` in
   `src/vla_curator/modalities/optical_flow.py`.

4. Register:

   ```python
   ModalityRegistry.register("optical_flow", OpticalFlow, OpticalFlowExtractor)
   ```

5. Call the extractor in a per-frame enrichment step (e.g. a new
   `curation/enrichment.py` module).

The same pattern applies for depth maps (planned with Depth-Anything/ZoeDepth)
and scene graphs (planned with grounded VLM detection).

---

## Output Format (JSONL)

Each line in `outputs/curated/episodes.jsonl` is:

```json
{
  "schema_version": "1.0",
  "episode_id": "bridge_v2/...",
  "task_description": "pick up the orange",
  "alignment_metadata": {
    "strategy": "nearest",
    "num_steps_bridge": 45,
    "reasoning_coverage": 1.0
  },
  "provenance": {
    "generation_backend": "gemini",
    "generation_model": "gemini-1.5-pro"
  },
  "steps": [
    {
      "step_index": 0,
      "action": [0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
      "is_first": true,
      "alignment_confidence": 1.0,
      "observation": { "image_path": "images/bridge_v2_.../step_00000.png" },
      "reasoning": {
        "task_reasoning": "Pick up the orange from the table.",
        "subtask_reasoning": "Move the arm above the orange.",
        "move_reasoning": "...",
        "gripper_reasoning": "...",
        "attribute_reasoning": "...",
        "spatial_reasoning": "..."
      }
    }
  ]
}
```

---

## License

MIT
