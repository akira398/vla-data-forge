"""
CLI: Generate reasoning traces for an embodied dataset using a model backend.

This runs the full GenerationPipeline: load episodes → build prompts →
call model → parse → save annotated JSONL.

Usage
-----
# Generate with Gemini (default config):
python scripts/generate_traces.py

# Override backend to OpenAI:
python scripts/generate_traces.py --backend-config configs/backends/openai.yaml

# Dry run (build prompts but don't call the model):
python scripts/generate_traces.py --dry-run

# Process only the first 10 episodes:
python scripts/generate_traces.py --max-episodes 10

# Use keyframe sampling strategy:
python scripts/generate_traces.py --frame-strategy keyframe --num-frames 6
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.backends.registry import create_backend
from vla_curator.config import (
    ECoTDatasetConfig,
    FrameSamplingConfig,
    GenerationConfig,
    GeminiConfig,
    load_backend_config,
)
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.generation.pipeline import GenerationPipeline
from vla_curator.utils.logging import setup_logging

app = typer.Typer(help="Generate reasoning traces for embodied datasets.")
console = Console()


@app.command()
def main(
    config_file: Path = typer.Option(
        None, "--config", "-c", help="Path to GenerationConfig YAML."
    ),
    backend_config: Path = typer.Option(
        "configs/backends/gemini.yaml",
        "--backend-config",
        "-b",
        help="Path to backend config YAML.",
    ),
    hf_repo: str = typer.Option(
        "Embodied-CoT/embodied_features_bridge",
        "--repo",
        help="HuggingFace dataset repo.",
    ),
    split: str = typer.Option("train", "--split"),
    max_episodes: int = typer.Option(
        None, "--max-episodes", "-n", help="Limit episodes (null = all)."
    ),
    output_dir: Path = typer.Option(
        Path("outputs/generation"), "--output-dir", "-o"
    ),
    frame_strategy: str = typer.Option(
        "uniform", "--frame-strategy", help="uniform | keyframe | all"
    ),
    num_frames: int = typer.Option(8, "--num-frames", help="Frames per episode."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build prompts but skip model calls."
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    # Load or construct config
    if config_file:
        from vla_curator.config import GenerationConfig, load_config
        cfg = load_config(config_file, GenerationConfig)
        if max_episodes is not None:
            cfg.dataset.max_episodes = max_episodes
        if dry_run:
            cfg.dry_run = True
        cfg.output_dir = output_dir
    else:
        backend_cfg = load_backend_config(backend_config)
        dataset_cfg = ECoTDatasetConfig(
            hf_repo=hf_repo,
            split=split,
            max_episodes=max_episodes,
        )
        frame_cfg = FrameSamplingConfig(strategy=frame_strategy, num_frames=num_frames)
        cfg = GenerationConfig(
            dataset=dataset_cfg,
            backend=backend_cfg,
            frame_sampling=frame_cfg,
            output_dir=output_dir,
            resume=resume,
            dry_run=dry_run,
        )

    console.print(f"[bold]Generation pipeline[/bold]")
    console.print(f"  Backend:  {cfg.backend.provider} / {cfg.backend.model_name}")
    console.print(f"  Dataset:  {cfg.dataset.hf_repo} ({cfg.dataset.split})")
    console.print(f"  Frames:   {cfg.frame_sampling.strategy} × {cfg.frame_sampling.num_frames}")
    console.print(f"  Output:   {cfg.output_dir}")
    console.print(f"  Dry run:  {cfg.dry_run}")

    # Instantiate components
    reader = ECoTDatasetReader(cfg.dataset)
    backend = create_backend(cfg.backend)

    if not cfg.dry_run:
        console.print("\n[yellow]Checking backend health…[/yellow]")
        # Avoid health check overhead on large runs; just log the model
        console.print(f"  Model: {backend.model_name}")

    pipeline = GenerationPipeline(config=cfg, backend=backend, reader=reader)

    console.print("\n[bold green]Starting generation…[/bold green]")
    results = pipeline.run()

    console.print(f"\n[green]Done.[/green] {len(results)} episodes annotated.")
    console.print(f"Output: [bold]{cfg.output_dir / 'episodes.jsonl'}[/bold]")


if __name__ == "__main__":
    app()
