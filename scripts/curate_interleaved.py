"""
CLI: Curate the interleaved ECoT + Bridge v2 dataset.

Loads both datasets, matches episodes by ID, aligns reasoning traces with
Bridge v2 observations, and writes InterleavedEpisode objects to JSONL.

Usage
-----
# Default: load from HF + TFDS, write to outputs/curated:
python scripts/curate_interleaved.py

# With custom config file:
python scripts/curate_interleaved.py --config configs/curation/interleaved.yaml

# Limit episodes for quick testing:
python scripts/curate_interleaved.py --max-episodes 50

# Use exact alignment (only keep steps with direct annotations):
python scripts/curate_interleaved.py --alignment exact

# Use HDF5 Bridge v2 source:
python scripts/curate_interleaved.py --bridge-source hdf5 --bridge-path /data/bridge_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.config import (
    BridgeV2DatasetConfig,
    CurationConfig,
    ECoTDatasetConfig,
    load_config,
)
from vla_curator.curation.export import ExportFormat, JSONLExporter
from vla_curator.curation.interleaver import EpisodeInterleaver
from vla_curator.curation.validator import DatasetValidator
from vla_curator.datasets.bridge_v2 import BridgeV2DatasetReader
from vla_curator.datasets.embodied_cot import ECoTDatasetReader
from vla_curator.utils.logging import setup_logging

app = typer.Typer(help="Curate interleaved ECoT + Bridge v2 dataset.")
console = Console()


@app.command()
def main(
    config_file: Path = typer.Option(
        None, "--config", "-c", help="CurationConfig YAML file."
    ),
    max_episodes: int = typer.Option(
        None, "--max-episodes", "-n"
    ),
    alignment: str = typer.Option(
        "nearest", "--alignment", "-a", help="exact | nearest | broadcast"
    ),
    bridge_source: str = typer.Option("tfds", "--bridge-source"),
    bridge_path: Path = typer.Option(None, "--bridge-path"),
    output_dir: Path = typer.Option(Path("outputs/curated"), "--output-dir", "-o"),
    no_validate: bool = typer.Option(False, "--no-validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    if config_file:
        cfg = load_config(config_file, CurationConfig)
        if max_episodes is not None:
            cfg.ecot.max_episodes = max_episodes
            cfg.bridge.max_episodes = max_episodes
    else:
        ecot_cfg = ECoTDatasetConfig(max_episodes=max_episodes)
        bridge_cfg = BridgeV2DatasetConfig(
            source=bridge_source,
            local_path=bridge_path,
            max_episodes=max_episodes,
        )
        cfg = CurationConfig(
            ecot=ecot_cfg,
            bridge=bridge_cfg,
            alignment_strategy=alignment,
            output_dir=output_dir,
            validate_output=not no_validate,
        )

    console.print("[bold]Curation pipeline[/bold]")
    console.print(f"  ECoT:      {cfg.ecot.hf_repo}")
    console.print(f"  Bridge v2: {cfg.bridge.source}")
    console.print(f"  Alignment: {cfg.alignment_strategy}")
    console.print(f"  Output:    {cfg.output_dir}")
    console.print(f"  Validate:  {cfg.validate_output}")

    ecot_reader   = ECoTDatasetReader(cfg.ecot)
    bridge_reader = BridgeV2DatasetReader(cfg.bridge)

    interleaver = EpisodeInterleaver(cfg, ecot_reader, bridge_reader)
    exporter = JSONLExporter(
        output_dir=cfg.output_dir,
        save_images=cfg.save_images,
        image_dir=cfg.image_output_dir,
    )
    validator = DatasetValidator() if cfg.validate_output else None

    total_written = 0
    validation_fails = 0

    console.print("\n[bold green]Starting curation…[/bold green]")
    for merged_ep in interleaver.iter_episodes():
        if validator:
            result = validator.validate_episode(merged_ep)
            if not result.passed:
                console.print(
                    f"[yellow]Validation fail[/yellow] {merged_ep.episode_id}: "
                    f"{result.errors}"
                )
                validation_fails += 1
                continue

        exporter.export_episode(merged_ep)
        total_written += 1

    # Write metadata
    exporter.write_metadata({
        "total_episodes": total_written,
        "validation_fails": validation_fails,
        "alignment_strategy": cfg.alignment_strategy,
        "schema_version": cfg.schema_version,
    })

    console.print(
        f"\n[green]Done.[/green] {total_written} episodes written, "
        f"{validation_fails} failed validation."
    )
    console.print(f"Output: [bold]{cfg.output_dir}[/bold]")


if __name__ == "__main__":
    app()
