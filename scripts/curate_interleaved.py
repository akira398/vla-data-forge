"""
CLI: Curate the interleaved ECoT + Bridge v2 dataset.

Loads both datasets, matches episodes by file-path key, aligns reasoning
traces with Bridge v2 observations, and writes the result as either RLDS
TFRecord files (default) or JSONL.

RLDS output (default, --format rlds)
--------------------------------------
Writes two TFDS-compatible datasets under output_dir:

  output_dir/vla_curated_dataset/full/1.0.0/
      — All Bridge v2 episodes.  Episodes without ECoT reasoning get
        empty strings for the reasoning fields.

  output_dir/vla_curated_dataset/reasoning_only/1.0.0/
      — Only episodes that have at least one reasoning annotation.

Load them later with:
  import tensorflow_datasets as tfds
  ds = tfds.builder_from_directory(
      "output_dir/vla_curated_dataset/full/1.0.0/"
  ).as_dataset(split="train")

JSONL output (--format jsonl)
------------------------------
Legacy format.  One JSON object per line; images saved separately.
Only matched episodes (ECoT + Bridge v2) are written.

Usage
-----
# RLDS, all episodes (default):
python scripts/curate_interleaved.py \\
    --bridge-path /datasets/bridge_orig \\
    --ecot-path /datasets/embodied_features_bridge

# RLDS, limit to 100 episodes for a quick test:
python scripts/curate_interleaved.py \\
    --bridge-path /datasets/bridge_orig \\
    --ecot-path /datasets/embodied_features_bridge \\
    --max-episodes 100

# JSONL (only matched episodes):
python scripts/curate_interleaved.py \\
    --bridge-path /datasets/bridge_orig \\
    --format jsonl

# Exact alignment (no reasoning propagation):
python scripts/curate_interleaved.py \\
    --bridge-path /datasets/bridge_orig \\
    --alignment exact
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
from vla_curator.curation.export import ExportFormat, JSONLExporter, create_exporter
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
        None, "--max-episodes", "-n",
        help="Cap on Bridge v2 episodes to process (useful for testing).",
    ),
    alignment: str = typer.Option(
        "nearest", "--alignment", "-a",
        help="Reasoning alignment strategy: exact | nearest | broadcast",
    ),
    bridge_source: str = typer.Option(
        "tfds", "--bridge-source",
        help="Bridge v2 data source: tfds | hdf5",
    ),
    bridge_path: Path = typer.Option(
        None, "--bridge-path",
        help="Path to Bridge v2 data (bridge_orig directory or HDF5 root).",
    ),
    ecot_path: Path = typer.Option(
        None, "--ecot-path",
        help="Path to embodied_features_bridge directory (or .json file).",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/curated"), "--output-dir", "-o",
    ),
    format: str = typer.Option(
        "rlds", "--format", "-f",
        help="Output format: rlds (default) | jsonl",
    ),
    no_validate: bool = typer.Option(False, "--no-validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    # ── Config ──────────────────────────────────────────────────────────────
    if config_file:
        cfg = load_config(config_file, CurationConfig)
        if max_episodes is not None:
            cfg.ecot.max_episodes = max_episodes
            cfg.bridge.max_episodes = max_episodes
        if ecot_path is not None:
            cfg.ecot.local_path = ecot_path
        if bridge_path is not None:
            cfg.bridge.local_path = bridge_path
    else:
        ecot_cfg = ECoTDatasetConfig(
            local_path=ecot_path,
            max_episodes=max_episodes,
        )
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

    fmt = ExportFormat(format.lower())

    console.print("[bold]Curation pipeline[/bold]")
    console.print(f"  ECoT:      {cfg.ecot.local_path or cfg.ecot.hf_repo}")
    console.print(f"  Bridge v2: {cfg.bridge.local_path or cfg.bridge.source}")
    console.print(f"  Alignment: {cfg.alignment_strategy}")
    console.print(f"  Format:    {fmt.value}")
    console.print(f"  Output:    {cfg.output_dir}")

    # ── Readers + interleaver ────────────────────────────────────────────────
    ecot_reader   = ECoTDatasetReader(cfg.ecot)
    bridge_reader = BridgeV2DatasetReader(cfg.bridge)
    interleaver   = EpisodeInterleaver(cfg, ecot_reader, bridge_reader)
    exporter      = create_exporter(fmt, cfg.output_dir)
    validator     = DatasetValidator() if cfg.validate_output else None

    # ── Episode source ───────────────────────────────────────────────────────
    # RLDS: iterate ALL Bridge v2 episodes (unmatched ones get empty reasoning).
    # JSONL: keep the old behaviour — only matched episodes.
    if fmt == ExportFormat.RLDS:
        console.print(
            "\n[bold]Mode:[/bold] all Bridge v2 episodes "
            "(unmatched → empty reasoning strings)"
        )
        episode_iter = interleaver.iter_all_episodes()
    else:
        console.print("\n[bold]Mode:[/bold] matched episodes only (ECoT + Bridge v2)")
        episode_iter = interleaver.iter_episodes()

    # ── Main loop ────────────────────────────────────────────────────────────
    total_written = 0
    validation_fails = 0

    console.print("[bold green]Starting curation…[/bold green]\n")
    for merged_ep in episode_iter:
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

        if total_written % 500 == 0:
            console.print(f"  …{total_written} episodes buffered")

    # ── Finalise ─────────────────────────────────────────────────────────────
    exporter.write_metadata({
        "total_episodes": total_written,
        "validation_fails": validation_fails,
        "alignment_strategy": cfg.alignment_strategy,
        "output_format": fmt.value,
        "schema_version": cfg.schema_version,
    })

    console.print(
        f"\n[green]Done.[/green] {total_written} episodes written, "
        f"{validation_fails} failed validation."
    )
    console.print(f"Output: [bold]{cfg.output_dir}[/bold]")

    if fmt == ExportFormat.RLDS:
        base = cfg.output_dir / "vla_curated_dataset"
        console.print("\nLoad datasets with:")
        console.print(
            f"  tfds.builder_from_directory('{base}/full/1.0.0/')"
        )
        console.print(
            f"  tfds.builder_from_directory('{base}/reasoning_only/1.0.0/')"
        )


if __name__ == "__main__":
    app()
