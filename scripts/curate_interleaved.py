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
    variants: str = typer.Option(
        "both", "--variants",
        help=(
            "Which RLDS variants to write (rlds format only): "
            "both (default) | full | reasoning_only"
        ),
    ),
    no_validate: bool = typer.Option(False, "--no-validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    # ── Config ──────────────────────────────────────────────────────────────
    if config_file:
        cfg = load_config(config_file, CurationConfig)
        if max_episodes is not None:
            # Only limit Bridge v2 — ECoT is a lookup table and must be
            # loaded in full, otherwise most Bridge episodes won't find
            # their ECoT match (different iteration order).
            cfg.bridge.max_episodes = max_episodes
            cfg.ecot.max_episodes = None
        if ecot_path is not None:
            cfg.ecot.local_path = ecot_path
        if bridge_path is not None:
            cfg.bridge.local_path = bridge_path
    else:
        ecot_cfg = ECoTDatasetConfig(
            local_path=ecot_path,
            # No max_episodes — ECoT is loaded as a lookup index, so all
            # entries are needed for matching regardless of Bridge v2 limit.
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

    # Parse --variants into a list for the RLDS exporter
    _variants_map = {
        "both":           ["full", "reasoning_only"],
        "full":           ["full"],
        "reasoning_only": ["reasoning_only"],
    }
    if variants.lower() not in _variants_map:
        console.print(
            f"[red]Unknown --variants value '{variants}'. "
            "Choose: both | full | reasoning_only[/red]"
        )
        raise typer.Exit(1)
    rlds_variants = _variants_map[variants.lower()]

    console.print("[bold]Curation pipeline[/bold]")
    console.print(f"  ECoT:      {cfg.ecot.local_path or cfg.ecot.hf_repo}")
    console.print(f"  Bridge v2: {cfg.bridge.local_path or cfg.bridge.source}")
    console.print(f"  Alignment: {cfg.alignment_strategy}")
    console.print(f"  Format:    {fmt.value}")
    if fmt == ExportFormat.RLDS:
        console.print(f"  Variants:  {', '.join(rlds_variants)}")
    console.print(f"  Output:    {cfg.output_dir}")

    # ── Readers + interleaver ────────────────────────────────────────────────
    ecot_reader   = ECoTDatasetReader(cfg.ecot)
    bridge_reader = BridgeV2DatasetReader(cfg.bridge)
    interleaver   = EpisodeInterleaver(cfg, ecot_reader, bridge_reader)
    exporter      = create_exporter(fmt, cfg.output_dir, variants=rlds_variants)
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
    total_written    = 0
    with_reasoning   = 0
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

        if merged_ep.has_reasoning():
            with_reasoning += 1

        exporter.export_episode(merged_ep)
        total_written += 1

        if total_written % 500 == 0:
            console.print(
                f"  …{total_written} episodes buffered "
                f"({with_reasoning} with reasoning)"
            )

    # ── Finalise ─────────────────────────────────────────────────────────────
    exporter.write_metadata({
        "total_episodes":    total_written,
        "validation_fails":  validation_fails,
        "alignment_strategy": cfg.alignment_strategy,
        "output_format":     fmt.value,
        "schema_version":    cfg.schema_version,
    })

    # ── Summary ──────────────────────────────────────────────────────────────
    without_reasoning = total_written - with_reasoning
    match_pct = 100.0 * with_reasoning / max(total_written, 1)

    from rich.table import Table
    t = Table(title="Curation summary", show_lines=True)
    t.add_column("Metric",  style="bold")
    t.add_column("Count",   justify="right")
    t.add_column("",        justify="right", style="dim")

    t.add_row("Total episodes",             str(total_written),    "")
    t.add_row("With ECoT reasoning",        str(with_reasoning),   f"{match_pct:.1f}%")
    t.add_row("Without ECoT reasoning",     str(without_reasoning),f"{100-match_pct:.1f}%")
    t.add_row("Validation failures",        str(validation_fails), "")
    console.print(t)

    console.print(f"\nOutput: [bold]{cfg.output_dir}[/bold]")

    if fmt == ExportFormat.RLDS:
        base = cfg.output_dir / "vla_curated_dataset"
        console.print("\n[bold]Load datasets with:[/bold]")
        if "full" in rlds_variants:
            console.print(f"  tfds.builder_from_directory('{base}/full/1.0.0/')")
        if "reasoning_only" in rlds_variants:
            if with_reasoning > 0:
                console.print(
                    f"  tfds.builder_from_directory('{base}/reasoning_only/1.0.0/')"
                )
            else:
                console.print(
                    "  [yellow]reasoning_only was skipped "
                    "(no ECoT matches found)[/yellow]"
                )


if __name__ == "__main__":
    app()
