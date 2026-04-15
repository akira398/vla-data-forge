"""
CLI: Validate a curated interleaved dataset.

Reads a JSONL output file produced by curate_interleaved.py, reconstructs
InterleavedEpisode objects (action arrays only, no images), and runs the
DatasetValidator.

Usage
-----
python scripts/validate_dataset.py --input outputs/curated/episodes.jsonl

# Strict mode (fail on any episode with <50% reasoning coverage):
python scripts/validate_dataset.py \\
    --input outputs/curated/episodes.jsonl \\
    --min-reasoning 0.5

# Output a failure report:
python scripts/validate_dataset.py \\
    --input outputs/curated/episodes.jsonl \\
    --report outputs/validation_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vla_curator.curation.validator import DatasetValidator
from vla_curator.schemas.interleaved import (
    AlignedStep,
    AlignmentMetadata,
    DataProvenance,
    EnrichedObservation,
    InterleavedEpisode,
)
from vla_curator.schemas.embodied_cot import ReasoningTrace
from vla_curator.utils.logging import setup_logging

app = typer.Typer(help="Validate a curated interleaved dataset.")
console = Console()


def _dict_to_episode(d: dict) -> InterleavedEpisode:
    """Reconstruct a lightweight InterleavedEpisode from a JSONL dict (no images)."""
    steps = []
    for s in d.get("steps", []):
        obs_d = s.get("observation", {})
        obs = EnrichedObservation(
            step_index=obs_d.get("step_index", s.get("step_index", 0)),
            image_path=obs_d.get("image_path"),
            state=np.array(obs_d["state"], dtype=np.float32) if obs_d.get("state") else None,
        )
        action_list = s.get("action")
        action = np.array(action_list, dtype=np.float32) if action_list else np.zeros(7, dtype=np.float32)

        reasoning = None
        r = s.get("reasoning")
        if r:
            reasoning = ReasoningTrace(**{
                k: r.get(k) for k in [
                    "task_reasoning", "subtask_reasoning", "move_reasoning",
                    "gripper_reasoning", "attribute_reasoning", "spatial_reasoning",
                ]
            })

        step = AlignedStep(
            step_index=s.get("step_index", 0),
            observation=obs,
            action=action,
            reasoning=reasoning,
            is_first=s.get("is_first", False),
            is_last=s.get("is_last", False),
            source_dataset=s.get("source_dataset", ""),
            alignment_confidence=s.get("alignment_confidence", 1.0),
        )
        steps.append(step)

    am_d = d.get("alignment_metadata", {})
    prov_d = d.get("provenance", {})

    return InterleavedEpisode(
        episode_id=d.get("episode_id", ""),
        task_description=d.get("task_description", ""),
        steps=steps,
        alignment_metadata=AlignmentMetadata(**{
            k: am_d.get(k, v)
            for k, v in AlignmentMetadata().__dict__.items()
        }),
        provenance=DataProvenance(**{
            k: prov_d.get(k, v)
            for k, v in DataProvenance().__dict__.items()
        }),
        schema_version=d.get("schema_version", "1.0"),
    )


@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to episodes.jsonl"),
    min_steps: int = typer.Option(1, "--min-steps"),
    min_reasoning: float = typer.Option(0.0, "--min-reasoning", help="Minimum reasoning coverage (0–1)."),
    report: Path = typer.Option(None, "--report", "-r", help="Save JSON report here."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Validating[/bold] {input_file}")

    validator = DatasetValidator(
        min_steps=min_steps,
        min_reasoning_coverage=min_reasoning,
    )

    episodes = []
    count = 0
    from vla_curator.utils.io import load_jsonl
    for d in load_jsonl(input_file):
        if max_episodes and count >= max_episodes:
            break
        try:
            ep = _dict_to_episode(d)
            episodes.append(ep)
        except Exception as exc:
            console.print(f"[yellow]Could not parse episode {count}: {exc}[/yellow]")
        count += 1

    console.print(f"Loaded {len(episodes)} episodes.")

    report_obj = validator.validate_dataset(episodes)

    # Summary
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Total episodes", str(report_obj.total))
    table.add_row("Passed", f"[green]{report_obj.passed}[/green]")
    table.add_row("Failed", f"[red]{report_obj.failed}[/red]")
    table.add_row("Pass rate", f"{report_obj.pass_rate:.1%}")
    console.print(table)

    # Failed episodes detail
    if report_obj.failed > 0:
        console.print(f"\n[bold red]Failed episodes:[/bold red]")
        for r in report_obj.failed_episodes()[:10]:
            console.print(f"  {r.episode_id}: {r.errors}")
        if report_obj.failed > 10:
            console.print(f"  … and {report_obj.failed - 10} more.")

    # Save report
    if report:
        report_data = {
            "total": report_obj.total,
            "passed": report_obj.passed,
            "failed": report_obj.failed,
            "pass_rate": report_obj.pass_rate,
            "failed_episodes": [
                {"episode_id": r.episode_id, "errors": r.errors, "warnings": r.warnings}
                for r in report_obj.failed_episodes()
            ],
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"\nReport saved to [bold]{report}[/bold]")

    if report_obj.failed > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
