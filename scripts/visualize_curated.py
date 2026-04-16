"""
CLI: Visualize a curated RLDS dataset episode (full or reasoning_only).

Shows both cameras side-by-side for each step with reasoning annotations,
action values, and alignment confidence — saved to disk (headless safe).

Usage
-----
# Random episode from the full dataset:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0

# Random episode from reasoning_only:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/reasoning_only/1.0.0

# Specific episode index:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0 \
    --episode-index 42

# Show more frames:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0 \
    --max-frames 24

# Custom output directory:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0 \
    --save-dir outputs/viz/curated
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Visualize a curated RLDS episode.")
console = Console()


# ---------------------------------------------------------------------------
# TFDS loading helpers
# ---------------------------------------------------------------------------


def _load_episode(dataset_path: Path, episode_index: int):
    """Load one episode from a TFDS dataset directory."""
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets required.  "
            "pip install 'vla-data-curator[bridge]'"
        ) from exc

    builder = tfds.builder_from_directory(str(dataset_path))
    ds = builder.as_dataset(split="train")

    for i, ep in enumerate(ds):
        if i == episode_index:
            return ep, builder.info

    raise IndexError(
        f"Episode index {episode_index} out of range "
        f"(dataset has fewer episodes)."
    )


def _tensor_to_numpy(val):
    if hasattr(val, "numpy"):
        return val.numpy()
    return val


def _decode_text(val) -> str:
    v = _tensor_to_numpy(val)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v) if v is not None else ""


def _decode_image(val) -> np.ndarray:
    arr = _tensor_to_numpy(val)
    return np.asarray(arr, dtype=np.uint8)


def _parse_steps(episode) -> list[dict]:
    """Convert a TFDS episode dict into a plain list of step dicts."""
    steps = []
    for step in episode["steps"]:
        obs = step["observation"]
        steps.append({
            "image_0":            _decode_image(obs["image_0"]),
            "image_1":            _decode_image(obs["image_1"]),
            "state":              _tensor_to_numpy(obs["state"]),
            "language_instruction": _decode_text(obs["language_instruction"]),
            "task_reasoning":     _decode_text(obs["task_reasoning"]),
            "subtask_reasoning":  _decode_text(obs["subtask_reasoning"]),
            "move_reasoning":     _decode_text(obs["move_reasoning"]),
            "action":             _tensor_to_numpy(step["action"]),
            "is_first":           bool(_tensor_to_numpy(step["is_first"])),
            "is_last":            bool(_tensor_to_numpy(step["is_last"])),
            "alignment_confidence": float(_tensor_to_numpy(step["alignment_confidence"])),
        })
    return steps


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 52) -> str:
    """Hard-wrap text to `width` characters per line."""
    if not text:
        return "—"
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            if line:
                lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def _confidence_color(conf: float) -> str:
    if conf >= 0.95:
        return "#2ecc71"   # green  — direct annotation
    if conf >= 0.5:
        return "#f39c12"   # orange — propagated
    return "#e74c3c"       # red    — no match


def _save_episode_grid(
    steps: list[dict],
    save_path: Path,
    max_frames: int = 16,
) -> None:
    """
    Save a grid figure: one column per step, two rows of images + reasoning text.

    Layout per column:
      row 0 — primary camera   (image_0)
      row 1 — wrist camera     (image_1)
      row 2 — reasoning text box
    """
    # Subsample frames evenly
    if len(steps) > max_frames:
        indices = [int(i * len(steps) / max_frames) for i in range(max_frames)]
        steps = [steps[i] for i in indices]

    n = len(steps)
    fig_w = max(n * 2.8, 10)
    fig_h = 14

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")

    # Title
    task = steps[0]["language_instruction"] if steps else ""
    n_with_r = sum(1 for s in steps if s["task_reasoning"])
    conf_avg = sum(s["alignment_confidence"] for s in steps) / max(len(steps), 1)
    fig.suptitle(
        f'Task: {task}\n'
        f'{n_with_r}/{len(steps)} steps with reasoning  |  '
        f'avg confidence: {conf_avg:.2f}',
        color="white", fontsize=10, y=0.98,
    )

    gs = gridspec.GridSpec(
        3, n,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
        top=0.92, bottom=0.02,
        left=0.01, right=0.99,
    )

    for col, step in enumerate(steps):
        conf  = step["alignment_confidence"]
        color = _confidence_color(conf)

        # ── Primary camera ──────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(step["image_0"])
        ax0.set_xticks([]); ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)
        if col == 0:
            ax0.set_ylabel("cam 0", color="white", fontsize=7)
        ax0.set_title(f"t={col}", color="white", fontsize=7, pad=2)

        # ── Wrist camera ────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(step["image_1"])
        ax1.set_xticks([]); ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)
        if col == 0:
            ax1.set_ylabel("cam 1", color="white", fontsize=7)

        # ── Reasoning text ──────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_facecolor("#0f0f23")
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(1.5)

        task_r    = step["task_reasoning"]
        subtask_r = step["subtask_reasoning"]
        move_r    = step["move_reasoning"]
        action    = step["action"]
        act_str   = "[" + ", ".join(f"{v:.2f}" for v in action) + "]"

        text_lines = []
        if task_r:
            text_lines.append(("task", task_r, "#3498db"))
        if subtask_r:
            text_lines.append(("subtask", subtask_r, "#9b59b6"))
        if move_r:
            text_lines.append(("move", move_r, "#1abc9c"))
        text_lines.append(("action", act_str, "#95a5a6"))
        text_lines.append(("conf", f"{conf:.2f}", color))

        y = 0.97
        for label, text, clr in text_lines:
            wrapped = _wrap(text, width=30)
            n_lines = wrapped.count("\n") + 1
            ax2.text(
                0.03, y,
                f"[{label}]\n{wrapped}",
                transform=ax2.transAxes,
                color=clr, fontsize=5.5,
                verticalalignment="top",
                fontfamily="monospace",
            )
            y -= 0.13 * (n_lines + 1)
            if y < 0.02:
                break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_reasoning_timeline(steps: list[dict], save_path: Path) -> None:
    """
    Save a vertical timeline figure: one row per step, showing confidence
    and reasoning presence as a coloured strip.
    """
    n = len(steps)
    fig, ax = plt.subplots(figsize=(12, max(n * 0.35, 4)), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for i, step in enumerate(steps):
        conf  = step["alignment_confidence"]
        color = _confidence_color(conf)
        has_r = bool(step["task_reasoning"])

        # Confidence bar
        ax.barh(i, conf, color=color, alpha=0.8, height=0.7)

        # Label
        task_r = step["task_reasoning"][:60] + "…" if len(step["task_reasoning"]) > 60 else step["task_reasoning"]
        label  = f"t={i:3d}  [{conf:.2f}]  {task_r or '(no reasoning)'}"
        ax.text(0.01, i, label, va="center", color="white", fontsize=6.5,
                fontfamily="monospace")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Alignment confidence", color="white")
    ax.set_title(
        f"Reasoning timeline — {n} steps\n"
        f"task: {steps[0]['language_instruction'] if steps else ''}",
        color="white",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    path: Path = typer.Option(
        ..., "--path", "-p",
        help="Path to a TFDS dataset version dir "
             "(e.g. outputs/curated/vla_curated_dataset/full/1.0.0).",
    ),
    episode_index: int = typer.Option(
        -1, "--episode-index", "-e",
        help="Episode index to visualize (-1 = random).",
    ),
    max_frames: int = typer.Option(
        16, "--max-frames",
        help="Max frames to show in the grid (evenly subsampled).",
    ),
    save_dir: Path = typer.Option(
        Path("outputs/viz/curated"), "--save-dir",
    ),
) -> None:

    # ── Pick episode index ────────────────────────────────────────────────────
    if episode_index < 0:
        episode_index = random.randint(0, 5000)
        console.print(f"Random episode index: [bold]{episode_index}[/bold]")

    # ── Load ─────────────────────────────────────────────────────────────────
    console.print(f"Loading episode {episode_index} from [bold]{path}[/bold]…")
    try:
        episode, info = _load_episode(path, episode_index)
    except IndexError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    steps = _parse_steps(episode)
    if not steps:
        console.print("[red]Episode has no steps.[/red]")
        raise typer.Exit(1)

    # ── Console summary ───────────────────────────────────────────────────────
    n_with_r  = sum(1 for s in steps if s["task_reasoning"])
    conf_vals = [s["alignment_confidence"] for s in steps]

    t = Table(title=f"Episode {episode_index}", show_lines=True)
    t.add_column("Field", style="bold")
    t.add_column("Value")
    t.add_row("Task",             steps[0]["language_instruction"])
    t.add_row("Steps",            str(len(steps)))
    t.add_row("With reasoning",   f"{n_with_r}/{len(steps)}  ({100*n_with_r/len(steps):.0f}%)")
    t.add_row("Avg confidence",   f"{sum(conf_vals)/len(conf_vals):.3f}")
    t.add_row("Min confidence",   f"{min(conf_vals):.3f}")
    console.print(t)

    # Print first annotated step's reasoning
    for s in steps:
        if s["task_reasoning"]:
            console.print(Panel(
                f"[bold blue]task:[/bold blue]    {s['task_reasoning']}\n"
                f"[bold magenta]subtask:[/bold magenta] {s['subtask_reasoning']}\n"
                f"[bold cyan]move:[/bold cyan]    {s['move_reasoning']}",
                title="First annotated step reasoning",
                expand=False,
            ))
            break

    # ── Save figures ──────────────────────────────────────────────────────────
    variant = path.parent.name   # "full" or "reasoning_only"
    base_name = f"ep{episode_index:05d}_{variant}"

    grid_path     = save_dir / f"{base_name}_grid.png"
    timeline_path = save_dir / f"{base_name}_timeline.png"

    console.print(f"\nSaving frame grid   → [bold]{grid_path}[/bold]")
    _save_episode_grid(steps, grid_path, max_frames=max_frames)

    console.print(f"Saving timeline     → [bold]{timeline_path}[/bold]")
    _save_reasoning_timeline(steps, timeline_path)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
