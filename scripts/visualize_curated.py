"""
CLI: Visualize a curated RLDS dataset episode frame by frame.

Produces one PNG per step.  Each PNG has a white background and shows:
  - Header:   "Step 0 / 27  —  <task instruction>"
  - Images:   primary camera (cam0) and wrist camera (cam1) side by side
  - Reasoning: task / subtask / move reasoning text
  - Metadata: alignment confidence + 7-DoF action values

Layout mirrors the Bridge v2 dual-camera viewer.

Usage
-----
# Random episode, full dataset:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0

# Specific episode:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/reasoning_only/1.0.0 \
    --episode-index 5

# Custom output dir:
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/full/1.0.0 \
    --save-dir outputs/viz/curated
"""

from __future__ import annotations

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
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Visualize a curated RLDS episode frame by frame.")
console = Console()


# ---------------------------------------------------------------------------
# TFDS loading
# ---------------------------------------------------------------------------


def _load_episode(dataset_path: Path, episode_index: int):
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
            return ep
    raise IndexError(
        f"Episode index {episode_index} is out of range for this dataset."
    )


def _t(val) -> str:
    """Decode a TF bytes tensor to a Python str."""
    v = val.numpy() if hasattr(val, "numpy") else val
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v) if v else ""


def _arr(val) -> np.ndarray:
    v = val.numpy() if hasattr(val, "numpy") else val
    return np.asarray(v)


def _parse_steps(episode) -> list[dict]:
    steps = []
    for step in episode["steps"]:
        obs = step["observation"]
        steps.append({
            "image_0":            _arr(obs["image_0"]).astype(np.uint8),
            "image_1":            _arr(obs["image_1"]).astype(np.uint8),
            "language_instruction": _t(obs["language_instruction"]),
            "task_reasoning":     _t(obs["task_reasoning"]),
            "subtask_reasoning":  _t(obs["subtask_reasoning"]),
            "move_reasoning":     _t(obs["move_reasoning"]),
            "action":             _arr(step["action"]).astype(np.float32),
            "is_first":           bool(_arr(step["is_first"])),
            "is_last":            bool(_arr(step["is_last"])),
            "alignment_confidence": float(_arr(step["alignment_confidence"])),
        })
    return steps


# ---------------------------------------------------------------------------
# Per-step figure
# ---------------------------------------------------------------------------

# Confidence colour (used for the confidence badge only)
def _conf_color(c: float) -> str:
    if c >= 0.95:
        return "#2ecc71"   # green  — direct annotation
    if c >= 0.4:
        return "#e67e22"   # orange — propagated
    return "#e74c3c"       # red    — no match


def _wrap(text: str, width: int = 55) -> str:
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


def _save_step_figure(
    step: dict,
    step_idx: int,
    total_steps: int,
    save_path: Path,
) -> None:
    """
    Save one PNG for a single step.

    Layout
    ------
    ┌──────────────────────────────────────────────────────────┐
    │  Step 3 / 27  —  pick up the orange from the table       │  ← suptitle
    ├──────────────────┬───────────────────┬───────────────────┤
    │                  │                   │  task_reasoning   │
    │    cam 0         │    cam 1          │  subtask_reas.    │
    │  (primary)       │  (wrist)          │  move_reasoning   │
    │                  │                   │  confidence       │
    │                  │                   │  action           │
    └──────────────────┴───────────────────┴───────────────────┘
    """
    fig = plt.figure(figsize=(14, 5.5), facecolor="white")

    task = step["language_instruction"] or "—"
    flags = []
    if step["is_first"]:
        flags.append("START")
    if step["is_last"]:
        flags.append("END")
    flag_str = f"  [{' · '.join(flags)}]" if flags else ""

    fig.suptitle(
        f"Step {step_idx} / {total_steps - 1}{flag_str}   —   {task}",
        fontsize=11, fontweight="bold", color="black", y=0.98,
    )

    # 3-column grid: cam0 | cam1 | text
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.02, right=0.98,
        top=0.88, bottom=0.04,
        wspace=0.06,
        width_ratios=[2, 2, 3],
    )

    # ── cam 0 ────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    img0 = step["image_0"]
    if img0.size > 0 and img0.max() > 0:
        ax0.imshow(img0)
    else:
        ax0.set_facecolor("#e0e0e0")
        ax0.text(0.5, 0.5, "No image", ha="center", va="center",
                 transform=ax0.transAxes, fontsize=9, color="#666")
    ax0.set_title("Camera 0  (primary)", fontsize=9, color="black", pad=4)
    ax0.axis("off")
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    # ── cam 1 ────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    img1 = step["image_1"]
    if img1.size > 0 and img1.max() > 0:
        ax1.imshow(img1)
    else:
        ax1.set_facecolor("#e0e0e0")
        ax1.text(0.5, 0.5, "No image", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=9, color="#666")
    ax1.set_title("Camera 1  (wrist)", fontsize=9, color="black", pad=4)
    ax1.axis("off")
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    # ── text panel ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("white")
    ax2.axis("off")

    conf  = step["alignment_confidence"]
    act   = step["action"]

    action_str = (
        f"Δxyz  [{act[0]:+.3f}, {act[1]:+.3f}, {act[2]:+.3f}]\n"
        f"Δrpy  [{act[3]:+.3f}, {act[4]:+.3f}, {act[5]:+.3f}]\n"
        f"grip   {act[6]:.3f}  ({'CLOSE' if act[6] > 0.5 else 'OPEN '})"
    )

    sections = [
        ("Task reasoning",    step["task_reasoning"],    "#1a1a1a"),
        ("Subtask reasoning", step["subtask_reasoning"], "#1a1a1a"),
        ("Move reasoning",    step["move_reasoning"],    "#1a1a1a"),
    ]

    y = 0.97
    LINE_H  = 0.055   # per text line
    LABEL_H = 0.065   # label row height
    GAP     = 0.025   # gap between sections

    for label, text, color in sections:
        wrapped = _wrap(text, width=52)
        n_lines = wrapped.count("\n") + 1

        # Section label
        ax2.text(
            0.02, y, label,
            transform=ax2.transAxes,
            fontsize=8, fontweight="bold", color="#555555",
            verticalalignment="top",
        )
        y -= LABEL_H

        # Section body
        ax2.text(
            0.02, y, wrapped,
            transform=ax2.transAxes,
            fontsize=8.5, color=color,
            verticalalignment="top",
            fontfamily="sans-serif",
        )
        y -= LINE_H * n_lines + GAP

        # Thin separator line
        ax2.axhline(y + GAP * 0.5, color="#dddddd", linewidth=0.7,
                    xmin=0.02, xmax=0.98)
        y -= GAP * 0.5

    # Confidence badge
    ax2.text(
        0.02, y, "Alignment confidence",
        transform=ax2.transAxes,
        fontsize=8, fontweight="bold", color="#555555",
        verticalalignment="top",
    )
    y -= LABEL_H
    conf_label = (
        "direct annotation (1.0)" if conf >= 0.95 else
        f"propagated ({conf:.2f})"    if conf >= 0.4  else
        f"no ECoT match ({conf:.2f})"
    )
    ax2.text(
        0.02, y, conf_label,
        transform=ax2.transAxes,
        fontsize=8.5, color=_conf_color(conf),
        verticalalignment="top", fontweight="bold",
    )
    y -= LINE_H + GAP
    ax2.axhline(y + GAP * 0.5, color="#dddddd", linewidth=0.7,
                xmin=0.02, xmax=0.98)
    y -= GAP * 0.5

    # Action
    ax2.text(
        0.02, y, "Action  (7-DoF)",
        transform=ax2.transAxes,
        fontsize=8, fontweight="bold", color="#555555",
        verticalalignment="top",
    )
    y -= LABEL_H
    ax2.text(
        0.02, y, action_str,
        transform=ax2.transAxes,
        fontsize=8.5, color="#1a1a1a",
        verticalalignment="top",
        fontfamily="monospace",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
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
        help="Episode index (-1 = random).",
    ),
    max_frames: int = typer.Option(
        None, "--max-frames",
        help="Save only the first N frames (default: all frames).",
    ),
    save_dir: Path = typer.Option(
        Path("outputs/viz/curated"), "--save-dir",
    ),
) -> None:

    if episode_index < 0:
        episode_index = random.randint(0, 5000)
        console.print(f"Random episode index: [bold]{episode_index}[/bold]")

    console.print(f"Loading episode {episode_index} from [bold]{path}[/bold] …")
    try:
        episode = _load_episode(path, episode_index)
    except IndexError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    steps = _parse_steps(episode)
    if not steps:
        console.print("[red]Episode has no steps.[/red]")
        raise typer.Exit(1)

    if max_frames is not None:
        steps = steps[:max_frames]

    total = len(steps)   # after possible truncation (for header display use original)
    total_display = len(_parse_steps(episode))  # original total for "X / N" header

    # Console summary
    n_with_r  = sum(1 for s in steps if s["task_reasoning"])
    conf_vals = [s["alignment_confidence"] for s in steps]

    t = Table(title=f"Episode {episode_index}", show_lines=True)
    t.add_column("Field",  style="bold")
    t.add_column("Value")
    t.add_row("Task",           steps[0]["language_instruction"])
    t.add_row("Total steps",    str(total_display))
    t.add_row("Saving frames",  str(total))
    t.add_row("With reasoning", f"{n_with_r}/{total}  ({100*n_with_r/max(total,1):.0f}%)")
    t.add_row("Avg confidence", f"{sum(conf_vals)/len(conf_vals):.3f}")
    console.print(t)

    # Save one PNG per step
    variant  = path.parent.name   # "full" or "reasoning_only"
    ep_dir   = save_dir / f"ep{episode_index:05d}_{variant}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\nSaving {total} frames to [bold]{ep_dir}[/bold] …")
    for i, step in enumerate(steps):
        out = ep_dir / f"step_{i:05d}.png"
        _save_step_figure(step, i, total_display, out)
        if (i + 1) % 10 == 0 or i == total - 1:
            console.print(f"  {i+1}/{total}")

    console.print(f"\n[bold green]Done.[/bold green]  {total} PNGs → {ep_dir}")


if __name__ == "__main__":
    app()
