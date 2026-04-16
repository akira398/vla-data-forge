"""
CLI: Visualize a curated RLDS dataset episode frame by frame.

Produces one PNG per step for each split (train, val, etc.).
Each PNG shows everything in the curated dataset:
  - Header:   "Step 0 / 27  —  <task instruction>"
  - Row 1:    4 camera views (image_0 … image_3)
  - Row 2:    Left panel  — 6 ECoT reasoning fields + caption
              Right panel — action, state, ECoT features, alignment confidence

Usage
-----
python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/matched/1.0.0

python scripts/visualize_curated.py \
    --path outputs/curated/vla_curated_dataset/matched/1.0.0 \
    --episode-index 5 --save-dir outputs/viz/curated
"""

from __future__ import annotations

import json
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


def _load_episode(dataset_path: Path, split: str, episode_index: int):
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets required.  "
            "pip install 'vla-data-curator[bridge]'"
        ) from exc

    builder = tfds.builder_from_directory(str(dataset_path))
    ds = builder.as_dataset(split=split)
    for i, ep in enumerate(ds):
        if i == episode_index:
            return ep
    raise IndexError(
        f"Episode index {episode_index} is out of range for split '{split}'."
    )


def _get_splits(dataset_path: Path) -> list[str]:
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        return ["train"]

    builder = tfds.builder_from_directory(str(dataset_path))
    return sorted(builder.info.splits.keys())


def _t(val) -> str:
    """Decode a TF bytes tensor to a Python str."""
    v = val.numpy() if hasattr(val, "numpy") else val
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v) if v else ""


def _arr(val) -> np.ndarray:
    v = val.numpy() if hasattr(val, "numpy") else val
    return np.asarray(v)


def _parse_episode_metadata(episode) -> dict:
    """Parse episode-level metadata."""
    meta = episode.get("episode_metadata", {})
    return {
        "file_path":   _t(meta.get("file_path", b"")),
        "episode_id":  int(_arr(meta.get("episode_id", -1))),
        "has_image_0": bool(_arr(meta.get("has_image_0", True))),
        "has_image_1": bool(_arr(meta.get("has_image_1", False))),
        "has_image_2": bool(_arr(meta.get("has_image_2", False))),
        "has_image_3": bool(_arr(meta.get("has_image_3", False))),
        "has_language": bool(_arr(meta.get("has_language", False))),
    }


def _parse_steps(episode) -> list[dict]:
    steps = []
    for step in episode["steps"]:
        obs = step["observation"]
        reasoning = step["reasoning"]
        ecot = step["ecot_features"]
        steps.append({
            # Images
            "image_0":     _arr(obs["image_0"]).astype(np.uint8),
            "image_1":     _arr(obs["image_1"]).astype(np.uint8),
            "image_2":     _arr(obs["image_2"]).astype(np.uint8),
            "image_3":     _arr(obs["image_3"]).astype(np.uint8),
            # State
            "state":       _arr(obs["state"]).astype(np.float32),
            # Action
            "action":      _arr(step["action"]).astype(np.float32),
            # Language
            "language_instruction": _t(step["language_instruction"]),
            "language_embedding":   _arr(step["language_embedding"]).astype(np.float32),
            # Flags
            "is_first":    bool(_arr(step["is_first"])),
            "is_last":     bool(_arr(step["is_last"])),
            "is_terminal": bool(_arr(step["is_terminal"])),
            "reward":      float(_arr(step["reward"])),
            "discount":    float(_arr(step["discount"])),
            # ECoT reasoning (6 fields)
            "task":            _t(reasoning["task"]),
            "plan":            _t(reasoning["plan"]),
            "subtask":         _t(reasoning["subtask"]),
            "subtask_reason":  _t(reasoning["subtask_reason"]),
            "move":            _t(reasoning["move"]),
            "move_reason":     _t(reasoning["move_reason"]),
            # ECoT features
            "caption":          _t(ecot["caption"]),
            "move_primitive":   _t(ecot["move_primitive"]),
            "gripper_position": _arr(ecot["gripper_position"]).astype(np.float32),
            "bboxes":           _t(ecot["bboxes"]),
            "state_3d":         _arr(ecot["state_3d"]).astype(np.float32),
            # Curation
            "alignment_confidence": float(_arr(step["alignment_confidence"])),
        })
    return steps


# ---------------------------------------------------------------------------
# Per-step figure
# ---------------------------------------------------------------------------

def _conf_color(c: float) -> str:
    if c >= 0.95:
        return "#2ecc71"   # green  — direct annotation
    if c >= 0.4:
        return "#e67e22"   # orange — propagated
    return "#e74c3c"       # red    — no match


def _wrap(text: str, width: int = 58) -> str:
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


def _has_content(img: np.ndarray) -> bool:
    """True if the image has any non-zero pixel."""
    return img.size > 0 and img.max() > 0


def _show_image(ax, img, title):
    """Render an image or 'No image' placeholder on an axes."""
    if _has_content(img):
        ax.imshow(img)
    else:
        ax.set_facecolor("#e0e0e0")
        ax.text(0.5, 0.5, "No image", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888")
    ax.set_title(title, fontsize=8, color="black", pad=3)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.6)


def _draw_text_section(ax, y, label, text, color="#1a1a1a",
                       label_size=7.5, body_size=8, width=58):
    """Draw a labelled text section and return the new y position."""
    LINE_H = 0.035
    LABEL_H = 0.042
    GAP = 0.012

    wrapped = _wrap(text, width=width)
    n_lines = wrapped.count("\n") + 1

    ax.text(0.02, y, label, transform=ax.transAxes,
            fontsize=label_size, fontweight="bold", color="#555555",
            verticalalignment="top")
    y -= LABEL_H

    ax.text(0.02, y, wrapped, transform=ax.transAxes,
            fontsize=body_size, color=color,
            verticalalignment="top", fontfamily="sans-serif")
    y -= LINE_H * n_lines + GAP

    ax.axhline(y + GAP * 0.4, color="#eeeeee", linewidth=0.5,
               xmin=0.02, xmax=0.98)
    y -= GAP * 0.4

    return y


def _save_step_figure(
    step: dict,
    step_idx: int,
    total_steps: int,
    save_path: Path,
    split: str = "",
    ep_meta: dict | None = None,
) -> None:
    fig = plt.figure(figsize=(18, 14), facecolor="white")

    task = step["language_instruction"] or "—"
    flags = []
    if step["is_first"]:
        flags.append("START")
    if step["is_last"]:
        flags.append("END")
    if step["is_terminal"]:
        flags.append("TERMINAL")
    flag_str = f"  [{' · '.join(flags)}]" if flags else ""
    split_str = f"  ({split})" if split else ""

    fig.suptitle(
        f"Step {step_idx} / {total_steps - 1}{flag_str}{split_str}   —   {task}",
        fontsize=11, fontweight="bold", color="black", y=0.99,
    )

    # =====================================================================
    # Layout: 2 rows
    #   Row 0: 4 camera images
    #   Row 1: left = reasoning text,  right = action/state/features/meta
    # =====================================================================
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.02, right=0.98,
        top=0.95, bottom=0.02,
        hspace=0.08,
        height_ratios=[1, 1.8],
    )

    # ── Row 0: 4 cameras ────────────────────────────────────────────────
    gs_cams = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0], wspace=0.04,
    )

    cam_labels = ["Camera 0 (primary)", "Camera 1 (wrist)",
                  "Camera 2", "Camera 3"]
    for ci, cam_key in enumerate(["image_0", "image_1", "image_2", "image_3"]):
        ax = fig.add_subplot(gs_cams[0, ci])
        _show_image(ax, step[cam_key], cam_labels[ci])

    # ── Row 1: two columns — reasoning | data ───────────────────────────
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.04,
        width_ratios=[1.1, 1],
    )

    # ── Left: reasoning + caption ────────────────────────────────────────
    ax_r = fig.add_subplot(gs_bottom[0, 0])
    ax_r.set_facecolor("white")
    ax_r.axis("off")

    y = 0.98
    reasoning_sections = [
        ("Caption",        step["caption"]),
        ("Task",           step["task"]),
        ("Plan",           step["plan"]),
        ("Subtask",        step["subtask"]),
        ("Subtask reason", step["subtask_reason"]),
        ("Move",           step["move"]),
        ("Move reason",    step["move_reason"]),
    ]
    for label, text in reasoning_sections:
        y = _draw_text_section(ax_r, y, label, text, width=62)

    # ── Right: action, state, ECoT features, alignment, metadata ──────���─
    ax_d = fig.add_subplot(gs_bottom[0, 1])
    ax_d.set_facecolor("white")
    ax_d.axis("off")

    y = 0.98
    act = step["action"]
    state = step["state"]
    conf = step["alignment_confidence"]
    grip_pos = step["gripper_position"]
    s3d = step["state_3d"]

    # Action
    action_str = (
        f"dxyz  [{act[0]:+.4f}, {act[1]:+.4f}, {act[2]:+.4f}]\n"
        f"drpy  [{act[3]:+.4f}, {act[4]:+.4f}, {act[5]:+.4f}]\n"
        f"grip   {act[6]:.4f}  ({'OPEN ' if act[6] > 0.5 else 'CLOSE'})"
    )
    y = _draw_text_section(ax_d, y, "Action (7-DoF)", action_str, body_size=7.5)

    # State
    state_str = (
        f"[{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f},\n"
        f" {state[3]:+.3f}, {state[4]:+.3f}, {state[5]:+.3f}, {state[6]:+.3f}]"
    )
    y = _draw_text_section(ax_d, y, "State (7-DoF)", state_str, body_size=7.5)

    # Reward / Discount
    rd_str = f"reward={step['reward']:.2f}   discount={step['discount']:.2f}"
    y = _draw_text_section(ax_d, y, "Reward / Discount", rd_str, body_size=7.5)

    # Alignment confidence
    conf_label = (
        "direct annotation (1.0)" if conf >= 0.95 else
        f"propagated ({conf:.2f})" if conf >= 0.4 else
        f"no ECoT match ({conf:.2f})"
    )
    ax_d.text(0.02, y, "Alignment confidence", transform=ax_d.transAxes,
              fontsize=7.5, fontweight="bold", color="#555555",
              verticalalignment="top")
    y -= 0.042
    ax_d.text(0.02, y, conf_label, transform=ax_d.transAxes,
              fontsize=8, color=_conf_color(conf),
              verticalalignment="top", fontweight="bold")
    y -= 0.035 + 0.012
    ax_d.axhline(y + 0.005, color="#eeeeee", linewidth=0.5,
                 xmin=0.02, xmax=0.98)
    y -= 0.005

    # Move primitive
    y = _draw_text_section(
        ax_d, y, "Move primitive",
        step["move_primitive"] or "—", body_size=7.5,
    )

    # ECoT state_3d
    s3d_str = f"[{s3d[0]:.4f}, {s3d[1]:.4f}, {s3d[2]:.4f}]"
    y = _draw_text_section(ax_d, y, "ECoT state_3d", s3d_str, body_size=7.5)

    # Gripper position
    gp_str = f"[{grip_pos[0]:.1f}, {grip_pos[1]:.1f}]"
    y = _draw_text_section(ax_d, y, "Gripper position (px)", gp_str, body_size=7.5)

    # Bboxes
    bbox_text = step["bboxes"] if step["bboxes"] else "—"
    if len(bbox_text) > 120:
        bbox_text = bbox_text[:120] + "..."
    y = _draw_text_section(ax_d, y, "Bboxes", bbox_text, body_size=7)

    # Language embedding (just show norm + first 5 dims)
    emb = step["language_embedding"]
    emb_norm = float(np.linalg.norm(emb))
    if emb_norm > 0:
        emb_str = (
            f"norm={emb_norm:.3f}  "
            f"[{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, {emb[3]:.3f}, {emb[4]:.3f}, ...]"
        )
    else:
        emb_str = "zeros (no embedding)"
    y = _draw_text_section(ax_d, y, "Language embedding (512-d)", emb_str, body_size=7)

    # Episode metadata (if available)
    if ep_meta:
        meta_str = (
            f"file_path: {ep_meta['file_path'][-60:]}\n"
            f"episode_id: {ep_meta['episode_id']}   "
            f"has_img: [{int(ep_meta['has_image_0'])},{int(ep_meta['has_image_1'])},"
            f"{int(ep_meta['has_image_2'])},{int(ep_meta['has_image_3'])}]   "
            f"has_lang: {int(ep_meta['has_language'])}"
        )
        y = _draw_text_section(ax_d, y, "Episode metadata", meta_str, body_size=6.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130,
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualize one split
# ---------------------------------------------------------------------------


def _visualize_split(
    dataset_path: Path,
    split: str,
    episode_index: int,
    max_frames: int | None,
    save_dir: Path,
) -> None:
    console.print(f"\n[bold]Split: {split}[/bold]")

    try:
        episode = _load_episode(dataset_path, split, episode_index)
    except IndexError as e:
        console.print(f"  [yellow]{e}[/yellow]")
        return

    ep_meta = _parse_episode_metadata(episode)
    steps = _parse_steps(episode)
    if not steps:
        console.print(f"  [yellow]Episode has no steps in split '{split}'.[/yellow]")
        return

    total_display = len(steps)
    if max_frames is not None:
        steps = steps[:max_frames]
    total = len(steps)

    # Console summary
    n_with_r = sum(1 for s in steps if s["task"])
    conf_vals = [s["alignment_confidence"] for s in steps]

    t = Table(title=f"Episode {episode_index} ({split})", show_lines=True)
    t.add_column("Field", style="bold")
    t.add_column("Value")
    t.add_row("Task",           steps[0]["language_instruction"])
    t.add_row("Caption",        steps[0]["caption"] or "—")
    t.add_row("File path",      ep_meta["file_path"][-80:] if ep_meta["file_path"] else "—")
    t.add_row("Episode ID",     str(ep_meta["episode_id"]))
    t.add_row("Total steps",    str(total_display))
    t.add_row("Saving frames",  str(total))
    t.add_row("With reasoning", f"{n_with_r}/{total}  ({100*n_with_r/max(total,1):.0f}%)")
    t.add_row("Avg confidence", f"{sum(conf_vals)/len(conf_vals):.3f}")
    t.add_row("Has images",
              f"0={int(ep_meta['has_image_0'])} 1={int(ep_meta['has_image_1'])} "
              f"2={int(ep_meta['has_image_2'])} 3={int(ep_meta['has_image_3'])}")
    t.add_row("Has language",   str(ep_meta["has_language"]))
    console.print(t)

    # Save one PNG per step
    variant = dataset_path.parent.name  # "matched" or "reasoning_only"
    ep_dir = save_dir / split / f"ep{episode_index:05d}_{variant}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Saving {total} frames to [bold]{ep_dir}[/bold] …")
    for i, step in enumerate(steps):
        out = ep_dir / f"step_{i:05d}.png"
        _save_step_figure(step, i, total_display, out, split=split, ep_meta=ep_meta)
        if (i + 1) % 10 == 0 or i == total - 1:
            console.print(f"  {i+1}/{total}")

    console.print(f"[green]Done.[/green]  {total} PNGs → {ep_dir}")

    # Assemble video (1 fps)
    _make_video(ep_dir, total)


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------


def _make_video(ep_dir: Path, num_frames: int) -> None:
    """Stitch step_*.png files into an MP4 video at 1 fps."""
    import imageio.v3 as iio

    png_paths = sorted(ep_dir.glob("step_*.png"))
    if len(png_paths) < 2:
        return

    video_path = ep_dir / "episode.mp4"
    frames = [iio.imread(p) for p in png_paths]
    iio.imwrite(video_path, frames, fps=1, codec="libx264")
    console.print(f"  Video → [bold]{video_path}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    path: Path = typer.Option(
        ..., "--path", "-p",
        help="Path to a TFDS dataset version dir "
             "(e.g. outputs/curated/vla_curated_dataset/matched/1.0.0).",
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

    splits = _get_splits(path)
    console.print(f"Dataset splits: [bold]{', '.join(splits)}[/bold]")
    console.print(f"Episode index:  [bold]{episode_index}[/bold]")

    for split in splits:
        _visualize_split(path, split, episode_index, max_frames, save_dir)

    console.print(f"\n[bold green]All done.[/bold green]  Output → {save_dir}")


if __name__ == "__main__":
    app()
