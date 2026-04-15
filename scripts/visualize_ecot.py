"""
CLI: Inspect the Embodied-CoT JSON file.

Prints keys, value structure, and sample entries — no figures, no parsing.

Usage
-----
python scripts/visualize_ecot.py --local-path /datasets/embodied_features_bridge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Inspect the Embodied-CoT JSON file.")
console = Console()


@app.command()
def main(
    local_path: Path = typer.Option(
        ..., "--local-path", "-p",
        help="Directory containing embodied_features_bridge.json",
    ),
    n: int = typer.Option(5, "--n", help="Number of sample entries to print."),
) -> None:

    # ── find the JSON file ────────────────────────────────────────────────────
    p = Path(local_path)
    if p.is_file():
        json_file = p
    else:
        candidates = sorted(
            f for f in p.rglob("*.json")
            if f.name not in ("dataset_info.json", "dataset_dict.json")
            and not f.name.startswith(".")
        )
        if not candidates:
            console.print(f"[red]No .json files found under {p}[/red]")
            console.print(f"Contents: {[x.name for x in p.iterdir()]}")
            raise typer.Exit(1)
        json_file = candidates[0]

    size_mb = json_file.stat().st_size / 1e6
    console.print(f"\n[bold]File:[/bold] {json_file}  ({size_mb:.0f} MB)")
    console.print("Loading… (may take a moment for large files)\n")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── top-level structure ───────────────────────────────────────────────────
    console.print(f"[bold]Top-level type:[/bold] {type(data).__name__}")

    if isinstance(data, dict):
        keys = list(data.keys())
        console.print(f"[bold]Total entries:[/bold] {len(keys):,}\n")

        # Key samples
        console.print(f"[bold]Sample keys ({min(n, len(keys))}):[/bold]")
        for k in keys[:n]:
            console.print(f"  {k}")

        # Value structure from first entry
        first_val = data[keys[0]]
        console.print(f"\n[bold]Value type:[/bold] {type(first_val).__name__}")

        if isinstance(first_val, dict):
            console.print(f"[bold]Value keys:[/bold] {list(first_val.keys())}\n")

            # Print sample entries as a table
            t = Table(title=f"First {min(n, len(keys))} entries", show_lines=True)
            t.add_column("Key (episode path)", overflow="fold", max_width=55)
            for field in first_val.keys():
                t.add_column(field, overflow="fold", max_width=30)

            for k in keys[:n]:
                v = data[k]
                if isinstance(v, dict):
                    t.add_row(k, *[str(v.get(field, ""))[:120] for field in first_val.keys()])
                else:
                    t.add_row(k, str(v)[:120])
            console.print(t)

        elif isinstance(first_val, list):
            console.print(f"[bold]List length (first entry):[/bold] {len(first_val)}")
            if first_val and isinstance(first_val[0], dict):
                console.print(f"[bold]Item keys:[/bold] {list(first_val[0].keys())}")
            console.print("\n[bold]First entry, first item:[/bold]")
            console.print(Panel(json.dumps(first_val[0], indent=2)[:800], expand=False))

        else:
            console.print(Panel(json.dumps(first_val, indent=2)[:800], expand=False))

    elif isinstance(data, list):
        console.print(f"[bold]Total entries:[/bold] {len(data):,}\n")
        first = data[0]
        console.print(f"[bold]Item type:[/bold] {type(first).__name__}")
        if isinstance(first, dict):
            console.print(f"[bold]Item keys:[/bold] {list(first.keys())}\n")
        console.print(f"[bold]First {min(n, len(data))} items:[/bold]")
        for item in data[:n]:
            console.print(Panel(json.dumps(item, indent=2)[:800], expand=False))

    else:
        console.print(Panel(str(data)[:1000], expand=False))

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
