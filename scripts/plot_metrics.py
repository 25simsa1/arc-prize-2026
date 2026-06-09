"""Writeup figures from a metrics run dir (or two, for comparison).

    .venv/bin/python scripts/plot_metrics.py results/baseline-c1
    .venv/bin/python scripts/plot_metrics.py --compare results/baseline-c1 results/r1-region

Outputs PNGs into the (first) run dir: coverage_<game>.png, phase_economics.png,
and compare_<game>.png in compare mode.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.metrics import PHASE_BUCKETS, read_events

PHASE_COLORS = {
    "exploration": "#4c72b0", "proposing": "#dd8452", "verifying": "#55a868",
    "planning": "#c44e52", "executing": "#8172b3", "env_stepping": "#937860",
}


def by_game_coverage(events: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["e"] == "coverage":
            out[e["game"].split("-")[0]].append(e)
    return out


def cumulative_actions(rows: list[dict]) -> list[int]:
    """action_index is per-play; accumulate across plays for one x-axis."""
    xs, offset, prev_play, prev_a = [], 0, None, 0
    for r in rows:
        if prev_play is not None and r["play"] != prev_play:
            offset += prev_a
        prev_play, prev_a = r["play"], r["a"]
        xs.append(offset + r["a"])
    return xs


def plot_coverage(rows: list[dict], game: str, label: str, ax) -> None:
    xs = cumulative_actions(rows)
    ax.plot(xs, [r["gp"] for r in rows], label=f"{label} grid coverage", lw=1.2)
    ax.plot(xs, [r["gx"] for r in rows], label=f"{label} grid exactness", lw=1.2, ls="--")
    ax.plot(xs, [r["ep"] for r in rows], label=f"{label} event coverage", lw=1.0, alpha=0.7)
    ax.set_xlabel("stored transitions observed (cumulative actions with new information)")
    ax.set_ylabel("fraction of stored transitions")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{game}: model coverage over experience")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def plot_run(run_dir: Path) -> None:
    events = read_events(run_dir)
    cov = by_game_coverage(events)
    for game, rows in cov.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_coverage(rows, game, run_dir.name, ax)
        out = run_dir / f"coverage_{game}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    phases = [e for e in events if e["e"] == "phase"]
    if phases:
        games = [e["game"] for e in phases]
        fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(games), 4.2))
        bottoms = [0.0] * len(games)
        for bucket in PHASE_BUCKETS:
            vals = [e.get(bucket, 0.0) for e in phases]
            ax.bar(games, vals, bottom=bottoms, label=bucket,
                   color=PHASE_COLORS[bucket])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, e in enumerate(phases):
            ax.text(i, bottoms[i] + 0.5, f"{e['actions']} acts\n{e['plays']} plays",
                    ha="center", fontsize=7)
        ax.set_ylabel("wall-clock seconds")
        ax.set_title(f"{run_dir.name}: where the time went, per game")
        ax.legend(fontsize=7)
        out = run_dir / "phase_economics.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def plot_compare(dir_a: Path, dir_b: Path) -> None:
    cov_a = by_game_coverage(read_events(dir_a))
    cov_b = by_game_coverage(read_events(dir_b))
    for game in sorted(set(cov_a) & set(cov_b)):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        plot_coverage(cov_a[game], game, dir_a.name, ax)
        plot_coverage(cov_b[game], game, dir_b.name, ax)
        ax.set_title(f"{game}: coverage, {dir_a.name} vs {dir_b.name}")
        out = dir_a / f"compare_{game}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = p.parse_args()
    if args.compare:
        plot_compare(Path(args.compare[0]), Path(args.compare[1]))
    elif args.run_dir:
        plot_run(Path(args.run_dir))
    else:
        p.error("need a run dir or --compare A B")


if __name__ == "__main__":
    main()
