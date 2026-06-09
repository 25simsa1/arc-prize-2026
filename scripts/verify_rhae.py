"""Unit-check our independent RHAE math against the shipped calculator.

Feeds identical synthetic per-level numbers to arc_agi's
EnvironmentScoreCalculator (authoritative) and harness.rhae (ours); any
disagreement means we misread the semantics. Covers: the 115 cap, exact
baseline, worse-than-baseline, uncompleted levels, partial-completion cap,
and full completion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_agi.scorecard import EnvironmentScoreCalculator

from harness import rhae

CASES = [
    # (name, baselines, level_actions, levels_completed)
    ("full win, mixed efficiency", [10, 20, 30], [10, 25, 15], 3),
    ("cap115: 2x better than human", [100, 50], [50, 25], 2),
    ("exactly at cap boundary", [115, 10], [100, 10], 2),
    ("partial: 2 of 3 levels", [10, 20, 30], [12, 18, 40], 2),
    ("nothing completed", [10, 20], [55, 0], 0),
    ("one-level game, slight win", [22], [20], 1),
]


def shipped_score(baselines, level_actions, levels_completed) -> float:
    calc = EnvironmentScoreCalculator()
    for i, b in enumerate(baselines):
        calc.add_level(
            level_index=i + 1,
            completed=i < levels_completed,
            actions_taken=level_actions[i],
            baseline_actions=b,
        )
    return calc.to_score(include_levels=True).score


def main() -> None:
    failures = 0
    for name, baselines, actions, completed in CASES:
        ours = rhae.game_score(baselines, actions, completed)
        theirs = shipped_score(baselines, actions, completed)
        ok = abs(ours - theirs) <= 1e-9
        failures += 0 if ok else 1
        print(f"{'OK ' if ok else 'FAIL'} {name:32s} ours={ours:10.4f} shipped={theirs:10.4f}")
    if failures:
        sys.exit(f"{failures} mismatch(es) — semantics misread, fix harness/rhae.py")
    print("All cases match the shipped calculator.")


if __name__ == "__main__":
    main()
