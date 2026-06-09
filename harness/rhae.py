"""RHAE recomputation, for verifying reported scores against the formula.

The authoritative implementation is arc_agi.scorecard (we never replace it);
this module independently recomputes scores from raw per-level numbers so a
harness bug or a misunderstanding of the semantics shows up as a mismatch.

Semantics (verified against arc_agi-0.9.8 scorecard.py and the official
methodology page, 2026-06-09):
  level_score   = min((baseline / actions)^2 * 100, 115.0)   [0 if not completed]
  game_score    = sum(w_i * level_score_i) / sum(w_i),  w_i = 1-indexed level
                  number over ALL baseline levels (uncompleted levels score 0
                  but keep their weight), then capped at
                  (sum of completed-level weights / sum of all weights) * 100.
"""

from typing import Optional, Sequence


def level_score(baseline: int, actions: int, completed: bool) -> float:
    if not completed or actions <= 0:
        return 0.0
    return min((baseline / actions) ** 2 * 100.0, 115.0)


def game_score(
    baselines: Sequence[int],
    level_actions: Sequence[int],
    levels_completed: int,
) -> float:
    total_w = 0
    total = 0.0
    completed_w = 0
    for i, baseline in enumerate(baselines):
        w = i + 1
        total_w += w
        completed = i < levels_completed
        acts = level_actions[i] if i < len(level_actions) else 0
        s = level_score(baseline, acts, completed)
        total += s * w
        if s > 0:
            completed_w += w
    if total_w == 0:
        return 0.0
    return min(total / total_w, completed_w / total_w * 100.0)


def verify_run(
    run: dict, baselines: Optional[Sequence[int]], tol: float = 1e-6
) -> tuple[bool, float, float]:
    """Recompute a single EnvironmentScore dict against its reported score.

    Returns (ok, reported, recomputed).
    """
    reported = float(run.get("score", 0.0))
    if not baselines:
        return reported == 0.0, reported, 0.0
    recomputed = game_score(
        baselines,
        run.get("level_actions") or [],
        int(run.get("levels_completed", 0)),
    )
    return abs(reported - recomputed) <= tol, reported, recomputed
