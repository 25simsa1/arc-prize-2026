"""WinSeeker: phase-1 policy. Objective is reaching WIN; model-building is a
side effect. Priority when the planner has no path (the agent tries the
planner FIRST and only consults this policy if planning fails):

  (ii)  actions the model is most uncertain about here = unseen
        (level, frame, action) keys in the store,
  (iii) untried is the same set under our exact-fit model — kept as one tier,
  (iv)  actions known to change the frame here, avoiding known GAME_OVER,
  (v)   anything available, avoiding known GAME_OVER.

Never emits an off-menu action: candidates are built FROM available_actions.
Click candidates come from visual salience (connected components), capped —
4096 raw coordinates is not an explorable space. GAME_OVER handling and the
no-progress bail-out live in the agent, which owns time budgets.
"""

import random
from collections import defaultdict
from typing import Optional

import numpy as np

from .store import EVENT_GAME_OVER, Transition


def salient_clicks(grid: np.ndarray, cap: int = 24) -> list[str]:
    """One representative click per 4-connected component, smallest
    components first (rare/small things are likelier to be interactive),
    skipping the largest color (background-ish). Deterministic."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, int, int, int]] = []  # (size, color, y, x)
    color_counts: dict[int, int] = defaultdict(int)
    for y in range(h):
        for x in range(w):
            color_counts[int(grid[y, x])] += 1
    bg = max(color_counts, key=lambda c: color_counts[c])
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(grid[y, x]) == bg:
                continue
            color = int(grid[y, x])
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            my = sum(c[0] for c in cells) // len(cells)
            mx = sum(c[1] for c in cells) // len(cells)
            # representative: centroid if inside component, else first cell
            ry, rx = (my, mx) if (my, mx) in set(cells) else cells[0]
            comps.append((len(cells), color, ry, rx))
    comps.sort()
    return [f"ACTION6:{x},{y}" for _, _, y, x in comps[:cap]]


class WinSeeker:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.step_count = 0

    def choose(
        self,
        ctx_actions: dict[str, Transition],
        grid: np.ndarray,
        available_simple: list[str],
        clicks_enabled: bool,
    ) -> tuple[str, str]:
        """Returns (action_key, source_tag).

        ctx_actions: actions already tried in THIS context, keyed by action,
        where context identity is the agent's (level, masked frame hash) —
        masking matters: with a ticking HUD in the hash, every state looks
        novel forever and unseen-first floods (the 26k-action explorations
        of Workstream B). Caller guarantees available_simple ⊆ the game's
        advertised available_actions."""
        self.step_count += 1
        candidates = list(available_simple)
        if clicks_enabled:
            candidates += salient_clicks(grid)

        unseen = [a for a in candidates if a not in ctx_actions]
        if unseen:
            # deterministic rotation so repeated visits try different actions
            return unseen[self.step_count % len(unseen)], "unseen"

        safe = [(a, t) for a, t in ctx_actions.items()
                if a in candidates and t.event != EVENT_GAME_OVER]
        changers = [a for a, t in safe if t.post_hash != t.pre_hash]
        if changers:
            return changers[self.step_count % len(changers)], "frame_changer"
        if safe:
            return safe[self.step_count % len(safe)][0], "safe_any"
        # everything known leads to GAME_OVER (or nothing available): random legal
        return self.rng.choice(candidates), "desperate"
