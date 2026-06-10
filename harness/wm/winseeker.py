"""WinSeeker v2: phase-1 exploration policy. Objective is reaching WIN with
model-building as a side effect; the agent consults the planner FIRST and
only calls this when planning fails.

Tier order (each mechanism cites the evidence that motivated it):

  (i)   unseen candidate actions at the current context;
  (ii)  INERT-START escalation — adaptive ACTION6 coverage. Motivated by
        ft09/lp85 in the 25-game sweep: 47k actions, 24 unique transitions,
        because the cap-24 salience generator never found the live control
        and every candidate was a frame-no-op. When NO candidate has ever
        changed the frame, sweep a coarse 8x8 click lattice over the board,
        then refine (3x3, step 3) around any click that produced change.
        "No live controls" may only be concluded AFTER the lattice sweep.
  (iii) evidence-seeking ordering of known frame-changers. Motivated by the
        starvation census (19/25 public games yielded zero LEVEL/WIN
        observations in 240s): (a) frontier/novelty first — prefer actions
        leading to states with untried actions, since level transitions
        live at the frontier of the reachable set; (b) meter-movers next —
        transitions that changed analyzer-flagged status cells track
        progress/lives counters, and a meter that drains to GAME_OVER is
        also event evidence (death teaches the win condition's complement);
        (c) death-path penalty — after a GAME_OVER, de-prioritize exact
        re-traversal of the dying trajectory.
  (iv)  safe known actions, then anything legal.

Never emits an off-menu action: candidates derive from available_actions.
The agent owns time budgets and the exploration ledger; this class exposes
the counters the ledger needs (lattice state, frame-change flag, novelty
staleness).
"""

import random
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from .store import EVENT_GAME_OVER, Transition, masked_hash


def salient_clicks(grid: np.ndarray, cap: int = 24) -> list[str]:
    """One representative click per 4-connected component, smallest
    components first, skipping the most common color. Deterministic."""
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, int, int, int]] = []
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
            ry, rx = (my, mx) if (my, mx) in set(cells) else cells[0]
            comps.append((len(cells), color, ry, rx))
    comps.sort()
    return [f"ACTION6:{x},{y}" for _, _, y, x in comps[:cap]]


def lattice_clicks(shape: tuple[int, int] = (64, 64), step: int = 8) -> list[str]:
    """Coarse full-board click coverage: cell centers of a step x step grid."""
    half = step // 2
    return [f"ACTION6:{x},{y}"
            for y in range(half, shape[0], step)
            for x in range(half, shape[1], step)]


def refine_clicks(x: int, y: int, radius: int = 3, step: int = 3) -> list[str]:
    out = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 64 and 0 <= ny < 64 and (dx, dy) != (0, 0):
                out.append(f"ACTION6:{nx},{ny}")
    return out


class WinSeeker:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.step_count = 0
        # inert-start escalation state
        self.frame_change_seen = False
        self._lattice: Optional[list[str]] = None
        # ordered + set-backed: the list-containment variant went quadratic
        # on click-heavy games (re86: 10 actions/s instead of ~300)
        self._refine: list[str] = []
        self._refine_set: set[str] = set()
        self.lattice_tried = 0
        self.lattice_total = 0
        # evidence-seeking state
        self.steps_since_new_transition = 0
        self._death_path: set[tuple[str, str]] = set()   # (ctx_key, action)
        self._recent_path: list[tuple[str, str]] = []    # rolling trajectory
        self._meter_mover: dict[str, int] = defaultdict(int)  # action -> count

    # ------------------------------------------------------------ feedback
    def observe(self, o: Transition, ctx_key: str,
                hud_mask: Optional[np.ndarray], is_new: bool) -> None:
        """Agent calls this for every observed transition."""
        self.steps_since_new_transition = 0 if is_new else self.steps_since_new_transition + 1
        changed = o.pre_hash != o.post_hash
        if changed:
            self.frame_change_seen = True
            if o.base_action == "ACTION6" and o.click_xy is not None and \
                    len(self._refine) < 512:
                x, y = o.click_xy
                for a in refine_clicks(x, y):
                    if a not in self._refine_set:
                        self._refine_set.add(a)
                        self._refine.append(a)
            if hud_mask is not None and bool(((o.pre != o.post) & hud_mask).any()):
                self._meter_mover[o.action_key] += 1
        self._recent_path.append((ctx_key, o.action_key))
        if len(self._recent_path) > 400:
            self._recent_path = self._recent_path[-400:]

    def on_game_over(self) -> None:
        """Penalize exact re-traversal of the trajectory that just died."""
        self._death_path = set(self._recent_path[-60:])
        self._recent_path = []

    # ------------------------------------------------------------- choosing
    def choose(
        self,
        ctx_actions: dict[str, Transition],
        grid: np.ndarray,
        available_simple: list[str],
        clicks_enabled: bool,
        ctx_key: str = "",
        untried_at: Optional[Callable[[Transition], int]] = None,
        hud_mask: Optional[np.ndarray] = None,
    ) -> tuple[str, str]:
        """Returns (action_key, source_tag). untried_at(transition) -> count
        of untried candidate actions at its successor state (frontier size)."""
        self.step_count += 1
        candidates = list(available_simple)
        if clicks_enabled:
            candidates += salient_clicks(grid)
            # escalation refinement targets are first-class candidates
            candidates += [a for a in self._refine[:24] if a not in candidates]

        unseen = [a for a in candidates if a not in ctx_actions]
        if unseen:
            return unseen[self.step_count % len(unseen)], "unseen"

        # (ii) INERT-START escalation: nothing here has ever changed the
        # frame and the normal candidates are exhausted -> lattice sweep.
        if clicks_enabled and not self.frame_change_seen:
            if self._lattice is None:
                self._lattice = lattice_clicks()
                self.lattice_total = len(self._lattice)
            while self._lattice:
                a = self._lattice.pop(0)
                self.lattice_tried += 1
                if a not in ctx_actions:
                    return a, "lattice"

        known = [(a, t) for a, t in ctx_actions.items() if a in candidates]
        safe = [(a, t) for a, t in known if t.event != EVENT_GAME_OVER]
        changers = [(a, t) for a, t in safe if t.post_hash != t.pre_hash]

        if changers:
            # (iii) evidence-seeking order: frontier novelty, meter movement,
            # death-path avoidance.
            def score(item):
                a, t = item
                frontier = untried_at(t) if untried_at else 0
                meter = self._meter_mover.get(a, 0)
                died_here = (ctx_key, a) in self._death_path
                return (-frontier, -min(meter, 3), died_here, a)

            ranked = sorted(changers, key=score)
            # rotate within the top tier to avoid hammering one action
            top = [it for it in ranked if score(it)[:3] == score(ranked[0])[:3]]
            return top[self.step_count % len(top)][0], "frontier"

        if safe:
            return safe[self.step_count % len(safe)][0], "safe_any"
        return self.rng.choice(candidates), "desperate"
