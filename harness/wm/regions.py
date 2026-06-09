"""Region factoring: segment the grid into STATIC / DYNAMIC / ALWAYS_CHANGING.

Evidence this layer exists (Workstream B store, cd82): the game ticks an
action-meter cell in row 63 on EVERY action — the meter front advances one
cell per action, sweeping the row — so no transition is ever frame-identity
and whole-frame templates fit nothing (grid coverage was a flat 0.0 over
31k actions). The meter is a REGION property, not a cell property: each
individual meter cell changes rarely (when the front passes), but the
region as a whole changes on ~every action regardless of action identity.

Classification runs in two stages, with membership grounded in the
sole-changer signal measured on cd82's store (48k transitions: 13,147
one-cell diffs, ALL of them in row 63; content diffs are always >=5 cells):
  1. membership: TIER-1 seeds = cells that are repeatedly the SOLE change of
     a transition AND action-exogenous — for ACTION6 the click must land
     elsewhere (chebyshev > 1). Exogeneity is what separates a HUD meter
     (changes regardless of where you click) from an interactive click-board
     (changes AT the clicked cell — masking those would gut click games).
     TIER-2 = cells repeatedly co-changing in exactly-2-cell diffs whose
     partner is a seed (catches HUD neighbors like a blinking indicator next
     to a ticker). Members cluster spatially (chebyshev <= 2 bridges gaps).
  2. region test: a cluster of such members is ALWAYS_CHANGING iff >=1
     member changes in at least `always_rate` of ALL transitions (majority
     test, 0.55 — cd82's meter skips no-op actions and measures 0.64, while
     genuinely interactive clusters built from exogenous sole-changers are
     rare), with enough evidence (min_transitions) and bounded size
     (max_frac — masking most of the board is over-masking by definition).

Over-masking guards (both enforced here, tested in test_wm_core):
  - unmaskable colors: any cell that ever held a color targeted by a
    LEVEL/WIN/GAME_OVER event rule is excluded — a win condition inside a
    noisy region must never be masked away;
  - confidence: no region is reported below min_transitions observations.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .store import Transition


@dataclass
class Region:
    region_id: int
    cells: list[tuple[int, int]]
    activity: float            # fraction of transitions with >=1 member changed
    transitions_observed: int  # confidence basis

    def as_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "cells": [list(c) for c in self.cells],
            "activity": round(self.activity, 4),
            "transitions_observed": self.transitions_observed,
        }


@dataclass
class RegionMap:
    shape: tuple[int, int]
    hud_regions: list[Region] = field(default_factory=list)
    transitions_observed: int = 0

    @property
    def hud_mask(self) -> Optional[np.ndarray]:
        """Bool mask, True = ALWAYS_CHANGING cell. None when nothing qualifies."""
        if not self.hud_regions:
            return None
        m = np.zeros(self.shape, dtype=bool)
        for r in self.hud_regions:
            for y, x in r.cells:
                m[y, x] = True
        return m

    def as_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "transitions_observed": self.transitions_observed,
            "hud_regions": [r.as_dict() for r in self.hud_regions],
        }


class RegionAnalyzer:
    def __init__(
        self,
        always_rate: float = 0.55,
        min_transitions: int = 30,
        max_frac: float = 0.10,
        seed_min_sole: int = 2,
        pair_min: int = 2,
        cluster_gap: int = 2,
    ) -> None:
        self.always_rate = always_rate
        self.min_transitions = min_transitions
        self.max_frac = max_frac
        self.seed_min_sole = seed_min_sole
        self.pair_min = pair_min
        self.cluster_gap = cluster_gap
        self._n = 0
        self._shape: Optional[tuple[int, int]] = None
        self._colors_seen: dict[tuple[int, int], set[int]] = {}
        self._changed_at: dict[tuple[int, int], list[int]] = {}
        self._sole: dict[tuple[int, int], int] = {}
        # partners in exactly-2-cell diffs: cell -> {partner: count}
        self._pair: dict[tuple[int, int], dict[tuple[int, int], int]] = {}

    def observe(self, t: Transition) -> None:
        if self._shape is None:
            self._shape = t.pre.shape
        self._n += 1
        diff = np.argwhere(t.pre != t.post)
        cells = [(int(y), int(x)) for y, x in diff]
        for cell in cells:
            self._colors_seen.setdefault(cell, set()).update(
                (int(t.pre[cell]), int(t.post[cell]))
            )
            self._changed_at.setdefault(cell, []).append(self._n - 1)
        if len(cells) == 1:
            cell = cells[0]
            xy = t.click_xy
            exogenous = xy is None or max(abs(xy[1] - cell[0]), abs(xy[0] - cell[1])) > 1
            if exogenous:
                self._sole[cell] = self._sole.get(cell, 0) + 1
        elif len(cells) == 2:
            a, b = cells
            self._pair.setdefault(a, {})[b] = self._pair.setdefault(a, {}).get(b, 0) + 1
            self._pair.setdefault(b, {})[a] = self._pair.setdefault(b, {}).get(a, 0) + 1

    def analyze(self, unmaskable_colors: Iterable[int] = ()) -> RegionMap:
        unmask = set(unmaskable_colors)
        if self._shape is None or self._n < self.min_transitions:
            return RegionMap(self._shape or (0, 0), [], self._n)

        seeds = {c for c, n in self._sole.items() if n >= self.seed_min_sole}
        tier2 = {
            c
            for c, partners in self._pair.items()
            if c not in seeds
            and any(p in seeds and n >= self.pair_min for p, n in partners.items())
        }
        members = []
        for cell in seeds | tier2:
            if self._colors_seen.get(cell, set()) & unmask:
                continue  # guard: never mask cells that held goal/hazard colors
            members.append(cell)

        clusters = self._cluster(members)
        regions: list[Region] = []
        max_cells = int(self.max_frac * self._shape[0] * self._shape[1])
        for i, cluster in enumerate(clusters):
            if len(cluster) > max_cells:
                continue  # guard: masking a large share of the board is over-masking
            changed_transitions: set[int] = set()
            for cell in cluster:
                changed_transitions.update(self._changed_at.get(cell, []))
            activity = len(changed_transitions) / self._n
            if activity >= self.always_rate:
                regions.append(Region(i, sorted(cluster), activity, self._n))
        return RegionMap(self._shape, regions, self._n)

    def _cluster(self, cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Union cells within chebyshev distance <= cluster_gap."""
        unvisited = set(cells)
        out = []
        g = self.cluster_gap
        while unvisited:
            seed = unvisited.pop()
            comp = [seed]
            frontier = [seed]
            while frontier:
                cy, cx = frontier.pop()
                near = [
                    c for c in unvisited
                    if abs(c[0] - cy) <= g and abs(c[1] - cx) <= g
                ]
                for c in near:
                    unvisited.remove(c)
                    comp.append(c)
                    frontier.append(c)
            out.append(comp)
        return out
