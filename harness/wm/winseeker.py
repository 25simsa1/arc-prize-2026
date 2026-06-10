"""WinSeeker v3: phase-1 exploration policy. Objective is reaching WIN (and,
under the eval-realistic 5x per-level cutoff, gathering LEVEL/WIN/GAME_OVER
and conflict evidence INSIDE small per-level windows) with model-building as
a side effect. The agent consults the planner FIRST and only calls this when
planning fails.

Tier order (each mechanism cites the evidence/paper that motivated it):

  (i)   UNSEEN candidates in SALIENCE-TIER order (Rudakov 2512.24156). The
        candidate set is salience-stratified over segmented components and is
        NOT capped: exhaust tier 1 (small/interactive-looking segments),
        fall through to larger tiers, then a productive-click refine tier,
        then a coarse-lattice FLOOR — so every segment (and every coarse-grid
        cell) is probed at least once before "no live controls" can be
        concluded. Replaces the old cap-24 salience generator that left
        ft09/lp85 inert for 47k actions.
  (ii)  EVIDENCE-SEEKING ordering of known frame-changers (the 88%
        eval-realistic starvation census): (a) segment-granularity novelty
        first (#Exploration 1611.04717) — prefer successors whose component
        kinds are rarely visited, so cosmetic HUD diffs don't drown
        event-bearing sub-changes; (b) frontier size; (c) meter-movers in
        EITHER direction (a meter draining to GAME_OVER is evidence too —
        death teaches the win condition's complement); (d) death-path
        penalty — de-prioritize exact re-traversal of a trajectory that died.
  (iii) safe known actions, then anything legal.

Never emits an off-menu action: candidates derive from available_actions.
The agent owns time/window budgets, the Go-Explore archive, and the
exploration ledger; this class exposes the counters the ledger needs.
"""

import random
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from .explore import (TIER_NAMES, _TIER_LATTICE, _TIER_REFINE, _TIER_SIMPLE,
                      tiered_click_candidates)
from .store import EVENT_GAME_OVER, Transition, masked_hash


def salient_clicks(grid: np.ndarray, cap: int = 24) -> list[str]:
    """One representative click per component, salience-ascending, capped.
    Kept for the PLANNER's click-target generation (a bounded candidate set is
    correct there); WinSeeker itself uses the uncapped tiered generator."""
    out = [ak for _, ak in tiered_click_candidates(grid)
           if ak.startswith("ACTION6:")]
    return out[:cap]


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
        # coverage / inert-start ledger signals
        self.frame_change_seen = False
        self.max_tier_reached = -1
        self.lattice_tried = 0
        self.lattice_total = 0
        self._lattice_seen: set[str] = set()
        # productive-click refine targets (tier between segments and lattice)
        self._refine: list[str] = []
        self._refine_set: set[str] = set()
        # evidence-seeking state
        self.steps_since_new_transition = 0
        self._death_path: set[tuple[str, str]] = set()
        self._recent_path: list[tuple[str, str]] = []
        self._meter_mover: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------ feedback
    def observe(self, o: Transition, ctx_key: str,
                hud_mask: Optional[np.ndarray], is_new: bool) -> None:
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
        self._death_path = set(self._recent_path[-60:])
        self._recent_path = []

    # ------------------------------------------------------------- choosing
    def _candidates(self, grid, available_simple, clicks_enabled,
                    hud_mask) -> list[tuple[int, str]]:
        """(tier, action_key) candidates in salience order. Tier 0 simple
        actions, then segment clicks by salience, then refine, then lattice."""
        cands: list[tuple[int, str]] = [(_TIER_SIMPLE, a) for a in available_simple]
        if clicks_enabled:
            cands += tiered_click_candidates(grid, hud_mask)
            cands += [(_TIER_REFINE, a) for a in self._refine[:64]]
        return cands

    def choose(
        self,
        ctx_actions: dict[str, Transition],
        grid: np.ndarray,
        available_simple: list[str],
        clicks_enabled: bool,
        ctx_key: str = "",
        untried_at: Optional[Callable[[Transition], int]] = None,
        hud_mask: Optional[np.ndarray] = None,
        seg_visits: Optional[dict] = None,
        succ_novelty: Optional[Callable[[Transition], int]] = None,
    ) -> tuple[str, str]:
        """Returns (action_key, source_tag)."""
        self.step_count += 1
        cands = self._candidates(grid, available_simple, clicks_enabled, hud_mask)

        # count lattice-floor coverage for the ledger's no_live_controls test
        for tier, ak in cands:
            if tier == _TIER_LATTICE and ak not in self._lattice_seen:
                self._lattice_seen.add(ak)
                self.lattice_total += 1

        # (i) UNSEEN at the LOWEST occupied tier, ROTATED within that tier.
        # Tier priority (small/interactive segments first) is preserved, but
        # within a tier we rotate by step_count instead of taking the
        # coordinate-string-first candidate: a strict string order buries
        # high-coordinate cells past the end of a tiny capped window (r11l's
        # winning click ACTION6:42,22 is one of ~100 single-cell dots and
        # sorts late as a string), which cost the capped L1 wins. Rotation
        # restores the old explorer's hit rate while keeping tier coverage.
        by_tier: dict[int, list[str]] = {}
        for tier, ak in cands:
            if ak not in ctx_actions:
                by_tier.setdefault(tier, []).append(ak)
        if by_tier:
            tier = min(by_tier)
            group = by_tier[tier]
            ak = group[self.step_count % len(group)]
            self.max_tier_reached = max(self.max_tier_reached, tier)
            if tier == _TIER_LATTICE:
                self.lattice_tried += 1
                return ak, "lattice"
            return ak, ("refine" if tier == _TIER_REFINE else "unseen")

        # (ii) EVIDENCE-SEEKING among known frame-changers
        known = [(a, t) for a, t in ctx_actions.items()
                 if any(a == ak for _, ak in cands)]
        safe = [(a, t) for a, t in known if t.event != EVENT_GAME_OVER]
        changers = [(a, t) for a, t in safe if t.post_hash != t.pre_hash]
        if changers:
            def score(item):
                a, t = item
                novelty = succ_novelty(t) if succ_novelty else 0
                frontier = untried_at(t) if untried_at else 0
                meter = self._meter_mover.get(a, 0)
                died_here = (ctx_key, a) in self._death_path
                return (-novelty, -frontier, -min(meter, 3), died_here, a)
            ranked = sorted(changers, key=score)
            tie = score(ranked[0])[:4]
            top = [it for it in ranked if score(it)[:4] == tie]
            return top[self.step_count % len(top)][0], "frontier"

        if safe:
            return safe[self.step_count % len(safe)][0], "safe_any"
        all_keys = [ak for _, ak in cands]
        return (self.rng.choice(all_keys) if all_keys else "ACTION1"), "desperate"

    def tier_reached_name(self) -> str:
        return TIER_NAMES.get(self.max_tier_reached, "none")
