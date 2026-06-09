"""Rule proposers. Interface: propose(store, current_model) -> list[Rule].

This is the interface a local coder-model backend implements later; nothing
here may assume how rules are produced. Both proposers are non-LLM:

  TemplateProposer — parameterized templates fit by enumeration over stored
  transitions; emits only templates with >= MIN_FITS exact fits.

  DiffMemorizer — exact (level, context_hash, action) -> outcome
  memorization. Zero generalization; the control arm of the ablation.

R1 region factoring: when current_model.hud_mask is set, grid templates
claim only the DYNAMIC region (mask=~hud_mask) — the cd82 evidence: a HUD
meter cell in row 63 ticks on every action, so unmasked whole-frame
templates can never fit (grid coverage measured flat 0.0 over 31k actions).
ALWAYS_CHANGING regions are NOT dropped on the floor: they get their own
template family (state-table ticker/counter, action-echo) proposed and
verified like any other rule — a meter that hits a threshold may be exactly
the latent precondition R3 needs later. With hud_mask=None (the ablation
switch) every code path below reduces to the pre-R1 behavior.
"""

from collections import defaultdict
from typing import Optional

import numpy as np

from .rules import Prediction, Rule, WorldModel, grids_match
from .store import (
    EVENT_GAME_OVER,
    EVENT_LEVEL,
    EVENT_NONE,
    EVENT_WIN,
    Transition,
    TransitionStore,
    frame_hash,
    masked_hash,
)

MIN_FITS = 2


# ---------------------------------------------------------------- helpers

def _positions(grid: np.ndarray, color: int,
               dyn_mask: Optional[np.ndarray]) -> set[tuple[int, int]]:
    sel = grid == color
    if dyn_mask is not None:
        sel = sel & dyn_mask
    ys, xs = np.where(sel)
    return {(int(y), int(x)) for y, x in zip(ys, xs)}


def _connected_component(grid: np.ndarray, y: int, x: int) -> set[tuple[int, int]]:
    """4-connected same-color component containing (y, x)."""
    color = grid[y, x]
    seen = {(y, x)}
    stack = [(y, x)]
    h, w = grid.shape
    while stack:
        cy, cx = stack.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and grid[ny, nx] == color:
                seen.add((ny, nx))
                stack.append((ny, nx))
    return seen


def _changed_cells(t: Transition, dyn_mask: Optional[np.ndarray]) -> np.ndarray:
    diff = t.pre != t.post
    if dyn_mask is not None:
        diff = diff & dyn_mask
    return np.argwhere(diff)


def _translate_candidates(
    t: Transition, dyn_mask: Optional[np.ndarray]
) -> list[tuple[int, int, int, int]]:
    """Candidate (color, dy, dx, bg) explaining t's DYNAMIC-region change as
    'color c moved by (dy,dx), vacated cells became bg'."""
    pre, post = t.pre, t.post
    changed = _changed_cells(t, dyn_mask)
    if changed.size == 0:
        return []
    out = []
    changed_set = {(int(y), int(x)) for y, x in changed}
    for c in {int(pre[y, x]) for y, x in changed_set} | {int(post[y, x]) for y, x in changed_set}:
        pre_c = _positions(pre, c, dyn_mask)
        post_c = _positions(post, c, dyn_mask)
        if not pre_c or len(pre_c) != len(post_c) or pre_c == post_c:
            continue
        p0 = next(iter(pre_c))
        for q in post_c:
            dy, dx = q[0] - p0[0], q[1] - p0[1]
            if (dy, dx) == (0, 0):
                continue
            if {(y + dy, x + dx) for y, x in pre_c} != post_c:
                continue
            vacated = pre_c - post_c
            bgs = {int(post[y, x]) for y, x in vacated}
            if len(bgs) != 1:
                continue
            out.append((c, dy, dx, bgs.pop()))
    return out


def _apply_translate(
    pre: np.ndarray, c: int, dy: int, dx: int, bg: int,
    dyn_mask: Optional[np.ndarray],
) -> Optional[tuple[Optional[np.ndarray], list]]:
    """Move color c by (dy,dx) if every non-internal target cell is bg.
    Returns (grid, blocked_cells); grid None when blocked. HUD cells are
    untouched and excluded from the move geometry."""
    pre_c = _positions(pre, c, dyn_mask)
    if not pre_c:
        return None
    h, w = pre.shape
    blocked = []
    for y, x in pre_c:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w):
            blocked.append((-1, -1))
        elif (ny, nx) not in pre_c and pre[ny, nx] != bg:
            blocked.append((ny, nx))
    if blocked:
        return None, blocked
    grid = pre.copy()
    for y, x in pre_c:
        grid[y, x] = bg
    for y, x in pre_c:
        grid[y + dy, x + dx] = c
    return grid, []


def _blocking_colors(pre: np.ndarray, c: int, dy: int, dx: int, bg: int,
                     dyn_mask: Optional[np.ndarray]) -> set[int]:
    pre_c = _positions(pre, c, dyn_mask)
    h, w = pre.shape
    colors: set[int] = set()
    for y, x in pre_c:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w):
            colors.add(-1)
        elif (ny, nx) not in pre_c and pre[ny, nx] != bg:
            colors.add(int(pre[ny, nx]))
    return colors


def _fits(rule: Rule, transitions: list[Transition]) -> int:
    """Exact fits of a rule over transitions (misses disqualify entirely)."""
    exact = 0
    for t in transitions:
        p = rule.predict(t.level, t.pre, t.action_key)
        if p is None:
            continue
        if not grids_match(p, t.post):
            return -1
        if p.event is not None and p.event != t.event:
            return -1
        exact += 1
    return exact


# ---------------------------------------------------------------- templates

def _rule_identity(base: str, dyn_mask: Optional[np.ndarray]) -> Rule:
    scope = "dyn" if dyn_mask is not None else "full"

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        return Prediction(grid=pre.copy(), mask=dyn_mask)

    return Rule(f"identity[{base},{scope}]", "identity", {"action": base}, fn,
                "template", specificity=10,
                region="dynamic" if dyn_mask is not None else "full")


def _rule_event_const(base: str, event: str) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        return Prediction(event=event)

    return Rule(f"event[{base}={event}]", "event_const", {"action": base, "event": event},
                fn, "template", specificity=5)


# Event observations are deduplicated singletons by construction (a level
# transition is stored once per (level, frame, action)), so a >=2-fits
# threshold would demand revisiting a level twice before the planner may even
# consider its goal. Level-scoped constant-event rules therefore use
# threshold 1; they start UNTESTED and the data decides from there.
MIN_FITS_EVENT_AT_LEVEL = 1


def _rule_event_at_level(base: str, level: int, event: str) -> Rule:
    def fn(lvl: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if lvl != level or action_key.split(":", 1)[0] != base:
            return None
        return Prediction(event=event)

    return Rule(f"event_at[{base},L{level}={event}]", "event_at_level",
                {"action": base, "level": level, "event": event},
                fn, "template", specificity=7)


def _rule_translate(base: str, c: int, dy: int, dx: int, bg: int,
                    dyn_mask: Optional[np.ndarray]) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        res = _apply_translate(pre, c, dy, dx, bg, dyn_mask)
        if res is None or res[0] is None:
            return None
        return Prediction(grid=res[0], mask=dyn_mask)

    return Rule(f"translate[{base},c{c},{dy:+d},{dx:+d},bg{bg}]", "translate",
                {"action": base, "color": c, "dy": dy, "dx": dx, "bg": bg},
                fn, "template", specificity=40,
                region="dynamic" if dyn_mask is not None else "full")


def _rule_blocked_identity(base: str, c: int, dy: int, dx: int, bg: int,
                           dyn_mask: Optional[np.ndarray]) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        res = _apply_translate(pre, c, dy, dx, bg, dyn_mask)
        if res is None:
            return None
        grid, blocked = res
        if grid is not None:
            return None  # not blocked: translate rule's territory
        return Prediction(grid=pre.copy(), mask=dyn_mask)

    return Rule(f"blocked[{base},c{c},{dy:+d},{dx:+d}]", "blocked_identity",
                {"action": base, "color": c, "dy": dy, "dx": dx, "bg": bg},
                fn, "template", specificity=35,
                region="dynamic" if dyn_mask is not None else "full")


def _rule_move_onto(dirmap: dict[str, tuple[int, int]], c: int, bg: int,
                    target: int, event: str,
                    dyn_mask: Optional[np.ndarray]) -> Rule:
    """Direction-agnostic: 'moving color c onto target color => event', with
    the per-action direction map taken from the fitted translate rules.
    Pooling across directions is what makes events fittable at all — each
    direction typically approaches a goal/hazard cell exactly once."""

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        d = dirmap.get(action_key.split(":", 1)[0])
        if d is None:
            return None
        if target in _blocking_colors(pre, c, d[0], d[1], bg, dyn_mask):
            return Prediction(event=event)
        return None

    return Rule(f"move_onto[c{c}->{target}={event}]", "move_onto",
                {"dirmap": dict(dirmap), "color": c, "bg": bg,
                 "target": target, "event": event},
                fn, "template", specificity=50)


def _rule_move_free(dirmap: dict[str, tuple[int, int]], c: int, bg: int,
                    dyn_mask: Optional[np.ndarray]) -> Rule:
    """Complement of move_onto: an unblocked fitted move triggers nothing."""

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        d = dirmap.get(action_key.split(":", 1)[0])
        if d is None:
            return None
        if _blocking_colors(pre, c, d[0], d[1], bg, dyn_mask):
            return None
        return Prediction(event=EVENT_NONE)

    return Rule(f"move_free[c{c}=NONE]", "move_free",
                {"dirmap": dict(dirmap), "color": c, "bg": bg},
                fn, "template", specificity=45)


def _rule_click_recolor(a: int, b: int, component: bool,
                        dyn_mask: Optional[np.ndarray]) -> Rule:
    kind = "component" if component else "cell"

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if not action_key.startswith("ACTION6:"):
            return None
        x, y = (int(v) for v in action_key.split(":", 1)[1].split(","))
        h, w = pre.shape
        if not (0 <= y < h and 0 <= x < w) or pre[y, x] != a:
            return None
        if dyn_mask is not None and not dyn_mask[y, x]:
            return None  # clicking inside the HUD: not this rule's claim
        grid = pre.copy()
        cells = _connected_component(pre, y, x) if component else {(y, x)}
        for cy, cx in cells:
            grid[cy, cx] = b
        return Prediction(grid=grid, mask=dyn_mask)

    return Rule(f"click_{kind}[{a}->{b}]", f"click_{kind}", {"from": a, "to": b},
                fn, "template", specificity=50,
                region="dynamic" if dyn_mask is not None else "full")


def _rule_hud_state_table(region_id: int, cells: list[tuple[int, int]],
                          table: dict[bytes, np.ndarray],
                          shape: tuple[int, int]) -> Rule:
    """ALWAYS_CHANGING region modeled as a state machine: current region
    contents -> next region contents, independent of action identity. Covers
    monotone counters and cyclic tickers alike; the verifier decides."""
    ys = np.array([c[0] for c in cells])
    xs = np.array([c[1] for c in cells])
    mask = np.zeros(shape, dtype=bool)
    mask[ys, xs] = True

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        nxt = table.get(pre[ys, xs].astype(np.int16).tobytes())
        if nxt is None:
            return None
        grid = pre.copy()
        grid[ys, xs] = nxt
        return Prediction(grid=grid, mask=mask)

    return Rule(f"hud_state[R{region_id}]", "hud_state_table",
                {"region": region_id, "cells": len(cells), "states": len(table)},
                fn, "template", specificity=60, region="hud")


def _rule_hud_action_echo(region_id: int, cells: list[tuple[int, int]],
                          table: dict[str, np.ndarray],
                          shape: tuple[int, int]) -> Rule:
    """ALWAYS_CHANGING region whose next contents depend on the action taken
    (e.g. a last-action indicator), independent of current contents."""
    ys = np.array([c[0] for c in cells])
    xs = np.array([c[1] for c in cells])
    mask = np.zeros(shape, dtype=bool)
    mask[ys, xs] = True

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        nxt = table.get(action_key.split(":", 1)[0])
        if nxt is None:
            return None
        grid = pre.copy()
        grid[ys, xs] = nxt
        return Prediction(grid=grid, mask=mask)

    return Rule(f"hud_echo[R{region_id}]", "hud_action_echo",
                {"region": region_id, "cells": len(cells), "actions": len(table)},
                fn, "template", specificity=58, region="hud")


class TemplateProposer:
    """Fits on a bounded SAMPLE of the store (propose-time cost must not grow
    with exploration length — it was 96% of wall-clock unbounded); the
    verifier remains the full-store ground truth, so a sample-fit rule that
    doesn't generalize gets CONTRADICTED there."""

    name = "template"

    def __init__(self, max_fit_per_base: int = 120) -> None:
        self.max_fit_per_base = max_fit_per_base

    def propose(self, store: TransitionStore, model: Optional[WorldModel] = None) -> list[Rule]:
        hud_mask = model.hud_mask if model is not None else None
        dyn_mask = ~hud_mask if hud_mask is not None else None
        region_map = getattr(model, "region_map", None) if model is not None else None

        rules: dict[str, Rule] = {}
        by_base: dict[str, list[Transition]] = defaultdict(list)
        for t in store.all():
            by_base[t.base_action].append(t)
        for base, ts in by_base.items():
            if len(ts) > self.max_fit_per_base:
                step = len(ts) // self.max_fit_per_base
                by_base[base] = ts[::step][: self.max_fit_per_base]

        def consider(rule: Rule, transitions: list[Transition],
                     min_fits: int = MIN_FITS) -> None:
            if rule.rule_id in rules:
                return
            fits = _fits(rule, transitions)
            if fits >= min_fits:
                rule.fit_count = fits
                rules[rule.rule_id] = rule

        all_ts = [t for ts in by_base.values() for t in ts]  # sampled view
        # fitted translation per (base -> (c, dy, dx, bg)), best fits wins
        fitted_translate: dict[str, tuple[int, tuple[int, int, int, int]]] = {}

        for base, ts in by_base.items():
            consider(_rule_identity(base, dyn_mask), ts)
            events = {t.event for t in ts}
            if len(events) == 1:
                consider(_rule_event_const(base, events.pop()), ts)
            for lvl in {t.level for t in ts}:
                lvl_events = {t.event for t in ts if t.level == lvl}
                if len(lvl_events) == 1:
                    consider(_rule_event_at_level(base, lvl, lvl_events.pop()),
                             ts, MIN_FITS_EVENT_AT_LEVEL)

            # translate family: candidates from each changed transition
            seen_cands: set[tuple[int, int, int, int]] = set()
            for t in ts:
                for cand in _translate_candidates(t, dyn_mask):
                    if cand in seen_cands:
                        continue
                    seen_cands.add(cand)
                    c, dy, dx, bg = cand
                    rule = _rule_translate(base, c, dy, dx, bg, dyn_mask)
                    consider(rule, ts)
                    if rule.rule_id in rules:
                        prev = fitted_translate.get(base)
                        if prev is None or rules[rule.rule_id].fit_count > prev[0]:
                            fitted_translate[base] = (
                                rules[rule.rule_id].fit_count, (c, dy, dx, bg)
                            )
                    consider(_rule_blocked_identity(base, c, dy, dx, bg, dyn_mask), ts)

            if base == "ACTION6":
                pairs: set[tuple[int, int]] = set()
                for t in ts:
                    xy = t.click_xy
                    if xy is None:
                        continue
                    x, y = xy
                    h, w = t.pre.shape
                    if not (0 <= y < h and 0 <= x < w):
                        continue
                    if dyn_mask is not None and not dyn_mask[y, x]:
                        continue
                    a, b = int(t.pre[y, x]), int(t.post[y, x])
                    if a != b:
                        pairs.add((a, b))
                for a, b in pairs:
                    consider(_rule_click_recolor(a, b, False, dyn_mask), ts)
                    consider(_rule_click_recolor(a, b, True, dyn_mask), ts)

        # move-onto events, pooled across directions via the fitted dirmap
        by_cbg: dict[tuple[int, int], dict[str, tuple[int, int]]] = defaultdict(dict)
        for base, (_, (c, dy, dx, bg)) in fitted_translate.items():
            by_cbg[(c, bg)][base] = (dy, dx)
        for (c, bg), dirmap in by_cbg.items():
            onto: dict[int, set[str]] = defaultdict(set)
            for u in all_ts:
                d = dirmap.get(u.base_action)
                if d is None:
                    continue
                for col in _blocking_colors(u.pre, c, d[0], d[1], bg, dyn_mask):
                    onto[col].add(u.event)
            for col, evs in onto.items():
                if len(evs) == 1:
                    consider(_rule_move_onto(dirmap, c, bg, col, evs.pop(), dyn_mask),
                             all_ts)
            consider(_rule_move_free(dirmap, c, bg, dyn_mask), all_ts)

        # HUD families: model ALWAYS_CHANGING regions instead of dropping them
        if region_map is not None and all_ts:
            shape = all_ts[0].pre.shape
            for region in region_map.hud_regions:
                cells = [tuple(c) for c in region.cells]
                ys = np.array([c[0] for c in cells])
                xs = np.array([c[1] for c in cells])
                state_table: dict[bytes, np.ndarray] = {}
                state_ok = True
                echo_table: dict[str, np.ndarray] = {}
                echo_ok = True
                for t in all_ts:
                    key = t.pre[ys, xs].astype(np.int16).tobytes()
                    nxt = t.post[ys, xs].astype(np.int16)
                    prev = state_table.get(key)
                    if prev is None:
                        state_table[key] = nxt
                    elif not np.array_equal(prev, nxt):
                        state_ok = False
                    eprev = echo_table.get(t.base_action)
                    if eprev is None:
                        echo_table[t.base_action] = nxt
                    elif not np.array_equal(eprev, nxt):
                        echo_ok = False
                if state_ok and state_table:
                    consider(_rule_hud_state_table(region.region_id, cells,
                                                   state_table, shape), all_ts)
                if echo_ok and echo_table:
                    consider(_rule_hud_action_echo(region.region_id, cells,
                                                   echo_table, shape), all_ts)

        return list(rules.values())


class DiffMemorizer:
    """Exact memorization of the live store. One rule, claims grid+event.
    Under region factoring the lookup keys on the masked context hash, so a
    ticking HUD doesn't make every stored state unreachable; the grid claim
    is scoped to the dynamic region accordingly."""

    name = "memo"

    def propose(self, store: TransitionStore, model: Optional[WorldModel] = None) -> list[Rule]:
        hud_mask = model.hud_mask if model is not None else None
        dyn_mask = ~hud_mask if hud_mask is not None else None

        if hud_mask is None:
            def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
                t = store.lookup(level, frame_hash(pre), action_key)
                if t is None:
                    return None
                return Prediction(grid=t.post.copy(), event=t.event)
        else:
            index: dict[tuple[int, str, str], Transition] = {}
            for t in store.all():
                index.setdefault(
                    (t.level, masked_hash(t.pre, hud_mask), t.action_key), t
                )

            def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
                t = index.get((level, masked_hash(pre, hud_mask), action_key))
                if t is None:
                    return None
                return Prediction(grid=t.post.copy(), event=t.event, mask=dyn_mask)

        rule = Rule("memo[store]", "memo", {}, fn, "memo",
                    fit_count=len(store), specificity=100,
                    region="dynamic" if hud_mask is not None else "full")
        return [rule]
