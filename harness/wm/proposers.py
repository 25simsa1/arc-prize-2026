"""Rule proposers. Interface: propose(store, current_model) -> list[Rule].

This is the interface a local coder-model backend implements later; nothing
here may assume how rules are produced. Both proposers are non-LLM:

  TemplateProposer — parameterized templates fit by enumeration over stored
  transitions; emits only templates with >= MIN_FITS exact fits.

  DiffMemorizer — exact (level, frame_hash, action) -> outcome memorization.
  Zero generalization; the control arm of the ablation. In a deterministic
  game it suffices to REPLAY any path already traversed — which is exactly
  why it's the floor a generalizing proposer must beat on coverage of
  UNSEEN transitions and plan quality.
"""

from collections import defaultdict
from typing import Optional

import numpy as np

from .rules import Prediction, Rule
from .store import (
    EVENT_GAME_OVER,
    EVENT_LEVEL,
    EVENT_NONE,
    EVENT_WIN,
    Transition,
    TransitionStore,
    frame_hash,
)

MIN_FITS = 2


# ---------------------------------------------------------------- helpers

def _positions(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    ys, xs = np.where(grid == color)
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


def _translate_candidates(t: Transition) -> list[tuple[int, int, int, int]]:
    """Candidate (color, dy, dx, bg) explaining t as 'color c moved by (dy,dx),
    vacated cells became bg, everything else unchanged'."""
    pre, post = t.pre, t.post
    changed = np.argwhere(pre != post)
    if changed.size == 0:
        return []
    out = []
    changed_set = {(int(y), int(x)) for y, x in changed}
    for c in {int(pre[y, x]) for y, x in changed_set} | {int(post[y, x]) for y, x in changed_set}:
        pre_c, post_c = _positions(pre, c), _positions(post, c)
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
    pre: np.ndarray, c: int, dy: int, dx: int, bg: int
) -> Optional[tuple[np.ndarray, list[tuple[int, int]]]]:
    """Move color c by (dy,dx) if every non-internal target cell is bg.
    Returns (grid, blocked_cells); grid None when blocked/absent."""
    pre_c = _positions(pre, c)
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
        return None, blocked  # type: ignore[return-value]
    grid = pre.copy()
    for y, x in pre_c:
        grid[y, x] = bg
    for y, x in pre_c:
        grid[y + dy, x + dx] = c
    return grid, []


def _blocking_colors(pre: np.ndarray, c: int, dy: int, dx: int, bg: int) -> set[int]:
    """Colors of in-bounds non-internal non-bg target cells (incl. out-of-bounds
    marker -1) for the would-be translation."""
    pre_c = _positions(pre, c)
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
        if p.grid is not None and not np.array_equal(p.grid, t.post):
            return -1
        if p.event is not None and p.event != t.event:
            return -1
        exact += 1
    return exact


# ---------------------------------------------------------------- templates

def _rule_identity(base: str) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        return Prediction(grid=pre.copy())

    return Rule(f"identity[{base}]", "identity", {"action": base}, fn,
                "template", specificity=10)


def _rule_event_const(base: str, event: str) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        return Prediction(event=event)

    return Rule(f"event[{base}={event}]", "event_const", {"action": base, "event": event},
                fn, "template", specificity=5)


def _rule_translate(base: str, c: int, dy: int, dx: int, bg: int) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        res = _apply_translate(pre, c, dy, dx, bg)
        if res is None or res[0] is None:
            return None
        return Prediction(grid=res[0])

    return Rule(f"translate[{base},c{c},{dy:+d},{dx:+d},bg{bg}]", "translate",
                {"action": base, "color": c, "dy": dy, "dx": dx, "bg": bg},
                fn, "template", specificity=40)


def _rule_blocked_identity(base: str, c: int, dy: int, dx: int, bg: int) -> Rule:
    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if action_key.split(":", 1)[0] != base:
            return None
        res = _apply_translate(pre, c, dy, dx, bg)
        if res is None:
            return None
        grid, blocked = res
        if grid is not None:
            return None  # not blocked: translate rule's territory
        return Prediction(grid=pre.copy())

    return Rule(f"blocked[{base},c{c},{dy:+d},{dx:+d}]", "blocked_identity",
                {"action": base, "color": c, "dy": dy, "dx": dx, "bg": bg},
                fn, "template", specificity=35)


def _rule_move_onto(dirmap: dict[str, tuple[int, int]], c: int, bg: int,
                    target: int, event: str) -> Rule:
    """Direction-agnostic: 'moving color c onto target color => event', with
    the per-action direction map taken from the fitted translate rules.
    Pooling across directions is what makes events fittable at all — each
    direction typically approaches a goal/hazard cell exactly once."""

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        d = dirmap.get(action_key.split(":", 1)[0])
        if d is None:
            return None
        if target in _blocking_colors(pre, c, d[0], d[1], bg):
            return Prediction(event=event)
        return None

    return Rule(f"move_onto[c{c}->{target}={event}]", "move_onto",
                {"dirmap": dict(dirmap), "color": c, "bg": bg,
                 "target": target, "event": event},
                fn, "template", specificity=50)


def _rule_move_free(dirmap: dict[str, tuple[int, int]], c: int, bg: int) -> Rule:
    """Complement of move_onto: an unblocked fitted move triggers nothing."""

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        d = dirmap.get(action_key.split(":", 1)[0])
        if d is None:
            return None
        if _blocking_colors(pre, c, d[0], d[1], bg):
            return None
        return Prediction(event=EVENT_NONE)

    return Rule(f"move_free[c{c}=NONE]", "move_free",
                {"dirmap": dict(dirmap), "color": c, "bg": bg},
                fn, "template", specificity=45)


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


def _rule_click_recolor(a: int, b: int, component: bool) -> Rule:
    kind = "component" if component else "cell"

    def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        if not action_key.startswith("ACTION6:"):
            return None
        x, y = (int(v) for v in action_key.split(":", 1)[1].split(","))
        h, w = pre.shape
        if not (0 <= y < h and 0 <= x < w) or pre[y, x] != a:
            return None
        grid = pre.copy()
        cells = _connected_component(pre, y, x) if component else {(y, x)}
        for cy, cx in cells:
            grid[cy, cx] = b
        return Prediction(grid=grid)

    return Rule(f"click_{kind}[{a}->{b}]", f"click_{kind}", {"from": a, "to": b},
                fn, "template", specificity=50)


class TemplateProposer:
    """Fits on a bounded SAMPLE of the store (propose-time cost must not grow
    with exploration length — it was 96% of wall-clock unbounded); the
    verifier remains the full-store ground truth, so a sample-fit rule that
    doesn't generalize gets CONTRADICTED there."""

    name = "template"

    def __init__(self, max_fit_per_base: int = 120) -> None:
        self.max_fit_per_base = max_fit_per_base

    def propose(self, store: TransitionStore, model=None) -> list[Rule]:
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
            consider(_rule_identity(base), ts)
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
                for cand in _translate_candidates(t):
                    if cand in seen_cands:
                        continue
                    seen_cands.add(cand)
                    c, dy, dx, bg = cand
                    rule = _rule_translate(base, c, dy, dx, bg)
                    consider(rule, ts)
                    if rule.rule_id in rules:
                        prev = fitted_translate.get(base)
                        if prev is None or rules[rule.rule_id].fit_count > prev[0]:
                            fitted_translate[base] = (
                                rules[rule.rule_id].fit_count, (c, dy, dx, bg)
                            )
                    consider(_rule_blocked_identity(base, c, dy, dx, bg), ts)

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
                for col in _blocking_colors(u.pre, c, d[0], d[1], bg):
                    onto[col].add(u.event)
            for col, evs in onto.items():
                if len(evs) == 1:
                    consider(_rule_move_onto(dirmap, c, bg, col, evs.pop()), all_ts)
            consider(_rule_move_free(dirmap, c, bg), all_ts)

            if base == "ACTION6":
                pairs_cell: set[tuple[int, int]] = set()
                pairs_comp: set[tuple[int, int]] = set()
                for t in ts:
                    xy = t.click_xy
                    if xy is None:
                        continue
                    x, y = xy
                    h, w = t.pre.shape
                    if not (0 <= y < h and 0 <= x < w):
                        continue
                    a, b = int(t.pre[y, x]), int(t.post[y, x])
                    if a != b:
                        pairs_cell.add((a, b))
                        pairs_comp.add((a, b))
                for a, b in pairs_cell:
                    consider(_rule_click_recolor(a, b, component=False), ts)
                for a, b in pairs_comp:
                    consider(_rule_click_recolor(a, b, component=True), ts)

        return list(rules.values())


class DiffMemorizer:
    """Exact memorization of the live store. One rule, claims grid+event."""

    name = "memo"

    def propose(self, store: TransitionStore, model=None) -> list[Rule]:
        def fn(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
            t = store.lookup(level, frame_hash(pre), action_key)
            if t is None:
                return None
            return Prediction(grid=t.post.copy(), event=t.event)

        rule = Rule("memo[store]", "memo", {}, fn, "memo",
                    fit_count=len(store), specificity=100)
        return [rule]
