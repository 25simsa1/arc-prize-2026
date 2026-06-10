# ============================================================================
# ARC-AGI-3 — Milestone 1 baseline: template-rule agent (single play)
#
# A small, self-contained baseline agent for the ARC Prize 2026 ARC-AGI-3
# track. It learns simple transition rules from its own observations while
# playing (movement templates, click effects, UI-region updates, event
# preconditions), verifies them against everything it has seen, and uses a
# short forward search to reach level goals when its rules support one.
# Also includes two trivial baselines (random, action-sweep) for reference.
#
# Open source under CC0 / MIT-0 per the competition's open-source rules.
# ============================================================================

import hashlib
import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 0. Environment: the competition runtime (arc_agi / arcengine) ships as
# wheels in the competition data. Install offline if not already present.
# ---------------------------------------------------------------------------
try:
    import arc_agi  # noqa: F401
except ImportError:
    import subprocess
    import sys
    wheel_dirs = []
    for root in ("/kaggle/input",):
        for dirpath, _, filenames in os.walk(root):
            if any(f.startswith("arc_agi") and f.endswith(".whl") for f in filenames):
                wheel_dirs.append(dirpath)
    assert wheel_dirs, "competition wheels not found under /kaggle/input"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", wheel_dirs[0], "arc-agi", "arcengine"],
                   check=True)

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import FrameDataRaw, GameAction, GameState

EVENT_NONE, EVENT_LEVEL, EVENT_WIN, EVENT_GO = "NONE", "LEVEL", "WIN", "GAME_OVER"
_ID2ACT = {a.value: a for a in GameAction}
_NAME2ACT = {a.name: a for a in GameAction}


def canon(frame_layers) -> np.ndarray:
    """Canonical state = last rendered layer (earlier layers are animation)."""
    return np.ascontiguousarray(np.asarray(frame_layers[-1], dtype=np.int16))


def fhash(grid: np.ndarray, exclude=None) -> str:
    h = hashlib.sha1()
    h.update(str(grid.shape).encode())
    h.update((grid[~exclude] if exclude is not None else grid).tobytes())
    return h.hexdigest()[:20]


def derive_event(pre_level: int, post_level: int, post_state: str) -> str:
    if post_state == "GAME_OVER":
        return EVENT_GO
    if post_state == "WIN":
        return EVENT_WIN
    return EVENT_LEVEL if post_level > pre_level else EVENT_NONE


# ---------------------------------------------------------------------------
# 1. Observation store: deduplicated (level, frame, action) -> outcome, with
# bounded memory (event transitions and small-diff observations are the most
# informative for rule fitting, so they are kept preferentially).
# ---------------------------------------------------------------------------
@dataclass
class Obs:
    level: int
    pre: np.ndarray
    pre_hash: str
    action_key: str
    post: np.ndarray
    post_hash: str
    event: str
    diff_cells: int

    @property
    def base_action(self) -> str:
        return self.action_key.split(":", 1)[0]

    @property
    def click_xy(self):
        if ":" not in self.action_key:
            return None
        x, y = self.action_key.split(":", 1)[1].split(",")
        return int(x), int(y)


class ObsStore:
    def __init__(self, game_id: str, cap: int = 120_000) -> None:
        self.game_id = game_id
        self.cap = cap
        self.by_key: dict = {}
        self._evictable: deque = deque()

    def _keep(self, o: Obs) -> bool:
        return o.event != EVENT_NONE or o.diff_cells <= 2

    def add(self, level, pre, action_key, post, post_level, post_state):
        ph, qh = fhash(pre), fhash(post)
        key = (level, ph, action_key)
        event = derive_event(level, post_level, post_state)
        ex = self.by_key.get(key)
        if ex is not None:
            return ("dup", ex)
        if len(self.by_key) >= self.cap:
            while self._evictable:
                k = self._evictable.popleft()
                o = self.by_key.get(k)
                if o is not None and not self._keep(o):
                    del self.by_key[k]
                    break
            else:
                return ("full", None)
        o = Obs(level, pre.copy(), ph, action_key, post.copy(), qh, event,
                int(np.count_nonzero(pre != post)))
        self.by_key[key] = o
        if not self._keep(o):
            self._evictable.append(key)
        return ("new", o)

    def __len__(self):
        return len(self.by_key)

    def all(self):
        return self.by_key.values()

    def lookup(self, level, ph, ak):
        return self.by_key.get((level, ph, ak))


# ---------------------------------------------------------------------------
# 2. UI-region detection: cells that repeatedly change ALONE (and, for
# clicks, far from the click) behave like status displays rather than game
# content. Masking them lets content rules fit; the regions themselves get
# their own rules below.
# ---------------------------------------------------------------------------
class RegionFinder:
    def __init__(self, always_rate=0.55, min_obs=30, max_frac=0.10, gap=2):
        self.always_rate, self.min_obs = always_rate, min_obs
        self.max_frac, self.gap = max_frac, gap
        self.n = 0
        self.shape = None
        self.sole: dict = defaultdict(int)
        self.pair: dict = defaultdict(lambda: defaultdict(int))
        self.changed_at: dict = defaultdict(list)
        self.colors_seen: dict = defaultdict(set)

    def observe(self, o: Obs) -> None:
        if self.shape is None:
            self.shape = o.pre.shape
        self.n += 1
        cells = [(int(y), int(x)) for y, x in np.argwhere(o.pre != o.post)]
        for c in cells:
            self.changed_at[c].append(self.n)
            self.colors_seen[c].update((int(o.pre[c]), int(o.post[c])))
        if len(cells) == 1:
            xy = o.click_xy
            far = xy is None or max(abs(xy[1] - cells[0][0]), abs(xy[0] - cells[0][1])) > 1
            if far:
                self.sole[cells[0]] += 1
        elif len(cells) == 2:
            a, b = cells
            self.pair[a][b] += 1
            self.pair[b][a] += 1

    def regions(self, protected_colors=()) -> list[list[tuple[int, int]]]:
        if self.shape is None or self.n < self.min_obs:
            return []
        seeds = {c for c, k in self.sole.items() if k >= 2}
        tier2 = {c for c, ps in self.pair.items()
                 if c not in seeds and any(p in seeds and k >= 2 for p, k in ps.items())}
        members = [c for c in seeds | tier2
                   if not (self.colors_seen[c] & set(protected_colors))]
        # spatial clustering (chebyshev <= gap)
        out, left = [], set(members)
        while left:
            seed = left.pop()
            comp, frontier = [seed], [seed]
            while frontier:
                cy, cx = frontier.pop()
                near = [c for c in left if abs(c[0] - cy) <= self.gap and abs(c[1] - cx) <= self.gap]
                for c in near:
                    left.remove(c)
                    comp.append(c)
                    frontier.append(c)
            out.append(comp)
        keep = []
        cap = int(self.max_frac * self.shape[0] * self.shape[1])
        for comp in out:
            if len(comp) > cap:
                continue
            hits = set()
            for c in comp:
                hits.update(self.changed_at[c])
            if len(hits) / self.n >= self.always_rate:
                keep.append(sorted(comp))
        return keep


# ---------------------------------------------------------------------------
# 3. Rules and the model: a rule predicts grid (within its claimed region)
# and/or the event for observations it covers; returning None means "not my
# transition". Rules are verified by exact comparison against the store.
# ---------------------------------------------------------------------------
class St(str, Enum):
    UNTESTED = "UNTESTED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class Pred:
    grid: Optional[np.ndarray] = None
    event: Optional[str] = None
    mask: Optional[np.ndarray] = None


@dataclass
class Rule:
    rule_id: str
    fn: Callable
    spec: int = 0
    region: str = "full"
    status: St = St.UNTESTED
    fits: int = 0

    def predict(self, level, pre, ak):
        return self.fn(level, pre, ak)


def grids_match(p: Pred, post: np.ndarray) -> bool:
    if p.grid is None:
        return True
    if p.mask is None:
        return bool(np.array_equal(p.grid, post))
    return bool(np.array_equal(p.grid[p.mask], post[p.mask]))


_RANK = {St.VERIFIED: 0, St.UNTESTED: 1, St.CONTRADICTED: 2}


class Model:
    def __init__(self):
        self.rules: list[Rule] = []
        self.ui_mask: Optional[np.ndarray] = None
        self.version = 0

    def ordered(self):
        return sorted(self.rules, key=lambda r: (_RANK[r.status], -r.spec, r.rule_id))

    def predict(self, level, pre, ak, allowed=(St.VERIFIED, St.UNTESTED)):
        grid = mask = event = None
        ui_grid = ui_mask = None
        g_st = e_st = None
        for r in self.ordered():
            if r.status not in allowed:
                continue
            if grid is not None and event is not None and ui_grid is not None:
                break
            p = r.predict(level, pre, ak)
            if p is None:
                continue
            if p.grid is not None:
                if r.region == "ui":
                    if ui_grid is None:
                        ui_grid, ui_mask = p.grid, p.mask
                elif grid is None:
                    grid, mask, g_st = p.grid, p.mask, r.status
            if p.event is not None and event is None:
                event, e_st = p.event, r.status
        return grid, mask, event, ui_grid, ui_mask, g_st, e_st

    def successor(self, pre, grid, mask, ui_grid, ui_mask):
        if grid is None:
            return None
        out = pre.copy()
        if mask is None:
            out = grid.copy()
        else:
            out[mask] = grid[mask]
        if ui_grid is not None and ui_mask is not None:
            out[ui_mask] = ui_grid[ui_mask]
        return out


def verify(rules: list[Rule], store: ObsStore, min_exact=3, budget_s=2.0) -> None:
    t0 = time.monotonic()
    obs = list(store.all())
    for r in rules:
        if time.monotonic() - t0 > budget_s:
            return
        exact = miss = 0
        for o in obs:
            p = r.predict(o.level, o.pre, o.action_key)
            if p is None:
                continue
            ok = grids_match(p, o.post) and (p.event is None or p.event == o.event)
            exact, miss = exact + ok, miss + (not ok)
        r.status = (St.CONTRADICTED if miss else
                    St.VERIFIED if exact >= min_exact else St.UNTESTED)


# ---------------------------------------------------------------------------
# 4. Template proposer: enumerate small parameterized rules that fit the
# observations (>=2 exact fits; event-at-level rules allowed at 1).
# ---------------------------------------------------------------------------
def positions(grid, color, dyn):
    sel = grid == color
    if dyn is not None:
        sel = sel & dyn
    return {(int(y), int(x)) for y, x in np.argwhere(sel)}


def apply_translate(pre, c, dy, dx, bg, dyn):
    pc = positions(pre, c, dyn)
    if not pc:
        return None
    h, w = pre.shape
    blocked = []
    for y, x in pc:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w):
            blocked.append((-1, -1))
        elif (ny, nx) not in pc and pre[ny, nx] != bg:
            blocked.append((ny, nx))
    if blocked:
        return None, blocked
    g = pre.copy()
    for y, x in pc:
        g[y, x] = bg
    for y, x in pc:
        g[y + dy, x + dx] = c
    return g, []


def blocking_colors(pre, c, dy, dx, bg, dyn):
    pc = positions(pre, c, dyn)
    h, w = pre.shape
    cols = set()
    for y, x in pc:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w):
            cols.add(-1)
        elif (ny, nx) not in pc and pre[ny, nx] != bg:
            cols.add(int(pre[ny, nx]))
    return cols


def fits_count(rule: Rule, obs: list[Obs]) -> int:
    n = 0
    for o in obs:
        p = rule.predict(o.level, o.pre, o.action_key)
        if p is None:
            continue
        if not grids_match(p, o.post) or (p.event is not None and p.event != o.event):
            return -1
        n += 1
    return n


def propose(store: ObsStore, model: Model, region_cells=None,
            max_fit_per_base=120) -> list[Rule]:
    ui = model.ui_mask
    dyn = ~ui if ui is not None else None
    rules: dict[str, Rule] = {}
    by_base: dict[str, list[Obs]] = defaultdict(list)
    for o in store.all():
        by_base[o.base_action].append(o)
    for b, obs in by_base.items():
        if len(obs) > max_fit_per_base:
            by_base[b] = obs[:: len(obs) // max_fit_per_base][:max_fit_per_base]
    all_obs = [o for v in by_base.values() for o in v]

    def consider(rule: Rule, obs, min_fits=2):
        if rule.rule_id in rules:
            return
        f = fits_count(rule, obs)
        if f >= min_fits:
            rule.fits = f
            rules[rule.rule_id] = rule

    fitted: dict[str, tuple[int, tuple]] = {}
    for base, obs in by_base.items():
        scope = "dyn" if dyn is not None else "full"

        def mk_ident(b=base):
            return Rule(f"identity[{b},{scope}]",
                        lambda lv, pre, ak, b=b: Pred(grid=pre.copy(), mask=dyn)
                        if ak.split(":")[0] == b else None,
                        spec=10, region="dynamic" if dyn is not None else "full")
        consider(mk_ident(), obs)
        evs = {o.event for o in obs}
        if len(evs) == 1:
            ev = evs.pop()
            consider(Rule(f"event[{base}={ev}]",
                          lambda lv, pre, ak, b=base, e=ev: Pred(event=e)
                          if ak.split(":")[0] == b else None, spec=5), obs)
        for lvl in {o.level for o in obs}:
            lev = {o.event for o in obs if o.level == lvl}
            if len(lev) == 1:
                e = lev.pop()
                consider(Rule(f"event_at[{base},L{lvl}={e}]",
                              lambda lv, pre, ak, b=base, L=lvl, e=e:
                              Pred(event=e) if lv == L and ak.split(":")[0] == b else None,
                              spec=7), obs, min_fits=1)

        seen = set()
        for o in obs:
            ch = np.argwhere((o.pre != o.post) & dyn) if dyn is not None \
                else np.argwhere(o.pre != o.post)
            chs = {(int(y), int(x)) for y, x in ch}
            for c in {int(o.pre[y, x]) for y, x in chs} | {int(o.post[y, x]) for y, x in chs}:
                pc, qc = positions(o.pre, c, dyn), positions(o.post, c, dyn)
                if not pc or len(pc) != len(qc) or pc == qc:
                    continue
                p0 = next(iter(pc))
                for q in qc:
                    dy, dx = q[0] - p0[0], q[1] - p0[1]
                    if (dy, dx) == (0, 0) or {(y + dy, x + dx) for y, x in pc} != qc:
                        continue
                    bgs = {int(o.post[y, x]) for y, x in pc - qc}
                    if len(bgs) != 1:
                        continue
                    cand = (c, dy, dx, bgs.pop())
                    if cand in seen:
                        continue
                    seen.add(cand)
                    cc, ddy, ddx, bg = cand

                    def t_fn(lv, pre, ak, b=base, cc=cc, ddy=ddy, ddx=ddx, bg=bg):
                        if ak.split(":")[0] != b:
                            return None
                        r = apply_translate(pre, cc, ddy, ddx, bg, dyn)
                        return Pred(grid=r[0], mask=dyn) if r and r[0] is not None else None

                    tr = Rule(f"translate[{base},c{cc},{ddy:+d},{ddx:+d},bg{bg}]",
                              t_fn, spec=40,
                              region="dynamic" if dyn is not None else "full")
                    consider(tr, obs)
                    if tr.rule_id in rules:
                        prev = fitted.get(base)
                        if prev is None or rules[tr.rule_id].fits > prev[0]:
                            fitted[base] = (rules[tr.rule_id].fits, cand)

                    def b_fn(lv, pre, ak, b=base, cc=cc, ddy=ddy, ddx=ddx, bg=bg):
                        if ak.split(":")[0] != b:
                            return None
                        r = apply_translate(pre, cc, ddy, ddx, bg, dyn)
                        if r is None or r[0] is not None:
                            return None
                        return Pred(grid=pre.copy(), mask=dyn)
                    consider(Rule(f"blocked[{base},c{cc},{ddy:+d},{ddx:+d}]", b_fn,
                                  spec=35,
                                  region="dynamic" if dyn is not None else "full"), obs)

        if base == "ACTION6":
            pairs = set()
            for o in obs:
                xy = o.click_xy
                if xy is None:
                    continue
                x, y = xy
                if not (0 <= y < 64 and 0 <= x < 64):
                    continue
                if dyn is not None and not dyn[y, x]:
                    continue
                a, b2 = int(o.pre[y, x]), int(o.post[y, x])
                if a != b2:
                    pairs.add((a, b2))
            for a, b2 in pairs:
                def c_fn(lv, pre, ak, a=a, b2=b2):
                    if not ak.startswith("ACTION6:"):
                        return None
                    x, y = (int(v) for v in ak.split(":")[1].split(","))
                    if not (0 <= y < 64 and 0 <= x < 64) or pre[y, x] != a:
                        return None
                    if dyn is not None and not dyn[y, x]:
                        return None
                    g = pre.copy()
                    g[y, x] = b2
                    return Pred(grid=g, mask=dyn)
                consider(Rule(f"click[{a}->{b2}]", c_fn, spec=50,
                              region="dynamic" if dyn is not None else "full"), obs)

    # event templates pooled over the fitted move directions
    dirmaps: dict[tuple, dict] = defaultdict(dict)
    for base, (_, (c, dy, dx, bg)) in fitted.items():
        dirmaps[(c, bg)][base] = (dy, dx)
    for (c, bg), dm in dirmaps.items():
        onto: dict[int, set] = defaultdict(set)
        for o in all_obs:
            d = dm.get(o.base_action)
            if d is None:
                continue
            for col in blocking_colors(o.pre, c, d[0], d[1], bg, dyn):
                onto[col].add(o.event)
        for col, evs in onto.items():
            if len(evs) == 1:
                e = evs.pop()
                def mo_fn(lv, pre, ak, dm=dict(dm), c=c, bg=bg, col=col, e=e):
                    d = dm.get(ak.split(":")[0])
                    if d is None:
                        return None
                    return Pred(event=e) if col in blocking_colors(pre, c, d[0], d[1], bg, dyn) else None
                consider(Rule(f"move_onto[c{c}->{col}={e}]", mo_fn, spec=50), all_obs)

        def mf_fn(lv, pre, ak, dm=dict(dm), c=c, bg=bg):
            d = dm.get(ak.split(":")[0])
            if d is None:
                return None
            return None if blocking_colors(pre, c, d[0], d[1], bg, dyn) else Pred(event=EVENT_NONE)
        consider(Rule(f"move_free[c{c}]", mf_fn, spec=45), all_obs)

    # UI-region rules: state-table and input-echo models of each region
    if region_cells and all_obs:
        shape = all_obs[0].pre.shape
        for i, cells in enumerate(region_cells):
            ys = np.array([c[0] for c in cells])
            xs = np.array([c[1] for c in cells])
            m = np.zeros(shape, dtype=bool)
            m[ys, xs] = True
            table, ok = {}, True
            echo, eok = {}, True
            for o in all_obs:
                k = o.pre[ys, xs].astype(np.int16).tobytes()
                nxt = o.post[ys, xs].astype(np.int16)
                if k in table and not np.array_equal(table[k], nxt):
                    ok = False
                table.setdefault(k, nxt)
                eb = echo.get(o.base_action)
                if eb is not None and not np.array_equal(eb, nxt):
                    eok = False
                echo.setdefault(o.base_action, nxt)
            if ok and table:
                def st_fn(lv, pre, ak, t=table, ys=ys, xs=xs, m=m):
                    nxt = t.get(pre[ys, xs].astype(np.int16).tobytes())
                    if nxt is None:
                        return None
                    g = pre.copy()
                    g[ys, xs] = nxt
                    return Pred(grid=g, mask=m)
                consider(Rule(f"ui_state[R{i}]", st_fn, spec=60, region="ui"), all_obs)
            if eok and echo:
                def ec_fn(lv, pre, ak, t=echo, ys=ys, xs=xs, m=m):
                    nxt = t.get(ak.split(":")[0])
                    if nxt is None:
                        return None
                    g = pre.copy()
                    g[ys, xs] = nxt
                    return Pred(grid=g, mask=m)
                consider(Rule(f"ui_echo[R{i}]", ec_fn, spec=58, region="ui"), all_obs)

    return list(rules.values())


# ---------------------------------------------------------------------------
# 5. Planner: budgeted forward search to the next LEVEL/WIN through the
# model; GAME_OVER claims prune; state identity ignores UI regions.
# ---------------------------------------------------------------------------
def plan_next(model: Model, level, start, simple, clicks_fn, deadline,
              max_depth=64, max_nodes=20000):
    import heapq
    import itertools
    cnt = itertools.count()
    ui = model.ui_mask
    frontier = [(0.0, next(cnt), start, [])]
    seen = {fhash(start, ui)}
    nodes = 0
    while frontier:
        if time.monotonic() > deadline or nodes >= max_nodes:
            return None
        _, _, grid, path = heapq.heappop(frontier)
        nodes += 1
        if len(path) >= max_depth:
            continue
        for ak in list(simple) + clicks_fn(grid):
            g, m, ev, ug, um, g_st, e_st = model.predict(level, grid, ak)
            if ev is None:
                continue
            if ev in (EVENT_LEVEL, EVENT_WIN):
                return path + [ak]
            if ev == EVENT_GO:
                continue
            succ = model.successor(grid, g, m, ug, um)
            if succ is None:
                continue
            nh = fhash(succ, ui)
            if nh in seen:
                continue
            seen.add(nh)
            np_ = path + [ak]
            pen = 0.25 * sum(1 for _ in np_)  # mild depth penalty
            heapq.heappush(frontier, (len(np_) + pen, next(cnt), succ, np_))
    return None


def salient_clicks(grid: np.ndarray, cap=24) -> list[str]:
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(grid[y, x]) == bg:
                continue
            color = int(grid[y, x])
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            my = sum(c[0] for c in cells) // len(cells)
            mx = sum(c[1] for c in cells) // len(cells)
            ry, rx = (my, mx) if (my, mx) in set(cells) else cells[0]
            comps.append((len(cells), color, ry, rx))
    comps.sort()
    return [f"ACTION6:{x},{y}" for _, _, y, x in comps[:cap]]


# ---------------------------------------------------------------------------
# 6. The agent (single play per game): explore to learn rules, plan to the
# level goal when the model supports it, stop at WIN or time budget.
# ---------------------------------------------------------------------------
class TemplateRuleAgent:
    name = "template-rules"

    def __init__(self, game_id: str, seed: int = 0, time_budget_s: float = 240.0):
        self.game_id = game_id
        self.rng = random.Random(seed)
        self.store = ObsStore(game_id)
        self.regions = RegionFinder()
        self.model = Model()
        self.t0 = time.monotonic()
        self.deadline = self.t0 + time_budget_s
        self.bailout_frac = 0.6
        self._last_progress = self.t0
        self._max_level = 0
        self._pending = None
        self._plan: list[str] = []
        self._replan = True
        self._plan_cache: dict = {}
        self._new_obs = 0
        self._ctx: dict = {}
        self._step_n = 0

    # -- observation --------------------------------------------------------
    def _observe(self, latest: FrameDataRaw) -> None:
        if self._pending is None:
            return
        lv, pre, ak = self._pending
        self._pending = None
        if latest.frame is None or not len(latest.frame):
            return
        post = canon(latest.frame)
        status, o = self.store.add(lv, pre, ak, post, latest.levels_completed,
                                   latest.state.name)
        if status == "new" and o is not None:
            self._new_obs += 1
            self.regions.observe(o)
            self._ctx.setdefault((o.level, fhash(o.pre, self.model.ui_mask)),
                                 {})[o.action_key] = o
        if latest.levels_completed > self._max_level:
            self._max_level = latest.levels_completed
            self._last_progress = time.monotonic()
            self._plan, self._replan = [], True

    def _refresh(self, force=False) -> None:
        if not force and self._new_obs < 40:
            return
        self._new_obs = 0
        protected = {r.rule_id.split("->")[1].split("=")[0]
                     for r in self.model.rules
                     if r.rule_id.startswith("move_onto[") and r.status != St.CONTRADICTED}
        cells = self.regions.regions(protected_colors={int(p) for p in protected
                                                       if p.lstrip("-").isdigit()})
        ui = None
        if cells:
            ui = np.zeros(self.regions.shape, dtype=bool)
            for comp in cells:
                for y, x in comp:
                    ui[y, x] = True
        mask_changed = (ui is None) != (self.model.ui_mask is None) or (
            ui is not None and self.model.ui_mask is not None
            and not np.array_equal(ui, self.model.ui_mask))
        if mask_changed:
            self.model.ui_mask = ui
            self._ctx = {}
            for o in self.store.all():
                self._ctx.setdefault((o.level, fhash(o.pre, ui)), {})[o.action_key] = o
        rules = propose(self.store, self.model, cells)
        verify(rules, self.store)
        sig = tuple(sorted((r.rule_id, r.status.value) for r in rules))
        old = tuple(sorted((r.rule_id, r.status.value) for r in self.model.rules))
        self.model.rules = rules
        if sig != old or mask_changed:
            self.model.version += 1
            self._plan_cache.clear()
            self._replan = True

    # -- the public agent contract -----------------------------------------
    def is_done(self, frames, latest: FrameDataRaw) -> bool:
        self._observe(latest)
        now = time.monotonic()
        if latest.state == GameState.WIN:
            return True  # single play: a win finishes the game
        if now > self.deadline:
            return True
        if (now - self._last_progress) > self.bailout_frac * (self.deadline - self.t0):
            return True  # stuck: free the remaining wall-clock for other games
        return False

    def choose_action(self, frames, latest: FrameDataRaw):
        self._observe(latest)
        if latest.state == GameState.GAME_OVER:
            self._plan, self._replan = [], True
            return GameAction.RESET, None
        grid = canon(latest.frame)
        level = latest.levels_completed
        self._step_n += 1
        available = latest.available_actions or []
        simple = [_ID2ACT[a].name for a in available
                  if a in _ID2ACT and _ID2ACT[a] not in (GameAction.RESET, GameAction.ACTION6)]
        clicks = GameAction.ACTION6.value in available
        self._refresh(force=not self.model.rules)

        ak = None
        if self._plan:
            ak = self._plan.pop(0)
        elif self._replan:
            self._replan = False
            key = (level, fhash(grid, self.model.ui_mask))
            if key in self._plan_cache:
                plan = self._plan_cache[key]
            else:
                plan = plan_next(
                    self.model, level, grid, simple,
                    (lambda g: salient_clicks(g)) if clicks else (lambda g: []),
                    deadline=min(time.monotonic() + 2.0, self.deadline))
                self._plan_cache[key] = plan
            if plan:
                self._plan = list(plan)
                ak = self._plan.pop(0)

        if ak is None:  # explore: untried first, then anything informative
            key = (level, fhash(grid, self.model.ui_mask))
            tried = self._ctx.get(key, {})
            cands = list(simple) + (salient_clicks(grid) if clicks else [])
            unseen = [a for a in cands if a not in tried]
            if unseen:
                ak = unseen[self._step_n % len(unseen)]
            else:
                safe = [a for a, o in tried.items()
                        if a in cands and o.event != EVENT_GO]
                changers = [a for a in safe if tried[a].post_hash != tried[a].pre_hash]
                pool = changers or safe or cands
                ak = pool[self._step_n % len(pool)]

        base = ak.split(":")[0]
        if _NAME2ACT[base].value not in set(available) and base != "RESET":
            ak = simple[0] if simple else "RESET"  # never send unavailable actions
        self._pending = (level, grid, ak)
        if ":" in ak:
            x, y = (int(v) for v in ak.split(":")[1].split(","))
            return _NAME2ACT[base], {"x": x, "y": y}
        return _NAME2ACT[base], None


# ---------------------------------------------------------------------------
# 7. Reference baselines.
# ---------------------------------------------------------------------------
class RandomAgent:
    name = "random"

    def __init__(self, game_id, seed=0, time_budget_s=60.0):
        self.rng = random.Random(f"{game_id}:{seed}")
        self.deadline = time.monotonic() + time_budget_s

    def is_done(self, frames, latest):
        return latest.state == GameState.WIN or time.monotonic() > self.deadline

    def choose_action(self, frames, latest):
        if latest.state == GameState.GAME_OVER:
            return GameAction.RESET, None
        ids = [a for a in (latest.available_actions or [1, 2, 3, 4])
               if a in _ID2ACT and a != 0]
        a = _ID2ACT[self.rng.choice(ids)]
        if a == GameAction.ACTION6:
            return a, {"x": self.rng.randrange(64), "y": self.rng.randrange(64)}
        return a, None


class SweepAgent:
    """Tries each available action in turn, round-robin."""
    name = "sweep"

    def __init__(self, game_id, seed=0, time_budget_s=60.0):
        self.i = 0
        self.deadline = time.monotonic() + time_budget_s

    def is_done(self, frames, latest):
        return latest.state == GameState.WIN or time.monotonic() > self.deadline

    def choose_action(self, frames, latest):
        if latest.state == GameState.GAME_OVER:
            return GameAction.RESET, None
        ids = [a for a in (latest.available_actions or [1, 2, 3, 4])
               if a in _ID2ACT and a != 0]
        self.i += 1
        a = _ID2ACT[ids[self.i % len(ids)]]
        if a == GameAction.ACTION6:
            return a, {"x": (self.i * 7) % 64, "y": (self.i * 13) % 64}
        return a, None


# ---------------------------------------------------------------------------
# 8. Main: play every available game once with the template-rule agent.
# ---------------------------------------------------------------------------
def play_game(arcade, card_id, game_id, agent) -> dict:
    env = arcade.make(game_id, scorecard_id=card_id)
    if env is None:
        return {"game": game_id, "error": "make failed"}
    fd = env.observation_space
    frames = [fd]
    steps = 0
    while fd is not None and not agent.is_done(frames, fd) and steps < 200_000:
        action, data = agent.choose_action(frames, fd)
        fd = env.reset() if action == GameAction.RESET else env.step(action, data)
        if fd is not None:
            frames.append(fd)
        steps += 1
    return {"game": game_id,
            "levels": fd.levels_completed if fd is not None else 0,
            "state": fd.state.name if fd is not None else "?",
            "steps": steps}


def main(per_game_seconds: float = 240.0) -> None:
    env_dir = os.environ.get("ENVIRONMENTS_DIR")
    if env_dir is None:
        for root, dirs, files in os.walk("/kaggle/input"):
            if "metadata.json" in files:
                env_dir = str(os.path.dirname(os.path.dirname(root)))
                break
    mode = os.environ.get("OPERATION_MODE", "offline")
    arcade = Arcade(operation_mode=OperationMode(mode),
                    environments_dir=env_dir or "environment_files")
    games = sorted({e.game_id.split("-")[0] for e in arcade.get_environments()})
    print(f"{len(games)} games discovered: {games}")
    card = arcade.open_scorecard(tags=["agent"])
    results = []
    for g in games:
        agent = TemplateRuleAgent(g, time_budget_s=per_game_seconds)
        r = play_game(arcade, card, g, agent)
        results.append(r)
        print(json.dumps(r))
    sc = arcade.close_scorecard(card)
    if sc is not None:
        print(f"\nscorecard: score={sc.score:.2f}")
        print(sc.model_dump_json(indent=1)[:2000])
    print(json.dumps({"summary": results}, indent=1))


if __name__ == "__main__":
    main(per_game_seconds=float(os.environ.get("PER_GAME_SECONDS", "240")))
