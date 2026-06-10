%%writefile /kaggle/working/my_agent.py
# =====================================================================
# FORGE v21  --  ARC-AGI-3 agent (ARC Prize 2026, Kaggle harness track)
#
# WHY THIS REWRITE, GROUNDED IN THE OFFICIAL SCORING
# --------------------------------------------------
# Scoring is RHAE. Per completed level:
#       level_score = (human_baseline_actions / agent_actions) ** 2
# capped at ~1.15x human (so the per-level ceiling is ~1.32), then a
# weighted average per game using the 1-indexed level number as the
# weight (later levels matter much more), then averaged across games.
# There is a HARD CUTOFF at 5x the human action count per level: if the
# agent has not finished a level within 5x the human's actions, it is
# cut off and scores 0 for that level.
#
# Two consequences drive every design choice here:
#   1. Completion is gated by a tight budget. Exhaustive breadth-first
#      flailing (the old v20 behaviour) blows past 5x human on anything
#      non-trivial and scores 0. Exploration must be FRUGAL.
#   2. Because the term is squared, every wasted action hurts a lot.
#      Halving the action count roughly quadruples the level score.
#
# WHAT ACTUALLY WORKED IN THE PREVIEW (and what we copy, cheaply):
#   - StochasticGoose (1st): a model predicting which actions change the
#     frame, used to BIAS exploration toward productive actions. We
#     reproduce the *idea* with a training-free running tally instead of
#     a CNN: per action-type, per clicked-colour, and a coarse spatial
#     click heatmap. No torch, no GPU, instant start, no weight loading.
#   - Blind Squirrel (2nd): a directed STATE GRAPH that prunes actions
#     creating loops or no change. We keep the graph, prune no-ops and
#     deadly actions, and add proper return-then-explore.
#
# CORE LOOP: a frugal, informed Go-Explore graph explorer.
#   * Robust state hashing: a per-level FROZEN mask removes volatile
#     border bands (step counters / score read-outs) so two genuinely
#     identical play states hash the same. The mask is frozen per level
#     so hashes never shift mid-level and corrupt the graph (a real bug
#     in v20, which recomputed the mask every 32 ticks).
#   * Segment each frame into connected components; propose a small,
#     prioritised candidate set: simple keys, then clicks on salient
#     objects (snapped to a real component cell), then coarse background
#     clicks only as a last resort.
#   * Order untested candidates by the effect model, so we try the
#     action most likely to do something useful FIRST. This is the main
#     lever on efficiency.
#   * Prefer the nearest reachable frontier without resetting. Only when
#     a whole reachable region is exhausted do we RESET and replay the
#     shortest stored path to the shallowest remaining frontier (the
#     deterministic engine makes replay exact).
#   * Every issued action is recorded TESTED with its outcome, so a
#     reset/death edge can never be re-selected forever (the loop bug
#     that cost the Helsinki team places). Deaths are attributed to the
#     action that caused them and recorded BEFORE the recovery reset.
#   * The effect model and learned action vocabulary CARRY ACROSS LEVELS
#     within a game. Mechanics tend to persist between levels, and the
#     scoring overweights later levels, so this is a real, legitimate
#     edge that compounds exactly where the points are.
#
# REMOVED: torch (heavy, GPU failure modes, never helped a from-scratch
# net learn a sparse single-reward level inside one budget), and the
# offline engine-introspection planner from v20. The official eval is
# sandboxed with no internet and the games are hardened against
# brute-force; the engine will not be importable, so that path returns
# nothing on the real set. It is also against the stated spirit of the
# competition (prize-eligible work is screened for it) and is dead weight
# and a disqualification risk. The legitimate explorer is the whole
# strategy, so that is all this file contains.
# =====================================================================
import hashlib
import logging
import random
import time
from collections import deque

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


# ==================== FRAME UTILITIES ====================

def background_colour(grid):
    return int(np.bincount(grid.reshape(-1), minlength=16).argmax())


def connected_components(grid, bg):
    """4-connected single-colour components of all non-background cells.

    Each component carries its colour, size, centroid, bounding box and a
    representative cell (rx, ry) that is guaranteed to lie inside the
    component and is the member nearest the centroid, so a click always
    lands on the object rather than on a hole in a concave shape.
    O(H*W); trivial on a 64x64 grid."""
    H, W = grid.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    comps = []
    cur = 0
    for i in range(H):
        row = grid[i]
        for j in range(W):
            if row[j] == bg or labels[i, j] != -1:
                continue
            colour = int(row[j])
            stack = [(i, j)]
            labels[i, j] = cur
            ys, xs = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y); xs.append(x)
                if y + 1 < H and labels[y + 1, x] == -1 and grid[y + 1, x] == colour:
                    labels[y + 1, x] = cur; stack.append((y + 1, x))
                if y - 1 >= 0 and labels[y - 1, x] == -1 and grid[y - 1, x] == colour:
                    labels[y - 1, x] = cur; stack.append((y - 1, x))
                if x + 1 < W and labels[y, x + 1] == -1 and grid[y, x + 1] == colour:
                    labels[y, x + 1] = cur; stack.append((y, x + 1))
                if x - 1 >= 0 and labels[y, x - 1] == -1 and grid[y, x - 1] == colour:
                    labels[y, x - 1] = cur; stack.append((y, x - 1))
            ys = np.asarray(ys); xs = np.asarray(xs)
            cy, cx = float(ys.mean()), float(xs.mean())
            # representative cell: member nearest the centroid
            k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            comps.append({
                'color': colour, 'size': int(ys.size),
                'cy': cy, 'cx': cx,
                'ry': int(ys[k]), 'rx': int(xs[k]),
                'y0': int(ys.min()), 'y1': int(ys.max()),
                'x0': int(xs.min()), 'x1': int(xs.max()),
            })
            cur += 1
    return comps


# ==================== EFFECT MODEL (training-free, cross-level) ====================

class EffectModel:
    """Running estimate of which actions tend to change the frame.

    This is the cheap stand-in for StochasticGoose's CNN. It is updated
    after every real transition and persists across levels of one game,
    because game mechanics usually carry over and the scoring overweights
    later levels. All estimates use Laplace smoothing so an untried
    action still has a sensible, optimistic prior."""

    def __init__(s):
        s.simple = {a: [1.0, 2.0] for a in range(1, 6)}   # id -> [changes, tries]
        s.color = {}                                       # colour -> [changes, tries]
        s.heat = np.ones((8, 8, 2), dtype=np.float64)      # coarse click-success grid
        s.heat[:, :, 1] += 1.0
        s.adv_keys = set()      # action-types that have ever advanced a level
        s.adv_colors = set()    # clicked colours that have ever advanced a level
        s.move_actions = set()  # simple actions that translate a small object
        s.move_votes = {}       # action -> {(dy,dx): count}
        s.avatar_color = None
        s.characterised = set() # simple actions tried at least a few times

    @staticmethod
    def _rate(pair):
        return pair[0] / pair[1] if pair[1] else 0.5

    def p_simple(s, a):
        return s._rate(s.simple.get(a, [1.0, 2.0]))

    def p_color(s, col):
        return s._rate(s.color.get(col, [1.0, 2.0]))

    def p_heat(s, x, y):
        gx, gy = min(7, x * 8 // 64), min(7, y * 8 // 64)
        cell = s.heat[gy, gx]
        return cell[0] / cell[1]

    def update_simple(s, a, changed, advanced=False):
        p = s.simple.setdefault(a, [1.0, 2.0])
        p[0] += 1.0 if changed else 0.0
        p[1] += 1.0
        if p[1] >= 4.0:
            s.characterised.add(a)
        if advanced:
            s.adv_keys.add(a)

    def update_click(s, col, x, y, changed, advanced=False):
        p = s.color.setdefault(col, [1.0, 2.0])
        p[0] += 1.0 if changed else 0.0
        p[1] += 1.0
        gx, gy = min(7, x * 8 // 64), min(7, y * 8 // 64)
        s.heat[gy, gx, 1] += 1.0
        s.heat[gy, gx, 0] += 1.0 if changed else 0.0
        if advanced:
            s.adv_colors.add(col)

    def learn_move(s, action, prev_grid, grid, bg):
        """If a simple action translated one small object, vote a
        displacement for it. Purely advisory: it only nudges priority
        toward movement actions, so a wrong guess is never fatal."""
        try:
            diff = prev_grid != grid
            n = int(diff.sum())
            if n == 0 or n > 28:
                return
            appeared = diff & (prev_grid == bg) & (grid != bg)
            vacated = diff & (prev_grid != bg) & (grid == bg)
            if appeared.sum() == 0 or vacated.sum() == 0:
                return
            ay, ax = np.where(appeared)
            vy, vx = np.where(vacated)
            dy = int(round(ay.mean() - vy.mean()))
            dx = int(round(ax.mean() - vx.mean()))
            if dy == 0 and dx == 0:
                return
            votes = s.move_votes.setdefault(action, {})
            votes[(dy, dx)] = votes.get((dy, dx), 0) + 1
            if votes[(dy, dx)] >= 2:
                s.move_actions.add(action)
                cols = grid[appeared]
                if cols.size:
                    s.avatar_color = int(np.bincount(cols, minlength=16).argmax())
        except Exception:
            pass

    def p_key(s, key, grid):
        """Expected usefulness of a candidate action, higher is better."""
        H, W = grid.shape
        if isinstance(key, tuple):
            _, x, y = key
            col = int(grid[y, x]) if (0 <= y < H and 0 <= x < W) else -1
            p = 0.5 * s.p_color(col) + 0.5 * s.p_heat(x, y)
            if col in s.adv_colors:
                p += 0.6
            return p
        p = s.p_simple(key)
        if key in s.adv_keys:
            p += 0.6
        if key in s.move_actions:
            p += 0.25
        return p


# ==================== PER-LEVEL STATE GRAPH ====================

class LevelGraph:
    """Directed graph of observed states for a single level.

    nodes[h] = {'path': shortest action-key walk from root,
                'proposer': ordered [(tier, key)] candidates,
                'actions': {key: (kind, succ_hash)} for tested keys,
                'dense': whether coarse-click escalation was applied}
    adj[h]   = [(key, succ_hash)] real transitions."""

    MAX_NODES = 6000  # guard against a degenerate hash exploding memory

    def __init__(s):
        s.nodes = {}
        s.adj = {}
        s.root = None

    def add(s, h, path, proposer):
        if h not in s.nodes and len(s.nodes) < s.MAX_NODES:
            s.nodes[h] = {'path': list(path), 'proposer': proposer,
                          'actions': {}, 'dense': False}
            s.adj.setdefault(h, [])

    def untested(s, h):
        node = s.nodes.get(h)
        if not node:
            return []
        tested = node['actions']
        return [(t, k) for (t, k) in node['proposer'] if k not in tested]

    def shallowest_frontier(s):
        best, best_len = None, 1 << 30
        for h, node in s.nodes.items():
            if len(node['path']) < best_len and s.untested(h):
                best, best_len = h, len(node['path'])
        return best


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10
    _SIMPLE = (1, 2, 3, 4, 5)

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time() * 1e6) + (hash(s.game_id) % 1000000)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        s.start_time = time.time()

        s.level = -1
        s.graphs = {}
        s.cur = None                 # (hash, path) of the node we are at
        s.pending = None             # (hash, path, key, prev_grid) last issued action
        s.plan = deque()             # queued keys for navigation / replay
        s._await_root = False        # re-anchor root on the next observed frame

        # volatility tracking, accumulated across the whole game so the
        # status-bar mask is good from level 2 onward with zero warmup
        s._prev_grid = None
        s._change = np.zeros((64, 64), dtype=np.int32)
        s._obs = 0
        s._level_mask = None         # FROZEN per level once warmup completes
        s._mask_ready = False        # has a usable game-wide mask been built
        s._calibrated = False        # have we sampled the action vocabulary

        s.effect = EffectModel()

        # goal-directed navigation state (reset each level). When an
        # avatar and its movement vectors are known we plan on the grid
        # toward salient objects instead of blindly exploring states.
        s._passable = set()        # (y,x) cells the avatar has occupied
        s._blocked_moves = set()   # (y,x,action) that produced no movement
        s._exhausted_targets = set()  # (y,x) targets visited without a win
        s._gs_target = None        # current target cell (y,x)
        s._gs_fail = {}            # target -> consecutive BFS failures
        s._last_avatar = None      # last known avatar cell (y,x)

    # --- harness plumbing (kept identical to the framework contract) ---
    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def is_done(s, frames, lf):
        try:
            return lf.state is GameState.WIN or (time.time() - s.start_time) >= 8 * 3600 - 300
        except Exception:
            return True

    # --- small helpers ---
    def _lvl(s, f):
        return getattr(f, 'score', None) or getattr(f, 'levels_completed', 0)

    def _grid(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _avail_ids(s, lf):
        out = set()
        for a in (getattr(lf, 'available_actions', None) or []):
            out.add(a.value if hasattr(a, 'value') else int(a))
        if not out:
            out = {1, 2, 3, 4, 5, 6}
        return out

    def _update_vol(s, grid):
        if s._prev_grid is not None and s._prev_grid.shape == grid.shape:
            s._change += (grid != s._prev_grid)
            s._obs += 1
        s._prev_grid = grid.copy()

    def _build_mask(s, grid):
        """Mask thin, highly volatile border bands (step counter, score /
        level read-outs) so the hash tracks the play area. Conservative:
        only border bands, and never masks away more than half the grid."""
        H, W = grid.shape
        mask = np.ones((H, W), dtype=bool)
        if s._obs < 3:
            return mask
        freq = s._change[:H, :W] / max(s._obs, 1)
        row_vol = freq.mean(axis=1)
        col_vol = freq.mean(axis=0)
        band = 8
        # whole volatile bands (a full-width status strip)
        for r in range(H):
            if (r < band or r >= H - band) and row_vol[r] > 0.5:
                mask[r, :] = False
        for c in range(W):
            if (c < band or c >= W - band) and col_vol[c] > 0.5:
                mask[:, c] = False
        # individual flickering cells inside the border region (a small
        # corner step counter / score read-out that no band test catches).
        # Restricted to the border so a frequently moving avatar in the
        # play area is never masked away.
        border = np.zeros((H, W), dtype=bool)
        border[:band, :] = True; border[-band:, :] = True
        border[:, :band] = True; border[:, -band:] = True
        mask &= ~(border & (freq > 0.5))
        if mask.sum() < 0.5 * H * W:
            return np.ones((H, W), dtype=bool)
        return mask

    def _hash(s, grid):
        m = s._level_mask
        if m is not None and m.shape == grid.shape:
            g = np.where(m, grid, -1)
        else:
            g = grid
        return hashlib.md5(g.astype(np.int16).tobytes()).hexdigest()[:20]

    # --- action proposal (segmentation + priority tiers) ---
    def _propose(s, grid, lf, dense=False):
        avail = s._avail_ids(lf)
        bg = background_colour(grid)
        out = {}

        def add(tier, key):
            if key not in out or tier < out[key]:
                out[key] = tier

        # tier 0: simple keys (cheap, and often the whole vocabulary)
        for a in s._SIMPLE:
            if a in avail:
                add(0, a)

        # clicks: one per component, snapped to a real member cell, tiered
        # by salience (small interactive-looking objects first)
        if 6 in avail:
            for c in connected_components(grid, bg):
                x, y = c['rx'], c['ry']
                if s._level_mask is not None and not s._level_mask[y, x]:
                    continue  # inside the masked status bar
                size = c['size']
                if size <= 6:
                    base = 1
                elif size <= 40:
                    base = 2
                elif size <= 200:
                    base = 3
                else:
                    base = 4
                add(base, ('c', x, y))
            # coarse background / empty-space clicks, lowest priority
            stride = 4 if dense else 12
            for yy in range(stride // 2, 64, stride):
                for xx in range(stride // 2, 64, stride):
                    if s._level_mask is None or s._level_mask[yy, xx]:
                        add(5, ('c', xx, yy))

        # undo: lowest priority, rarely worth a real action
        if 7 in avail:
            add(6, 7)

        return sorted(((t, k) for k, t in out.items()), key=lambda tk: tk[0])

    def _ensure_node(s, G, h, grid, path, lf, dense=False):
        if h not in G.nodes:
            G.add(h, path, s._propose(grid, lf, dense=dense))
        elif dense and not G.nodes[h]['dense']:
            G.nodes[h]['proposer'] = s._propose(grid, lf, dense=True)
            G.nodes[h]['dense'] = True

    # --- transition recording ---
    def _record(s, G, grid, lf, h, lvl):
        """Graph-only recording. Effect-model learning happens separately
        and unconditionally, so the movement model builds even during the
        calibration phase before the graph exists."""
        ph, ppath, pk, pgrid = s.pending
        node = G.nodes.get(ph)
        changed = (h != ph)
        advanced = lvl > s.level
        if node is None:
            return
        if advanced:
            node['actions'][pk] = ('advance', h)
            return
        if not changed:
            node['actions'][pk] = ('noop', h)
            return
        node['actions'][pk] = ('edge', h)
        G.adj.setdefault(ph, []).append((pk, h))
        if h not in G.nodes:
            G.add(h, ppath + [pk], s._propose(grid, lf))
        elif len(ppath) + 1 < len(G.nodes[h]['path']):
            G.nodes[h]['path'] = ppath + [pk]

    def _learn(s, pk, pgrid, grid, changed, advanced):
        try:
            if isinstance(pk, tuple):
                _, x, y = pk
                col = int(pgrid[y, x]) if pgrid is not None else -1
                s.effect.update_click(col, x, y, changed, advanced)
            elif isinstance(pk, int):
                s.effect.update_simple(pk, changed, advanced)
                if pgrid is not None:
                    bg = background_colour(pgrid)
                    if changed:
                        s.effect.learn_move(pk, pgrid, grid, bg)
                    s._track_avatar(pk, pgrid, grid)
        except Exception:
            pass

    def _play_grid(s, grid):
        """Grid with the masked status region blanked to background, so
        segmentation for navigation ignores counters and score read-outs
        (which can share colours with the avatar or with real targets)."""
        m = s._level_mask
        if m is None or m.shape != grid.shape:
            return grid
        bg = background_colour(grid)
        pg = grid.copy()
        pg[~m] = bg
        return pg

    def _avatar_cell(s, grid):
        """Representative cell of the avatar-colour component, or None."""
        if s.effect.avatar_color is None:
            return None
        try:
            pg = s._play_grid(grid)
            bg = background_colour(pg)
            comps = [c for c in connected_components(pg, bg)
                     if c['color'] == s.effect.avatar_color]
            if not comps:
                return None
            if s._last_avatar is not None:
                ly, lx = s._last_avatar
                comps.sort(key=lambda c: (c['size'],
                                          abs(c['cy'] - ly) + abs(c['cx'] - lx)))
            else:
                comps.sort(key=lambda c: c['size'])
            return (comps[0]['ry'], comps[0]['rx'])
        except Exception:
            return None

    def _track_avatar(s, action, pgrid, grid):
        """Update the passable map and blocked-move set from a move."""
        before = s._avatar_cell(pgrid)
        after = s._avatar_cell(grid)
        if before is None:
            return
        if after is not None and after != before:
            s._passable.add(after)
            s._passable.add(before)
        elif after == before or after is None:
            # the avatar did not move: this direction is blocked here
            s._blocked_moves.add((before[0], before[1], action))

    # --- navigation: BFS over known edges to the nearest frontier ---
    def _bfs_to_frontier(s, G, start):
        if G.untested(start):
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u != start and G.untested(u):
                path, cur = [], u
                while prev[cur] is not None:
                    pu, pk = prev[cur]
                    path.append(pk); cur = pu
                return list(reversed(path))
            for (k, v) in G.adj.get(u, []):
                if v not in prev:
                    prev[v] = (u, k); q.append(v)
        return None

    # --- goal-directed navigation (movement games) ---
    def _movement_vectors(s):
        """Confident per-action displacement vectors for the avatar."""
        vecs = {}
        for a, votes in s.effect.move_votes.items():
            if not votes:
                continue
            (dy, dx), cnt = max(votes.items(), key=lambda kv: kv[1])
            if cnt >= 2 and (dy, dx) != (0, 0):
                vecs[a] = (dy, dx)
        return vecs

    def _pick_target(s, grid, avatar):
        """Most salient non-avatar object to head for: rare colours and
        small objects first, then proximity. Exhausted targets skipped."""
        try:
            pg = s._play_grid(grid)
            bg = background_colour(pg)
            comps = connected_components(pg, bg)
            counts = {}
            for c in comps:
                counts[c['color']] = counts.get(c['color'], 0) + 1
            ay, ax = avatar
            cands = []
            for c in comps:
                if c['color'] == s.effect.avatar_color:
                    continue
                cell = (c['ry'], c['rx'])
                if cell in s._exhausted_targets:
                    continue
                cands.append((counts[c['color']], c['size'],
                              abs(c['cy'] - ay) + abs(c['cx'] - ax), cell))
            if not cands:
                return None
            cands.sort()
            return cands[0][3]
        except Exception:
            return None

    def _bfs_grid(s, grid, start, goal, vecs):
        """Shortest action path from avatar `start` to `goal` over the
        visible grid, treating known non-background cells as obstacles and
        cells the avatar has already occupied as passable. Returns the
        action list, or None if no path is currently believed to exist."""
        grid = s._play_grid(grid)
        bg = background_colour(grid)
        H, W = grid.shape
        prev = {start: None}
        q = deque([start])
        moves = list(vecs.items())
        while q:
            u = q.popleft()
            if u == goal:
                path, cur = [], u
                while prev[cur] is not None:
                    pu, pa = prev[cur]
                    path.append(pa); cur = pu
                return list(reversed(path))
            for a, (dy, dx) in moves:
                ny, nx = u[0] + dy, u[1] + dx
                v = (ny, nx)
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if (u[0], u[1], a) in s._blocked_moves:
                    continue
                passable = (grid[ny, nx] == bg) or v == goal or v in s._passable
                if not passable or v in prev:
                    continue
                prev[v] = (u, a); q.append(v)
        return None

    def _goal_seek(s, grid):
        """Return the next movement action toward a salient target, or
        None to defer to the graph explorer (e.g. no avatar = click or
        transform puzzle, where this never activates)."""
        vecs = s._movement_vectors()
        if len(vecs) < 2:
            return None
        avatar = s._avatar_cell(grid)
        if avatar is None:
            return None
        s._last_avatar = avatar
        # if we are standing on the previous target and the level did not
        # advance, that target was a dead end: retire it and move on
        if s._gs_target is not None and avatar == s._gs_target:
            s._exhausted_targets.add(s._gs_target)
            s._gs_target = None
        target = s._pick_target(grid, avatar)
        if target is None:
            return None
        s._gs_target = target
        path = s._bfs_grid(grid, avatar, target, vecs)
        if not path:
            # A failure here often just means we have not yet learned a
            # move on the axis we need, not that the target is truly
            # unreachable. Only retire it once we can move on both axes
            # and have still failed to reach it several times; otherwise
            # defer one step, learn more moves, and try again.
            both_axes = (any(dy != 0 for dy, dx in vecs.values()) and
                         any(dx != 0 for dy, dx in vecs.values()))
            s._gs_fail[target] = s._gs_fail.get(target, 0) + 1
            if both_axes and s._gs_fail[target] >= 3:
                s._exhausted_targets.add(target)
            s._gs_target = None
            return None
        s._gs_fail.pop(target, None)
        return path[0]

    # --- choosing the next action at node h ---
    def _next_action(s, G, h, grid, lf):
        if s.plan:
            return s.plan.popleft()

        # goal-directed move toward a salient object, if we have an
        # avatar and a movement model. Returns None on non-movement games.
        gs = s._goal_seek(grid)
        if gs is not None:
            return gs

        unt = G.untested(h)
        if unt:
            # informed choice: blend the static tier with the effect
            # model so the action most likely to do something useful is
            # tried first. score is "cost", lower is better.
            best, best_score = None, 1e9
            for (t, k) in unt:
                score = t - 1.5 * s.effect.p_key(k, grid) + random.uniform(0, 0.05)
                if score < best_score:
                    best, best_score = k, score
            return best

        # nothing untested here: walk known edges to the nearest frontier
        path = s._bfs_to_frontier(G, h)
        if path:
            s.plan = deque(path)
            return s.plan.popleft()

        # reachable region exhausted: reset and replay to the shallowest
        # remaining frontier anywhere in the graph (engine is deterministic)
        target = G.shallowest_frontier()
        if target is not None and target != h and G.nodes[target]['path']:
            s._await_root = True
            s.plan = deque(['RESET'] + list(G.nodes[target]['path']))
            return s.plan.popleft()

        # escalate click density once before giving up on this region
        if h in G.nodes and not G.nodes[h]['dense']:
            s._ensure_node(G, h, grid, G.nodes[h]['path'], lf, dense=True)
            unt = G.untested(h)
            if unt:
                best, best_score = None, 1e9
                for (t, k) in unt:
                    score = t - 1.5 * s.effect.p_key(k, grid) + random.uniform(0, 0.05)
                    if score < best_score:
                        best, best_score = k, score
                return best

        # last resort: informed-random among available actions
        avail = list(s._avail_ids(lf))
        simple = [a for a in avail if 1 <= a <= 5]
        if simple:
            simple.sort(key=lambda a: -s.effect.p_simple(a))
            return simple[0] if random.random() < 0.7 else random.choice(simple)
        if 6 in avail:
            return ('c', random.randint(0, 63), random.randint(0, 63))
        return ('RESET',)

    # --- calibration: sample the action vocabulary on the first level ---
    def _calibration_action(s, lf):
        """Round-robin over the available simple actions until each has
        been tried a few times, enough to learn confident movement
        vectors (two consistent votes) and to reveal the status bar in
        the volatility map. Returns an action id, or None when complete.
        Runs once per game: the effect model then carries forward to every
        later level for free, which is where the weighted score sits."""
        avail = [a for a in s._SIMPLE if a in s._avail_ids(lf)]
        if not avail:
            return None
        tries = {a: int(s.effect.simple.get(a, [1.0, 2.0])[1] - 2) for a in avail}
        if sum(tries.values()) >= 4 * len(avail) + 4:
            return None  # safety cap on calibration length
        if min(tries.values()) >= 3:
            return None  # every action sampled enough
        return min(avail, key=lambda a: (tries[a], a))

    # --- key -> GameAction ---
    def _mk(s, action, reason):
        action.reasoning = reason
        return action

    def _mk_key(s, key):
        if key == ('RESET',):
            return s._mk(GameAction.RESET, "nav:reset")
        if isinstance(key, tuple):
            _, x, y = key
            a = GameAction.ACTION6
            a.set_data({"x": int(x), "y": int(y)})
            return s._mk(a, f"click({x},{y})")
        a = GameAction.from_id(int(key))
        return s._mk(a, f"key{key}")

    # --- level entry ---
    def _enter_level(s, lvl, grid, h, lf):
        s.level = lvl
        if lvl not in s.graphs:
            s.graphs[lvl] = LevelGraph()
        G = s.graphs[lvl]

        # freeze the per-level mask. Once the game has been calibrated a
        # good game-wide mask already exists, so we commit immediately.
        # On the very first level we hold off until calibration has seen
        # enough frames for the status bar to reveal itself.
        if s._mask_ready:
            s._level_mask = s._build_mask(grid)
        else:
            s._level_mask = None

        # new level, fresh navigation memory
        s._passable = set()
        s._blocked_moves = set()
        s._exhausted_targets = set()
        s._gs_target = None
        s._gs_fail = {}
        s._last_avatar = None

        h = s._hash(grid)
        G.root = h
        s._ensure_node(G, h, grid, [], lf)
        s.cur = (h, [])
        s.plan.clear()
        s._await_root = False
        return h

    # --- main loop ---
    def choose_action(s, frames, lf):
        try:
            grid = s._grid(lf)
            s._update_vol(grid)

            state = lf.state
            lvl = s._lvl(lf)

            # reset / game-over: attribute a death to the action that
            # caused it (so it is never retried), then restart cleanly.
            if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                if state is GameState.GAME_OVER and s.pending is not None \
                        and s.level in s.graphs:
                    ph, ppath, pk, pgrid = s.pending
                    node = s.graphs[s.level].nodes.get(ph)
                    if node is not None and pk not in node['actions']:
                        node['actions'][pk] = ('death', None)
                    s._learn(pk, pgrid, grid, True, False)
                    # never route toward a cell that just killed us
                    if s._gs_target is not None:
                        s._exhausted_targets.add(s._gs_target)
                        s._gs_target = None
                s.pending = None
                s.plan.clear()
                s._await_root = True
                return s._mk(GameAction.RESET, "reset")

            h = s._hash(grid)

            # learn from the action issued last turn. The effect model is
            # updated unconditionally so movement vectors form during
            # calibration; the graph is only written once calibrated.
            if s.pending is not None and not s._await_root:
                ph, ppath, pk, pgrid = s.pending
                advanced = lvl > s.level
                changed = (h != ph)
                s._learn(pk, pgrid, grid, changed or advanced, advanced)
                if s._calibrated and s.level in s.graphs:
                    s._record(s.graphs[s.level], grid, lf, h, lvl)

            # level transition (advance edge recorded just above)
            if lvl != s.level or s.level not in s.graphs:
                if not s._mask_ready and s._obs >= 6:
                    s._mask_ready = True
                h = s._enter_level(lvl, grid, h, lf)

            G = s.graphs[s.level]

            # re-anchor root after a reset (deterministic, so this is root)
            if G.root is None or s._await_root:
                G.root = h
                s._await_root = False
            s._ensure_node(G, h, grid,
                           [] if h == G.root else (s.cur[1] if s.cur else []), lf)
            s.cur = (h, G.nodes[h]['path'] if h in G.nodes else [])

            # first-game calibration: sample the action vocabulary so the
            # movement model is confident before we trust the planner.
            if not s._calibrated:
                if not s._mask_ready and s._obs >= 6:
                    # the status bar has now shown itself: freeze the mask
                    # and re-anchor the root to the masked hash
                    s._mask_ready = True
                    s._level_mask = s._build_mask(grid)
                    h = s._hash(grid)
                    G.root = h
                    s._ensure_node(G, h, grid, [], lf)
                    s.cur = (h, [])
                cal = s._calibration_action(lf)
                if cal is not None:
                    s.pending = (h, [], cal, grid.copy())
                    return s._mk_key(cal)
                # calibration complete: commit the mask, anchor root, and
                # hand over to the planner / graph explorer
                s._calibrated = True
                s._mask_ready = True
                s._level_mask = s._build_mask(grid)
                h = s._hash(grid)
                G.root = h
                s._ensure_node(G, h, grid, [], lf)
                s.cur = (h, [])

            key = s._next_action(G, h, grid, lf)
            path = list(G.nodes[h]['path']) if h in G.nodes else []
            s.pending = (h, path, key, grid.copy())
            return s._mk_key(key)

        except Exception as e:
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            return s._mk(GameAction.from_id(random.choice(s._SIMPLE)), f"err:{e}")