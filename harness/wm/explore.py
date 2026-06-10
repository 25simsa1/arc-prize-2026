"""Exploration substrate: segmentation, salience tiers, segment-granularity
novelty, and a Go-Explore archive. No LLM anywhere in this layer.

Motivation (cited at each mechanism):
  - Tier-escalating click coverage: Rudakov 2512.24156 (salience-stratified
    candidate tiers + frontier return) — fixes the INERT-START pair
    (ft09/lp85: 47k actions, 24 unique transitions under the old cap-24
    salience generator that never found the live control).
  - Segment-granularity visit counts: #Exploration 1611.04717 (hashed-state
    counts). We hash at COMPONENT level, not whole-frame, so cosmetic
    diffs (HUD tickers) don't drown event-bearing sub-changes.
  - Go-Explore archive + return-via-replay: 1901.10995 / 2004.12919. The
    engine is deterministic, so returning to a state = exact replay of its
    recorded action prefix.

Budget note that shapes the archive: under the eval-realistic 5x per-level
cutoff (tech report 2603.24621; see results/cap_study/REPORT.md), returning
to a state COSTS prefix-length actions out of a 30-2890 window, so prefix
length is a first-class ranking feature and the capped policy prefers
short-prefix cells. Uncapped, return is action-free and prefix length only
breaks ties.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# salience tiers by component size (small interactive-looking objects first,
# large background masses last), mirroring the FORGE explorer's tiering.
_TIER_SIMPLE = 0      # ACTION1..5 — cheapest, often the whole vocabulary
_TIER_SMALL = 1       # <=6 cells
_TIER_MED = 2         # <=40
_TIER_BIG = 3         # <=200
_TIER_HUGE = 4        # >200
_TIER_REFINE = 5      # neighborhood refine of a productive click
_TIER_LATTICE = 6     # coarse-grid floor: every region gets one probe
TIER_NAMES = {0: "simple", 1: "seg_small", 2: "seg_med", 3: "seg_big",
              4: "seg_huge", 5: "refine", 6: "lattice"}


def _tier_for_size(n: int) -> int:
    if n <= 6:
        return _TIER_SMALL
    if n <= 40:
        return _TIER_MED
    if n <= 200:
        return _TIER_BIG
    return _TIER_HUGE


@dataclass
class Component:
    color: int
    size: int
    cells: list[tuple[int, int]]
    cy: float
    cx: float
    ry: int          # representative cell (member nearest centroid)
    rx: int
    sig: str         # translation-invariant signature (color + shape)


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def components(grid: np.ndarray, bg: Optional[int] = None) -> list[Component]:
    """4-connected single-color components of non-background cells.
    Representative cell is the member nearest the centroid (never a hole in a
    concave shape), so a click always lands on the object."""
    if bg is None:
        bg = background_color(grid)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[Component] = []
    for y in range(h):
        row = grid[y]
        for x in range(w):
            if seen[y, x] or int(row[x]) == bg:
                continue
            color = int(row[x])
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                            and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = np.fromiter((c[0] for c in cells), dtype=np.int32, count=len(cells))
            xs = np.fromiter((c[1] for c in cells), dtype=np.int32, count=len(cells))
            cyf, cxf = float(ys.mean()), float(xs.mean())
            k = int(np.argmin((ys - cyf) ** 2 + (xs - cxf) ** 2))
            y0, x0 = int(ys.min()), int(xs.min())
            # translation-invariant signature: color + bbox dims + relative
            # cell offsets (so "the same object anywhere" counts as one kind)
            rel = sorted((int(cy_ - y0), int(cx_ - x0)) for cy_, cx_ in cells)
            sig = hashlib.sha1(
                f"{color}:{int(ys.max())-y0}:{int(xs.max())-x0}:{rel}".encode()
            ).hexdigest()[:12]
            out.append(Component(color, len(cells), cells, cyf, cxf,
                                 int(ys[k]), int(xs[k]), sig))
    return out


def tiered_click_candidates(grid: np.ndarray,
                            hud_mask: Optional[np.ndarray] = None,
                            lattice_step: int = 8) -> list[tuple[int, str]]:
    """All click candidates as (tier, action_key), salience-ascending then a
    coarse-lattice floor. NOT capped — the caller picks the first untried in
    tier order, so tier 1 is exhausted before tier 2, and the lattice floor
    guarantees every region is probed once before "no live controls" can be
    concluded (Rudakov 2512.24156; fixes INERT-START)."""
    bg = background_color(grid)
    out: list[tuple[int, str]] = []
    seen_xy: set[tuple[int, int]] = set()
    for c in components(grid, bg):
        if hud_mask is not None and hud_mask[c.ry, c.rx]:
            continue  # inside a masked HUD region
        xy = (c.rx, c.ry)
        if xy in seen_xy:
            continue
        seen_xy.add(xy)
        out.append((_tier_for_size(c.size), f"ACTION6:{c.rx},{c.ry}"))
    # deterministic salience order; representative-cell coords break ties
    out.sort(key=lambda tk: (tk[0], tk[1]))
    # lattice floor (lowest priority): one click per coarse cell center
    half = lattice_step // 2
    for y in range(half, grid.shape[0], lattice_step):
        for x in range(half, grid.shape[1], lattice_step):
            if hud_mask is not None and hud_mask[y, x]:
                continue
            if (x, y) in seen_xy:
                continue
            out.append((_TIER_LATTICE, f"ACTION6:{x},{y}"))
    return out


def segment_signatures(grid: np.ndarray) -> frozenset[str]:
    """Translation-invariant component signatures present in a frame. Used for
    segment-granularity visit counts and state novelty (#Exploration)."""
    return frozenset(c.sig for c in components(grid))


def state_novelty(grid: np.ndarray, seg_visits: dict) -> int:
    """Count of segments in this frame that are RARE (<=1 prior visit). A
    state introducing unseen component kinds scores high even if a HUD ticker
    makes its whole-frame hash trivially 'new' every step."""
    sigs = segment_signatures(grid)
    return sum(1 for s in sigs if seg_visits.get(s, 0) <= 1)


@dataclass
class ArchiveEntry:
    cell: str                       # discretized state key (segment-set hash)
    frame_hash: str
    action_prefix: tuple            # actions from level start to reach it
    prefix_len: int
    level: int
    novelty: int                    # rare-segment count when archived
    near_event: bool                # within k actions of LEVEL/WIN/GAME_OVER
    meter_extreme: bool             # touched an analyzer-flagged region extreme
    visits: int = 0                 # times returned-to (anti-entrenchment)


@dataclass
class Archive:
    """Go-Explore archive of interesting frames, keyed by a discretized state
    cell (segment-signature-set hash). Ranking prefers short prefixes
    (cap-aware), high novelty, near-event and meter-extreme states.

    Capped policy: return costs prefix_len actions, so callers pass
    cap_remaining and we only surface cells whose prefix fits. Uncapped,
    prefix_len only breaks ties (return is action-free via replay)."""
    k_near_event: int = 6
    entries: dict[str, ArchiveEntry] = field(default_factory=dict)
    _recent: list = field(default_factory=list)  # rolling (cell, action) prefix

    @staticmethod
    def cell_key(grid: np.ndarray) -> str:
        sigs = sorted(segment_signatures(grid))
        return hashlib.sha1(("|".join(sigs)).encode()).hexdigest()[:16]

    def note_step(self, action_key: str, cell: str) -> None:
        self._recent.append((cell, action_key))
        if len(self._recent) > 4000:
            self._recent = self._recent[-4000:]

    def reset_prefix(self) -> None:
        """Level start / new play: the recorded prefix restarts from here."""
        self._recent = []

    def consider(self, grid: np.ndarray, frame_hash: str, level: int,
                 prefix: tuple, novelty: int, near_event: bool,
                 meter_extreme: bool) -> None:
        cell = self.cell_key(grid)
        prev = self.entries.get(cell)
        # keep the SHORTER prefix to reach a given cell (cheaper to return to)
        if prev is not None and prev.prefix_len <= len(prefix) \
                and not (near_event or meter_extreme) :
            prev.novelty = max(prev.novelty, novelty)
            return
        self.entries[cell] = ArchiveEntry(
            cell=cell, frame_hash=frame_hash, action_prefix=tuple(prefix),
            prefix_len=len(prefix), level=level, novelty=novelty,
            near_event=near_event, meter_extreme=meter_extreme,
            visits=prev.visits if prev else 0,
        )

    def pick_return(self, level: int, cap_remaining: Optional[int],
                    capped: bool) -> Optional[ArchiveEntry]:
        """Best cell to return to and explore outward from, or None. Filters by
        level and (capped) by prefix fitting the remaining window."""
        cands = [e for e in self.entries.values() if e.level == level
                 and e.prefix_len > 0]
        if capped and cap_remaining is not None:
            # leave room to actually do something after returning
            cands = [e for e in cands if e.prefix_len + 2 <= cap_remaining]
        if not cands:
            return None

        def rank(e: ArchiveEntry):
            # higher is better: novelty, event/meter interest, fewer revisits;
            # short prefix is primary under the cap, a tiebreak otherwise
            interest = e.novelty + (3 if e.near_event else 0) + (2 if e.meter_extreme else 0)
            if capped:
                return (-e.prefix_len, interest, -e.visits)
            return (interest, -e.visits, -e.prefix_len)

        best = max(cands, key=rank)
        return best
