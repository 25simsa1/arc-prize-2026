"""Append-only transition log for ONE game, deduplicated, persisted per run.

Key = (level, pre_hash, action_key). Level is part of the key because frames
alias across levels (tt01's two levels render identically; real games reuse
layouts) — levels_completed is agent-observable, so conditioning on it is
legal. A conflicting duplicate (same key, different outcome) would falsify
the determinism assumption verification relies on; we keep the first
observation, log the conflict, and expose it loudly.

Transitions accumulate ACROSS plays of the same game. RESET transitions are
not stored — RESET is runner/agent plumbing, not game dynamics.
"""

import hashlib
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

EVENT_NONE = "NONE"
EVENT_LEVEL = "LEVEL"
EVENT_WIN = "WIN"
EVENT_GAME_OVER = "GAME_OVER"


def canon_frame(frame_layers: Any) -> np.ndarray:
    """Canonical state = LAST rendered layer of the action (layers before it
    are intra-action animation frames), as a contiguous int16 array."""
    last = frame_layers[-1]
    return np.ascontiguousarray(np.asarray(last, dtype=np.int16))


def frame_hash(grid: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(str(grid.shape).encode())
    h.update(grid.tobytes())
    return h.hexdigest()[:20]


def masked_hash(grid: np.ndarray, exclude_mask) -> str:
    """Context hash ignoring excluded (e.g. ALWAYS_CHANGING/HUD) cells, so a
    ticking meter doesn't make every state look novel. None mask = full hash."""
    if exclude_mask is None:
        return frame_hash(grid)
    h = hashlib.sha1()
    h.update(str(grid.shape).encode())
    h.update(grid[~exclude_mask].tobytes())
    return h.hexdigest()[:20]


def derive_event(pre_level: int, post_level: int, post_state: str) -> str:
    if post_state == "GAME_OVER":
        return EVENT_GAME_OVER
    if post_state == "WIN":
        return EVENT_WIN
    if post_level > pre_level:
        return EVENT_LEVEL
    return EVENT_NONE


@dataclass
class Transition:
    level: int
    pre: np.ndarray
    pre_hash: str
    action_key: str  # "ACTION1".."ACTION7" or "ACTION6:x,y"
    post: np.ndarray
    post_hash: str
    post_level: int
    event: str
    play_index: int = 0
    diff_cells: int = -1  # changed-cell count; eviction protects <=2 (HUD seeds)

    @property
    def base_action(self) -> str:
        return self.action_key.split(":", 1)[0]

    @property
    def click_xy(self) -> Optional[tuple[int, int]]:
        if ":" not in self.action_key:
            return None
        x, y = self.action_key.split(":", 1)[1].split(",")
        return int(x), int(y)


@dataclass
class TransitionStore:
    game_id: str
    by_key: dict[tuple[int, str, str], Transition] = field(default_factory=dict)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    appended_total: int = 0
    created_at: float = field(default_factory=time.monotonic)
    # Memory rail: full grids cost ~16KB/transition; fast games reach ~100k
    # transitions in one 240s budget (~1.6GB). At the cap we EVICT rather
    # than refuse, preferentially KEEPING proposer food — conflict pairs,
    # sole/pair-changer evidence (RegionAnalyzer seeds), and every
    # LEVEL/WIN/GAME_OVER transition — and evicting ordinary NONE-event
    # multi-cell transitions oldest-first. Known limitation, documented: a
    # conflict whose FIRST observation was evicted cannot be detected on
    # re-observation (the comparison baseline is gone); protected entries
    # are never evicted, so established conflict keys keep detecting.
    max_transitions: Optional[int] = None
    capped_drops: int = 0      # new transitions refused (nothing evictable)
    evicted_total: int = 0
    # Observation-level event census — counted on every non-duplicate add
    # BEFORE any storage decision, so eviction/capping can never skew the
    # evidence-starvation table.
    event_counts: dict = field(default_factory=dict)
    _evictable: deque = field(default_factory=deque)   # candidate keys, FIFO
    _conflict_keys: set = field(default_factory=set)

    def _is_protected(self, t: Transition, key) -> bool:
        return (
            t.event != EVENT_NONE
            or key in self._conflict_keys
            or 0 <= t.diff_cells <= 2
        )

    def _evict_one(self) -> bool:
        while self._evictable:
            key = self._evictable.popleft()
            t = self.by_key.get(key)
            if t is None:
                continue  # already gone
            if self._is_protected(t, key):
                continue  # promoted (e.g. conflicted) since enqueued
            del self.by_key[key]
            self.evicted_total += 1
            return True
        return False

    def add(
        self,
        level: int,
        pre: np.ndarray,
        action_key: str,
        post: np.ndarray,
        post_level: int,
        post_state: str,
        play_index: int = 0,
    ) -> tuple[str, Transition]:
        """Returns (status, transition); status in {new, dup, conflict,
        capped}. "capped" returns transition=None — the store was full and
        nothing was evictable (callers must None-check before use)."""
        ph, qh = frame_hash(pre), frame_hash(post)
        key = (level, ph, action_key)
        event = derive_event(level, post_level, post_state)
        existing = self.by_key.get(key)
        if existing is not None and existing.post_hash == qh and existing.event == event:
            return "dup", existing
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        if existing is not None:
            self.conflicts.append(
                {
                    "key": str(key),
                    "first": f"{existing.post_hash}/{existing.event}",
                    "second": f"{qh}/{event}",
                }
            )
            self._conflict_keys.add(key)  # conflict pairs are proposer food
            return "conflict", existing  # keep first; determinism is suspect
        if self.max_transitions is not None and len(self.by_key) >= self.max_transitions:
            if not self._evict_one():
                self.capped_drops += 1  # everything left is protected
                return "capped", None  # type: ignore[return-value]
        diff_cells = int(np.count_nonzero(pre != post))
        t = Transition(level, pre.copy(), ph, action_key, post.copy(), qh,
                       post_level, event, play_index, diff_cells)
        self.by_key[key] = t
        self.appended_total += 1
        if not self._is_protected(t, key):
            self._evictable.append(key)
        return "new", t

    def __len__(self) -> int:
        return len(self.by_key)

    def all(self) -> Iterable[Transition]:
        return self.by_key.values()

    def for_base_action(self, base: str) -> list[Transition]:
        return [t for t in self.by_key.values() if t.base_action == base]

    def at_context(self, level: int, pre_hash: str) -> list[Transition]:
        return [
            t for (lvl, ph, _), t in self.by_key.items()
            if lvl == level and ph == pre_hash
        ]

    def lookup(self, level: int, pre_hash: str, action_key: str) -> Optional[Transition]:
        return self.by_key.get((level, pre_hash, action_key))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {"game_id": self.game_id, "transitions": list(self.by_key.values()),
                 "conflicts": self.conflicts,
                 # eviction/census state — without these a loaded store can't
                 # evict (nothing enqueued => every add at the cap is refused),
                 # loses conflict-key protection, and under-reports the census
                 "conflict_keys": list(self._conflict_keys),
                 "appended_total": self.appended_total,
                 "event_counts": dict(self.event_counts)},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(game_id=data["game_id"])
        for t in data["transitions"]:
            store.by_key[(t.level, t.pre_hash, t.action_key)] = t
        store.conflicts = data["conflicts"]
        # legacy pickles lack these fields; fall back to what's recoverable
        # from the retained transitions (evicted observations are gone)
        store._conflict_keys = {tuple(k) for k in data.get("conflict_keys", [])}
        store.appended_total = data.get("appended_total", len(store.by_key))
        counts = data.get("event_counts")
        if counts is None:
            counts = {}
            for t in store.by_key.values():
                counts[t.event] = counts.get(t.event, 0) + 1
        store.event_counts = counts
        # rebuild the evictable queue in insertion (oldest-first) order
        for key, t in store.by_key.items():
            if not store._is_protected(t, key):
                store._evictable.append(key)
        return store
