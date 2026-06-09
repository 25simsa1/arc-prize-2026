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
        """Returns (status, transition); status in {new, dup, conflict}."""
        ph, qh = frame_hash(pre), frame_hash(post)
        key = (level, ph, action_key)
        event = derive_event(level, post_level, post_state)
        existing = self.by_key.get(key)
        if existing is not None:
            if existing.post_hash == qh and existing.event == event:
                return "dup", existing
            self.conflicts.append(
                {
                    "key": str(key),
                    "first": f"{existing.post_hash}/{existing.event}",
                    "second": f"{qh}/{event}",
                }
            )
            return "conflict", existing  # keep first; determinism is suspect
        t = Transition(level, pre.copy(), ph, action_key, post.copy(), qh,
                       post_level, event, play_index)
        self.by_key[key] = t
        self.appended_total += 1
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
                 "conflicts": self.conflicts},
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
        return store
