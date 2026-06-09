"""Agent contract, mirroring the official ARC-AGI-3-Agents repo.

Same method names/semantics as the official `Agent` class so an agent here
ports to the official repo (and its HTTP loop) with only a thin shim:
`choose_action(frames, latest_frame)` and `is_done(frames, latest_frame)`.

One deliberate difference: the official repo attaches click coordinates to
its own action wrapper; `arcengine.GameAction` is a bare enum, so here
`choose_action` returns `(GameAction, data)` where `data` is e.g.
`{"x": 32, "y": 17}` for ACTION6 and None otherwise.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from arcengine import FrameDataRaw, GameAction


class Agent(ABC):
    name: str = "agent"

    def __init__(self, game_id: str, seed: int = 0) -> None:
        self.game_id = game_id
        self.seed = seed

    @abstractmethod
    def choose_action(
        self, frames: list[FrameDataRaw], latest_frame: FrameDataRaw
    ) -> tuple[GameAction, Optional[dict[str, Any]]]:
        """Pick the next action given full frame history and the latest frame."""
        raise NotImplementedError

    def on_play_start(self, play_index: int) -> None:
        """Called when a new play of the same game begins (two-phase mode).

        play_index is 0 for the first play. Agents switch policy here, e.g.
        explore/sloppy-win on play 0, execute from the world model afterward.
        """

    def is_done(
        self, frames: list[FrameDataRaw], latest_frame: FrameDataRaw
    ) -> bool:
        """Stop condition beyond the runner's own WIN/budget checks."""
        return False
