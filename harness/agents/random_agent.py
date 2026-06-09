"""Random baseline: uniform over the game's advertised available_actions.

Floor for any real agent, and an end-to-end pipeline check. Respects
available_actions (off-menu actions are accepted by the engine, do nothing,
and still cost an action) and never RESETs except after GAME_OVER, where
RESET is the only way to keep playing (engine restarts the current level
and charges one action).
"""

import random
from typing import Any, Optional

from arcengine import FrameDataRaw, GameAction, GameState

from .base import Agent

_ID_TO_ACTION = {a.value: a for a in GameAction}


class RandomAgent(Agent):
    name = "random"

    def __init__(self, game_id: str, seed: int = 0) -> None:
        super().__init__(game_id, seed)
        self.rng = random.Random(f"{game_id}:{seed}")

    def choose_action(
        self, frames: list[FrameDataRaw], latest_frame: FrameDataRaw
    ) -> tuple[GameAction, Optional[dict[str, Any]]]:
        if latest_frame.state == GameState.GAME_OVER:
            return GameAction.RESET, None

        candidates = [
            aid
            for aid in (latest_frame.available_actions or [])
            if aid in _ID_TO_ACTION and aid != GameAction.RESET.value
        ]
        if not candidates:  # nothing advertised: keyboard-ish guess
            candidates = [1, 2, 3, 4]

        action = _ID_TO_ACTION[self.rng.choice(candidates)]
        if action == GameAction.ACTION6:
            return action, {"x": self.rng.randrange(64), "y": self.rng.randrange(64)}
        return action, None
