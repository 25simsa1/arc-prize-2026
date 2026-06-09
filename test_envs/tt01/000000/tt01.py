"""Trivial 2-level test game for play-semantics experiments.

ACTION1 completes the current level (auto-WIN on the last); ACTION5 is a
counted no-op, used to make a play deliberately sloppy. Human baseline in
metadata.json is 3 actions/level, so a 1-action level scores at the 115 cap
and a 6-action level scores (3/6)^2 = 25%.
"""

from arcengine import ARCBaseGame, GameAction, Level


class Tt01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(sprites=[], grid_size=(64, 64), name="L1"),
            Level(sprites=[], grid_size=(64, 64), name="L2"),
        ]
        super().__init__(
            game_id="tt01-000000",
            levels=levels,
            available_actions=[1, 5],
            seed=seed,
        )

    def step(self) -> None:
        if self.action.id == GameAction.ACTION1:
            self.next_level()
        self.complete_action()
