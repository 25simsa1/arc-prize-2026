"""Stage-2 smoke test: drive one environment in-process via the offline runtime.

Verifies: environment discovery, make/reset/step, FrameData fields, and how
the shipped scorecard counts actions/resets/levels. Run from the repo root:

    .venv/bin/python scripts/smoke_env.py
"""

import json
from pathlib import Path

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

ENV_DIR = str(
    Path.home()
    / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files"
)


def frame_summary(fd, label):
    grid = fd.frame
    dims = (
        f"{len(grid)} layer(s) of "
        f"{len(grid[0])}x{len(grid[0][0])}" if grid is not None and len(grid) else "empty"
    )
    cells = set()
    if grid is not None and len(grid):
        for row in grid[0]:
            cells.update(int(v) for v in row)
    print(
        f"[{label}] state={fd.state.name} levels={fd.levels_completed}/{fd.win_levels} "
        f"frame={dims} palette⊆0-15:{max(cells) <= 15 if cells else 'n/a'} "
        f"available={fd.available_actions} full_reset={getattr(fd, 'full_reset', None)}"
    )
    return fd


def main():
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENV_DIR,
        recordings_dir="recordings",
    )
    envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
    print(f"Discovered {len(envs)} environments:")
    for e in envs:
        nb = len(e.baseline_actions or [])
        print(f"  {e.game_id:16s} levels={nb:2d} baselines={e.baseline_actions} tags={e.tags}")

    card_id = arcade.open_scorecard(tags=["smoke"])
    env = arcade.make("ls20", seed=0, scorecard_id=card_id)
    assert env is not None, "make('ls20') failed"

    fd = env.observation_space  # set by the reset() inside the wrapper __init__
    frame_summary(fd, "init/RESET")

    # A few simple actions, then a click, then a mid-game RESET.
    for i, act in enumerate(
        [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    ):
        fd = env.step(act)
        frame_summary(fd, f"step{i + 1}:{act.name}")

    fd = env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
    frame_summary(fd, "step5:ACTION6(32,32)")

    fd = env.reset()
    frame_summary(fd, "mid-game RESET")

    sc = arcade.close_scorecard(card_id)
    print("\n=== EnvironmentScorecard (close) ===")
    print(json.dumps(json.loads(sc.model_dump_json()), indent=2))

    # Expectation per scorecard source: 5 scored actions (ACTION1-6) + 1 scored
    # reset (mid-game, non-full) = 6 total actions, resets=1 — unless the
    # engine flagged the mid-game RESET as full_reset (then a 2nd play with 0).
    print("\nExpectation: actions≈6 incl. resets=1 on a single play; "
          "init RESET uncounted (new_play).")


if __name__ == "__main__":
    main()
