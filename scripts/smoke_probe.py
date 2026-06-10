"""Local smoke for the gateway diagnostic: drive Diagnostician against the
in-process arc_agi engine (OFFLINE) on a few public games, to prove the
state machine runs and prints sane traces BEFORE the human spends a Kaggle
submission. This is NOT a gateway test (the local engine has no cutoff) —
it only validates the probe's plumbing and that RESET-after-WIN / null-coord
branches fire.

    .venv/bin/python scripts/smoke_probe.py --games tt01 --env-dir test_envs
    .venv/bin/python scripts/smoke_probe.py --games lf52 sp80 r11l
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

from harness.runner import DEFAULT_ENV_DIR
from kaggle_probe.probe_agent import Diagnostician, MAX_PROBE_ACTIONS


def drive(arcade, card_id, game_id, cap_actions):
    env = arcade.make(game_id, scorecard_id=card_id, include_frame_data=False)
    if env is None:
        return {"game": game_id, "error": "make failed"}
    fd = env.observation_space
    diag = Diagnostician(game_id)
    frames = [fd]
    steps = 0
    while fd is not None and steps < cap_actions and not diag.done(frames, fd):
        diag.observe_post()
        act_id, data = diag.next_action(frames, fd)
        action = GameAction.from_id(act_id)
        if data is not None:
            action.set_data(data)
        fd = env.reset() if act_id == 0 else env.step(action, data)
        if fd is not None:
            frames.append(fd)
        steps += 1
    return diag.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=["tt01"])
    p.add_argument("--env-dir", default=None)
    p.add_argument("--cap", type=int, default=300,
                   help="smoke action cap (real probe uses %d)" % MAX_PROBE_ACTIONS)
    args = p.parse_args()
    env_dir = args.env_dir or (
        "test_envs" if args.games == ["tt01"] else DEFAULT_ENV_DIR)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    card = arcade.open_scorecard(tags=["smoke-probe"])
    out = []
    for g in args.games:
        out.append(drive(arcade, card, g, args.cap))
    arcade.close_scorecard(card)
    Path("/tmp/smoke_probe_out.json").write_text(json.dumps(out, indent=2))
    for s in out:
        print(f"{s.get('game'):6s} win@={s.get('win_seen_at')} "
              f"mint={s.get('reset_after_win_frame') is not None} "
              f"GOs={s.get('game_overs')} maxL0={s.get('max_level0_reached')} "
              f"nullcoord={s.get('null_coord_result')} stuck={s.get('stuck_terminal')}",
              flush=True)


if __name__ == "__main__":
    main()
