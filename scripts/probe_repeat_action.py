"""Verify AERA's Table-9 claim (arXiv 2605.25931 s5.8): several public games
are solvable by repeating a SINGLE simple action, within 50-200 steps.

For every game x ACTION1..5: fresh env, repeat the action up to MAX_REPS or
WIN, record levels completed and the step indices of level advances.

    .venv/bin/python scripts/probe_repeat_action.py

Writes results/cap_study/repeat_action_probe.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction, GameState

from harness.runner import DEFAULT_ENV_DIR

MAX_REPS = 500
# AERA Table 9: game -> steps claimed sufficient (single repeated action)
AERA_T9 = {"tu93": 50, "re86": 100, "tr87": 128, "ka59": 100,
           "ls20": 129, "sc25": 52, "g50t": 130, "wa30": 200}

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
           GameAction.ACTION4, GameAction.ACTION5]


def main():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=DEFAULT_ENV_DIR)
    games = sorted(e.game_id.split("-")[0] for e in arcade.get_environments())
    out = {"max_reps": MAX_REPS, "games": {}}
    t0 = time.time()
    for g in games:
        rows = {}
        for act in ACTIONS:
            card = arcade.open_scorecard(tags=["repeat-probe"])
            env = arcade.make(g, scorecard_id=card, include_frame_data=False)
            if env is None:
                continue
            fd = env.observation_space
            if act.value not in [a.value if hasattr(a, "value") else int(a)
                                 for a in (fd.available_actions or [])]:
                arcade.close_scorecard(card)
                continue  # off-menu: skip (counted no-ops, never productive)
            advances = []
            lvl = fd.levels_completed
            state = fd.state.name
            for i in range(MAX_REPS):
                fd = env.step(act, None)
                if fd is None:
                    state = "STEP_NONE"
                    break
                if fd.levels_completed > lvl:
                    advances.append(i + 1)
                    lvl = fd.levels_completed
                state = fd.state.name
                if fd.state == GameState.WIN:
                    break
                if fd.state == GameState.GAME_OVER:
                    # pure repeat: a level RESET would deviate; stop here
                    break
            arcade.close_scorecard(card)
            rows[act.name] = {"levels": lvl, "advances_at": advances,
                              "end_state": state, "steps": i + 1}
        best = max(rows.values(), key=lambda r: (r["levels"],
                   -r["steps"]), default=None)
        out["games"][g] = {"per_action": rows, "best_levels": best["levels"] if best else 0,
                           "aera_t9_steps": AERA_T9.get(g)}
        flag = " <== AERA-T9" if g in AERA_T9 else ""
        print(f"{g:6s} best_levels={best['levels'] if best else 0} "
              f"{ {a: (r['levels'], r['end_state'], r['steps']) for a, r in rows.items()} }{flag}")
    Path("results/cap_study/repeat_action_probe.json").write_text(
        json.dumps(out, indent=2))
    print(f"\ndone in {time.time()-t0:.1f}s -> results/cap_study/repeat_action_probe.json")


if __name__ == "__main__":
    main()
