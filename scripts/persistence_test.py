"""Empirical settlement of the AERA repeated-single-action claim (V1
follow-up): for each candidate game, repeat each basic action up to N steps
(RESET on GAME_OVER, which costs an action) and record what actually
happens: WIN / level progress / death loop / nothing.

Also reports actions-per-level against the 5x-baseline per-level cutoff
(tech report 2603.24621 eval policy) so the cap interaction is settled in
the same run.

    .venv/bin/python scripts/persistence_test.py ft09 vc33 lp85 s5i5
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction, GameState

ENV_DIR = str(Path.home()
              / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files")
N_STEPS = 1500
ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
           GameAction.ACTION4, GameAction.ACTION5]


def probe(arcade, card, game_id: str, action: GameAction, baselines) -> dict:
    env = arcade.make(game_id, scorecard_id=card)
    if env is None:
        return {"error": "make failed"}
    fd = env.observation_space
    if action.value not in (fd.available_actions or []):
        return {"skip": "not available"}
    level_hits = []          # (scored_action_index, new_level)
    game_overs = 0
    scored = 0
    cap_violations = []
    level_start = 0
    for _ in range(N_STEPS):
        if fd.state == GameState.WIN:
            break
        if fd.state == GameState.GAME_OVER:
            fd = env.reset()   # counted level-reset
            scored += 1
            game_overs += 1
            continue
        prev_level = fd.levels_completed
        fd = env.step(action)
        if fd is None:
            return {"error": "step None"}
        scored += 1
        if fd.levels_completed > prev_level:
            spent = scored - level_start
            cap = (5 * baselines[prev_level]
                   if baselines and prev_level < len(baselines) else None)
            level_hits.append({"level": fd.levels_completed, "actions": spent,
                               "cap_5x": cap,
                               "within_cap": (cap is None or spent <= cap)})
            if cap is not None and spent > cap:
                cap_violations.append(prev_level)
            level_start = scored
    return {
        "outcome": fd.state.name,
        "levels": fd.levels_completed,
        "scored_actions": scored,
        "game_overs": game_overs,
        "level_hits": level_hits,
        "cap_violations": cap_violations,
    }


def probe_click(arcade, card, game_id: str, xy, baselines, n=N_STEPS) -> dict:
    """Repeated ACTION6 at one fixed coordinate (xy=None => the
    null-coordinate audit: send the click with NO coordinates at all)."""
    env = arcade.make(game_id, scorecard_id=card)
    fd = env.observation_space
    data = {"x": xy[0], "y": xy[1]} if xy is not None else {}
    level_hits, game_overs, scored, level_start = [], 0, 0, 0
    errors = 0
    for _ in range(n):
        if fd.state == GameState.WIN:
            break
        if fd.state == GameState.GAME_OVER:
            fd = env.reset()
            scored += 1
            game_overs += 1
            continue
        prev_level = fd.levels_completed
        nxt = env.step(GameAction.ACTION6, dict(data))
        if nxt is None:
            errors += 1
            if errors >= 3:
                return {"outcome": "STEP_ERRORS", "errors": errors,
                        "scored_actions": scored, "levels": fd.levels_completed}
            continue
        fd = nxt
        scored += 1
        if fd.levels_completed > prev_level:
            spent = scored - level_start
            cap = (5 * baselines[prev_level]
                   if baselines and prev_level < len(baselines) else None)
            level_hits.append({"level": fd.levels_completed, "actions": spent,
                               "cap_5x": cap,
                               "within_cap": (cap is None or spent <= cap)})
            level_start = scored
    return {"outcome": fd.state.name, "levels": fd.levels_completed,
            "scored_actions": scored, "game_overs": game_overs,
            "level_hits": level_hits, "errors": errors}


def main() -> None:
    games = sys.argv[1:] or ["ft09", "vc33", "lp85", "s5i5"]
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    base = {e.game_id.split("-")[0]: e.baseline_actions for e in arcade.get_environments()}
    for g in games:
        card = arcade.open_scorecard(tags=["persistence-test"])
        print(f"\n=== {g} (baselines {base.get(g)})")
        for a in ACTIONS:
            r = probe(arcade, card, g, a, base.get(g))
            if "skip" in r:
                continue
            print(f"  {a.name}: {json.dumps(r)}")
        for xy in [(32, 32), (0, 0), (63, 63), (16, 48), (48, 16)]:
            r = probe_click(arcade, card, g, xy, base.get(g))
            print(f"  ACTION6@{xy}: {json.dumps(r)[:220]}")
        r = probe_click(arcade, card, g, None, base.get(g), n=30)
        print(f"  ACTION6(NULL-COORD AUDIT): {json.dumps(r)[:220]}")
        arcade.close_scorecard(card)


if __name__ == "__main__":
    main()
