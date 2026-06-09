"""Workstream A ground truth: multi-play scoring semantics, three experiments.

Uses the tt01 test fixture (2 levels, baseline 3 actions/level; ACTION1
completes a level, ACTION5 is a counted no-op).

  1. NORMAL/OFFLINE mode, re-make: sloppy winning play 1 (12 actions), then a
     fresh wrapper on the same scorecard plays clean (2 actions). Expect two
     runs; game score = clean play's score, unpolluted by play 1.
  2. RESET-after-WIN on ONE wrapper (the competition-legal path): sloppy win,
     RESET (engine full-resets because state==WIN), clean win. Same
     expectations, same guid.
  3. COMPETITION mode over HTTP: (a) starting a second environment for an
     already-played game is refused; (b) RESET at action_count==0 is
     intercepted (counted, no new play); (c) RESET after WIN still creates a
     new play; game score = max(plays).

    .venv/bin/python scripts/test_play_semantics.py
"""

import json
from pathlib import Path

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction

TEST_ENVS = str(Path(__file__).resolve().parent.parent / "test_envs")


def sloppy_win(step):
    """5 no-ops + complete, per level: 12 actions total, (3/6)^2=25% per level."""
    for _ in range(2):
        for _ in range(5):
            step(GameAction.ACTION5)
        step(GameAction.ACTION1)


def clean_win(step):
    """2 actions total: each level at the 115 cap, game capped to 100."""
    step(GameAction.ACTION1)
    step(GameAction.ACTION1)


def game_json(sc, game_id="tt01-000000"):
    data = json.loads(sc.model_dump_json())
    return next(e for e in data["environments"] if e["id"] == game_id)


def summarize(env):
    runs = [
        {
            "score": round(r["score"], 2),
            "levels": r["levels_completed"],
            "actions": r["actions"],
            "level_actions": r.get("level_actions"),
            "state": r["state"],
        }
        for r in env["runs"]
    ]
    return {"game_score": round(env["score"], 2), "runs": runs}


def part1_remake():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=TEST_ENVS)
    card = arcade.open_scorecard(tags=["plays-p1"])

    env1 = arcade.make("tt01", scorecard_id=card)
    sloppy_win(lambda a: env1.step(a))
    assert env1.observation_space.state.name == "WIN"

    env2 = arcade.make("tt01", scorecard_id=card)  # fresh wrapper => new play
    clean_win(lambda a: env2.step(a))
    assert env2.observation_space.state.name == "WIN"

    out = summarize(game_json(arcade.close_scorecard(card)))
    print("P1 re-make (normal mode):", json.dumps(out))
    assert len(out["runs"]) == 2, out
    assert out["runs"][0]["actions"] == 12 and out["runs"][0]["score"] == 25.0, out
    assert out["runs"][1]["actions"] == 2 and out["runs"][1]["score"] == 100.0, out
    assert out["game_score"] == 100.0, out
    print("P1 PASS: game score = clean play alone; sloppy play's 12 actions inert\n")


def part2_reset_after_win():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=TEST_ENVS)
    card = arcade.open_scorecard(tags=["plays-p2"])

    env = arcade.make("tt01", scorecard_id=card)
    guid = env._guid
    sloppy_win(lambda a: env.step(a))

    fd = env.reset()  # state==WIN => engine full_reset => new play, same guid
    assert fd.full_reset is True, "expected full reset after WIN"
    clean_win(lambda a: env.step(a))

    out = summarize(game_json(arcade.close_scorecard(card)))
    print("P2 RESET-after-WIN (one wrapper):", json.dumps(out))
    assert len(out["runs"]) == 2, out
    assert out["runs"][1]["actions"] == 2 and out["game_score"] == 100.0, out
    print(f"P2 PASS: same wrapper/guid ({guid[:8]}…), WIN->RESET minted play 2\n")


def part3_competition_http():
    from arc_agi.server import create_app

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=TEST_ENVS)
    app, _ = create_app(arcade, competition_mode=True)
    c = app.test_client()

    card = c.post("/api/scorecard/open", json={"tags": ["plays-p3"]}).get_json()["card_id"]

    fd = c.post("/api/cmd/RESET", json={"game_id": "tt01-000000", "card_id": card}).get_json()
    guid = fd["guid"]

    # (b) RESET at action_count==0: intercepted — must NOT mint a new play.
    fd0 = c.post(
        "/api/cmd/RESET",
        json={"game_id": "tt01-000000", "card_id": card, "guid": guid},
    ).get_json()
    assert "error" not in fd0, fd0

    # sloppy win over HTTP
    for _ in range(2):
        for _ in range(5):
            c.post("/api/cmd/ACTION5", json={"game_id": "tt01-000000", "guid": guid})
        fd = c.post("/api/cmd/ACTION1", json={"game_id": "tt01-000000", "guid": guid}).get_json()
    assert fd["state"] == "WIN", fd

    # (a) second environment for an already-played game: refused.
    blocked = c.post("/api/cmd/RESET", json={"game_id": "tt01-000000", "card_id": card})
    assert blocked.status_code != 200 or "error" in (blocked.get_json() or {}), (
        "competition mode allowed a second environment instance!"
    )

    # (c) RESET after WIN: allowed, mints play 2; clean win.
    fd = c.post(
        "/api/cmd/RESET",
        json={"game_id": "tt01-000000", "card_id": card, "guid": guid},
    ).get_json()
    for _ in range(2):
        fd = c.post("/api/cmd/ACTION1", json={"game_id": "tt01-000000", "guid": guid}).get_json()
    assert fd["state"] == "WIN", fd

    closed = c.post("/api/scorecard/close", json={"card_id": card}).get_json()
    env = next(e for e in closed["environments"] if e["id"] == "tt01-000000")
    out = summarize(env)
    print("P3 competition mode (HTTP):", json.dumps(out))
    assert len(out["runs"]) == 2, out
    assert out["game_score"] == 100.0, out
    # play 1 carries the intercepted reset: 12 scored actions + 1 counted reset
    assert out["runs"][0]["actions"] == 13, out
    assert out["runs"][1]["actions"] == 2, out
    print(
        "P3 PASS: 2nd env refused; RESET@0 counted, no play minted; "
        "WIN->RESET minted play 2; game score = max(plays)\n"
    )


if __name__ == "__main__":
    part1_remake()
    part2_reset_after_win()
    part3_competition_http()
    print("VERDICT EVIDENCE COMPLETE: see NOTES.md 'Play semantics'")
