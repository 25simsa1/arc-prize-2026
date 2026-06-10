"""Win-gate flow test on the tt01 fixture: sloppy play-1 win -> planner
replay -> game score = replay score, with >=80% of replay steps matching
model predictions.

    .venv/bin/python scripts/test_wm_tt01.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import RunConfig, run_suite
from harness.agents.wm_agent import WorldModelAgent

TEST_ENVS = str(Path(__file__).resolve().parent.parent / "test_envs")


def main() -> None:
    agents: list[WorldModelAgent] = []

    def factory(game_id: str, seed: int) -> WorldModelAgent:
        a = WorldModelAgent(game_id, seed, proposer="template", time_budget_s=60)
        agents.append(a)
        return a

    factory.agent_name = "wm-template"

    cfg = RunConfig(
        max_actions_per_game=500,
        environments_dir=TEST_ENVS,
        mode="two_phase",
        tag="wm-tt01",
        quiet=True,
    )
    record = run_suite(["tt01"], factory, cfg)

    env = record["scorecard"]["environments"][0]
    runs = env["runs"]
    agent = agents[0]
    rep = agent.report()
    print(json.dumps({"game_score": env["score"], "runs": [
        {"score": round(r["score"], 2), "actions": r["actions"]} for r in runs
    ], "agent": rep["plays"], "status": rep["status"]}, indent=2))

    assert len(runs) >= 2, "expected at least 2 plays (win-gate replay)"
    assert env["score"] == 100.0, f"expected 100.0 from clean replay, got {env['score']}"
    # tt01's baseline (3/level) is generous enough that a lucky play-1 win can
    # also hit the 100 cap — when scores tie at the cap, fewer actions is the
    # honest efficiency comparison.
    replay = runs[-1]
    assert replay["actions"] == 2, f"replay should take 2 actions, took {replay['actions']}"
    # WinSeeker v3 tries ACTION1 (first simple) before clicks, so play 0 now
    # wins tt01 optimally too (2 actions) instead of the old rotation's sloppy
    # win — so replay ties rather than strictly beats. The substantive checks
    # below (fully plan-driven, prediction-matched, score 100) still hold.
    assert replay["actions"] <= runs[0]["actions"], "replay worse than play 1 on actions"
    replay_logs = [p for p in rep["plays"] if p["play"] >= 1]
    assert replay_logs, "no replay play recorded by agent"
    final = replay_logs[-1]
    m = final["match"]  # honest triple: matched / predicted_steps / total_actions
    assert m["predicted_steps"] > 0 and m["matched"] / m["predicted_steps"] >= 0.8, (
        f"replay match too low: {m}"
    )
    assert m["predicted_steps"] == m["total_actions"], (
        "replay steps must all be model-predicted"
    )
    assert final["sources"].get("plan", 0) == final["actions"], (
        "replay must be fully planner-driven"
    )
    print("\nTT01 WIN-GATE TEST PASS")


if __name__ == "__main__":
    main()
