"""Unit tests for the PROVISIONAL per-level action cap (RunConfig.
per_level_action_cap_multiplier), modeling the tech-report eval cutoff
(2603.24621: cut off after 5x the level's human baseline actions).

Uses the tt01 fixture (2 levels, baseline 3/level => cap 15/level at 5.0;
ACTION1 completes a level, ACTION5 is a counted no-op).

    .venv/bin/python scripts/test_level_cap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction

from harness import RunConfig, run_suite
from harness.agents.base import Agent
from harness.agents.random_agent import RandomAgent
from harness.runner import level_action_cap

TEST_ENVS = str(Path(__file__).resolve().parent.parent / "test_envs")
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


class Scripted(Agent):
    """Plays a fixed action script per play index; ACTION5 when exhausted."""

    def __init__(self, game_id, seed=0, scripts=None):
        super().__init__(game_id, seed)
        self.scripts = scripts or {}
        self.play = 0
        self.i = 0

    def on_play_start(self, play_index):
        self.play = play_index
        self.i = 0

    def choose_action(self, frames, latest_frame):
        script = self.scripts.get(self.play, [])
        a = script[self.i] if self.i < len(script) else GameAction.ACTION5
        self.i += 1
        return a, None


def run(agent_scripts, cap, mode="single_play", budget=100, seed=0, tag="cap-test"):
    def factory(game_id, s):
        return Scripted(game_id, s, scripts=agent_scripts)

    factory.agent_name = "scripted"
    cfg = RunConfig(
        max_actions_per_game=budget, seed=seed, mode=mode, tag=tag, quiet=True,
        environments_dir=TEST_ENVS, runs_dir="runs/cap-tests",
        per_level_action_cap_multiplier=cap,
    )
    return run_suite(["tt01"], factory, cfg)


def result(rec):
    return rec["results"][0]


A1, A5 = GameAction.ACTION1, GameAction.ACTION5

print("T0: level_action_cap unit semantics")
check("cap = int(mult*baseline)", level_action_cap([3, 3], 0, 5.0) == 15)
check("missing level index -> None", level_action_cap([3], 1, 5.0) is None)
check("baseline<=0 -> None", level_action_cap([0, -1], 0, 5.0) is None)
check("multiplier None -> None", level_action_cap([3], 0, None) is None)

print("T1: cap fires at exactly 5x (15 counted no-ops on level 0)")
r = result(run({0: [A5] * 50}, cap=5.0))
check("state LEVEL_CAPPED", r["state"] == "LEVEL_CAPPED", r["state"])
check("capped_at_level == 0", r["capped_at_level"] == 0)
check("fired at exactly 15 scored actions", r["scored_actions"] == 15,
      str(r["scored_actions"]))
check("no levels completed", r["levels_completed"] == 0)

print("T2: completing ON the cap-th action is allowed")
r = result(run({0: [A5] * 14 + [A1] + [A5] * 14 + [A1]}, cap=5.0))
check("no cap fired", r["capped_at_level"] is None, r["state"])
check("both levels completed", r["levels_completed"] == 2)
check("state WIN", r["state"] == "WIN", r["state"])

print("T3: counter resets per level (L0 in 1 action, L1 capped)")
r = result(run({0: [A1] + [A5] * 50}, cap=5.0))
check("capped_at_level == 1", r["capped_at_level"] == 1, str(r["capped_at_level"]))
check("scored = 1 (L0) + 15 (L1)", r["scored_actions"] == 16, str(r["scored_actions"]))
check("one level completed", r["levels_completed"] == 1)

print("T4: cap=None reproduces pre-cap behavior bit-for-bit on tt01")


def strip_wall(rec):
    d = dict(rec["results"][0])
    d.pop("wall_seconds")
    d.pop("env_step_s")
    return d, rec["game_scores"]


def rand_run(cap, seed=0):
    def factory(game_id, s):
        return RandomAgent(game_id, s)

    factory.agent_name = "random"
    cfg = RunConfig(
        max_actions_per_game=40, seed=seed, mode="two_phase", tag="cap-none",
        quiet=True, environments_dir=TEST_ENVS, runs_dir="runs/cap-tests",
        per_level_action_cap_multiplier=cap,
    )
    return run_suite(["tt01"], factory, cfg)


a = rand_run(None)
b = rand_run(None)
c = rand_run(1e9)  # cap machinery active but unreachable
ra, sa = strip_wall(a)
rb, sb = strip_wall(b)
rc, sc_ = strip_wall(c)
check("None vs None deterministic (control)", ra == rb and sa == sb)
check("None vs unreachable-cap identical (counting does not perturb)",
      ra == rc and sa == sc_)

print("T5: two_phase — capped replay ends the play, banked win survives")
r5 = run({0: [A5] * 5 + [A1] + [A5] * 5 + [A1], 1: [A5] * 50},
         cap=5.0, mode="two_phase", budget=100)
res5 = result(r5)
check("cap fired on replay play 1", res5["capped_play"] == 1,
      str(res5["capped_play"]))
check("capped at level 0 of the replay", res5["capped_at_level"] == 0)
check("banked levels survive", res5["levels_completed"] == 2)
gs = r5["game_scores"].get("tt01-000000", 0.0)
check("game score = sloppy play's 25.0 (max over plays unaffected)",
      abs(gs - 25.0) < 0.01, f"{gs:.2f}")
check("scored = 12 (play 0) + 15 (capped replay)",
      res5["scored_actions"] == 27, str(res5["scored_actions"]))

n_fail = sum(1 for _, ok in PASS if not ok)
print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed")
sys.exit(1 if n_fail else 0)
