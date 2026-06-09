"""World-model core tests on a synthetic 8x8 environment with KNOWN dynamics.

True dynamics: player (color 3) moves up/down/left/right (ACTION1-4); walls
(color 5) block; moving onto goal (color 2) advances the level; moving onto
hazard (color 9) is GAME_OVER. Asserts:
  1. TemplateProposer recovers translate/blocked/move-onto rules and the
     verifier VERIFIES them (>=3 exact, 0 misses).
  2. Planner finds the known-optimal path using the learned model.
  3. A CONTRADICTED rule (planted falsehood) is excluded from default
     planning and the plan is still correct.

    .venv/bin/python scripts/test_wm_core.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.planner import plan_to_next_level
from harness.wm.proposers import TemplateProposer
from harness.wm.rules import Prediction, Rule, RuleStatus, WorldModel
from harness.wm.store import EVENT_GAME_OVER, EVENT_LEVEL, EVENT_NONE, TransitionStore
from harness.wm.verifier import verify_rules

PLAYER, GOAL, WALL, HAZARD, BG = 3, 2, 5, 9, 0
MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


def make_level() -> np.ndarray:
    g = np.zeros((8, 8), dtype=np.int16)
    g[3, 1:7] = WALL          # horizontal wall with a gap
    g[3, 4] = BG
    g[1, 1] = PLAYER
    g[6, 6] = GOAL
    g[6, 1] = HAZARD
    return g


def true_step(grid: np.ndarray, action: str) -> tuple[np.ndarray, str]:
    dy, dx = MOVES[action]
    (py, px) = [(int(y), int(x)) for y, x in np.argwhere(grid == PLAYER)][0]
    ny, nx = py + dy, px + dx
    if not (0 <= ny < 8 and 0 <= nx < 8) or grid[ny, nx] == WALL:
        return grid.copy(), EVENT_NONE
    if grid[ny, nx] == GOAL:
        return grid.copy(), EVENT_LEVEL
    if grid[ny, nx] == HAZARD:
        return grid.copy(), EVENT_GAME_OVER
    out = grid.copy()
    out[py, px] = BG
    out[ny, nx] = PLAYER
    return out, EVENT_NONE


def collect_transitions(store: TransitionStore, depth: int = 20) -> None:
    """BFS over true dynamics from the start state, recording everything."""
    start = make_level()
    frontier = [start]
    seen = {start.tobytes()}
    for _ in range(depth):
        nxt = []
        for g in frontier:
            for a in MOVES:
                post, event = true_step(g, a)
                state = {"NONE": "NOT_FINISHED", "LEVEL": "NOT_FINISHED",
                         "GAME_OVER": "GAME_OVER"}[event]
                post_level = 1 if event == EVENT_LEVEL else 0
                store.add(0, g, a, post, post_level, state)
                if event == EVENT_NONE and post.tobytes() not in seen:
                    seen.add(post.tobytes())
                    nxt.append(post)
        frontier = nxt


def main() -> None:
    store = TransitionStore("toy")
    collect_transitions(store)
    print(f"toy store: {len(store)} transitions, {len(store.conflicts)} conflicts")
    assert len(store.conflicts) == 0

    rules = TemplateProposer().propose(store)
    verify_rules(rules, store)
    model = WorldModel(rules=rules)
    model.recompute_coverage(store)
    print("model:", model.summary())

    def find(prefix: str, status: RuleStatus) -> list[Rule]:
        return [r for r in rules if r.rule_id.startswith(prefix) and r.status == status]

    # 1. true rules recovered & verified
    for a, (dy, dx) in MOVES.items():
        tr = find(f"translate[{a},c{PLAYER},{dy:+d},{dx:+d}", RuleStatus.VERIFIED)
        assert tr, f"translate rule for {a} not VERIFIED"
        bl = find(f"blocked[{a},c{PLAYER}", RuleStatus.VERIFIED)
        assert bl, f"blocked-identity rule for {a} not VERIFIED"
    goal_rules = [r for r in rules if r.name == "move_onto"
                  and r.params["target"] == GOAL and r.params["event"] == EVENT_LEVEL
                  and r.status != RuleStatus.CONTRADICTED]
    hazard_rules = [r for r in rules if r.name == "move_onto"
                    and r.params["target"] == HAZARD and r.params["event"] == EVENT_GAME_OVER
                    and r.status != RuleStatus.CONTRADICTED]
    assert goal_rules and hazard_rules, "move-onto goal/hazard rules missing"
    print(f"recovered: 4x translate+blocked VERIFIED, goal rules={len(goal_rules)}, "
          f"hazard rules={len(hazard_rules)}")
    assert model.coverage_exact == 1.0, "grid predictions must be exact on stored data"

    # 2. planner finds the known-optimal path (start (1,1) -> goal (6,6),
    # through the wall gap at (3,4): manhattan-with-gap optimum = 10 moves)
    plan = plan_to_next_level(
        model, 0, make_level(), list(MOVES), lambda g: [],
        deadline=time.monotonic() + 10.0, allow_untested=True,
    )
    assert plan.found_goal, f"no plan found: {plan.reason}"
    print(f"plan: {len(plan.steps)} steps, confidence={plan.confidence:.2f}, "
          f"actions={plan.actions}")
    assert len(plan.steps) == 10, f"expected optimal 10 steps, got {len(plan.steps)}"

    # simulate the plan against TRUE dynamics: must reach the goal w/o hazard
    g = make_level()
    for i, a in enumerate(plan.actions):
        g, event = true_step(g, a)
        assert event != EVENT_GAME_OVER, f"plan stepped into hazard at {i}"
    assert event == EVENT_LEVEL, "plan did not end on the goal"

    # 3. planted falsehood -> CONTRADICTED -> excluded from default planning
    lie = Rule(
        "lie[ACTION1-identity]", "lie", {},
        lambda level, pre, ak: Prediction(grid=pre.copy()) if ak == "ACTION1" else None,
        "test", specificity=99,  # outranks real rules if not excluded
    )
    rules_with_lie = rules + [lie]
    verify_rules(rules_with_lie, store)
    assert lie.status == RuleStatus.CONTRADICTED, "false rule must be CONTRADICTED"
    model2 = WorldModel(rules=rules_with_lie)
    plan2 = plan_to_next_level(
        model2, 0, make_level(), list(MOVES), lambda g: [],
        deadline=time.monotonic() + 10.0, allow_untested=True,
    )
    assert plan2.found_goal and len(plan2.steps) == 10, "CONTRADICTED rule leaked into planning"
    g = make_level()
    for a in plan2.actions:
        g, event = true_step(g, a)
    assert event == EVENT_LEVEL
    print("CONTRADICTED exclusion: plan unchanged and still correct")

    print("\nALL CORE TESTS PASS")


if __name__ == "__main__":
    main()
