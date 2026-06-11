"""World-model core tests on synthetic 8x8 environments with KNOWN dynamics.

Scenario 0 (pre-R1 regression): clean env — true rules recovered VERIFIED,
optimal plan, CONTRADICTED rule excluded from default planning.

Scenario A (R1): same env + a planted 1-cell ticker at (0,7) cycling
10->11->12 on EVERY action (the cd82 HUD phenomenon in miniature). Asserts:
  (a) translate/blocked rules reach VERIFIED with the ticker masked,
  (b) the ticker region itself gets a VERIFIED hud_state rule,
  (d) with factoring DISABLED the pre-R1 pathology reproduces exactly:
      no grid template fits, grid coverage 0.0 — the paper's ablation.
Also: the planner still finds the optimal 10-step path under masked state
hashing (ticker phases must not triple the search space).

Scenario C (over-masking trap): a goal-colored cell at (0,6), adjacent to
the ticker, blinking every 7th action — spatially it clusters WITH the
ticker and would be masked; the unmaskable-colors guard must keep it out.
(Kept separate from Scenario A because a rarely-blinking unmasked cell
legitimately CONTRADICTS translate rules — that interaction is real, and
mixing it in would mask assertion failures.)

Scenario R1'-counter (rendered-counter trap, the r11l failure mode): TWO
HUD cells change on EVERY action — a countdown and a spinner, both exact
functions of actions-since-level-start, resetting on respawn — so no diff
is ever a sole change and the sole-changer seed can never fire. Asserts
R1'-on masks exactly the counter pair and grid templates verify; R1'-off
reproduces today's behavior bit-for-bit (nothing masked, coverage 0.0).

Scenario R1'-click-guard: a pure-click game where a board cell is clicked
at the SAME action indices every episode (its toggle values are perfectly
idx-predictable — the short-window trap from the r11l diagnosis), plus the
dual rendered counter. Asserts the click-board guard keeps the interactive
cell unmasked while the counter is masked; disabling the guard proves the
trap is armed.

    .venv/bin/python scripts/test_wm_core.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.planner import plan_to_next_level
from harness.wm.proposers import TemplateProposer
from harness.wm.regions import RegionAnalyzer
from harness.wm.rules import Prediction, Rule, RuleStatus, WorldModel
from harness.wm.store import EVENT_GAME_OVER, EVENT_LEVEL, EVENT_NONE, TransitionStore
from harness.wm.verifier import verify_rules

PLAYER, GOAL, WALL, HAZARD, BG = 3, 2, 5, 9, 0
TICKER_COLORS = [10, 11, 12]
MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


def make_level(ticker: bool = False, blink_goal: bool = False) -> np.ndarray:
    g = np.zeros((8, 8), dtype=np.int16)
    g[3, 1:7] = WALL          # horizontal wall with a gap
    g[3, 4] = BG
    # HUD corner: walled off like real games separate HUD from play area
    # (cd82: meter row 63 vs content rows 21-37) — without spatial
    # separation, multi-base player-path cells cluster into the HUD region
    # and the size guard rejects the merged blob.
    g[0:3, 5:8] = WALL
    g[1, 1] = PLAYER
    g[6, 6] = GOAL
    g[6, 1] = HAZARD
    if ticker:
        g[0, 7] = TICKER_COLORS[0]
    if blink_goal:
        g[0, 6] = GOAL        # goal-colored cell right next to the ticker
    return g


def true_step(grid: np.ndarray, action: str, step_idx: int,
              ticker: bool = False, blink_goal: bool = False) -> tuple[np.ndarray, str]:
    dy, dx = MOVES[action]
    (py, px) = [(int(y), int(x)) for y, x in np.argwhere(grid == PLAYER)][0]
    ny, nx = py + dy, px + dx
    out = grid.copy()
    hud_cells = ({(0, 7)} if ticker else set()) | ({(0, 6)} if blink_goal else set())
    if (
        not (0 <= ny < 8 and 0 <= nx < 8)
        or grid[ny, nx] == WALL
        or (ny, nx) in hud_cells  # HUD sits outside the play area
    ):
        event = EVENT_NONE
    elif grid[ny, nx] == GOAL and (ny, nx) == (6, 6):
        event = EVENT_LEVEL
    elif grid[ny, nx] == HAZARD:
        event = EVENT_GAME_OVER
    else:
        out[py, px] = BG
        out[ny, nx] = PLAYER
        event = EVENT_NONE
    if ticker:  # ticks on EVERY action, like cd82's meter
        out[0, 7] = TICKER_COLORS[(step_idx + 1) % 3]
    if blink_goal and (step_idx + 1) % 7 == 0:
        out[0, 6] = 12 if grid[0, 6] == GOAL else GOAL
    return out, event


def collect_bfs(store: TransitionStore, depth: int = 20) -> None:
    """Full closure of the CLEAN env (no ticker; dynamics are stationary)."""
    start = make_level()
    frontier = [start]
    seen = {start.tobytes()}
    for _ in range(depth):
        nxt = []
        for g in frontier:
            for a in MOVES:
                post, event = true_step(g, a, 0)
                state = {"NONE": "NOT_FINISHED", "LEVEL": "NOT_FINISHED",
                         "GAME_OVER": "GAME_OVER"}[event]
                store.add(0, g, a, post, 1 if event == EVENT_LEVEL else 0, state)
                if event == EVENT_NONE and post.tobytes() not in seen:
                    seen.add(post.tobytes())
                    nxt.append(post)
        frontier = nxt


def collect_walk(store: TransitionStore, steps: int, ticker: bool,
                 blink_goal: bool, analyzer: RegionAnalyzer | None = None,
                 make_fn=None) -> None:
    """Deterministic random walk through the NON-stationary env (ticker needs
    a global action counter, so BFS closure doesn't apply)."""
    make_fn = make_fn or make_level
    rng = random.Random(7)
    g = make_fn(ticker, blink_goal)
    for i in range(steps):
        a = rng.choice(list(MOVES))
        post, event = true_step(g, a, i, ticker, blink_goal)
        state = {"NONE": "NOT_FINISHED", "LEVEL": "NOT_FINISHED",
                 "GAME_OVER": "GAME_OVER"}[event]
        status, t = store.add(0, g, a, post, 1 if event == EVENT_LEVEL else 0, state)
        if analyzer is not None and status == "new":
            analyzer.observe(t)
        if event == EVENT_LEVEL or event == EVENT_GAME_OVER:
            # respawn, preserving ticker/blink phase
            fresh = make_fn(ticker, blink_goal)
            fresh[0, 7] = post[0, 7] if ticker else fresh[0, 7]
            if blink_goal:
                fresh[0, 6] = post[0, 6]
            g = fresh
        else:
            g = post


def build_model(store: TransitionStore, hud_mask, region_map) -> WorldModel:
    model = WorldModel()
    model.hud_mask = hud_mask
    model.region_map = region_map
    model.rules = TemplateProposer().propose(store, model)
    verify_rules(model.rules, store)
    model.recompute_coverage(store)
    return model


def scenario_0_regression() -> None:
    store = TransitionStore("toy")
    collect_bfs(store)
    model = build_model(store, None, None)
    assert model.coverage_predicted == 1.0 and model.coverage_exact == 1.0
    plan = plan_to_next_level(model, 0, make_level(), list(MOVES), lambda g: [],
                              deadline=time.monotonic() + 10.0, allow_untested=True)
    assert plan.found_goal and len(plan.steps) == 10, plan.reason

    lie = Rule("lie[ACTION1-identity]", "lie", {},
               lambda level, pre, ak: Prediction(grid=pre.copy()) if ak == "ACTION1" else None,
               "test", specificity=99)
    rules2 = model.rules + [lie]
    verify_rules(rules2, store)
    assert lie.status == RuleStatus.CONTRADICTED
    model2 = WorldModel(rules=rules2)
    model2.hud_mask = None
    plan2 = plan_to_next_level(model2, 0, make_level(), list(MOVES), lambda g: [],
                               deadline=time.monotonic() + 10.0, allow_untested=True)
    assert plan2.found_goal and len(plan2.steps) == 10
    print("scenario 0 (pre-R1 regression): PASS")


def scenario_a_ticker() -> None:
    store = TransitionStore("toy-ticker")
    analyzer = RegionAnalyzer()
    collect_walk(store, 900, ticker=True, blink_goal=False, analyzer=analyzer)

    # (d) ablation first: factoring disabled reproduces the cd82 pathology
    off = build_model(store, None, None)
    grid_rules_off = [r for r in off.rules if r.region in ("full", "dynamic")
                      and r.name in ("identity", "translate", "blocked_identity")]
    assert not grid_rules_off, f"ticker should kill all grid templates, got {grid_rules_off}"
    assert off.coverage_predicted == 0.0, "ablation must reproduce grid coverage 0.0"

    # factoring on
    region_map = analyzer.analyze(unmaskable_colors={GOAL, HAZARD})
    assert region_map.hud_regions, "ticker region not detected"
    hud = region_map.hud_mask
    assert hud is not None and hud[0, 7] and hud.sum() == 1, "exactly the ticker cell"

    model = build_model(store, hud, region_map)
    ver = {r.rule_id: r.status for r in model.rules}

    # (a) translate/blocked VERIFIED with the ticker masked
    for a, (dy, dx) in MOVES.items():
        tr = [r for r in model.rules if r.name == "translate"
              and r.params["action"] == a and r.status == RuleStatus.VERIFIED]
        assert tr, f"translate[{a}] not VERIFIED under masking: {ver}"
    bl = [r for r in model.rules if r.name == "blocked_identity"
          and r.status == RuleStatus.VERIFIED]
    assert bl, "blocked_identity not VERIFIED under masking"
    assert model.coverage_exact == 1.0, "masked grid predictions must be exact"
    assert model.coverage_predicted > 0.9, "grid coverage must be unblocked"

    # (b) the ticker itself gets a VERIFIED hud rule (not dropped on the floor)
    hud_rules = [r for r in model.rules if r.region == "hud"
                 and r.status == RuleStatus.VERIFIED]
    assert hud_rules, f"no VERIFIED hud rule: {ver}"
    assert model.hud_exact == 1.0 and model.hud_predicted > 0.9

    # planner under masked hashing: ticker phases must not blow up the search
    plan = plan_to_next_level(model, 0, make_level(ticker=True), list(MOVES),
                              lambda g: [], deadline=time.monotonic() + 10.0,
                              allow_untested=True)
    assert plan.found_goal and len(plan.steps) == 10, (plan.reason, len(plan.steps))
    print("scenario A (ticker masked, hud modeled, ablation reproduces): PASS")


def scenario_c_goal_trap() -> None:
    store = TransitionStore("toy-trap")
    analyzer = RegionAnalyzer()

    # Trap geometry: the blink cell + ticker must form an ISOLATED cluster.
    # Player-path cells are multi-base changers too, and in this cramped 8x8
    # they bridge clusters within chebyshev-2; sealing the corridor column
    # above the wall gap isolates the HUD corner. (This scenario asserts
    # masking only — the goal becomes unreachable and that's fine here.)
    def make_trap(ticker: bool, blink_goal: bool) -> np.ndarray:
        g = make_level(ticker, blink_goal)
        g[0:3, 4] = WALL
        return g

    collect_walk(store, 900, ticker=True, blink_goal=True, analyzer=analyzer,
                 make_fn=make_trap)

    # without the guard, the blinking goal-colored cell clusters with the
    # ticker and gets masked — the trap is real...
    unguarded = analyzer.analyze(unmaskable_colors=())
    masked_cells = {c for r in unguarded.hud_regions for c in map(tuple, r.cells)}
    assert (0, 6) in masked_cells, (
        "trap not armed: blink cell did not cluster with ticker "
        f"(masked={masked_cells})"
    )

    # ...and the unmaskable-colors guard disarms it
    guarded = analyzer.analyze(unmaskable_colors={GOAL, HAZARD})
    masked_cells = {c for r in guarded.hud_regions for c in map(tuple, r.cells)}
    assert (0, 6) not in masked_cells, "goal-colored cell was masked despite guard"
    assert (0, 7) in masked_cells, "ticker should still be masked"
    print("scenario C (goal cell never masked): PASS")


# ---- R1' scenarios: rendered counters that NEVER solo (r11l class) ----
CTR_A, CTR_B = (0, 6), (0, 7)  # dual HUD display inside the walled corner


def counter_post(out: np.ndarray, idx: int) -> None:
    """HUD values after the action at index `idx` since level start: BOTH
    cells change every action (so no diff is ever a sole change or a bare
    pair) and both are exact functions of idx — r11l's rendered 60->0 step
    counter in miniature, with the per-level reset."""
    out[CTR_A] = 149 - (idx % 49)   # countdown
    out[CTR_B] = 200 + ((idx + 1) % 3)  # spinner


def make_counter_level() -> np.ndarray:
    g = make_level()
    g[CTR_A] = 150
    g[CTR_B] = 200
    return g


def counter_step(grid: np.ndarray, action: str, idx: int) -> tuple[np.ndarray, str]:
    out, event = true_step(grid, action, 0, ticker=False, blink_goal=False)
    counter_post(out, idx)
    return out, event


def collect_counter_walk(store: TransitionStore, steps: int,
                         analyzers: list[RegionAnalyzer]) -> None:
    rng = random.Random(11)
    g = make_counter_level()
    idx = 0
    for _ in range(steps):
        a = rng.choice(list(MOVES))
        post, event = counter_step(g, a, idx)
        state = {"NONE": "NOT_FINISHED", "LEVEL": "NOT_FINISHED",
                 "GAME_OVER": "GAME_OVER"}[event]
        status, t = store.add(0, g, a, post, 1 if event == EVENT_LEVEL else 0, state)
        if status == "new":
            for an in analyzers:
                an.observe(t, idx=idx)
        if event in (EVENT_LEVEL, EVENT_GAME_OVER):
            g = make_counter_level()  # rendered counter RESETS at level start
            idx = 0
        else:
            g = post
            idx += 1


def scenario_r1prime_counter() -> None:
    store = TransitionStore("toy-counter")
    on = RegionAnalyzer()                 # r1prime defaults ON
    off = RegionAnalyzer(r1prime=False)
    collect_counter_walk(store, 900, [on, off])

    # regression baseline: the counters never solo, so R1'-off (today's
    # detector) must NOT mask them...
    off_map = off.analyze(unmaskable_colors={GOAL, HAZARD})
    assert not off_map.hud_regions, (
        f"R1'-off must reproduce today's miss, got {off_map.hud_regions}")
    # ...which reproduces the r11l pathology: no grid template fits
    off_model = build_model(store, None, None)
    grid_rules_off = [r for r in off_model.rules if r.region in ("full", "dynamic")
                      and r.name in ("identity", "translate", "blocked_identity")]
    assert not grid_rules_off, f"counter should kill grid templates, got {grid_rules_off}"
    assert off_model.coverage_predicted == 0.0

    # R1' on: exactly the counter pair masked, nothing from the board
    on_map = on.analyze(unmaskable_colors={GOAL, HAZARD})
    masked = {c for r in on_map.hud_regions for c in map(tuple, r.cells)}
    assert masked == {CTR_A, CTR_B}, f"expected counter pair masked, got {masked}"

    model = build_model(store, on_map.hud_mask, on_map)
    ver = {r.rule_id: r.status for r in model.rules}
    for a in MOVES:
        tr = [r for r in model.rules if r.name == "translate"
              and r.params["action"] == a and r.status == RuleStatus.VERIFIED]
        assert tr, f"translate[{a}] not VERIFIED under R1' masking: {ver}"
    bl = [r for r in model.rules if r.name == "blocked_identity"
          and r.status == RuleStatus.VERIFIED]
    assert bl, "blocked_identity not VERIFIED under R1' masking"
    assert model.coverage_exact == 1.0, "masked grid predictions must be exact"
    assert model.coverage_predicted > 0.9, "grid coverage must be unblocked"
    print("scenario R1'-counter (never-solo counter masked, templates verify, "
          "off-switch preserves baseline): PASS")


def scenario_r1prime_click_guard() -> None:
    TRAP = (5, 5)  # interactive cell, clicked on a fixed schedule

    def make_click_level() -> np.ndarray:
        g = np.zeros((8, 8), dtype=np.int16)
        g[0:3, 5:8] = WALL
        g[CTR_A] = 150
        g[CTR_B] = 200
        return g

    def click_step(grid: np.ndarray, x: int, y: int, idx: int) -> np.ndarray:
        out = grid.copy()
        out[y, x] = 7 if grid[y, x] == 0 else 0  # click toggles the board cell
        counter_post(out, idx)
        return out

    store = TransitionStore("toy-click")
    on = RegionAnalyzer()
    noguard = RegionAnalyzer(click_dep_rate=1.1)  # guard off: arms the trap
    off = RegionAnalyzer(r1prime=False)
    rng = random.Random(13)
    board = [(y, x) for y in range(3, 8) for x in range(8) if (y, x) != TRAP]
    # 6-click episodes: idx 0 is a random board click (episodes diverge there,
    # defeating store dedup), idx 1..5 always click TRAP — so TRAP's toggle
    # sequence 7,0,7,0,7 is a perfect function of idx, with heavy repeat
    # evidence. Only the click-board guard can tell it from a counter.
    for _ in range(120):
        g = make_click_level()
        for idx in range(6):
            y, x = TRAP if idx >= 1 else board[rng.randrange(len(board))]
            post = click_step(g, x, y, idx)
            state = "GAME_OVER" if idx == 5 else "NOT_FINISHED"
            status, t = store.add(0, g, f"ACTION6:{x},{y}", post, 0, state)
            if status == "new":
                for an in (on, noguard, off):
                    an.observe(t, idx=idx)
            g = post

    # trap armed: without the guard, the idx-predictable board cell IS masked
    armed = noguard.analyze(unmaskable_colors={GOAL, HAZARD})
    armed_cells = {c for r in armed.hud_regions for c in map(tuple, r.cells)}
    assert TRAP in armed_cells, (
        f"trap not armed: guardless R1' did not mask the board cell "
        f"(masked={armed_cells})")

    # ...and the click-board guard disarms it: counters masked, board intact
    guarded = on.analyze(unmaskable_colors={GOAL, HAZARD})
    masked = {c for r in guarded.hud_regions for c in map(tuple, r.cells)}
    assert masked == {CTR_A, CTR_B}, (
        f"expected exactly the counters masked, got {masked}")
    assert TRAP not in masked, "interactive board cell masked despite guard"

    # regression baseline: nothing ever solos, R1'-off masks nothing
    off_map = off.analyze(unmaskable_colors={GOAL, HAZARD})
    assert not off_map.hud_regions
    print("scenario R1'-click-guard (board survives, counters masked): PASS")


if __name__ == "__main__":
    scenario_0_regression()
    scenario_a_ticker()
    scenario_c_goal_trap()
    scenario_r1prime_counter()
    scenario_r1prime_click_guard()
    print("\nALL CORE TESTS PASS")
