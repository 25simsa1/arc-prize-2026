"""Time-budgeted forward search through the WorldModel to the next
LEVEL/WIN event.

Plans one level at a time: a level-advance changes the layout to a frame the
model may never have seen, so the agent replans after each level-up.
Expansion requires an event claim (GAME_OVER claims prune the branch); a
NONE event additionally needs a grid claim to produce the successor state.
Default admits VERIFIED rules only; allow_untested admits UNTESTED with a
per-step cost penalty so verified paths win ties.
"""

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .rules import ModelPrediction, RuleStatus, WorldModel
from .store import EVENT_GAME_OVER, EVENT_LEVEL, EVENT_WIN, frame_hash

GOAL_EVENTS = (EVENT_LEVEL, EVENT_WIN)


@dataclass
class PlanStep:
    action_key: str
    grid_status: Optional[RuleStatus]
    event_status: Optional[RuleStatus]

    @property
    def fully_verified(self) -> bool:
        ok = self.event_status == RuleStatus.VERIFIED
        if self.grid_status is not None:
            ok = ok and self.grid_status == RuleStatus.VERIFIED
        return ok


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    found_goal: bool = False
    partial: bool = False
    nodes_expanded: int = 0
    reason: str = ""

    @property
    def actions(self) -> list[str]:
        return [s.action_key for s in self.steps]

    @property
    def confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.fully_verified) / len(self.steps)


def plan_to_next_level(
    model: WorldModel,
    level: int,
    start: np.ndarray,
    simple_actions: list[str],
    click_targets: Callable[[np.ndarray], list[str]],
    deadline: float,
    allow_untested: bool = False,
    untested_penalty: float = 0.25,
    max_depth: int = 64,
    max_nodes: int = 20000,
) -> Plan:
    allowed = (
        (RuleStatus.VERIFIED, RuleStatus.UNTESTED)
        if allow_untested
        else (RuleStatus.VERIFIED,)
    )
    counter = itertools.count()
    start_h = frame_hash(start)
    frontier: list = [(0.0, next(counter), start, [])]
    seen: set[str] = {start_h}
    best_partial: list[PlanStep] = []
    nodes = 0

    while frontier:
        if time.monotonic() > deadline or nodes >= max_nodes:
            return Plan(best_partial, False, True, nodes, "budget_exhausted")
        cost, _, grid, path = heapq.heappop(frontier)
        nodes += 1
        if len(path) >= max_depth:
            continue

        candidates = list(simple_actions)
        candidates += click_targets(grid)

        for ak in candidates:
            p: ModelPrediction = model.predict(level, grid, ak, allowed=allowed)
            if p.event is None:
                continue  # no event claim: cannot risk stepping blind
            step = PlanStep(ak, p.grid_status if p.grid is not None else None, p.event_status)
            if p.event in GOAL_EVENTS:
                return Plan(path + [step], True, False, nodes, "goal")
            if p.event == EVENT_GAME_OVER:
                continue  # hazard: prune
            if p.grid is None:
                continue  # eventless step with unknown successor
            nh = frame_hash(p.grid)
            if nh in seen:
                continue
            seen.add(nh)
            new_path = path + [PlanStep(ak, p.grid_status, p.event_status)]
            if len(new_path) > len(best_partial):
                best_partial = new_path
            penalty = untested_penalty * sum(
                1
                for s in new_path
                if s.event_status == RuleStatus.UNTESTED
                or s.grid_status == RuleStatus.UNTESTED
            )
            heapq.heappush(frontier, (len(new_path) + penalty, next(counter), p.grid, new_path))

    return Plan(best_partial, False, True, nodes, "exhausted_no_goal")
