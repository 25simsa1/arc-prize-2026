"""Rule and WorldModel.

A Rule predicts grid and/or event for transitions it claims; returning None
means NO_PREDICTION ("not my transition"), which is legal partial coverage,
not failure. CONTRADICTED rules are retained (their misses are refinement
data) but excluded from prediction and default planning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .store import TransitionStore


class RuleStatus(str, Enum):
    UNTESTED = "UNTESTED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class Prediction:
    grid: Optional[np.ndarray] = None  # None = no grid claim
    event: Optional[str] = None        # None = no event claim
    mask: Optional[np.ndarray] = None  # bool: cells the grid claim covers
    #                                    (None = whole frame). Verification
    #                                    compares ONLY within the mask.


# predict(level, pre_grid, action_key) -> Prediction | None
PredictFn = Callable[[int, np.ndarray, str], Optional[Prediction]]


@dataclass
class Rule:
    rule_id: str
    name: str
    params: dict
    fn: PredictFn
    proposer: str
    fit_count: int = 0           # exact fits at proposal time (provenance)
    status: RuleStatus = RuleStatus.UNTESTED
    n_exact: int = 0
    n_miss: int = 0
    specificity: int = 0         # higher = more specific; ordering tiebreak
    region: str = "full"         # "full" | "dynamic" | "hud" — claim scope class

    def predict(self, level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        return self.fn(level, pre, action_key)


def grids_match(pred: Prediction, post: np.ndarray) -> bool:
    """Exact comparison within the prediction's claimed region."""
    if pred.grid is None:
        return True
    if pred.mask is None:
        return bool(np.array_equal(pred.grid, post))
    return bool(np.array_equal(pred.grid[pred.mask], post[pred.mask]))


_STATUS_RANK = {RuleStatus.VERIFIED: 0, RuleStatus.UNTESTED: 1, RuleStatus.CONTRADICTED: 2}


@dataclass
class ModelPrediction:
    grid: Optional[np.ndarray] = None        # dynamic-scope grid claim
    grid_mask: Optional[np.ndarray] = None
    event: Optional[str] = None
    grid_status: Optional[RuleStatus] = None
    event_status: Optional[RuleStatus] = None
    grid_rule: Optional[str] = None
    event_rule: Optional[str] = None
    hud_grid: Optional[np.ndarray] = None    # ALWAYS_CHANGING-region claim
    hud_mask: Optional[np.ndarray] = None
    hud_status: Optional[RuleStatus] = None
    hud_rule: Optional[str] = None

    @property
    def complete(self) -> bool:
        return self.grid is not None and self.event is not None

    def successor(self, pre: np.ndarray) -> Optional[np.ndarray]:
        """Compose claimed regions into a full successor grid; unclaimed cells
        carry the pre values (deterministic placeholder for planning)."""
        if self.grid is None:
            return None
        out = pre.copy()
        if self.grid_mask is None:
            out = self.grid.copy()
        else:
            out[self.grid_mask] = self.grid[self.grid_mask]
        if self.hud_grid is not None and self.hud_mask is not None:
            out[self.hud_mask] = self.hud_grid[self.hud_mask]
        return out


@dataclass
class WorldModel:
    rules: list[Rule] = field(default_factory=list)
    # Region factoring (R1): bool mask of ALWAYS_CHANGING cells, set by the
    # agent from RegionAnalyzer output. None = factoring off/nothing found —
    # exact pre-R1 behavior (this is the paper's ablation switch).
    hud_mask: Optional[np.ndarray] = None
    model_version: int = 0  # bumped whenever rules/mask change (plan caching)
    # Coverage counters. Maintained incrementally via observe_for_coverage()
    # as transitions arrive (O(rules) per action) and rebuilt from scratch by
    # recompute_coverage() whenever the rule set changes — full per-action
    # recomputation would be O(store x rules), the same cost trap that made
    # unthrottled propose() eat 96% of wall-clock.
    _cov_n: int = 0
    _g_claimed: int = 0
    _g_exact: int = 0
    _e_claimed: int = 0
    _e_exact: int = 0
    _h_claimed: int = 0   # HUD-region grid claims
    _h_exact: int = 0

    @property
    def coverage_predicted(self) -> float:
        return self._g_claimed / self._cov_n if self._cov_n else 0.0

    @property
    def coverage_exact(self) -> float:
        return self._g_exact / self._g_claimed if self._g_claimed else 0.0

    @property
    def event_predicted(self) -> float:
        return self._e_claimed / self._cov_n if self._cov_n else 0.0

    @property
    def event_exact(self) -> float:
        return self._e_exact / self._e_claimed if self._e_claimed else 0.0

    @property
    def hud_predicted(self) -> float:
        return self._h_claimed / self._cov_n if self._cov_n else 0.0

    @property
    def hud_exact(self) -> float:
        return self._h_exact / self._h_claimed if self._h_claimed else 0.0

    def status_counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in RuleStatus}
        for r in self.rules:
            out[r.status.value] += 1
        return out

    def ordered_rules(self) -> list[Rule]:
        return sorted(
            self.rules,
            key=lambda r: (_STATUS_RANK[r.status], -r.specificity, r.rule_id),
        )

    def predict(
        self,
        level: int,
        pre: np.ndarray,
        action_key: str,
        allowed: tuple[RuleStatus, ...] = (RuleStatus.VERIFIED, RuleStatus.UNTESTED),
    ) -> ModelPrediction:
        """Compose the first dynamic-scope grid claim, first HUD-region grid
        claim, and first event claim from eligible rules in order.
        CONTRADICTED rules never participate unless asked."""
        out = ModelPrediction()
        for rule in self.ordered_rules():
            if rule.status not in allowed:
                continue
            if out.grid is not None and out.event is not None and out.hud_grid is not None:
                break
            p = rule.predict(level, pre, action_key)
            if p is None:
                continue
            if p.grid is not None:
                if rule.region == "hud":
                    if out.hud_grid is None:
                        out.hud_grid = p.grid
                        out.hud_mask = p.mask
                        out.hud_status = rule.status
                        out.hud_rule = rule.rule_id
                elif out.grid is None:
                    out.grid = p.grid
                    out.grid_mask = p.mask
                    out.grid_status = rule.status
                    out.grid_rule = rule.rule_id
            if p.event is not None and out.event is None:
                out.event = p.event
                out.event_status = rule.status
                out.event_rule = rule.rule_id
        return out

    def observe_for_coverage(self, t) -> None:
        """Incremental coverage update for ONE newly stored transition."""
        p = self.predict(t.level, t.pre, t.action_key)
        self._cov_n += 1
        if p.grid is not None:
            self._g_claimed += 1
            if grids_match(Prediction(grid=p.grid, mask=p.grid_mask), t.post):
                self._g_exact += 1
        if p.hud_grid is not None:
            self._h_claimed += 1
            if grids_match(Prediction(grid=p.hud_grid, mask=p.hud_mask), t.post):
                self._h_exact += 1
        if p.event is not None:
            self._e_claimed += 1
            if p.event == t.event:
                self._e_exact += 1

    def recompute_coverage(self, store: TransitionStore) -> None:
        self._cov_n = self._g_claimed = self._g_exact = 0
        self._e_claimed = self._e_exact = 0
        self._h_claimed = self._h_exact = 0
        for t in store.all():
            self.observe_for_coverage(t)

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for r in self.rules:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "rules": len(self.rules),
            "by_status": by_status,
            "grid_coverage": round(self.coverage_predicted, 3),
            "grid_exact": round(self.coverage_exact, 3),
            "event_coverage": round(self.event_predicted, 3),
            "event_exact": round(self.event_exact, 3),
            "hud_coverage": round(self.hud_predicted, 3),
            "hud_exact": round(self.hud_exact, 3),
            "hud_cells": int(self.hud_mask.sum()) if self.hud_mask is not None else 0,
        }
