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

    def predict(self, level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
        return self.fn(level, pre, action_key)


_STATUS_RANK = {RuleStatus.VERIFIED: 0, RuleStatus.UNTESTED: 1, RuleStatus.CONTRADICTED: 2}


@dataclass
class ModelPrediction:
    grid: Optional[np.ndarray] = None
    event: Optional[str] = None
    grid_status: Optional[RuleStatus] = None
    event_status: Optional[RuleStatus] = None
    grid_rule: Optional[str] = None
    event_rule: Optional[str] = None

    @property
    def complete(self) -> bool:
        return self.grid is not None and self.event is not None


@dataclass
class WorldModel:
    rules: list[Rule] = field(default_factory=list)
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
        """Compose the first grid claim and first event claim from eligible
        rules in order. CONTRADICTED rules never participate unless asked."""
        out = ModelPrediction()
        for rule in self.ordered_rules():
            if rule.status not in allowed:
                continue
            if out.grid is not None and out.event is not None:
                break
            p = rule.predict(level, pre, action_key)
            if p is None:
                continue
            if p.grid is not None and out.grid is None:
                out.grid = p.grid
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
            if np.array_equal(p.grid, t.post):
                self._g_exact += 1
        if p.event is not None:
            self._e_claimed += 1
            if p.event == t.event:
                self._e_exact += 1

    def recompute_coverage(self, store: TransitionStore) -> None:
        self._cov_n = self._g_claimed = self._g_exact = 0
        self._e_claimed = self._e_exact = 0
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
        }
