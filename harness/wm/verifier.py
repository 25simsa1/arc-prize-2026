"""Verifier: exact replay of rules against all in-scope stored transitions.

Determinism is load-bearing (verified empirically per game elsewhere): a
prediction either matches the stored outcome exactly or the rule is wrong.
No fuzzy matching, no tolerance.

VERIFIED:     >= min_exact exact predictions and zero misses.
CONTRADICTED: any miss. Retained, marked — misses are refinement data.
UNTESTED:     otherwise (including rules that never fire on stored data).
"""

import time
from typing import Optional

import numpy as np

from .rules import Rule, RuleStatus
from .store import TransitionStore

DEFAULT_MIN_EXACT = 3


def verify_rules(
    rules: list[Rule],
    store: TransitionStore,
    min_exact: int = DEFAULT_MIN_EXACT,
    deadline: Optional[float] = None,  # time.monotonic() deadline
) -> dict:
    """Set status/n_exact/n_miss on every rule. Returns a summary dict.

    On deadline exhaustion remaining rules keep their previous status; the
    summary marks the verification as truncated (callers treat truncated
    verification as 'no status upgrades', never as silent success).
    """
    transitions = list(store.all())
    truncated = False
    for rule in rules:
        if deadline is not None and time.monotonic() > deadline:
            truncated = True
            break
        exact = miss = 0
        for t in transitions:
            p = rule.predict(t.level, t.pre, t.action_key)
            if p is None:
                continue  # NO_PREDICTION: not my transition
            ok = True
            if p.grid is not None and not np.array_equal(p.grid, t.post):
                ok = False
            if p.event is not None and p.event != t.event:
                ok = False
            if ok:
                exact += 1
            else:
                miss += 1
        rule.n_exact, rule.n_miss = exact, miss
        if miss > 0:
            rule.status = RuleStatus.CONTRADICTED
        elif exact >= min_exact:
            rule.status = RuleStatus.VERIFIED
        else:
            rule.status = RuleStatus.UNTESTED

    counts: dict[str, int] = {}
    for r in rules:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    return {"counts": counts, "transitions": len(transitions), "truncated": truncated}
