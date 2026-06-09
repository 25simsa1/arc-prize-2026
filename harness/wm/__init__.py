"""Executable world model: propose -> verify -> plan -> execute.

Per-game only. Cross-game transfer is a later, separate component — nothing
in this package may key knowledge off game_id beyond namespacing artifacts.
"""

from .store import Transition, TransitionStore, canon_frame, frame_hash
from .rules import Prediction, Rule, RuleStatus, WorldModel
from .verifier import verify_rules

__all__ = [
    "Transition",
    "TransitionStore",
    "canon_frame",
    "frame_hash",
    "Prediction",
    "Rule",
    "RuleStatus",
    "WorldModel",
    "verify_rules",
]
