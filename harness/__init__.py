"""Local development harness for ARC Prize 2026 (ARC-AGI-3 track).

Drives the official `arc_agi` offline runtime in-process and reports exact
RHAE using the shipped scorecard implementation. Nothing here may depend on
network access at runtime — the evaluated submission path is offline.
"""

from .runner import RunConfig, run_suite

__all__ = ["RunConfig", "run_suite"]
