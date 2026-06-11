# Overnight task 4 — Audit harness/wm/ for correctness and test gaps (time-permitting)

You are working in /Users/simonsang/arc-prize-2026 (git repo, branch main). Use venv/bin/python.

This is the lowest-priority task; the runner enforces a hard time cap, so work in small commit-able increments.

## What to do
If the `cloud-coder:audit-loop` skill is available in your session, invoke it scoped to **harness/wm/** with the goal "correctness bugs and test gaps", capped at 2 audit-fix cycles. If it is not available, do the loop manually:

1. **Audit:** read every module in harness/wm/ (store.py, regions.py, verifier.py, proposers.py, llm_proposer.py, and the rest). Hunt specifically for: off-by-one and boundary errors, mutation/aliasing of shared grids, hash-key collisions in the (level, pre_hash, action_key) store, deadline/signal races (a SIGALRM race in repair's `count()` was recently fixed — look for siblings), silent exception swallowing, and untested branches in the verifier's VERIFIED/CONTRADICTED/UNTESTED logic.
2. **Fix:** for each suspected bug, write a failing test FIRST in the matching scripts/test_*.py file, then fix it. Behavior-preserving refactors only where a real bug is demonstrated; do not restyle working code.
3. **Verify:** re-run `venv/bin/python scripts/test_wm_core.py`, `scripts/test_explore.py`, `scripts/test_body_move.py` (and other offline scripts/test_*.py) after each fix.

Commit locally (no push) after each audit cycle with a specific message describing the bug(s) fixed and tests added, and append a dated NOTES.md entry (2026-06-10, overnight task 4) summarizing findings — including suspected issues you did NOT fix (with file:line) so they aren't lost.

## GUARDRAILS (hard)
- No Kaggle actions of any kind (no kaggle CLI, no submissions).
- No spend: no paid APIs, no cloud LLM calls; the only LLM allowed is local ollama.
- No new dependencies: no pip/npm installs, no model pulls.
- Code edits only inside harness/ and scripts/; artifacts only under results/; NOTES.md is editable. Never touch git config, never push.
- Do not change public interfaces that scripts/run_wm.py or scripts/llm_quality.py depend on.
- If blocked or repeatedly failing, commit what exists, record it in NOTES.md, and exit rather than thrashing.
