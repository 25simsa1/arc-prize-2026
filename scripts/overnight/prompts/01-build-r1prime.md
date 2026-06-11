# Overnight task 1 — Build R1′: exogenous-aware HUD masking

You are working in /Users/simonsang/arc-prize-2026 (git repo, branch main). Use venv/bin/python for everything.

## Context — read these first
- harness/wm/regions.py — current HUD detection. `RegionAnalyzer` seeds on cells that are repeatedly the SOLE change and action-exogenous (TIER-1), expands to co-changers in exactly-2-cell diffs (TIER-2), then classifies clusters as ALWAYS_CHANGING with `always_rate=0.55`, `max_frac=0.10`, an unmaskable-colors guard (cells that held LEVEL/WIN/GAME_OVER event colors), and `min_transitions=30`. Output is `RegionMap.hud_mask`.
- results/diagnosis_r11l_sp80.md — the failure mode this task fixes. r11l renders a decrementing step counter (60→0) that changes on EVERY action but NEVER solos: clicks also move/select pieces, so the sole-changer seed never fires. Counter+content co-change makes nearly every frame unique → state explosion → 0 template coverage. cd82-style tickers (which do solo) are caught today; r11l-style rendered counters are not.

## What to build
1. Add a **change-content-predictability detector** to `RegionAnalyzer` in harness/wm/regions.py: a region is HUD-like if the CONTENT of its changes is a low-entropy function of action count — i.e., per-cell (or per-cluster) value sequences are a deterministic/near-deterministic function of the number of actions taken since level start — even when it never solo-changes. Use a conditional-entropy or exact-functional test of value-given-action-index; require the same `min_transitions` confidence floor before classifying.
2. **Board guard (critical — from the r11l diagnosis):** in click games (ACTION6), never let the new detector mask cells whose changes are click-dependent — change content or timing varies with click coordinates (changes concentrated near click points, or value depends on which cell was clicked). Interactive board cells must survive even if their value sequence happens to look low-entropy in a short window. Keep the existing unmaskable-colors guard and `max_frac` size guard in force for the new detector too.
3. **R1′ must be independently switchable:** constructor flag (e.g. `RegionAnalyzer(..., r1prime=True)`) defaulting ON, plumbed through to a CLI flag `--r1prime {on,off}` on scripts/run_wm.py (read run_wm.py to wire it idiomatically). With `--r1prime off`, behavior must be bit-identical to today — a later task A/Bs this.

## Tests
Extend scripts/test_wm_core.py (run: `venv/bin/python scripts/test_wm_core.py`):
- **Rendered-counter trap:** synthetic scenario with a counter region that decrements every action while board content also changes (so it never solos). Assert R1′-on masks the counter and grid templates reach VERIFIED; assert R1′-off does NOT mask it (regression baseline preserved).
- **Click-board guard:** a click game where board cells respond to clicks (and could look value-predictable over a short window); assert the interactive board is NOT masked with R1′ on.
- All existing scenarios must still pass. Also run scripts/test_explore.py and scripts/test_body_move.py (and scripts/test_wm_tt01.py if it runs offline); report pass/fail honestly.

## Finish
- Append a dated NOTES.md entry (2026-06-10, overnight task 1): the detector design, the click-board guard, flag plumbing, test results.
- Commit locally (do NOT push): `git add harness scripts NOTES.md` then commit with message:
  `R1': exogenous-aware HUD masking via change-content predictability; click-board guard; rendered-counter trap tests`
- If you cannot finish, commit what exists with a WIP message, record the blocker in NOTES.md, and stop.

## GUARDRAILS (hard)
- No Kaggle actions of any kind (no kaggle CLI, no submissions).
- No spend: no paid APIs, no cloud LLM calls; the only LLM allowed is local ollama.
- No new dependencies: no pip/npm installs, no model pulls.
- Code edits only inside harness/ and scripts/; artifacts only under results/; NOTES.md is editable. Run artifacts written by existing scripts to runs/ are acceptable side effects. Never touch git config, never push.
- If blocked or repeatedly failing, commit what exists, record it in NOTES.md, and exit rather than thrashing.
