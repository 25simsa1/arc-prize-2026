# Overnight task 2 — R1′ A/B template verification on the five state-exploded games

You are working in /Users/simonsang/arc-prize-2026 (git repo, branch main). Use venv/bin/python.

## Precondition
Task 1 should have added `--r1prime {on,off}` to scripts/run_wm.py and an `r1prime` flag on `RegionAnalyzer`. Verify this (read the code, run `venv/bin/python scripts/run_wm.py --help`). If absent or broken, record the blocker in a dated NOTES.md entry (2026-06-10, overnight task 2), commit NOTES.md, and stop.

## Run both arms
Target games (the Part 2 correction set): **ft09 lp85 r11l sp80 tn36**. Same budget both arms. Read scripts/run_wm.py first and adapt flags to its actual interface; the intended shape is:

```
venv/bin/python scripts/run_wm.py --games ft09 lp85 r11l sp80 tn36 --proposer template --time-budget 240 --tag overnight-r1prime-off --r1prime off --save-stores
venv/bin/python scripts/run_wm.py --games ft09 lp85 r11l sp80 tn36 --proposer template --time-budget 240 --tag overnight-r1prime-on  --r1prime on  --save-stores
```

`--save-stores` matters: task 3 consumes the R1′-on stores. If the game environment is unreachable (API down, auth missing), log the blocker in NOTES.md, commit, and stop. Do not retry endlessly.

## Report
Write results/overnight-20260610/r1prime_coverage.md with a per-game table: unique states observed, template coverage, verified-rule counts, RHAE if available — each for R1′ off vs on, plus a delta column — and a one-line interpretation per game (unblocked / unchanged / regressed). A game counts as **unblocked** when R1′ produces a material coverage gain or collapses the state explosion (unique-state count drops sharply while coverage rises).

End the file with exactly one line so task 3 can parse it:
```
UNBLOCKED: <space-separated game ids, or 'none'>
```

## Finish
- NOTES.md entry (2026-06-10, overnight task 2) with the headline deltas.
- Commit locally (no push): `git add results NOTES.md`, message:
  `R1' A/B on the five state-exploded games: coverage deltas (ft09 lp85 r11l sp80 tn36)`

## GUARDRAILS (hard)
- No Kaggle actions of any kind (no kaggle CLI, no submissions).
- No spend: no paid APIs, no cloud LLM calls; the only LLM allowed is local ollama.
- No new dependencies: no pip/npm installs, no model pulls.
- Code edits only inside harness/ and scripts/ (this task should not need code edits beyond small flag fixes); artifacts only under results/; NOTES.md is editable. Run artifacts written by existing scripts to runs/ are acceptable side effects. Never touch git config, never push.
- If blocked or repeatedly failing, commit what exists, record it in NOTES.md, and exit rather than thrashing.
