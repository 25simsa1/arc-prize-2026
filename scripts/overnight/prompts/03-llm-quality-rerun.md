# Overnight task 3 — 14B llm_quality re-run on R1′-unblocked games

You are working in /Users/simonsang/arc-prize-2026 (git repo, branch main). Use venv/bin/python.

## Preconditions
1. Read results/overnight-20260610/r1prime_coverage.md and take the trailing `UNBLOCKED:` line. If the file is missing or the list is `none`, write a dated NOTES.md entry (2026-06-10, overnight task 3) saying the task was skipped and why, commit NOTES.md, and stop.
2. Check ollama: `curl -s http://localhost:11434/api/tags`. If it is down or `qwen2.5-coder:14b` is not in the model list, do NOT install or pull anything — log blocked in NOTES.md, commit, stop.

## Run
Use the R1′-on stores captured by task 2 (`--save-stores` under tag `overnight-r1prime-on`; read scripts/llm_quality.py for its `--stores` convention and find the store directory, likely under runs/wm/overnight-r1prime-on/). Then:

```
venv/bin/python scripts/llm_quality.py --stores <r1prime-on store dir> --games <unblocked games> --rounds 4 --llm-model qwen2.5-coder:14b --llm-url http://localhost:11434 --llm-backend ollama
```

Adapt flags to the script's actual interface. Save/copy the output report under results/overnight-20260610/llm-quality-r1prime/.

## Report
Per game: format-error rate, verified-rule count, beyond-template coverage, and tokens/seconds per verified rule if emitted. Compare explicitly against the Part 2 baseline finding (14B produced 0 verified rules on su15/sb26/ar25 — fabricated mechanics, not a harness wall): does R1′ masking change the 14B's hit rate on these games, or does the fabrication failure mode persist?

## Finish
- NOTES.md entry (2026-06-10, overnight task 3) with the headline.
- Commit locally (no push): `git add results NOTES.md`, message:
  `14B llm_quality re-run on R1'-unblocked games: <one-line headline result>`

## GUARDRAILS (hard)
- No Kaggle actions of any kind (no kaggle CLI, no submissions).
- No spend: no paid APIs, no cloud LLM calls; the only LLM allowed is local ollama at localhost:11434.
- No new dependencies: no pip/npm installs, no `ollama pull`.
- Code edits only inside harness/ and scripts/ (avoid code edits in this task unless a trivial flag fix is needed); artifacts only under results/; NOTES.md is editable. Run artifacts written by existing scripts to runs/ are acceptable side effects. Never touch git config, never push.
- If blocked or repeatedly failing, commit what exists, record it in NOTES.md, and exit rather than thrashing.
