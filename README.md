# ARC Prize 2026 — ARC-AGI-3 local harness

Local development scaffolding for the ARC-AGI-3 Kaggle track. Drives the
official `arc_agi` offline runtime in-process, scores with the shipped
scorecard, and double-checks every reported score with an independent RHAE
recomputation. No network access in any runtime path; no API LLMs anywhere
(submission constraint).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install \
  ~/.cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/arc_agi-0.9.8-py3-none-any.whl \
  ~/.cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/arcengine-0.9.3-py3-none-any.whl
```

The 25 public environments are read from the kagglehub cache
(`environment_files/`); override with `RunConfig.environments_dir`.

## Commands

```bash
.venv/bin/python scripts/smoke_env.py        # runtime + scorecard semantics check
.venv/bin/python scripts/verify_rhae.py      # our RHAE math vs shipped calculator
.venv/bin/python scripts/run_baseline.py --games all --budget 300   # random baseline
.venv/bin/python scripts/smoke_http.py       # official HTTP route shape, offline
```

Run records land in `runs/*.json` (scorecard + independent verification).
Research log: `NOTES.md`. Competition framing and prior art: `phase1-v2.md`.

## Layout

- `harness/runner.py` — suite runner; budgets count *scored* actions
  (ids 1–7 + non-full RESETs), mirroring the scorecard.
- `harness/rhae.py` — independent RHAE recomputation (cap 115/level,
  level-index weights, completed-weight ceiling).
- `harness/agents/` — agent contract mirroring the official repo
  (`choose_action(frames, latest_frame)` / `is_done`), plus the random
  baseline. Agents must respect `available_actions` — off-menu actions cost
  budget and do nothing.
