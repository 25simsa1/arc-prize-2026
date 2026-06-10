# Kaggle sitting — your exact steps (one sitting, in order)

Three things to do at the Kaggle machine, ordered so the **free** checks and
the milestone work happen first and *tell us whether the paid probe is even
worth a submission slot*. Read the channel reality below before step 3.

## Channel reality (from Task A — read this first)

- **Interactive gateway access: almost certainly does NOT exist.** The
  official harness only contacts `gateway:8001` when
  `KAGGLE_IS_COMPETITION_RERUN=1`; the hidden test set isn't exposed to a
  live session. Step 1 confirms this in 30 seconds.
- **Owner-visible rerun logs: UNKNOWN, leaning NO** for a hidden-test
  gateway. This is the make-or-break channel and we test it **for free**
  using the M1 submission (step 2) before spending any slot on the probe.
- The gateway is a separate black box running possibly-newer code than our
  pip `arc_agi` (which has no cutoff), so Q1/Q2 genuinely need a live
  submission — there is no local shortcut.
- **What one probe submission buys depends entirely on the log channel:**
  - logs visible → the ProbeAgent answers Q1, Q3, Q5 outright and bounds Q2.
  - logs hidden → the ProbeAgent yields **almost nothing** (it scores ~0 on
    every game by design, so the leaderboard number carries no signal).
    In that case **do not submit it** — come back and we build the
    score-encoding probe (control + deliberate-overshoot, compared by
    aggregate score) for your approval. You should know that's the fork
    before you spend a slot.

## Step 1 — free interactive checks (no submission burned)

In an interactive competition notebook (Internet OFF, competition dataset
attached), run these cells and copy back the output:

```python
# 1a. Are we in a rerun? Is the gateway reachable interactively?
import os, subprocess
print("RERUN_ENV:", os.getenv("KAGGLE_IS_COMPETITION_RERUN"))   # expect None
print(subprocess.run(["bash","-lc",
    "curl -s -m 5 http://gateway:8001/api/games || echo NO_GATEWAY"],
    capture_output=True, text=True).stdout[:500])
```
- `NO_GATEWAY` (expected) → Channel 2 confirmed absent; proceed.
- A real games list (unexpected) → tell me immediately; we can probe free.

```python
# 1b. Offline wheel install proof + session limits (the vendored-wheel trap)
import subprocess, sys
print(subprocess.run(["bash","-lc",
  "pip install -q --no-index --find-links "
  "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels "
  "arc-agi python-dotenv && echo WHEELS_OK || echo WHEELS_FAIL"],
  capture_output=True, text=True).stdout[-300:])
```
Copy back: the gateway result, `RERUN_ENV`, `WHEELS_OK/FAIL`, and **the
wall-clock limit shown in the session banner** (expect 6h — flag if not).

## Step 2 — M1 notebook: submit + publish + milestone (and a free log test)

Follow **`kaggle_m1/STEPS.md`** verbatim (pre-flight already done: audit
clean at 37 patterns, dry-run OK, single-play only). It covers notebook
creation, the status-region veto point, run+observe, submit, make public,
milestone registration.

**Then — the free Channel-1 test:** after the M1 submission reruns, open the
submission and check whether you can view its **execution logs / notebook
output**. M1 prints ordinary logs, so:
- M1 rerun logs **viewable** → Channel 1 is OPEN → the ProbeAgent (step 3)
  is worth a slot.
- M1 rerun logs **not viewable / suppressed** → Channel 1 is CLOSED → **skip
  step 3**, tell me, and we build the score-encoding probe for approval.

Copy back everything from the M1 "run + observe" checklist in its STEPS.md
(how games are served, any per-game/total limits visible, the submission
artifact, the wall-clock limit, anything unlike our local runtime).

## Step 3 — ProbeAgent submission (ONLY if M1 logs were viewable)

**Check the daily submission limit first** (competition page → My
Submissions). M1 already used one today; if the daily cap is small, the
probe may need the next day — don't blow your M1 slot on it.

Assemble a **private** notebook (keep it private — it's a diagnostic, not a
scoring entry; it will score ~0):

- Cell 1 — install wheels (same line as 1b above).
- Cell 2 — `%%writefile /kaggle/working/my_agent.py` then paste the entire
  contents of **`kaggle_probe/probe_agent.py`**.
- Cell 3 — the rerun harness: copy the rerun block from the FORGE notebook
  (`research/competitors/ash-forge/`, the `if os.getenv('KAGGLE_IS_COMPETITION_RERUN')`
  cell) verbatim — it stages the agents repo, writes `agents/__init__.py`
  with `MyAgent`, writes `.env`, and runs `python main.py --agent myagent`.
- Cell 4 — the dummy-submission fallback for the non-rerun branch (also from
  FORGE, the `if ... != "1"` cell that writes a minimal `submission.parquet`).

Submit to competition. The probe prints `PROBE_SENTINEL_7Qf3xR2 ...` lines
and a per-game SUMMARY JSON, and `main.py` logs the closed scorecard.

## Step 4 — what to bring back (drop into `results/gateway_probe/`, then say the word)

- `log.txt` — the full rerun log text, if viewable (this is the prize).
- `scorecard.json` — the closed scorecard JSON if exposed.
- `score.txt` — the leaderboard aggregate (and `control_score.txt` if we
  ever run the score-encoding control).
- any error output **verbatim**.

Then I run `scripts/analyze_gateway_probe.py` — it parses whatever landed,
matches it against the hypothesis tables, and emits the five verdicts
(CONFIRMED / REFUTED / INCONCLUSIVE, each with its evidence) to
`results/gateway_probe/verdicts.json`. (Self-tested on a synthetic fixture.)

## End of sitting

Stop here. After the verdicts land I update `results/cap_study/REPORT.md`
and `NOTES.md`, and flag which vertical-slice parts change under the
confirmed semantics (especially: does win-gated two-phase survive Q2, and
does the budget math shift on the confirmed Q4=6h).

## Guardrails honored
- I do not submit or publish anything — every Kaggle action above is yours.
- M1 was dry-run locally before this handoff (mandatory mirror check done).
- Channel reality stated plainly: interactive access ~absent, log visibility
  unknown-lean-no; the score-encoding-only fallback is **not** finalized —
  it needs your go-ahead once step 2 tells us logs are hidden, because it
  buys less per slot and you should choose that knowingly.
