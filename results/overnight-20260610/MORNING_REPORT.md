# Overnight 2026-06-10 — Morning report

Queue: `scripts/overnight/overnight-queue-20260610.yaml`, start SHA
`53d216d`. **The queue ran twice.** The first pass (23:58) hit a Claude
session/usage limit: all five tasks exited rc=1 within seconds, producing
only the five empty fallback commits `0f18773..da66f8e`. The runner was then
patched (`0a961ab`) to treat usage-limit errors as wait-and-retry and to
reset the poisoned `completed.txt`. The real pass ran **2026-06-11
07:50–08:42**, all five tasks rc=0, no SIGALRM/time-cap kills.

## 1. What changed

**Task 1 — build R1′ (`4acab94`, fallback `4fcd4fe`).** Built the R1′
exogenous-HUD masking stage in `harness/wm/regions.py`: cells whose
post-value is a near-deterministic function of actions-since-level-start
(majority-value share ≥0.9 over repeated index bins, singleton bins
excluded, evidence floors 8 changes / 6 repeated obs) are masked even though
they never change solo. Includes a click-board guard (changes concentrating
within chebyshev ≤2 of clicks in ≥20% of click transitions are exempt),
index plumbing via `observe(t, idx=...)`, and a full on/off switch down to
`scripts/run_wm.py --r1prime {on,off}` (off is bit-identical to the old
detector). Tests all green: test_wm_core 5/5 including two new scenarios
(rendered-counter trap, click-guard), test_explore 15/15, test_body_move,
test_wm_tt01.

**Task 2 — R1′ A/B on the five state-exploded games (`ea7091f`, fallback
`eaaf263`).** Both arms ran clean offline (template proposer, 240s/game,
seed 0, tags `overnight-r1prime-off/on`). Headline: **sp80 unblocked**
(distinct contexts 4168→473, −89%; steady event coverage 0→0.096 at exact
rate 1.0; first VERIFIED rule). **r11l improved, not unblocked** (contexts
−31%, HUD coverage 0→0.064, verified rules 0→1, matched prediction steps
17→104, but grid/event template coverage still 0.000 — consistent with R1′
being necessary-not-sufficient; R2/R3 pending). **ft09, lp85, tn36
unchanged** — ft09/lp85 were never context-exploded (590/242 contexts);
tn36 flat within noise. RHAE identical per game across arms (mean 0.634):
at this budget R1′ moves model quality, not score. Full table in §3 and
`r1prime_coverage.md`; R1′-on stores at `runs/wm/overnight-r1prime-on/`
(~690MB, gitignored).

**Task 3 — llm_quality re-run on sp80: NO RESULTS (fallback `5d5c41e`
only).** Preconditions passed (`UNBLOCKED: sp80` parsed; ollama up with
qwen2.5-coder:14b) and the task launched
`llm_quality.py --stores runs/wm/overnight-r1prime-on --games sp80 --rounds 4`
— but it launched it **in the background** and ended its session saying
"I'll pick this up when the background task finishes." The runner moved on
at 08:32:57 and the process died with it.
`results/overnight-20260610/llm-quality-r1prime/` contains only an empty
`run.log`; there is no quality table and no NOTES entry. The task exited
rc=0, so this reads as success in `completed.txt` — it was not.

**Task 4 — harness/wm/ correctness audit, 2 cycles (`f4cee1a`, `af41132`,
fallback `7cd030b`).** Four real bugs fixed, each with a failing test first:
(1) verifier counted a claim-less `Prediction()` as an exact match — a
vacuous rule reached VERIFIED on any store with ≥3 transitions (192 phantom
exacts demonstrated on the toy store); now NO_PREDICTION. (2) verifier
deadline was only checked between rules, so one slow rule overshot the
budget unboundedly; now checked every 32 transitions, aborted rules keep
their previous status. (3) `TransitionStore.load()` dropped all eviction
state — a loaded store at cap refused every new transition; save/load now
round-trip it with graceful legacy-pickle fallback. (4) Cycle 2: the
identical vacuous-claim hole in `proposers._fits()`. Five further suspects
deliberately left unfixed and recorded with file:line (see §2). All offline
suites pass. Note: the `cloud-coder:audit-loop` skill's backing script
doesn't exist in this repo and its design spawns cloud CLI calls (no-spend
guardrail), so the loop ran manually as the task spec allows.

## 2. What's blocked

- **Task 3 deliverable missing (the only outright failure).** The LLM
  quality run on the R1′-on sp80 store produced nothing because it was
  backgrounded and orphaned. *Unblock:* re-run in the foreground —
  `.venv/bin/python scripts/llm_quality.py --stores runs/wm/overnight-r1prime-on --games sp80 --rounds 4`
  with output to `results/overnight-20260610/llm-quality-r1prime/` (stores
  and ollama model are already in place; ~minutes-to-an-hour of local GPU,
  no spend).
- **r11l: improved, not unblocked.** R1′ dampened the explosion (−31%
  contexts) but template coverage is still 0.000. *Unblock:* build R2
  (selection-parameterized rigid-body) and R3 (latent state) from the Part 1
  diagnosis — R1′ alone was predicted to be necessary, not sufficient.
- **ft09, lp85, tn36: blockers are not HUD exogeneity.** Contexts were
  never exploded (ft09/lp85) or flat (tn36); R1′ is a no-op. *Unblock:*
  fresh per-game diagnosis to name their actual blockers before spending
  more A/B budget on them.
- **Audit suspects needing supervised decisions** (file:line in the task 4
  NOTES entry): `explore.py:199-214` Archive.consider drops
  near-event/meter flags on shorter-prefix replacement (fix changes tuned
  exploration ranking); `proposers.py:627-649` DiffMemorizer masked branch
  reads a frozen index while the unmasked branch reads the live store (a
  behavioral asymmetry between ablation arms); fixed 64×64 shape
  assumptions at `store.py:168`/`regions.py:168` and
  `winseeker.py:57`; dup check at `store.py:150-152` ignores `post_level`.
  *Unblock:* review each with the tuning context in hand and decide
  fix-vs-keep.
- **`venv/` is a bare environment (pip only — no numpy).** Every task
  prompt said `venv/bin/python`; all four tasks independently fell back to
  `.venv/bin/python` (the project env per the scripts' docstrings).
  *Unblock:* either delete/rebuild `venv/` or fix the overnight prompt
  templates to say `.venv/bin/python`.
- **`cloud-coder:audit-loop` skill is broken in this repo.** Its backing
  script `devops/cloud-coder/audit-fix-loop.sh` doesn't exist, and its
  design spawns paid cloud CLI calls. *Unblock:* add the script or stop
  referencing the skill in audit task prompts.
- **Runner robustness (fixed mid-night, worth keeping).** The usage-limit
  pass marked all five tasks complete with empty work; `0a961ab` now
  wait-and-retries (20m backoff, ~8h max). Residual gap exposed by task 3:
  the runner trusts rc=0 even when the task's deliverable doesn't exist.
  *Unblock:* add a per-task artifact check (expected output path in
  tasks.tsv) before marking complete.

## 3. R1′ coverage table (verbatim from r1prime_coverage.md)

| game | uniq states off | uniq states on | Δ states | ev-cov off | ev-cov on | hud-cov off | hud-cov on | verified off | verified on | RHAE off | RHAE on | matched steps off→on | verdict |
|------|----------------:|---------------:|---------:|-----------:|----------:|------------:|-----------:|-------------:|------------:|---------:|--------:|---------------------:|---------|
| ft09 | 590  | 587  | −0.5% | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.00 | 0.00 | 694→694   | unchanged |
| lp85 | 242  | 238  | −1.7% | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.00 | 0.00 | 4232→4232 | unchanged |
| r11l | 4577 | 3140 | −31%  | 0.000 | 0.000 | 0.000 | 0.064 | 0 | 1 | 2.94 | 2.94 | 17→104    | improved, not unblocked |
| sp80 | 4168 | 473  | −89%  | 0.000 | 0.096 | 0.000 | 0.000 | 0 | 1 | 0.03 | 0.03 | 700→407*  | **unblocked** |
| tn36 | 1748 | 1797 | +2.8% | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.19 | 0.19 | 50→50     | unchanged |

\* sp80 total actions also fell 10360→3851; the matched *rate* rose
(699/704 → 407/408 on predicted steps) while exploration found the same
frontier with far fewer steps.
