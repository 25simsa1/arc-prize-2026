# R1′ A/B — template proposer on the five state-exploded games

Overnight task 2, 2026-06-10 queue (executed 2026-06-11 morning).
Arms: `--r1prime off` vs `--r1prime on`, otherwise identical
(`--proposer template --time-budget 240 --seed 0 --save-stores`, two_phase,
action budget 50000). Tags: `overnight-r1prime-off`, `overnight-r1prime-on`
(results/<tag>/, stores under runs/wm/<tag>/).

Metric notes:
- **Unique states** = `explorer.distinct_contexts`, the agent's
  (level, masked-frame-hash) context count — the quantity R1′ is meant to
  collapse. Raw frame uniqueness is unchanged by construction.
- **Coverage** = steady-state predicted fraction over the final quarter of
  coverage events (peak values saturate transiently and are not meaningful).
  Grid coverage (`gp`) was 0.0000 steady-state for every game in both arms;
  the table shows event coverage (`ep`) and HUD coverage (`hp`), where the
  movement actually happened.
- **Verified rules** = end-of-run model `by_status.VERIFIED`.
- **RHAE** = per-game `best_rhae` from summary.json.

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

Per-game interpretation:
- **ft09 — unchanged.** Context count was never exploded here (590); R1′
  found nothing to mask and behavior is near-bit-identical (matched steps
  identical). Whatever blocks ft09 is not HUD exogeneity.
- **lp85 — unchanged.** Same story at 242 contexts; coverage stays 0 in both
  arms. Not a state-explosion problem.
- **r11l — improved, not unblocked.** R1′ detected the exogenous HUD
  (steady HUD coverage 0→0.064, peak 0.90), cut distinct contexts 31%
  (4577→3140), produced the game's first VERIFIED rule, and raised matched
  prediction steps 17→104. But unique contexts remain in the thousands and
  grid/event template coverage is still 0.000 — the explosion is dampened,
  not collapsed. Does not meet the unblocked bar.
- **sp80 — unblocked.** Distinct contexts collapse 4168→473 (−89%) while
  event coverage rises 0→0.096 with exact rate 1.0 and a VERIFIED rule
  appears (off-arm had only contradicted rules). This is exactly the
  state-explosion-collapse-plus-coverage-rise signature.
- **tn36 — unchanged.** Contexts flat (1748→1797, within noise), all
  coverage 0, no verified rules either arm. tn36's uniqueness is not
  HUD-driven; R1′ is a no-op here.

Headline: mean RHAE over games is identical between arms (0.634 — driven by
r11l's 2.94 in both); R1′'s effect at this budget shows up in model quality
(contexts, verified rules, prediction matching), not yet in RHAE.

UNBLOCKED: sp80
