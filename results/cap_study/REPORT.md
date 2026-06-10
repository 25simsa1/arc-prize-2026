# Cap study — morning report (overnight run, 2026-06-10, ~01:23–02:50 EDT)

## THE CAVEAT, FIRST

Everything below models a cutoff whose **gateway enforcement is
UNCONFIRMED**. Source: the ARC-AGI-3 technical report (2603.24621,
Leaderboards) — "hard cutoff of 5x human performance per level" — which is
absent from the shipped scoring code; whether the Kaggle gateway enforces
it, and with what mechanics, is exactly what the daytime Kaggle-Starter
probe must answer. This report prepares both branches: every census now
exists in capped AND uncapped form. The cap implementation's five
provisional assumptions are documented in
`harness/runner.py::level_action_cap` for the probe to confirm/refute,
the two most consequential being: a cutoff ends the game run, and replay
plays get fresh per-level counters.

Second caveat: the capped runs put tonight's agent — designed for an
uncapped regime — under the cap unchanged. They are a **lower bound** on a
cap-aware agent. Third: sweep25 (the original census) was an older agent;
tonight's capped and uncapped arms both use the current agent (post
explorer/deadline fixes), so the arms are internally comparable; sweep25 is
shown for continuity.

## 1. Capped vs uncapped census table

3 capped seeds (0,1,2) and 3 uncapped runs (sweep25 = seed 0/older agent;
uncap-s1, uncap-s2 = current agent). Mean [min–max]:

| metric | capped (5×) | uncapped | single-seed original (sweep25) |
|---|---|---|---|
| mean RHAE % | 0.219 [0.219–0.219] | 0.230 [0.219–0.253] | 0.253 |
| evidence-starved games | **88%** [88–88] (22/25) | 73% [72–76] | 76% |
| non-Markov (conflict sig) | **0%** [0–0] (artifact, §3) | 51% [48–52] | 48% |
| games won | 0% | 0% | 0% |
| levels completed (sum) | 3 | 8 | 8 |
| actions spent (sum) | ~4,836 | ~1.24M | 1.32M |
| sweep wall-clock | **~4 min** | ~45–140 min | ~2.3 h |

Retroactive audit cross-check (truncating the OLD sweep25 trajectories at
the hypothetical cap): starvation 92%, levels surviving 2/8 — consistent
with the live capped runs (88%, 3 levels; the current explorer finds
lp85's level where the old one didn't).

Seed variance: near-zero. The three capped seeds differ by ONE action
total (the agent's rng only reaches a rarely-hit "desperate" branch);
uncap-s1 and uncap-s2 have identical censuses. Agent VERSION variance
(sweep25 vs tonight: starvation 76→72%, non-Markov 48→52%) exceeds seed
variance. Per-2601.00042 multi-seed reporting is now in place, but for
this agent the honest statement is "deterministic up to wall-clock
jitter"; version-to-version drift is the real reproducibility risk.

## 2. Two-phase survivability verdict

**Structurally survives, practically idle.** The cap cannot poison the
architecture: a capped replay ends its play and banked plays keep the
max-over-plays score (unit-tested, T5). But on real games the machinery
never engages — zero wins in any arm — and the cap moves the win-gate
much closer: the FIRST win must now itself be frugal, because every
level of the winning play must complete within 5× its baseline.
"Win sloppily, then replay cleanly" survives only in the narrow sense
that *post-win* replays remain free; the sloppy win's allowed sloppiness
shrinks from "whatever wall-clock permits" (26k–110k actions observed)
to **30–2,890 actions per level**. The honest restatement of our
architecture under the cap: **win frugally, then replay cleanly** — and
"win frugally" is currently unsolved (margin data: the only wins on
record, tt01, fit at 0.13–0.20 of cap; no real game has any win).

## 3. The non-Markov census is eval-blind (new finding, paper-relevant)

Under the cap, conflict signatures collapse 51% → 0%: in ≤5×-baseline
actions per level, the store never observes the same (level, frame,
action) key twice with different outcomes. The latent state has not
gone anywhere — the cap makes it **undetectable before the cutoff**.
Implications: (a) R3 conflict-triggered machinery (the AutumnSynth-style
"invent a hidden variable when conflicts appear" route from the lit
sweep) cannot trigger inside the eval envelope as designed — it needs
either within-window detection (cross-play comparison: replays re-visit
the same early states, and conflicts between PLAYS are observable
cheaply) or stateful-by-default priors; (b) the paper's hidden-110
predictor ("~half the eval needs stateful models") must be framed as a
property measured OFF-CAP, with the explicit note that an eval-time
agent cannot measure it.

## 4. Budget math under the cap

- Eval-realistic per-game evidence: **34–465 actions** before cutoff
  (median ~150), vs 1.5k–198k uncapped. The evidence sets the LLM
  proposer will see are tiny: serialization cost stops being the
  constraint; evidence SELECTION stops mattering (you ship everything).
- Mechanism payoff inside the window (capped trajectories): unseen-first
  enumeration does essentially all the work (>95% of actions on 23/25
  games); lattice/frontier/safe_any barely engage. Level advances, when
  they come, come EARLY: actions 8 (r11l), 59 (lf52), 65 (lp85) — ≤40%
  of the level-0 cap. GAME_OVER evidence arrives within cap on 17/25
  games. Pattern: **the capped window is enough for cheap evidence iff
  the game yields evidence cheaply at all** — grinding never fits.
- Wall-clock inverts: a capped 25-game sweep is ~4 min. Under cap-style
  eval, ~all of the 6–9h notebook budget is available for LLM calls —
  the proposer's affordability constraint relaxes by ~an order of
  magnitude, while its evidence constraint tightens by ~two orders.
  Part-2 planning should flip from "minimize calls per rule" toward
  "maximize induction quality from ~150-action evidence sets, spend
  freely on internal compute" (replay/simulation against the store stays
  free).
- Part-0 priority changes: (1) frugal-exploration mechanisms (FORGE-style
  effect tallies, tier escalation) are now first-order — they act inside
  the window; (2) the AERA persistence probe is demoted (refuted, §6, and
  it doesn't fit: 200 reps × 5 actions exceeds every level-0 cap; even
  one action's 200 reps fits only 7/25 games); (3) cross-play conflict
  detection (cheap, fits the win-gate) is the new R3 evidence route.

## 5. FLAGS — numbers in NOTES.md this study invalidates or re-scopes

1. **"76% evidence starvation"** → 72–76% is the *uncapped, current-agent*
   range; the **eval-realistic number is 88%** (live capped) / 92%
   (retro on old trajectories). Use 88% for eval-context claims.
2. **"48% non-Markov census as hidden-110 predictor"** → 48–52% uncapped
   and REAL, but **unobservable under the cap** (0% detected). Keep the
   census, add the eval-blindness caveat (§3).
3. **"0.253% mean RHAE, zero LLM calls"** → current agent measures
   0.219% (both arms); 0.253 was the older agent. The "inside the
   frontier band" framing survives; the specific number doesn't.
   Notably: **the cap costs ~nothing in RHAE** (0.219 vs 0.219–0.253) —
   quadratic scoring already gave grinding zero credit.
4. **110-games-in-9h projection** (7.3h harness + ~3.5h model) → under
   the cap the harness share collapses (~minutes); the projection's
   binding constraint becomes model time + per-game evidence scarcity,
   not stepping or wall-clock.
5. **Triage buckets**: WALLED-R3-trans (12 games) is an off-cap
   diagnosis; under eval conditions those games are indistinguishable
   from DEAD before cutoff. Bucket definitions need a "detectable
   in-window?" column before the paper uses them.
6. **AERA-derived routing** (sweep doc): persistence probe demoted;
   AERA's 21%/30% claims and validity critique treated as unverified
   (Table 9 refuted on the real environments — §6).

## 6. Bonus verifications (optional phase, completed)

- **V1 (AERA units/protocol)**: RHAE figures in 2605.25931 are 0–1
  FRACTIONS (0.2116 = 21.16%, "4/25 solved"; code-track 0.30 = 30%
  private) — but direct probe (scripts/probe_repeat_action.py) shows all
  8 of their Table-9 "solvable by one repeated action" games hit
  GAME_OVER at exactly the step count they report as sufficient
  (tu93@50, sc25@52, tr87@128, ka59@100, re86@100, ls20@129, g50t@130,
  wa30@200; zero levels anywhere). Their evaluation evidently counted
  episode end as success. Threat downgraded; correction filed in the
  sweep doc §8.
- **V2 (cross-game legality)**: already verified 2026-06-10 daytime
  (sweep doc §7): no official rule forbids cross-game learning at eval;
  Competition Mode's forced behaviors don't mention it; the eval swarm
  runs all games in one process (state sharing structurally available);
  only soft control = ARC Prize overfitting discretion. Kaggle rules
  page remains the one unchecked source (login-gated).

## 7. What the gateway probe must now answer (priority order)

1. Is the 5× per-level cutoff enforced on Kaggle at all?
2. If yes: what does a cutoff DO — end the level (forced advance-denial),
   end the game, end the run? Is the counter per-play or cumulative
   across plays? (Decides whether win-gated replay survives at all.)
3. Does RESET-after-WIN still mint a fresh play (the "Game Resets become
   Level Resets" doc sentence vs our arc_agi-0.9.8 evidence)?
4. The actual wall-clock limit: 6h (mirror claim) / 8h (FORGE's guard) /
   9h (our assumption)?
5. The null-coordinate behavior (AERA's exploit claim — unverified but
   cheap to check there).

## Artifacts

- `harness/runner.py::level_action_cap` + `RunConfig.per_level_action_cap_multiplier`
  (+ `scripts/test_level_cap.py`, 21/21)
- `results/cap_study/retro_audit.json` (+ `scripts/audit_cap_retro.py`)
- `results/cap5-s{0,1,2}/`, `results/uncap-s{1,2}/` (committed run dirs)
- `results/cap_study/census_table.json` (+ `scripts/cap_census_table.py`)
- `results/cap_study/capped_mechanism_breakdown.json`
- `results/cap_study/repeat_action_probe.json` (+ `scripts/probe_repeat_action.py`)
- `results/cap_study/PHASE_LEDGER.md` (timestamped phase log)
