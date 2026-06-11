# Research log — ARC Prize 2026, ARC-AGI-3 track

Working log of decisions, surprises, and dead ends while building the local
development harness and (later) the agent. Honest record; feeds the
competition writeup. Newest entries at the bottom of each dated section.

## 2026-06-09 — Harness build, day 1

### Ground rules the harness is built around

- Submissions run in offline Kaggle notebooks: no internet at evaluation, ≤9h,
  open-weight models only — nothing API-based in the evaluated path. Local
  harness may use internet at *install* time, but every runtime component must
  work offline.
- Scoring is RHAE. Verified against the shipped scoring code
  (`arc_agi-0.9.8`, `scorecard.py:170-171`): per level,
  `score = min((baseline_actions / agent_actions)^2 * 100, 115.0)` — a flat
  115-point per-level ceiling (≈ reached at ≤93% of baseline actions; the
  inline comment says "max 100" but it's stale, the code governs). Game score
  = level-index-weighted average (1-indexed level numbers as weights), capped
  at (completed-weight fraction × 100), so 100% needs the final level. Final
  score = mean over games. Only environment interactions count as actions;
  internal compute is free.
- Design consequences baked into the harness: (a) over-performing easy levels
  (up to 115%) subsidizes under-performing hard levels within a game's
  weighted average; (b) level-index weights mean actions saved on later
  levels are worth more than on early ones.
- Starter repo's `MAX_ACTIONS=80` is a repo default, not a competition rule —
  our runner takes an explicit per-run budget.
- Starter repo's LLM agent templates call OpenAI APIs. Deliberately not
  copying that pattern anywhere in the harness: dev-time scaffolding in the
  official repo, but a trap for submissions.

### Environment facts discovered

- The bundled wheel set targets **Linux / CPython 3.12** (all compiled deps
  are `cp312 manylinux`: numpy, pillow, pydantic_core, matplotlib...). So the
  Kaggle eval image is py3.12. The two competition packages (`arc_agi-0.9.8`,
  `arcengine-0.9.3`) are pure Python (`py3-none-any`).
- Local dev machine is macOS arm64 / Python 3.13.3 → plan: install the two
  local wheels into a venv and let pip resolve compiled deps from PyPI for
  macOS. Version skew risk (numpy 2.4.4 etc.) noted; pin closely to the
  bundled versions where it matters.
- 25 public environments at
  `~/.cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files/`,
  each `{game_id}/{version_hash}/` with `metadata.json` (incl. per-level human
  `baseline_actions`) + plain readable Python game source (ARCBaseGame
  subclass). Exact RHAE is computable locally from these baselines.

### Decisions

- **D1: in-process first, HTTP second.** The official agents repo drives games
  over the Flask API (`arc_agi` serves it; agents POST actions). For fast
  iteration the harness drives `arc_agi`'s local environment wrapper
  in-process; the HTTP path gets one smoke test since it's the
  evaluation-shaped path. Re-verify parity before any submission.
- **D2: reuse shipped scoring code, don't reimplement.** RHAE report wraps
  `arc_agi.scorecard` classes so our numbers can't drift from the reference
  implementation. We hand-check one example against the formula anyway.
- **D3: agent contract mirrors the official one** (`choose_action(frames,
  latest_frame)` / `is_done(frames, latest_frame)`, returning
  `arcengine.GameAction`) so agents port to the official repo unchanged.

### Open questions — RESOLVED same day (source + experiment)

- **Q1: RESET costs an action — except full resets.** `Card.inc_reset_count`
  (scorecard.py:701) bumps `resets` AND `actions`. But `update_scorecard`
  (scorecard.py:834) routes a RESET with `full_reset=True` to `new_play()`
  instead — fresh counters, nothing charged. Engine default (`ARCBaseGame.
  handle_reset`, base_game.py:305): full reset only when `_action_count == 0`
  or state == WIN; otherwise RESET = *level* reset (restart current level,
  same play, +1 action) — **including after GAME_OVER**. Verified empirically:
  init RESET `full_reset=True` uncounted; 5 actions + 1 mid-game RESET →
  `actions=6, resets=1`. (Games *can* override `handle_reset`; engine also
  honors `ONLY_RESET_LEVELS=true` env var.)
- **Q2: deterministic — and seed is IGNORED (surprise).** Same 24-action
  sequence on ls20/vc33/ft09: seed 0 twice → identical frame hashes, and
  seed 0 vs seed 1 → also identical. These games take no randomness from the
  seed param at all (at least these 3, 24-action horizon). Exact replay is
  reliable → executable-world-model verification can assume deterministic
  transitions until a counterexample shows up. Re-test per game before
  relying on it broadly.
- **Q3: per-level action attribution is automatic.** Wrapper
  `_set_last_response` → `ScorecardManager.update_scorecard` →
  `Card.set_levels_completed` appends `(levels_completed,
  cumulative_actions)` whenever the level counter changes;
  `_calculate_score` diffs consecutive snapshots. Runner needs no level
  bookkeeping of its own — just use open_scorecard/make/close_scorecard.

### More findings with strategy weight

- **Per-game score = MAX over plays** (`EnvironmentScoreList.score`,
  scorecard.py:241); `levels_completed` = max, `actions` = sum
  (informational). A bad play does not poison the score → explore/execute
  split across plays is structurally free *if* you can start a new play.
  Within one wrapper that needs action_count==0 or a WIN; locally,
  re-`make()`-ing the env creates a new guid = new play. **VERIFY before
  relying on it: does the Kaggle eval notebook let the agent re-make games /
  start multiple plays?** (We don't have the eval notebook template locally;
  check the Kaggle competition Data/Code tab when preparing M1 submission.)
- **Off-menu actions are accepted, do nothing, and still cost an action.**
  ls20 advertises `available_actions=[1,2,3,4]`; sending ACTION6(32,32)
  anyway was counted (actions went up) with no frame change. Agents MUST
  filter by `available_actions` or bleed score silently.
- The initial `make()` already performs the first RESET (wrapper __init__
  calls reset()) — agents start from a live first frame; sending another
  RESET first thing would full-reset again (action_count==0 → free) but is
  pointless.
- Frames on ls20: 1 layer of 64×64 (the "stack" can be a single grid),
  16-color palette confirmed. `state` starts NOT_FINISHED right after init
  reset (never NOT_PLAYED from the agent's perspective).
- Tags in metadata: `keyboard` / `click` / `keyboard_click` — tells which
  action families a game uses. 25-game split visible in smoke output; useful
  prior for exploration (e.g. pure-`click` games like lf52/lp85/su15 need
  ACTION6 coordinate search, pure-`keyboard` like ls20/g50t/tr87/wa30 don't).
- Human baselines vary 6..578 actions per level (dc22 L6=578, sc25 L2=6) —
  per-level RHAE stakes are wildly uneven across games.

### Stage 2 verification record

- `smoke_env.py`: 25 environments discovered from kagglehub cache; ls20
  make/step/reset round-trip OK; scorecard math matched prediction exactly
  (actions=6, resets=1, level_actions=[6,0,...]).
- Determinism probe: 3 games × (seed0, seed0, seed1) × 24 actions —
  identical hashes everywhere.

### Stages 4–6: harness, baseline, HTTP path (all verified)

- **RHAE recomputation matches the shipped calculator 6/6** on synthetic
  cases (`scripts/verify_rhae.py`), including the subtle ones: a fully
  completed game caps at 100 even when 115%-capped levels push the weighted
  average above it; partially completed games cap at exactly
  completed-weight-fraction × 100 (e.g. 2-of-3 levels → 50.0 cap, observed
  49.9074 — the cap binds only when earned scores exceed it).
- **Random baseline, all 25 games, 300 scored actions each: mean RHAE
  0.30%.** Two lucky level-1 completions: cd82 2.86% (~71 actions vs
  baseline 55), sp80 4.76%. 25/25 reported scores match our independent
  recomputation — including the non-zero ones, so the verification now has
  teeth. Context: ARC Prize's May 2026 analysis put GPT-5.5 at 0.43% and
  Opus 4.7 at 0.18% — frontier LLM agents sit at the random floor.
- **Completion without efficiency is worth ~zero.** 5,000 random actions on
  the click games: lp85 and r11l completed level 1 yet scored 0.00% — at
  ~3,000 actions vs baseline 17, (17/3000)² ≈ 0.003%. The quadratic
  obliterates inefficient wins; even 2× baseline is only 25%. Random's
  nonzero all-25 mean is pure luck on two games, not signal.
- **Environment stepping is computationally free.** Whole 25-game sweep at
  300 actions/game: ~3.3s wall in-process (~0.04–1s per game; lf52/r11l the
  slowest). The 9-hour Kaggle budget will be spent ~entirely on the model,
  not the environments. Sets the exchange rate for "internal compute is
  free": simulation/planning loops are cheap even at large scale.
- **HTTP path (what the official agents repo speaks) verified offline** via
  Flask test client against `create_app()` (`scripts/smoke_http.py`):
  /api/games serves all 25; RESET takes `{game_id, card_id}` and returns the
  play `guid`; subsequent actions take `{game_id, guid}`; close_scorecard
  agreed with in-process counting (initial RESET uncounted, actions=1 after
  one ACTION1). In-process and HTTP paths share the same scorecard code, so
  parity is structural, not coincidental.
- Dead ends today: none material. One process note: probing the HTTP
  payload shape empirically via validation errors was faster than reading
  the API internals.

## 2026-06-09 — Workstream A: Play semantics

### VERDICT: MULTI_PLAY_FREE — win-gated in competition mode

Per-game score is the **max over plays**, each play's score is computed from
**that play's action counters alone** (earlier plays never pollute it), and
the official runtime lets an agent mint a new play — but in competition mode
**only by winning first**. "Explore in a throwaway play, then execute" is
NOT available under competition rules; "**win sloppily, then replay
cleanly**" is fully sanctioned by the code. Local non-competition modes
additionally allow new plays via re-`make()`.

### Code evidence (arc_agi 0.9.8 installed wheel; line numbers from source)

1. **A "play" is a positional entry in `Card`'s parallel lists** — fresh
   zeroed counters per play:
   `Card.inc_play_count` (scorecard.py:692-699) appends to `guids`,
   `levels_completed`, `states`, `actions`, `resets`, `actions_by_level`.
2. **What mints a play**: `Scorecard.update_scorecard` (scorecard.py:838-843)
   — `if data.action_input.id.value in [0]: if full_reset: self.new_play(...)
   else: self.reset(...)`. The engine sets `full_reset` only when
   `_action_count == 0 or self._state == GameState.WIN`
   (`ARCBaseGame.handle_reset`, base_game.py:305-316); otherwise RESET is a
   *level* reset: same play, +1 action (`inc_reset_count`,
   scorecard.py:701-704).
3. **Per-play isolation**: `_calculate_score(card, idx)` reads only
   `card.actions[idx]` / `card.actions_by_level[idx]` (scorecard.py:390-392,
   466-491). No cross-play term exists anywhere in the scoring path.
4. **Aggregation**: game score `= max(run.score for run in self.runs)`
   (`EnvironmentScoreList.score`, scorecard.py:237-241); overall score
   `= mean over games of that max` (`from_scorecard`, scorecard.py:612-618).
   `EnvironmentScoreList.actions` sums plays but is informational only.
5. **Competition gate #1 — no second environment instance per game**:
   `RestAPI._get_or_create_environment` (api.py:424-425):
   `if scorecard.competition_mode and scorecard.has_environment(game_id):
   return None, False` → re-make path refused with an error.
6. **Competition gate #2 — RESET@0 interception**: api.py:325-340 ("This is
   quite hacky as we have to look inside the underlying ARCBaseGame to check
   if this is the first action of the level and would cause a full reset") —
   in competition mode a RESET that *would* full-reset at `action_count==0`
   does not step the engine and is recorded as a **counted** reset on the
   current play. Empty plays cannot be minted.
7. **The WIN path is deliberately untouched**: RESET with state==WIN goes
   through `g.step()` → engine `full_reset()` → frame `full_reset=True` →
   `new_play` — same wrapper, same guid, fresh counters. Works in
   competition mode.
8. **Unplayed games count as zeros**: competition-mode `close_scorecard`
   pre-makes every available environment that has no card (api.py:202-215),
   so the final mean's denominator is ALL games, not attempted ones. Budget
   allocation must cover the full game set.
9. **Agents repo has no real ceiling**: `MAX_ACTIONS: int = 80  # to avoid
   looping forever if agent doesnt exit` (agents/agent.py:22); the Playback
   agent sets 1,000,000 (agent.py:202). No competition/eval template in the
   repo; agents speak HTTP to localhost:8001 — the competition_mode server
   tested below is the local mirror of evaluation semantics.

### Empirical ground truth (scripts/test_play_semantics.py, tt01 fixture)

Authored a 2-level test game (test_envs/tt01; baseline 3 actions/level;
ACTION1 completes a level, ACTION5 = counted no-op). Three experiments, all
PASS:

- **P1 re-make, normal mode**: sloppy win (12 actions → 25.0) then fresh
  wrapper, clean win (2 actions → 100.0). Game score **100.0**; play 2's
  `level_actions=[1,1]` untouched by play 1's 12 actions.
- **P2 RESET-after-WIN, one wrapper**: identical numbers, same guid — the
  competition-legal path works in-process.
- **P3 competition mode over HTTP**: second environment instance refused;
  RESET@0 intercepted (charged: play 1 shows `level_actions=[7,6]`, score
  22.79 vs 25.0); WIN→RESET minted play 2; clean replay → game score
  **100.0** from `max(plays)`.

**Kaggle caveat (flagged honestly):** all of the above is the local runtime
and its competition_mode mirror. The actual Kaggle evaluation wrapper is not
in the local repo (submission docs point at a form/notebook we don't have
here). Before relying on replay-after-WIN at submission time, eyeball the
Kaggle notebook template's agent loop for anything that closes the game or
scorecard at first WIN. Until then: architecture assumes win-gated
multi-play, with single-play as a config fallback.

### Architecture implication

The two-phase design survives, but its phase 1 is not "explore freely" — it
is "**reach a WIN with exploration debt allowed**." Exploration cannot be
quarantined from scoring until the first win exists; after that, replays are
score-free and the best play wins, so the world model's payoff compounds:
every replay is a fresh chance to execute the learned policy near-baseline,
and a failed replay costs nothing but wall-clock (which is the real budget:
9h across all games, incl. unplayed-games-count-as-zero pressure from #8).
Both modes are now in the harness behind `RunConfig.mode`
("single_play" | "two_phase"); a mid-competition rules patch costs a config
change, not a rewrite. Agents get `on_play_start(play_index)` to switch
policy between plays.

Demonstrated on tt01 with the random agent (budget 40): single_play stopped
at its first WIN → 85.42%; two_phase replayed 11 times and the max play hit
**100.0%** — +14.6 points from replay alone, with zero agent intelligence
added. RHAE verification matched on all 11 runs.

## 2026-06-09 — Workstream B: executable world model, first end-to-end loop

### What was built

`harness/wm/`: TransitionStore (per-game, keyed (level, frame_hash, action) —
levels alias frames, so level is part of observable state; conflicting
duplicates flagged as determinism violations), Rule/WorldModel (grid and/or
event claims; NO_PREDICTION = legal partial coverage; coverage report),
Verifier (exact replay; ≥3 exact + 0 miss = VERIFIED, any miss =
CONTRADICTED-but-retained), TemplateProposer (identity, translate +
blocked-identity, click recolor cell/component, move-onto-event pooled
across directions via fitted dirmaps, move-free, event-at-level),
DiffMemorizer (control arm), time-budgeted BFS planner (event claim required
to expand; GAME_OVER claims prune; VERIFIED-only default + allow-untested
penalty), WinSeeker (plan-first, then unseen → frame-changers → safe-any;
salience-capped clicks; never off-menu), WorldModelAgent (win-then-replay
via on_play_start; miss → append + replan + re-propose; bail-out →
ABANDONED). Tests: synthetic 8×8 env (true rules recovered VERIFIED,
optimal 10-step plan, hazard avoided, planted CONTRADICTED rule excluded);
tt01 win-gate flow.

### Acceptance verdict

- **tt01: MET for both proposers** — play 1 WIN (WinSeeker, 4 actions),
  play 2 a fully planner-driven 2-action replay, prediction match 2/2
  (=100% ≥ 80%), game score 100. Caveat recorded: play 1 also hit the 100
  cap (baseline 3/level is generous), so "strictly better RHAE" was
  realized as strictly fewer actions (4→2), not a higher capped score.
- **cd82, sb26: NOT MET — and the failure is informative.** Play 1 never
  reached WIN, so the replay machinery never engaged. The win-gate (from
  Workstream A) is the binding constraint on real games: the bottleneck is
  *winning at all*, not replay quality. cd82: 2/6 levels, game score 0.11%
  (26k exploration actions — efficiency-or-zero reconfirmed at scale).
  sb26: 0/8 levels despite 26k actions. Both ABANDONED by the bail-out as
  designed (a sloppy partial play beats starving other games).

### Wall-clock per phase (first real datapoint for 110-games-in-9h)

| run | total | propose | verify | plan | stepping+rest |
|---|---|---|---|---|---|
| cd82 template (240s budget) | 148s play | 34.6s | 3.1s | 63.7s | ~47s (26.5k actions) |
| sb26 template | 144s | 57.8s | 0.1s | 48.4s | ~38s (26.2k actions) |
| cd82 memo (120s) | 89s | 0 | 2.6s | 82.4s | ~4s |
| sb26 memo | 72s | 0 | 7.4s | 57.9s | ~7s |

Lessons: (1) unthrottled propose initially consumed **96%** of wall-clock
(139.7s/145s) — fixed with sample-capped fitting (≤120 transitions/action;
verifier stays full-store) + an adaptive cap (modeling ≤25% of elapsed),
which took action throughput from ~14/s to ~180/s. (2) The planner is the
new hot spot: a failed plan attempt runs every step while no plan exists —
needs "replan only on model/context change" throttling. (3) Environment
stepping remains ~free; in-process actions are not the cost, thinking is.

### Where TemplateProposer ran out of expressiveness (requirements)

Diagnosed from stored transitions (runs/wm/*-store.pkl), not speculation:

1. **R1 — region factoring / HUD masking.** cd82 ticks an action-meter cell
   in row 63 (color 4→5) on EVERY action, so no transition is ever
   "identity" and whole-frame templates can never fit. Requirement: factor
   the frame into independently-modeled regions ("identity outside region
   R" + per-region sub-rules), discovered from change-location statistics.
2. **R2 — multi-color rigid-body motion.** cd82's content moves ~200
   mixed-color cells {2,5,15} as a unit; single-color translate cannot
   express it. Requirement: region/sprite translation with occlusion-aware
   vacated-fill.
3. **R3 — latent-state event preconditions.** sb26 (ACTION1-4 inert; 5/6/7
   drive a click-sequence mechanism with animation timers) advances levels
   on conditions like "k-th correct match", i.e. hidden counters. Pixel-
   enumeration templates cannot express hidden state; this is the concrete
   case for the coder-model proposer (write a Python world model with
   variables), per phase1-v2 §3.

### Luck vs competence (honesty section)

- tt01's play-1 win is **luck by construction**: 2 available actions, the
  winning one found by unseen-first enumeration almost immediately, and
  near-baseline incidentally. tt01 proves loop *plumbing*, not modeling.
- cd82's 2 levels: WinSeeker unseen-flooding, not model competence —
  **grid_coverage was 0.0** (the model never predicted a single full grid
  on either real game; every grid template was blocked by R1/R2).
- The high match rates (0.97–0.99) on real games are misleading in
  isolation: they cover only `predicted_steps` (a small minority of
  actions, mostly event-claims), not dynamics mastery. Report
  predicted_steps/actions alongside match_rate always.

### Determinism + guards held at scale

Zero store conflicts across ~32k transitions on two real games (exact-replay
verification stays sound); zero off-menu violations (dev-mode assert never
fired); RHAE verification matched the shipped scorecard on all runs incl.
the 11-play two_phase tt01 record.

## 2026-06-09 — Workstream C part 1: instrumentation + recorded baseline

### Measurement layer (no behavior changes — baseline reflects B's agent exactly)

- `harness/wm/metrics.py`: JSONL event stream per run (gzipped) + summary.
  Events: `coverage` (after every store update AND after every rule refresh;
  incremental O(rules)-per-action coverage counters in WorldModel so
  per-action emission doesn't recreate the propose cost trap), `plan`
  (planned vs executed length, confidence, retirement reason, planner call
  count), `phase` (six buckets: exploration/proposing/verifying/planning/
  executing/env_stepping — agent owns five, runner times env stepping),
  `outcome` (ledger row per game).
- **HonestMatch triple** (matched, predicted_steps, total_actions) is the
  ONLY way match quality leaves the system — encodes the Workstream B
  honesty rule structurally. The summary table prints all three columns.
- `scripts/plot_metrics.py`: coverage-over-experience per game,
  phase-economics stacked bars, and `--compare` mode for run-vs-run deltas.
  These are the writeup figures; axes labeled.
- Outcome taxonomy: WIN_REPLAYED / WIN_UNREPLAYED / ABANDONED / TIMEOUT.

### Recorded baseline (results/baseline-c1*, committed as the reference)

Template proposer, two_phase, 240s/game, action budget 50k — the Workstream
B configuration, now instrumented. Two run dirs because tt01 lives in
test_envs (single-env-dir runtime): `baseline-c1-tt01` + `baseline-c1`.

| game | status | best RHAE | levels | matched | predicted | actions |
|---|---|---|---|---|---|---|
| tt01 | WIN_REPLAYED | 100.0 | 2/2 | 2 | 2 | 6 |
| cd82 | ABANDONED | 0.11 | 2/6 | 1114 | 1124 | 31,570 |
| sb26 | ABANDONED | 0.00 | 0/8 | 2878 | 2947 | 31,819 |

Reading the honest triples: cd82's "99.1% match" covers only **3.6%** of
actions (1124/31570 predicted); sb26's "97.7%" covers 9.3%. The coverage
figure makes the R1 gap visceral: **grid coverage is a flat 0.0 line across
31k actions on cd82** (every grid template blocked by the HUD tick), and
event coverage *decays* toward 0 as the store grows faster than event rules
generalize. This is the "before" plot that Part 2 (R1 region factoring) must
move; the comparison mode is ready for that delta.

Baseline replication note: behavior matches the B run (cd82 2 levels →
0.11%, sb26 0 levels, tt01 win-then-replay 100%), so the instrumentation
itself didn't perturb the agent.

## 2026-06-09 — Workstream C parts 2+3: region factoring + planner economics

### Delta table vs results/baseline-c1 (template, two_phase, 240s/game)

| metric | cd82 base → R1+fix | sb26 base → R1+fix | tt01 |
|---|---|---|---|
| planning s/game | 64.6 → **0.19** (340×) | 55.8 → **0.13** (429×) | — |
| actions in 240s | 31,570 → **110,510** | 31,819 → 61,099 | 6 → 6 |
| grid coverage | 0.0 → **0.0 (NOT unblocked)** | 0.0 → 0.0 | n/a (events only) |
| HUD detected | none → **64 cells, row 63** | none (none exists) | none |
| best RHAE / levels | 0.11 / 2 → 0.11 / 2 | 0.0 / 0 → 0.0 / 0 | 100 → **100 (canary ✓)** |
| match (honest) | 1114/1124 of 31,570 → 738/749 of 110,510 | 2878/2947 → 5533/5652 of 61,099 | 2/2 of 6 |

### Part 3 (planner economics): ACCEPTED, decisively

Replan-on-trigger + per-(context, model_version) plan caching killed the
every-step replanning cost: planning fell from 48–82s/game to **0.1–0.2s**,
and the freed time turned into 1.9–3.5× more exploration actions in the
same budget. Trigger histograms now in the phase events — cd82:
{game_over: 1104, model_change: 75, miss: 11} (it dies constantly; that's
the meter running out), sb26: {miss: 119, game_over: 112}. Cache hit rates
~10:1 vs planner calls.

### Part 2 (R1): split verdict — detection unblocked, coverage NOT

**What works:** the RegionAnalyzer finds cd82's meter exactly — 64 cells,
all of row 63, masked. It took three membership criteria to get there
(research log, honestly):
1. per-cell change-rate ≥0.9 — failed: meter cells individually change
   rarely (the front sweeps past each cell), it's a REGION property;
2. multi-base "ever-changed" clustering — failed two ways: play-area cells
   are multi-base changers too and bridge clusters (toy), and the
   cluster-activity bar of 0.9 missed cd82's 0.64 (meter skips no-op
   actions);
3. **sole-changer seeds + action-exogeneity** (final): cells repeatedly the
   SOLE change of a transition, where for clicks the change must be far
   from the click coordinate. Grounded in store evidence: 13,147 of 48k
   cd82 transitions are 1-cell diffs, all 64 cells in row 63, soloing
   242–493× each; content diffs are always ≥5 cells. Exogeneity is what
   separates a meter from an interactive click-board (whose cells change AT
   the click — masking those would gut click games). Activity bar lowered
   to majority (0.55) accordingly.

Guard lessons: the unmaskable-colors guard initially read CONTRADICTED
rules — one bogus sampled move_onto naming content colors 4/5 stripped the
whole meter row (meter shares those colors). Falsified rules must never
feed the mask. Toy trap tests (goal-colored blinker beside the ticker:
armed unguarded, disarmed by the color guard) and the ablation flag
(factoring off reproduces grid coverage 0.0 + zero grid templates — the
cd82 pathology in miniature) both pass.

**What doesn't:** grid coverage stayed **0.0 on every real game tested**
(cd82, sb26, plus a ls20/tr87 probe to check the machinery beyond cd82 —
ls20 emits zero rules at all). Behind cd82's HUD sit two further walls,
both already named in the B-session requirements:
- **R2 evidence, sharpened:** a single cd82 click triggers ~95-cell
  structured redraws (colors {0,3,15}→{0,3}); movement shifts ~200
  mixed-color cells as a body. No identity rule is even emittable — every
  action sometimes changes content, so unconditional per-action identity
  dies on the sample. cd82's HUD state-table rule also failed consistency:
  the meter ticks iff the action is "valid" — even the HUD needs
  conditional rules.
- **R3 evidence, new and quantified:** cd82 store conflicts (same level +
  frame + action → different outcome) grew to **1,791** — the frame is not
  Markov; there is latent state. sb26 had exactly 1 conflict — its
  blocker is latent state of a different kind (the level-advance condition,
  not the transition function). Determinism-as-assumed held only 3-for-3 on
  the B-session probe games; cd82 falsifies it as a universal.

**Acceptance verdict, straight:** Part 3 accepted; tt01 canary accepted;
sb26-stuck-as-predicted confirmed; **Part 2's binary criterion (cd82 grid
coverage off 0.0) NOT met** — masking was necessary but not sufficient.
The honest-match columns make the same point from the action side: cd82's
"98.5% match" covers 749 of 110,510 actions (0.7%).

### Updated 110-games-in-9h projection

Harness overhead is no longer the constraint: at 240s/game, 110 games =
7.3h of budget with planning at ~0.2s/game and stepping ~10s/game; at
180s/game = 5.5h, leaving ~3.5h for model inference — the entire remaining
budget question is the proposer's LLM calls (as designed: internal compute
is the scored-free resource). Two operational caveats before any full-set
run: (1) the store holds full grids — ~100k transitions ≈ 1.6GB RAM on
cd82-class games in one 240s budget; needs bounding/compression. (2)
propose() still costs 40–63s/game under the 25% modeling cap; the
coder-model proposer will live inside that same cap.

## 2026-06-09 — LLM proposer smoke test (throwaway, gates Workstream D)

### Hardware + model (recorded per spec)

Local: **Apple M4, 24GB unified memory, 10 cores, Metal 3** — passes the
7-8B bar comfortably. Model: **qwen2.5-coder:14b (ollama, q4_K_M, ~9GB)** —
strongest *reliable* fit; Qwen3-Coder-30B-A3B is stronger on paper but its
~18-19GB q4 footprint collides with the ~18GB Metal wired-memory ceiling on
24GB, and a flaky load would have burned the day. Throughput: 9–82s/sample
(taskA, 2.7k-char prompt), 19–82s (taskB after the fix below). 20 samples ≈
30 min wall. Local dev benchmarking is viable on this machine; a dev box
becomes worthwhile at the bake-off stage (Part 2 of this workstream), not
before.

### Task construction (deviations logged honestly)

- **sb26 substitution:** the spec's task (a) wanted sb26's level-advance
  precondition — but sb26 has ZERO level-advance observations in any store
  (31,469 transitions: 31,413 NONE, 56 GAME_OVER). You cannot prompt for
  evidence never gathered; the exploration problem precedes the proposer
  problem there. Substituted: **cd82 GAME_OVER precondition** (314
  positives; same R3 task shape). The meter makes it learnable: GAME_OVER
  pre-frames show the row-63 meter drained to ≤1 remaining '4' cell.
  Held-out (30 GO / 30 NONE) built adversarially with near-drained NONE
  meters so "any drained cell ⇒ GO" can't win free. Reference rule
  (`count('4') <= 1`) scores 1.0; always-NONE floor 0.5.
- **Task B = cd82 click structured response.** Scorer validation caught an
  unfair first construction: the response set depends on game phase, so a
  fixed-set hypothesis scores 0.0 on held-out — evidence didn't determine
  the answer. Fixed by including each observation's color-15 cell positions
  (the toggle's visible state). Calibration: naive "changed = current
  15-cells" reference scores **0.294** mean Jaccard (0.58–0.61 off-toggle,
  0.0 on-toggle — the appearing pattern isn't in the pre-frame; that half
  stays legitimately hard).
- **Serialization is load-bearing:** the first task-B prompt (raw
  coordinate JSON, 20k chars) silently exceeded num_ctx after tokenization;
  ollama truncated and the model emitted a bare ``` fence — all 10 samples,
  3 bytes each. Row-span encoding (`r7:35-47`) cut it to 4.8k chars and
  made the structure legible. Lesson for D: token-budget the evidence
  serializer, and treat <50-char responses as harness errors, not model
  failures.

### Results (qwen2.5-coder:14b, temp 0.8, 10 samples/task, scratch/smoke/)

Task A (event precondition), accuracy on 60 balanced held-out:
**best 0.90**, then 0.70, rest 0.17–0.53 (floor 0.5). Every sample engaged
the meter structure; the failures are directional/threshold errors, not
incoherence.

Task B (structured response), mean Jaccard on 6 held-out clicks:
**best 0.294 — exactly the naive reference** (1/10), the other nine ≈ 0.0.

Representative generations (verbatim, full set in scratch/smoke/):

- A best (0.90 — right feature, wrong boundary; the 6 FPs are the
  near-drained NONE meters): `if "5" in meter_row63 and
  meter_row63.count("4") >= 5: return "NONE" else: return "GAME_OVER"`
- A failure mode (0.17 — right feature, INVERTED): `if '4'*5 in
  meter_row63 and action == "ACTION6": return "GAME_OVER"`
- B best (0.294 — engages cells15, collapses to identity-on-15-cells via a
  bogus "symmetry around the click"): reflects each 15-cell through the
  click coordinate.
- B modal failure (0.0): pure click-centric geometry fantasy ("vertical
  strip centered on the clicked column, width = click row") — a generic
  puzzle prior overriding the actual evidence, which shows changed cells
  FAR from the click.

### VERDICT: **PARTIAL** (not rounding up)

No sample was substantially correct: A's best is one threshold constant
away from 1.0 but carries 6 false positives on a 60-item held-out; B never
beat the naive baseline. But every A sample and most B samples are coherent
attempts wrong in *describable* ways: (1) right feature, wrong
threshold/direction (A); (2) generic click-centric geometric priors
overriding evidence (B); (3) nothing engaged the on-toggle hidden-pattern
half at all. Implication for D's framing: failure mode (1) is exactly what
a verify-and-repair loop fixes mechanically — feed back the 6
counterexamples and the constant snaps into place. Mode (2) needs
counterexample-driven REFRAMING, not constant-tuning — D's repair prompt
must be able to say "your prediction was geometrically anchored to the
click; the misses are all far from it." Mode (3) suggests the proposer
needs temporal context (previous toggle states) in the evidence, not just
single transitions. PARTIAL is the gate-friendly outcome the spec
anticipated; the D sandbox exists to convert it.

## 2026-06-09 — Bake-off Phase A: candidate slate + bench harness (no spend)

### Slate of 4 (availability + licenses re-verified today, not from memory)

| arm | model (exact ID) | params | license | deploy quant to test | serving stack |
|---|---|---|---|---|---|
| ≤8B | Qwen/Qwen2.5-Coder-7B-Instruct | 7B dense | Apache-2.0 | AWQ-INT4 | vLLM |
| ~14B (reference) | Qwen/Qwen2.5-Coder-14B-Instruct | 14B dense | Apache-2.0 | AWQ-INT4 | vLLM |
| 30B-class | zai-org/GLM-4.7-Flash | 30B-A3B MoE | MIT/Apache lineage — **re-verify license field at pull time** | FP8 (NVFP4 variant exists; MXFP4-MoE GGUF fallback) | vLLM |
| top arm | Qwen/Qwen3-Coder-Next | 80B-A3B MoE | apache-2.0 (verified on HF) | INT4/AWQ ~44GB (FP8 is ~80GB — too thin on 96GB with KV) | vLLM |

Notes: (1) Qwen3-Coder-Next is 80B TOTAL — above the literal 30–70B band —
but 3B-active with a 44GB INT4 footprint, i.e. inside the band's resource
envelope, and it's the strongest Apache-2.0 agentic coder currently
available (Feb 2026 release; 256K ctx). Dense Llama-3.3-70B was rejected on
license (fails the competition's OSI-checklist open-source requirement);
Qwen2.5-Coder-32B is the named fallback if Next's INT4 disappoints.
(2) Serving stack is vLLM for all four on GPU — the stack we'd actually
ship in the submission notebook (throughput numbers must transfer); ollama
remains local-dev-only on the Mac. (3) qwen2.5-coder:14b stays as the
smoke-calibrated reference point.

### Bench harness (bench/, ported from scratch/smoke which stays as record)

- Tasks rebuilt with three splits for A (prompt / feedback / final held-out
  15GO+15NONE) so T-repair's counterexamples never touch the scored set.
- **T-repair**: one-round counterexample feedback (up to 6 misses from the
  feedback split, the smoke test's 6-false-positive class) — directly
  measures the repair-loop lift the thesis depends on. **T-reframe**:
  Task B + adversarial reframing line ("changed cells are FAR from the
  click; anchor on the color-15 state") + temporal context (previous
  transition per exemplar; temporal order = store insertion order, true for
  single-play stores — logged assumption).
- Prompt versions hashed in bench/tasks/manifest.json: A a2997353…, B
  db85e0d8…, reframe a36db5ad… N=8/task/model, temperature 0.8 (logged in
  every results file).
- Calibration (bench/tasks/calibration.json): A reference 1.0 / floor 0.5
  on the new final held-out; B and reframe naive reference 0.295.
- Local end-to-end proof: repair loop ran on qwen2.5-coder:14b (2 samples,
  0.467→0.5 and 0.5→0.5) — the rental session will not be the harness's
  first full run.

### Pre-spend estimate (logged BEFORE any spend, per ground rules)

Rental: 1× RTX PRO 6000 Blackwell 96GB — matches Kaggle eval hardware, so
throughput observations transfer. Marketplace rate ~$1.7–2.5/h (2026).
Workload: 4 models × 40 generations (A/B/reframe @8 + repair @8×2 rounds)
= 160 gens; MoE-3B-active and dense ≤14B at vLLM speeds ≈ 25 min total
generation; budget dominated by weights download (~85–120GB) + 4 vLLM
loads ≈ 1.5–2h. **Estimate: ~3 GPU-hours ≈ $5–8; hard ceiling $15** incl.
retries and a quant fallback. Local Phase-A spend: $0.

Disclosure boundary (standing): the public-leaning throughput notebook gets
ONLY generic load/timing code with neutral stand-in prompts of matched
length/structure; real evidence prompts run on the rental and locally only.

## 2026-06-09 — Bake-off Phase B: runbook handed off (spend gate with user)

bench/RUNBOOK.md written for an ephemeral RunPod/Vast box: instance order
(RTX PRO 6000 96GB preferred — Kaggle-matching; H100/A100 fallbacks with
quant consequences spelled out), bring-up with reproducibility-by-recording
(pip freeze + nvidia-smi land in every tarball), per-model run commands
with the Qwen3-Coder-Next quant decision tree (official AWQ → community AWQ
→ named fallback Qwen2.5-Coder-32B-AWQ; do not burn hours on quant bugs),
collect-verify-destroy procedure. `bench/run_quality.py` is the single
boxside entrypoint: serves via vLLM, runs A/B/reframe/repair at N=8, writes
one self-contained tarball per model (generations verbatim, scores, prompt
hashes, calibration, vllm log, env snapshot, timing). bench/ verified
import-clean of harness/numpy (the box ships only bench/ via scp — no git
credentials on hostile boxes). `bench/analyze.py` ready for when tarballs
land: rates vs calibrated refs, repair/reframe lifts, lucky-single-gen
callouts, hypothesis-comment corpus for failure-pattern notes.

Spend: user executes the rental (est. $5–8, ceiling $15 at estimate rates;
phase budget $20–60 covers hot marketplaces/retries). Waiting on tarballs.

## 2026-06-09 — Bake-off Phase C: Kaggle throughput notebook prepared (free)

Two companion notebooks in bench/kaggle/ + STEPS.md, respecting the
disclosure boundary (neutral synthetic sensor-log prompts, matched in
length/shape only: 2.7k/3.8k/5.4k chars ≈ the A/B/reframe distribution;
no evidence, no task framing; notebooks stay private regardless).

- make_wheels_notebook.py (internet ON): pip-downloads vLLM + deps into a
  private dataset — attaching it to the offline notebook and installing
  with `--no-index` IS the proof that the serving stack can exist in the
  competition environment (the known vendored-wheels trap, tested head-on).
- throughput_kaggle.py (competition notebook, internet OFF): env facts
  (GPU/VRAM/RAM/disk/internet-reachability), offline-install proof with a
  stack-level disqualification exit if it fails, then per candidate IN A
  FRESH SUBPROCESS: cold-load seconds, prefill s + prefill tps per prompt
  size, decode tps (600-token budget), peak GPU GB. Auto-discovers attached
  weight dirs; finalists = Phase B top 2 (+7B iff repair-lift real) — flags
  flip without editing code, since Phase B hasn't landed yet.
- Hard rule logged: cannot-serve = DISQUALIFIED regardless of quality.
  Decision thresholds written into STEPS.md (cold-load vs 9h fleet math,
  decode tps → proposer-calls-per-game arithmetic for D).

User actions: run wheels notebook, publish weights datasets (Kaggle Models
preferred; HF snapshot fallback documented), run throughput notebook, drop
throughput_results.json into bench/results/. Spend: $0. Waiting on numbers
(and on Phase B tarballs, which decide the finalist flags).

## 2026-06-09 — Bake-off Phase D: DECISION MEMO (provisional by construction)

**Status of inputs, stated first because honesty:** Phase B tarballs (rental)
and Phase C throughput JSON (Kaggle) have NOT landed — both are user-run
gates still open. D-0 Part 2 (25-game sweep) never ran (redirected), so
harness overhead uses the spec's ~115s/game placeholder, flagged. Nothing
below is invented: every number is tagged MEASURED (local), PLACEHOLDER
(flagged), or PENDING (slot defined, recompute on arrival).

### Input table

| quantity | value | status |
|---|---|---|
| harness overhead /game (non-LLM) | ~115s | PLACEHOLDER (sweep not run; r1-c2 two-game data ≈ 95–145s) |
| model-time envelope | 9h − 110×115s ≈ **5.4h** ≈ 177s/game | derived from placeholder |
| 14B quality, task A | best 0.90 (1/10 ≥0.9; mean 0.52; floor 0.5) | MEASURED (local, N=10) |
| 14B quality, task B | best 0.294 = naive ref (1/10); rest ≈0 | MEASURED (local, N=10) |
| 14B repair lift | +0.033, +0.000 | MEASURED but N=2 — not evidence yet |
| tokens/attempt | A ≈ 0.7k in + ~85 out; B ≈ 1.1k in + ~245 out | MEASURED (local gen corpus) |
| 14B decode on M4 (ollama q4) | ~8–15 tok/s | MEASURED, NON-TARGET hardware |
| target-hw decode (RTX 6000, vLLM AWQ) | 14B ~60–100 tok/s; MoE-3B-active ~80–150 | ESTIMATE, flagged → Phase C |
| big-model quality (GLM-4.7-Flash, Next) | — | PENDING Phase B |
| offline servability + cold-load | — | PENDING Phase C |

### The arithmetic, end to end (with today's inputs)

verified_rules_affordable/game = envelope/game ÷ (attempts/rule × s/attempt)

- s/attempt (14B-class, target-hw estimate): prefill ~1k tok ≈ 0.3s +
  decode ~150 tok @ ~70 tok/s ≈ 2.1s + overhead ≈ **~3s** (M4 measured 11s).
- attempts/rule: the load-bearing measured fact — raw sampling at 14B
  produced **zero** verified-clean rules in 10 tries on task A; one 0.9-class
  near-miss. The whole affordability calculation only closes if the repair
  loop converts near-misses, i.e. attempts/rule ≈ 10 samples + ~2 repair
  rounds ≈ **12 attempts ≈ 36s/rule** — vs ∞ without repair. The thesis is
  now a measured arithmetic dependency, not a design preference.
- ⇒ at 14B-class: 177s ÷ 36s ≈ **~5 verified rules/game** affordable — IF
  the repair conversion holds (PENDING T-repair at N=8 across the slate).
- Structured-response (B-class) rules: not affordable at any sampling depth
  measured so far (best = naive baseline); gated on reframing lift, PENDING.

### Decisions (conditional, switch conditions explicit)

- **Provisional pick: Qwen3-Coder-Next-80B-A3B, INT4/AWQ, vLLM**, on three
  conditions: (i) serves offline on Kaggle (Phase C install + no DQ row),
  (ii) decode ≥ ~40 tok/s there, (iii) Phase B shows a real quality margin
  over the 14B reference on task A and repair-lift. 3B-active means its
  s/attempt should be near-14B while quality should be far better — if both
  hold, the pick is arithmetic, not taste.
- **Runner-up: GLM-4.7-Flash.** Switch if: Next is DQ'd offline (quant/wheel
  trouble — its decision tree already has the named fallback
  Qwen2.5-Coder-32B-AWQ), or GLM lands within ~10% of Next's task-A success
  at materially better tokens/s.
- **≤8B verdict: OPEN, leaning no.** The only repair data (N=2, 14B) showed
  +0.033/+0.0 — nothing yet says repair can carry a small model. Viability
  condition, pre-registered: 7B is the fallback (or pick) only if its
  repaired task-A success reaches 0.9-class within ≤3 rounds in Phase B.
- **Quality-vs-throughput tiebreak, pre-registered** so it can't be resolved
  silently later: small wins only if attempts_small/attempts_big <
  s_big/s_small. With 3B-active MoEs, s_big/s_small ≈ 1 — so quality should
  dominate; if Phase B/C contradict this, the memo gets the tradeoff table,
  not a quiet pick.
- **Top failure pattern (only measured model, 14B): click-centric geometric
  prior overriding evidence** — 9/10 task-B samples anchored on the click
  coordinate against the data. This is Workstream D Part 2's
  prompt-engineering target; T-reframe already operationalizes it and its
  lift is the first number to read out of the tarballs.
- **Cold-load amortization: ONE RESIDENT MODEL across all 110 games** —
  decided now on structure, not pending data: even a 60s per-game reload
  costs 110×60s = 1.8h = a third of the model-time envelope; Next-AWQ at
  ~44GB + KV fits 96GB with no need to swap. The Phase C cold_load_s only
  needs to clear "one load at startup ≪ budget" (anything under ~10 min is
  fine); per-game reload is arithmetically dead regardless.

### Update protocol (when artifacts land)

Phase B tarballs → bench/analyze.py → replace rows 4–6+9, recompute
attempts/rule per model, settle the ≤8B verdict and the failure-pattern
note per finalist. Phase C JSON → replace throughput estimates + servability
+ cold-load; apply DQ rule mechanically. If the recomputed pick differs
from the provisional one, the memo gets amended in place with a dated
correction — not silently rewritten.

## 2026-06-10 — Bake-off Phase B: RESULTS (rented RTX PRO 6000, ~1.7h box time)

All four slate tarballs verified locally (plus a fifth: GLM's thinking-mode
negative record). N=8/task, temp 0.8, vLLM 0.22.1, AWQ/BF16 as slated.
Distributions below, not best-runs; lifts arrays shown in full.

| model | A best/mean | A≥0.9 | repair lifts (mean) | reframe best | format errors | load |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-AWQ | 0.63 / 0.52 | 0 | [-.07,0,-.07,0,0,**+.50**,+.33,0] (+0.087) | 0.08 (lucky-1-gen) | 0 | 90s |
| Qwen2.5-Coder-14B-AWQ | 0.87 / 0.66 | 0 | [+.03,-.07,0,0,0,-.03,-.10,-.10] (**−0.033**) | 0.18 | 0 | 65s |
| GLM-4.7-Flash (nothink) | 0.53 / 0.50 | 0 | [-.23,**+.50**,0,+.07,+.07,+.37,0,+.10] (+0.108) | 0.29 | 1/24 | 55s |
| Qwen3-Coder-Next-AWQ | **0.90** / 0.56 | **1** | [+.03,-.03,**+.50**,0,0,**+.50**] (+0.167) | **0.41** | 17/24 (!) | 220s |

### Verdicts per model

- **Qwen3-Coder-Next (bullpoint AWQ-4bit): quality leader with a named
  defect.** Only model with a substantial (≥0.9) raw task-A sample; best
  repair lift incl. TWO 0.5→1.0 full conversions; and the project's first
  structured-response hypothesis to beat the naive baseline — reframe
  0.411 mean with off-toggle Jaccards 0.84/0.78/0.84 vs naive's 0.58-0.61.
  HONESTY CALLOUT the lucky-flag heuristic can't see: that 0.411 is the
  ONLY completed reframe gen of 8 — the other 7 truncated. Failure pattern
  (the D Part 2 prompt-engineering target): verbose analysis prose
  exhausts the 1500-token budget before the code block — the analysis is
  GOOD (it was deriving per-click offsets from the data), it just never
  ships code. Mechanical mitigations exist (bigger budget, terse-output
  instruction, two-turn analyze-then-code).
- **GLM-4.7-Flash (license field verified: `mit`): the efficient
  workhorse.** Mediocre raw A (0.53 best) but solid repair (+0.108, one
  0.5→1.0) and reframe 0.29 ≈ naive. Fastest load (55s). Thinking mode is
  UNUSABLE at deployment budgets: 24/24 gens consumed 1500 tokens of
  reasoning without emitting code (archived as glm47-flash-think.tar.gz);
  nothink via chat_template_kwargs is mandatory and logged in results.
- **14B reference: raw quality second (0.87/0.66) but REPAIR-NEGATIVE**
  (−0.033; 4 of 8 degraded). Its decent round-1 rules get overcorrected
  toward the counterexamples. Design consequence for the D sandbox,
  now evidence-backed: **verify-gated acceptance** — keep the revision
  only if it improves on the feedback split; never adopt blind.
- **7B: repair CAN occasionally carry it** (0.5→1.0 once, +0.33 once) but
  raw quality is floor-hugging (mean 0.52) and it ignores the reframing
  instruction entirely (still writes "rectangular area centered at the
  clicked position" directly under the warning saying changes are far from
  the click). Pre-registered viability bar (0.9-class within ≤3 rounds)
  NOT met on this evidence: fallback-only.

### Cross-model facts

- Repair-lift ranking: Next +0.167 > GLM +0.108 > 7B +0.087 > 14B −0.033.
  Three of four models produced at least one full 0.5→1.0 repair — the
  repair loop is real, but it is high-variance and occasionally
  destructive: the sandbox must gate acceptance on verification.
- Task B (un-reframed) stayed ≈0 for everyone — click-centric geometric
  priors are universal across families at these scales. The reframing
  prompt is what unlocked Next's 0.41; prompt engineering moves this
  needle, scale alone does not.
- Throughput texture (target-class GPU): 40-gen suites ran in 52-272s;
  decode ~115 tok/s observed on GLM-30B-A3B mid-run. Cold loads 55-220s —
  all trivially amortized by the one-resident-model strategy.

## 2026-06-10 — Blackwell deployment quirks (Kaggle-relevant intel)

Learned on an RTX PRO 6000 (sm_120) rental — Kaggle's eval pool runs the
same silicon, so these feed the Phase C notebook design directly:

1. **FlashInfer (0.6.11) fails its arch check on sm_120** ("requires sm75
   or higher" — the check predates the card) from THREE separate entry
   points: attention backend, JIT sampling (`gen_sampling_module` — crashed
   the dense Qwens during the engine's profile_run), and MoE autotune
   (`compilation_context.get_nvcc_flags_list`: "No supported CUDA
   architectures found for major versions [12]" — crashed GLM). Partial
   env-var fixes are whack-a-mole; **the durable fix is uninstalling
   flashinfer from the venv** — vLLM 0.22.1 falls back to FLASH_ATTN /
   native sampler / triton MoE cleanly. Do NOT pip-upgrade flashinfer on a
   cu13 stack: pip's resolver downgrades torch to cu12 and poisons the
   venv (cost ~30 min of the manual session). For the Kaggle wheels
   dataset: EXCLUDE flashinfer wheels entirely.
2. `VLLM_ATTENTION_BACKEND` is an UNKNOWN env var to vLLM 0.22.1 (warning
   in logs) — the earlier "fix" was a no-op; dense models were auto-
   selecting FLASH_ATTN anyway. `VLLM_USE_FLASHINFER_SAMPLER=0` IS still
   honored and was the real fix for the dense models (now moot post-
   uninstall, kept as defense-in-depth in run_quality's spawn env).
3. ~8GB of VRAM can be squatted by zombie allocations on rental boxes —
   `--gpu-memory-utilization 0.85` as standing policy.
4. HF cache must be pinned off quota'd network volumes
   (HF_HOME=/root/.cache/huggingface; /workspace had a 10GB quota).
5. Non-interactive ssh skips bashrc: every box command carries its env
   inline; run_quality.py now bakes the quirk env in as overridable
   defaults (env-passthrough through Popen was verified NOT the bug —
   os.environ is merged; the log's "Using FLASH_ATTN" line proved env
   reached the engine).
6. `huggingface-cli` is deprecated/broken on current images (prints help,
   downloads nothing) — the CLI is `hf` now. Cost: parallel pre-downloads
   silently didn't run; vLLM's own downloader saved the schedule.

### Phase D memo updates (rows formerly PENDING-PhaseB)

- Big-model quality: MEASURED (table above). Provisional pick survives
  with a sharpened condition: **Qwen3-Coder-Next remains the pick** IF
  Phase C confirms it serves offline AND D Part 2 fixes the
  truncation-verbosity defect (mechanical mitigations available);
  runner-up GLM-4.7-Flash is now genuinely attractive (MIT, 55s loads,
  +0.108 repair, 4× cheaper VRAM) — switch trigger unchanged.
- ≤8B verdict: bar NOT met → fallback-only.
- attempts/rule (14B-class row): with repair-gating, the Next data implies
  ~6 attempts/verified-A-rule (2 conversions in 6 completed pairs + 1 raw
  substantial in 8) — better than the 12 placeholder; still
  structured-response-blocked on everything but reframed-Next.
- Phase C finalists per the spec rule (top 2 + 7B-iff-repair-real):
  **Qwen3-Coder-Next + GLM-4.7-Flash; 7B excluded** (bar missed; its two
  conversions are real but rest on a floor-quality base).

## 2026-06-10 — Milestone 1 public notebook (built; submission = user action)

### Public/private boundary decisions (recorded per spec)

SHIPPED (kaggle_m1/m1_notebook.py, single file, clean-room rewrite — no
modules copied): observation store (bounded, neutral "keep informative
observations" framing), status-region detection (NEUTRAL framing: "UI/score
displays", no evidence history, no game-specific diagnostics), template
proposer (identity/translate/blocked/click/move-onto/event-at + UI-region
state tables), exact verification, budgeted forward search to next level,
SINGLE-PLAY agent (is_done returns True at WIN — full stop), random +
action-sweep baselines, plain-words scoring description.

WITHHELD (and why): everything multi-play — two_phase, on_play_start, play
minting via RESET-after-WIN, max-over-plays scoring (the single largest
strategic asset); the play-semantics analysis; conflict ledger /
non-Markov framing AND the conflict-detection code path itself (the
vendored store treats a re-observed key as a plain dup — silently weaker,
boundary-safe); determinism probes (the sweep baseline's docstring
originally said "determinism probe" — caught by the audit, reworded);
internal RHAE shorthand; eviction's evidence-history rationale; everything
about the LLM-proposer track, bench/, the bake-off, and NOTES.md content.

JUDGMENT CALL flagged to user as a veto point in STEPS.md: status-region
detection ships (the agent is hollow without it) in neutral language. Veto
before submission if too revealing.

AUDIT: kaggle_m1/audit.py — 28 forbidden patterns over the public files;
must be CLEAN before anything goes public; currently CLEAN. It caught one
real leak on first run ("determinism probe" docstring).

### Status

Notebook compiles, end-to-end tested locally (all 25 public games, offline
mode, short budgets). WRITEUP.md is modest/factual. STEPS.md hands the
user: M1-instructions reading (surface ambiguity, don't guess), notebook
creation settings, the first-contact OBSERVATION CHECKLIST (eval
invocation pattern, per-game limits, submission artifact mechanics,
wall-clock limit, divergences from the local runtime — these directly
test the play-semantics caveat), submit + make public. Pending user:
submission, score, observations → to be logged here.

## 2026-06-10 — Full 25-game sweep: triage + the two censuses (paper tables)

Template proposer, two_phase, R1 on, planner fix on, 240s/game, eviction-
bounded stores, compact-between-games. All 25 public games, ~2.3h wall.
Full ledger + coverage curves + honest triples: results/sweep25/
(triage.json, per-game PNGs, events.jsonl.gz).

### Headline

**Mean RHAE over the full public set: 0.253% — zero LLM calls.** Reference
points, with the caveat that they're not the same eval set: ARC Prize's
May-2026 frontier analysis measured GPT-5.5 at 0.43% and Opus 4.7 at 0.18%
on semi-private. A pure template agent sits inside the frontier-LLM band.
**No game was WON** — the win-gate from Workstream A remains the binding
constraint everywhere, now measured at full breadth.

### Triage (rules in scripts/triage_sweep.py; full table in triage.json)

| bucket | n | games |
|---|---|---|
| SCORED | 0 | — |
| WINNABLE-MAYBE | 2 | r11l (4.76 RHAE, L1), sp80 (0.01, L1) |
| WALLED-R3-trans (conflict sig ≥0.2%) | 12 | bp35 cd82 dc22 g50t ka59 lf52 m0r0 sc25 sk48 tr87 vc33 wa30 |
| WALLED-R3-win (progress-gated, ~0 conflicts, traction) | 8 | ar25 cn04 ft09 lp85 re86 sb26 su15 tn36 |
| DEAD | 3 | ls20 s5i5 tu93 |

**3× time-budget test (720s) on the WINNABLE-MAYBE pair: time converts
NOTHING.** r11l reproduced its run almost exactly (L1, 4.76, bailed);
sp80 ground 85k actions for the same L1 at 0.01. Both are capability-
walled at level 2, not budget-starved. WINNABLE-MAYBE is effectively
empty — on the current template family, NO public game is within reach of
more wall-clock.

### Census 1 — non-Markov fraction (hidden-110 predictor)

**12/25 (48%) of the public set shows conflict signatures** (same
level+frame+action → different outcome at ≥0.2% of keys). Extremes: g50t
at ~46% of additions conflicting (77,365 conflicts on 7,776 stored keys) —
a "live world" whose state evolves between actions (timers/animation as
hidden state), a distinct R3 subtype worth naming. Four conflict-games
still made level progress (cd82, vc33 2 levels; lf52, sk48 1) — latent
state and progress coexist. If the hidden 110 matches this distribution,
roughly half the evaluation requires a stateful world model — the paper's
strongest single argument for R3.

### Census 2 — evidence starvation (D's explorer requirement, quantified)

**19/25 (76%) produced ZERO level-advance or win observations** in 240s of
exploration. The smoke test's sb26 finding generalizes to three-quarters
of the public set: a future LLM proposer, however good, has nothing to
induce win conditions from on these games. Evidence-seeking exploration is
not an optimization — it is the precondition for the entire
propose-verify loop on most of the benchmark.

### Surprises (fit no bucket cleanly; design input)

- **INERT-START class — ft09, lp85**: only 24 unique transitions in ~47k
  actions; every tried action was a frame-no-op from the start state, so
  the winseeker exhausted its 24 salient-click candidates and looped
  "safe" no-ops for the rest of the budget (the 100%-predicted triples are
  the identity rule trivially confirming no-ops). Exploration failure, not
  modeling failure: the cap-24 salience generator never finds the active
  control. D explorer requirement: adaptive click coverage when the
  candidate set is exhausted and inert.
- s5i5/tu93/ls20 (DEAD) still racked 74-175k actions — high-throughput
  thrash with no traction; cheap to detect early (predicted-fraction near
  zero after N actions) → budget-reallocation hook for the fleet runner.
- Eviction rail: zero capped_drops anywhere; stores stayed well under cap
  (max ~62k stored) — the bound exists for safety, not as an active
  constraint at 240s budgets.

## 2026-06-10 — Part 1: r11l + sp80 hand diagnosis (no crack; blockers named)

### r11l (best target: 4.76 RHAE, L1 efficient, L2 stalls)

Source reading (public env file): click-only piece-routing puzzle — clicking
a piece launches an ANIMATED travel toward a slot; colliding with a
`defgjl` obstacle costs a strike (hidden 0–4 counter, 5 ⇒ lose); all pieces
seated ⇒ level. **Level 1 has no obstacle; level 2 introduces it** plus the
strike counter.

Trajectory reading found a bug BEFORE the game's walls: with zero rules
ever fitting, `_refresh_model(force=not rules)` re-proposed EVERY STEP —
144s of the 240s budget went to proposing and the agent took 114 actions
total. Fixed (bootstrap-force now requires ≥10 new transitions);
**rerun: 114 → 42,191 actions in the same budget (370×)**. Still no L2.

Diff-structure probe (5,228 stored click transitions): every click also
places/moves a small CURSOR overlay — 33% of changed-click diffs are the
1-cell cursor alone; 65% are piece-moves CONTAMINATED by cursor bookkeeping
(lost=25 vs gained=30 cells). Two general blockers, routed:
- **Composite diffs** (rigid move ⊕ overlay artifact): single-rule
  templates can't claim them. → Part 2 (LLM proposer — exactly the
  "expressiveness beyond templates" case). A general `body_move` template
  (rigid MULTI-COLOR translation with stable destinations — the R2 family)
  was implemented anyway with synthetic generalization tests
  (scripts/test_body_move.py: predicts unseen click cells of a known body);
  it will serve cleaner click-to-move games but cannot claim r11l's
  cursor-contaminated diffs.
- **Latent state**: cursor occludes content (restore value not in frame) +
  the L2 strike counter. → Part 4 first integration target.

### sp80 (L1 done at 0.01 RHAE, L2 stalls)

Source: per-rotation ACTION REMAP tables (`mfkgvxzkbj`/`othselxnik`, keyed
by rotation 1/2/3) — the same input moves different directions depending on
the player sprite's orientation. The orientation is FRAME-VISIBLE ⇒ Markov;
the missing capability is a **body-signature-conditioned translate family**
(translate rules conditioned on the moving sprite's appearance — the same
conditioning mechanism body_move already uses). Routed: next template
iteration, and flagged as the FIRST LLM-proposer test case — sp80 is in the
Markov + evidence-rich intersection (census L:1 W:0 G:40, conflicts 0).

### Part 1 acceptance: outcome class 2 (named blockers, routed)

No crack (no WIN, no RHAE>50). Collateral wins: the propose-force
throughput bug (would have silently strangled EVERY zero-rule game in the
hidden set), and the general body_move template. r11l → Part 2 + Part 4;
sp80 → template iteration / Part 2 test case.

## 2026-06-10 — V1 + V2 verifications (gate everything; both resolved)

### V1 — AERA (arXiv 2605.25931) units: **FRACTIONS. The bar is ~30%.**

Verbatim from the paper: "achieving RHAE=0.2116 (4/25 solved) on these 25
games"; "The linked code track entry achieves RHAE=0.30 on the full
55-game private evaluation"; "random and no-explore baselines score
0.0000". So **21.16% public / 30% private** — not 0.21%/0.30%. Our
template-only 0.253% is ~84× below the public-set state of the art.
Recalibration: the near-term target is no longer "beat the frontier-LLM
band (0.43%)" but "approach AERA's 21% public — and the mechanisms that
get them there are, by their own admission, largely non-intelligent."

The mechanism read (full text):
- Architecture: EXPLORE (entropy-reduction over a belief state) / VERIFY
  (1–3-step falsification) / PLAN (MAP hypothesis) — conceptually adjacent
  to our propose-verify-plan, independently derived.
- **Their own validity critique is the headline**: "all 25 public games
  are reachable through non-intelligent strategies: 10 in single blind
  step, 5 after probing, 1 via repeated ACTION1... 18 via null-coordinate
  vulnerability" (a library-level crash exploit), and "the 25 public games
  cannot discriminate between intelligent exploration and trivial
  heuristics". 8 games "solvable by repeated single action with
  sufficient budget (50–200 steps)".
- Solved set (b=1): **FT09, VC33, LP85, S5I5** — plus R11L, TN36 in some
  of 8 independent runs (seed variance corroborating the multi-seed
  caveat). **These are exactly our INERT-START (ft09, lp85) and DEAD
  (s5i5) classes** — the persistence probe and click-tier escalation in
  the Part 0 addendum attack precisely this set, and our win-gate replay
  (which AERA does not exploit) converts any win to near-100% per game.
- No cross-game learning; per-episode memory only.

AERA validity caveat (now attached to every public-set number we quote):
public-set scores partly measure exploitation of non-discriminative
structure, not intelligence; the private 55 is the genuine test.

### V2 — cross-game learning legality: **AMBIGUOUS**

Searched: docs.arcprize.org/methodology (no mention — verified by direct
fetch), arcprize.org/competitions/2026/arc-agi-3 (no mention; only "No
internet access during evaluation" and open-sourcing requirements),
preview-era materials ("Each level will be scored in isolation" — scoring
language, not a learning restriction). No rule text forbids OR allows
retaining learned state across games at evaluation; mechanically the eval
(one notebook process, sequential games, one scorecard) permits in-process
carry-over. Verdict: AMBIGUOUS → per directive, the novelty claim
re-scopes to the conservative formulation:

> **Re-scoped novelty sentence:** executable-world-model agency with a
> rule/skill library accumulated across the PUBLIC games during
> development and FROZEN at submission — per-game learning only at
> evaluation time — under the offline open-weight compute envelope.

This changes paper claims and Part 4+ wording, not the near-term build.

### Priority reassessment (per the V1=fraction branch)

1. **Part 0 addendum jumps to the front**: persistence probe (~hours) +
   tier-escalating click coverage target AERA's solved set, which overlaps
   our INERT-START/DEAD games 1:1; win-gate replay then multiplies each
   win toward 100%. This is the highest score-per-day work available.
2. Harness audit (null-coordinate vulnerability) is mandatory before any
   census is quoted publicly — it affects BOTH our censuses' validity and
   raises a legality question we record both answers to.
3. Multi-seed (≥3) census rerun scheduled before the paper freezes
   76%/48%/0.253% as facts (AERA's own 8-run variance corroborates).
4. Part 2 (LLM proposer) continues as specced with the addendum's
   literature-corrected fixes; plumbing is already proven (0 format
   errors live).

## 2026-06-10 — DATED AMENDMENT to V1: two-track reconciliation + the
## reproduction test (per the memo's amend-in-place protocol)

### Empirical settlement (local, minutes, decisive)

Ran the repeated-single-action strategy in OUR harness on AERA's solved set
(FT09, VC33, LP85, S5I5): all four are ACTION6-only games (no basic action
even available — already a strike against "repeated ACTION1" applying
here); repeated clicks at 5 fixed coordinates × 1,500 actions each:
**zero level completions anywhere** (vc33/s5i5 death-cycle ~1 GAME_OVER per
50 clicks; ft09/lp85 inert). **The claim does not reproduce in the current
public runtime (arc_agi 0.9.8).**

Null-coordinate audit (the 18-games exploit): ACTION6 with NO coordinates
executes as a counted no-op in our wheel — no crash, no error, no progress.
**Patched or version-specific.** Audit answers: (validity) our censuses are
NOT contaminated by the exploit — it isn't exploitable here; (legality)
moot locally; if it exists in the hosted runtime, using it would be a
disqualifiable exploit, not a strategy.

### Leaderboard reconciliation (live Kaggle API, primary source)

Current Kaggle offline leaderboard (fetched via the API today): **Tufa Labs
1.21**, then 0.68, 0.66, 0.66, 0.65… Scores above 1.0 exist ⇒ the scale is
PERCENT. Therefore AERA's "0.30 on the full 55-game private evaluation"
**cannot be the Kaggle offline track** — it lives on the hosted API
evaluation (online, no compute envelope, runtime version uncertain — and
the reproduction test above suggests version-dependent mechanics).

### Corrected calibration: TWO bars, one per track

| track | bar | who | our position |
|---|---|---|---|
| Kaggle offline (the one we compete on) | **1.21%** | Tufa Labs | 0.253% template-only ≈ 4.8× below the LEADER |
| hosted API / paper context | 21.16% public / 30% private-55 | AERA | not directly comparable: online compute, mechanisms that don't reproduce on the current offline runtime |

Supersedes this morning's "the bar is ~30%" and the original "frontier
0.43%" framings. Implications: (1) the Kaggle gap is one cracked game away
— a single full WIN+replay on a 25-game-style set ≈ +4% mean, far above
the current leader; (2) AERA's public number inherits a reproducibility
caveat of its own — the paper cites both ways (their validity critique of
the public set; our non-reproduction of their mechanisms on the current
runtime); (3) the Part-0 persistence probe is demoted from "free public
score" to "cheap hidden-set insurance", and its acceptance wording becomes
"report which of our 25 it cracks" (empirically today: none of the four
AERA-solved games fall to pure repetition in this runtime).

### Next (tomorrow+)

1. World-model loop prototype: propose transition rules as Python from
   observed (frame, action, frame′) triples, verify by exact replay
   (determinism holds), plan through the verified model.
2. Kaggle eval-notebook reconnaissance: does the evaluated agent get to
   start multiple plays per game (re-make / RESET-at-zero)? Best-play-of-N
   scoring makes explore-then-execute free if yes. Check the competition
   Data/Code tab template.
3. Re-run the determinism probe per game (all 25, longer horizons) before
   the world model assumes it globally.
4. Benchmark candidate open-weight coder models for world-model code quality
   (the load-bearing engineering unknown from phase1-v2 §7).

## 2026-06-10 — Overnight cap study, Phase 1: the 5× per-level cutoff modeled

Driving fact (verified earlier today, research/competitors/
forge-analysis_2026-06.md): the tech report (2603.24621, Leaderboards)
specifies a hard eval-time cutoff of 5× human baseline actions PER LEVEL —
absent from the shipped scoring code, so every census/budget number we have
was measured in a regime the evaluation may not permit. Gateway enforcement
UNCONFIRMED (Kaggle-Starter probe = daytime task).

`RunConfig.per_level_action_cap_multiplier` (None = off, 5.0 = eval policy)
implemented in `harness/runner.py::level_action_cap` — one isolated function
carrying the PROVISIONAL interpretation, all five assumptions documented in
its docstring for the gateway probe to confirm or refute:

1. counter = SCORED actions attributed to the current level, cumulative
   across GAME_OVER level-resets within a play (mirrors scorecard
   attribution);
2. completing ON the cap-th action is allowed ("cut off AFTER 5×");
3. cutoff ends the game run (levels are sequential; capped = stuck) — in
   two_phase it ends the play, banked plays keep their max-over-plays score;
4. replay plays get fresh per-level counters (per-play isolation);
5. levels with no usable baseline are uncapped.

Tests (scripts/test_level_cap.py, 21/21): fires at exactly 5× on tt01
(15th counted no-op), completing on the 15th action allowed, counter resets
per level, cap=None bit-for-bit identical to unreachable-cap AND
deterministic across runs (counting doesn't perturb), and the two_phase
interaction: a capped replay ends the play while the banked sloppy win's
25.0 survives as the game score. `run_wm.py --per-level-cap 5.0` wires it
into sweeps.

## 2026-06-10 — Overnight cap study, Phases 2–4: results (full report:
results/cap_study/REPORT.md)

Headline: **the 5× cap costs ~nothing in RHAE (0.219 capped vs
0.219–0.253 uncapped) but blinds the agent** — eval-realistic evidence
starvation is **88%** (was 76% uncapped), and the non-Markov census
collapses from 51% to **0% detectable** within the capped window (the
latent state is still there; conflicts just can't be observed twice in
≤5× baseline actions). Capped sweeps run in ~4 minutes (vs hours), so
under cap-style eval nearly the whole notebook budget becomes LLM time —
while per-game evidence shrinks to 34–465 actions.

Corrections to numbers previously logged here as fact (details REPORT §5):
76% starvation → 88% eval-realistic; 0.253% mean RHAE → 0.219% on the
current agent; the 48% non-Markov census is real but UNOBSERVABLE under
eval conditions; the 110-games-in-9h projection's harness share collapses
to minutes; triage buckets need a "detectable in-window?" column.

Two-phase verdict: structurally survives (capped replays can't poison
banked plays — unit-tested), practically idle (zero wins in any arm). The
honest restatement: **win frugally, then replay cleanly** — sloppiness
allowance per level drops from ~26k–110k observed actions to 30–2,890.
New R3 evidence route that fits the envelope: cross-PLAY conflict
detection (replays revisit early states; conflicts between plays are
observable without grinding within one).

Seed variance: near-zero (capped seeds differ by 1 action; uncapped
censuses identical across seeds) — agent-version drift dominates seeds.

Bonus: AERA (2605.25931) Table 9 REFUTED by direct probe — all 8 of their
"solvable by one repeated action" games hit GAME_OVER at exactly the step
count they quote as sufficient; their RHAE figures are 0–1 fractions
(0.2116 = 21%), now suspect wholesale. Sweep doc §8 has the correction.
DEAD bucket reconfirmed; persistence probe demoted.

Cap semantics remain PROVISIONAL (5 assumptions in
harness/runner.py::level_action_cap). Gateway probe priority list:
REPORT §7 — top three: is the cutoff enforced; what does it end
(level/game/run, per-play or cumulative); does RESET-after-WIN still mint
a play.

## 2026-06-10 — Gateway probe: output channels determined + probe built
(full design: results/cap_study/gateway_probe_design.md)

Task A done. Decisive inference: the eval gateway is a SEPARATE container
(gateway:8001, up only when KAGGLE_IS_COMPETITION_RERUN=1) and our pip
arc_agi-0.9.8 has NO cutoff anywhere — so if the gateway enforces the 5×
cutoff it runs different/newer code, and Q1/Q2 can only be settled by
probing the live gateway (a submission; the online dev API is same-vendor
proxy evidence at best).

Channels: (1) rerun logs visible — UNKNOWN, lean NO for a hidden-test
gateway; cheaply settled by a sentinel submission. (2) interactive gateway
— almost certainly NO (forge gates the curl behind the rerun env var;
exposing it interactively would leak the hidden set). (3) score-encoding —
always works but only writable if the agent wins ≥1 level (RHAE scores only
completed levels); ride it on whatever games we win, modulate action count
into score bands. (4) online three.arcprize.org API — proxy, needs an ARC
key we don't have. (5) local mirror — answers Q3 mechanics, nothing on
Q1/Q2 enforcement.

Key reframing: the agent can DETECT Q1/Q2/Q3/Q5 from the frame/score stream
in one run; the whole problem is getting bits OUT — hence channels first.

Built kaggle_probe/probe_agent.py (merged Probe 0 sentinel + Probe 1 full
diagnostic: cutoff trace that RESETs past normal deaths to accumulate
level-0 actions toward 5×baseline, RESET-after-WIN mint test, null-coord
test) + scripts/smoke_probe.py. Smoke-verified on the local engine (no-cutoff
signature clean, all branches fire). Q4 (wall-clock): 6h, now corroborated
by two independent overview mirrors (supersedes FORGE's 8h guess / our 9h
assumption). Submitting = human action; recommended order in the design doc
§"Recommended order".

## 2026-06-10 — Exploration layer upgrade for eval-realistic budgets (no LLM)

Binding constraint shifted by the cap study: not total actions but INFORMATIVE
actions inside 30–2890 per-level windows. Rebuilt the exploration layer
(harness/wm/explore.py new; winseeker.py v3; wm_agent.py wiring). Every
mechanism cites its motivating paper/census in-code. Seven items, all landed
and tested (scripts/test_explore.py 15/15; full WM regression suite green;
m1 audit still clean):

1. **Tier-escalating click coverage** (Rudakov 2512.24156) — replaced the
   cap-24 salience generator with uncapped salience-stratified tiers over
   segmented components + a coarse-lattice FLOOR; first-untried-in-tier-order.
   **Acceptance, honest:** ft09 24→**2649** unique transitions, lp85 24→**3490**
   (uncapped, 60s) — both now frame_change=True (the old generator never found
   the live control). The literal ">500 within a 5×-capped run" is UNREACHABLE
   on this pair because their level-0 caps are 215/85 (cap0=5×{43,17}); capped
   they hit 214/84 unique = essentially one NEW transition per action (~100%
   novel rate) then verdict window_exhausted_capped. So: the mechanism is
   validated by the rate (≈100% vs the old plateau at 24); the >500 figure only
   exists uncapped. Flagged rather than forced.
2. **Go-Explore archive + return-via-replay** (1901.10995/2004.12919) —
   archive of interesting frames (novel segment-set / near-event / meter
   extreme), keyed by segment-set hash, returned to by exact replay of the
   recorded prefix. Cap-aware policy: prefix length is the PRIMARY rank under
   the cap (return costs prefix-length actions) and a tiebreak uncapped; the
   leading RESET returns to level start and is not stored. Wired + unit-tested
   (return path drives cleanly on live frames). Note: rarely TRIGGERS in short
   runs because tier coverage keeps novelty high — it fires only on genuine
   frontier exhaustion, which is correct.
3. **Segment-granularity visit counts** (#Exploration 1611.04717) —
   translation-invariant component signatures; state novelty = count of rare
   (<=1-visit) segments. Wired into WinSeeker's frontier ranking and the
   archive's interesting-criterion, so HUD tickers don't drown event-bearing
   sub-changes.
4. **Evidence-seeking mode** (the 88% census) — segment-novelty-first ranking
   of known changers, then frontier size, then bidirectional meter movement
   (a meter draining to GAME_OVER is evidence), then death-path penalty.
5. **Persistence probe** (AERA 2605.25931, demoted) — once/game, pre-WIN, each
   basic action repeated up to min(200, cap-remaining), abort on first event.
   Local result as expected: cracks nothing, but yields first-evidence
   (GAME_OVER) on games like sp80/ar25. Kept as cheap insurance; --no-probe
   disables it.
6. **APEX quota** (2605.21240 pitfall) — min apex_quota=12 exploration steps
   per level until first WIN before committing to plan-following; logged
   (apex_forced).
7. **Per-game exploration ledger** — added window_exhausted_capped to the
   failure taxonomy (no_live_controls / loop_detected / frontier_exhausted /
   window_exhausted_capped / time); report() now also carries tier_reached,
   distinct_segments, archive_cells, probe_first_evidence, and the explore_stats.

Cap-awareness reaches the agent via run_wm passing per_level_cap_mult +
per-game baselines (mirrors runner.level_action_cap; runner stays
authoritative). No LLM anywhere in this layer.

## 2026-06-10 — Explorer run matrix (25 games × {uncapped 240s, capped 5×}), upgraded explorer

Both branches, current template agent, seed 0. tt01 canary run separately.
Artifacts: results/explore-uncap/, results/explore-cap5/ (+ -tt01), 
results/explore_matrix/report.json. Baseline for deltas = the cap-study
CURRENT-agent uncapped run (uncap-s1, mean 0.219), NOT sweep25 (0.253, older
agent).

### Headline, stated straight: a real tradeoff, NOT a clean win
The upgrade does what it was designed to (evidence breadth) but it
**regresses public-set RHAE**, and the LEVEL/WIN-starvation number did not
drop. Both verified, not assumed.

### Census deltas — two definitions, because they say opposite things
- **LEVEL/WIN-starvation** (the original "76%/88%" metric): baseline 76%
  (19/25) → uncapped **80%** (20/25) → capped **92%** (23/25). It did NOT
  drop. Reaching a level on these games is capability-bound (the template
  proposer can't model them to PLAN a win) and luck-of-exploration-order;
  breadth-first coverage REDISTRIBUTES which games level (gained lp85, tn36;
  lost cd82, lf52, sk48 vs the old explorer) without growing the count.
  Capped, it's structurally floored — you cannot advance a level in a
  30–290-action window on a walled game.
- **Zero-event starvation** (no LEVEL/WIN/GAME_OVER at all — the
  proposer-relevant metric, since a GAME_OVER teaches the win condition's
  complement): uncapped **8%→0%** (every one of the 25 now yields ≥1 event;
  the ft09/lp85 INERT pathology is gone), capped **28%** (7/25 zero-event;
  18/25 get in-window GAME_OVER evidence, 13 of them previously
  LEVEL/WIN-starved). This is the material drop — but on the evidence
  metric, not the LEVEL/WIN one the task named.

### INERT-START acceptance (item 1): MET uncapped, ceiling-bound capped
ft09 24→**13,330** unique transitions, lp85 24→**14,089** (uncapped 240s),
both now frame_change=True — the cap-24 generator's blindness is fixed.
Capped they hit 214/84 (= their 5×{43,17} windows) ≈ one new transition per
action. The literal ">500 within a 5×-capped run" stays unreachable for this
pair because their windows are 215/85; reported honestly.

### RHAE: REGRESSION, attributed precisely (does not meet the no-regression bar)
Uncapped mean **0.219 → 0.127**; capped **0.219(cap-study)→0.125**. The
entire uncapped drop is a handful of marginal first-level wins:
- r11l 4.76→**2.94** (L1 reached with ~128 vs ~101 level-1 actions — same
  win, lower efficiency): −0.073 of the −0.092.
- lost lf52 0.52, lp85 0.18 (old explorer's lucky L1 clicks not hit);
  gained tn36 0.19, vc33 0.01→(2 levels), sp80 0.03.
Probe/APEX are NOT the cause (lean --no-probe --apex-quota 0 reruns give
nearly identical RHAE: sp80 0.03→0.05, r11l unchanged). The cause is the
breadth-first explorer reaching first-levels with different/more level-1
actions — luck-of-coordinate-order on click games with ~100 candidates.
**Root reason the win-gate doesn't save us:** NO public game is fully WON,
so the replay machinery never engages; 100% of public-set RHAE is raw
first-level exploration efficiency, which breadth-first coverage trades away
for evidence. The earlier assumption "win-gate makes exploration debt free
uncapped" holds only AFTER a full win — false on the public set. Chasing the
old per-game efficiency would mean overfitting public click-coordinates that
won't transfer to the hidden 55 — declined on principle.

### tt01 canary: unchanged — 100.0, WIN_REPLAYED, both branches.

### Explorer mechanism usage
- Uncapped tier_reached: lattice 17 / simple 6 / refine 2 — on 17 games the
  explorer exhausted every segment and descended to the coarse-lattice floor
  (full coverage). Go-Explore returns fired **1,529** times (frontier
  genuinely exhausts uncapped); max archive 2,628 cells.
- Capped tier_reached: none 9 / simple 6 / seg_med 5 / seg_small 3 /
  seg_big 2 — tiny windows rarely descend past simple/small tiers; Go-Explore
  fired 0 (no frontier exhaustion in 30–290 actions — correct). Probe yielded
  first-evidence on 10 games.

### Triage movement (uncapped, vs sweep25 buckets)
- lp85: WALLED-R3-win → now reaches L1 (exploration was the blocker; now isn't).
- tn36: WALLED-R3-win → now reaches L1.
- vc33: WALLED-R3-trans → now 2 levels (was 1).
- Regressions vs old explorer: cd82 (2→0 capped-window? uncapped 2→0 levels),
  lf52 (1→0), sk48 (1→0) lost their level — same luck-of-order. Net level
  count roughly flat; the SET moved.

### Exploration is NO LONGER the blocker (Part 1/2 targets — do NOT start here)
Games that now reach ≥1 level uncapped, i.e. the proposer/planner (not the
explorer) is the bottleneck:
- **r11l** — 1 level, RHAE 2.94, best public scorer; the clearest Part 1/2
  target (needs a model to WIN the game and trigger replay, not more
  exploration).
- vc33 — 2 levels, 0.01. sp80 — 1 level, 0.03. tn36 — 1 level, 0.19.
- lp85 — 1 level, 0.0.
All ABANDONED (no full WIN), so all are replay-gated on modeling, not
exploration.

### Seed note
Agent is deterministic up to the rare rng "desperate" fallback (only fires
when no candidate is selectable). It was touched **6 times total across all
25 uncapped games, 0 times capped** — negligible; single-seed numbers are
effectively exact. Multi-seed not run (the rng branch is too rarely hit to
move any census; flagged if a future change increases desperate-rate).

## 2026-06-10 — r11l + sp80 diagnosis (GATE): precise blocker routed, not a crack

Full evidence: results/diagnosis_r11l_sp80.md. Diagnosed both Workstream-B
style (read source + trajectory). Both are the SAME class and route to the
SAME place; neither cracks now.

**r11l** is a click-select-then-click-place puzzle (L1: 1 piece/1 target/0
obstacles = 2-click, WinSeeker stumbles it → 4.76; L2: 2 pieces/2 targets/1
obstacle = sequenced select-place×2 with pairing). **sp80** is the same
family with arrow-move-selected + a multi-step animated "spill" phase.

Trajectories (upgraded explorer, uncapped): r11l 5,292 transitions / **0
conflicts** / hud_regions=[] / coverage **0.0**; sp80 10,056 / 0 / [] /
**0.0**. Both REACH L2 and gather thousands of transitions, then die out
(261/290 GAME_OVERs). **Exploration is NOT the blocker** (Part 0 confirmed:
upgraded explorer still 0 coverage at L2). Modeling is.

The wall is a compound, gated in order:
1. **R1′ (NEW HUD subtype):** a step-counter UI renders into the frame and
   changes EVERY action; combined with selection/piece changes, ~92% of
   r11l-L2 frames are unique → state explosion → no recurring context → 0
   coverage. RegionAnalyzer misses it because the counter never SOLOS (its
   seed is the sole-changer signal; cd82's meter soloed on no-ops, this one
   doesn't). 0 conflicts because latent state is fully RENDERED (observable)
   — explosion, not aliasing (unlike cd82). General fix: high-frequency /
   monotone-countdown region detection, but EXOGENOUS-AWARE — must not mask
   the interactive click board (both are click games; the sole-changer guard
   exists precisely to protect click boards).
2. **R2:** click-/selection-parameterized rigid-body motion (general
   template gap, entangled with selection).
3. **R3:** latent selection + phase + animation timers (Part 4).

**Routing (protocol option c):** r11l = **Part 4's first integration
target** (cleanest R3: rendered latent state, 0 conflicts, fully
observable); sp80 = second (adds spill-animation phase, harder). General
prerequisites Part 4 needs: R1′ HUD masking (exogenous-aware) + R2
selection-parameterized rigid body. The project's first real win runs
through Part 4 + these prerequisites, not through exploration or one
template. cd82-class R1′ benefit is a bonus.

## 2026-06-10 — Part 2: minimal LLM proposer integration (frame-only), GATE

The bake-off machinery is now wired into the live agent (it already was, via
wm_agent._llm_step; this part validated it end-to-end on real game stores and
fixed a live-scale bug). Quality vehicle: scripts/llm_quality.py — loads a
captured store, runs the LIVE LLMProposer.propose() + gated repair N rounds,
verifies by exact replay, checks beyond-template coverage, emits Phase-D
economics + verbatim corpus. Backend-agnostic (--llm-url/model/backend);
ready to point at a rental serving the Phase-C pick.

### Target class (from triage.json ∩ Part-0 explore-uncap census)
"Markov+evidence" = 0 conflict signature AND event evidence. **Important
correction:** "0 conflicts ⇒ frame-only suffices" is TOO LOOSE. Of 8
candidates, only **su15, sb26, ar25** are genuinely frame-modelable
(template proposer verifies 129 / 79 / 1 transitions there). The other 5
(ft09, re86, cn04, tn36, lp85) have **templates verifying 0** — their latent
state is RENDERED (UI tickers), so 0 conflicts but state-exploded and
unmaskable (the r11l class; hud detection found 0 regions on 7/8). Those were
never valid frame-only targets. The real frame-only set is small.

### Results (local 14B = qwen2.5-coder:14b, plumbing/dev; 8 games × 4 rounds)
- **TRUNCATION FIX: decisively met. 1/157 format errors (0.6%) vs the 17/24
  (71%) bake-off baseline.** Code-first scaffold + one mechanical retry; the
  single failure was one tn36 gen that still missed after retry.
- **GATED REPAIR working: 5 accepted / 56 rejected.** The 14B is
  repair-negative (bake-off), and the gate correctly refuses degradations;
  no accepted repair reached VERIFIED either.
- **VERIFIED rules: 0 across all 8 games → quality acceptance NOT met on the
  14B.** On the genuinely-modelable games (su15/sb26/ar25, where templates
  verify), the 14B produced 0 verified rules — **pure model quality**, not a
  harness wall. Failure mode (sb26, verbatim): coherent but FABRICATED
  mechanics ("ACTION5 → GAME_OVER recoloring cell (53,0)"; "click → recolor a
  vertical-line-of-4 to 14") that exact-replay correctly rejects. Exactly the
  bake-off's "plausible-but-invented" pattern; the 14B guesses game rules
  rather than fitting evidence.

### Bug fixed (exposed only at live scale)
Gated-repair's count() iterated the FULL store (10k+ transitions) under the
per-predict SIGALRM kill switch; over thousands of back-to-back predicts the
alarm raced during timer teardown and escaped (_PredictTimeout crash). Fixed:
sample-bound count()/coverage to 800 (event-priority), disarm the timer
inside the try ASAP, and swallow a late alarm in finally. The bench harness
never hit this (bench tasks are tiny). wm_core + tt01 regressions green.

### GATE outcome
Plumbing ✅, truncation fix ✅ (0.6% vs 71%), gated repair ✅, live-scale bug
✅ fixed. **Quality gate (≥1 beyond-template VERIFIED rule on 3 games): NOT
met on the 14B — routed to the Phase-C pick (Next, else GLM) on a ~$3
rental**, exactly the task's stated fallback (the verbatim corpus is empty on
the 14B; harness is rental-ready, just repoint --llm-url/model). Also routed:
the frame-only target set is su15/sb26/ar25 (not the looser 0-conflict list);
ft09/re86/cn04/tn36/lp85 need the R1′ HUD fix before any frame rule can
verify. Corpus + economics: results/llm_quality/report.json (+ gens.tar.gz
verbatim).

## 2026-06-10 — Overnight task 1: R1′ exogenous-aware HUD masking (change-content predictability)

Implements the R1′ prerequisite named by the r11l/sp80 diagnosis: rendered
step counters that change on EVERY action but NEVER solo (clicks also
move/select pieces) defeat the sole-changer seed → ~every frame unique →
state explosion → 0 template coverage. cd82-style soloing tickers were
caught; r11l-style rendered counters were not.

### Detector design (harness/wm/regions.py)
- New R1′ stage in `RegionAnalyzer.analyze()`, after (and disjoint from) the
  existing TIER-1/TIER-2 sole-changer stage. A cell is HUD-like when the
  VALUE it changes to is a near-deterministic function of actions-since-
  level-start: per changed cell we record (action_idx, post_value); over idx
  bins observed ≥2 times, the majority-value share must reach `pred_rate`
  (0.9). Singleton bins are excluded — they are deterministic vacuously, the
  exact "looks low-entropy in a short window" trap — with evidence floors
  `pred_min_changes=8` total and `pred_min_repeat_obs=6` repeated-bin obs.
- The action index resets on LEVEL/WIN/GAME_OVER (and on a level change seen
  without an event). `observe(t, idx=...)` accepts the true index from the
  caller; the agent passes `len(self._level_prefix)-1` (its replay prefix),
  because the analyzer only sees status=="new" transitions and an internal
  counter would desync on dedup-skips. Internal counter remains the fallback
  for idx-less callers (llm_quality.py, old tests).
- Candidates then pass the SAME guards as the base stage: spatial clustering,
  cluster `always_rate` (0.55) activity test, `max_frac` size guard,
  unmaskable-colors guard, `min_transitions` floor.

### Click-board guard (the diagnosis' flagged danger)
A cell whose changes concentrate near click points — chebyshev ≤
`click_radius` (2) of the click in ≥ `click_dep_rate` (0.2) of its
click-transition changes — is interactive board and is EXEMPT from R1′
masking no matter how idx-predictable its values look. A real HUD counter
changes wherever the click lands, so its near-click fraction sits at the
area-ratio noise floor (~25/4096 on 64×64). Cells whose post-value varies
with WHICH cell was clicked already fail the predictability test itself.
Source-cells of moved pieces (change when the click is far away) are
protected by value-entropy, not the guard.

### Flag plumbing (A/B-ready)
- `RegionAnalyzer(r1prime=True)` default ON; `r1prime=False` skips all R1′
  recording and the analyze stage → bit-identical to the pre-R1′ detector.
- `WorldModelAgent(r1prime=...)` → analyzer + `report()["r1prime"]`.
- `scripts/run_wm.py --r1prime {on,off}` (default on), recorded in run_meta.

### Tests (scripts/test_wm_core.py)
- Scenario R1′-counter: dual HUD cells (countdown + spinner), BOTH change
  every action as exact functions of idx, reset per level — never a solo or
  bare pair. Asserts: R1′-off masks nothing and reproduces the r11l
  pathology (no grid template, coverage 0.0); R1′-on masks exactly the
  counter pair, translate/blocked_identity reach VERIFIED, coverage_exact
  1.0, coverage_predicted >0.9.
- Scenario R1′-click-guard: pure-click game; a board cell clicked at fixed
  idxs 1..5 every episode (toggle values perfectly idx-predictable, heavy
  repeat evidence) + the counter pair. Asserts: with the guard disabled
  (click_dep_rate=1.1) the trap fires (board cell masked); with defaults the
  mask is exactly the counters and the board survives; R1′-off masks nothing.
- Results: test_wm_core 5/5 scenarios PASS (0, A, C, R1′-counter,
  R1′-click-guard); test_explore 15/15 PASS; test_body_move PASS;
  test_wm_tt01 PASS (offline). Note: ran with .venv/bin/python (the task
  said venv/bin/python, but that env is bare — no numpy; .venv is the
  project env per the scripts' own docstrings).

Next (separate task): A/B `--r1prime on/off` on the live r11l/sp80 stores —
the prediction is hud_regions≠[] and nonzero template coverage on r11l with
R1′ on. R2 (selection-parameterized rigid-body) and R3 (latent state) remain;
R1′ alone is necessary, not sufficient, per the diagnosis.

## 2026-06-10 — Overnight task 2: R1′ A/B on the five state-exploded games

Ran both arms (template proposer, 240s/game, seed 0, `--save-stores`) on
ft09 lp85 r11l sp80 tn36; tags `overnight-r1prime-off` / `overnight-r1prime-on`.
Full table: results/overnight-20260610/r1prime_coverage.md. Stores for task 3
at runs/wm/overnight-r1prime-on/*-store.pkl. (Used .venv/bin/python; `venv/`
is a bare env — same discrepancy task 1 hit.)

Headline deltas (off → on):

- **sp80 UNBLOCKED**: distinct contexts 4168 → 473 (−89%), steady event
  coverage 0 → 0.096 (exact rate 1.0), first VERIFIED rule. The textbook
  R1′ signature: explosion collapses, coverage rises.
- **r11l improved, not unblocked**: contexts 4577 → 3140 (−31%), HUD
  coverage 0 → 0.064 steady (0.90 peak), VERIFIED rules 0 → 1, matched
  prediction steps 17 → 104. Grid/event template coverage still 0.000 —
  consistent with the Part 1 diagnosis that R1′ is necessary, not
  sufficient, for r11l (R2/R3 still needed).
- **ft09, lp85, tn36 unchanged**: contexts flat (590/242/1748 — ft09 and
  lp85 were never context-exploded), coverage 0 both arms, no verified
  rules. Their blockers are not HUD exogeneity.
- RHAE identical per game in both arms (mean 0.634); at this budget R1′
  moves model quality, not score.

## 2026-06-10 — Overnight task 4: harness/wm/ correctness audit (2 cycles)

Manual audit-fix loop over all 11 modules (the cloud-coder:audit-loop skill's
backing script `devops/cloud-coder/audit-fix-loop.sh` doesn't exist in this
repo, and its design spawns cloud Claude CLI calls — no-spend guardrail — so
the loop ran manually per the task spec). Each fix had a failing test written
first; all offline suites pass after each cycle (.venv/bin/python again —
`venv/` is still the bare env).

### Fixed (cycle 1, commit f4cee1a)

- **verifier.py: claim-less Prediction counted as exact.** A rule returning
  `Prediction()` (grid=None, event=None) matched everything — `grids_match`
  returns True with no grid claim and the event check is skipped — so a
  vacuous rule reached VERIFIED on any store with ≥ min_exact transitions
  (demonstrated: 192 "exact" matches on the toy store). Now treated as
  NO_PREDICTION. Test: scenario_verifier_edges in test_wm_core.py.
- **verifier.py: deadline only checked BETWEEN rules.** One slow rule × a
  large store overshot the budget unboundedly (sibling of the repair count()
  SIGALRM race the task flagged). Now checked every 32 transitions; an
  aborted rule keeps its previous status/counts so a partial pass can never
  upgrade or downgrade. Same test.
- **store.py: load() dropped all eviction/census state.** `_evictable` was
  never rebuilt (a loaded store at max_transitions refused EVERY new
  transition — `capped_drops` forever), `_conflict_keys` protection was lost
  (conflict pairs evictable post-load), and appended_total/event_counts reset
  to zero. save() now persists conflict_keys/appended_total/event_counts;
  load() rebuilds _evictable oldest-first and falls back gracefully on legacy
  pickles (census rebuilt from retained transitions). Test: persistence
  round-trip + legacy-pickle case in test_store_eviction.py.
- **store.py: add() docstring** omitted the ("capped", None) return. Callers
  (wm_agent._observe) already None-check; docstring fixed.

### Fixed (cycle 2, committed with this note)

- **proposers.py: _fits() had the same vacuous-claim bug** — a claim-less
  prediction counted as a proposal-time fit, so a vacuous rule would clear
  MIN_FITS on any sample. Same guard as the verifier; test added to
  scenario_verifier_edges. (No current template emits claim-less predictions;
  the LLM wrapper already guards — this closes the shared-helper hole.)

### Suspected, NOT fixed (so they aren't lost)

- **explore.py:199-214 Archive.consider flag/prefix loss.** A plain
  (non-near-event) observation with a shorter prefix REPLACES an entry,
  silently dropping its near_event/meter_extreme flags; conversely a
  near-event observation replaces a shorter-prefix entry unconditionally,
  contradicting the "keep the SHORTER prefix" comment. Merging (min prefix,
  OR flags, max novelty) looks right but changes tuned exploration ranking —
  left for a supervised change.
- **proposers.py:627-649 DiffMemorizer masked-branch staleness.** The
  unmasked branch lookups the LIVE store; the masked branch builds a frozen
  index at propose() time, so transitions stored after the last propose are
  unreachable until the next one. Never wrong, just stale — but it's a
  behavioral asymmetry between the ablation arms.
- **store.py:168 / regions.py:168** — `pre != post` raises on shape change
  mid-game; fine while frames are fixed 64×64 (ARC-AGI-3 spec) but worth a
  guard if that assumption ever moves.
- **store.py:150-152 dup check ignores post_level**: same (level, pre_hash,
  action_key), same post_hash, same event but different post_level (frames
  alias across levels) silently keeps the first post_level. Marginal.
- **winseeker.py:57 refine_clicks hardcodes the 64×64 bounds** instead of
  taking the grid shape.
- **llm_proposer.py:229-237 _build_evidence** can overshoot its 26-pick cap
  by one per pool (break re-checked only after an append); harmless, the
  char cap bounds the prompt anyway.
- **rules.py:171 WorldModel.predict** never early-breaks when no HUD rules
  exist (the hud_grid term can't become non-None) — perf only.


## 2026-06-10 — Overnight morning report

Full report: results/overnight-20260610/MORNING_REPORT.md. Tasks 1/2/4
landed (R1' built + A/B: sp80 unblocked, r11l improved; 4 wm bugs fixed).
Task 3 (llm_quality on sp80) produced NOTHING despite rc=0 — the run was
backgrounded and orphaned when the session ended; needs a foreground re-run.
First queue pass (23:58) was killed by a usage limit; the real pass ran
07:50-08:42 after the runner fix (0a961ab).
