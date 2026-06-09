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
