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
