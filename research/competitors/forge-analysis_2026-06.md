# Kernel recon: jonathanchan/arc26-3-agent-v15 (score 0.46) → FORGE lineage

Pulled 2026-06-10 (`kaggle kernels pull`, sources committed alongside).
Chain: jonathanchan's notebook is a fork of **ashvinsingh's "Ash's
ARC-AGI-3 Agent"** (FORGE); jonathanchan v1/v2 were "almost exact copies"
and are what scored **0.46** on the leaderboard. The fork's own v15
"revision" is broken by construction — ChatGPT-refactored with a literal
`# ... (ALL YOUR ORIGINAL INIT CODE)` placeholder, references to
undefined methods, and no `is_done` implementation (abstract in the
framework's `Agent` ABC, so `MyAgent` cannot even instantiate). The
agent that actually scored is upstream FORGE. Upstream's current head is
**FORGE v21** (extracted to `ash-forge/forge_v19_extracted.py`).

## What FORGE v21 actually is

A training-free Go-Explore-style graph explorer, no torch, no GPU
dependence in the loop:

- **EffectModel** — Laplace-smoothed running tallies of "does this action
  change the frame": per simple-action, per clicked-colour, plus an 8×8
  click-success heatmap; actions/colours that ever advanced a level get a
  +0.6 priority bonus. Explicitly a cheap stand-in for StochasticGoose's
  CNN. **Persists across levels within a game** (not across games) —
  argued as a deliberate edge because level weights grow.
- **Volatility masking** — per-cell change-frequency map; thin border
  bands and border cells changing >50% of frames are masked out of the
  state hash, and the mask is **frozen per level** (they note recomputing
  it mid-level corrupted the graph in v20). This is an independent
  reinvention of our R1 HUD masking, border-band-restricted.
- **LevelGraph** — per-level directed graph of masked-hash states; every
  issued action recorded with outcome (noop/edge/advance/death); deaths
  attributed to the causing action *before* the recovery reset so they're
  never retried ("the loop bug that cost the Helsinki team places").
  Frontier policy: untested-at-current-node first (tier blended with
  effect-model score), else BFS over known edges to nearest frontier,
  else RESET + replay the stored shortest path to the shallowest frontier
  (exploits determinism), else one-time click-density escalation
  (stride 12 → 4), else informed-random.
- **Click proposals** — connected components, one candidate per
  component, snapped to the member cell nearest the centroid (clicks
  never land in concave holes), tiered by size (small objects first),
  coarse background grid as last resort. Same family as Rudakov
  2512.24156's tiers; the density escalation is their answer to our
  INERT-START.
- **Goal-directed navigation** — learns per-action displacement vectors
  by voting on small-object translations (≥2 consistent votes), infers
  avatar colour, then BFS *on the grid* (background = passable +
  visited-cells memory + blocked-move set) toward salient targets (rare
  colours, small, near), retiring targets that fail or kill. Activates
  only on movement games; click/transform games fall through to the
  graph explorer.
- **Calibration phase** — round-robins each available simple action ~3
  times at game start to seed movement vectors and reveal the status bar
  before trusting the planner. Our persistence-probe idea (from AERA) is
  adjacent but distinct: theirs samples each action a few times, not
  ~200 reps.
- `is_done`: WIN, or a **wall-clock guard at 8h−300s** — the author
  believes the notebook limit is ~8h (third datapoint alongside our 9h
  assumption and the mirror's 6h claim; still unverified).

Single-play architecture: stops at first WIN per game. No replay-for-
efficiency phase, no cross-game state, no LLM, no world model. Their
entire bet is frugal first-win efficiency.

## The two load-bearing claims in FORGE's header (verified)

### 1. The 5× cutoff is REAL — and our harness doesn't model it

FORGE's header: "HARD CUTOFF at 5x the human action count per level: if
the agent has not finished a level within 5x the human's actions, it is
cut off and scores 0 for that level."

**Verified against the official technical report (arXiv 2603.24621,
Leaderboards section), verbatim:** "Due to the operational intensity of
running an ARC-AGI-3 full evaluation set using high-reasoning frontier
model APIs (which could run in the tens of thousands of dollars as of
early 2026), we set a hard cutoff of 5x human performance per level. If
a human takes 10 actions to beat a certain level on average, then we
will cut the AI agent off after 50 actions."

Cross-checks: NOT in the shipped scoring code (arc_agi 0.9.8
scorecard.py has only the 115 cap — confirmed by grep, nothing
enforces any cutoff locally); NOT in docs.arcprize.org/methodology; NOT
in the human-baseline blog post. So it is an **evaluation-runtime
policy, not a scoring-library rule** — stated for the official
leaderboard; whether the Kaggle gateway enforces it is unconfirmed but
FORGE's author asserts it for the Kaggle track and the gateway is
ARC-Prize-operated. Assume it applies until disproven.

**Implications if enforced (all major, routed to strategy):**

- Per-level action budget at first contact ≈ **5× baseline = 30–2,890
  actions** (baselines 6–578). Our 76% starvation census explored
  26k–110k actions/game; under a 5× cutoff most of that exploration
  never happens. Evidence-seeking exploration must work inside tiny
  budgets, which raises the value of every cheap transplant in the
  sweep (tier escalation, persistence probe, effect-model ordering) and
  lowers the value of anything sample-hungry.
- "Win sloppily, then replay cleanly" is bounded: exploration debt
  within a level is capped at 5× baseline, so the *sloppy win itself*
  must land within 5× per level. The win-gate analysis (Workstream A)
  needs re-running with a per-level 5× cap to see what survives.
  Open mechanics questions: cutoff per play or cumulative? What state
  does a cutoff produce (GAME_OVER? level reset? play termination)?
- Local harness work item: add an optional 5× per-level cutoff to
  RunConfig so census numbers and agent development reflect the real
  constraint. Without it all our local RHAE/exploration numbers are
  optimistic.
- Test surface: the ARC-AGI-3-Kaggle-Starter + a long deliberate
  overshoot on a known level would reveal whether and how the Kaggle
  gateway enforces it (same session as the WIN→RESET test).

### 2. The engine-introspection exploit is dead AND a DQ risk (per FORGE)

The jonathanchan v15 notebook still carries FORGE **v20's** `BFSSolver`:
it locates the game's Python source (`environment_info.local_dir`, glob
fallbacks over `/kaggle/*` and `/tmp/*`), instantiates the game class
in-process, and BFS-plans by deep-copying full game state — including
`_probe_hidden_fields`: diffing the game object's `__dict__` scalars to
discover hidden state (introspective R3), and level-to-level solution
transfer with object-centroid offset remapping.

FORGE v21 **removed** it, header verbatim: "the official eval is
sandboxed with no internet and the games are hardened against
brute-force; the engine will not be importable, so that path returns
nothing on the real set. It is also against the stated spirit of the
competition (prize-eligible work is screened for it) and is dead weight
and a disqualification risk."

Routed implications: (a) reassuring for our gap — perfect-simulator
shortcuts don't reach the hidden set, so learned world models stay
necessary; (b) the screening claim is another reason our public/private
disclosure discipline matters; (c) their `_probe_hidden_fields` trick is
dev-time-legal on public games — could cheaply label which of our 48%
conflict-census games have which latent-variable type, as ground truth
for R3 development (never shipped).

## Eval-harness mechanics confirmed (cell 2/4 of both notebooks)

This is the recon NOTES.md wanted (open question #2 from 2026-06-09):

- Rerun detection: `KAGGLE_IS_COMPETITION_RERUN=1`.
- The eval runs a **gateway at http://gateway:8001** serving the
  official REST API; agents speak HTTP (our D1 HTTP-parity decision was
  right). Notebooks curl-retry `/api/games` for up to 5–10 min while it
  boots.
- The agent process is the official **ARC-AGI-3-Agents swarm**
  (`python main.py --agent myagent`) — **all games, one process,
  threads**; one agent instance per game but shared module/class state
  is structurally available → cross-game learning is possible at eval
  (consistent with the §7 correction in the sweep doc).
- Agent-side `.env` sets `OPERATION_MODE=online` — competition-mode
  enforcement is server-side at the gateway.
- A dummy `submission.parquet` is written only in the interactive phase
  ("important to keep" — Phase A validation needs an artifact); the
  rerun's real artifact comes from the gateway side.
- FORGE keeps frames bounded (`_MAX_FRAMES=10`) and `MAX_ACTIONS=inf`.

## Competitive read

FORGE (ashvinsingh) is the strongest *public* Kaggle-track baseline
found: legitimate, training-free, frugality-first, and explicitly
designed around the 5× cutoff. It does everything cheap that we do
(masking, salience clicks, graph exploration, determinism replay) minus
the world model, the LLM proposer, and multi-play replay. The 0.46
fork score is near the random/frontier floor (random 0.30, GPT-5.5
0.43), i.e. the public-LB value of even a good explorer is currently
tiny — consistent with our thesis that modeling+efficiency, not
exploration alone, is where score lives. Preview intel embedded in
their header: StochasticGoose won the preview with an action-effect
CNN; "Blind Squirrel" (2nd) was a pruned directed state graph; a
"Helsinki team" lost places to a re-selected death-action loop.
