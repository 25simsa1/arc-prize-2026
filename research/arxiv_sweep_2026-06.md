# arXiv sweep — June 2026

Systematic literature sweep, run 2026-06-10. Six tracks: scooping watch,
executable/program world models, exploration-for-evidence, LLM code repair
loops, privileged-context gap measurement, competitor profiling. Method:
arXiv export API (bulk abstracts) + web/citation chasing (Semantic Scholar);
full reads only for triage-passing papers. ~60+ abstracts skimmed, ~12
papers read properly. Triage criteria applied to every abstract:
(a) threatens our novelty claim, (b) mechanism implementable inside the
offline envelope, (c) baseline/number to compare against, (d) related-work
citation. Read-only session — implications routed, nothing implemented.

---

## 1. SCOOPING VERDICT: **WATCH** (no fire alarm)

As of 2026-06-10, **no paper does cross-environment skill/program-library
learning on ARC-AGI-3** — under offline compute or otherwise. The gap is
open. But it is narrowing from two directions:

1. **The ARC-AGI-3 agent space is publishing fast** — four new agent papers
   since March 2026 (DreamTeam 2605.09650, MAP 2605.13037, AERA 2605.25931,
   Sensi 2603.17683), all per-game, all API-frontier-LLM-based except AERA
   (Qwen2.5-0.5B). None carries anything across games.
2. **The generic self-evolving-skill-library literature has exploded**
   (40 arXiv hits for "skill library", almost all May–June 2026), including
   executable-program skill libraries in interactive games (Evolving
   Programmatic Skill Networks 2601.03509, MineDojo/Crafter) and evolving
   cross-instance libraries in agentic tasks (EvoLib 2605.14477, MSR —
   cites ArcMemo, no ARC). Someone combining these with ARC-AGI-3 is a
   matter of time.

**Rodionov 2605.05138 revision status: NOT revised past v2** (v1 May 6,
v2 Jun 6 2026). v2 is security/eval-integrity hardening only; results
unchanged (GPT-5.5 15/25 public, mean RHAE 58.12%). Confirmed no collision:
"Each playthrough starts from a fresh agent instance and clean workspace";
reusable routines are explicitly future work. Zero indexed citations yet.

**Watch items (re-check at next sweep, ~monthly):**

- **2605.09650 — DreamTeam "Workspace Optimization" (NVIDIA/Technion).**
  Highest-priority watch. Multi-agent harness on ARC-AGI-3 building an
  executable world model; "workspace as trainable substrate"
  (artifacts↔parameters, counterexamples↔losses). Public-set
  protocol-matched SOTA 36%→38.4%, 31% fewer actions. No collision today —
  workspace rebuilt per game/level, frontier APIs, not offline — but the
  "trainable workspace" rhetoric is one persistence step from our claim.
- **2605.14477 — EvoLib (MSR).** Evolving cross-instance skill library at
  test time, no parameter updates, cites ArcMemo. No ARC of any version.
  Nearest conceptual scoop outside ARC; a port to ARC-AGI-3 lands on our gap.
- **2605.25931 — AERA "Explore Before You Solve".** Two distinct threats:
  (i) *benchmark-validity critique* — all 25 public games reachable by
  non-intelligent strategies (10 in one blind step, 8 via a single repeated
  action at 50–200 steps, plus a null-coordinate exploit bypassing 18) —
  public-set results, ours and Rodionov's alike, may face reviewer
  discounting; pre-empt in the paper. (ii) *Competitive* — RHAE 0.2116
  (4/25) with Qwen2.5-0.5B offline-compatible, and its code-track entry
  claims **RHAE 0.30 on the full 55-game private set**.
- **Tufa Labs' unpublished May-2026 Kaggle jump** (0.68%→1.17%, "novel
  approach", details unreleased). Right compute regime; if it involves
  transfer, it becomes a direct collision. See §6.
- **Framing pressure, not a scoop:** "online skill library" alone is no
  longer novel (SAGE, SkillRL, SkillOS, HiSME, AEL cluster). Our claim must
  stay anchored to the conjunction: *interactive ARC-AGI-3 + executable
  exact-replay-verified world models + cross-environment library + offline
  open-weight compute*. The ARC living survey (2603.13372, 82 approaches)
  catalogues no skill-library-on-ARC-AGI-3 approach — independent
  confirmation the gap is unoccupied.

---

## 2. ROUTED FINDINGS

### → Exploration / WinSeeker (the 76% starvation problem)

**2512.24156 — Graph-Based Exploration for ARC-AGI-3 (Rudakov), Dec 2025.**
Training-free: connected-component segmentation, status-bar masking,
**five-tier salience-stratified click candidates**, directed graph of
state-hashes, and frontier-seeking (move to the nearest state with untested
state-action pairs via shortest path) — Go-Explore-style return-by-retrace.
Median 30/52 levels on 6 preview games, 3rd on the private preview
leaderboard. *Implications:* (a) partial novelty narrowing — cheap
salience-tiered graph exploration on ARC-AGI-3 is published; WinSeeker's
contribution must be positioned as **event-targeted exploration feeding LLM
rule induction**, not exploration per se. (b) **Direct INERT-START fix:
replace the hard 24-candidate salience cap with tier escalation** — exhaust
tier 1, fall through to lower-salience tiers until every segment (or coarse
grid) is probed. Effort ~1–2 days, pure heuristic. (c) "Shortest path to
untested state-action pair" is a drop-in upgrade of unseen-state-first.
(d) Baseline numbers to cite.

**2605.25931 — AERA.** Beyond the watch item: 8/25 public games fall to
*a single repeated action with a 50–200 step budget* — a strategy our
heuristic stack likely never tries, and a candidate explanation for part of
the 76% zero-event census. *Implication:* add a **per-action persistence
probe** (each basic action repeated ~200 steps, once per game) to
WinSeeker — hours of work, ~1200 actions/game, trivially affordable at our
~180 actions/s. Also: check our harness for the null-coordinate
vulnerability (both to exploit legally and to keep census data valid).

**Go-Explore lineage — 1901.10995 / 2004.12919 (Go-Explore, "First return,
then explore"), 2405.15143 (Intelligent Go-Explore).** Our setting
(deterministic, near-free stepping, exact replay) is exactly Go-Explore's
sweet spot. *Implication:* archive of interesting frames + return-via-replay,
optionally letting the already-resident LLM rank archive states (IGE-style,
no training). Effort ~2–4 days. Required foundational citations.

**1611.04717 — #Exploration (hashed-state counts).** Foundational citation
for frame-hash visit counting; suggests counting at **segment granularity**
(hash of segmented components) so event-bearing sub-changes aren't drowned
by cosmetic frame diffs. ≤1 day if adopted.

**Pitfall citations:** 2605.21240 (APEX — exploration collapse in
memory-accumulating agents; don't let plan-first entrench early routines);
2601.00042 (Go-Explore red-team study — seed variance dominates, 8×
outcome spread; multi-seed any census/ablation we report). 2605.16143
("Look Before You Leap" — Exploration Checkpoint Coverage metric; citable
formalization of what our zero-event census measures informally).
2605.16024 (ScreenSearch — affordance discovery as frontier expansion under
dedup, GUI analogue of INERT-START; PUCT-bandit mechanism held in reserve
if tier escalation proves insufficient). 2603.17683 (Sensi — perception
hallucination cascades; supports grounding the proposer on heuristically
verified events, not LLM-perceived ones).

**Priority order of in-envelope transplants:** (1) tier-escalating click
coverage (INERT-START fix, ~1–2 d); (2) persistence probe (~hours);
(3) Go-Explore archive+return (~2–4 d).

### → World model: R1 region factoring (prior art found — cite, don't reinvent)

**2505.10819 — PoE-World (Piriyakulkij et al.).** World model = weighted
product of hundreds of small LLM-synthesized programmatic experts, each
restricted to one attribute of one object type, with **uniform distribution
over everything an expert doesn't predict** — partial coverage /
NO_PREDICTION built into the formalism; hard constraints as indicator
experts. Montezuma from minutes of demos: score 100 vs WorldCoder 0.
*Implications:* primary R1 citation. Position our contribution as **region
factoring on raw 64×64 grids without an object-centric API** vs their
per-object-attribute factoring on extracted object lists. Their
uniform-on-unset semantics is cleaner than a sentinel NO_PREDICTION — worth
adopting in verifier scoring (~days). Soft weights are overkill for
deterministic envs; their hard-constraint experts map to our exact-replay
gate.

**2510.12088 — OneLife.** Precondition-gated programmatic laws in a
probabilistic-programming frame; only laws whose preconditions fire
participate — the precondition-gated variant of partial program WMs. Their
**state-ranking metric** (does the WM rank the true next state above
distractors) is a cheap intermediate eval for partially-covered models
before full planning (~days to add). "One life" single-episode framing
matches our budget narrative.

### → World model: R3 latent state / non-Markov (48% census)

**AutumnSynth (Das, Solar-Lezama, Tavares — POPL 2023).** THE pre-LLM
anchor. Synthesizes 2D grid-game source from one play episode; when event
synthesis fails to explain an observation, it **invents a latent variable
via automata synthesis** and retries — solved 27/30 games incl.
complex latent state. *Implication:* must-cite; frame R3 as the LLM-era
version of failure-triggered latent invention. Mechanism to adopt: use our
conflict signatures as the symbolic trigger for a dedicated "propose a
hidden variable + update rule" prompt. Effort ~1 week (prompt +
control-flow).

**2605.30880 — PatchWorld.** Executable **belief-state programs** (code
carrying latent state across actions) induced from offline trajectories via
counterexample-guided repair, gradient-free. *Implications:* cite as
executable-WMs-under-partial-observability prior art; adopt their
belief-state-program interface (state dict threaded through transitions)
rather than inventing one (~days). **Pitfall:** their
fidelity/discriminativeness tradeoff — improving observation-prediction
fidelity can hurt action-discriminative dynamics; exact replay optimizes
fidelity, so also check the model ranks *actions* correctly.

**2605.13740 — Pinductor.** LLM proposes POMDP models with explicit hidden
state, refined against a belief-based likelihood. In our deterministic
setting this simplifies to an **existence check**: does there exist a
latent trajectory under the proposed update rule consistent with all
observed triples — an exact constraint check, natural R3 acceptance
criterion. Effort ~1–2 weeks (verifier extension).

**2605.16725 — "Baba in Wonderland" (Alice).** Online executable-WM
learning under prior misalignment (lexical priors useless — analogous to
our no-instructions setting). Mechanism: "preservation conflicts" (new
program breaks previously-explained transitions) → refine into hypothesis
classes → **class-stratified counterexamples** fed back, plus class-aware
exploration. *Implication:* when our verifier rejects a rule for breaking
old triples, cluster broken triples (by region/action) and feed stratified
counterexamples instead of a flat list. ~1 week. Closest published analogue
of our transition-store regression check.

**Supporting R3 citations:** 2601.18620 (CASSANDRA — deterministic-code /
latent-probabilistic split; supports the region-factoring intuition),
2605.12978 ("Useful Memories Become Faulty…" — LLM-consolidated memory
across ARC episodes degrades; supports exact-replay-verified programs over
free-text memory).

### → Part 2: LLM proposer, repair loop, truncation fix

**Repair-negative has published company.**
- **2604.10508 — "How Many Tries Does It Take?"** (the 2026 Olausson
  follow-up; 7 models, ≤5 rounds, T=0). Repair universally helps at 70B+
  (and uses 39–54% fewer tokens than resampling); **the 8B is the one model
  where resampling beats repair** (79.9% vs 76.8%); two rounds capture
  76–95% of achievable gain; assertion/logic errors hardest (~45% repair
  success vs ~77% name errors). They see *no* degradation — but at greedy
  decoding with hidden ground-truth tests. Cite as consistent-with our
  14B-negative / Next-positive split; contrast our −0.033 at temp 0.8 on
  open-ended rule induction as the domain/temperature difference. Their
  tokens-per-pp (9.5K–20.8K) are citable economics baselines.
- **2604.01029 — "Revision or Re-Solving?"** Weak draft content is
  *actively harmful* — revision anchored to a bad artifact degrades output.
  The published mechanism behind our "decent round-1 rules overcorrected
  toward counterexamples". Their null-draft ablation (repair prompt without
  the round-1 rule body) is a ~1-day experiment we could replicate.
- **2306.09896 — Olausson, "Is Self-Repair a Silver Bullet?"** Anchor
  citation; pair with 2604.10508 as "both regimes coexist".
- **2512.02389 — Synthetic error injection fails to elicit
  self-correction.** Detection ≠ correction; supports externalizing
  verification.

**Gated acceptance — novelty re-scope required.**
- **2605.24613 — GuardedRepair** publishes verify-gated repair acceptance
  (accept answer-changing repairs only when deterministic guards support
  replacement; zero broken-correct) — in *math reasoning*. Must cite. Our
  gating novelty re-scopes to: **held-out counterexample-split gating for
  executable rule repair under an interaction budget**.
- **2510.03217 — Abstain and Validate (Google).** Industrial confirmation
  that *rejecting* repair attempts is where the win is (+39 pp combined).
  Suggests a predict-before-repair abstention heuristic (skip repair when
  round-1 score is below a floor) — ~1–2 days.

**Truncation fix (the Next 17/24 defect) — literature verdict:**
budget increase + two-turn analyze-then-code; *avoid* terse-instructions
and constrained decoding as primary fixes.
- 2504.14350 (strict output-length constraints degrade code correctness —
  our 1500-token truncations are a known budget-binding regime, not a model
  defect; "bigger budget" is the boring-correct first fix; see also
  2602.14444 "Broken Chains").
- 2511.01807 (Plan-and-Write — training-free budget-allocating planning
  stage; precedent for two-turn analyze-then-code, ~1 day of prompt
  plumbing).
- **Pitfalls:** 2510.15211 (ReasonIF — models ignore instructions applying
  to the reasoning region; predicts terse-output instructions underdeliver;
  worth a quick N=8 A/B, ~hours); 2604.06066 ("structure snowballing" —
  constrained decoding during reflection *worsens* semantic correction on
  small dense models).

**Economics / comparison targets:** 2502.14382 (S* — hybrid
sample-AND-repair with execution feedback; the standard citation matching
our repair-gated hybrid; their distinguishing-input selection we get free
via exact replay), 2605.07248 (PaT — escalate expensive model only on
verification failure, ~69% cost cut; motivates a 14B-drafts /
Next-repairs-on-failure cascade as a bake-off arm, ~2–3 days),
2605.13414 (TRIAGE — models can't self-allocate budgets; justifies
controller-side gating), 2503.23145 (CodeARC — interactive inductive
synthesis benchmark, closest structural relative of our loop; skim for
refinement-round statistics), 2604.05560 (FixAudit — trained-fixer upper
bound), 2511.19422 (SLMFix — small models repair *syntactic* failures;
matches the error-type split: our 14B-negative is on *semantic* repair).

**Proposer validity checks:** 2605.24375 (distilling game-CWM generation
into 3B models — citation that weak open models can emit valid CWMs when
wrapped in verification; their structural/semantic property checks are a
checklist for pre-replay static validation, ~days), 2512.22336
(Agent2World — static validation misses behavior-level errors; supports
exact-replay-over-unit-tests).

### → Fleet runner / strategy (budget, evaluation posture)

- AERA's benchmark-validity findings change *reporting*, not architecture:
  our public-set numbers need a private-set-facing caveat, and the
  null-coordinate check belongs in the harness audit before any census is
  quoted in the paper.
- RHAE means wasted probe actions cost score: the persistence probe
  (~1200 actions/game) is affordable but should be capped and only run
  pre-win (exploration debt is forgiven by the win-gate replay anyway).
- Multi-seed reporting (2601.00042's 8× seed spread) applies to the sweep
  censuses if we keep citing them as facts: rerun the 25-game census at
  ≥3 seeds before the paper freezes the 76%/48% numbers.

### → Paper (positioning, baselines, claims hygiene)

- **Numbers to beat / cite:** Rodionov GPT-5.5 15/25 public, mean RHAE
  58.12% (v2; v1 abstract had 32.58% — cite v2); DreamTeam 38.4%
  protocol-matched public SOTA; Symbolica Arcgentica 36.08% day-1
  community leaderboard (~$40/game, API); AERA 0.2116 public RHAE with a
  0.5B model and claimed 0.30 on the 55-game private set; Rudakov median
  30/52 preview levels; ARC Prize frontier analysis (GPT-5.5 0.43%,
  Opus 4.7 0.18%) vs our template-only 0.253%.
- **Claims hygiene:** scope the novelty claim as the conjunction (see §1);
  cite GuardedRepair before claiming gated repair; cite Rudakov before
  claiming salience-guided exploration; cite PoE-World/OneLife before
  claiming partial/factored program WMs; cite AutumnSynth before claiming
  latent-variable invention. Each of these is a "we'd have been scooped in
  review" landmine now defused.
- One terminology footnote: "code world model" collides with Meta's CWM
  (2510.02387, an LLM trained on execution traces) — disambiguate once.

---

## 3. PRIVILEGED-CONTEXT EXPERIMENT SPEC (track 5 → routed to Part 2)

**Source paper, read in full: arXiv:2605.30070 — "A Predictive Law for
On-Policy Self-Distillation From World Feedback" (He, Sieber, Saponati —
Tufa Labs, 2026-05-28).** Their gap: *self-teacher* = same weights (EMA
copy) conditioned on a privileged context c the student never sees; the
**initial student–self-teacher gap** = teacher val accuracy − student val
accuracy, measured before training, decoding held fixed. Their law: final
OPSD improvement is OLS-linear in that initial gap across context *kinds*
(Qwen3-8B slope 1.492, R²=0.949; Olmo-3-7B slope 0.663, R²=0.996;
scale-invariant 0.6B→8B, slope 1.508, R²=0.977). The gap is the headroom
training consumes. Refs [4]–[9] triaged: SDPO 2601.20802 (feedback as
privileged context — structurally our conflict ledger; test-time matches
best-of-k with 3× fewer attempts), OPSD 2601.18734 (canonical definition;
"verified traces" ≈ our current-model view), OEL 2603.16856 (**structured
experiential knowledge beats raw trajectories as context** — direct support
for the conflict-ledger design over raw transition dumps), CRISP
2603.05433 (even a trivial context carries a large gap), SDFT 2601.19897 +
OPCD 2602.12275 (related-work; OPCD is the exit ramp if we ever internalize
context into a fine-tune).

**The experiment ("gap table for the proposer prompt"):**

- **Question.** How much rule quality does each prompt component — **L**
  (conflict ledger), **T** (temporal context), **M** (current-model view) —
  buy per token, for Qwen3-Coder-Next?
- **Conditions: full 2³ factorial (8 conditions).** Not
  ablate-one-at-a-time: smoke tests showed T is *enabling* (hidden-pattern
  tasks fail entirely without it), so interactions are expected — L may be
  worthless without T to anchor when conflicts occurred. The factorial
  yields both marginals per component: Δ⁺(X) = gap(X alone),
  Δ⁻(X) = gap(full) − gap(full∖X). Base prompt identical; only L/T/M
  blocks inserted/removed. Temp 0.8 (established protocol).
- **Tasks/metrics.** Task A (event precondition): accuracy on the fixed
  60-item held-out. Task B (reframed structured response): mean Jaccard.
  N=8 samples/condition/task.
- **Gap statistic, mirroring 2605.30070:** gap(c) = mean metric(c) − mean
  metric(∅), same model, same decoding, same evidence exemplars. Explicit
  adaptations: (1) no training on Kaggle, so the gaps themselves are the
  deliverable — per their law they also estimate what an OPCD/SDPO
  internalization pass could later extract, so the table doubles as a
  screening table for a hypothetical fine-tune. (2) Their points are
  context kinds; ours are factorial combinations → report the two
  marginals. (3) Uncertainty: paired bootstrap over held-out items pooled
  over samples; noise floor = bootstrap SD of ∅; treat |Δ| < 2× pooled SE
  as zero.
- **Cost.** 8 × 2 × 8 = **128 generations** ≈ 80 min generation at our
  rental vLLM rate (~40 gens ≈ 25 min) + deterministic scoring. One rental
  session.
- **Decision rules.** (a) Δ⁻(X) > noise floor on either task → X stays in
  the deployed serializer. (b) Δ⁻(X) ≈ 0 but Δ⁺(X) > 0 → redundant; keep
  the cheaper of the overlapping pair. (c) Δ⁻(X) < 0 → distractor; drop,
  reclaim tokens for exemplars. (d) T dominates and L ≈ 0 → latent state is
  inferred from raw history; spend ledger tokens on a deeper temporal
  window. (e) All gaps ≈ 0 on B but not A → context helps classification
  not synthesis; reframing, not context, is the lever. Final output: table
  of gap × marginal token cost so the serializer budget allocates by
  gap-per-token.

---

## 4. RELATED-WORK SKELETON (for the October draft)

**§ ARC-AGI-3 benchmark + agents:**
2603.24621 (benchmark paper; humans 100%, frontier <1%); 2601.10904 (ARC
Prize 2025 report); 2605.05138v2 Rodionov (executable WMs, primary
baseline); 2605.09650 DreamTeam (workspace optimization, public SOTA);
2605.13037 MAP (map-then-act; validates explore-then-induce); 2605.25931
AERA (validity critique + small-model entry); 2512.24156 Rudakov (graph
exploration); 2603.17683 Sensi (perception-bottleneck failure mode);
2603.13372 (living survey, 82 approaches).

**§ Program/executable world models:**
WorldCoder 2402.12275; GIF-MCTS 2405.15383; PoE-World 2505.10819 (factored,
partial-coverage); OneLife 2510.12088 (precondition-gated); PatchWorld
2605.30880 (belief-state programs); Baba/Alice 2605.16725 (online,
preservation conflicts); Pinductor 2605.13740 (POMDP induction);
CASSANDRA 2601.18620 (deterministic/stochastic split); AutumnSynth
(POPL 2023 — latent-variable invention); GGP CWMs 2510.04542 (rules-given
contrast); Agent2World 2512.22336 (behavior-level validation); 2605.24375
(weak-model CWM generation); cognitive lineage: TBRL 2503.20124,
2509.00074; terminology footnote vs Meta CWM 2510.02387.

**§ Skill/program libraries & memory (the novelty-gap fence):**
Pang 2025 (static-ARC program library); ArcMemo 2509.04439; SOAR
2507.14172; EvoLib 2605.14477; Evolving Programmatic Skill Networks
2601.03509; skill-library cluster one-liner (SAGE 2512.17102, SkillRL
2602.08234, SkillOS 2605.06614, HiSME 2605.28390, AEL 2604.21725,
SkillEvolBench 2605.24117); 2605.12978 (memory consolidation degrades —
motivates verified programs).

**§ Exploration:**
Go-Explore 1901.10995 / 2004.12919; IGE 2405.15143; #Exploration
1611.04717; Rudakov 2512.24156; AERA 2605.25931; ScreenSearch 2605.16024;
Look Before You Leap 2605.16143 (coverage metric); pitfalls: APEX
2605.21240, 2601.00042 (seed variance); SIERL 2602.00460 (frontier
subgoals, RL contrast).

**§ Repair / verification loops:**
Olausson 2306.09896; 2604.10508 (scale-dependent repair); 2604.01029
(revision-anchoring harm); GuardedRepair 2605.24613; Abstain-and-Validate
2510.03217; 2512.02389 (detection ≠ correction); S* 2502.14382; PaT
2605.07248; CodeARC 2503.23145; SLMFix 2511.19422; FixAudit 2604.05560;
TRIAGE 2605.13414; truncation: 2504.14350, 2602.14444, ReasonIF
2510.15211, 2604.06066, Plan-and-Write 2511.01807.

**§ Privileged context / distillation (if the gap experiment ships):**
2605.30070 (predictive law); SDPO 2601.20802; OPSD 2601.18734; OEL
2603.16856; CRISP 2603.05433; SDFT 2601.19897; OPCD 2602.12275; EDGE-OPD
2605.23493; ACE 2510.04618.

---

## 5. COMPETITOR PROFILES

**Tufa Labs (MindsAI lineage: Cole, Osman, Smit).** No ARC-AGI-3 arXiv
paper, but the most active competitor in practice: Smit's StochasticGoose
won the Aug-2025 preview (4-layer CNN action prediction + simple RL on
sparse level rewards, hash-table dedup, retraining between levels;
12.58% preview / 0.25% full), and in May 2026 they raised the leading
Kaggle score 0.68%→1.17% with an unpublished "novel approach". arXiv
output is RLVR/self-distillation (LADDER 2503.00735, 2505.08827,
2605.30070). **Bet:** small trained models + online within-game RL/TTT,
anti-pure-LLM. **Compute:** open-weight/local — *our* regime, and they
lead it. **Collision: adjacent** — no library, no cross-env transfer
published; the per-game CNN is the opposite of compositional reuse. Watch
the unpublished jump.

**Symbolica AI (Arcgentica).** Community-leaderboard 36.08% (113/182
levels, ~$1,005/run) via orchestrator–subagent harness; subagents write and
test Python, return compressed summaries. **Bet:** frontier-LLM harness
engineering. **Compute:** API, ~$40/game — Kaggle-ineligible.
**Collision: no** — per-game, no persistent library; a leaderboard ceiling
reference.

**NVIDIA/Technion DreamTeam (2605.09650).** "Artifacts in place of
parameters, textual feedback in place of gradients"; public-set 36%→38.4%
with 31% fewer actions. **Bet:** persistent workspace optimization for
frozen frontier agents. **Compute:** API frontier. **Collision: adjacent —
closest conceptual neighbor**; no cross-environment transfer claimed, wrong
compute regime. Cite and differentiate explicitly; flagged for a full read.

**Quiet groups.** NVARC (NVIDIA KGMoN, ARC Prize 2025 winners; synthetic
data + TTT on 4B): nothing on v3 — highest-capability sleeping competitor,
their posture fits the Kaggle track. ARChitects: ARC-AGI-2 only (PoE,
masked diffusion). Jeremy Berman: static ARC only. Giotto.ai: pivoted to
product. Rodionov: zero citations, no follow-on ecosystem yet.

**Bottom line:** nobody found is publicly doing online cross-environment
skill/program-library learning on ARC-AGI-3 under open-weight offline
compute. Nearest neighbors: DreamTeam (library-ish artifacts, wrong regime,
no transfer) and Tufa (right regime, online learning, no library).

---

## 6. QUERY LOG (reproducibility)

Method notes: arXiv export API requires HTTPS and rate-limits aggressively
(~6 s between requests; 429s rerun or routed through abs/html pages +
WebSearch). Citation chasing via Semantic Scholar API. "0 hits" rows are
genuine nothing-found results, reported as such.

### Track 1 — scooping watch
| query | source | hits | useful |
|---|---|---|---|
| all:"ARC-AGI-3" (date-sorted, 50) | arxiv-api | 9 | yes — full v3 paper census |
| all:"ARC-AGI" AND all:interactive | arxiv-api | 14 | yes — 2605.20784, 2605.12978 |
| all:"ARC Prize 2026" | arxiv-api | 0 | nothing-found |
| all:"skill library" (40) | arxiv-api | 40 (saturated) | yes — field explosion, no v3 overlap |
| all:"cross-environment transfer" | arxiv-api | 5 | partly |
| all:"program library" (40) | arxiv-api | 40 | mostly software-lib noise |
| all:"cross-task transfer" AND "program library" | arxiv-api | 0 | nothing-found |
| citations of 2605.05138 | semantic-scholar | 0 | yes — no citing papers |
| citations of 2509.04439 (ArcMemo) | semantic-scholar | 18 | yes — EvoLib, SkillOS, DreamTeam |
| citations of 2507.14172 (SOAR) | semantic-scholar | 25 | yes — AERA, DreamTeam |
| 8 web queries (skill library × ARC-AGI-3, Pang follow-up, Rudakov citations, offline-envelope papers) | websearch | ~6–10 ea | gap confirmed unoccupied; no offline-envelope paper exists |

### Track 2 — executable/program world models
| query | source | hits | useful |
|---|---|---|---|
| all:"executable world model" | arxiv-api | 7 | 4 (PatchWorld, Alice, Rodionov v2, Agent2World) |
| "code world model(s)" / "world model as code" | arxiv-api | 10 | 4 + term-collision discards |
| "program synthesis" AND "world model" | arxiv-api | 6 | 4 (PoE-World, TBRL, …) |
| "world model" AND LLM AND planning/interactive | arxiv-api | ~37 | 6 new, mostly filler |
| "factored world model" OR "object-centric world model" | arxiv-api | 26 | 0 adoptable (all gradient-trained) |
| "hidden state" AND "world model" AND agent | arxiv-api | 11 | 1 strong (Pinductor) |
| "stateful" AND "world model" AND LLM | arxiv-api | ~30 | overlap only |
| WorldCoder/ARC-AGI-3/non-Markov follow-ups | websearch (429 fallback) | ~7–10 ea | OneLife, CASSANDRA, AutumnSynth |

### Track 3 — exploration for evidence
| query | source | hits | useful |
|---|---|---|---|
| "directed exploration" AND "sparse reward" | arxiv-api | 12 | 0 (all RL-training) |
| exploration AND "LLM agents" | arxiv-api | 30 | 3 (APEX, LBYL, MAP) |
| "novelty search" AND grid | arxiv-api | 1 | 0 |
| "intrinsic motivation" AND "rare events" | arxiv-api | 0 | nothing-found |
| affordance AND exploration AND interactive | arxiv-api | 25 | 3 |
| "Go-Explore" | arxiv-api | 25 | 5 |
| all:"ARC-AGI-3" | arxiv-api | 9 | 6 in papers-that-matter |
| "frontier-based exploration" AND agent | arxiv-api | 9 | 0 (robotics) |
| exploration AND "state transitions" AND sparse | arxiv-api | 14 | 0 new |
| "GUI agent" AND exploration | arxiv-api | 25 | 1 (ScreenSearch) |
| "interactive elements" AND agent AND discover | arxiv-api | 0 | nothing-found |
| controllability / "action effects" exploration | arxiv-api | 25+13 | 0 |
| "goal-conditioned exploration" | arxiv-api | 429 ×2 | failed, not rerun |
| ARC-AGI-3 skill library cross-game | websearch | 9 | gap confirmed; ~~note: competition forbids cross-game learning at eval~~ **CORRECTED 2026-06-10, see §7 — no such rule exists in any official source** |

### Track 4 — repair/verification loops
| query | source | hits | useful |
|---|---|---|---|
| "self-repair" AND "code generation" | arxiv-api | 8 | 2604.10508, 2306.09896 |
| "execution feedback" AND code AND LLM | arxiv-api | 40 | 6 |
| "self-correction" AND LLM AND fail | arxiv-api | 40 | 5 |
| resampling AND repair AND code | arxiv-api | 1 | confirms thin field |
| "test-time compute" AND "code generation" | arxiv-api | 14 | 3 |
| truncation AND "code generation" | arxiv-api | 15 | 0 direct |
| "length control" AND LLM | arxiv-api | 30 | 3 |
| "small language models" AND code AND repair | arxiv-api | 2 | 1 (SLMFix) |
| Olausson follow-ups, validation-gated repair, truncation mitigation | websearch | ~7–9 ea | 2604.01029, GuardedRepair, 2510.03217, 2504.14350 |

### Track 5 — privileged context
| query | source | hits | useful |
|---|---|---|---|
| 2605.30070 abs + PDF (full read) | arxiv.org | 1 | primary |
| refs [4]–[9] via export api | arxiv-api | 429 (all) | fell back to abs pages, 5/5 + websearch resolved retitled [8] |
| "privileged information" LLM distillation | arxiv html search | 11 | EDGE-OPD, 2606.10385 |
| "context ablation" LLM agents | arxiv html search | 0 | nothing-found |
| "teacher-student gap" language model context | arxiv html search | 0 | nothing-found |
| 2 web fallbacks for the above | websearch | 6–8 ea | marginal (ACE, LUPI) |

### Track 6 — competitors
| query | source | hits | useful |
|---|---|---|---|
| all:"Tufa Labs" / abs:"Tufa" AND abs:"ARC" | arxiv-api | 0 | affiliation not indexed |
| tufalabs.ai + /research/ | webfetch | 2 | pub list; "ARC v3" a stated priority |
| citations of 2512.24156 | semantic-scholar | 1 | DreamTeam |
| citations of 2605.05138 | semantic-scholar | 0 | quiet |
| ~10 web queries (StochasticGoose, Arcgentica, Berman/ARChitects/Giotto/NVARC on v3, arcprize.org) | websearch/webfetch | 5–10 ea | profiles above; quiet groups confirmed quiet |

---

## 7. CORRECTION (2026-06-10): the "no cross-game learning at eval" claim is unsupported

Track 3's query log carried an unsourced note that "the competition forbids
cross-game learning at eval." Verified against every official source the
same day — **no such rule exists.**

Sources checked, none of which mention cross-game learning, memory
persistence across games, or fresh-instance requirements:

- arcprize.org/competitions/2026 (rules section: open-source, eligibility,
  no-internet — nothing else) and /competitions/2026/arc-agi-3.
- ARC-AGI-3 technical report (arXiv 2603.24621) — no evaluation-protocol
  rule on inter-game persistence at all.
- docs.arcprize.org, full index — including the **Competition Mode** page,
  which is the authoritative list of forced behaviors: API-only
  interaction, scored against all environments, only level resets, one
  `make` per environment, one scorecard, no inflight scorecard reads.
  Nothing about cross-game state.
- The official **Swarms** doc runs one agent instance per game
  *concurrently in one process* — an architectural default, not a
  prohibition; threads can share state, and nothing says they may not.
- Local ARC-AGI-3-Agents repo (README, llms.txt): nothing.
- Kaggle rules page is login-gated (JS-rendered, unfetchable here) —
  residual uncertainty; **eyeball it once when accepting the rules.**

Rodionov's "fresh agent instance per playthrough" is *their own protocol
choice*, not a competition rule. The only official lever against transfer
is soft: ARC Prize discretion to exclude apparently-overfit submissions
from private-game testing. Consequence for the paper: the novelty claim
does NOT need an "applied frozen at eval" hedge — online cross-environment
learning during the evaluated run appears to be permitted, and arguably is
exactly the anti-overfitting story the discretion clause rewards.

### Two adjacent findings from the same verification (both material)

1. **Competition Mode doc wording vs our Workstream A finding.** The doc
   says "Only *Level Resets* are permitted, *Game Resets* are not allowed
   and become *Level Resets*." Our arc_agi-0.9.8 code reading + empirical
   P3 test showed RESET-after-WIN *does* full-reset and mints a new play in
   competition mode — the two-phase replay strategy depends on it. The doc
   sentence is ambiguous about whether the WIN-path full reset counts as a
   disallowed "Game Reset" (locally it works; the doc may simply be
   describing the RESET@0/mid-game interception we already verified). This
   is the exact Kaggle caveat NOTES.md flagged, now with an official test
   surface: the **ARC-AGI-3-Kaggle-Starter repo**
   (github.com/arcprize/ARC-AGI-3-Kaggle-Starter) is the real submission
   scaffold ("hosts the same game engine the Kaggle gateway runs").
   **Action: clone it and run the tt01-style WIN→RESET experiment through
   its exact path before relying on two-phase in a submission** (~half a
   day; resolves the single biggest strategic unknown).
2. **Runtime-limit discrepancy.** A mirror of the Kaggle overview states a
   **6-hour** notebook limit; NOTES.md and all budget arithmetic assume
   ≤9h. The starter confirms RTX 6000 (`g4-standard-48`) is real and
   ARC-AGI-3-exclusive. If 6h is right, the model-time envelope shrinks
   from ~5.4h to ~2.4h at current overhead — material to the
   attempts-per-rule math. **Verify the limit on the Kaggle overview page
   when logged in.**

---

## Actionable backlog distilled (effort-tagged, NOT implemented — routing only)

| item | routes to | effort |
|---|---|---|
| Tier-escalating click coverage (INERT-START fix, per 2512.24156) | WinSeeker | 1–2 d |
| Per-action persistence probe, ~200 reps (per AERA) | WinSeeker | hours |
| Go-Explore archive + return-via-replay (+optional LLM ranking) | WinSeeker | 2–4 d |
| Segment-granularity state hashing for visit counts | WinSeeker | ≤1 d |
| Null-coordinate vulnerability check | harness audit | hours |
| Conflict-triggered "propose hidden variable" prompt (AutumnSynth-style) | R3 | ~1 wk |
| Latent-trajectory existence check as R3 acceptance (Pinductor-derived) | R3 verifier | 1–2 wk |
| Stratified counterexample feedback (cluster broken triples, per Alice) | R3/proposer | ~1 wk |
| Uniform-on-unset NO_PREDICTION semantics (PoE-World) | verifier scoring | days |
| State-ranking intermediate eval (OneLife) | harness eval | days |
| Pre-replay static validation checklist (2605.24375) | proposer | days |
| Repair-abstention floor (skip repair below round-1 score floor) | proposer | 1–2 d |
| Null-draft repair ablation (2604.01029 replication) | bake-off | ~1 d |
| Terse-instruction A/B (ReasonIF prediction check) | bake-off | hours |
| Two-turn analyze-then-code prompt split | proposer | ~1 d |
| Cheap-drafts/expensive-repairs cascade arm (PaT-style) | bake-off | 2–3 d |
| Privileged-context 2³ gap experiment (§3) | Part 2 | 128 gens, ~80 min GPU + 1 d analysis |
| Multi-seed rerun of the 25-game censuses before paper freeze | sweep | ~7 h wall × seeds |
