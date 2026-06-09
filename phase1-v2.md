# ARC Prize 2026 — Phase 1 Document, v2 (post-verification)
## Taxonomy, corrected record, rules, and the rescoped thesis

**Status:** Revision of the v1 skeleton after external verification (June 2026). Historical figures below are corrected per that verification; items post-dating my own reliable knowledge (the 2025 final results, the 2026 rules, Rodionov 2026, etc.) are taken from the verification report and its cited primary sources — I have not independently confirmed them, so spot-check anything load-bearing against the linked arXiv/Kaggle pages before it goes into the paper. Where v1 said [VERIFY], this version says what was found.

---

## 0. The rules that shape everything (new section)

Three tracks, $2M total: ARC-AGI-3 ($850K), ARC-AGI-2 ($700K), paper track ($450K). **Binding deadlines (Kaggle rules govern where sources conflict):** Milestone 1 — June 30, 2026, requires a public open-source notebook; Milestone 2 — Sept 30; entry deadline Oct 26; final code Nov 2; paper Nov 9.

**Compute envelope (the constraint that defines the feasible region):** Kaggle notebooks, ≤9 hours, internet disabled, *no API LLMs during evaluation* — verbatim "no API-based systems like GPT/Claude/etc." Open-weight models and pretrained weights as external data are allowed. ARC-AGI-3 runs on RTX 6000 Blackwell-class machines; ARC-AGI-2 on L4x4/12h. Mandatory open-sourcing before private scores, all tracks.

**Scoring is RHAE — Relative Human Action Efficiency — and it is the single most strategy-shaping fact.** Per level: score = (human_baseline_actions / your_actions)², capped at 1.15× baseline. Game score is the level-index-weighted average; you must finish the final level to unlock 100%; final score averages across games. **Reasoning, tool calls, and internal compute are not counted — only environment interactions are.** Implications, in order of importance: (1) internal simulation is free, so an agent that thinks long and acts seldom is directly rewarded; (2) the quadratic means every wasted exploratory action compounds against you — exploration must be *informative per action*, not exhaustive; (3) a world model good enough to plan through before acting converts free compute into the scored currency. The scoring rule is, in effect, a subsidy for model-based agency.

**Hidden eval:** 110 private games (55 public-LB + 55 private-LB) vs. 25 public ones. Human baseline is the upper-median first-time human player per level; humans solve 100% of environments. The 25 public environments ship as readable Python (ARCBaseGame subclasses) with per-level `baseline_actions` in metadata.json — **exact RHAE is computable locally**, and a fully offline local runtime exists in the arc_agi wheel.

**Agent interface (confirmed implementable):** actions are RESET, ACTION1–5 (directional/select-style), ACTION6(x, y) with x,y ∈ [0,63], ACTION7; `available_actions` arrives per frame, so the action set is per-game-discoverable. Observations are FrameData: stacks of 64×64 16-color grids plus state (WIN/GAME_OVER/etc.) and level counters. Contract: subclass Agent, implement choose_action and is_done. The starter repo's MAX_ACTIONS=80 is a repo default, not a rule. Caveat: the starter's LLM agent templates call OpenAI APIs — local development scaffolding only; nothing API-based can run in the evaluated submission.

**Paper rubric (now verbatim, six criteria 0–5, averaged):** Accuracy, Universality, Progress ("how much does the paper increase the overall chance of anyone achieving 85%/top score"), Theory ("why … as opposed to merely how"), Completeness, Novelty. Papers must link to a Kaggle code submission, which "need not achieve a high score." 1,500-word cap. Scores above 4.5/5 qualify for a $375K discretionary pool beyond guaranteed prizes — the rubric is not just a ranking device; it has money attached at the top end.

Minor primary-source conflicts exist (milestone prize splits, Nov 8 vs 9 paper date, license wording); treat Kaggle's rules pages as binding in each case.

---

## 1. The corrected historical record (Families 1–6, condensed)

The v1 mechanism/why-it-works/why-it-breaks analysis stands; figures now corrected, with the eval-set caveat that public, semi-private, and private numbers are *not comparable* — a conflation that has embarrassed prior writeups.

**Family 1 — DSL + brute-force search.** Icecuber 2020: 20.6% private (top-3 guesses). Failure modes stand: frozen primitives, depth explosion, no learning.

**Family 2 — Library learning.** Best-ever ARC-specific result: DreamCoder+PeARL (Bober-Irizar & Banerjee 2024) at 4.5% public eval — worse than 2020 brute force, with the ablation finding that the abstraction-sleep phase *degraded* later iterations. The v1 "gap" paragraph claiming nobody had done online library learning within a single eval run is **withdrawn**: Pang's evolutionary synthesis grows its program library during semi-private evaluation (77.1% AGI-1 / 26% AGI-2), and ArcMemo (arXiv:2509.04439) updates a natural-language concept memory online. Both are now nearest prior art to cite, not gaps to claim. What remains true: neither operates in an interactive setting, and both consolidate over static puzzle solutions rather than environment dynamics.

**Family 3 — Direct transduction.** Pre-o3 frontier results on semi-private: GPT-4o 5%, Gemini 1.5 4.5%, Claude 3.5 Sonnet 14%, o1-preview 18%. No frontier LLM has ever touched the true private set. Failure modes stand.

**Family 4 — Test-time training.** ARChitects 2024: 53.5% private (their ICML'25 paper's 71.6% is public eval — do not conflate). MindsAI 55.5% private, unclaimed. Akyürek et al.: Llama-3 8B to 53% public validation, 61.9% ensembled with BARC, 47.5% verified semi-private. The didn't-transfer claim is confirmed and now quantified: 53.5→16.5 and 55.5→12.6 across the AGI-1→AGI-2 transition — though note the TTT family *still won* the 2025 cycle, so "didn't transfer" means "degraded," not "displaced."

**Family 5 — LLM-guided synthesis.** Greenblatt: 50% public eval at ~8,000 samples/task (5k + 3k revision), 43% verified semi-private. Berman: 53.6% semi-private (2024, Sonnet 3.5); 2025 update evolving English instructions with Grok-4 reached 79.6% AGI-1 / 29.4% AGI-2. BARC: induction 38%, transduction 43%, ensemble+TTT 56.75% public validation pass@2; 19% Kaggle private scaled-down. Verification-asymmetry framing stands and now extends naturally to the interactive setting (see §3).

**Family 6 — Frontier reasoning.** o3: 75.7%/87.5% semi-private, with ARC Prize's repriced costs of ~$200 and ~$4,560 per task; released o3 on ARC-AGI-2: 1.9–3%. Verified ARC-AGI-2 SOTA as of Dec 2025: GPT-5.2 Pro, 54.2% at $15.72/task. The efficiency framing survives intact: 24% at $0.20/task under Kaggle limits versus 54% uncapped is the cleanest single illustration that the leaderboard measures a different quantity than the public-eval literature.

---

## 2. The 2025 cycle, now filled in (Family 7)

Final ARC-AGI-2 private leaderboard: NVARC (NVIDIA — Sorokin & Puget) 24.03%, ARChitects-style TTT on a ~4B model plus TRM components and massive synthetic data at ~$0.20/task; ARChitects 16.53% with a 2D-aware masked-diffusion LLM and recursive refinement; MindsAI 12.64% (TTFT); Lonnie 6.67% (no public method description located — read their Kaggle writeup directly); Barbadillo 6.53%. Grand prize unclaimed. The official report (arXiv:2601.10904) names per-task refinement loops as the defining 2025 theme.

**Paper awards — your rubric case studies, read these as studies in scoring, not just method:** 1st TRM (Jolicoeur-Martineau; 7M-parameter recursive network, 45% AGI-1 / 8% AGI-2 — note a tiny model with a strong *why* beat bigger scores, which is the Theory criterion working as advertised); 2nd SOAR (Pourcel et al.; evolutionary synthesis + hindsight self-training); 3rd CompressARC (MDL, zero pretraining). The pattern across all three: a single crisp theoretical commitment, executed and ablated. That is the bar.

---

## 3. ARC-AGI-3 prior art and the rescoped thesis (Family 8, rewritten)

The v1 claim that this space was "nearly empty" is dead. Current map:

**Preview competition (July 2025 — there was no March preview):** winner StochasticGoose, a CNN trained online to predict action→frame-change, 12.58% — which then **collapsed to 0.25% on the full 2026 benchmark**. That collapse is the cleanest empirical exhibit for the no-cross-task-learning failure mode: per-environment online learning with nothing carried across environments shattered on a broader distribution. Lead with it.

**Rudakov et al. (arXiv:2512.24156):** training-free state-graph exploration, beat LLM agents on preview games. Establishes that explicit state structure outperforms raw policies here.

**Rodionov, "Executable World Models for ARC-AGI-3" (arXiv:2605.05138, May 2026, AGI-26) — the big one.** Agent maintains a Python world model, verifies it against observed transitions, refactors toward simpler abstractions, plans through it. With GPT-5.5: 15/25 public games fully solved, mean RHAE 58.12%. This is v1's components (a)+(b) — exploration building an explicit, testable model verified against transitions — already published and strong.

**Frontier baselines remain dismal:** <1% at launch; GPT-5.5 at 0.43% in ARC Prize's May 2026 analysis, with named failure modes — local perception without global world models, false-success masking, overfitting to game analogies. Useful as the foil and as a checklist of what your agent must demonstrably not do.

**The surviving gap, rescoped.** Two facts compose it. First, component (c) is unclaimed: no published ARC-AGI-3 work grows a *cross-environment* skill/program library (targeted searches empty as of June 2026 — re-verify monthly; this is the claim most at risk of being scooped between now and November). Second, Rodionov's agent runs on GPT-5.5 via API, which **cannot enter the Kaggle track at all** — offline, 9 hours, open weights only. So the defensible thesis is:

> **Executable-world-model agency with online cross-environment skill abstraction, under the offline open-weight compute envelope.**

In failure-mode terms: it attacks *frozen priors* (the skill library grows) and *no-cross-task-learning* (StochasticGoose's collapse is the motivating exhibit), while importing Family 5's *verification asymmetry* into the interactive setting (hypotheses about dynamics are executable and checkable against observed transitions, exactly where transduction-style agents exhibit false-success masking). The strongest prior art (Rodionov) is structurally excluded from the leaderboard, so the open-weights replication-plus-extension is simultaneously the novelty delta and the only version that can score.

**Why RHAE actively favors this design:** planning through an internal model costs zero scored actions; confirmed skills from earlier games reduce exploratory actions in later ones; and the quadratic means those saved actions compound. The scoring rule and the thesis are aligned to an unusual degree — say so explicitly in the paper's Theory section, because it is a *why it works* argument grounded in the benchmark's own methodology.

**Honest difficulty assessment (carry into Phase 2 planning):** the hard part is no longer the concept — it's that Rodionov's result leans on GPT-5.5's code-writing strength, and the open-weight models that fit a 9-hour Blackwell budget write substantially worse world-model code. The core engineering risk is whether verification-and-refactor loops can compensate for weaker per-sample code generation. That is also, framed positively, the Progress contribution: demonstrating (or honestly bounding) how far the executable-world-model recipe degrades under open-weight constraints is information the whole field needs for anyone to hit the grand prize.

**Nearest prior art to cite in the novelty section:** Rodionov (executable world models, API-only), Pang and ArcMemo (online library growth, static setting), SOAR (hindsight self-training), Rudakov (training-free state graphs), StochasticGoose (per-environment online learning and its collapse). The novelty paragraph writes itself as a delta against these five.

---

## 4. Cross-cutting failure modes (updated synthesis)

The four v1 patterns survive contact with verification, now with sharper exhibits. **Frozen priors:** Families 1/5; PeARL's abstraction-sleep actively hurting is the cautionary tale for naive library growth — your abstraction-admission criterion must be stricter than recurrence-counting. **No verification:** the ARC Prize analysis of frontier ARC-AGI-3 failures (false-success masking) is this failure mode observed in the wild, in the interactive setting. **No cross-task learning:** StochasticGoose, 12.58% → 0.25%. **Compute mismatch:** Rodionov scoring 58.12% RHAE in a regime the competition forbids; 24% at $0.20 vs 54% at $15.72 on AGI-2. The thesis sentence answers modes 1 and 3 directly and weaponizes mode 2's solution; the paper's framing answers mode 4.

---

## 5. Updated reading list (priority order)

1. docs.arcprize.org/methodology — RHAE in full; recompute the strategy implications yourself.
2. Rodionov, arXiv:2605.05138 — the method to replicate, extend, and differentiate from. Read twice.
3. ARC-AGI-3 technical report (arXiv:2603.24621) and the 2025 technical report (arXiv:2601.10904).
4. Pang's writeup and ArcMemo (arXiv:2509.04439) — what "online library growth" already means; your delta must be precise.
5. SOAR, TRM, CompressARC paper-award writeups — rubric case studies.
6. Rudakov (arXiv:2512.24156) and the StochasticGoose preview writeup.
7. The 25 public environment sources + ls20 metadata — ground truth for the local RHAE harness.

## 6. Phase 2, reshaped by the deadline structure

Milestone 1 (June 30, public open-source notebook) converts Phase 2 from "prototype two ideas, kill one" into "ship a minimal viable agent in three weeks." Concretely: weeks 1–2 — local harness with exact RHAE scoring against the public environments, plus the simplest possible executable-world-model loop (an open-weight coder model proposing transition rules as Python, verified against observed frames, random-but-informative exploration). Week 3 — clean it into a public notebook and submit for Milestone 1; even a modest score plants the flag, de-risks the pipeline end-to-end, and starts the research log with a real artifact. The skill-library component — the actual novelty — becomes the July–September arc, with Milestone 2 (Sept 30) as its checkpoint and an ablation already implied by the architecture: same agent with the library frozen versus growing. The kill-one-idea decision is also reshaped: the choice is no longer between two concepts but between two library representations (executable program fragments à la Pang versus natural-language concepts à la ArcMemo), and the Milestone 1 agent is the testbed for deciding empirically.

## 7. Open items

Re-verify monthly that the cross-environment-library gap is still unclaimed (highest scooping risk). Resolve the minor rules conflicts against Kaggle's pages before relying on any date or split. Benchmark candidate open-weight coder models for world-model code quality within the 9-hour envelope early — it is the load-bearing engineering unknown, and if it fails, the fallback paper ("how far does the executable-world-model recipe degrade under open weights, and why") is still a legitimate Progress contribution.

---

## Addendum: spot-check results (appended 2026-06-09, Claude)

Load-bearing claims re-checked against primary sources after v2 was written. Document text above is verbatim as authored; corrections noted here, not edited in.

**1. RHAE formula — confirmed twice over, with one wording correction for §0.** The methodology page (docs.arcprize.org/methodology) matches verbatim: `level_score = (human_baseline_actions / ai_actions)^2`, level-index-weighted game average, final-level requirement, "Internal operations that do not alter the environment (tool calls, reasoning steps, retries) are not counted as actions," upper-median first-time-human baseline. The reference implementation in the shipped wheel (`arc_agi-0.9.8`, `scorecard.py` lines 166–206) agrees AND settles the cap semantics: per-level score is `min((baseline/actions)² × 100, 115.0)` — i.e. **the cap is a 115% ceiling on the level score**, not "1.15× baseline" actions as §0 phrases it. Reaching 115 requires using ≤ ~93.25% of baseline actions (1/√1.15). The game score is then capped at (sum of completed-level weights / total weights) × 100, so a fully completed game tops out at 100% but **115% levels can subsidize sub-100% levels within a game's weighted average** — and since weights are the 1-indexed level numbers, efficiency on later levels matters more. Uncompleted levels enter the average as 0. (The code comment at line 168 says "max 100" but the code caps at 115.0 — the comment is stale; trust the code.)

**2. Rodionov (arXiv:2605.05138) — confirmed, one new fact.** Abstract verbatim: GPT-5.5 high reasoning "fully solved 15 games and achieved a mean per-game RHAE of 58.12%"; a weaker-model config solved 8 games at 41.29%. Abstract also states private-set performance "remains to be tested." **New: the paper was revised to v2 on June 6, 2026** — three days before this spot-check. It is a moving target; re-read v2 before writing the novelty delta.

**3. arcprize.org/competitions/2026 — confirmed.** Milestones June 30 / Sept 30, submissions Nov 2, papers Nov 8 (Kaggle says Nov 9 — conflict already flagged in §0, Kaggle governs), "Internet access is not available during Kaggle evaluation (no API-based systems like GPT/Claude/etc.)" verbatim, CC0/MIT-0 license wording verbatim.

**4. Kaggle code-requirements page (9h, internet disabled) — NOT directly re-fetchable** (JS-rendered). The quote rests on the verification agent's retrieval and is independently consistent with arcprize.org's no-API statement. Eyeball the Code Requirements tab once in a browser before relying on the exact 9-hour figure for compute budgeting.
