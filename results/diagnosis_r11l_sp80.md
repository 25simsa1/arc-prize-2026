# Diagnosis: r11l and sp80 (Workstream-B-style, 2026-06-10)

GATE outcome: **precise named blocker routed — NOT a crack.** Both games are
the same class and route to the same place. Evidence below; no code changed
(this is routing, per the protocol's verdict).

## What each level requires (from source, public games)

**r11l** (`available_actions=[6]`, pure click; 60-action game limit via a
rendered step counter). A click-to-select-then-click-to-place puzzle:
- click a sub-element → SELECT a piece (`wiayqaumjug`/held); click a
  destination → the multi-cell piece DRAGS there, animated over frames
  (`yfbjozweime`, interpolated `qvnmfoxseus`); win = every piece sprite
  collides with its matching target (`flkdtg-*` by name suffix); `defgjl`
  obstacles kill after 5 collisions.
- **L1 (level 0): 1 piece, 1 target, 0 obstacles** → 2 clicks (select,
  place); the WinSeeker stumbles it (won in ~8 wandering clicks).
- **L2 (level 1): 2 pieces, 2 targets, 1 obstacle** → requires a SEQUENCED
  select-A → place-A → select-B → place-B with correct piece↔target pairing,
  avoiding the obstacle, inside the 60-action limit.

**sp80** (`available_actions=[1..6]`). Two phases:
- "change": click (A6) to SELECT a sprite (`vsoxmtrhqt`), arrow keys
  (A1-4) move it 1 cell, A5 commits; then a "spill" phase floods/recolors
  over a multi-step animation (`trhynadhiz` 0-6) and checks whether the
  filled region matches the target set → next_level, else penalty
  (`zlhbnhpcq`-=, ≤0 → lose).
- L1 → L5 grow in sprite count (7 → 21 refs); L2 adds pieces/structure.

## What the explorer + model actually did (trajectories)

| game | transitions | conflicts | hud_regions | model | coverage | reached |
|---|---|---|---|---|---|---|
| r11l | 5,292 | **0** | **[]** | 1 CONTRADICTED | grid/event **0.0** | L2, then 261 GAME_OVER |
| sp80 | 10,056 | **0** | **[]** | 1 CONTRADICTED | grid/event **0.0** | L2, then 290 GAME_OVER |

r11l L2: 105 actions, **92% of post-frames unique** (97/105 distinct
hashes); the few repeats are post-reset initial states. Exploration is NOT
the blocker — both games reach L2 and gather thousands of transitions; the
MODEL has zero coverage, so the planner is blind and the agent luck-clicks
until the action limit / obstacle kills it.

## Which wall — named precisely (a compound, gated in this order)

1. **R1′ — HUD ticker that CO-CHANGES with content (the gating wall, NEW
   subtype).** Both games render UI displays into the 64×64 frame: r11l a
   decrementing step counter (`rjtqizgnlf` 60→0) + a piece-status display
   (`xeuvojjxyk`); sp80 two `RenderableUserDisplay`s. The counter changes
   EVERY action, so combined with piece/selection changes ~all frames are
   unique → state explosion → no (level, frame, action) context ever
   recurs → 0 coverage. The current RegionAnalyzer MISSES it because its
   seed is the *sole-changer* signal (cd82's meter soloed on no-op actions);
   here the counter never solos (a click also moves/selects), so it is never
   seeded → never masked. `hud_regions=[]` confirms it. This differs from
   cd82: the latent state is fully RENDERED (observable), so there are **0
   conflicts** — the pathology is explosion, not aliasing.
   - General fix needed: detect HUD cells by "changes in ≥X% of ALL
     transitions regardless of being the sole change" (monotone-countdown /
     high-frequency-region detection), INDEPENDENT of the sole-changer seed.
   - **Danger, flagged:** r11l/sp80 are CLICK games; a naive "frequently
     changing ⇒ mask" rule would mask the interactive board itself (cells
     change AT the click) — exactly what the sole-changer+exogeneity guard
     was built to avoid. The R1′ detector must stay exogenous-aware (mask
     cells that change independently of WHERE you click/which action), so it
     catches the counter but not the play area.

2. **R2 — click-/selection-parameterized rigid-body motion (general
   template gap).** Placing/moving a piece translates a multi-cell sprite to
   an arbitrary destination. r11l: "click(x,y) → held sprite moves to
   (x,y)"; sp80: arrow-moves a *selected* sprite. No current template family
   expresses a rigid body whose motion is parameterized by the action AND a
   latent selection. (The known R2 gap, here entangled with selection.)

3. **R3 — latent selection + phase + animation timers (Part 4).** Which
   sprite responds is the held-selection (`wiayqaumjug`/`vsoxmtrhqt`);
   r11l moves animate over frames; sp80 has change/spill phases and a
   multi-step spill animation with commit/penalty counters. A hidden-state
   machine the (frame, action) store cannot express even with R1′+R2.

## Verdict + routing (protocol option (c))

- (a) Part 0 explorer fixes do NOT unblock — **confirmed by rerun**: the
  upgraded explorer reaches L2 and gathers 5k–10k transitions, yet coverage
  stays 0.0. Exploration was never the blocker here.
- (b) A single missing general template (R2) is necessary but **not
  sufficient** — R1′ (gating) and R3 (latent state) also block; implementing
  R2 alone cracks neither game, so option (b) does not apply as a crack.
- (c) **Latent state needed → route to Part 4.** r11l is Part 4's **first
  integration target** (cleanest self-contained mechanic: rendered latent
  selection + rigid-body drag + animation, ZERO conflicts so the latent
  state is fully observable — the friendliest possible R3 case). sp80 is the
  **second** (adds a multi-step spill-animation phase + arrow-move-selected,
  strictly harder). General prerequisites Part 4 will need, in order:
  **R1′ HUD-ticker masking** (exogenous-aware, must not mask click boards)
  and **R2 selection-parameterized rigid-body motion**.

## Why no "first real win" here, stated plainly
r11l L1's 4.76 RHAE is the WinSeeker stumbling a 2-click puzzle; L2 is a
3-wall compound (R1′+R2+R3) that no current component can model, and the
cap study already showed 3× time converts nothing on it. The path to the
project's first real win runs through Part 4 (latent state) + the R1′/R2
prerequisites — not through more exploration or a single template.
