# Gateway probe — output channels + probe designs (Task A)

2026-06-10. Goal: empirically answer Q1–Q5 (REPORT §7) about the Kaggle
eval gateway. Task A first: determine HOW a finding gets out of a
competition rerun, because that — not detection — is the hard part.

## The reframing that makes Task A central

An instrumented agent can **detect** Q1/Q2/Q3/Q5 directly from the
frame/score stream inside a single run, with no privileged access:

- **Q1 (cutoff enforced?)** — drive one level past 5×baseline of no-op
  actions; watch whether the gateway returns a terminal/“cut” state at
  ~5×baseline that the agent did not cause. Detectable from `fd.state`.
- **Q2 (what does it end? per-play or cumulative?)** — after a detected
  cutoff, probe whether further actions on the SAME level do anything,
  whether a RESET re-opens the level, whether other games still respond,
  and (cumulative test) whether a second play’s counter starts fresh or
  continues. All observable from returned frames.
- **Q3 (RESET-after-WIN mints a play?)** — win a level, RESET, check
  whether the next frame is a fresh level-0 play (mint) or a level reset.
- **Q5 (null-coordinate)** — send ACTION6 with x=y=0 / missing data;
  observe the frame response and whether it’s counted.

So the agent computes the answers internally. **The difficulty is
reporting a few bits out of a sandboxed rerun.** Hence channels first.

Q4 (wall-clock) is the exception — not frame-observable; addressed by
external fact (below) + an optional self-timing encode.

## Decisive inference: the gateway ≠ our `arc_agi`

The pip `arc_agi-0.9.8` action path has **no cutoff anywhere**
(grepped the whole package; only the 115 cap in scorecard.py:168–171).
The forge harness curls a **separate container** `gateway:8001`, mounted
only when `KAGGLE_IS_COMPETITION_RERUN=1`. Therefore:

- If the gateway enforces the 5× cutoff, it runs **different/newer server
  code** than our package. Q1 cannot be settled by local inspection.
- The online dev API (three.arcprize.org) is the same vendor but a
  different deployment; it may or may not run the leaderboard cutoff, so
  it is **proxy evidence at best**, not a substitute for the gateway.
- Conclusion: Q1/Q2 require probing the **actual Kaggle gateway** → a
  submission. Submission is the human’s action (daily-slot cost, no money);
  this doc designs what to submit.

## Channel determination

### Channel 1 — rerun logs visible to the submitter? (the easy channel)
**Status: UNKNOWN, lean NO for the hidden set; cheaply testable.**
- If visible: trivial. `main.py` already prints the full closed scorecard
  JSON to stdout at SIGINT (per-game, per-level scores), and we can print
  the agent’s entire frame/state/action trace. One instrumented submission
  answers Q1–Q5 outright.
- Why lean NO: the gateway serves the **hidden 55-game test set**; an agent
  sees hidden frames and could print them, so Kaggle commonly suppresses or
  truncates rerun logs for hidden-test code competitions. Could not confirm
  on Kaggle docs/forum (JS-gated, unfetchable here).
- **Cheap decider (recommended FIRST, 1 submission):** the *sentinel
  submission* below. If its unique string is readable in any post-rerun
  log surface, Channel 1 is open and the rest is easy.

### Channel 2 — gateway reachable in an INTERACTIVE session?
**Status: almost certainly NO for the hidden games.**
- The forge harness gates the gateway curl behind
  `if os.getenv('KAGGLE_IS_COMPETITION_RERUN')` — strong signal the
  gateway is up only during the rerun.
- Exposing the hidden test gateway to a live, log-visible interactive
  session would leak the private set, which Kaggle structurally prevents.
- Residual possibility: an interactive session reaches a **public-games**
  gateway. Unconfirmed; even if so it doesn’t answer Q1 (cutoff is a
  leaderboard-eval policy, unlikely on a dev endpoint). Treat as not
  available.

### Channel 3 — score-encoding (the always-works fallback)
**Status: AVAILABLE, with one precondition.**
- The leaderboard returns at least the **aggregate RHAE number**; assume
  that is the only reliably-visible quantity (the per-game scorecard URL
  main.py prints points at the gateway host, unreachable post-run; whether
  Kaggle surfaces a per-game breakdown is unconfirmed — design for 1 number).
- **Precondition:** RHAE scores only COMPLETED levels, so the score channel
  is writable only if the agent wins ≥1 level on ≥1 game. On the hidden set
  this is plausible but not guaranteed (76–88% starvation). Mitigation: the
  encoder rides on whichever games the agent *does* win; a control
  submission establishes the baseline winnable set/score.
- Encoding: the agent computes answer bit(s) internally, then **modulates
  the action count on a level it wins** to land the aggregate in
  distinguishable bands (win frugally = high; pad actions ×k = score/k²).
  Multiple bits via multiple winnable games or multiple padding tiers.

### Channel 4 — online dev API (three.arcprize.org) proxy
**Status: AVAILABLE to the human with an ARC API key (we hold none;
`~/.kaggle/access_token` is Kaggle-only).** Same-vendor black box, live,
free, rate-limited, internet-at-dev-time (allowed). Can probe Q1/Q2/Q5
*by proxy* on PUBLIC games with known baselines — but a negative result
(no cutoff on the dev API) does NOT clear the gateway, since the cutoff is
a leaderboard policy that the dev endpoint may omit. Use as a cheap
sanity pass before spending submissions, not as the answer.

### Channel 5 — local mirror (what we already own)
Our `arc_agi` competition-mode server reproduces play semantics exactly
(Workstream A) but by construction has **no cutoff** — so it answers Q3
mechanics offline and nothing about Q1/Q2 enforcement.

## Probe designs

### Probe 0 — Channel-1 sentinel (run FIRST; 1 submission)
Minimal agent: in the rerun branch, `print("PROBE_SENTINEL_7Qf3…")` plus a
one-line dump of `len(/api/games)` and the first frame’s shape, take a
handful of actions, exit cleanly so a valid submission is produced. After
the rerun, look for the sentinel in every available log surface.
- Sentinel visible → **Channel 1 open** → run Probe 1 (full diagnostic).
- Not visible → fall to Probes 2–5 (score-encoded).
Also reads out, for free if logs show: the number of games the gateway
mounts (public+hidden? hidden only?) — decides whether the known-baseline
tripwire (Probe 2A) is usable.

### Probe 1 — full diagnostic agent (only if Channel 1 open)
One submission answers everything by printing a structured trace:
- Per game: take no-ops on level 0 and log `(action_index, fd.state,
  fd.levels_completed)` until terminal or N≫5×max-baseline (2890). The
  index where `state` goes terminal without an agent-caused death, divided
  by the (printed) level baseline if available, gives the multiplier → Q1.
- On the first winnable game: win L0, log the frame; RESET; log whether the
  next frame is fresh L0 (mint) or same-level reset → Q3. Continue to L1;
  deliberately overshoot L1; log whether L2 becomes reachable → Q2 “ends
  level vs game”. Track a second play’s per-level counter start → Q2
  “per-play vs cumulative”.
- Send ACTION6 {x:0,y:0} and ACTION6 with no data; log response + whether
  counted → Q5.
Print a final JSON block; the closed scorecard JSON from `main.py` is a
second copy of the score side.

### Probe 2 — Q1/Q2 score-encoded (if Channel 1 closed)
Two-bit encoding on a winnable game, robust to all-hidden starvation by
riding on whatever the control wins.
- **2A known-baseline tripwire (preferred, iff gateway mounts a public
  game):** on a public game with known per-level baseline b0, take exactly
  `floor(5*b0)+2` no-op actions on L0, then win L0.
  - cutoff ends level → never completes L0 → that game scores 0.
  - no cutoff → completes L0 (score ~0 from the padding) AND continues; win
    L1 frugally → nonzero L1 contribution. Presence of the L1 contribution
    = “no cutoff” bit. Compare against a control notebook that wins L0+L1
    frugally on the same game.
- **2B baseline-agnostic (hidden-only set):** control notebook = frugal
  agent, record winnable set W and aggregate S_ctrl. Probe notebook =
  identical, but on every game inject a large L0 overshoot (50k no-ops)
  *before* its first completion, then continue normally.
  - no cutoff: still reaches/score later levels on games in W where ≥2
    levels are reachable → S_probe tracks S_ctrl on those.
  - cutoff: overshoot ends the game pre-completion everywhere → S_probe≈0.
  S_probe vs S_ctrl gives Q1. Q2 “ends level vs game/run” needs a game with
  ≥2 reachable levels in W; **honest limitation: if W has no multi-level
  game, score-encoding cannot separate ends-level from ends-game** — only
  Channel 1 / the known-baseline tripwire can. Flag for the human:
  Probe 0’s game-count readout tells us which regime we’re in before
  spending Probe-2 submissions.

### Probe 3 — Q3 RESET-after-WIN (score-encoded)
Agent wins L0 on a winnable game, RESETs once, then attempts a clean
replay. If RESET mints a fresh play, max-over-plays keeps the best replay →
score reflects two attempts (and a deliberately-better second play raises
it). If RESET is a level reset only, the single play’s counter continues
and the encoded second-attempt improvement cannot appear. Encode the
observed branch into a banded score. (Local mirror already says “mints” for
arc_agi-0.9.8; this confirms the gateway matches.)

### Probe 4 — Q4 wall-clock
External fact: **6 hours**, now corroborated by two independent mirrors of
the Kaggle overview (supersedes FORGE’s 8h guard guess and our 9h
assumption — both were unsourced). Optional confirmation: an agent that
wins L0 frugally early (banks score), then idles in a timed no-op loop and
every ~30 min nudges its winning margin on a sacrificial game so the final
aggregate encodes the elapsed-time bucket it reached before the kill — only
worth it if a submission is already being spent on Probe 2/3.

### Probe 5 — Q5 null-coordinate (free rider)
Fold into Probe 1 (if logs) or piggyback on a Probe-2 submission: send
ACTION6 with {x:0,y:0} and with data omitted on a few games; if logs,
print the response + scored flag; if score-only, this is observational
(note whether it ever advances/over-counts) and mainly a validity/legality
note, low priority.

## Recommended order (minimize submissions)
1. **Probe 0 sentinel** — one submission decides the whole strategy and
   reads the mounted game count.
2. If Channel 1 open → **Probe 1** (one submission, answers Q1,Q2,Q3,Q5).
3. If closed → **Probe 2 control + probe** (two submissions; 2A if a public
   game is mounted, else 2B) for Q1 and as-much-of-Q2-as-possible, then
   **Probe 3** for Q3.
4. Q4 = 6h (documented); confirm opportunistically only.
5. Optional: human runs the **Channel-4 online proxy** with an ARC key as a
   free pre-check before spending any submission.

## Implementation status (built + smoke-tested this session)

`kaggle_probe/probe_agent.py` implements the merged Probe 0+1 as a neutral
`Diagnostician` (base-class-independent) + an official-contract `MyAgent`
wrapper. `scripts/smoke_probe.py` drives it against the in-process
`arc_agi` engine (NOT the gateway — local engine has no cutoff, so this is
plumbing validation only). Verified on the local engine:

- **Cutoff trace (Q1/Q2 detector):** repeats action-1; on each GAME_OVER,
  RESETs and keeps accumulating level-0 actions. sp80 dies every 31 actions
  (its own rule) yet the probe sails to the 300-action smoke cap with
  `stuck_terminal=None` — the correct **"no cutoff"** signature. On the real
  gateway a cutoff would instead show as `stuck_terminal` set (a RESET that
  won't clear the terminal), actions refused, or accumulation stalling near
  5×baseline. The full `(index, state, levels)` trace + `game_over_indices`
  + `max_level0_reached` are printed per game.
- **Null-coord (Q5):** fires at action index 2 on click games (lp85, su15,
  lf52), recording advance/state — runs independent of winning.
- **Mint (Q3):** on tt01, RESET-after-WIN returns `full_reset=True,
  levels_completed=0` → MINT confirmed (matches arc_agi-0.9.8).

**Known limitation (flagged):** the diagnostic does not try to WIN, so its
Q3 mint test only fires on games it wins incidentally by repeating action-1
(tt01 does; most hidden games won't). Q3 at the gateway therefore needs
either (a) reliance on the strong local baseline (mint works in our
package; only the docs' "Game Resets become Level Resets" sentence is in
doubt), or (b) a follow-on probe that reuses the real winseeker on one
game to reach a WIN, then runs the mint test. Q1/Q2 — the
architecture-deciding questions — do not need a win and are fully exercised
by the cutoff trace.

## What I verified vs what needs the human
- Verified locally: gateway is a separate container; pip `arc_agi` has no
  cutoff (so Q1 needs the live gateway); competition-mode `get_scorecard`
  is 403 (no inflight read) but `close_scorecard` returns the full card and
  `main.py` prints it; RESET-after-WIN mints locally (Q3 baseline).
- Needs the human: every probe that submits (Probe 0/1/2/3); obtaining an
  ARC API key for the Channel-4 proxy. No monetary spend; each submission
  costs one daily slot.
- Open externally-unconfirmable-from-here: Channel-1 log visibility (Probe
  0 settles it); whether the gateway mounts public games (Probe 0 reads it).
