# Template-rule baseline agent (Milestone 1)

A small, fully open baseline for the ARC-AGI-3 track. One self-contained
notebook; no external models, no network access, CC0/MIT-0.

## What it does

The agent plays each game once, within a fixed per-game time budget:

1. **Observes**: every (frame, action, next-frame) it experiences goes into
   a deduplicated, memory-bounded observation store.
2. **Detects status regions**: cells that repeatedly change on their own
   (independent of where you click) behave like score/status displays;
   they are modeled separately so they don't block rule fitting on the
   actual game content.
3. **Fits small rules**: parameterized templates fit by enumeration —
   "this action moves color c by (dy,dx)", "moving onto color t ends the
   level", "clicking a cell of color a recolors it to b", per-level event
   regularities, and simple state-table models of the status regions.
   Every rule is then checked exactly against the full observation store.
4. **Plans**: when the verified rules support it, a short forward search
   finds an action sequence to the next level; otherwise the agent
   explores (untried actions first, in a loop-avoiding rotation).

## Honest expectations

This is a baseline: it scores when a game's mechanics happen to fall
inside the template family above, and explores blindly otherwise. Scores,
per-game results, and the agent's own observation counts are printed by
the notebook. Reference baselines (uniform random, action-sweep) are
included for comparison.

## Layout

Everything is in one notebook file: observation store, region detection,
rule templates + verification, planner, agent loop, baselines. Runs
offline against the competition runtime wheels.
