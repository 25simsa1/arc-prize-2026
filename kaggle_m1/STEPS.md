# Milestone 1 submission — your exact steps

Pre-flight already done on this side: disclosure audit CLEAN (28 forbidden
patterns, see audit.py), notebook compiles, end-to-end tested locally
against the 25 public games in offline mode.

**One veto point before you start:** the public agent includes the
status-region detection (neutral framing, no evidence history). It is part
of what makes the template agent function at all, but it does reveal that
our agent separates UI regions from content. If you want it withheld,
say so BEFORE step 3 and I'll strip it to a plainer agent.

## 1. Read the current M1 instructions (5 min)

Competition page → Overview → look for "Milestone" sections, plus any
pinned discussion thread about Milestone 1 eligibility. Per the rules
research: milestone prizes require the notebook to be **public under an
open-source license by June 30**. If the page asks for a specific
registration/tagging mechanism for M1 and it's ambiguous, STOP and tell me
what it says — don't guess.

## 2. Create the notebook

1. Competition page → Code → New Notebook.
2. Settings: competition GPU not required (the agent is CPU-only) but
   harmless; **Internet OFF**; attach the competition dataset (it provides
   environment_files/ + the runtime wheels).
3. Paste `kaggle_m1/m1_notebook.py` into a single cell.
4. License: CC0 or MIT-0 (competition requires a permissive license for
   milestone eligibility).

## 3. Run + observe (this is also first contact with real eval semantics)

Run all. While it runs, note ANYTHING you see about how evaluation works —
paste me verbatim afterwards:

- how the runtime discovers/serves games inside the notebook (which
  OPERATION_MODE the environment sets, any competition server messages);
- any per-game or total action/time limits enforced by their side;
- what the submission artifact actually is (auto-generated file? scorecard
  id?) and where it appears;
- the exact wall-clock limit shown for the session (9h figure or other);
- anything that looks different from the local runtime's behavior.

If the notebook errors on their template assumptions (e.g. paths), paste
me the first traceback — likely a one-line env fix.

## 4. Submit + make public

1. Save Version (full run) → Submit to Competition.
2. Make the notebook PUBLIC (required for milestone eligibility) with the
   `WRITEUP.md` text as the notebook description — modest and factual, as
   written; don't add roadmap/strategy language.
3. Tell me: the public URL, the leaderboard score, and your eval
   observations. I'll log everything in NOTES.md.
