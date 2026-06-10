# Cap-study overnight run — phase ledger (2026-06-10)

Driving fact: ARC-AGI-3 tech report (2603.24621, Leaderboards) specifies a
hard eval-time cutoff of 5× human baseline actions per level; absent from
shipped scoring code and our harness. Gateway enforcement UNCONFIRMED
(daytime Kaggle-Starter probe is ground truth). Tonight: capped + uncapped
numbers so both verdicts are covered.

| time (EDT) | phase | event |
|---|---|---|
| 01:23 | setup | start; repo clean at cb1f1f8; read runner.py, run_wm.py; verified sweep25 trajectories carry per-step level+event for all 25 games (audit is computable without reruns) |
| 01:30 | P1 done | cap implemented + 21/21 tests + CLI flag; committed de05b92 (~10 min, 50 min under budget) |
| 01:38 | P2 done | retro audit: cap fires at action 35–442 on 23/25 games; 4 of 6 level-completions vanish (cd82, vc33, sk48, sp80); starvation 76% → 92% eval-realistic; AERA 200-rep probe doesn't fit any game at 5 actions, fits 7/25 at one action; only wins on record (tt01) fit at margin 0.13–0.20. Caveat: truncating an uncapped-regime explorer is a LOWER bound on a cap-aware explorer. |
| 01:38 | P3 plan | seeds adjusted for comparability: existing sweep25 = uncapped seed 0; run capped seeds 0,1,2 then uncapped seeds 1,2. Capped runs expected fast (games end at cap). |
