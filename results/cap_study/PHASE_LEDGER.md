# Cap-study overnight run — phase ledger (2026-06-10)

Driving fact: ARC-AGI-3 tech report (2603.24621, Leaderboards) specifies a
hard eval-time cutoff of 5× human baseline actions per level; absent from
shipped scoring code and our harness. Gateway enforcement UNCONFIRMED
(daytime Kaggle-Starter probe is ground truth). Tonight: capped + uncapped
numbers so both verdicts are covered.

| time (EDT) | phase | event |
|---|---|---|
| 01:23 | setup | start; repo clean at cb1f1f8; read runner.py, run_wm.py; verified sweep25 trajectories carry per-step level+event for all 25 games (audit is computable without reruns) |
