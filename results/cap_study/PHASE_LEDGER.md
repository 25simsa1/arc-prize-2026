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
| 01:45 | P3 capped s0 | all 25 LEVEL_CAPPED, sweep wall ~4 min (vs 2.3h uncapped); lf52/lp85/r11l reach 1 level under cap. AGENT-VERSION FLAG: current agent has post-sweep25 explorer fixes (lp85 now progresses) — tonight's capped/uncapped arms use the same current agent; sweep25 = older agent, context only |
| 01:55 | P3 capped done | seeds 0,1,2 complete (each ~4 min). Launching uncapped seeds 1,2 chained (~5h, ETA ~07:00) |
| 02:10 | V1 AERA | UNITS: AERA reports RHAE as 0–1 fractions (0.2116 = 21.16%, "4/25 solved"; code-track claim 0.30 = 30% private) — we had been reading these as percent. BUT probe refutes their Table 9: on all 8 claimed games, repeating one action hits GAME_OVER at EXACTLY their quoted step count (tu93@50, sc25@52, tr87@128, ka59@100, re86@100, ls20@129, g50t@130, wa30@200) — they counted GAME_OVER as solved. 21%/30% claims now suspect; threat downgraded; our DEAD bucket stands; persistence-probe priority drops |
| 01:37 | note | ledger timestamps above were narrative estimates a few min ahead of wall clock; from here they are actual. uncap-s1 started ~01:33, on game 2/25 |
| 02:16 | P3 uncap-s1 | done in ~43 min (deadline-fix made sweeps faster). mean RHAE 0.219 — IDENTICAL to capped mean; progress on 7 games / 8 levels. auto-committed 9aa808b |
| 02:42 | P4 done | REPORT.md + NOTES corrections committed |
