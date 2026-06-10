# results/gateway_probe/ — drop zone for returned probe artifacts

Put whatever the Kaggle sitting produced here, then run
`scripts/analyze_gateway_probe.py`:

- `log.txt` — raw rerun/notebook log text (if owner-visible). Parsed for
  `PROBE_SENTINEL_7Qf3xR2 SUMMARY {...}` lines.
- `scorecard.json` — the closed scorecard, if exposed.
- `score.txt` — the leaderboard aggregate (free text ok).
- `control_score.txt` — optional, the frugal-control aggregate for the
  score-encoding comparison.

The analyzer writes `verdicts.json` (Q1–Q5, each CONFIRMED/REFUTED/
INCONCLUSIVE with evidence). See `results/cap_study/gateway_probe_design.md`
for the channel analysis and probe designs.
