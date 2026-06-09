"""Measurement layer: JSONL events during a run + summary artifacts.

Honest accounting rule (NOTES.md, Workstream B): a prediction-match figure
is only meaningful next to how much was predicted at all. HonestMatch is the
ONLY way this module emits match quality — always the full triple
(matched, predicted_steps, total_actions); there is deliberately no API that
yields a bare percentage.

Event stream (events.jsonl, gzipped at close):
  coverage — after every store update and after every rule refresh
  plan     — one per adopted plan, on retirement (executed vs planned)
  phase    — per-game wall-clock split, emitted at game end
  outcome  — per-game ledger row
"""

import gzip
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class HonestMatch:
    matched: int
    predicted_steps: int
    total_actions: int

    def as_dict(self) -> dict[str, int]:
        return {
            "matched": self.matched,
            "predicted_steps": self.predicted_steps,
            "total_actions": self.total_actions,
        }

    def __str__(self) -> str:
        pct = (100.0 * self.matched / self.predicted_steps) if self.predicted_steps else 0.0
        return (
            f"{self.matched}/{self.predicted_steps} predicted-steps matched "
            f"({pct:.1f}%) over {self.total_actions} actions"
        )


PHASE_BUCKETS = (
    "exploration", "proposing", "verifying", "planning", "executing", "env_stepping",
)


class MetricsLogger:
    def __init__(self, results_dir: str | Path, run_meta: Optional[dict] = None) -> None:
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "events.jsonl"
        self._fh = open(self._path, "w")
        self._t0 = time.monotonic()
        self.emit("run_meta", **(run_meta or {}))

    def emit(self, event: str, **fields: Any) -> None:
        rec = {"e": event, "t": round(time.monotonic() - self._t0, 3)}
        rec.update(fields)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def coverage(
        self,
        game: str,
        play: int,
        action_index: int,
        transitions_stored: int,
        grid_predicted_frac: float,
        grid_exact_rate: float,
        event_predicted_frac: float,
        event_exact_rate: float,
        n_verified: int,
        n_contradicted: int,
        n_untested: int,
        trigger: str,
        hud_predicted_frac: float = 0.0,
        hud_exact_rate: float = 0.0,
    ) -> None:
        self.emit(
            "coverage",
            game=game, play=play, a=action_index, n=transitions_stored,
            gp=round(grid_predicted_frac, 4), gx=round(grid_exact_rate, 4),
            ep=round(event_predicted_frac, 4), ex=round(event_exact_rate, 4),
            hp=round(hud_predicted_frac, 4), hx=round(hud_exact_rate, 4),
            rv=n_verified, rc=n_contradicted, ru=n_untested, trig=trigger,
        )

    def plan(self, game: str, play: int, planned_len: int, executed_len: int,
             confidence: float, retired_by: str, planner_calls: int) -> None:
        self.emit("plan", game=game, play=play, planned=planned_len,
                  executed=executed_len, conf=round(confidence, 3),
                  retired_by=retired_by, planner_calls=planner_calls)

    def phase(self, game: str, buckets: dict[str, float], actions: int,
              plays: int, replans: int,
              replan_triggers: Optional[dict[str, int]] = None) -> None:
        assert set(buckets) <= set(PHASE_BUCKETS), f"unknown phase bucket in {buckets}"
        self.emit("phase", game=game,
                  **{k: round(buckets.get(k, 0.0), 2) for k in PHASE_BUCKETS},
                  actions=actions, plays=plays, replans=replans,
                  replan_triggers=replan_triggers or {})

    def outcome(self, game: str, status: str, best_rhae: float, levels: int,
                win_levels: int, match: HonestMatch, diagnostics: dict) -> None:
        self.emit("outcome", game=game, status=status, best_rhae=round(best_rhae, 2),
                  levels=levels, win_levels=win_levels, match=match.as_dict(),
                  diagnostics=diagnostics)

    def close(self, summary: Optional[dict] = None) -> Path:
        self._fh.close()
        with open(self._path, "rb") as src, gzip.open(f"{self._path}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        self._path.unlink()
        if summary is not None:
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return self.dir


def read_events(results_dir: str | Path) -> list[dict]:
    p = Path(results_dir)
    gz, plain = p / "events.jsonl.gz", p / "events.jsonl"
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return [json.loads(line) for line in f]
    with open(plain) as f:
        return [json.loads(line) for line in f]
