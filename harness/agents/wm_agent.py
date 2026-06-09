"""WorldModelAgent: propose -> verify -> plan -> execute, wired to the
win-gated play semantics (NOTES.md "Play semantics").

Play 1: WinSeeker drives (planner consulted first each step); the
TransitionStore and proposers run continuously. On WIN the runner mints a
fresh play (two_phase mode); plays 2+ are planner-driven replays. Any
prediction miss appends the observed transition, forces re-propose, and
replans from the current frame. In single_play mode the same logic
degenerates to interleaved explore/execute — the fallback if Kaggle's
notebook semantics diverge from the local mirror.

Time is the real budget: 110 hidden games / 9h ≈ 4.5-5 min/game average,
and unplayed games score zero — overrunning one game starves another. Every
looping component here takes a deadline.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from arcengine import FrameDataRaw, GameAction, GameState

from ..wm.planner import Plan, plan_to_next_level
from ..wm.proposers import DiffMemorizer, TemplateProposer
from ..wm.rules import WorldModel
from ..wm.store import TransitionStore, canon_frame, frame_hash
from ..wm.verifier import verify_rules
from ..wm.winseeker import WinSeeker, salient_clicks
from .base import Agent

_ID_TO_ACTION = {a.value: a for a in GameAction}
_NAME_TO_ACTION = {a.name: a for a in GameAction}


def _to_game_action(action_key: str) -> tuple[GameAction, Optional[dict[str, Any]]]:
    base, _, coords = action_key.partition(":")
    action = _NAME_TO_ACTION[base]
    if coords:
        x, y = (int(v) for v in coords.split(","))
        return action, {"x": x, "y": y}
    return action, None


class WorldModelAgent(Agent):
    name = "wm"

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        proposer: str = "template",
        time_budget_s: float = 240.0,   # per game; see module docstring
        bailout_frac: float = 0.6,      # no-level-progress bail-out threshold
        repropose_every: int = 25,      # new transitions between re-proposals
        max_replays: int = 4,
        plan_time_s: float = 2.0,
        verify_time_s: float = 2.0,
        dev_mode: bool = True,          # off-menu action: raise (dev) / clamp+log (run)
    ) -> None:
        super().__init__(game_id, seed)
        self.proposer = TemplateProposer() if proposer == "template" else DiffMemorizer()
        self.proposer_name = proposer
        self.store = TransitionStore(game_id=game_id)
        self.model = WorldModel()
        self.winseeker = WinSeeker(seed=seed)
        self.time_budget_s = time_budget_s
        self.bailout_frac = bailout_frac
        self.repropose_every = repropose_every
        self.max_replays = max_replays
        self.plan_time_s = plan_time_s
        self.verify_time_s = verify_time_s
        self.dev_mode = dev_mode

        self.t0 = time.monotonic()
        self.deadline = self.t0 + time_budget_s
        self.play_idx = 0
        self.status = "RUNNING"  # RUNNING | WON | ABANDONED | TIMEOUT | SATISFIED
        self._pending: Optional[dict] = None  # last (ctx, action, prediction)
        self._plan: list[str] = []
        self._plan_meta: Optional[Plan] = None
        self._new_since_propose = 0
        self._last_progress_t = self.t0
        self._max_level_seen = 0
        self._satisfied = False
        self._win_seen = False

        # accounting
        self.phase_time: dict[str, float] = {"propose": 0.0, "verify": 0.0,
                                             "plan": 0.0, "act": 0.0}
        self.plays: list[dict] = []
        self._cur_play = self._new_play_log(0)

    # ------------------------------------------------------------- logging

    def _new_play_log(self, idx: int) -> dict:
        return {
            "play": idx,
            "steps": [],
            "actions": 0,
            "predicted_steps": 0,
            "matched_steps": 0,
            "misses": 0,
            "sources": {},
            "started_at": round(time.monotonic() - self.t0, 2),
            "ended_at": None,
            "end_state": None,
            "levels": 0,
        }

    def _finalize_play(self, end_state: str, levels: int) -> None:
        if self._cur_play["ended_at"] is not None:
            return
        self._cur_play["ended_at"] = round(time.monotonic() - self.t0, 2)
        self._cur_play["end_state"] = end_state
        self._cur_play["levels"] = levels
        self.plays.append(self._cur_play)

    # ---------------------------------------------------------- observation

    def _observe(self, latest: FrameDataRaw) -> None:
        """Record the pending transition. Called from both is_done and
        choose_action (whichever sees the frame first); idempotent."""
        if self._pending is None:
            return
        pend, self._pending = self._pending, None
        post = canon_frame(latest.frame) if latest.frame is not None and len(latest.frame) else None
        if post is None:
            return
        status, _ = self.store.add(
            level=pend["level"],
            pre=pend["pre"],
            action_key=pend["action_key"],
            post=post,
            post_level=latest.levels_completed,
            post_state=latest.state.name,
            play_index=self.play_idx,
        )
        if status == "new":
            self._new_since_propose += 1
        if status == "conflict":
            # determinism violation: model trust is suspect; force re-propose
            self._new_since_propose = self.repropose_every

        entry = {
            "play": self.play_idx,
            "level": pend["level"],
            "action": pend["action_key"],
            "observed": frame_hash(post),
            "event": self.store.lookup(pend["level"], pend["pre_hash"], pend["action_key"]).event
            if self.store.lookup(pend["level"], pend["pre_hash"], pend["action_key"])
            else None,
            "source": pend["source"],
        }
        self._cur_play["actions"] += 1
        self._cur_play["sources"][pend["source"]] = (
            self._cur_play["sources"].get(pend["source"], 0) + 1
        )
        pred = pend.get("prediction")
        if pred is not None and (pred.grid is not None or pred.event is not None):
            self._cur_play["predicted_steps"] += 1
            match = True
            if pred.grid is not None and not np.array_equal(pred.grid, post):
                match = False
            if pred.event is not None:
                observed_event = entry["event"]
                if observed_event is not None and pred.event != observed_event:
                    match = False
            entry["predicted"] = True
            entry["match"] = match
            if match:
                self._cur_play["matched_steps"] += 1
            else:
                self._cur_play["misses"] += 1
                self._plan = []  # model was wrong here: replan from reality
                self._new_since_propose = self.repropose_every
        self._cur_play["steps"].append(entry)

        if latest.levels_completed > self._max_level_seen:
            self._max_level_seen = latest.levels_completed
            self._last_progress_t = time.monotonic()
            self._plan = []  # next level: fresh layout, fresh plan

    # ------------------------------------------------------------ lifecycle

    def on_play_start(self, play_index: int) -> None:
        # Runner already observed WIN via is_done (we finalized there).
        self._pending = None
        self._plan = []
        self.play_idx = play_index
        self._cur_play = self._new_play_log(play_index)
        self._last_progress_t = time.monotonic()
        self._refresh_model(force=True)

    def is_done(self, frames: list[FrameDataRaw], latest: FrameDataRaw) -> bool:
        self._observe(latest)
        now = time.monotonic()

        if latest.state == GameState.WIN:
            self._win_seen = True
            self._finalize_play("WIN", latest.levels_completed)
            last = self.plays[-1]
            replays_done = sum(1 for p in self.plays if p["play"] >= 1)
            # Replay while the model is still learning: a replay with zero
            # prediction misses executed the model-optimal path under current
            # knowledge, so another replay cannot improve it.
            improvable = last["misses"] > 0 or last["play"] == 0
            if (
                now < self.deadline
                and replays_done < self.max_replays
                and improvable
            ):
                return False  # runner mints the next play
            self.status = "WON" if last["play"] == 0 else "SATISFIED"
            return True

        if now > self.deadline:
            self._finalize_play("TIMEOUT", latest.levels_completed)
            self.status = "TIMEOUT" if self._win_seen else "ABANDONED"
            return True

        # bail-out: no level progress within bailout_frac of the time budget
        if (
            not self._win_seen
            and (now - self._last_progress_t) > self.bailout_frac * self.time_budget_s
        ):
            self._finalize_play("BAILOUT", latest.levels_completed)
            self.status = "ABANDONED"
            return True
        return False

    # ------------------------------------------------------------- modeling

    def _refresh_model(self, force: bool = False) -> None:
        if not force and self._new_since_propose < self.repropose_every:
            return
        # Adaptive throttle: modeling must never starve acting (observed
        # failure: unthrottled propose consumed 96% of a game's wall-clock).
        elapsed = max(time.monotonic() - self.t0, 1e-6)
        modeling = self.phase_time["propose"] + self.phase_time["verify"]
        if not force and modeling > 0.25 * elapsed:
            return
        self._new_since_propose = 0
        t = time.monotonic()
        rules = self.proposer.propose(self.store, self.model)
        self.phase_time["propose"] += time.monotonic() - t
        t = time.monotonic()
        verify_rules(rules, self.store, deadline=time.monotonic() + self.verify_time_s)
        self.model.rules = rules
        self.model.recompute_coverage(self.store)
        self.phase_time["verify"] += time.monotonic() - t

    # -------------------------------------------------------------- acting

    def _simple_actions(self, latest: FrameDataRaw) -> tuple[list[str], bool]:
        available = latest.available_actions or []
        simple = [
            _ID_TO_ACTION[a].name
            for a in available
            if a in _ID_TO_ACTION and _ID_TO_ACTION[a] not in (GameAction.RESET, GameAction.ACTION6)
        ]
        clicks = any(a == GameAction.ACTION6.value for a in available)
        return simple, clicks

    def _validated(self, action_key: str, latest: FrameDataRaw, source: str) -> str:
        base = action_key.split(":", 1)[0]
        available = set(latest.available_actions or [])
        base_id = _NAME_TO_ACTION[base].value
        if base_id in available or base == "RESET":
            return action_key
        msg = f"off-menu action {action_key} (available={sorted(available)}, source={source})"
        if self.dev_mode:
            raise AssertionError(msg)
        simple, _ = self._simple_actions(latest)
        fallback = simple[0] if simple else "RESET"
        self._cur_play.setdefault("clamped", []).append(msg)
        return fallback

    def choose_action(
        self, frames: list[FrameDataRaw], latest: FrameDataRaw
    ) -> tuple[GameAction, Optional[dict[str, Any]]]:
        self._observe(latest)

        if latest.state == GameState.GAME_OVER:
            # Counted level-restart; free debt in a play we won't score.
            self._plan = []
            return GameAction.RESET, None

        grid = canon_frame(latest.frame)
        level = latest.levels_completed
        simple, clicks = self._simple_actions(latest)
        self._refresh_model(force=not self.model.rules)

        action_key: Optional[str] = None
        source = ""

        if self._plan:
            action_key, source = self._plan.pop(0), "plan"
        else:
            def click_targets(g: np.ndarray) -> list[str]:
                if not clicks:
                    return []
                # salience candidates + clicks already known to do something
                # at this exact state (lets the planner retrace stored paths
                # whose coordinates salience would miss)
                known = [
                    t.action_key
                    for t in self.store.at_context(level, frame_hash(g))
                    if t.base_action == "ACTION6"
                ]
                return list(dict.fromkeys(salient_clicks(g) + known))

            t = time.monotonic()
            plan = plan_to_next_level(
                self.model,
                level,
                grid,
                simple,
                click_targets,
                deadline=min(time.monotonic() + self.plan_time_s, self.deadline),
                allow_untested=True,
            )
            self.phase_time["plan"] += time.monotonic() - t
            if plan.found_goal:
                self._plan_meta = plan
                self._plan = plan.actions
                action_key, source = self._plan.pop(0), "plan"

        if action_key is None:
            action_key, source = self.winseeker.choose(
                self.store, level, grid, simple, clicks
            )

        action_key = self._validated(action_key, latest, source)
        prediction = self.model.predict(level, grid, action_key)
        self._pending = {
            "level": level,
            "pre": grid,
            "pre_hash": frame_hash(grid),
            "action_key": action_key,
            "prediction": prediction,
            "source": source,
        }
        return _to_game_action(action_key)

    # -------------------------------------------------------------- report

    def report(self) -> dict:
        return {
            "game_id": self.game_id,
            "proposer": self.proposer_name,
            "status": self.status,
            "plays": [
                {k: v for k, v in p.items() if k != "steps"} | {
                    "match_rate": round(p["matched_steps"] / p["predicted_steps"], 3)
                    if p["predicted_steps"]
                    else None
                }
                for p in self.plays
            ],
            "model": self.model.summary(),
            "store": {"transitions": len(self.store), "conflicts": len(self.store.conflicts)},
            "phase_time_s": {k: round(v, 2) for k, v in self.phase_time.items()},
            "wall_s": round(time.monotonic() - self.t0, 2),
        }

    def dump_trajectories(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"report": self.report(), "plays": self.plays}, indent=2))


class WMTemplateAgent(WorldModelAgent):
    name = "wm-template"

    def __init__(self, game_id: str, seed: int = 0) -> None:
        super().__init__(game_id, seed, proposer="template")


class WMMemoAgent(WorldModelAgent):
    name = "wm-memo"

    def __init__(self, game_id: str, seed: int = 0) -> None:
        super().__init__(game_id, seed, proposer="memo")
