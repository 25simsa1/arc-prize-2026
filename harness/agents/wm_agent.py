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
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from arcengine import FrameDataRaw, GameAction, GameState

from ..wm.llm_proposer import LLMProposer
from ..wm.metrics import HonestMatch, MetricsLogger
from ..wm.planner import Plan, plan_to_next_level
from ..wm.proposers import DiffMemorizer, TemplateProposer
from ..wm.regions import RegionAnalyzer
from ..wm.rules import Prediction, RuleStatus, WorldModel, grids_match
from ..wm.store import TransitionStore, canon_frame, frame_hash, masked_hash
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
        propose_time_s: float = 3.0,    # hard per-call template-fitting budget
        dev_mode: bool = True,          # off-menu action: raise (dev) / clamp+log (run)
        metrics: Optional[MetricsLogger] = None,
        region_factoring: bool = True,  # R1; False = pre-R1 behavior (ablation)
        store_cap: int = 150_000,       # memory rail; see TransitionStore
        llm: Optional[dict] = None,     # LLMProposer kwargs; None = templates only
    ) -> None:
        super().__init__(game_id, seed)
        self.proposer = TemplateProposer() if proposer == "template" else DiffMemorizer()
        self.proposer_name = proposer
        self.llm = LLMProposer(**llm) if llm else None
        self._llm_rules: list = []  # persistent across refreshes (expensive)
        self.store = TransitionStore(game_id=game_id, max_transitions=store_cap)
        self.model = WorldModel()
        self.winseeker = WinSeeker(seed=seed)
        self.time_budget_s = time_budget_s
        self.bailout_frac = bailout_frac
        self.repropose_every = repropose_every
        self.max_replays = max_replays
        self.plan_time_s = plan_time_s
        self.verify_time_s = verify_time_s
        self.propose_time_s = propose_time_s
        self.dev_mode = dev_mode

        self.metrics = metrics
        self.region_factoring = region_factoring
        self.analyzer = RegionAnalyzer() if region_factoring else None
        self.t0 = time.monotonic()
        self.deadline = self.t0 + time_budget_s
        self.play_idx = 0
        self.status = "RUNNING"  # RUNNING | WON | ABANDONED | TIMEOUT | SATISFIED
        self._pending: Optional[dict] = None  # last (ctx, action, prediction)
        self._plan: list[str] = []
        self._plan_meta: Optional[Plan] = None
        self._plan_record: Optional[dict] = None
        self._new_since_propose = 0
        self._last_progress_t = self.t0
        self._max_level_seen = 0
        self._satisfied = False
        self._win_seen = False
        # masked-context index: (level, masked_hash) -> {action_key: Transition}
        # — novelty/exploitation bookkeeping that ignores ALWAYS_CHANGING cells
        self._ctx_index: dict[tuple[int, str], dict[str, Any]] = {}
        self._mask_sig: Optional[bytes] = None
        # Part 3: replan-on-trigger + plan cache keyed (level, masked_hash),
        # valid for the current model_version only
        self._replan_needed = True
        self.replan_triggers: Counter = Counter({"play_start": 1})
        self._plan_cache: dict[tuple[int, str], Optional[Plan]] = {}
        self.plan_cache_hits = 0

        self._final_report: Optional[dict] = None  # set by compact()
        self._last_propose_size = -1  # store size at last propose attempt
        # accounting (buckets match metrics.PHASE_BUCKETS minus env_stepping,
        # which the runner owns)
        self.phase_time: dict[str, float] = {
            "exploration": 0.0, "proposing": 0.0, "verifying": 0.0,
            "planning": 0.0, "executing": 0.0,
        }
        self.planner_calls = 0
        self.replans = 0
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

    def _exploration_verdict(self) -> str:
        """Why exploration failed — the per-game ledger that taxonomizes
        DEAD/starved games (sweep censuses: 76% of public games yielded no
        LEVEL/WIN evidence; ft09/lp85 no-op'd for 47k actions). Heuristic
        definitions, stated so the paper table is reproducible:
          no_live_controls  nothing EVER changed the frame, and the click
                            lattice (when clicks exist) was fully swept;
          loop_detected     frame changes exist but novelty stalled while
                            the agent cycled few contexts;
          frontier_exhausted novelty stalled with a broad visited set —
                            the reachable space appears consumed;
          time              still discovering when the budget ended."""
        ws = self.winseeker
        actions = self._cur_play["actions"] or 1
        if not ws.frame_change_seen:
            lattice_done = ws.lattice_total > 0 and ws.lattice_tried >= ws.lattice_total
            no_clicks = ws.lattice_total == 0 and ws._lattice is None
            if lattice_done or no_clicks:
                return "no_live_controls"
            return "time"  # inert so far but the sweep wasn't finished
        if ws.steps_since_new_transition > min(5000, 0.3 * actions):
            if len(self._ctx_index) < 0.01 * actions:
                return "loop_detected"
            return "frontier_exhausted"
        return "time"

    def _finalize_play(self, end_state: str, levels: int) -> None:
        if self._cur_play["ended_at"] is not None:
            return
        self._cur_play["ended_at"] = round(time.monotonic() - self.t0, 2)
        self._cur_play["end_state"] = end_state
        self._cur_play["levels"] = levels
        if end_state in ("BAILOUT", "TIMEOUT", "BUDGET_EXHAUSTED"):
            self._cur_play["exploration_verdict"] = self._exploration_verdict()
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
        status, stored_t = self.store.add(
            level=pend["level"],
            pre=pend["pre"],
            action_key=pend["action_key"],
            post=post,
            post_level=latest.levels_completed,
            post_state=latest.state.name,
            play_index=self.play_idx,
        )
        if status == "new" and stored_t is not None:
            self._new_since_propose += 1
            self.model.observe_for_coverage(stored_t)
            if self.analyzer is not None:
                self.analyzer.observe(stored_t)
            mkey = masked_hash(stored_t.pre, self.model.hud_mask)
            self._ctx_index.setdefault(
                (stored_t.level, mkey), {}
            )[stored_t.action_key] = stored_t
            self.winseeker.observe(stored_t, f"{stored_t.level}:{mkey}",
                                   self.model.hud_mask, is_new=True)
            self._emit_coverage("store")
        elif stored_t is not None:
            self.winseeker.observe(
                stored_t,
                f"{stored_t.level}:{masked_hash(stored_t.pre, self.model.hud_mask)}",
                self.model.hud_mask, is_new=False)
        if status == "conflict":
            # determinism violation: model trust is suspect; force re-propose
            self._new_since_propose = self.repropose_every
            self._emit_coverage("conflict")

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
        if pred is not None and (
            pred.grid is not None or pred.event is not None or pred.hud_grid is not None
        ):
            self._cur_play["predicted_steps"] += 1
            match = True
            if pred.grid is not None and not grids_match(
                Prediction(grid=pred.grid, mask=pred.grid_mask), post
            ):
                match = False
            if pred.hud_grid is not None and not grids_match(
                Prediction(grid=pred.hud_grid, mask=pred.hud_mask), post
            ):
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
                self._retire_plan("miss")  # model was wrong here: replan from reality
                self._need_replan("miss")
                self._new_since_propose = self.repropose_every
        self._cur_play["steps"].append(entry)

        if latest.levels_completed > self._max_level_seen:
            self._max_level_seen = latest.levels_completed
            self._last_progress_t = time.monotonic()
            self._retire_plan("level_up")  # next level: fresh layout, fresh plan
            self._need_replan("level_up")

    # ----------------------------------------------------- metrics plumbing

    def _need_replan(self, reason: str) -> None:
        self._replan_needed = True
        self.replan_triggers[reason] += 1

    def _llm_step(self) -> list:
        """Propose/repair via the LLM when due. Honors the modeling-time cap
        through the caller (_refresh_model); its own min-interval keeps LLM
        seconds bounded — under the scoring rule thinking is free, but time
        is the fleet constraint (110 games / 9h)."""
        from ..wm.rules import RuleStatus as St

        if self.llm.due(self.store):
            fresh = self.llm.propose(self.store, self.model, self.game_id)
            self._llm_rules.extend(fresh)
            # gated repair for contradicted LLM rules (max 2 per cycle)
            repaired = 0
            for r in list(self._llm_rules):
                if repaired >= 2 or r.status != St.CONTRADICTED:
                    continue
                misses = self._collect_misses(r, cap=6)
                fixed = self.llm.repair(r, misses, self.store, self.model,
                                        self.game_id)
                if fixed is not None:
                    self._llm_rules.remove(r)
                    self._llm_rules.append(fixed)
                    repaired += 1
            if self.metrics is not None:
                self.metrics.emit("llm", game=self.game_id,
                                  **self.llm.stats.as_dict())
        return list(self._llm_rules)

    def _collect_misses(self, rule, cap: int = 6) -> list:
        out = []
        for t in self.store.all():
            p = rule.predict(t.level, t.pre, t.action_key)
            if p is None:
                continue
            ok = grids_match(p, t.post) and (p.event is None or p.event == t.event)
            if not ok:
                out.append(t)
                if len(out) >= cap:
                    break
        return out

    def _emit_coverage(self, trigger: str) -> None:
        if self.metrics is None:
            return
        counts = self.model.status_counts()
        self.metrics.coverage(
            game=self.game_id,
            play=self.play_idx,
            action_index=self._cur_play["actions"],
            transitions_stored=len(self.store),
            grid_predicted_frac=self.model.coverage_predicted,
            grid_exact_rate=self.model.coverage_exact,
            event_predicted_frac=self.model.event_predicted,
            event_exact_rate=self.model.event_exact,
            n_verified=counts["VERIFIED"],
            n_contradicted=counts["CONTRADICTED"],
            n_untested=counts["UNTESTED"],
            trigger=trigger,
            hud_predicted_frac=self.model.hud_predicted,
            hud_exact_rate=self.model.hud_exact,
        )

    def _retire_plan(self, reason: str) -> None:
        if self._plan_record is not None:
            self.replans += 1
            if self.metrics is not None:
                self.metrics.plan(
                    game=self.game_id,
                    play=self.play_idx,
                    planned_len=self._plan_record["planned"],
                    executed_len=self._plan_record["executed"],
                    confidence=self._plan_record["confidence"],
                    retired_by=reason,
                    planner_calls=self.planner_calls,
                )
            self._plan_record = None
        self._plan = []

    # ------------------------------------------------------------ lifecycle

    def on_play_start(self, play_index: int) -> None:
        # Runner already observed WIN via is_done (we finalized there).
        self._pending = None
        self._retire_plan("play_start")
        self._need_replan("play_start")
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
        # Adaptive throttle: modeling must never starve acting (observed
        # failure: unthrottled propose consumed 96% of a game's wall-clock).
        # The cap binds FORCED calls too — force bypasses the new-transition
        # counter only, never the time budget. (Learned twice: r11l's
        # zero-rule force-every-step ate 144s/240s; then the bootstrap-force
        # growth threshold re-created it on fast-stepping games where the
        # store grows +10 in milliseconds — re86: 329s wall on a 40s budget.)
        elapsed = max(time.monotonic() - self.t0, 1e-6)
        modeling = self.phase_time["proposing"] + self.phase_time["verifying"]
        if modeling > 0.25 * elapsed:
            return
        if not force and self._new_since_propose < self.repropose_every:
            return
        self._new_since_propose = 0
        self._last_propose_size = len(self.store)

        # R1: refresh the region map BEFORE proposing, so templates scope to
        # the dynamic region. Unmaskable colors come from the current event
        # rules — goal/hazard cells must never be masked away.
        mask_changed = False
        if self.analyzer is not None:
            # CONTRADICTED rules are falsified — letting them name unmaskable
            # colors lets one bogus sampled rule poison the mask forever
            # (observed on cd82: contradicted move_onto rules naming content
            # colors 4/5 stripped the whole meter row from the mask).
            unmaskable = {
                r.params["target"]
                for r in self.model.rules
                if r.name == "move_onto"
                and r.status != RuleStatus.CONTRADICTED
                and r.params.get("event") in ("LEVEL", "WIN", "GAME_OVER")
            }
            region_map = self.analyzer.analyze(unmaskable)
            self.model.region_map = region_map
            new_mask = region_map.hud_mask
            new_sig = new_mask.tobytes() if new_mask is not None else None
            if new_sig != self._mask_sig:
                mask_changed = True
                self._mask_sig = new_sig
                self.model.hud_mask = new_mask
                # context identity changed: rebuild the masked-context index
                self._ctx_index = {}
                for tr in self.store.all():
                    self._ctx_index.setdefault(
                        (tr.level, masked_hash(tr.pre, new_mask)), {}
                    )[tr.action_key] = tr

        t = time.monotonic()
        rules = self.proposer.propose(self.store, self.model,
                                      deadline=t + self.propose_time_s)
        if self.llm is not None:
            rules = rules + self._llm_step()
        self.phase_time["proposing"] += time.monotonic() - t
        t = time.monotonic()
        verify_rules(rules, self.store, deadline=time.monotonic() + self.verify_time_s)
        rule_sig = tuple(sorted((r.rule_id, r.status.value) for r in rules))
        old_sig = tuple(sorted((r.rule_id, r.status.value) for r in self.model.rules))
        self.model.rules = rules
        self.model.recompute_coverage(self.store)
        self.phase_time["verifying"] += time.monotonic() - t
        if rule_sig != old_sig or mask_changed:
            self.model.model_version += 1
            self._plan_cache.clear()  # cache keys are only valid per version
            self._need_replan("model_change")
        self._emit_coverage("rules")

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
        t_choose = time.monotonic()
        self._observe(latest)

        if latest.state == GameState.GAME_OVER:
            # Counted level-restart; free debt in a play we won't score.
            self._retire_plan("game_over")
            self._need_replan("game_over")
            self.winseeker.on_game_over()  # vary the path next attempt
            self.phase_time["exploration"] += time.monotonic() - t_choose
            return GameAction.RESET, None

        grid = canon_frame(latest.frame)
        level = latest.levels_completed
        simple, clicks = self._simple_actions(latest)
        snap = dict(self.phase_time)
        # Bootstrap-force only when there's NEW data to propose from. The
        # unconditional `force=not rules` variant re-proposed every step on
        # games where no template ever fits (r11l in the sweep: 144s of a
        # 240s budget spent proposing, 114 actions taken).
        self._refresh_model(
            force=not self.model.rules
            and len(self.store) - self._last_propose_size >= 10
        )

        action_key: Optional[str] = None
        source = ""

        mkey = (level, masked_hash(grid, self.model.hud_mask))

        if self._plan:
            action_key, source = self._plan.pop(0), "plan"
            if self._plan_record is not None:
                self._plan_record["executed"] += 1
                if not self._plan:
                    self._retire_plan("completed")
                    self._need_replan("plan_exhausted")
        elif self._replan_needed:
            # Part 3: planning runs on triggers (miss / model change / plan
            # exhausted / play boundaries), never every step — every-step
            # replanning cost 48-82s/game in the baseline. Results, including
            # failures, are cached per (context, model_version).
            self._replan_needed = False
            plan = self._plan_cache.get(mkey, "absent")
            if plan != "absent":
                self.plan_cache_hits += 1
            else:
                def click_targets(g: np.ndarray) -> list[str]:
                    if not clicks:
                        return []
                    # salience candidates + clicks already tried in this
                    # masked context (lets the planner retrace stored paths
                    # whose coordinates salience would miss)
                    known = [
                        ak for ak in self._ctx_index.get(
                            (level, masked_hash(g, self.model.hud_mask)), {}
                        )
                        if ak.startswith("ACTION6:")
                    ]
                    return list(dict.fromkeys(salient_clicks(g) + known))

                t = time.monotonic()
                self.planner_calls += 1
                plan = plan_to_next_level(
                    self.model,
                    level,
                    grid,
                    simple,
                    click_targets,
                    deadline=min(time.monotonic() + self.plan_time_s, self.deadline),
                    allow_untested=True,
                )
                self.phase_time["planning"] += time.monotonic() - t
                if not plan.found_goal:
                    plan = None
                self._plan_cache[mkey] = plan
            if plan is not None:
                self._plan_meta = plan
                self._plan = list(plan.actions)
                self._plan_record = {
                    "planned": len(plan.steps),
                    "executed": 1,
                    "confidence": plan.confidence,
                }
                action_key, source = self._plan.pop(0), "plan"
                if not self._plan:
                    self._retire_plan("completed")
                    self._need_replan("plan_exhausted")

        if action_key is None:
            def untried_at(t) -> int:
                # frontier size at a known successor: candidate actions minus
                # tried ones (approximated by the simple-action count; click
                # candidates vary per frame and are counted when visited)
                pk = masked_hash(t.post, self.model.hud_mask)
                tried = self._ctx_index.get((t.post_level, pk), {})
                return max(0, len(simple) + (8 if clicks else 0) - len(tried))

            action_key, source = self.winseeker.choose(
                self._ctx_index.get(mkey, {}), grid, simple, clicks,
                ctx_key=f"{level}:{mkey[1]}", untried_at=untried_at,
                hud_mask=self.model.hud_mask,
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
        # attribute the un-itemized remainder of this call to acting
        spent_in_subphases = sum(self.phase_time.values()) - sum(snap.values())
        remainder = max(0.0, (time.monotonic() - t_choose) - spent_in_subphases)
        bucket = "executing" if source == "plan" else "exploration"
        self.phase_time[bucket] += remainder
        return _to_game_action(action_key)

    # -------------------------------------------------------------- report

    def _llm_report(self) -> Optional[dict]:
        if self.llm is None:
            return None
        from ..wm.rules import RuleStatus as St

        verified = [r for r in self._llm_rules if r.status == St.VERIFIED]
        s = self.llm.stats
        # the Phase-D memo arithmetic, measured live
        tok = s.prompt_tokens + s.output_tokens
        return {
            **s.as_dict(),
            "rules_live": len(self._llm_rules),
            "rules_verified": len(verified),
            "verified_rule_ids": [r.rule_id for r in verified],
            "tokens_per_verified_rule": round(tok / len(verified), 1) if verified else None,
            "seconds_per_verified_rule": round(s.seconds / len(verified), 1) if verified else None,
        }

    def match_accounting(self, plays: Optional[list[dict]] = None) -> HonestMatch:
        """The only exit point for match quality: always the full triple."""
        src = plays if plays is not None else self.plays
        return HonestMatch(
            matched=sum(p["matched_steps"] for p in src),
            predicted_steps=sum(p["predicted_steps"] for p in src),
            total_actions=sum(p["actions"] for p in src),
        )

    def compact(self, trajectories_path: Optional[str | Path] = None) -> None:
        """Snapshot reporting artifacts, then free per-game heavy state.
        A 25-game sweep holding every store concurrently would cost tens of
        GB; after this, report()/match_accounting() keep working from the
        snapshot and per-play counters (step lists are dumped, then dropped)."""
        if self._final_report is not None:
            return
        if trajectories_path is not None:
            self.dump_trajectories(trajectories_path)
        self._final_report = self.report()
        for p in self.plays:
            p["steps"] = []
        self.store.by_key.clear()
        self.store.conflicts = self.store.conflicts[:0]
        self._ctx_index.clear()
        self._plan_cache.clear()
        self.analyzer = None
        self.model.rules = []

    def report(self) -> dict:
        if self._final_report is not None:
            return self._final_report
        # A runner exiting on its ACTION budget (rather than our time budget)
        # never routes through is_done's finalization — close the books here
        # so the ledger can't silently show an empty run.
        if self._cur_play["ended_at"] is None and self._cur_play["actions"] > 0:
            self._finalize_play("BUDGET_EXHAUSTED", self._max_level_seen)
        return {
            "game_id": self.game_id,
            "proposer": self.proposer_name,
            "status": self.status,
            "plays": [
                {k: v for k, v in p.items() if k != "steps"} | {
                    "match": HonestMatch(
                        p["matched_steps"], p["predicted_steps"], p["actions"]
                    ).as_dict()
                }
                for p in self.plays
            ],
            "match": self.match_accounting().as_dict(),
            "model": self.model.summary(),
            "regions": (
                self.model.region_map.as_dict()
                if getattr(self.model, "region_map", None) is not None
                else None
            ),
            "store": {
                "transitions": len(self.store),
                "conflicts": len(self.store.conflicts),
                "evicted": self.store.evicted_total,
                "capped_drops": self.store.capped_drops,
            },
            "event_census": dict(self.store.event_counts),
            "explorer": {
                "frame_change_seen": self.winseeker.frame_change_seen,
                "lattice_tried": self.winseeker.lattice_tried,
                "lattice_total": self.winseeker.lattice_total,
                "steps_since_new": self.winseeker.steps_since_new_transition,
                "distinct_contexts": len(self._ctx_index),
            },
            "phase_time_s": {k: round(v, 2) for k, v in self.phase_time.items()},
            "llm": self._llm_report(),
            "planner_calls": self.planner_calls,
            "plan_cache_hits": self.plan_cache_hits,
            "replans": self.replans,
            "replan_triggers": dict(self.replan_triggers),
            "region_factoring": self.region_factoring,
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
