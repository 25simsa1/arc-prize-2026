"""Suite runner: drive agents over environments in-process, score with the
shipped scorecard, verify with the independent RHAE recomputation, and dump
a machine-readable run record under runs/.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction, GameState

from . import rhae
from .agents.base import Agent

DEFAULT_ENV_DIR = str(
    Path.home()
    / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files"
)


@dataclass
class RunConfig:
    max_actions_per_game: int = 300  # scored actions; budget, not a rule
    seed: int = 0
    environments_dir: str = DEFAULT_ENV_DIR
    recordings_dir: str = "recordings"
    save_recording: bool = False
    runs_dir: str = "runs"
    tag: str = "dev"
    quiet: bool = False
    # Play semantics (see NOTES.md "Play semantics"):
    #   single_play — stop at first WIN.
    #   two_phase  — after a WIN, RESET (engine full-resets => fresh play,
    #                uncounted) and replay until the budget runs out; the
    #                scorecard takes the best play. Competition-legal: new
    #                plays are only mintable via WIN, which this respects.
    mode: str = "single_play"  # "single_play" | "two_phase"
    # Eval-policy model of the ARC-AGI-3 technical report's per-level action
    # cutoff (arXiv 2603.24621, "Leaderboards"): "we set a hard cutoff of 5x
    # human performance per level. If a human takes 10 actions to beat a
    # certain level on average, then we will cut the AI agent off after 50
    # actions." None = off (pre-cap behavior, bit-for-bit). 5.0 = eval policy.
    # PROVISIONAL semantics — see level_action_cap() — until the Kaggle
    # gateway probe confirms actual enforcement and mechanics.
    per_level_action_cap_multiplier: Optional[float] = None


@dataclass
class GameResult:
    game_id: str
    scored_actions: int
    levels_completed: int
    win_levels: int
    state: str
    wall_seconds: float
    plays: int = 1
    env_step_s: float = 0.0  # wall-clock inside env.step()/env.reset()
    error: Optional[str] = None
    capped_at_level: Optional[int] = None  # level index where the cutoff fired
    capped_play: Optional[int] = None      # 0-based play index of the cutoff


def level_action_cap(
    baselines: Optional[list[int]],
    level_index: int,
    multiplier: Optional[float],
) -> Optional[int]:
    """Action cap for one level, or None for "no cap".

    PROVISIONAL local interpretation of the eval-time cutoff (tech report
    2603.24621, Leaderboards: cut off after multiplier x the level's human
    baseline actions), pending the Kaggle gateway probe. Assumptions, each
    flagged for the probe:
      1. The counter is SCORED actions attributed to the current level,
         cumulative across GAME_OVER level-resets within a play (mirrors
         the shipped scorecard's per-level attribution).
      2. Completing the level ON the cap-th action is allowed ("cut off
         AFTER 5x" => the cutoff fires on the cap-th non-completing action).
      3. A cutoff ends the run for that game: levels cannot be skipped, so
         being cut off on a level means no further progress is possible.
         In two_phase mode this also ends the play; earlier completed plays
         keep their banked scores (max-over-plays is unaffected).
      4. Replay plays get fresh per-level counters (per-play isolation,
         mirroring the scorecard's per-play accounting).
      5. Levels with no usable baseline (missing list, short list, or
         baseline <= 0) are uncapped.
    """
    if multiplier is None or not baselines:
        return None
    if level_index < 0 or level_index >= len(baselines):
        return None
    b = baselines[level_index]
    if b is None or b <= 0:
        return None
    return int(multiplier * b)


def _is_scored(action: GameAction, fd) -> bool:
    # Mirrors Scorecard.update_scorecard: ids 1-7 always count; RESET counts
    # only when the engine did NOT treat it as a full reset (new play).
    if action == GameAction.RESET:
        return not getattr(fd, "full_reset", False)
    return True


def run_game(
    arcade: Arcade,
    card_id: str,
    agent: Agent,
    game_id: str,
    cfg: RunConfig,
    baselines: Optional[list[int]] = None,
) -> GameResult:
    t0 = time.time()
    env = arcade.make(
        game_id,
        seed=cfg.seed,
        scorecard_id=card_id,
        save_recording=cfg.save_recording,
        include_frame_data=False,
    )
    if env is None:
        return GameResult(game_id, 0, 0, 0, "MAKE_FAILED", 0.0, error="make() failed")

    fd = env.observation_space  # make() already performed the initial RESET
    frames = [fd]
    scored = 0
    plays = 1
    best_levels = 0
    env_step_s = 0.0
    error = None
    agent.on_play_start(0)

    # Per-level cutoff state (see level_action_cap for the PROVISIONAL
    # semantics). cur_level is the level being played (0-based, == the
    # play's levels_completed); level_actions counts scored actions
    # attributed to it, cumulative across GAME_OVER level-resets.
    cap_mult = cfg.per_level_action_cap_multiplier
    cur_level = fd.levels_completed if fd is not None else 0
    level_actions = 0
    capped_at_level: Optional[int] = None
    capped_play: Optional[int] = None

    while (
        fd is not None
        and scored < cfg.max_actions_per_game
        and not agent.is_done(frames, fd)
    ):
        if fd.state == GameState.WIN:
            best_levels = max(best_levels, fd.levels_completed)
            if cfg.mode != "two_phase":
                break
            # WIN => engine full-resets on RESET: fresh play, not counted.
            # Best play wins the game score, so replaying is score-free.
            t_env = time.time()
            fd = env.reset()
            env_step_s += time.time() - t_env
            if fd is None:
                error = "reset() after WIN returned None"
                break
            frames = [fd]
            plays += 1
            agent.on_play_start(plays - 1)
            cur_level = fd.levels_completed  # fresh play, fresh counters
            level_actions = 0
            continue

        action, data = agent.choose_action(frames, fd)
        t_env = time.time()
        nxt = env.reset() if action == GameAction.RESET else env.step(action, data)
        env_step_s += time.time() - t_env
        if nxt is None:
            error = f"step({action.name}) returned None"
            break
        counted = _is_scored(action, nxt)
        if counted:
            scored += 1
        if nxt.levels_completed > cur_level:
            # the completing action belongs to the finished level; the new
            # level starts with a fresh counter (cap never fires on advance)
            cur_level = nxt.levels_completed
            level_actions = 0
        elif counted:
            level_actions += 1
            cap = level_action_cap(baselines, cur_level, cap_mult)
            if cap is not None and level_actions >= cap:
                capped_at_level = cur_level
                capped_play = plays - 1
                fd = nxt
                frames.append(fd)
                break
        fd = nxt
        frames.append(fd)

    if fd is not None:
        best_levels = max(best_levels, fd.levels_completed)

    return GameResult(
        game_id=game_id,
        scored_actions=scored,
        levels_completed=best_levels,
        win_levels=fd.win_levels if fd is not None else 0,
        state="LEVEL_CAPPED" if capped_at_level is not None
        else (fd.state.name if fd is not None else "UNKNOWN"),
        wall_seconds=round(time.time() - t0, 2),
        plays=plays,
        env_step_s=round(env_step_s, 2),
        error=error,
        capped_at_level=capped_at_level,
        capped_play=capped_play,
    )


def run_suite(
    games: list[str],
    agent_factory: Callable[[str, int], Agent],
    cfg: Optional[RunConfig] = None,
) -> dict[str, Any]:
    cfg = cfg or RunConfig()
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=cfg.environments_dir,
        recordings_dir=cfg.recordings_dir,
    )
    baselines = {
        e.game_id: e.baseline_actions or [] for e in arcade.get_environments()
    }
    # games may be passed as short ids ("ls20") while environments carry a
    # version suffix ("ls20-9607627b"): index baselines under both
    for gid in list(baselines):
        baselines.setdefault(gid.split("-")[0], baselines[gid])

    card_id = arcade.open_scorecard(tags=[cfg.tag])
    results: list[GameResult] = []
    for game_id in games:
        agent = agent_factory(game_id, cfg.seed)
        res = run_game(arcade, card_id, agent, game_id, cfg,
                       baselines=baselines.get(game_id))
        results.append(res)
        if not cfg.quiet:
            print(
                f"  {game_id:6s} {res.state:12s} levels={res.levels_completed}/{res.win_levels}"
                f" actions={res.scored_actions:4d} plays={res.plays}"
                f" t={res.wall_seconds:6.2f}s"
                + (f" ERROR={res.error}" if res.error else "")
            )

    sc = arcade.close_scorecard(card_id)
    sc_data = json.loads(sc.model_dump_json()) if sc is not None else {}

    # Independent RHAE verification of every reported run.
    verification = []
    for env_list in sc_data.get("environments", []):
        env_id = env_list["id"]
        for run in env_list.get("runs", []):
            ok, reported, recomputed = rhae.verify_run(run, baselines.get(env_id))
            verification.append(
                {"game": env_id, "ok": ok, "reported": reported, "recomputed": recomputed}
            )

    game_scores = {
        e["id"]: max((r.get("score", 0.0) for r in e.get("runs", [])), default=0.0)
        for e in sc_data.get("environments", [])
    }
    mean_score = sum(game_scores.values()) / len(game_scores) if game_scores else 0.0

    record = {
        "tag": cfg.tag,
        "agent": getattr(agent_factory, "agent_name", "unknown"),
        "config": {
            "max_actions_per_game": cfg.max_actions_per_game,
            "seed": cfg.seed,
            "games": games,
            "mode": cfg.mode,
            "per_level_action_cap_multiplier": cfg.per_level_action_cap_multiplier,
        },
        "mean_score_over_games_run": mean_score,
        "game_scores": game_scores,
        "results": [vars(r) for r in results],
        "rhae_verification": verification,
        "scorecard": sc_data,
    }

    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.tag}.json"
    out.write_text(json.dumps(record, indent=2))

    if not cfg.quiet:
        bad = [v for v in verification if not v["ok"]]
        print(f"\nMean RHAE over {len(game_scores)} game(s): {mean_score:.2f}%")
        print(
            f"RHAE verification: {len(verification) - len(bad)}/{len(verification)} runs match"
            + (f" — MISMATCHES: {bad}" if bad else "")
        )
        print(f"Run record: {out}")

    record["run_file"] = str(out)
    return record
