"""Instrumented world-model runs: JSONL metrics, outcome ledger, honest
match accounting (always matched/predicted/actions — never a bare rate).

    .venv/bin/python scripts/run_wm.py --games tt01 cd82 sb26 --proposer template \
        --time-budget 240 --tag baseline-c1

Artifacts: results/<tag>/{events.jsonl.gz, summary.json} (committed) and
runs/wm/<tag>/ trajectories + stores (gitignored, large).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import RunConfig, run_suite
from harness.agents.wm_agent import WorldModelAgent
from harness.wm.metrics import HonestMatch, MetricsLogger


def outcome_status(rep: dict) -> str:
    plays = rep["plays"]
    won_replay = any(p["play"] >= 1 and p["end_state"] == "WIN" for p in plays)
    won_any = any(p["end_state"] == "WIN" for p in plays)
    if won_replay:
        return "WIN_REPLAYED"
    if won_any:
        return "WIN_UNREPLAYED"
    return rep["status"] if rep["status"] in ("ABANDONED", "TIMEOUT") else "ABANDONED"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--proposer", default="template", choices=["template", "memo"])
    p.add_argument("--time-budget", type=float, default=240.0)
    p.add_argument("--budget", type=int, default=50000)
    p.add_argument("--mode", default="two_phase", choices=["single_play", "two_phase"])
    p.add_argument("--env-dir", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-region-factoring", action="store_true",
                   help="R1 ablation switch: pre-factoring behavior")
    p.add_argument("--save-stores", action="store_true",
                   help="persist full transition stores (disk-heavy: ~16KB/transition)")
    args = p.parse_args()

    results_dir = Path("results") / args.tag
    metrics = MetricsLogger(
        results_dir,
        run_meta={
            "tag": args.tag, "proposer": args.proposer, "mode": args.mode,
            "games": args.games, "time_budget_s": args.time_budget,
            "action_budget": args.budget, "seed": args.seed,
        },
    )
    agents: dict[str, WorldModelAgent] = {}
    order: list[str] = []
    raw_dir = Path("runs/wm") / args.tag

    def factory(game_id: str, seed: int) -> WorldModelAgent:
        if order:  # previous game is finished: free its heavy state now
            prev = agents[order[-1]]
            prev.compact(raw_dir / f"{order[-1]}-trajectories.json")
        a = WorldModelAgent(
            game_id, seed, proposer=args.proposer,
            time_budget_s=args.time_budget, metrics=metrics,
            region_factoring=not args.no_region_factoring,
        )
        agents[game_id] = a
        order.append(game_id)
        return a

    factory.agent_name = f"wm-{args.proposer}"

    cfg = RunConfig(
        max_actions_per_game=args.budget, seed=args.seed, mode=args.mode, tag=args.tag,
    )
    if args.env_dir:
        cfg.environments_dir = args.env_dir

    games = args.games
    if games == ["all"]:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode

        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=cfg.environments_dir)
        games = sorted(e.game_id.split("-")[0] for e in arcade.get_environments())

    record = run_suite(games, factory, cfg)
    if order:  # compact the final game too
        agents[order[-1]].compact(raw_dir / f"{order[-1]}-trajectories.json")

    ledger: list[dict] = []
    env_step_by_game = {r["game_id"]: r["env_step_s"] for r in record["results"]}

    for env in record["scorecard"].get("environments", []):
        gid = env["id"]
        base = gid.split("-")[0]
        agent = agents.get(base) or agents.get(gid)
        if agent is None:
            continue
        rep = agent.report()
        runs = env.get("runs", [])
        best_rhae = max((r.get("score", 0.0) for r in runs), default=0.0)
        levels = max((r.get("levels_completed", 0) for r in runs), default=0)
        match = agent.match_accounting()
        status = outcome_status(rep)
        last_play = rep["plays"][-1] if rep["plays"] else {}

        buckets = dict(rep["phase_time_s"])
        buckets["env_stepping"] = env_step_by_game.get(base, 0.0)
        metrics.phase(
            game=base, buckets=buckets,
            actions=sum(p["actions"] for p in rep["plays"]),
            plays=len(rep["plays"]), replans=rep["replans"],
            replan_triggers=rep.get("replan_triggers"),
        )
        metrics.outcome(
            game=base, status=status, best_rhae=best_rhae, levels=levels,
            win_levels=max((len(r.get("level_scores") or []) for r in runs), default=0),
            match=match,
            diagnostics={
                "end_state_last_play": last_play.get("end_state"),
                "store_transitions": rep["store"]["transitions"],
                "store_conflicts": rep["store"]["conflicts"],
                "planner_calls": rep["planner_calls"],
                "model": rep["model"],
            },
        )
        ledger.append({
            "game": base, "status": status, "best_rhae": round(best_rhae, 2),
            "levels": levels, "plays": len(rep["plays"]),
            "match": match.as_dict(), "phase_s": buckets,
        })
        if agent._final_report is None:  # not compacted: dump now
            agent.dump_trajectories(raw_dir / f"{base}-trajectories.json")
        if args.save_stores and len(agent.store):
            agent.store.save(raw_dir / f"{base}-store.pkl")

    summary = {
        "meta": {"tag": args.tag, "proposer": args.proposer, "mode": args.mode,
                 "time_budget_s": args.time_budget, "games": args.games},
        "mean_rhae_over_games_run": record["mean_score_over_games_run"],
        "ledger": ledger,
        "run_record": record["run_file"],
    }
    metrics.close(summary)

    # honest summary table: matched | predicted | actions side by side, always
    print(f"\n=== {args.tag}: outcome ledger ===")
    hdr = f"{'game':6s} {'status':16s} {'RHAE':>6s} {'lvls':>4s} {'plays':>5s} " \
          f"{'matched':>8s} {'predicted':>9s} {'actions':>8s}"
    print(hdr)
    for row in ledger:
        m = row["match"]
        print(
            f"{row['game']:6s} {row['status']:16s} {row['best_rhae']:6.2f} "
            f"{row['levels']:4d} {row['plays']:5d} {m['matched']:8d} "
            f"{m['predicted_steps']:9d} {m['total_actions']:8d}"
        )
        print(f"       {HonestMatch(**m)}")
    print(f"\nresults: {results_dir}/  (events.jsonl.gz, summary.json)")


if __name__ == "__main__":
    main()
