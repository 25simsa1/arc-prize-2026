"""Acceptance runs: world-model agent on real games, with trajectory logs.

    .venv/bin/python scripts/run_wm.py --games cd82 sb26 --proposer template \
        --time-budget 180 --tag wm-accept
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import RunConfig, run_suite
from harness.agents.wm_agent import WorldModelAgent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--proposer", default="template", choices=["template", "memo"])
    p.add_argument("--time-budget", type=float, default=180.0)
    p.add_argument("--budget", type=int, default=50000)
    p.add_argument("--mode", default="two_phase", choices=["single_play", "two_phase"])
    p.add_argument("--env-dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tag = args.tag or f"wm-{args.proposer}-{args.mode}"
    agents: dict[str, WorldModelAgent] = {}

    def factory(game_id: str, seed: int) -> WorldModelAgent:
        a = WorldModelAgent(
            game_id, seed, proposer=args.proposer, time_budget_s=args.time_budget
        )
        agents[game_id] = a
        return a

    factory.agent_name = f"wm-{args.proposer}"

    cfg = RunConfig(
        max_actions_per_game=args.budget,
        seed=args.seed,
        mode=args.mode,
        tag=tag,
    )
    if args.env_dir:
        cfg.environments_dir = args.env_dir

    record = run_suite(args.games, factory, cfg)

    out_dir = Path("runs/wm")
    print("\n=== per-game detail ===")
    for env in record["scorecard"].get("environments", []):
        gid = env["id"]
        base = gid.split("-")[0]
        agent = agents.get(base) or agents.get(gid)
        runs = [
            {"play": i, "score": round(r["score"], 2), "actions": r["actions"],
             "levels": r["levels_completed"], "state": r["state"]}
            for i, r in enumerate(env.get("runs", []))
        ]
        rep = agent.report() if agent else {}
        print(f"\n{gid}: game_score={round(env['score'], 2)}")
        print("  scorecard runs:", json.dumps(runs))
        if agent:
            print("  agent:", json.dumps({k: rep[k] for k in ("status", "phase_time_s", "wall_s")}))
            print("  plays:", json.dumps(rep["plays"]))
            print("  model:", json.dumps(rep["model"]))
            traj = out_dir / f"{tag}-{base}.json"
            agent.dump_trajectories(traj)
            agent.store.save(out_dir / f"{tag}-{base}-store.pkl")
            print(f"  trajectories: {traj}")


if __name__ == "__main__":
    main()
