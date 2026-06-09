"""Run a baseline agent through the harness.

    .venv/bin/python scripts/run_baseline.py --games ls20 vc33 ft09 --budget 300
    .venv/bin/python scripts/run_baseline.py --games all --budget 200 --tag random-all25
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import RunConfig, run_suite
from harness.agents import AVAILABLE_AGENTS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random", choices=sorted(AVAILABLE_AGENTS))
    p.add_argument("--games", nargs="+", default=["ls20"])
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--mode", default="single_play", choices=["single_play", "two_phase"])
    p.add_argument("--env-dir", default=None, help="override environments dir (e.g. test_envs)")
    args = p.parse_args()

    cfg = RunConfig(
        max_actions_per_game=args.budget,
        seed=args.seed,
        tag=args.tag or f"{args.agent}-{args.mode}-b{args.budget}",
        mode=args.mode,
    )
    if args.env_dir:
        cfg.environments_dir = args.env_dir

    games = args.games
    if games == ["all"]:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode

        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE, environments_dir=cfg.environments_dir
        )
        games = sorted(e.game_id.split("-")[0] for e in arcade.get_environments())

    agent_cls = AVAILABLE_AGENTS[args.agent]

    def factory(game_id: str, seed: int):
        return agent_cls(game_id, seed)

    factory.agent_name = args.agent

    print(f"Agent={args.agent} budget={args.budget}/game seed={args.seed} games={games}")
    run_suite(games, factory, cfg)


if __name__ == "__main__":
    main()
