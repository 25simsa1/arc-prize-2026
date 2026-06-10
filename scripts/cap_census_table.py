"""Cross-run census aggregation for the cap study: per run tag, recompute
the three paper censuses (evidence starvation, non-Markov conflict
signature, mean RHAE) + outcome counts, then report mean +/- range across
seeds per arm (capped / uncapped).

    .venv/bin/python scripts/cap_census_table.py \
        --capped cap5-s0 cap5-s1 cap5-s2 --uncapped sweep25 uncap-s1 uncap-s2

Reads results/<tag>/summary.json + runs/wm/<tag>/*-trajectories.json.
Writes results/cap_study/census_table.json.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFLICT_R3 = 0.002  # same rule as triage_sweep.py


def run_census(tag: str) -> dict:
    summary = json.loads((ROOT / "results" / tag / "summary.json").read_text())
    raw = ROOT / "runs/wm" / tag
    games = {}
    for led in summary["ledger"]:
        g = led["game"]
        traj = raw / f"{g}-trajectories.json"
        rep = json.loads(traj.read_text())["report"] if traj.exists() else {}
        store = rep.get("store", {})
        n = max(store.get("transitions", 0) + store.get("evicted", 0), 1)
        census = rep.get("event_census", {})
        diag = led.get("diagnostics", {})
        games[g] = {
            "rhae": led["best_rhae"],
            "levels": led["levels"],
            "status": led["status"],
            "starved": census.get("LEVEL", 0) == 0 and census.get("WIN", 0) == 0,
            "conflict_rate": store.get("conflicts", 0) / n,
            "transitions": store.get("transitions", 0),
            "actions": led["match"]["total_actions"],
        }
    n_games = len(games)
    return {
        "tag": tag,
        "mean_rhae": summary["mean_rhae_over_games_run"],
        "n_games": n_games,
        "starved_n": sum(1 for v in games.values() if v["starved"]),
        "starved_games": sorted(g for g, v in games.items() if v["starved"]),
        "nonmarkov_n": sum(1 for v in games.values()
                           if v["conflict_rate"] >= CONFLICT_R3),
        "nonmarkov_games": sorted(g for g, v in games.items()
                                  if v["conflict_rate"] >= CONFLICT_R3),
        "won_n": sum(1 for v in games.values() if v["status"].startswith("WIN")),
        "levels_total": sum(v["levels"] for v in games.values()),
        "games_with_progress": sorted(g for g, v in games.items() if v["levels"] > 0),
        "total_actions": sum(v["actions"] for v in games.values()),
        "games": games,
    }


def agg(rows: list[dict], key: str):
    vals = [r[key] for r in rows]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
            "values": vals}


def fmt(a, pct=False, n=None):
    if a is None:
        return "-"
    if n:  # express counts as percent of n
        m, lo, hi = (100 * v / n for v in (a["mean"], a["min"], a["max"]))
        return f"{m:.0f}% [{lo:.0f}–{hi:.0f}]"
    return f"{a['mean']:.3f} [{a['min']:.3f}–{a['max']:.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capped", nargs="*", default=[])
    p.add_argument("--uncapped", nargs="*", default=[])
    args = p.parse_args()

    arms = {"capped": [run_census(t) for t in args.capped],
            "uncapped": [run_census(t) for t in args.uncapped]}
    out = {"arms": {}}
    print(f"{'metric':28s} {'capped (5x)':>24s} {'uncapped':>24s}")
    metrics = [
        ("mean_rhae", "mean RHAE %", False),
        ("starved_n", "evidence-starved games", True),
        ("nonmarkov_n", "non-Markov (conflict sig)", True),
        ("won_n", "games won", True),
        ("levels_total", "levels completed (sum)", False),
        ("total_actions", "actions spent (sum)", False),
    ]
    for key, label, as_pct in metrics:
        row = {}
        for arm, rows in arms.items():
            n = rows[0]["n_games"] if rows else None
            row[arm] = agg(rows, key)
            row[f"{arm}_fmt"] = fmt(row[arm], n=n if as_pct else None)
        print(f"{label:28s} {row['capped_fmt']:>24s} {row['uncapped_fmt']:>24s}")
        out["arms"][key] = row
    out["runs"] = {arm: rows for arm, rows in arms.items()}

    # per-seed detail lines
    for arm, rows in arms.items():
        for r in rows:
            print(f"  [{arm}] {r['tag']}: RHAE {r['mean_rhae']:.3f}, "
                  f"starved {r['starved_n']}/{r['n_games']}, "
                  f"nonmarkov {r['nonmarkov_n']}, won {r['won_n']}, "
                  f"levels {r['levels_total']}, "
                  f"progress on {r['games_with_progress']}")

    (ROOT / "results/cap_study/census_table.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nwrote results/cap_study/census_table.json")


if __name__ == "__main__":
    main()
