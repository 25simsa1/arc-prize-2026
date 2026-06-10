"""Triage the 25-game sweep into the four buckets + surprises, and build
the two paper censuses (conflict signature, evidence starvation).

Bucket rules, in precedence order (documented so the table is reproducible):
  SCORED            any play ended WIN (replay machinery then maximizes it).
  WALLED-R3-trans   conflict signature: conflicts/transitions >= 0.2% —
                    cd82-class latent TRANSITION function. Wins precedence
                    over level progress (cd82 itself reaches 2 levels).
  WINNABLE-MAYBE    levels >= 1 with low conflicts — stalled, candidates
                    for the 3x-budget rerun.
  WALLED-R3-win     sb26-class: zero levels, near-zero conflicts, but real
                    model traction (predicted steps >= 2% of actions or a
                    VERIFIED non-event rule) — latent WIN CONDITION.
  DEAD              zero levels, no traction.
Anything contradicting its bucket's spirit lands in SURPRISES instead of
being force-fitted (surprises are design input).

    .venv/bin/python scripts/triage_sweep.py results/sweep25 runs/wm/sweep25
"""

import json
import sys
from pathlib import Path

CONFLICT_R3 = 0.002   # >=0.2% conflicting (level,frame,action) keys
TRACTION_PRED = 0.02  # predicted steps as fraction of actions


def load(results_dir: Path, raw_dir: Path) -> list[dict]:
    summary = json.loads((results_dir / "summary.json").read_text())
    rows = []
    for led in summary["ledger"]:
        game = led["game"]
        traj = raw_dir / f"{game}-trajectories.json"
        rep = json.loads(traj.read_text())["report"] if traj.exists() else {}
        rows.append({"led": led, "rep": rep, "game": game})
    rows.sort(key=lambda r: r["game"])
    return rows, summary


def classify(row: dict) -> tuple[str, str]:
    led, rep = row["led"], row["rep"]
    store = rep.get("store", {})
    n = max(store.get("transitions", 0) + store.get("evicted", 0), 1)
    conflict_rate = store.get("conflicts", 0) / n
    match = led["match"]
    pred_frac = match["predicted_steps"] / max(match["total_actions"], 1)
    model = rep.get("model", {})
    n_verified = model.get("by_status", {}).get("VERIFIED", 0)
    levels = led["levels"]
    note = (f"conf={conflict_rate:.3%} pred={pred_frac:.1%} "
            f"V={n_verified} evict={store.get('evicted', 0)}")

    if led["status"].startswith("WIN"):
        return "SCORED", note
    if conflict_rate >= CONFLICT_R3:
        return "WALLED-R3-trans", note
    if levels >= 1:
        return "WINNABLE-MAYBE", note
    if pred_frac >= TRACTION_PRED or n_verified >= 1:
        return "WALLED-R3-win", note
    return "DEAD", note


def main() -> None:
    results_dir, raw_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows, summary = load(results_dir, raw_dir)

    buckets: dict[str, list] = {}
    table = []
    starved = []
    nonmarkov = []
    for row in rows:
        bucket, note = classify(row)
        led, rep = row["led"], row["rep"]
        census = rep.get("event_census", {})
        ev = (f"L:{census.get('LEVEL', 0)} W:{census.get('WIN', 0)} "
              f"G:{census.get('GAME_OVER', 0)}")
        if census.get("LEVEL", 0) + census.get("WIN", 0) == 0:
            starved.append(row["game"])
        store = rep.get("store", {})
        n = max(store.get("transitions", 0) + store.get("evicted", 0), 1)
        if store.get("conflicts", 0) / n >= CONFLICT_R3:
            nonmarkov.append(row["game"])
        m = led["match"]
        table.append({
            "game": row["game"], "bucket": bucket, "status": led["status"],
            "rhae": led["best_rhae"], "levels": led["levels"],
            "events": ev, "conflicts": store.get("conflicts", 0),
            "matched": m["matched"], "predicted": m["predicted_steps"],
            "actions": m["total_actions"], "note": note,
        })
        buckets.setdefault(bucket, []).append(row["game"])

    print(f"{'game':6s} {'bucket':16s} {'RHAE':>6s} {'lvl':>3s} {'events':14s} "
          f"{'confl':>6s} {'matched/pred/actions':>22s}")
    for t in table:
        print(f"{t['game']:6s} {t['bucket']:16s} {t['rhae']:6.2f} {t['levels']:3d} "
              f"{t['events']:14s} {t['conflicts']:6d} "
              f"{t['matched']:7d}/{t['predicted']:6d}/{t['actions']:7d}")
    print(f"\nmean RHAE over {len(rows)} games: "
          f"{summary['mean_rhae_over_games_run']:.3f}%")
    print(f"buckets: " + json.dumps({k: sorted(v) for k, v in buckets.items()}, indent=1))
    print(f"non-Markov (conflict>=0.2%): {len(nonmarkov)}/{len(rows)} {sorted(nonmarkov)}")
    print(f"evidence-starved (zero LEVEL+WIN observed): {len(starved)}/{len(rows)} {sorted(starved)}")

    (results_dir / "triage.json").write_text(json.dumps({
        "table": table, "buckets": buckets,
        "mean_rhae": summary["mean_rhae_over_games_run"],
        "nonmarkov": sorted(nonmarkov), "evidence_starved": sorted(starved),
    }, indent=1))
    print(f"\nwrote {results_dir}/triage.json")


if __name__ == "__main__":
    main()
