"""Acceptance + census report for the explorer run matrix (capped vs uncapped),
compared against the pre-upgrade single-seed sweep25 baseline.

    .venv/bin/python scripts/explore_matrix_report.py \
        --uncapped explore-uncap --capped explore-cap5 --baseline sweep25

Reads results/<tag>/summary.json + runs/wm/<tag>/*-trajectories.json. Writes
results/explore_matrix/report.json and prints the tables.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# triage buckets the sweep25 census assigned (NOTES 2026-06-10 full sweep)
SWEEP25_BUCKETS = {
    "r11l": "WINNABLE-MAYBE", "sp80": "WINNABLE-MAYBE",
    "bp35": "WALLED-R3-trans", "cd82": "WALLED-R3-trans", "dc22": "WALLED-R3-trans",
    "g50t": "WALLED-R3-trans", "ka59": "WALLED-R3-trans", "lf52": "WALLED-R3-trans",
    "m0r0": "WALLED-R3-trans", "sc25": "WALLED-R3-trans", "sk48": "WALLED-R3-trans",
    "tr87": "WALLED-R3-trans", "vc33": "WALLED-R3-trans", "wa30": "WALLED-R3-trans",
    "ar25": "WALLED-R3-win", "cn04": "WALLED-R3-win", "ft09": "WALLED-R3-win",
    "lp85": "WALLED-R3-win", "re86": "WALLED-R3-win", "sb26": "WALLED-R3-win",
    "su15": "WALLED-R3-win", "tn36": "WALLED-R3-win",
    "ls20": "DEAD", "s5i5": "DEAD", "tu93": "DEAD",
}


def load(tag):
    summ = json.loads((ROOT / "results" / tag / "summary.json").read_text())
    raw = ROOT / "runs/wm" / tag
    games = {}
    for led in summ["ledger"]:
        g = led["game"]
        rep = {}
        tr = raw / f"{g}-trajectories.json"
        if tr.exists():
            rep = json.loads(tr.read_text())["report"]
        census = rep.get("event_census", {})
        ex = rep.get("explorer", {})
        # first-evidence-inside-window: from the capped run, did any
        # LEVEL/WIN/GAME_OVER appear at all (capped run ends at the window)?
        games[g] = {
            "rhae": led["best_rhae"], "levels": led["levels"], "status": led["status"],
            "L": census.get("LEVEL", 0), "W": census.get("WIN", 0),
            "G": census.get("GAME_OVER", 0),
            "uniq": rep.get("store", {}).get("transitions", 0),
            "actions": led["match"]["total_actions"],
            "matched": led["match"]["matched"], "predicted": led["match"]["predicted_steps"],
            "tier": ex.get("tier_reached"), "archive": ex.get("archive_cells", 0),
            "go_returns": ex.get("go_explore_returns", 0),
            "probe_ev": ex.get("probe_first_evidence"),
            "frame_change": ex.get("frame_change_seen"),
            "verdict": (rep.get("plays") or [{}])[-1].get("exploration_verdict"),
            "starved": census.get("LEVEL", 0) == 0 and census.get("WIN", 0) == 0,
        }
    return {"mean_rhae": summ["mean_rhae_over_games_run"], "games": games}


def starv(d):
    g = d["games"]
    n = len(g)
    s = [k for k, v in g.items() if v["starved"]]
    return len(s), n, sorted(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uncapped", required=True)
    p.add_argument("--capped", required=True)
    p.add_argument("--baseline", default="sweep25")
    args = p.parse_args()

    unc = load(args.uncapped)
    cap = load(args.capped)
    base = load(args.baseline)
    out = {"uncapped": unc, "capped": cap, "baseline": base}

    print("=" * 78)
    print("EVIDENCE STARVATION CENSUS (no LEVEL and no WIN evidence)")
    for name, d in [("baseline sweep25", base), ("uncapped (new)", unc),
                    ("capped 5x (new)", cap)]:
        n_s, n, games = starv(d)
        print(f"  {name:20s}: {n_s}/{n} starved ({100*n_s/n:.0f}%)  {games}")

    print("\n" + "=" * 78)
    print("PER-GAME EVENT COUNTS  (uncapped L/W/G | capped L/W/G | uniq u/c)")
    print(f"{'game':6s} {'unc L/W/G':>14s} {'cap L/W/G':>14s} "
          f"{'uniq u/c':>14s} {'tier(cap)':>10s} {'verdict(cap)':>22s}")
    for g in sorted(unc["games"]):
        u, c = unc["games"][g], cap["games"][g]
        print(f"{g:6s} {f'{u[\"L\"]}/{u[\"W\"]}/{u[\"G\"]}':>14s} "
              f"{f'{c[\"L\"]}/{c[\"W\"]}/{c[\"G\"]}':>14s} "
              f"{f'{u[\"uniq\"]}/{c[\"uniq\"]}':>14s} {str(c['tier']):>10s} "
              f"{str(c['verdict']):>22s}")

    print("\n" + "=" * 78)
    print("RHAE (mean over 25):")
    print(f"  baseline {base['mean_rhae']:.3f} | uncapped {unc['mean_rhae']:.3f} "
          f"| capped {cap['mean_rhae']:.3f}")

    # INERT-START acceptance (item 1)
    print("\nINERT-START (ft09/lp85) uncapped unique transitions (was 24):")
    for g in ("ft09", "lp85"):
        print(f"  {g}: {unc['games'][g]['uniq']} unique, frame_change="
              f"{unc['games'][g]['frame_change']}; capped {cap['games'][g]['uniq']} "
              f"(ceiling = window)")

    # first-evidence inside capped window on previously-starved games
    base_starved = set(starv(base)[2])
    newly = [g for g in base_starved
             if not cap["games"][g]["starved"]
             or cap["games"][g]["G"] > 0]
    print("\nPreviously-starved games with event evidence INSIDE the capped window:")
    for g in sorted(newly):
        c = cap["games"][g]
        print(f"  {g}: capped L/W/G = {c['L']}/{c['W']}/{c['G']} "
              f"probe_ev={c['probe_ev']}")

    # tt01 canary (run separately; read if present)
    for tag in (args.uncapped, args.capped):
        tt = ROOT / "results" / f"{tag}-tt01" / "summary.json"
        if tt.exists():
            s = json.loads(tt.read_text())
            row = next((l for l in s["ledger"] if l["game"] == "tt01"), {})
            print(f"  tt01 canary [{tag}]: RHAE={row.get('best_rhae')} "
                  f"levels={row.get('levels')}")

    # triage bucket changes (a game "unstarves" => may leave WALLED-R3-win/DEAD)
    print("\nTriage movement (games that gained LEVEL/WIN evidence uncapped):")
    for g in sorted(unc["games"]):
        was = SWEEP25_BUCKETS.get(g, "?")
        u = unc["games"][g]
        gained = (not u["starved"]) and was in ("WALLED-R3-win", "DEAD")
        if gained:
            print(f"  {g}: {was} -> evidence now present (L={u['L']} W={u['W']})")

    # exploration-no-longer-the-blocker shortlist (Part 1/2 targets)
    print("\nExploration NO LONGER the blocker (Part 1/2 targets):")
    for g in sorted(unc["games"]):
        u = unc["games"][g]
        if u["levels"] > 0 or u["W"] > 0 or (u["L"] > 0):
            print(f"  {g}: levels={u['levels']} L/W={u['L']}/{u['W']} "
                  f"rhae={u['rhae']} status={u['status']}")

    (ROOT / "results/explore_matrix/report.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/explore_matrix/report.json")


if __name__ == "__main__":
    main()
