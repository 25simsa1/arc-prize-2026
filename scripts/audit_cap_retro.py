"""Retroactive 5x-per-level cap audit over stored trajectories (no reruns).

For every stored play (sweep25 real games + tt01 baseline runs):
  - replay the step stream against a hypothetical cap of MULT x the level's
    human baseline, with the SAME provisional semantics as
    harness.runner.level_action_cap (counter = actions on current level,
    cumulative across GAME_OVER resets; completing on the cap-th action
    allowed; first cap fire truncates the play);
  - record where the cap would have fired, what survives before it, and
    which observed events (LEVEL advances / WINs / GAME_OVERs) arrive
    within the truncated window — the eval-realistic evidence census.
  - for wins: per-level margin (actions used / cap).

    .venv/bin/python scripts/audit_cap_retro.py

Writes results/cap_study/retro_audit.json and prints the tables.
"""

import glob
import json
import sys
from pathlib import Path

MULT = 5.0
ROOT = Path(__file__).resolve().parent.parent
ENV = Path.home() / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files"
TT01_BASE = [3, 3]  # test_envs/tt01 fixture

# AERA-style persistence probe: each available simple action repeated ~200x
PROBE_REPS = 200


def baselines() -> dict[str, list[int]]:
    out = {}
    for md in ENV.glob("*/*/metadata.json"):
        d = json.load(open(md))
        out[d["game_id"].split("-")[0]] = d.get("baseline_actions") or []
    out["tt01"] = TT01_BASE
    return out


def audit_play(steps: list[dict], base: list[int], mult: float = MULT):
    """Walk one play's step stream under the hypothetical cap.

    Returns dict with: cap_fired (level, step_index) or None, events_total,
    events_within (events observed strictly before the cap fires),
    per-level action counts, levels completed before/after cap.
    """
    level_actions: dict[int, int] = {}
    cur = steps[0]["level"] if steps else 0
    fired = None
    events_total, events_within = {}, {}
    levels_before_cap = 0
    completed_levels_actions: dict[int, int] = {}

    for i, st in enumerate(steps):
        lvl = st["level"]
        ev = st.get("event", "NONE")
        if ev != "NONE":
            events_total[ev] = events_total.get(ev, 0) + 1
            if fired is None:
                events_within[ev] = events_within.get(ev, 0) + 1
        if lvl > cur:
            # the action at i completed level cur
            completed_levels_actions[cur] = level_actions.get(cur, 0) + 1
            if fired is None:
                levels_before_cap = lvl
            cur = lvl
            continue
        # action attributed to current level
        level_actions[cur] = level_actions.get(cur, 0) + 1
        if fired is None and cur < len(base) and base[cur] and base[cur] > 0:
            cap = int(mult * base[cur])
            if level_actions[cur] >= cap:
                fired = {"level": cur, "step_index": i,
                         "actions_at_fire": i + 1, "cap": cap}
    return {
        "cap_fired": fired,
        "events_total": events_total,
        "events_within_cap": events_within,
        "levels_completed": cur,
        "levels_before_cap": levels_before_cap if fired else cur,
        "completed_level_actions": completed_levels_actions,
        "total_actions": len(steps),
    }


def win_margins(plays, base, game):
    """Per-level (actions used / cap) for every play that ended in WIN."""
    rows = []
    for p in plays:
        if p.get("end_state") != "WIN":
            continue
        a = audit_play(p["steps"], base)
        for lvl, used in sorted(a["completed_level_actions"].items()):
            cap = int(MULT * base[lvl]) if lvl < len(base) and base[lvl] > 0 else None
            rows.append({
                "game": game, "play": p["play"], "level": lvl,
                "actions": used, "baseline": base[lvl] if lvl < len(base) else None,
                "cap": cap,
                "margin": round(used / cap, 3) if cap else None,
                "fits": (used <= cap) if cap else None,
            })
    return rows


def main():
    bases = baselines()
    audit = {"mult": MULT, "games": {}, "win_margins": [], "probe_fit": []}

    traj_files = sorted(glob.glob(str(ROOT / "runs/wm/sweep25/*-trajectories.json")))
    tt01_candidates = sorted(
        glob.glob(str(ROOT / "runs/wm/baseline-c1-tt01/tt01-trajectories.json"))
    )
    for f in traj_files + tt01_candidates:
        d = json.load(open(f))
        game = d["report"]["game_id"].split("-")[0]
        base = bases.get(game, [])
        plays = d["plays"]
        per_play = [audit_play(p["steps"], base) for p in plays]
        audit["win_margins"] += win_margins(plays, base, game)

        evs_tot, evs_in = {}, {}
        for a in per_play:
            for k, v in a["events_total"].items():
                evs_tot[k] = evs_tot.get(k, 0) + v
            for k, v in a["events_within_cap"].items():
                evs_in[k] = evs_in.get(k, 0) + v
        first = per_play[0]
        audit["games"][game] = {
            "source": str(Path(f).parent.name),
            "baseline": base,
            "plays": len(plays),
            "total_actions": sum(a["total_actions"] for a in per_play),
            "cap_fired_play0": first["cap_fired"],
            "levels_uncapped": max(a["levels_completed"] for a in per_play),
            "levels_under_cap": max(a["levels_before_cap"] for a in per_play),
            "events_total": evs_tot,
            "events_within_cap": evs_in,
        }

    # persistence-probe fit: does 200 reps/action fit in level 0's cap?
    # tags tell which action families a game uses; conservative count = 5
    # simple actions (click games would need far more via coordinates).
    for game, base in sorted(bases.items()):
        if game == "tt01" or not base:
            continue
        cap0 = int(MULT * base[0]) if base[0] > 0 else None
        audit["probe_fit"].append({
            "game": game, "baseline_l0": base[0], "cap_l0": cap0,
            "one_action_probe_fits": (cap0 or 0) >= PROBE_REPS,
            "five_action_probe_fits": (cap0 or 0) >= 5 * PROBE_REPS,
            "max_reps_per_action_5acts": (cap0 or 0) // 5,
        })

    out = ROOT / "results/cap_study/retro_audit.json"
    out.write_text(json.dumps(audit, indent=2))

    # ---- printed tables ----
    g = audit["games"]
    print(f"=== retro audit at {MULT}x (per-play truncation) ===")
    print(f"{'game':6s} {'lvls':>9s} {'cap@lvl':>7s} {'fire@act':>9s} "
          f"{'of total':>9s} {'events(total)':>22s} {'events(within cap)':>22s}")
    for game, a in sorted(g.items()):
        cf = a["cap_fired_play0"]
        print(f"{game:6s} {a['levels_under_cap']}/{a['levels_uncapped']:<7d} "
              f"{(cf['level'] if cf else '-'):>7} "
              f"{(cf['actions_at_fire'] if cf else '-'):>9} "
              f"{a['total_actions']:>9d} "
              f"{str(a['events_total']):>22s} {str(a['events_within_cap']):>22s}")

    print("\n=== win margins (every WIN play, per level) ===")
    for r in audit["win_margins"]:
        print(f"  {r['game']} play{r['play']} L{r['level']}: {r['actions']} acts "
              f"vs cap {r['cap']} (baseline {r['baseline']}) -> "
              f"margin {r['margin']} {'FITS' if r['fits'] else 'EXCEEDS'}")
    if not audit["win_margins"]:
        print("  (no WIN plays found in audited trajectories)")

    n_fit1 = sum(1 for r in audit["probe_fit"] if r["one_action_probe_fits"])
    n_fit5 = sum(1 for r in audit["probe_fit"] if r["five_action_probe_fits"])
    print(f"\n=== persistence-probe fit under cap (level 0) ===")
    print(f"  200 reps of ONE action fits: {n_fit1}/{len(audit['probe_fit'])}")
    print(f"  200 reps of FIVE actions fits: {n_fit5}/{len(audit['probe_fit'])}")
    for r in audit["probe_fit"]:
        print(f"  {r['game']:6s} b0={r['baseline_l0']:<4d} cap0={r['cap_l0']:<5d} "
              f"1-act:{'Y' if r['one_action_probe_fits'] else 'n'} "
              f"5-act:{'Y' if r['five_action_probe_fits'] else 'n'} "
              f"reps/act@5acts={r['max_reps_per_action_5acts']}")

    # eval-realistic starvation: games with zero LEVEL/WIN evidence within cap
    starved_uncapped = [k for k, a in g.items()
                        if k != "tt01" and not any(e in a["events_total"] for e in ("LEVEL", "WIN"))]
    starved_capped = [k for k, a in g.items()
                      if k != "tt01" and not any(e in a["events_within_cap"] for e in ("LEVEL", "WIN"))]
    n_real = len([k for k in g if k != "tt01"])
    print(f"\n=== starvation census (LEVEL/WIN evidence) ===")
    print(f"  uncapped (original): {len(starved_uncapped)}/{n_real} starved "
          f"({100*len(starved_uncapped)/n_real:.0f}%)")
    print(f"  within-cap (eval-realistic): {len(starved_capped)}/{n_real} starved "
          f"({100*len(starved_capped)/n_real:.0f}%)")
    print(f"  newly starved under cap: {sorted(set(starved_capped)-set(starved_uncapped))}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
