"""Before/after evidence-census comparison for the Part-0 acceptance gate.

    .venv/bin/python scripts/census_compare.py runs/wm/sweep25 runs/wm/p0-accept \
        ar25 cn04 sc25 tu93 wa30 re86 ft09 lp85
"""

import json
import sys
from pathlib import Path


def load(d: Path, game: str) -> dict:
    p = d / f"{game}-trajectories.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())["report"]
    return {
        "census": r.get("event_census", {}),
        "transitions": r.get("store", {}).get("transitions", 0),
        "verdict": (r.get("plays") or [{}])[-1].get("exploration_verdict"),
        "explorer": r.get("explorer", {}),
        "levels": max((p.get("levels", 0) for p in r.get("plays", [])), default=0),
    }


def main() -> None:
    before_dir, after_dir = Path(sys.argv[1]), Path(sys.argv[2])
    games = sys.argv[3:]
    first_evidence = []
    print(f"{'game':6s} {'uniq before->after':>20s} {'census before':22s} "
          f"{'census after':22s} {'lvl':>3s} verdict")
    for g in games:
        b, a = load(before_dir, g), load(after_dir, g)
        cb, ca = b.get("census", {}), a.get("census", {})
        def fmt(c):
            return f"L{c.get('LEVEL',0)} W{c.get('WIN',0)} G{c.get('GAME_OVER',0)}"
        new_classes = [k for k in ("LEVEL", "WIN", "GAME_OVER")
                       if ca.get(k, 0) > 0 and cb.get(k, 0) == 0]
        if new_classes:
            first_evidence.append((g, new_classes))
        print(f"{g:6s} {b.get('transitions',0):8d} -> {a.get('transitions',0):8d} "
              f"{fmt(cb):22s} {fmt(ca):22s} {a.get('levels',0):3d} {a.get('verdict')}")
    print(f"\nfirst-ever evidence classes: {first_evidence}")
    print(f"games with new evidence: {len(first_evidence)} (gate target: >=2)")


if __name__ == "__main__":
    main()
