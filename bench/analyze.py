"""Local analysis of bake-off tarballs (run after scp'ing them down).

Per model per task: success/partial rates against the calibrated references,
repair-loop lift, reframing lift, sample distribution (single-lucky-
generation results called out explicitly), and the raw material for the
qualitative failure-pattern note (hypothesis comment lines per model on the
structured-response tasks).

    .venv/bin/python bench/analyze.py bench/results/*.tar.gz
"""

import argparse
import json
import re
import statistics
import tarfile
import tempfile
from pathlib import Path

CAL = json.loads(Path(__file__).with_name("tasks").joinpath("calibration.json").read_text())
A_REF = CAL["A"]["reference"]["acc"]                      # 1.0
A_FLOOR = CAL["A"]["floor"]["acc"]                        # 0.5
B_NAIVE = CAL["B"]["naive_reference"]["mean_jaccard"]     # 0.295


def lucky(vals: list[float]) -> bool:
    """One sample carries the result: best is far above the rest."""
    if len(vals) < 3:
        return False
    top = sorted(vals, reverse=True)
    return top[0] > 0 and (top[1] <= 0.5 * top[0])


def hypothesis_lines(gen_dir: Path, prefix: str) -> list[str]:
    out = []
    for p in sorted(gen_dir.glob(f"{prefix}-gen*.md")):
        m = re.search(r"#\s*(Hypothes\w*.*)$", p.read_text(), re.MULTILINE | re.IGNORECASE)
        if m:
            out.append(m.group(1).strip()[:140])
    return out


def analyze_one(tar_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar_path) as tar:
            tar.extractall(td)
        root = next(Path(td).iterdir())
        res = json.loads((root / "results.json").read_text())
        rep: dict = {"model": res["model_id"], "repo": res.get("repo"),
                     "load_s": res.get("load_seconds"),
                     "run_s": res.get("total_run_seconds")}

        if "A" in res:
            accs = [r.get("acc", 0.0) for r in res["A"] if "acc" in r]
            errs = sum(1 for r in res["A"] if "error" in r)
            rep["A"] = {
                "best": max(accs, default=0.0), "mean": round(statistics.mean(accs), 3) if accs else 0.0,
                "n_substantial": sum(1 for a in accs if a >= 0.9),
                "n_above_floor": sum(1 for a in accs if a > A_FLOOR + 0.05),
                "errors": errs, "lucky_single_gen": lucky(accs),
                "reference": A_REF, "floor": A_FLOOR,
            }
        for t in ("B", "reframe"):
            if t in res:
                js = [r.get("mean_jaccard", 0.0) for r in res[t] if "mean_jaccard" in r]
                errs = sum(1 for r in res[t] if "error" in r)
                rep[t] = {
                    "best": round(max(js, default=0.0), 3),
                    "mean": round(statistics.mean(js), 3) if js else 0.0,
                    "n_beats_naive": sum(1 for j in js if j > B_NAIVE + 0.02),
                    "errors": errs, "lucky_single_gen": lucky(js),
                    "naive_reference": round(B_NAIVE, 3),
                }
        if "repair" in res:
            lifts = []
            for r in res["repair"]:
                b, a = r.get("before", {}), r.get("after", {})
                if "acc" in b and "acc" in a:
                    lifts.append(round(a["acc"] - b["acc"], 3))
            rep["repair"] = {
                "n_pairs": len(lifts),
                "mean_lift": round(statistics.mean(lifts), 3) if lifts else None,
                "lifts": lifts,
                "n_positive": sum(1 for v in lifts if v > 0),
                "n_negative": sum(1 for v in lifts if v < 0),
            }
        if "reframe" in rep and "B" in rep:
            rep["reframe_lift_best"] = round(rep["reframe"]["best"] - rep["B"]["best"], 3)
            rep["reframe_lift_mean"] = round(rep["reframe"]["mean"] - rep["B"]["mean"], 3)

        rep["failure_patterns_B"] = hypothesis_lines(root, "B")[:8]
        rep["failure_patterns_reframe"] = hypothesis_lines(root, "reframe")[:8]
        return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarballs", nargs="+")
    args = ap.parse_args()
    out = [analyze_one(Path(t)) for t in args.tarballs]
    print(json.dumps(out, indent=1))
    Path("bench/results/analysis.json").write_text(json.dumps(out, indent=1))
    print("\nsaved bench/results/analysis.json", flush=True)


if __name__ == "__main__":
    main()
