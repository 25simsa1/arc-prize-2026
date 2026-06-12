"""One-command analysis for the rental-2 tarball. When the tarball lands:

    tar xzf rental2.tar.gz -C results/rental2
    .venv/bin/python scripts/rental_report.py results/rental2

Emits (per the Task-4 spec):
  1. verified-rules-per-game under the FULL FIXED verifier — recomputed
     locally from the verbatim generation corpus against the STAGED stores
     (process rule: never trust in-run verified counts);
  2. format-error rate vs the 0.6% local-14B baseline;
  3. gated-repair accept counts and accepted-repair lift;
  4. the 2^3 gap table: condition means, factor marginals (delta_plus /
     delta_minus), paired-bootstrap SE over held-out items pooled over
     samples, noise floor 2x pooled SE, and keep/drop/reallocate decisions
     by gap-per-token;
  5. economics: tokens+seconds per verified rule against the 6h/110-game
     envelope.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from harness.wm.llm_proposer import LLMProposer
from harness.wm.rules import RuleStatus
from harness.wm.store import TransitionStore
from harness.wm.verifier import verify_rules

STORES = {
    "sp80": "runs/wm/overnight-r1prime-on/sp80-store.pkl",
    "su15": "runs/wm/rental-stores/su15-store.pkl",
    "sb26": "runs/wm/rental-stores/sb26-store.pkl",
    "ar25": "runs/wm/rental-stores/ar25-store.pkl",
}
LOCAL_14B_FORMAT_BASELINE = "1/157 (0.6%)"
ENVELOPE_S = 6 * 3600   # 6h session (confirmed figure), 110 games
FACTORS = ("L", "T", "M")


def reverify_corpus(run_dir: Path) -> dict:
    llm = LLMProposer(log_dir=None)
    out = {}
    gen_files = sorted(run_dir.rglob("*-gen-*.md")) + sorted(run_dir.rglob("*-repair-*.md"))
    for game, store_rel in STORES.items():
        pkl = ROOT / store_rel
        if not pkl.exists():
            out[game] = {"error": f"staged store missing: {store_rel}"}
            continue
        store = TransitionStore.load(pkl)
        n = v = 0
        verified_ids = []
        for f in [f for f in gen_files if f.name.startswith(f"{game}-")]:
            m = re.search(r"```python\n(.*?)```", f.read_text(), re.DOTALL)
            if not m or "def predict" not in m.group(1):
                continue
            fn = llm._compile(m.group(1))
            if fn is None:
                continue
            rule = llm._wrap(fn, m.group(1), game, None)
            n += 1
            verify_rules([rule], store, deadline=time.monotonic() + 90.0)
            if rule.status == RuleStatus.VERIFIED:
                v += 1
                verified_ids.append(f.name)
        out[game] = {"recompiled": n, "verified_full": v, "files": verified_ids[:6]}
    return out


def bootstrap_gap(bits_c: list[list[int]], bits_0: list[list[int]],
                  iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap over held-out ITEMS, pooled over samples.
    bits_*: per-gen lists of per-item correctness (same item order)."""
    rng = random.Random(seed)
    n_items = len(bits_0[0]) if bits_0 and bits_0[0] else 0
    if n_items == 0:
        return 0.0, 0.0
    def mean_over(bits, idx):
        vals = [sum(b[i] for i in idx) / len(idx) for b in bits if len(b) == n_items]
        return sum(vals) / len(vals) if vals else 0.0
    base_gap = mean_over(bits_c, range(n_items)) - mean_over(bits_0, range(n_items))
    gaps = []
    for _ in range(iters):
        idx = [rng.randrange(n_items) for _ in range(n_items)]
        gaps.append(mean_over(bits_c, idx) - mean_over(bits_0, idx))
    mu = sum(gaps) / len(gaps)
    se = (sum((g - mu) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5
    return base_gap, se


def analyze_gap(run_dir: Path) -> dict:
    raw_path = run_dir / "gap_experiment" / "raw.json"
    if not raw_path.exists():
        cands = list(run_dir.rglob("raw.json"))
        if not cands:
            return {"error": "gap raw.json not found"}
        raw_path = cands[0]
    raw = json.loads(raw_path.read_text())
    block_chars = raw.get("block_chars", {})
    res = raw["results"]

    def cond_bits(name, task):
        rows = res.get(f"{name}/{task}", [])
        if task == "A":
            return [r["bits"] for r in rows if "bits" in r]
        # task B: per-item jaccards = off_toggle + on_toggle lists
        return [r["off_toggle"] + r["on_toggle"] for r in rows
                if "off_toggle" in r]

    table = {}
    for task in ("A", "B"):
        b0 = cond_bits("none", task)
        rows = {}
        for name in res:
            if not name.endswith(f"/{task}"):
                continue
            cname = name.split("/")[0]
            gap, se = bootstrap_gap(cond_bits(cname, task), b0)
            rows[cname] = {"gap": round(gap, 4), "se": round(se, 4)}
        # marginals
        marg = {}
        for i, X in enumerate(FACTORS):
            alone = X
            full = "LTM"
            full_minus = "".join(f for f in "LTM" if f != X) or "none"
            dp = rows.get(alone, {}).get("gap")
            dm = (rows.get(full, {}).get("gap", 0)
                  - rows.get(full_minus, {}).get("gap", 0))
            se_pool = max(rows.get(alone, {}).get("se", 0),
                          rows.get(full, {}).get("se", 0))
            tokens = block_chars.get(X if X != "T" else "T_a", 0) / 4
            decision = "drop"
            if dm is not None and abs(dm) >= 2 * se_pool and dm > 0:
                decision = "keep"
            elif dp is not None and abs(dp) >= 2 * se_pool and dp > 0:
                decision = "keep (marginal-alone)"
            marg[X] = {"delta_plus": dp, "delta_minus": round(dm, 4),
                       "2xSE": round(2 * se_pool, 4),
                       "gap_per_ktok": round((dm or 0) / max(tokens / 1000, 1e-9), 3),
                       "decision": decision}
        table[task] = {"conditions": rows, "marginals": marg}
    return table


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/rental2")
    report = {}

    qual = next(iter(run_dir.rglob("report.json")), None)
    if qual:
        q = json.loads(qual.read_text())
        report["quality_in_run"] = q
        fmt = {g: d.get("format_error_rate") for g, d in q.get("games", {}).items()}
        rep = {g: f"{d.get('repair_accepted')}/{d.get('repair_accepted', 0) + d.get('repair_rejected', 0)}"
               for g, d in q.get("games", {}).items()}
        report["format_errors"] = {"per_game": fmt,
                                   "baseline_local_14b": LOCAL_14B_FORMAT_BASELINE}
        report["gated_repair_accepts"] = rep

    print("== re-verifying generation corpus under the FULL fixed verifier")
    report["verified_full"] = reverify_corpus(run_dir)
    print(json.dumps(report["verified_full"], indent=1))

    print("== 2^3 gap analysis")
    report["gap"] = analyze_gap(run_dir)
    print(json.dumps(report["gap"], indent=1))

    # economics
    if qual:
        q = report["quality_in_run"]
        tot_v = sum(d.get("verified_full", 0) for d in report["verified_full"].values()
                    if isinstance(d, dict))
        secs = sum(d.get("seconds", 0) for d in q.get("games", {}).values()
                   if isinstance(d, dict))
        report["economics"] = {
            "verified_rules_total_full": tot_v,
            "llm_seconds_total": secs,
            "seconds_per_verified_rule": round(secs / tot_v, 1) if tot_v else None,
            "envelope": f"{ENVELOPE_S}s / 110 games = {ENVELOPE_S // 110}s/game",
            "rules_affordable_per_game": (
                round((ENVELOPE_S / 110) * 0.5 / (secs / tot_v), 2)
                if tot_v and secs else None),  # 50% of per-game wall for the LLM
        }
        print("== economics", json.dumps(report["economics"], indent=1))

    out = run_dir / "RENTAL_REPORT.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
