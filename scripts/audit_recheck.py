"""TASK 1 — audit impact recheck. Re-verify every previously reported
VERIFIED/coverage number on the RECORDED artifacts, under (a) the FIXED
code at HEAD and (b) a faithful replica of the pre-audit buggy semantics
(vacuous claim counts as exact; no _fits guard). The delta on identical
inputs isolates the bugs' inflation. No new exploration.

Faithfulness notes:
- A/B arms (overnight-r1prime-on/off): the agent's R1' mask depends on
  action-index bins not recoverable from stores, so the mask is rebuilt
  from each run's RECORDED trajectories JSON ("regions"), exactly what the
  reported numbers used.
- llm_quality: mirrors its own pipeline (RegionAnalyzer().observe(t) with
  no idx, analyze() with no protected colors), incl. covered(sample=800).
- LLM rules are recompiled from the verbatim gens tarball and re-verified
  under the fixed verifier.

    .venv/bin/python scripts/audit_recheck.py
"""

import json
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.wm.proposers as proposers_mod
from harness.wm.llm_proposer import LLMProposer
from harness.wm.proposers import TemplateProposer
from harness.wm.regions import RegionAnalyzer
from harness.wm.rules import RuleStatus, WorldModel, grids_match
from harness.wm.store import TransitionStore
from harness.wm.verifier import verify_rules


def verify_rules_buggy(rules, store, min_exact=3):
    """Replica of the PRE-AUDIT verifier semantics: a claim-less Prediction
    counts as an exact match; deadline checked only between rules (we run
    with no deadline here, so that part is moot)."""
    transitions = list(store.all())
    for rule in rules:
        exact = miss = 0
        for t in transitions:
            try:
                p = rule.predict(t.level, t.pre, t.action_key)
            except Exception:
                continue
            if p is None:
                continue
            # NOTE: no vacuous-claim guard — the old bug, on purpose
            ok = grids_match(p, t.post) and (p.event is None or p.event == t.event)
            exact, miss = exact + ok, miss + (not ok)
        rule.n_exact, rule.n_miss = exact, miss
        rule.status = (RuleStatus.CONTRADICTED if miss else
                       RuleStatus.VERIFIED if exact >= min_exact else
                       RuleStatus.UNTESTED)


def _fits_buggy(rule, transitions):
    n = 0
    for o in transitions:
        p = rule.predict(o.level, o.pre, o.action_key)
        if p is None:
            continue
        if not grids_match(p, o.post) or (p.event is not None and p.event != o.event):
            return -1
        n += 1
    return n


def claimful_fraction(rule, store, n=100) -> float:
    """Fraction of sampled predictions that carry a REAL claim — 0.0 means
    the rule is vacuous (the exact thing the audit guard now rejects)."""
    ts = list(store.all())[:n]
    fired = claims = 0
    for t in ts:
        try:
            p = rule.predict(t.level, t.pre, t.action_key)
        except Exception:
            continue
        if p is None:
            continue
        fired += 1
        if p.grid is not None or p.event is not None:
            claims += 1
    return (claims / fired) if fired else 1.0


def mask_from_trajectories(traj_path: Path):
    rep = json.loads(traj_path.read_text()).get("report", {})
    regions = rep.get("regions")
    if not regions or not regions.get("hud_regions"):
        return None
    shape = tuple(regions.get("shape") or (64, 64))
    m = np.zeros(shape, dtype=bool)
    for r in regions["hud_regions"]:
        for y, x in r["cells"]:
            m[y, x] = True
    return m


def covered(rule, store, sample=800) -> set:
    ts = list(store.all())
    idx = list(range(len(ts)))
    if len(idx) > sample:
        ev = [i for i in idx if ts[i].event != "NONE"]
        rest = [i for i in idx if ts[i].event == "NONE"][: max(0, sample - len(ev))]
        idx = ev + rest
    out = set()
    for i in idx:
        t = ts[i]
        try:
            p = rule.predict(t.level, t.pre, t.action_key)
        except Exception:
            continue
        if p is None:
            continue
        if p.grid is None and p.event is None:
            continue  # FIXED semantics here; buggy variant handled by caller
        if grids_match(p, t.post) and (p.event is None or p.event == t.event):
            out.add(i)
    return out


def recheck_arm(arm_dir: Path, game: str) -> dict:
    store = TransitionStore.load(arm_dir / f"{game}-store.pkl")
    mask = mask_from_trajectories(arm_dir / f"{game}-trajectories.json")
    model = WorldModel()
    model.hud_mask = mask

    out = {}
    for mode in ("buggy", "fixed"):
        orig_fits = proposers_mod._fits
        if mode == "buggy":
            proposers_mod._fits = _fits_buggy
        try:
            rules = TemplateProposer().propose(store, model,
                                               deadline=time.monotonic() + 30.0)
        finally:
            proposers_mod._fits = orig_fits
        if mode == "buggy":
            verify_rules_buggy(rules, store)
        else:
            verify_rules(rules, store, deadline=time.monotonic() + 180.0)
        ver = [r for r in rules if r.status == RuleStatus.VERIFIED]
        out[mode] = {
            "verified": len(ver),
            "ids": [r.rule_id for r in ver][:8],
            "vacuous": [r.rule_id for r in ver
                        if claimful_fraction(r, store) == 0.0],
        }
    return out


def recheck_llm_gens(store_dir: Path, games: list[str]) -> dict:
    gens = Path("results/llm_quality/gens.tar.gz")
    out = {}
    if not gens.exists():
        return {"error": "gens tarball missing"}
    llm = LLMProposer(log_dir=None)
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(gens) as tar:
            tar.extractall(td)
        files = sorted(Path(td).rglob("*.md"))
        for g in games:
            pkl = store_dir / f"{g}-store.pkl"
            if not pkl.exists():
                continue
            store = TransitionStore.load(pkl)
            n_rules = n_verified = 0
            for f in [f for f in files if f.name.startswith(f"{g}-")]:
                m = re.search(r"```python\n(.*?)```", f.read_text(), re.DOTALL)
                if not m or "def predict" not in m.group(1):
                    continue
                fn = llm._compile(m.group(1))
                if fn is None:
                    continue
                rule = llm._wrap(fn, m.group(1), g, None)
                n_rules += 1
                verify_rules([rule], store, deadline=time.monotonic() + 60.0)
                if rule.status == RuleStatus.VERIFIED:
                    n_verified += 1
            out[g] = {"recompiled": n_rules, "verified_fixed": n_verified}
    return out


def main() -> None:
    report = {}

    # --- A/B arms: the sp80/r11l "first VERIFIED rule" claims + the rest
    for arm in ("overnight-r1prime-on", "overnight-r1prime-off"):
        arm_dir = Path("runs/wm") / arm
        for game in ("sp80", "r11l", "ft09", "lp85", "tn36"):
            if not (arm_dir / f"{game}-store.pkl").exists():
                continue
            key = f"{arm.split('-')[-1]}/{game}"
            report[key] = recheck_arm(arm_dir, game)
            b, f = report[key]["buggy"], report[key]["fixed"]
            print(f"{key:10s} buggy={b['verified']:3d} fixed={f['verified']:3d} "
                  f"vacuous_in_buggy={len(b['vacuous'])} "
                  f"fixed_ids={f['ids'][:3]}")

    # --- llm_quality: template coverage counts + LLM rules
    candidates = [d for d in Path("runs/wm").iterdir()
                  if (d / "su15-store.pkl").exists()]
    print(f"\nllm_quality store dir candidates: {[str(c) for c in candidates]}")
    if candidates:
        sd = candidates[0]
        for g, claimed in (("su15", 129), ("sb26", 79), ("ar25", 1)):
            pkl = sd / f"{g}-store.pkl"
            if not pkl.exists():
                continue
            store = TransitionStore.load(pkl)
            model = WorldModel()
            an = RegionAnalyzer()
            for t in store.all():
                an.observe(t)
            rmap = an.analyze()
            model.region_map = rmap
            model.hud_mask = rmap.hud_mask
            tmpl = TemplateProposer().propose(store, model,
                                              deadline=time.monotonic() + 30.0)
            cov = {}
            for mode in ("buggy", "fixed"):
                if mode == "buggy":
                    verify_rules_buggy(tmpl, store)
                else:
                    verify_rules(tmpl, store, deadline=time.monotonic() + 180.0)
                s = set()
                for r in tmpl:
                    if r.status == RuleStatus.VERIFIED:
                        s |= covered(r, store)
                cov[mode] = len(s)
            report[f"llmq/{g}"] = {"claimed": claimed, **cov}
            print(f"llmq/{g}: claimed={claimed} buggy={cov['buggy']} fixed={cov['fixed']}")
        report["llmq/llm_rules"] = recheck_llm_gens(sd, ["su15", "sb26", "ar25", "sp80"])
        print("llm rules recheck:", json.dumps(report["llmq/llm_rules"]))

    Path("results/audit_recheck.json").write_text(json.dumps(report, indent=1))
    print("\nwrote results/audit_recheck.json")


if __name__ == "__main__":
    main()
