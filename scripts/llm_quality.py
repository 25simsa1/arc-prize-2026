"""Offline LLM-proposer quality pass against REAL captured game stores.

Decouples the proposer's quality measurement from the agent's modeling-time
cap / bailout: load a store, run the LIVE LLMProposer.propose() (+ gated
repair) N rounds, verify every rule by exact replay against the store, and
report per game: format-error rate, verified-rule count, whether each
verified rule is BEYOND-TEMPLATE (covers transitions no verified template
rule covers), the Phase-D economics (tokens & seconds per verified rule),
and the verbatim verified-rule corpus.

Backend-agnostic: --llm-url/--llm-model/--llm-backend point at the local
14B (plumbing/dev) or a rental serving the Phase-C pick (quality).

    .venv/bin/python scripts/llm_quality.py --stores runs/wm/store-capture \
        --games cn04 re86 sb26 su15 tn36 --rounds 4 --llm-model qwen2.5-coder:14b
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.llm_proposer import LLMProposer
from harness.wm.proposers import TemplateProposer
from harness.wm.regions import RegionAnalyzer
from harness.wm.rules import RuleStatus, WorldModel, grids_match
from harness.wm.store import TransitionStore
from harness.wm.verifier import verify_rules


def covered(rule, store, sample=800) -> set:
    """Indices of transitions a rule predicts exactly (grid+event match),
    over a capped sample (huge stores make a full pass too slow under the
    per-predict kill switch). Event transitions are always included."""
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
        if grids_match(p, t.post) and (p.event is None or p.event == t.event):
            out.add(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", required=True)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--llm-url", default="http://localhost:11434")
    ap.add_argument("--llm-model", default="qwen2.5-coder:14b")
    ap.add_argument("--llm-backend", default="ollama")
    ap.add_argument("--extra-body", default=None,
                    help='JSON merged into chat bodies, e.g. '
                         '\'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--out", default="results/llm_quality")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    report = {"model": args.llm_model, "rounds": args.rounds, "games": {}}
    corpus = []

    for g in args.games:
        pkl = Path(args.stores) / f"{g}-store.pkl"
        if not pkl.exists():
            print(f"{g}: no store, skip"); continue
        store = TransitionStore.load(pkl)
        # rebuild HUD mask the way the agent would (so evidence/masking match)
        model = WorldModel()
        analyzer = RegionAnalyzer()
        for t in store.all():
            analyzer.observe(t)
        rmap = analyzer.analyze()
        model.region_map = rmap
        model.hud_mask = rmap.hud_mask

        # template baseline: which transitions do VERIFIED templates cover?
        tmpl = TemplateProposer().propose(store, model,
                                          deadline=time.monotonic() + 5.0)
        verify_rules(tmpl, store, deadline=time.monotonic() + 5.0)
        tmpl_cov = set()
        for r in tmpl:
            if r.status == RuleStatus.VERIFIED:
                tmpl_cov |= covered(r, store)

        llm = LLMProposer(url=args.llm_url, model=args.llm_model,
                          backend=args.llm_backend, samples_per_call=3,
                          min_interval_s=0.0,
                          extra_body=json.loads(args.extra_body) if args.extra_body else None,
                          log_dir=str(outdir / "gens"))
        verified = []
        for rnd in range(args.rounds):
            rules = llm.propose(store, model, game_id=g)
            verify_rules(rules, store, deadline=time.monotonic() + 8.0)
            for r in rules:
                if r.status == RuleStatus.VERIFIED:
                    verified.append(r)
                elif r.status == RuleStatus.CONTRADICTED:
                    # gated repair: feed up to 6 misses, accept only if it
                    # strictly improves on the store
                    misses = []
                    for t in store.all():
                        p = r.predict(t.level, t.pre, t.action_key)
                        if p is not None and not (grids_match(p, t.post)
                                                  and (p.event is None or p.event == t.event)):
                            misses.append(t)
                            if len(misses) >= 6:
                                break
                    fixed = llm.repair(r, misses, store, model, game_id=g)
                    if fixed is not None:
                        verify_rules([fixed], store, deadline=time.monotonic() + 8.0)
                        if fixed.status == RuleStatus.VERIFIED:
                            verified.append(fixed)

        # beyond-template check + record corpus
        beyond = []
        for r in verified:
            cov = covered(r, store)
            extra = cov - tmpl_cov
            if extra:
                beyond.append(r)
                corpus.append({"game": g, "rule_id": r.rule_id,
                               "exact_transitions": len(cov),
                               "beyond_template_transitions": len(extra),
                               "code": getattr(r, "source_code", "")})
        s = llm.stats
        report["games"][g] = {
            "store_transitions": len(store),
            "template_verified_cover": len(tmpl_cov),
            "llm_verified": len(verified),
            "llm_verified_beyond_template": len(beyond),
            "format_error_rate": f"{s.format_errors}/{s.generations}",
            "format_retries": s.format_retries,
            "repair_accepted": s.repair_accepted,
            "repair_rejected": s.repair_rejected,
            "seconds_per_verified_rule":
                round(s.seconds / len(verified), 1) if verified else None,
            "tokens_per_verified_rule":
                round((s.prompt_tokens + s.output_tokens) / len(verified), 1) if verified else None,
            "generations": s.generations,
        }
        r = report["games"][g]
        print(f"{g}: verified={r['llm_verified']} "
              f"beyond_template={r['llm_verified_beyond_template']} "
              f"fmt_err={r['format_error_rate']} "
              f"repair acc/rej={r['repair_accepted']}/{r['repair_rejected']} "
              f"s/verified={r['seconds_per_verified_rule']}")

    report["corpus"] = corpus
    n_games_pass = sum(1 for v in report["games"].values()
                       if v["llm_verified_beyond_template"] > 0)
    report["games_with_beyond_template_verified"] = n_games_pass
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nGames with >=1 beyond-template VERIFIED rule: {n_games_pass}")
    print(f"wrote {outdir}/report.json")


if __name__ == "__main__":
    main()
