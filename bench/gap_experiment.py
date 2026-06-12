"""The 2^3 privileged-context gap experiment (research/arxiv_sweep_2026-06.md
section 3; law + protocol from arXiv:2605.30070).

Factors, inserted as discrete prompt blocks into otherwise-identical base
prompts: L = conflict ledger, T = temporal context, M = current-model view.
Full factorial (8 conditions) x tasks {A: event-precondition accuracy on the
fixed 60-item held-out; B: reframed structured response, mean Jaccard} x
N=8 samples = 128 generations. Gap(c) = mean metric(c) - mean metric(none),
same model/decoding/exemplars. Marginals per factor X:
  delta_plus(X)  = gap(X alone)
  delta_minus(X) = gap(full) - gap(full minus X)
Noise floor: paired bootstrap over held-out items pooled over samples;
|delta| < 2x pooled SE is treated as zero. Decision rules (keep/drop/
reallocate by gap-per-token) applied in scripts/rental_report.py.

Three stages so the rental box stays dumb:
  --build    (local) -> bench/tasks/gap_conditions.json (prompts + hashes)
  --run      (box)   -> results/gap_experiment/raw.json (+ verbatim gens)
  (analysis lives in scripts/rental_report.py)
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

TEMPERATURE = 0.8
N_SAMPLES = 8
FACTORS = ("L", "T", "M")


# ---------------------------------------------------------------- building

def _conflict_ledger_block(store) -> str:
    """L: the conflict ledger as privileged context (SDPO-style feedback;
    OEL 2603.16856: structured experiential knowledge beats raw dumps)."""
    rows = store.conflicts[:8]
    if not rows:
        return ("CONFLICT LEDGER: no contradictions observed — every repeated "
                "(state, action) produced the identical outcome so far.\n")
    lines = "\n".join(json.dumps(r) for r in rows)
    return ("CONFLICT LEDGER (same state+action later produced a DIFFERENT "
            "outcome — evidence of hidden state your rule may need to "
            f"respect):\n{lines}\n")


def _model_view_block(store, model, rules) -> str:
    """M: current verified-model view (OPSD 'verified traces' analogue)."""
    from harness.wm.rules import RuleStatus
    ver = [r for r in rules if r.status == RuleStatus.VERIFIED][:10]
    if not ver:
        return ("CURRENT MODEL: no rules have been verified yet — the "
                "dynamics below are so far unexplained.\n")
    lines = "\n".join(f"- {r.rule_id} (exact on {r.n_exact} observations)"
                      for r in ver)
    return ("CURRENT MODEL (already-VERIFIED rules — do not re-derive these; "
            f"hypothesize what they MISS):\n{lines}\n")


def _temporal_block_a(history_rows) -> str:
    """T for task A: the action history preceding each exemplar window."""
    lines = "\n".join(history_rows)
    return ("RECENT ACTION HISTORY (the consecutive observations immediately "
            f"before the window below, oldest first):\n{lines}\n")


def build(store_path: str) -> dict:
    import evidence as ev
    from harness.wm.proposers import TemplateProposer
    from harness.wm.rules import WorldModel
    from harness.wm.store import TransitionStore
    from harness.wm.verifier import verify_rules

    store = TransitionStore.load(store_path)
    taskA = ev.build_task_a(store)
    taskB_plain = ev.build_task_b(store, reframe=False)
    taskB_temporal = ev.build_task_b(store, reframe=True)  # carries T inside

    # the reframe INSTRUCTION (anti-click-geometry) is prompt-base for all B
    # conditions; reframe=True additionally carries previous_transition rows
    # — that addition is exactly the T factor for task B.
    reframe_note = ("\nIMPORTANT, learned from failed prior attempts: "
                    "hypotheses anchored on geometry near the click "
                    "coordinate were all wrong — the changed cells are "
                    "typically FAR from the click. Anchor on the color-15 "
                    "pattern state.\n")

    model = WorldModel()
    rules = TemplateProposer().propose(store, model,
                                       deadline=time.monotonic() + 30.0)
    verify_rules(rules, store, deadline=time.monotonic() + 120.0)

    L = _conflict_ledger_block(store)
    M = _model_view_block(store, model, rules)
    # T for task A: serialize the 6 transitions preceding the exemplar window
    ts = list(store.all())
    hist = [json.dumps({"action": t.action_key, "event": t.event})
            for t in ts[:6]]
    T_a = _temporal_block_a(hist)

    conditions = {}
    for bits in itertools.product((0, 1), repeat=3):
        name = "".join(f for f, b in zip(FACTORS, bits) if b) or "none"
        blocks = ""
        if bits[0]:
            blocks += L + "\n"
        if bits[2]:
            blocks += M + "\n"
        # task A prompt: base evidence + optional T history + blocks
        pa = taskA["prompt"]
        if bits[1]:
            pa = pa.replace("OBSERVATIONS:", T_a + "\nOBSERVATIONS:")
        pa = blocks + pa
        # task B prompt: plain or temporal variant + reframe note + blocks
        pb = (taskB_temporal if bits[1] else taskB_plain)["prompt"]
        pb = pb.replace("OBSERVATIONS:", reframe_note + "\nOBSERVATIONS:")
        pb = blocks + pb
        conditions[name] = {
            "bits": dict(zip(FACTORS, bits)),
            "prompt_A": pa, "prompt_B": pb,
            "sha_A": hashlib.sha256(pa.encode()).hexdigest()[:12],
            "sha_B": hashlib.sha256(pb.encode()).hexdigest()[:12],
            "chars_A": len(pa), "chars_B": len(pb),
        }
    out = {
        "store": store_path,
        "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
        "block_chars": {"L": len(L), "T_a": len(T_a), "M": len(M)},
        "heldout_A": taskA["heldout"],
        "heldout_B": taskB_plain["heldout"],
        "conditions": conditions,
    }
    path = HERE / "tasks" / "gap_conditions.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"built 8 conditions -> {path}")
    for n, c in conditions.items():
        print(f"  {n:4s} A={c['chars_A']:5d}ch B={c['chars_B']:5d}ch")
    return out


# ----------------------------------------------------------------- running

def run(url: str, model: str, backend: str, out_dir: Path,
        samples_override: int = 0) -> None:
    from run_bench import gen_ollama, gen_openai
    from scoring import extract_code, score

    spec = json.loads((HERE / "tasks" / "gap_conditions.json").read_text())
    if samples_override:
        spec["n_samples"] = samples_override  # plumbing smokes only
    # the scorers read heldout from a spec file path; write a shim spec
    shim = out_dir / "heldout_shim.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    shim.write_text(json.dumps({"heldout": spec["heldout_A"],
                                "heldout_B": spec["heldout_B"]}))

    gen = (lambda p, s: gen_ollama(model, p, s, url)) if backend == "ollama" \
        else (lambda p, s: gen_openai(model, p, s, url))

    raw = {"model": model, "temperature": spec["temperature"],
           "block_chars": spec["block_chars"], "results": {}}
    for name, cond in spec["conditions"].items():
        for task in ("A", "B"):
            rows = []
            for i in range(spec["n_samples"]):
                t0 = time.time()
                text = gen(cond[f"prompt_{task}"], 9000 + i)
                dt = time.time() - t0
                (out_dir / f"{name}-{task}-gen{i}.md").write_text(text)
                code = extract_code(text)
                if code is None:
                    rows.append({"gen": i, "error": "no python block",
                                 "seconds": round(dt, 1)})
                    continue
                kind = "event" if task == "A" else "click"
                split = "heldout" if task == "A" else "heldout_B"
                s = score(code, kind, str(shim), split)
                s.pop("misses", None)
                rows.append({"gen": i, "seconds": round(dt, 1), **s})
                print(f"{name}-{task} gen{i}: {json.dumps(rows[-1])[:110]}")
            raw["results"][f"{name}/{task}"] = rows
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=1))
    print(f"wrote {out_dir}/raw.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--store", default="runs/wm/baseline-c1/cd82-store.pkl")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--llm-url", default="http://localhost:8000")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-backend", default="openai")
    ap.add_argument("--out", default="results/gap_experiment")
    ap.add_argument("--samples", type=int, default=0,
                    help="override n_samples (plumbing smokes only)")
    args = ap.parse_args()
    if args.build:
        build(args.store)
    if args.run:
        run(args.llm_url, args.llm_model, args.llm_backend, Path(args.out),
            samples_override=args.samples)


if __name__ == "__main__":
    main()
