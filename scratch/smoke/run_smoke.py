"""THROWAWAY smoke-test runner. Samples one local model via ollama, scores
generations against held-out evidence with hand-rolled checks, saves every
generation verbatim. Quick and dirty by design — generated code is exec'd in
a subprocess with a timeout, nothing more (local model output, scratch only).

    .venv/bin/python scratch/smoke/run_smoke.py --model qwen2.5-coder:14b --samples 10
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434/api/generate"


def sample(model: str, prompt: str, seed: int) -> str:
    r = requests.post(OLLAMA, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.8, "seed": seed, "num_ctx": 16384,
                    "num_predict": 1500},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def score_in_subprocess(code: str, harness: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + harness)
        path = f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return {"error": out.stderr.strip().splitlines()[-1][:200] if out.stderr else "nonzero exit"}
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"harness: {e}"}
    finally:
        Path(path).unlink(missing_ok=True)


HARNESS_A = """
import json
HELD = json.loads(open({heldout_path!r}).read())["heldout"]
tp = tn = fp = fn = err = 0
for h in HELD:
    try:
        pred = predict_event(h["meter_row63"], h["action"])
    except Exception:
        err += 1
        continue
    go = h["event"] == "GAME_OVER"
    pgo = (pred == "GAME_OVER")
    if go and pgo: tp += 1
    elif go and not pgo: fn += 1
    elif not go and pgo: fp += 1
    else: tn += 1
n = tp + tn + fp + fn + err
print(json.dumps({{"acc": (tp + tn) / n if n else 0.0,
                  "tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": err}}))
"""

HARNESS_B = """
import json
HELD = json.loads(open({heldout_path!r}).read())["heldout"]
jaccards, offs, ons = [], [], []
errs = 0
for h in HELD:
    try:
        pred = predict_changed_positions(h["click"][0], h["click"][1],
                                         h["clicked_value"],
                                         [tuple(c) for c in h["cells15"]])
        pred = {{(int(r), int(c)) for r, c in pred}}
    except Exception:
        errs += 1
        continue
    true = {{(r, c) for r, c in h["changed_positions"]}}
    j = len(pred & true) / (len(pred | true) or 1)
    jaccards.append(j)
    (offs if h["toggle_off"] else ons).append(j)
print(json.dumps({{"mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
                   "off_toggle": [round(j, 3) for j in offs],
                   "on_toggle": [round(j, 3) for j in ons], "errors": errs}}))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", default=["A", "B"])
    args = ap.parse_args()

    results = {}
    for task in args.tasks:
        spec_path = HERE / f"task{task}.json"
        spec = json.loads(spec_path.read_text())
        harness = (HARNESS_A if task == "A" else HARNESS_B).format(
            heldout_path=str(spec_path)
        )
        out_dir = HERE / f"task{task}-gens"
        out_dir.mkdir(exist_ok=True)
        rows = []
        for i in range(args.samples):
            t0 = time.time()
            text = sample(args.model, spec["prompt"], seed=1000 + i)
            dt = time.time() - t0
            (out_dir / f"gen{i:02d}.md").write_text(text)
            code = extract_code(text)
            if len(text.strip()) < 50:
                score = {"error": f"empty/truncated response ({len(text)} chars)"}
            elif code is None:
                score = {"error": "no python block"}
            else:
                score = score_in_subprocess(code, harness)
            rows.append({"gen": i, "seconds": round(dt, 1), **score})
            print(f"task{task} gen{i:02d} ({dt:5.1f}s): {json.dumps(score)}")
        results[task] = rows
        (out_dir / "scores.json").write_text(json.dumps(rows, indent=1))

    (HERE / "summary.json").write_text(json.dumps(
        {"model": args.model, "results": results}, indent=1))
    print("\nsaved scratch/smoke/summary.json")


if __name__ == "__main__":
    main()
