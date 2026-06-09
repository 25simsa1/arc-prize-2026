"""Bench runner: N samples per task per model, two backends.

  --backend ollama   local dev (Mac): http://localhost:11434
  --backend openai   any OpenAI-compatible server (vLLM on the rental box).
                     DEV-TIME MEASUREMENT ONLY — the import-graph rule
                     (no API clients in agent code) is untouched; this
                     module is bench/, never imported by harness/.

Tasks: A, B, reframe (single-round) and repair (two rounds: initial sample
scored on the FEEDBACK split; its misses become counterexamples in a
round-2 prompt; only the FINAL held-out is reported, before vs after).

    .venv/bin/python bench/run_bench.py --model qwen2.5-coder:14b \
        --backend ollama --tasks A B reframe repair --samples 8
"""

import argparse
import json
import time
from pathlib import Path

import requests

from evidence import build_repair_feedback  # bench-local import
from scoring import extract_code, score

HERE = Path(__file__).parent
TASKS_DIR = HERE / "tasks"
TEMPERATURE = 0.8  # logged in every results file


def gen_ollama(model: str, prompt: str, seed: int, url: str) -> str:
    r = requests.post(f"{url}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": seed,
                    "num_ctx": 16384, "num_predict": 1500},
    }, timeout=900)
    r.raise_for_status()
    return r.json()["response"]


def gen_openai(model: str, prompt: str, seed: int, url: str) -> str:
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE, "seed": seed, "max_tokens": 1500,
    }, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_single(task: str, kind: str, model: str, gen, n: int, out_dir: Path) -> list[dict]:
    spec_path = str(TASKS_DIR / f"task_{task}.json")
    prompt = json.loads(Path(spec_path).read_text())["prompt"]
    rows = []
    for i in range(n):
        t0 = time.time()
        text = gen(prompt, 1000 + i)
        dt = time.time() - t0
        (out_dir / f"{task}-gen{i:02d}.md").write_text(text)
        code = extract_code(text)
        if len(text.strip()) < 50:
            s = {"error": f"empty/truncated ({len(text)} chars)"}
        elif code is None:
            s = {"error": "no python block"}
        else:
            s = score(code, kind, spec_path)
            s.pop("misses", None)
        rows.append({"gen": i, "seconds": round(dt, 1), **s})
        print(f"{model} {task} gen{i:02d} ({dt:5.1f}s): {json.dumps(rows[-1])[:140]}")
    return rows


def run_repair(model: str, gen, n: int, out_dir: Path) -> list[dict]:
    spec_path = str(TASKS_DIR / "task_A.json")
    spec = json.loads(Path(spec_path).read_text())
    rows = []
    for i in range(n):
        text = gen(spec["prompt"], 2000 + i)
        (out_dir / f"repair-r1-gen{i:02d}.md").write_text(text)
        code = extract_code(text)
        if code is None:
            rows.append({"gen": i, "error": "no python block (round 1)"})
            print(f"{model} repair gen{i:02d}: no code r1")
            continue
        before = score(code, "event", spec_path)            # final held-out
        fb = score(code, "event", spec_path, "feedback")    # counterexample source
        misses = fb.pop("misses", []) if isinstance(fb, dict) else []
        if not misses:
            rows.append({"gen": i, "before": before, "after": before,
                         "note": "no feedback misses; round 2 skipped"})
            print(f"{model} repair gen{i:02d}: clean on feedback, before==after")
            continue
        prompt2 = spec["prompt"] + "\n\n" + (
            f"```python\n{code}```\n\n" + build_repair_feedback(spec, misses)
        )
        text2 = gen(prompt2, 3000 + i)
        (out_dir / f"repair-r2-gen{i:02d}.md").write_text(text2)
        code2 = extract_code(text2)
        after = ({"error": "no python block (round 2)"} if code2 is None
                 else score(code2, "event", spec_path))
        if isinstance(after, dict):
            after.pop("misses", None)
        before.pop("misses", None)
        rows.append({"gen": i, "before": before, "after": after,
                     "n_counterexamples": len(misses)})
        print(f"{model} repair gen{i:02d}: before={before.get('acc')} after={after.get('acc')}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--url", default=None)
    ap.add_argument("--tasks", nargs="+", default=["A", "B", "reframe", "repair"])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = args.url or ("http://localhost:11434" if args.backend == "ollama" else "http://localhost:8000")
    gen = (lambda p, s: gen_ollama(args.model, p, s, url)) if args.backend == "ollama" \
        else (lambda p, s: gen_openai(args.model, p, s, url))

    safe = args.model.replace("/", "_").replace(":", "_")
    out_dir = Path(args.out or (HERE / "results" / safe))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"model": args.model, "backend": args.backend,
               "temperature": TEMPERATURE, "samples": args.samples,
               "manifest": json.loads((TASKS_DIR / "manifest.json").read_text())}
    for task in args.tasks:
        if task == "repair":
            results["repair"] = run_repair(args.model, gen, args.samples, out_dir)
        else:
            kind = "event" if task == "A" else "click"
            results[task] = run_single(task, kind, args.model, gen, args.samples, out_dir)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {out_dir}/results.json")


if __name__ == "__main__":
    main()
