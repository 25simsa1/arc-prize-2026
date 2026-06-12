"""Scorers (ported from scratch/smoke) + calibrated references and floors.

Generated code is exec'd in a subprocess with a timeout — dev-time only.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS_EVENT = """
import json, sys
HELD = json.loads(open(sys.argv[1]).read())[sys.argv[2]]
tp = tn = fp = fn = err = 0
misses = []
bits = []  # per-item correctness, held-out order (paired bootstrap support)
for h in HELD:
    try:
        pred = predict_event(h["meter_row63"], h["action"])
    except Exception:
        err += 1
        bits.append(0)
        continue
    go = h["event"] == "GAME_OVER"
    pgo = (pred == "GAME_OVER")
    ok = (go == pgo)
    bits.append(1 if ok else 0)
    if go and pgo: tp += 1
    elif go and not pgo: fn += 1; misses.append(h)
    elif not go and pgo: fp += 1; misses.append(h)
    else: tn += 1
n = tp + tn + fp + fn + err
print(json.dumps({"acc": (tp + tn) / n if n else 0.0, "tp": tp, "tn": tn,
                  "fp": fp, "fn": fn, "errors": err, "bits": bits,
                  "misses": misses[:12]}))
"""

HARNESS_CLICK = """
import json, sys
HELD = json.loads(open(sys.argv[1]).read())[sys.argv[2]]
jaccards, offs, ons = [], [], []
errs = 0
for h in HELD:
    try:
        pred = predict_changed_positions(h["click"][0], h["click"][1],
                                         h["clicked_value"],
                                         [tuple(c) for c in h["cells15"]])
        pred = {(int(r), int(c)) for r, c in pred}
    except Exception:
        errs += 1
        continue
    true = {(r, c) for r, c in h["changed_positions"]}
    j = len(pred & true) / (len(pred | true) or 1)
    jaccards.append(j)
    (offs if h["toggle_off"] else ons).append(j)
print(json.dumps({"mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
                  "off_toggle": [round(j, 3) for j in offs],
                  "on_toggle": [round(j, 3) for j in ons], "errors": errs}))
"""


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def score(code: str, kind: str, spec_path: str, split: str = "heldout") -> dict:
    harness = HARNESS_EVENT if kind == "event" else HARNESS_CLICK
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + harness)
        path = f.name
    try:
        out = subprocess.run([sys.executable, path, spec_path, split],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            tail = out.stderr.strip().splitlines()[-1][:200] if out.stderr else "nonzero exit"
            return {"error": tail}
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"harness: {e}"}
    finally:
        Path(path).unlink(missing_ok=True)


REF_EVENT = ('def predict_event(meter_row63, action):\n'
             '    return "GAME_OVER" if meter_row63.count("4") <= 1 else "NONE"')
FLOOR_EVENT = 'def predict_event(m, a):\n    return "NONE"'
REF_CLICK = ('def predict_changed_positions(x, y, v, cells15):\n'
             '    return set(map(tuple, cells15))')


def calibrate(tasks_dir: str = "bench/tasks") -> dict:
    d = Path(tasks_dir)
    out = {
        "A": {
            "reference": score(REF_EVENT, "event", str(d / "task_A.json")),
            "floor": score(FLOOR_EVENT, "event", str(d / "task_A.json")),
            "reference_on_feedback": score(REF_EVENT, "event", str(d / "task_A.json"), "feedback"),
        },
        "B": {"naive_reference": score(REF_CLICK, "click", str(d / "task_B.json"))},
        "reframe": {"naive_reference": score(REF_CLICK, "click", str(d / "task_reframe.json"))},
    }
    (d / "calibration.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=1))
