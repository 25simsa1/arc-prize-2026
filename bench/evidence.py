"""Evidence builders for the proposer bench (ported from scratch/smoke,
which stays as the historical record).

Four tasks, all built from REAL stored evidence (cd82 baseline-c1 store):

  A         event precondition (game-over from the meter row).
            Splits: prompt examples / FEEDBACK set (counterexample source
            for T-repair) / FINAL held-out (the only scored set).
  B         structured click response (row-span encoding; includes the
            pre-frame color-15 cells after the smoke test's fairness fix).
  T-repair  Task A plus a one-round counterexample feedback loop — measures
            the repair-loop lift the thesis depends on (smoke finding: best
            sample was one threshold constant off).
  T-reframe Task B plus the adversarial reframing line (smoke finding:
            models anchor on click-centric geometry the evidence
            contradicts) and TEMPORAL context — each exemplar carries the
            transition observed immediately before it. Temporal order =
            store insertion order, which is trajectory order for our
            single-play stores (logged assumption).

Serialization stays row-span ("r7:35-47"): raw coordinate JSON silently
blew the context window in the smoke test.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.wm.store import TransitionStore  # noqa: E402

DEFAULT_STORE = "runs/wm/baseline-c1/cd82-store.pkl"
rng = random.Random(0)


def spans(cells) -> str:
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(int(r), []).append(int(c))
    parts = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        runs, start, prev = [], cols[0], cols[0]
        for c in cols[1:] + [None]:
            if c is None or c != prev + 1:
                runs.append(f"{start}" if start == prev else f"{start}-{prev}")
                if c is not None:
                    start = c
            prev = c if c is not None else prev
        parts.append(f"r{r}:" + ",".join(runs))
    return " ".join(parts)


def meter_str(t) -> str:
    return "".join(f"{int(v):x}" for v in t.pre[63])


def fours_left(t) -> int:
    return int((t.pre[63] == 4).sum())


# ----------------------------------------------------------------- task A

def build_task_a(store: TransitionStore) -> dict:
    gos = [t for t in store.all() if t.event == "GAME_OVER"]
    nones = [t for t in store.all() if t.event == "NONE"]
    near = [t for t in nones if 2 <= fours_left(t) <= 6]
    mid = [t for t in nones if 7 <= fours_left(t) <= 40]
    full = [t for t in nones if fours_left(t) > 40]
    for pool in (gos, near, mid, full):
        rng.shuffle(pool)

    ex = gos[:8] + near[:3] + mid[:3] + full[:2]
    feedback = gos[8:23] + near[3:8] + mid[3:10] + full[2:5]   # 15 GO / 15 NONE
    final = gos[23:38] + near[8:13] + mid[10:17] + full[5:8]   # 15 GO / 15 NONE

    def fmt(t):
        return {"meter_row63": meter_str(t), "action": t.base_action, "event": t.event}

    lines = [json.dumps(fmt(t)) for t in ex]
    rng.shuffle(lines)
    prompt = f"""You are reverse-engineering the rules of an unknown 64x64 grid game from observed transitions.

Each observation below is one action taken in the game. `meter_row63` is the bottom row of the screen (64 cells, one hex digit per cell color). `event` is what happened: "GAME_OVER" or "NONE".

OBSERVATIONS:
{chr(10).join(lines)}

Write a single Python function that predicts the event BEFORE the action is taken:

```python
def predict_event(meter_row63: str, action: str) -> str:
    # return "GAME_OVER" or "NONE"
```

Hypothesize the actual game rule (you may posit counters derived from the meter contents). Reply with ONLY the function in a ```python code block, plus one comment line stating your hypothesized rule."""
    return {
        "prompt": prompt,
        "feedback": [fmt(t) for t in feedback],
        "heldout": [fmt(t) for t in final],
    }


def build_repair_feedback(taskA: dict, code_misses: list[dict]) -> str:
    """Round-2 prompt for T-repair: original prompt + the model's function +
    up to 6 counterexamples it got wrong on the FEEDBACK split."""
    ce = "\n".join(json.dumps(m) for m in code_misses[:6])
    return (
        "Your previous function misclassified the following observations "
        "(same format; `event` is the TRUE outcome your function failed to "
        f"predict):\n{ce}\n\n"
        "Revise your hypothesis to account for these counterexamples. Reply "
        "with ONLY the corrected function in a ```python code block, plus "
        "one comment line stating the revised rule."
    )


# ----------------------------------------------------------------- task B

def _click_obs(store: TransitionStore):
    content = np.arange(64)[:, None] < 63
    order = []  # insertion order == trajectory order for single-play stores
    for idx, t in enumerate(store.all()):
        if t.base_action != "ACTION6":
            continue
        diff = np.argwhere((t.pre != t.post) & content)
        if len(diff) < 10:
            continue
        x, y = t.click_xy
        c15 = np.argwhere((t.pre == 15) & content)
        order.append({
            "store_idx": idx,
            "click_x": x, "click_y": y,
            "clicked_value": int(t.pre[y, x]) if 0 <= y < 64 and 0 <= x < 64 else -1,
            "cells15": [[int(a), int(b)] for a, b in c15],
            "toggle_off": bool((t.pre[diff[:, 0], diff[:, 1]] == 15).any()),
            "changed": [[int(a), int(b)] for a, b in diff],
        })
    return order


def build_task_b(store: TransitionStore, reframe: bool = False) -> dict:
    obs = _click_obs(store)
    rng2 = random.Random(1)
    rng2.shuffle(obs)
    offs = [c for c in obs if c["toggle_off"]]
    ons = [c for c in obs if not c["toggle_off"]]
    examples = offs[:4] + ons[:3]
    heldout = offs[4:7] + ons[3:6]
    rng2.shuffle(examples)

    all_by_idx = {c["store_idx"]: c for c in obs}

    def prev_summary(c):
        prior = [i for i in all_by_idx if i < c["store_idx"]]
        if not prior:
            return "none"
        p = all_by_idx[max(prior)]
        return json.dumps({"click": [p["click_x"], p["click_y"]],
                           "changed_positions": spans(p["changed"])})

    ex_lines = []
    for c in examples:
        d = {
            "click": [c["click_x"], c["click_y"]],
            "clicked_value": c["clicked_value"],
            "color15_cells_before": spans(c["cells15"]),
            "changed_positions": spans(c["changed"]),
        }
        if reframe:
            d["previous_transition"] = prev_summary(c)
        ex_lines.append(json.dumps(d))

    reframe_note = ""
    if reframe:
        reframe_note = """
IMPORTANT, learned from failed prior attempts: hypotheses anchored on
geometry near the click coordinate (strips, rectangles, reflections around
the click) were all wrong — the changed cells are typically FAR from the
click. Anchor your hypothesis on the color-15 pattern state and how it
evolves between transitions, not on click geometry.
"""

    prompt = f"""You are reverse-engineering an unknown 64x64 grid game. Clicking certain cells causes a structured change elsewhere on the board.

Below are observed click transitions. Cell sets are encoded as row spans: "r7:35-38,40" means row 7, columns 35,36,37,38 and 40. Each observation lists the click coordinate (x=column, y=row), the color of the clicked cell, all cells showing color 15 BEFORE the click, and the cells that changed.{reframe_note}
OBSERVATIONS:
{chr(10).join(ex_lines)}

Write a single Python function predicting WHICH CELLS will change for a new click:

```python
def predict_changed_positions(click_x: int, click_y: int, clicked_value: int,
                              cells_with_color15_before: list[tuple[int, int]]) -> set[tuple[int, int]]:
    # return the set of (row, col) positions expected to change
```

Look for the underlying structure. Reply with ONLY the function in a ```python code block, plus one comment line stating your hypothesized rule."""
    return {
        "prompt": prompt,
        "heldout": [{"click": [c["click_x"], c["click_y"]],
                     "clicked_value": c["clicked_value"],
                     "cells15": c["cells15"],
                     "toggle_off": c["toggle_off"],
                     "changed_positions": c["changed"]}
                    for c in heldout],
    }


# ----------------------------------------------------------------- assembly

def build_all(store_path: str = DEFAULT_STORE, out_dir: str = "bench/tasks") -> dict:
    store = TransitionStore.load(store_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {
        "A": build_task_a(store),
        "B": build_task_b(store, reframe=False),
        "reframe": build_task_b(store, reframe=True),
    }
    # T-repair reuses task A's artifacts; its round-2 prompt is built at run
    # time from each model's own misses on the feedback split.
    manifest = {}
    for name, spec in tasks.items():
        (out / f"task_{name}.json").write_text(json.dumps(spec, indent=1))
        manifest[name] = {
            "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest()[:16],
            "prompt_chars": len(spec["prompt"]),
            "heldout": len(spec["heldout"]),
            "store": store_path,
        }
    manifest["repair"] = {"derived_from": "A", "feedback_set": len(tasks["A"]["feedback"]),
                          "max_counterexamples": 6}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=1))
