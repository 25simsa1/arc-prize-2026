"""THROWAWAY smoke-test evidence builder (Workstream D gate). Not harness code.

Builds two prompts from REAL stored evidence (runs/wm/baseline-c1 stores):

  Task A — latent/structured EVENT PRECONDITION. The spec wanted sb26's
  level-advance condition, but sb26 has ZERO level-advance observations in
  any store (31,469 transitions, 0 LEVEL events) — you cannot prompt for
  evidence that was never gathered. Substituted: cd82's GAME_OVER
  precondition (314 positive examples; the meter row makes it learnable).
  Held-out is adversarial: NONE examples include nearly-drained meters so
  "any drained cell => game over" cannot win for free.

  Task B — STRUCTURED RESPONSE: predict the changed-cell positions of a
  held-out cd82 click from 7 example click transitions.

    .venv/bin/python scratch/smoke/build_evidence.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from harness.wm.store import TransitionStore

OUT = Path(__file__).parent
rng = random.Random(0)


def meter_str(t) -> str:
    return "".join(f"{int(v):x}" for v in t.pre[63])


def fours_left(t) -> int:
    return int((t.pre[63] == 4).sum())


def build_task_a(store: TransitionStore) -> dict:
    gos = [t for t in store.all() if t.event == "GAME_OVER"]
    nones = [t for t in store.all() if t.event == "NONE"]
    # bucket NONE by meter drain so examples + heldout cover the boundary
    nones_near = [t for t in nones if 2 <= fours_left(t) <= 6]
    nones_full = [t for t in nones if fours_left(t) > 40]
    nones_mid = [t for t in nones if 7 <= fours_left(t) <= 40]
    rng.shuffle(gos), rng.shuffle(nones_near), rng.shuffle(nones_full), rng.shuffle(nones_mid)

    ex_go, ex_none = gos[:8], nones_near[:3] + nones_mid[:3] + nones_full[:2]
    held_go = gos[8:38]
    held_none = nones_near[3:13] + nones_mid[3:18] + nones_full[2:7]

    def fmt(t):
        return {"meter_row63": meter_str(t), "action": t.base_action,
                "event": t.event}

    lines = [json.dumps(fmt(t)) for t in ex_go + ex_none]
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
        "heldout": [fmt(t) for t in held_go + held_none],
        "n_heldout_go": len(held_go),
        "n_heldout_none": len(held_none),
    }


def build_task_b(store: TransitionStore) -> dict:
    """Fairness note (found by scorer validation): the click response depends
    on game state — a fixed-set hypothesis from examples scores 0.0 on
    held-out clicks from other phases. Each observation therefore includes
    the pre-frame's color-15 cell positions (the toggle's visible state).
    The off-toggle direction (15 -> gone) becomes learnable from evidence;
    the on-toggle direction stays legitimately hard (the appearing pattern
    is not visible in the pre-frame). Held-out mixes both directions."""
    content = np.arange(64)[:, None] < 63
    clicks = []
    for t in store.all():
        if t.base_action != "ACTION6":
            continue
        diff = np.argwhere((t.pre != t.post) & content)
        if len(diff) < 10:
            continue
        x, y = t.click_xy
        c15 = np.argwhere((t.pre == 15) & content)
        toggle_off = bool((t.pre[diff[:, 0], diff[:, 1]] == 15).any())
        clicks.append({
            "click_x": x, "click_y": y,
            "clicked_value": int(t.pre[y, x]) if 0 <= y < 64 and 0 <= x < 64 else -1,
            "cells15": [[int(a), int(b)] for a, b in c15],
            "toggle_off": toggle_off,
            "changed": [[int(a), int(b), int(t.pre[a, b]), int(t.post[a, b])]
                        for a, b in diff],
        })
    rng.shuffle(clicks)
    offs = [c for c in clicks if c["toggle_off"]]
    ons = [c for c in clicks if not c["toggle_off"]]
    examples = offs[:4] + ons[:3]
    heldout = offs[4:7] + ons[3:6]
    rng.shuffle(examples)

    def spans(cells: list[list[int]]) -> str:
        """Compact row-span encoding — raw coordinate JSON blew past the
        model's context (20k chars tokenized >8k tokens; ollama silently
        truncated and the model emitted a bare fence)."""
        by_row: dict[int, list[int]] = {}
        for r, c in cells:
            by_row.setdefault(r, []).append(c)
        parts = []
        for r in sorted(by_row):
            cols = sorted(by_row[r])
            runs, start = [], cols[0]
            prev = start
            for c in cols[1:] + [None]:
                if c is None or c != prev + 1:
                    runs.append(f"{start}" if start == prev else f"{start}-{prev}")
                    if c is not None:
                        start = c
                prev = c if c is not None else prev
            parts.append(f"r{r}:" + ",".join(runs))
        return " ".join(parts)

    ex_lines = []
    for c in examples:
        post15 = [cell[:2] for cell in c["changed"] if cell[3] == 15]
        gone15 = [cell[:2] for cell in c["changed"] if cell[2] == 15]
        ex_lines.append(json.dumps({
            "click": [c["click_x"], c["click_y"]],
            "clicked_value": c["clicked_value"],
            "color15_cells_before": spans(c["cells15"]),
            "changed_positions": spans([cell[:2] for cell in c["changed"]]),
            "cells_becoming_15": spans(post15) if post15 else "",
            "cells_losing_15": spans(gone15) if gone15 else "",
        }))
    prompt = f"""You are reverse-engineering an unknown 64x64 grid game. Clicking certain cells causes a structured change elsewhere on the board.

Below are observed click transitions. Cell sets are encoded as row spans: "r7:35-38,40" means row 7, columns 35,36,37,38 and 40. Each observation lists the click coordinate (x=column, y=row), the color of the clicked cell, all cells showing color 15 BEFORE the click, and the cells that changed.

OBSERVATIONS:
{chr(10).join(ex_lines)}

Write a single Python function predicting WHICH CELLS will change for a new click:

```python
def predict_changed_positions(click_x: int, click_y: int, clicked_value: int,
                              cells_with_color15_before: list[tuple[int, int]]) -> set[tuple[int, int]]:
    # return the set of (row, col) positions expected to change
```

Look for the underlying structure (toggles, fixed patterns, relations between the color-15 cells and the changed set). Reply with ONLY the function in a ```python code block, plus one comment line stating your hypothesized rule."""

    return {
        "prompt": prompt,
        "heldout": [{"click": [c["click_x"], c["click_y"]],
                     "clicked_value": c["clicked_value"],
                     "cells15": c["cells15"],
                     "toggle_off": c["toggle_off"],
                     "changed_positions": [[r, ccol] for r, ccol, _, _ in c["changed"]]}
                    for c in heldout],
    }


def main() -> None:
    store = TransitionStore.load("runs/wm/baseline-c1/cd82-store.pkl")
    a = build_task_a(store)
    b = build_task_b(store)
    (OUT / "taskA.json").write_text(json.dumps(a, indent=1))
    (OUT / "taskB.json").write_text(json.dumps(b, indent=1))
    print(f"taskA: prompt {len(a['prompt'])} chars, heldout {len(a['heldout'])} "
          f"({a['n_heldout_go']} GO / {a['n_heldout_none']} NONE)")
    print(f"taskB: prompt {len(b['prompt'])} chars, heldout {len(b['heldout'])} clicks")


if __name__ == "__main__":
    main()
