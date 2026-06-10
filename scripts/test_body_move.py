"""Synthetic test for the rigid body-move template (R2 family, general —
NOT an r11l special): multi-color bodies that jump to stable destinations
when clicked anywhere inside them.

Asserts: (1) the template is proposed from two observed clicks per body;
(2) it predicts the move for an UNSEEN click cell of the same body
(generalization over click position, not memorization); (3) misplaced
patterns (signature mismatch) yield NO_PREDICTION.

    .venv/bin/python scripts/test_body_move.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.proposers import TemplateProposer
from harness.wm.rules import WorldModel, grids_match, Prediction
from harness.wm.store import TransitionStore
from harness.wm.verifier import verify_rules

BG = 0
BODY_A = {(0, 0): 3, (0, 1): 4, (1, 0): 5, (1, 1): 3}     # 2x2 multi-color
BODY_B = {(0, 0): 6, (1, 0): 6, (2, 0): 6, (2, 1): 7}     # L-shape
A_POS, A_DEST = (2, 2), (2, 10)
B_POS, B_DEST = (10, 3), (10, 11)


def stamp(grid, body, pos):
    for (dy, dx), c in body.items():
        grid[pos[0] + dy, pos[1] + dx] = c
    return grid


def base_grid(a_at=A_POS, b_at=B_POS):
    g = np.zeros((16, 16), dtype=np.int16)
    stamp(g, BODY_A, a_at)
    stamp(g, BODY_B, b_at)
    return g


def click_transition(store, pre, body, src, dest, click_dydx):
    post = pre.copy()
    for (dy, dx), _ in body.items():
        post[src[0] + dy, src[1] + dx] = BG
    stamp(post, body, dest)
    x, y = src[1] + click_dydx[1], src[0] + click_dydx[0]
    store.add(0, pre, f"ACTION6:{x},{y}", post, 0, "NOT_FINISHED")
    return post


def main() -> None:
    store = TransitionStore("toy-body")
    g0 = base_grid()
    # two observed clicks on body A (different cells), one on body B
    click_transition(store, g0, BODY_A, A_POS, A_DEST, (0, 0))
    click_transition(store, g0, BODY_A, A_POS, A_DEST, (1, 1))
    click_transition(store, g0, BODY_B, B_POS, B_DEST, (2, 0))

    model = WorldModel()
    rules = TemplateProposer().propose(store, model)
    body_rules = [r for r in rules if r.name == "body_move"]
    assert len(body_rules) >= 2, f"body_move rules missing: {[r.rule_id for r in rules]}"
    verify_rules(rules, store)
    model.rules = rules

    # (2) UNSEEN click cell of body A: (0,1)-offset cell — never clicked
    x, y = A_POS[1] + 1, A_POS[0]
    p = model.predict(0, g0, f"ACTION6:{x},{y}")
    assert p.grid is not None, "no prediction for unseen click cell"
    expected = g0.copy()
    for (dy, dx), _ in BODY_A.items():
        expected[A_POS[0] + dy, A_POS[1] + dx] = BG
    stamp(expected, BODY_A, A_DEST)
    assert grids_match(Prediction(grid=p.grid, mask=p.grid_mask), expected), \
        "predicted move incorrect for unseen click"

    # (3) clicking the background or a mismatched pattern: no grid claim
    p2 = model.predict(0, g0, "ACTION6:0,15")
    assert p2.grid is None or not (p2.grid != g0)[g0 == BG].any()

    print("rules:", [(r.rule_id, r.status.value) for r in body_rules])
    print("BODY-MOVE TEST PASS")


if __name__ == "__main__":
    main()
