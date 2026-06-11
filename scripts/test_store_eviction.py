"""Eviction-policy test: the cap must keep proposer food and stay honest.

Asserts: (1) size respects the cap; (2) LEVEL/WIN/GAME_OVER transitions,
conflict pairs, and sole/pair-changer evidence survive eviction; (3) what
gets evicted is ordinary NONE multi-cell transitions, oldest first;
(4) conflict detection still works on retained keys after evictions;
(5) the event census is unaffected by eviction.

    .venv/bin/python scripts/test_store_eviction.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.store import TransitionStore, frame_hash


def grid(seed: int) -> np.ndarray:
    # injective for any 64-consecutive-seed window (flat position) plus a
    # value channel for wider ranges — a periodic generator here silently
    # collapses the flood into dups and the cap never engages
    g = np.zeros((8, 8), dtype=np.int16)
    g.flat[seed % 64] = 3
    g[0, 0] = (seed // 64) % 14 + 1
    return g


def main() -> None:
    store = TransitionStore("evict-test", max_transitions=30)

    # 3 LEVEL transitions (protected)
    for i in range(3):
        store.add(0, grid(i), "ACTION1", grid(i + 50), 1, "NOT_FINISHED")
    # 2 sole-changer transitions (protected: diff <= 2)
    for i in range(3, 5):
        g = grid(i)
        post = g.copy()
        post[7, 7] = 9
        store.add(0, g, "ACTION2", post, 0, "NOT_FINISHED")
    # 1 conflict pair on an early key (protected once conflicted)
    g_conf = grid(99)
    store.add(0, g_conf, "ACTION3", grid(60), 0, "NOT_FINISHED")
    s, _ = store.add(0, g_conf, "ACTION3", grid(61), 0, "NOT_FINISHED")
    assert s == "conflict" and len(store.conflicts) == 1

    # flood with 60 ordinary NONE multi-cell transitions -> forces eviction
    for i in range(100, 160):
        store.add(0, grid(i), "ACTION4", grid(i + 200) + 1, 0, "NOT_FINISHED")

    assert len(store) <= 30, f"cap violated: {len(store)}"
    assert store.evicted_total > 0, "nothing was evicted"

    kept_events = [t for t in store.all() if t.event != "NONE"]
    assert len(kept_events) == 3, f"LEVEL evidence evicted! kept={len(kept_events)}"
    kept_sole = [t for t in store.all() if 0 <= t.diff_cells <= 2 and t.event == "NONE"]
    assert len(kept_sole) >= 2, "sole-changer evidence evicted"
    assert store.lookup(0, store.conflicts[0]["key"].split("'")[1], "ACTION3") is not None or any(
        t.action_key == "ACTION3" for t in store.all()
    ), "conflicted transition evicted"

    # conflict detection still works post-eviction on a retained key
    survivor = next(t for t in store.all() if t.event == "NONE" and t.diff_cells > 2)
    s, _ = store.add(survivor.level, survivor.pre, survivor.action_key,
                     grid(7) + 5, 0, "GAME_OVER")
    assert s == "conflict", f"conflict detection broken post-eviction: {s}"

    # census reflects ALL observations incl. evicted ones
    assert store.event_counts["NONE"] >= 60, store.event_counts
    assert store.event_counts["LEVEL"] == 3

    print(f"size={len(store)} evicted={store.evicted_total} "
          f"census={store.event_counts} conflicts={len(store.conflicts)}")

    # ---- persistence round-trip must preserve the eviction machinery ----
    # load() that doesn't rebuild the evictable queue leaves a capped store
    # unable to accept ANY new transition (everything "protected" by absence);
    # conflict-key protection and the observation census must survive too.
    with tempfile.TemporaryDirectory() as td:
        pkl = Path(td) / "store.pkl"
        store.save(pkl)
        loaded = TransitionStore.load(pkl)
        loaded.max_transitions = 30
        assert len(loaded) == len(store)
        assert loaded.appended_total == store.appended_total, (
            f"appended_total lost on load: {loaded.appended_total}")
        assert loaded.event_counts == store.event_counts, (
            f"event census lost on load: {loaded.event_counts}")

        for i in range(300, 340):  # flood: must evict, not refuse
            loaded.add(0, grid(i), "ACTION5", grid(i + 200) + 1, 0, "NOT_FINISHED")
        assert len(loaded) <= 30, f"cap violated post-load: {len(loaded)}"
        assert loaded.capped_drops == 0, (
            f"post-load adds refused (evictable queue not rebuilt): "
            f"{loaded.capped_drops} drops")
        assert loaded.evicted_total > 0, "nothing evicted post-load"
        kept_events = [t for t in loaded.all() if t.event != "NONE"]
        assert len(kept_events) == 3, (
            f"LEVEL evidence evicted post-load! kept={len(kept_events)}")
        # both conflicted keys (ACTION3 from setup, survivor's from above)
        # are unprotected by diff_cells/event — only persisted conflict keys
        # keep them alive through the post-load flood
        assert loaded.lookup(0, frame_hash(g_conf), "ACTION3") is not None, (
            "conflicted transition evicted post-load (conflict keys not "
            "persisted)")

        # legacy pickles (no census/conflict-key fields) must still load and
        # evict — fallbacks: appended_total=len, census rebuilt from retained
        legacy_pkl = Path(td) / "legacy.pkl"
        with open(legacy_pkl, "wb") as f:
            pickle.dump({"game_id": store.game_id,
                         "transitions": list(store.by_key.values()),
                         "conflicts": store.conflicts}, f)
        old = TransitionStore.load(legacy_pkl)
        old.max_transitions = 30
        assert old.appended_total == len(old)
        for i in range(400, 420):
            old.add(0, grid(i), "ACTION5", grid(i + 200) + 2, 0, "NOT_FINISHED")
        assert len(old) <= 30 and old.evicted_total > 0 and old.capped_drops == 0

    print("PERSISTENCE ROUND-TRIP PASS")
    print("EVICTION TEST PASS")


if __name__ == "__main__":
    main()
