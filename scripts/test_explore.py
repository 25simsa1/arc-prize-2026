"""Unit tests for the exploration substrate (harness/wm/explore.py) and a
forced Go-Explore integration check that the return-via-replay path executes.

    .venv/bin/python scripts/test_explore.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.wm.explore import (Archive, components, segment_signatures,
                                state_novelty, tiered_click_candidates,
                                _TIER_LATTICE)

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# grid: bg=0, a 1-cell object (color 3, tier 1) and a 9x9=81-cell block
# (color 5, tier >1)
g = np.zeros((32, 32), dtype=np.int16)
g[2, 2] = 3
g[10:19, 10:19] = 5

print("components / signatures")
comps = components(g)
check("two components found", len(comps) == 2, str(len(comps)))
# translation invariance: a 1-cell object elsewhere -> same signature
g2 = np.zeros((32, 32), dtype=np.int16)
g2[20, 20] = 3
sig_dot = next(c.sig for c in components(g2))
sig_dot_orig = next(c.sig for c in comps if c.size == 1)
check("dot sig translation-invariant", sig_dot == sig_dot_orig)
check("distinct shapes distinct sigs",
      len({c.sig for c in comps}) == 2)

print("tiered click candidates")
cands = tiered_click_candidates(g)
tiers = [t for t, _ in cands]
small_tier = min(t for t, a in cands if a == "ACTION6:2,2")     # 1-cell -> tier 1
big_tier = min(t for t, a in cands if a == "ACTION6:14,14")     # 81-cell block
check("small component tier < big component tier",
      small_tier < big_tier, f"{small_tier} vs {big_tier}")
check("lattice floor present", any(t == _TIER_LATTICE for t in tiers))
# uncapped: candidates = every segment + every lattice cell (no cap-24)
n_lat = sum(1 for t in tiers if t == _TIER_LATTICE)
check("not capped (all segments + all lattice)", len(cands) == 2 + n_lat,
      f"{len(cands)} = 2 + {n_lat}")
# hud_mask suppresses masked clicks
mask = np.zeros((32, 32), dtype=bool)
mask[2, 2] = True
cands_m = tiered_click_candidates(g, hud_mask=mask)
check("masked component click suppressed",
      "ACTION6:2,2" not in [a for _, a in cands_m])

print("segment novelty")
visits = {}
for s in segment_signatures(g):
    visits[s] = 0
check("all-rare frame novelty = segment count",
      state_novelty(g, visits) == len(segment_signatures(g)))
for s in list(visits):
    visits[s] = 9
check("well-visited frame novelty = 0", state_novelty(g, visits) == 0)

print("archive consider / pick_return")
ar = Archive()
ga = np.zeros((16, 16), dtype=np.int16); ga[3, 3] = 7
gb = np.zeros((16, 16), dtype=np.int16); gb[4, 4] = 8
# cell A reachable by a long prefix, then a shorter one -> keep shorter
ar.consider(ga, "ha", 0, prefix=("ACTION1",) * 10, novelty=2,
            near_event=False, meter_extreme=False)
ar.consider(ga, "ha", 0, prefix=("ACTION2",) * 3, novelty=2,
            near_event=False, meter_extreme=False)
ka = Archive.cell_key(ga)
check("shorter prefix kept", ar.entries[ka].prefix_len == 3,
      str(ar.entries[ka].prefix_len))
# cell B: long prefix, high interest (near_event)
ar.consider(gb, "hb", 0, prefix=("ACTION1",) * 8, novelty=5,
            near_event=True, meter_extreme=False)
# uncapped: interest dominates -> B chosen
e_unc = ar.pick_return(0, cap_remaining=None, capped=False)
check("uncapped picks high-interest cell", Archive.cell_key(gb) == e_unc.cell)
# capped with a tight window: only the short-prefix cell A fits (3+2<=6)
e_cap = ar.pick_return(0, cap_remaining=6, capped=True)
check("capped picks short-prefix fitting cell", e_cap.cell == ka,
      e_cap.cell if e_cap else None)
# capped with no room: nothing returnable
check("capped excludes when prefix won't fit",
      ar.pick_return(0, cap_remaining=3, capped=True) is None)

print("Go-Explore return-via-replay path executes (direct drive on live frames)")
# Frontier exhaustion (the natural trigger) is rare while tier coverage keeps
# finding novelty, so we drive the consumption path directly: run a live game
# a few steps to populate the archive, force a return queue, then verify the
# leading RESET and the replayed prefix actions emit with source "return"
# without any crash / off-menu assertion.
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine import GameAction
from harness.agents.wm_agent import WorldModelAgent

ENV = str(Path.home() / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files")
arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV)
card = arc.open_scorecard(tags=["explore-forced"])
env = arc.make("ls20", scorecard_id=card, include_frame_data=False)
agent = WorldModelAgent("ls20", 0, proposer="template", time_budget_s=30,
                        persistence_probe=False, dev_mode=True)
agent.on_play_start(0)
fd = env.observation_space
for _ in range(120):                      # populate the archive
    act, data = agent.choose_action([fd], fd)
    fd = env.reset() if act == GameAction.RESET else env.step(act, data)
ge_entry = agent.archive.pick_return(fd.levels_completed, None, False)
check("archive has a returnable cell after exploration", ge_entry is not None,
      f"cells={len(agent.archive.entries)}")
if ge_entry is not None:
    agent._return_queue = ["RESET"] + list(ge_entry.action_prefix)[:5]
    seen_return = False
    crashed = False
    try:
        for _ in range(len(agent._return_queue) + 1):
            act, data = agent.choose_action([fd], fd)
            if agent._pending and agent._pending.get("source") == "return":
                seen_return = True
            fd = env.reset() if act == GameAction.RESET else env.step(act, data)
    except Exception as e:
        crashed = True
        print("    crash:", e)
    check("return-via-replay emitted 'return'-sourced actions, no crash",
          seen_return and not crashed)
arc.close_scorecard(card)

n_fail = sum(1 for ok in PASS if not ok)
print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed")
sys.exit(1 if n_fail else 0)
