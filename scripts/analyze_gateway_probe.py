"""Turn returned probe artifacts into the five gateway verdicts (REPORT §7).

Inputs (drop whatever you got into results/gateway_probe/, then run this):
  - log.txt        : raw notebook/rerun log text, IF owner-visible. We grep
                     it for the probe SENTINEL summary lines.
  - scorecard.json : the closed scorecard (per-game / per-level), if exposed.
  - score.txt      : the single leaderboard aggregate (free-text fine), for
                     the score-encoding fallback path.
  - control_score.txt (optional): the frugal control run's aggregate, for
                     the Probe-2 comparison.

    .venv/bin/python scripts/analyze_gateway_probe.py

Prints a verdict per question with the evidence it used, and writes
results/gateway_probe/verdicts.json. Every verdict is one of
CONFIRMED / REFUTED / INCONCLUSIVE with the reason — never guesses.
"""

import json
import re
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "results/gateway_probe"
SENTINEL = "PROBE_SENTINEL_7Qf3xR2"
MAX_HUMAN_BASELINE = 578           # observed max over public set
CUTOFF_REACH = 5 * MAX_HUMAN_BASELINE  # 2890; the probe caps at 4000


def read(name):
    p = DIR / name
    return p.read_text() if p.exists() else None


def parse_summaries(log):
    """Extract the per-game JSON the probe printed (lines: '<SENTINEL> SUMMARY {..}')."""
    out = []
    if not log:
        return out
    for m in re.finditer(re.escape(SENTINEL) + r" SUMMARY (\{.*\})", log):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return out


def verdict(q, status, reason, evidence=None):
    print(f"\n[{q}] {status}\n    {reason}")
    if evidence:
        print(f"    evidence: {evidence}")
    return {"question": q, "status": status, "reason": reason, "evidence": evidence}


def main():
    log = read("log.txt")
    scorecard = read("scorecard.json")
    score = read("score.txt")
    control = read("control_score.txt")
    summaries = parse_summaries(log)
    logs_visible = bool(summaries) or (log is not None and SENTINEL in log)

    print("=" * 64)
    print("CHANNEL STATUS")
    print(f"  log.txt present: {log is not None}; sentinel found: "
          f"{log is not None and SENTINEL in log}; "
          f"per-game summaries parsed: {len(summaries)}")
    print(f"  scorecard.json present: {scorecard is not None}; "
          f"score.txt: {score is not None}; control_score: {control is not None}")
    print("=" * 64)

    V = []

    # ---- Channel-1 path: the probe's own trace answers everything ----
    if summaries:
        # Q1: did any game's level-0 accumulation stall / get locked near a
        # baseline multiple (cutoff), or did it sail to the probe cap?
        reached_cap = [s for s in summaries if s.get("max_level0_reached", 0) >= CUTOFF_REACH]
        stuck = [s for s in summaries if s.get("stuck_terminal")]
        # a cutoff: many games stop accumulating well below the cap without a
        # self-caused death pattern explaining it / a sticky terminal appears
        capped_early = [s for s in summaries
                        if s.get("max_level0_reached", 0) < CUTOFF_REACH
                        and s.get("actions", 0) < CUTOFF_REACH]
        if reached_cap and not stuck:
            V.append(verdict("Q1 cutoff enforced?", "REFUTED",
                "at least one game accumulated level-0 actions past 5x the max "
                "human baseline with no lockout — no per-level cutoff at the gateway.",
                {"reached_cap_games": [s["game"] for s in reached_cap][:5]}))
        elif stuck or (capped_early and not reached_cap):
            V.append(verdict("Q1 cutoff enforced?", "CONFIRMED",
                "level-0 accumulation halted below the probe cap (sticky terminal / "
                "actions refused) consistent with a per-level cutoff.",
                {"stuck": [s["game"] for s in stuck][:5],
                 "halt_indices": [s.get("max_level0_reached") for s in (stuck or capped_early)][:5]}))
        else:
            V.append(verdict("Q1 cutoff enforced?", "INCONCLUSIVE",
                "trace neither clearly reached the cap nor showed a lockout; "
                "inspect transitions/game_over_indices by hand.",
                {"per_game": [(s["game"], s.get("max_level0_reached"),
                               s.get("stuck_terminal")) for s in summaries][:8]}))

        # Q2: only answerable if a cutoff was seen AND we have multi-level reach
        multi = [s for s in summaries if (s.get("win_seen_at") is not None)]
        if stuck:
            V.append(verdict("Q2 what does it end / per-play?", "INCONCLUSIVE",
                "cutoff seen; the diagnostic does not pursue level-2 or a second "
                "play, so end-scope and per-play-vs-cumulative need the follow-on "
                "win-seeker probe (design doc, Probe 1 extension).",
                {"stuck_games": [s["game"] for s in stuck][:5]}))
        else:
            V.append(verdict("Q2 what does it end / per-play?", "INCONCLUSIVE",
                "no cutoff observed (or none reached), so there is nothing to "
                "characterize; revisit only if Q1 confirms."))

        # Q3: RESET-after-WIN mint, from any game the probe incidentally won
        mints = [s for s in summaries if s.get("reset_after_win_frame")]
        if mints:
            f = mints[0]["reset_after_win_frame"]
            minted = f.get("full_reset") or f.get("levels_completed") == 0
            V.append(verdict("Q3 RESET-after-WIN mints a play?",
                "CONFIRMED" if minted else "REFUTED",
                f"post-RESET frame after a win: {f} -> "
                f"{'fresh level-0 play (MINT)' if minted else 'same-level reset (NO mint)'}.",
                {"game": mints[0]["game"]}))
        else:
            V.append(verdict("Q3 RESET-after-WIN mints a play?", "INCONCLUSIVE",
                "the diagnostic won no game (it does not try to), so the mint test "
                "never fired; needs the win-seeker follow-on probe or rely on the "
                "local arc_agi baseline (mint confirmed there)."))

        # Q5: null-coordinate
        nulls = [s for s in summaries if s.get("null_coord_result")]
        if nulls:
            adv = any(n["null_coord_result"].get("advanced") for n in nulls)
            V.append(verdict("Q5 null-coordinate behavior",
                "CONFIRMED",
                f"ACTION6 (0,0) observed on {len(nulls)} game(s); advanced a level: "
                f"{adv}. See per-game null_coord_result for state/counting.",
                {"sample": nulls[0]["null_coord_result"]}))
        else:
            V.append(verdict("Q5 null-coordinate behavior", "INCONCLUSIVE",
                "no click game reached the null-coord probe (no ACTION6 available "
                "where tested)."))

    # ---- Score-encoding fallback path (no visible logs) ----
    else:
        V.append(verdict("Q1 cutoff enforced?",
            "CONFIRMED" if (score and control and _num(score) is not None
                            and _num(control) is not None
                            and _num(score) < 0.5 * _num(control))
            else "INCONCLUSIVE",
            "score-encoding path: compare probe aggregate vs frugal control. "
            "A probe score collapsing far below control (overshoot ended progress) "
            "=> cutoff. Provide score.txt AND control_score.txt for a verdict.",
            {"probe_score": _num(score), "control_score": _num(control)}))
        for q in ["Q2 what does it end / per-play?",
                  "Q3 RESET-after-WIN mints a play?",
                  "Q5 null-coordinate behavior"]:
            V.append(verdict(q, "INCONCLUSIVE",
                "logs not visible; score-encoding alone underdetermines this. "
                "See design doc: needs the known-baseline tripwire (Q1/Q2), the "
                "replay probe (Q3), or is observational only (Q5)."))

    # Q4 is external, not probe-derived
    V.append(verdict("Q4 wall-clock limit", "CONFIRMED",
        "6 hours — corroborated by two independent overview mirrors; confirm the "
        "session banner during the run and flag any difference.",
        {"source": "overview mirrors; FORGE 8h guard and our 9h were unsourced"}))

    (DIR / "verdicts.json").write_text(json.dumps(V, indent=2))
    print(f"\nwrote {DIR/'verdicts.json'}")
    print("CHANNEL:", "logs-visible (rich)" if logs_visible else
          "no logs (score-encoding fallback)")


def _num(s):
    if not s:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


if __name__ == "__main__":
    main()
