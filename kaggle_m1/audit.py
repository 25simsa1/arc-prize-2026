"""Disclosure audit for the M1 public bundle. Run before ANYTHING goes
public; exits nonzero on any hit. The forbidden list covers strategy and
analysis that must not ship: multi-play/replay mechanics, play-semantics
findings, the research log, the LLM-proposer track, and internal framings.

    .venv/bin/python kaggle_m1/audit.py
"""

import re
import sys
from pathlib import Path

FORBIDDEN = [
    r"two_phase", r"on_play_start", r"new_play", r"play.?mint", r"replay",
    r"max.?over.?plays", r"best.?play", r"win.?gated", r"play.?semantics",
    r"full_reset", r"conflict", r"determinism", r"non.?markov",
    r"NOTES\.md", r"smoke", r"bake.?off", r"\bLLM\b", r"proposer model",
    r"qwen", r"glm", r"ollama", r"vllm", r"bench/", r"evidence.?starv",
    r"RHAE",  # we describe scoring in plain words publicly, not internal shorthand
    r"scratch", r"workstream", r"phase [a-d]\b",
]

PUBLIC_FILES = ["m1_notebook.py", "WRITEUP.md"]


def main() -> None:
    here = Path(__file__).parent
    hits = []
    for name in PUBLIC_FILES:
        text = (here / name).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            for pat in FORBIDDEN:
                if re.search(pat, line, re.IGNORECASE):
                    hits.append(f"{name}:{i}: [{pat}] {line.strip()[:100]}")
    if hits:
        print("AUDIT FAILED:")
        print("\n".join(hits))
        sys.exit(1)
    print(f"AUDIT CLEAN: {PUBLIC_FILES} contain none of {len(FORBIDDEN)} forbidden patterns")


if __name__ == "__main__":
    main()
