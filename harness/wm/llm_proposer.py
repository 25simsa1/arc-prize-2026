"""LLM proposer: frame-only rules from a LOCAL open-weight model.

Implements the same propose(store, model) -> list[Rule] interface as the
template proposer. The backend is an OpenAI-compatible or ollama HTTP
endpoint on localhost serving an open-weight model — never a hosted API
(the import-graph rule: nothing API-based in the evaluated path; the
submission notebook serves its own model the same way).

Bake-off findings built in as FIRST-CLASS requirements:
  TRUNCATION FIX  (Next: 7/8 hard-task gens wrote analysis prose and died
  before code; format-error baseline 17/24): code-first scaffold — the
  model must answer with ONE python function, analysis as comments INSIDE
  it; generous max_tokens; a generation without a parseable function gets
  exactly one mechanical retry with a harder constraint. Format errors are
  counted and reported.

  GATED REPAIR  (the 14B was repair-NEGATIVE: blind round-2 adoption
  degraded 4/8 rules): a revision is accepted ONLY if it strictly improves
  on the feedback evidence (fewer misses, no fewer exact fits); otherwise
  the original is kept. Accept/reject counts are logged.

Sandboxing (Workstream D): generated code compiles into a namespace with a
whitelisted builtin set (+numpy); source containing imports/dunders/IO is
rejected outright; every predict call runs under a SIGALRM kill switch and
a rule is permanently disabled on its first timeout or exception. Every
generation is logged verbatim.
"""

import json
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

from .rules import Prediction, Rule, WorldModel
from .store import Transition, TransitionStore

SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "len", "range", "enumerate", "min", "max", "abs", "sum", "sorted",
        "set", "dict", "list", "tuple", "int", "float", "bool", "str",
        "any", "all", "zip", "map", "filter", "reversed", "round", "isinstance",
    )
}
FORBIDDEN_SRC = re.compile(
    r"\bimport\b|__|\bopen\(|\bexec\b|\beval\b|\bglobals\b|\blocals\b|\bgetattr\b"
)


class _PredictTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _PredictTimeout()


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


def serialize_transition(t: Transition, dyn_mask) -> str:
    diff = t.pre != t.post
    if dyn_mask is not None:
        diff = diff & dyn_mask
    cells = np.argwhere(diff)
    changed = [[int(y), int(x), int(t.pre[y, x]), int(t.post[y, x])]
               for y, x in cells[:80]]
    return json.dumps({
        "level": t.level, "action": t.action_key, "event": t.event,
        "n_changed": int(len(cells)),
        "changed_y_x_pre_post": changed,
        "changed_spans": spans([(y, x) for y, x in cells]) if len(cells) else "",
    })


@dataclass
class LLMStats:
    calls: int = 0
    generations: int = 0
    format_errors: int = 0
    format_retries: int = 0
    repair_accepted: int = 0
    repair_rejected: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    rules_emitted: int = 0
    kills: int = 0  # rules disabled by the per-predict kill switch

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LLMProposer:
    name = "llm"

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b",
        backend: str = "ollama",          # "ollama" | "openai"
        log_dir: Optional[str] = None,
        samples_per_call: int = 3,
        max_tokens: int = 3000,           # generous: the truncation fix
        temperature: float = 0.8,
        predict_timeout_s: float = 0.05,
        min_interval_s: float = 45.0,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.url, self.model, self.backend = url, model, backend
        self.log_dir = Path(log_dir) if log_dir else None
        self.samples = samples_per_call
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.predict_timeout_s = predict_timeout_s
        self.min_interval_s = min_interval_s
        self.extra_body = extra_body or {}
        self.stats = LLMStats()
        self._last_call = 0.0
        self._gen_counter = 0
        self._rule_counter = 0

    # ------------------------------------------------------------- backend
    def _generate(self, prompt: str, seed: int) -> tuple[str, int, int]:
        t0 = time.monotonic()
        if self.backend == "ollama":
            r = requests.post(f"{self.url}/api/generate", json={
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": self.temperature, "seed": seed,
                            "num_ctx": 16384, "num_predict": self.max_tokens},
            }, timeout=600)
            r.raise_for_status()
            d = r.json()
            text = d["response"]
            pt, ot = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        else:
            body = {"model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature, "seed": seed,
                    "max_tokens": self.max_tokens, **self.extra_body}
            r = requests.post(f"{self.url}/v1/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            d = r.json()
            text = d["choices"][0]["message"]["content"]
            u = d.get("usage", {})
            pt, ot = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        self.stats.seconds += time.monotonic() - t0
        self.stats.generations += 1
        self.stats.prompt_tokens += pt
        self.stats.output_tokens += ot
        return text, pt, ot

    def _log(self, kind: str, game_id: str, text: str) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._gen_counter += 1
        (self.log_dir / f"{game_id}-{kind}-{self._gen_counter:03d}.md").write_text(text)

    # ------------------------------------------------------------- prompts
    PROMPT = """You are reverse-engineering the rules of an unknown 64x64 grid game from observed transitions. Cell values are colors 0-15. `action` is one of ACTION1..ACTION7 or "ACTION6:x,y" (a click at column x, row y). `event` is what the action caused: "NONE", "LEVEL" (level completed), "WIN", or "GAME_OVER".

OBSERVED TRANSITIONS (each line: one action; changed cells as [row, col, color_before, color_after]):
{evidence}

Write ONE Python function hypothesizing a rule of the game's dynamics:

```python
def predict(level, grid, action):
    # grid: 64x64 list of lists of ints (colors). action: string as above.
    # Return None if this rule does not apply to (level, grid, action).
    # Otherwise return a dict with one or both keys:
    #   "grid":  the full 64x64 next grid (list of lists)
    #   "event": "NONE" | "LEVEL" | "WIN" | "GAME_OVER"
    ...
```

CRITICAL FORMAT RULES: respond with ONLY the ```python code block. Put ALL analysis as # comments INSIDE the function body — no prose outside the code block. Cover the regularity you are most confident about; returning None for everything else is correct behavior, not failure."""

    RETRY_SUFFIX = """

YOUR PREVIOUS RESPONSE CONTAINED NO PARSEABLE FUNCTION. Respond with ONLY a code block in exactly this shape, nothing before or after it:
```python
def predict(level, grid, action):
    ...
```"""

    REPAIR_PROMPT = """Your hypothesized rule for this grid game was wrong on the following observed transitions (same format; these are the TRUE outcomes):
{misses}

Your function was:
```python
{code}
```

Revise the hypothesis to account for these counterexamples WITHOUT breaking the cases it already predicted correctly. Respond with ONLY the corrected function in a ```python code block, analysis as comments inside it."""

    # ------------------------------------------------------------ evidence
    def _build_evidence(self, store: TransitionStore, model: WorldModel,
                        cap_chars: int = 5000) -> str:
        dyn = ~model.hud_mask if model.hud_mask is not None else None
        ts = list(store.all())
        # priority: event transitions, then smallest diffs, action-diverse
        events = [t for t in ts if t.event != "NONE"]
        nones = sorted((t for t in ts if t.event == "NONE"),
                       key=lambda t: t.diff_cells if t.diff_cells >= 0 else 9999)
        picked, seen_actions = [], set()
        for pool in (events[:10], nones):
            for t in pool:
                key = (t.base_action, t.event)
                bonus = key not in seen_actions
                if bonus or len(picked) < 18:
                    picked.append(t)
                    seen_actions.add(key)
                if len(picked) >= 26:
                    break
        lines, total = [], 0
        for t in picked:
            line = serialize_transition(t, dyn)
            if total + len(line) > cap_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    # ------------------------------------------------------------ sandbox
    def _compile(self, code: str) -> Optional[Any]:
        if FORBIDDEN_SRC.search(code):
            return None
        ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "np": np}
        try:
            exec(compile(code, "<llm_rule>", "exec"), ns)  # noqa: S102
        except Exception:
            return None
        fn = ns.get("predict")
        return fn if callable(fn) else None

    def _wrap(self, fn: Any, code: str, game_id: str,
              dyn_mask) -> Rule:
        self._rule_counter += 1
        rule_id = f"llm[{game_id}#{self._rule_counter:03d}]"
        state = {"dead": False}
        stats = self.stats
        timeout = self.predict_timeout_s

        def predict(level: int, pre: np.ndarray, action_key: str) -> Optional[Prediction]:
            if state["dead"]:
                return None
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                out = fn(level, [list(map(int, row)) for row in pre], action_key)
                signal.setitimer(signal.ITIMER_REAL, 0)  # disarm ASAP, inside try
            except Exception:  # incl. _PredictTimeout: first strike kills
                state["dead"] = True
                stats.kills += 1
                return None
            finally:
                # belt-and-suspenders: a late alarm can fire during teardown
                # (race over many back-to-back predicts); swallow it here so it
                # never escapes into the caller's loop.
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                except _PredictTimeout:
                    pass
                signal.signal(signal.SIGALRM, old)
            if out is None:
                return None
            if not isinstance(out, dict):
                state["dead"] = True
                stats.kills += 1
                return None
            grid = out.get("grid")
            event = out.get("event")
            g = None
            if grid is not None:
                try:
                    g = np.asarray(grid, dtype=np.int16)
                    if g.shape != pre.shape:
                        return None
                except Exception:
                    return None
            if event is not None and event not in ("NONE", "LEVEL", "WIN", "GAME_OVER"):
                return None
            if g is None and event is None:
                return None
            return Prediction(grid=g, event=event, mask=dyn_mask)

        rule = Rule(rule_id, "llm", {"code_chars": len(code)}, predict,
                    "llm", specificity=70,
                    region="dynamic" if dyn_mask is not None else "full")
        rule.source_code = code  # type: ignore[attr-defined]
        return rule

    @staticmethod
    def _extract(text: str) -> Optional[str]:
        m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if m and "def predict" in m.group(1):
            return m.group(1)
        return None

    # ----------------------------------------------------------- main API
    def due(self, store: TransitionStore) -> bool:
        return (len(store) >= 30
                and time.monotonic() - self._last_call >= self.min_interval_s)

    def propose(self, store: TransitionStore, model: Optional[WorldModel] = None,
                game_id: str = "game") -> list[Rule]:
        model = model or WorldModel()
        self._last_call = time.monotonic()
        self.stats.calls += 1
        dyn = ~model.hud_mask if model.hud_mask is not None else None
        evidence = self._build_evidence(store, model)
        prompt = self.PROMPT.format(evidence=evidence)
        rules: list[Rule] = []
        for i in range(self.samples):
            seed = 1000 * self.stats.calls + i
            text, _, _ = self._generate(prompt, seed)
            self._log("gen", game_id, text)
            code = self._extract(text)
            if code is None:
                # TRUNCATION FIX: one mechanical retry, harder constraint
                self.stats.format_retries += 1
                text, _, _ = self._generate(prompt + self.RETRY_SUFFIX, seed + 500)
                self._log("retry", game_id, text)
                code = self._extract(text)
            if code is None:
                self.stats.format_errors += 1
                continue
            fn = self._compile(code)
            if fn is None:
                self.stats.format_errors += 1
                continue
            rules.append(self._wrap(fn, code, game_id, dyn))
        self.stats.rules_emitted += len(rules)
        return rules

    def repair(self, rule: Rule, misses: list[Transition], store: TransitionStore,
               model: WorldModel, game_id: str = "game") -> Optional[Rule]:
        """GATED: returns a replacement rule only if the revision strictly
        improves on the feedback evidence; otherwise None (keep original)."""
        code = getattr(rule, "source_code", None)
        if code is None or not misses:
            return None
        dyn = ~model.hud_mask if model.hud_mask is not None else None
        miss_lines = "\n".join(serialize_transition(t, dyn) for t in misses[:6])
        text, _, _ = self._generate(
            self.REPAIR_PROMPT.format(misses=miss_lines, code=code), seed=77)
        self._log("repair", game_id, text)
        new_code = self._extract(text)
        fn = self._compile(new_code) if new_code else None
        if fn is None:
            self.stats.repair_rejected += 1
            return None
        candidate = self._wrap(fn, new_code, game_id, dyn)

        from .rules import grids_match

        # Sample-bound the feedback evaluation: live stores reach 10k+
        # transitions, and a full per-predict pass under the kill switch is
        # both slow and alarm-racy. Event transitions are always included
        # (they carry the signal repair must not break).
        all_ts = list(store.all())
        if len(all_ts) > 800:
            ev = [t for t in all_ts if t.event != "NONE"]
            rest = [t for t in all_ts if t.event == "NONE"][: max(0, 800 - len(ev))]
            eval_ts = ev + rest
        else:
            eval_ts = all_ts

        def count(r: Rule) -> tuple[int, int]:
            exact = miss = 0
            for t in eval_ts:
                try:
                    p = r.predict(t.level, t.pre, t.action_key)
                except Exception:
                    continue
                if p is None:
                    continue
                ok = grids_match(p, t.post) and (p.event is None or p.event == t.event)
                exact, miss = exact + ok, miss + (not ok)
            return exact, miss

        old_exact, old_miss = count(rule)
        new_exact, new_miss = count(candidate)
        if new_miss < old_miss and new_exact >= old_exact:
            self.stats.repair_accepted += 1
            return candidate
        self.stats.repair_rejected += 1
        return None
