# Kaggle companion notebook 2/2 — THROUGHPUT ON TARGET HARDWARE (offline).
# Paste into a single-cell notebook attached to the ARC-AGI-3 competition,
# accelerator set to the competition GPU, INTERNET DISABLED (that's the
# point), with these inputs attached:
#   - the "vllm-offline-wheels" dataset from companion notebook 1
#   - one Kaggle Model / Dataset per candidate (weights dirs)
#
# DISCLOSURE BOUNDARY: prompts below are NEUTRAL stand-ins — synthetic
# sensor-log classification with random hex payloads, matched only in
# LENGTH and SHAPE (instructions + JSON observation lines + code-block
# request) to generic rule-induction prompts. No competition evidence, no
# task framing, no game content appears in this file.
#
# Measures per model, each in a FRESH SUBPROCESS (guaranteed-clean VRAM,
# honest cold-load): cold-load seconds, prefill seconds per prompt size,
# decode tokens/sec, peak GPU memory. Plus environment facts (GPU, VRAM,
# RAM, disk, internet reachability) and the offline-install proof.

import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
RESULTS = {"env": {}, "install": {}, "models": {}}

# ---------------------------------------------------------------- env facts
def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"

RESULTS["env"]["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
RESULTS["env"]["ram"] = sh("grep MemTotal /proc/meminfo")
RESULTS["env"]["disk"] = sh("df -h /kaggle/working | tail -1")
RESULTS["env"]["cpus"] = os.cpu_count()
try:
    socket.create_connection(("8.8.8.8", 53), timeout=3).close()
    RESULTS["env"]["internet"] = True  # should be False in the real config!
except OSError:
    RESULTS["env"]["internet"] = False
print(json.dumps(RESULTS["env"], indent=1))

# ----------------------------------------------------- offline install proof
wheel_dirs = [str(p) for p in Path("/kaggle/input").glob("*/wheels")] + \
             [str(p) for p in Path("/kaggle/input").glob("*") if list(p.glob("vllm-*.whl"))]
assert wheel_dirs, "attach the vllm-offline-wheels dataset"
t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index",
     "--find-links", wheel_dirs[0], "vllm"],
    capture_output=True, text=True)
RESULTS["install"] = {
    "ok": proc.returncode == 0,
    "seconds": round(time.time() - t0, 1),
    "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
}
print(json.dumps(RESULTS["install"], indent=1))
if not RESULTS["install"]["ok"]:
    # Hard finding: the serving stack cannot exist offline => stack-level
    # disqualification evidence. Still write results and stop.
    (WORK / "throughput_results.json").write_text(json.dumps(RESULTS, indent=1))
    raise SystemExit("vllm cannot install offline — stack disqualified, see results json")

# ------------------------------------------------- neutral matched prompts
def neutral_prompt(target_chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    head = (
        "You are analyzing logs from an industrial sensor array. Each "
        "observation lists `payload` (64 hex digits, one per channel) and "
        "`status`, which is either \"ALERT\" or \"NONE\".\n\nOBSERVATIONS:\n"
    )
    tail = (
        "\n\nWrite a single Python function that predicts the status:\n\n"
        "```python\ndef predict_status(payload: str, source: str) -> str:\n"
        "    # return \"ALERT\" or \"NONE\"\n```\n\n"
        "Hypothesize the underlying condition. Reply with ONLY the function "
        "in a ```python code block, plus one comment line stating your rule."
    )
    body = ""
    while len(head) + len(body) + len(tail) < target_chars:
        payload = "".join(rng.choice("0123456789abcdef") for _ in range(64))
        body += json.dumps({"payload": payload,
                            "source": rng.choice(["S1", "S2", "S3"]),
                            "status": rng.choice(["ALERT", "NONE"])}) + "\n"
    return head + body + tail

# length/shape matched to the real task distribution (sizes only, no content)
PROMPTS = {"small_2700": neutral_prompt(2700), "mid_3700": neutral_prompt(3700),
           "large_5400": neutral_prompt(5400)}
DECODE_TOKENS = 600  # typical real generation budget

# ------------------------------------------------------- per-model runner
RUNNER = r'''
import json, sys, time
import torch
from vllm import LLM, SamplingParams

model_path, prompts_path = sys.argv[1], sys.argv[2]
prompts = json.loads(open(prompts_path).read())
out = {}
t0 = time.time()
llm = LLM(model=model_path, max_model_len=16384, gpu_memory_utilization=0.92,
          enforce_eager=False, trust_remote_code=True)
out["cold_load_s"] = round(time.time() - t0, 1)
tok = llm.get_tokenizer()
llm.generate(["warmup"], SamplingParams(max_tokens=8))  # jit/warmup

for name, prompt in prompts.items():
    n_in = len(tok.encode(prompt))
    t1 = time.time()
    llm.generate([prompt], SamplingParams(max_tokens=1))
    prefill_s = time.time() - t1
    t2 = time.time()
    r = llm.generate([prompt], SamplingParams(max_tokens=600, temperature=0.8,
                                              seed=7))
    full_s = time.time() - t2
    n_out = len(r[0].outputs[0].token_ids)
    decode_tps = (n_out - 1) / max(full_s - prefill_s, 1e-6)
    out[name] = {"prompt_tokens": n_in,
                 "prefill_s": round(prefill_s, 2),
                 "prefill_tps": round(n_in / max(prefill_s, 1e-6), 1),
                 "decode_tokens": n_out,
                 "decode_tps": round(decode_tps, 1)}
out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("RESULT_JSON:" + json.dumps(out))
'''
(WORK / "runner.py").write_text(RUNNER)
(WORK / "prompts.json").write_text(json.dumps(PROMPTS))

# Candidate weights: auto-discover attached model dirs (any input dir that
# holds a config.json). Flip OFF non-finalists here once Phase B lands.
CANDIDATES = {}
for p in Path("/kaggle/input").rglob("config.json"):
    d = p.parent
    if any(d.glob("*.safetensors")):
        CANDIDATES[d.name] = str(d)
print("discovered candidates:", json.dumps(CANDIDATES, indent=1))

for name, path in CANDIDATES.items():
    print(f"\n==== {name}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(WORK / "runner.py"), path,
                           str(WORK / "prompts.json")],
                          capture_output=True, text=True, timeout=3600)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("RESULT_JSON:")), None)
    if line:
        RESULTS["models"][name] = json.loads(line[len("RESULT_JSON:"):])
        RESULTS["models"][name]["wall_total_s"] = round(time.time() - t0, 1)
    else:
        # DISQUALIFICATION evidence: could not serve in this environment.
        RESULTS["models"][name] = {
            "DISQUALIFIED": True,
            "exit": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-5:],
        }
    print(json.dumps(RESULTS["models"][name], indent=1))

(WORK / "throughput_results.json").write_text(json.dumps(RESULTS, indent=1))
print("\nWROTE /kaggle/working/throughput_results.json — download it.")
