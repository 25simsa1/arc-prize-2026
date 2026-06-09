# Bake-off runbook — rented GPU, ephemeral box

Spin up, run 4 models, scp one tarball per model down, destroy the box.
Assume the box is hostile to state: nothing on it survives, nothing on it
matters once the tarballs are local. Pre-spend estimate (NOTES.md): ~3
GPU-hours, $5–8 expected, **$15 ceiling at the estimate rate — phase budget
allows up to $60 if marketplace prices or retries run hot**.

## 1. Instance

Preferred, in order (one box, one GPU):

| choice | why | notes |
|---|---|---|
| RTX PRO 6000 Blackwell 96GB | matches Kaggle eval hardware — throughput observations transfer | FP8 native |
| H100 80GB / H200 | plentiful on RunPod/Vast | Next must run INT4/AWQ (FP8 80B ≈ 80GB won't fit 80GB card) |
| A100 80GB | cheapest fallback | **no FP8 on Ampere**: GLM runs BF16 (~60GB), Next runs AWQ |

Disk: ≥250GB (weights total ~85–120GB). Image: any CUDA 12.4+ PyTorch base
(RunPod "pytorch" templates fine) or `vllm/vllm-openai` docker.

## 2. Bring-up (run on the box)

```bash
tmux new -s bench            # everything inside tmux; ssh drops are free
python3 -m venv ~/v && source ~/v/bin/activate
pip install -U pip
pip install vllm requests huggingface_hub
pip freeze > ~/env-pin.txt   # reproducibility by RECORDING — run_quality
                             # also captures pip freeze + nvidia-smi into
                             # every tarball, so exact versions are logged
```

No HF token needed — all four repos are public/non-gated. **Do not put git
credentials on the box.** Ship the bench dir up from local instead:

```bash
# local machine:
tar czf /tmp/bench.tgz bench/
scp /tmp/bench.tgz root@BOX:~/ 
# box:
tar xzf ~/bench.tgz && cd ~/bench
```

## 3. Models — exact run commands

One at a time (each serves, runs all 4 tasks at N=8, writes a tarball,
kills the server). `--vllm-extra` is where fit problems get solved.

```bash
# 1. 7B arm
python run_quality.py --model-id qwen25coder-7b \
  --repo Qwen/Qwen2.5-Coder-7B-Instruct-AWQ

# 2. 14B reference arm
python run_quality.py --model-id qwen25coder-14b \
  --repo Qwen/Qwen2.5-Coder-14B-Instruct-AWQ

# 3. GLM-4.7-Flash (30B-A3B). CHECK THE LICENSE FIELD on the repo page at
#    pull time (expected MIT/Apache lineage) and note it in NOTES.md.
#    Blackwell/H100: FP8 if an official FP8 repo exists, else BF16:
python run_quality.py --model-id glm47-flash \
  --repo zai-org/GLM-4.7-Flash --vllm-extra "--trust-remote-code"
#    A100 fallback: same command (BF16, ~60GB).

# 4. Qwen3-Coder-Next (80B-A3B). Decision tree, try in order:
#    a) official quant if it exists:
huggingface-cli repo info Qwen/Qwen3-Coder-Next-AWQ 2>/dev/null && \
python run_quality.py --model-id qwen3-coder-next \
  --repo Qwen/Qwen3-Coder-Next-AWQ
#    b) else best community AWQ/INT4 (check downloads/discussions first;
#       record exactly which repo was used — it goes in the tarball):
# python run_quality.py --model-id qwen3-coder-next --repo <community-awq-repo>
#    c) else SKIP and note it — do not burn hours fighting quant bugs;
#       Qwen2.5-Coder-32B-Instruct-AWQ is the named fallback model:
# python run_quality.py --model-id qwen25coder-32b \
#   --repo Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
```

If a model OOMs: add `--max-model-len 8192` first, then
`--vllm-extra "--gpu-memory-utilization 0.92"`.

## 4. Collect + destroy

```bash
# local machine:
scp 'root@BOX:~/bench/results/*.tar.gz' bench/results/
# verify all tarballs open locally BEFORE destroying the box:
for t in bench/results/*.tar.gz; do tar tzf "$t" | head -2; done
# then destroy the instance from the provider console.
```

Each tarball is self-contained: every generation verbatim, scores,
prompt-hash manifest, calibration, vllm log, env snapshot, timing.

## 5. Local analysis (after tarballs land)

```bash
.venv/bin/python bench/analyze.py bench/results/*.tar.gz
```

Produces per model: success/partial rates vs calibrated references
(A ref 1.0 / floor 0.5; B naive 0.295), repair-loop lift, reframing lift,
distribution with single-lucky-generation callouts, and the hypothesis-
comment corpus for the qualitative failure-pattern notes.

## Boundaries (standing)

- Dev-time measurement only; nothing here enters the evaluated path and
  bench/ is never imported by harness/.
- Real evidence prompts run ONLY on this rental and locally. Any
  Kaggle-side throughput notebook uses neutral stand-in prompts of matched
  length/structure — never this content.
