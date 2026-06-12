# Rental sitting 2 — quality pass + 2³ gap experiment (known-good recipe)

One Blackwell box, two runs, one tarball. Everything below is the recipe
that worked in rental 1, with the deployed fixes baked in. Est. $3–6,
ceiling $10.

## Instance + bring-up (the established quirks, do not relitigate)

- RTX PRO 6000 Blackwell 96GB preferred (H100 80GB fallback). Disk ≥120GB.
- `--gpu-memory-utilization 0.85` on every serve (VRAM-squat insurance).
- `export HF_HOME=/root/.cache/huggingface` — NEVER /workspace (10GB quota).
- **`pip uninstall -y flashinfer-python flashinfer`** right after install —
  it fails its arch check on sm_120 from three entry points (attention,
  JIT sampling, MoE autotune); vLLM falls back cleanly. Do NOT pip-upgrade
  it (resolver downgrades torch to cu12 and poisons the venv).
- Belt-and-braces env on every command:
  `VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/.cache/huggingface`
  (`VLLM_ATTENTION_BACKEND` is an unknown var to vLLM ≥0.22 — skip it.)

```bash
tmux new -s rental
python3 -m venv ~/v && source ~/v/bin/activate
pip install -U pip && pip install vllm requests numpy
pip uninstall -y flashinfer-python flashinfer
pip freeze > ~/env-pin.txt
```

## Ship the bundle up (no git creds on the box)

`rental_sitting/make_bundle.sh` builds `/tmp/rental2-bundle.tgz` locally
(harness/, the bench pieces, scripts/llm_quality.py, gap_conditions.json,
and the four staged stores). Then:

```bash
scp -P PORT /tmp/rental2-bundle.tgz root@BOX:~/ && ssh ... 'tar xzf rental2-bundle.tgz'
```

## Model serve (Phase-D pick, with the on-box fallback)

```bash
# primary: Qwen3-Coder-Next AWQ (the rental-1 quality leader)
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/.cache/huggingface \
python -m vllm.entrypoints.openai.api_server \
  --model bullpoint/Qwen3-Coder-Next-AWQ-4bit \
  --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.85 \
  > ~/vllm.log 2>&1 &
# wait for: curl -s localhost:8000/health
```

Fallback if Next misbehaves (>20 min of fighting): kill it and serve
`zai-org/GLM-4.7-Flash --trust-remote-code` instead; then EVERY client
command below adds:
`--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`
(thinking mode burns the whole token budget before code — 24/24 measured).

## Run 1 — quality pass (4 target games, R1′-clean sp80 store)

```bash
cd ~/bundle
.venv-less: python scripts/llm_quality.py \
  --stores stores --games sp80 su15 sb26 ar25 --rounds 4 \
  --llm-url http://localhost:8000 --llm-backend openai \
  --llm-model bullpoint/Qwen3-Coder-Next-AWQ-4bit \
  --out results/llm_quality_rental 2>&1 | tee quality.log
```

Target rationale: su15/sb26/ar25 = the genuinely frame-modelable set
(templates verify there — the LLM must beat them); sp80 = R1′-unblocked
with clean context structure (store staged from overnight-r1prime-on).

## Run 2 — the 2³ privileged-context gap experiment (~128 gens, marginal cost ≈ 0)

```bash
python bench/gap_experiment.py --run \
  --llm-url http://localhost:8000 --llm-backend openai \
  --llm-model bullpoint/Qwen3-Coder-Next-AWQ-4bit \
  --out results/gap_experiment 2>&1 | tee gap.log
```

(Conditions are prebuilt + hashed in bench/tasks/gap_conditions.json —
the box only generates and scores.)

## Collect

```bash
tar czf ~/rental2.tar.gz results quality.log gap.log ~/env-pin.txt ~/vllm.log
# local: scp it down, verify `tar tzf`, THEN destroy the box.
```

Local analysis (one command):
`.venv/bin/python scripts/rental_report.py results/rental2`
— re-verifies the corpus under the FULL fixed verifier (in-run verified
counts are never headline numbers), format-error rate vs the 0.6% local
baseline, gated-repair accepts, the gap table with keep/drop decisions,
and tokens+seconds per verified rule against the 6h/110-game envelope.
