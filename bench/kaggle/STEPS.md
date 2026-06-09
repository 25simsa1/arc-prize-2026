# Phase C — Kaggle throughput run: exact steps (user actions)

Everything here is free. The point: measure the slate on the ACTUAL
competition runtime, and prove (or disprove) that vLLM can exist in the
offline notebook at all. **Deployability is a hard constraint — a finalist
that cannot serve here is DISQUALIFIED regardless of Phase B quality.**

Disclosure boundary: both notebook files contain only generic loading/
timing code and neutral synthetic prompts (sensor-log classification,
matched in length/shape only). Keep both notebooks PRIVATE anyway.

## Step 1 — wheels dataset (one-time, ~10 min)

1. kaggle.com → Create Notebook (no GPU needed), Internet **ON**.
2. Paste `bench/kaggle/make_wheels_notebook.py` into a cell, run.
3. Save Version → from the version's Output, "Create Dataset" from
   `/kaggle/working/wheels`, name it `vllm-offline-wheels` (private).

## Step 2 — weights as Kaggle inputs (one-time per model)

Preferred: attach existing **Kaggle Models** entries (search the Models hub
for "Qwen2.5-Coder-7B-Instruct-AWQ", "Qwen2.5-Coder-14B-Instruct-AWQ" —
the qwen-lm org publishes most of these). For anything missing
(GLM-4.7-Flash, Qwen3-Coder-Next quant): a throwaway internet-ON notebook:

```python
from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-4.7-Flash", local_dir="/kaggle/working/glm47flash")
```

Save Version → publish the output dir as a private Dataset. (Mind dataset
size limits; the Next AWQ at ~44GB is large but within bounds. Use the SAME
quant repos Phase B used, recorded in its tarballs.)

## Step 3 — the throughput notebook

1. Create Notebook **inside the ARC-AGI-3 competition** (so the upgraded
   accelerator pool applies). Settings: competition GPU, Internet **OFF**.
2. Attach inputs: `vllm-offline-wheels` + the weights for the finalists —
   Phase B top 2 plus the 7B if (and only if) its repair-lift was real.
   (The notebook auto-discovers any attached dir containing config.json +
   safetensors; attach only what should run.)
3. Paste `bench/kaggle/throughput_kaggle.py`, run all.
4. Expected wall: ~5–15 min per model after load; cold-load dominates for
   the big MoEs. The 9h limit is nowhere close.
5. Download `/kaggle/working/throughput_results.json` and drop it at
   `bench/results/throughput_results.json` locally; tell Claude.

## What the numbers decide

- `install.ok == false` → vLLM cannot exist offline → stack-level
  disqualification; we pivot serving stacks (llama.cpp wheels) and re-prove.
- A model row with `DISQUALIFIED: true` (OOM, quant unsupported, crash) →
  that finalist is OUT, regardless of quality scores. Logged as such.
- `cold_load_s` matters at fleet scale: 110 games / 9h leaves ~3.5h of
  model time at 180s/game budgets — a 10-min cold load is 5% of the whole
  evaluation, fine; a 40-min load is not.
- `decode_tps` × real generation budgets feeds the proposer-calls-per-game
  arithmetic for Workstream D.
