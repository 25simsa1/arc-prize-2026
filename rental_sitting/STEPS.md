# Rental sitting 2 — your steps (paste-and-wait)

Estimate **$3–6, ceiling $10** (Blackwell ~$2/h; both runs ≈ 1–1.5h plus
bring-up; quality pass ≈ 40 min, gap experiment ≈ 20–30 min at vLLM
speeds). Full recipe with rationale: `bench/RENTAL2.md`.

1. **Spin up**: RunPod/Vast, RTX PRO 6000 Blackwell 96GB (H100 80GB ok),
   ≥120GB disk, any CUDA 12.4+ image.
2. **Local, one command**: `bash rental_sitting/make_bundle.sh`
   → `/tmp/rental2-bundle.tgz` (harness + bench pieces + 4 staged stores;
   sp80 is the R1′-on store, confirmed).
3. **On the box** (paste blocks from `bench/RENTAL2.md` in order):
   bring-up (incl. the flashinfer uninstall) → scp bundle up → serve
   Next-AWQ → **Run 1 quality pass** → **Run 2 gap experiment** → tar
   `rental2.tar.gz`.
   - If Next fights for >20 min: switch to the GLM fallback block (adds
     the nothink `--extra-body` to both runs). Note which model ran.
4. **Collect**: scp `rental2.tar.gz` down, `tar tzf` it locally to verify,
   THEN destroy the box.
5. **Drop it** at `results/rental2/` (`mkdir -p results/rental2 && tar xzf
   rental2.tar.gz -C results/rental2`) and say the word — analysis is one
   command on my side and the report covers: full-verifier verified rules
   per game, format errors vs 0.6%, repair accepts, the 2³ gap table with
   keep/drop calls, and seconds/tokens per verified rule vs the 6h
   envelope.
