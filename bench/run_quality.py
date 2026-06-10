"""Single entrypoint for the rented box: serve one model with vLLM, run all
four tasks, write a self-contained results tarball, kill the server.

Designed for an ephemeral, state-hostile box: everything the analysis needs
(generations verbatim, scores, prompt hashes, calibration, timing, env
snapshot) lands in ONE tarball; nothing on the box matters afterwards.

    python bench/run_quality.py --model-id qwen2.5-coder-14b \
        --repo Qwen/Qwen2.5-Coder-14B-Instruct-AWQ --samples 8

Dev-time measurement only; never imported by harness/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_bench import TEMPERATURE, gen_openai, run_repair, run_single  # noqa: E402


def wait_health(url: str, proc: subprocess.Popen, timeout_s: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited early with code {proc.returncode}")
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("vllm never became healthy")


def env_snapshot(out: Path) -> None:
    snap = {}
    for name, cmd in [
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"]),
        ("pip_freeze", [sys.executable, "-m", "pip", "freeze"]),
    ]:
        try:
            snap[name] = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            snap[name] = f"unavailable: {e}"
    (out / "env.json").write_text(json.dumps(snap, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="short name for files/results")
    ap.add_argument("--repo", required=True, help="HF repo vLLM serves (deploy quant)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--vllm-extra", default="", help="extra vllm serve args")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into chat request bodies "
                         "(e.g. '{\"chat_template_kwargs\":{\"enable_thinking\":false}}')")
    ap.add_argument("--tasks", nargs="+", default=["A", "B", "reframe", "repair"])
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    out_dir = HERE / "results" / args.model_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    vllm_cmd = (
        f"{sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {args.repo} --port {args.port} "
        f"--max-model-len {args.max_model_len} {args.vllm_extra}"
    )
    print(f"starting: {vllm_cmd}")
    # Blackwell (sm_120) deployment quirks, learned on an RTX PRO 6000 rental:
    # the shipped flashinfer wheel fails its arch check on sm_120 in BOTH the
    # attention path and (separately) the JIT sampling path — keep it out of
    # both. HF cache must never land on a quota'd network volume. These are
    # overridable defaults (shell exports win), so non-interactive shells
    # that skip bashrc can't lose them.
    spawn_env = {
        **os.environ,
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0"),
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    }
    server = subprocess.Popen(vllm_cmd.split(),
                              stdout=open(out_dir / "vllm.log", "w"),
                              stderr=subprocess.STDOUT,
                              env=spawn_env)
    try:
        t_load0 = time.time()
        wait_health(url, server)
        load_s = round(time.time() - t_load0, 1)
        print(f"vllm healthy after {load_s}s")

        extra_body = json.loads(args.extra_body) if args.extra_body else None

        def gen(prompt: str, seed: int) -> str:
            return gen_openai(args.repo, prompt, seed, url, extra_body)

        results = {
            "model_id": args.model_id, "repo": args.repo,
            "temperature": TEMPERATURE, "samples": args.samples,
            "max_model_len": args.max_model_len, "load_seconds": load_s,
            "extra_body": extra_body,
            "manifest": json.loads((HERE / "tasks" / "manifest.json").read_text()),
        }
        t_run0 = time.time()
        for task in args.tasks:
            if task == "repair":
                results["repair"] = run_repair(args.model_id, gen, args.samples, out_dir)
            else:
                kind = "event" if task == "A" else "click"
                results[task] = run_single(task, kind, args.model_id, gen,
                                           args.samples, out_dir)
        results["total_run_seconds"] = round(time.time() - t_run0, 1)
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    # self-contained tarball: results + calibration + task specs + env
    env_snapshot(out_dir)
    for extra in ["manifest.json", "calibration.json"]:
        src = HERE / "tasks" / extra
        if src.exists():
            shutil.copy(src, out_dir / extra)
    tar_path = HERE / "results" / f"{args.model_id}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=args.model_id)
    print(f"\nTARBALL: {tar_path}")


if __name__ == "__main__":
    main()
