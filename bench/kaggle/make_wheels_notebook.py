# Kaggle companion notebook 1/2 — WHEELS BUILDER (internet ON).
# Paste into a single-cell Kaggle notebook with Internet enabled, GPU off.
# Run, then "Save Version" and publish /kaggle/working/wheels as a Dataset
# (e.g. "vllm-offline-wheels"). The throughput notebook installs from it
# with --no-index, which is the PROOF that the serving stack can exist in
# the offline competition environment at all.
#
# Disclosure boundary: this file contains nothing project-specific.

import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/wheels")
OUT.mkdir(parents=True, exist_ok=True)

# Pin nothing here; record everything. The download resolves against THIS
# notebook's python/platform, which matches the offline runtime.
PKGS = ["vllm", "huggingface_hub"]

for pkg in PKGS:
    print(f"== pip download {pkg}")
    subprocess.run(
        [sys.executable, "-m", "pip", "download", pkg,
         "-d", str(OUT), "--prefer-binary"],
        check=True,
    )

names = sorted(p.name for p in OUT.glob("*"))
print(f"\n{len(names)} files, ~{sum(p.stat().st_size for p in OUT.glob('*'))/1e9:.1f} GB")
print("\n".join(names[:40]))
(Path("/kaggle/working/wheel_manifest.txt")).write_text("\n".join(names))
print("\nNow: Save Version -> publish /kaggle/working/wheels as a private Dataset.")
