#!/usr/bin/env bash
# Build the rental-2 upload bundle. Run from the repo root:
#   bash rental_sitting/make_bundle.sh
set -e
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)
mkdir -p "$STAGE/bundle/stores" "$STAGE/bundle/bench/tasks" "$STAGE/bundle/scripts"

cp -R harness "$STAGE/bundle/"
cp bench/run_bench.py bench/scoring.py bench/evidence.py bench/gap_experiment.py \
   "$STAGE/bundle/bench/"
cp bench/tasks/gap_conditions.json bench/tasks/manifest.json \
   bench/tasks/calibration.json "$STAGE/bundle/bench/tasks/" 2>/dev/null || true
cp scripts/llm_quality.py "$STAGE/bundle/scripts/"

# staged stores: sp80 from the R1'-on A/B arm; the rest from the recapture
cp runs/wm/overnight-r1prime-on/sp80-store.pkl "$STAGE/bundle/stores/"
for g in su15 sb26 ar25; do
  cp "runs/wm/rental-stores/$g-store.pkl" "$STAGE/bundle/stores/"
done

tar czf /tmp/rental2-bundle.tgz -C "$STAGE" bundle
rm -rf "$STAGE"
echo "built /tmp/rental2-bundle.tgz ($(du -h /tmp/rental2-bundle.tgz | cut -f1))"
tar tzf /tmp/rental2-bundle.tgz | head -8
