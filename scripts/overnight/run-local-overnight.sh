#!/usr/bin/env bash
# Sequential overnight queue runner (no `set -e` by design: a failed task must
# not kill the loop). Usage:
#   nohup bash scripts/overnight/run-local-overnight.sh scripts/overnight/overnight-queue-20260610.yaml > /tmp/overnight.log 2>&1 &
#
# Per task: run claude CLI headless with a hard per-task time cap (perl alarm;
# macOS has no `timeout`), fallback-commit any uncommitted work the task left,
# mark the id in logs/completed.txt so reruns skip finished tasks.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${1:?usage: run-local-overnight.sh <queue.yaml>}"
RUN_TAG="${RUN_TAG:-overnight-20260610}"
OUTDIR="$ROOT/results/$RUN_TAG"
LOGDIR="$OUTDIR/logs"
COMPLETED="$LOGDIR/completed.txt"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

mkdir -p "$LOGDIR"
touch "$COMPLETED"
cd "$ROOT" || exit 1
[ -f "$OUTDIR/start-sha.txt" ] || git rev-parse HEAD > "$OUTDIR/start-sha.txt"

RUNLOG="$LOGDIR/runner.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUNLOG"; }

say "queue start: $QUEUE (run tag $RUN_TAG, start sha $(cat "$OUTDIR/start-sha.txt"))"

# Extract id / max_seconds / prompt_file triples (fields appear in this order
# per task; see the queue file header for why this isn't a real YAML parse).
TASKS_TSV="$LOGDIR/tasks.tsv"
paste \
  <(grep -E '^[[:space:]]*- id:' "$QUEUE" | awk '{print $3}') \
  <(grep -E '^[[:space:]]*max_seconds:' "$QUEUE" | awk '{print $2}') \
  <(grep -E '^[[:space:]]*prompt_file:' "$QUEUE" | awk '{print $2}') \
  > "$TASKS_TSV"

N_TASKS=$(wc -l < "$TASKS_TSV" | tr -d ' ')
say "parsed $N_TASKS tasks"

while IFS=$'\t' read -r id maxs pfile; do
  [ -z "$id" ] && continue
  if grep -qx "$id" "$COMPLETED"; then
    say "SKIP $id (already completed)"
    continue
  fi
  if [ ! -f "$ROOT/$pfile" ]; then
    say "ERROR $id: prompt file missing: $pfile — skipping"
    echo "$id" >> "$COMPLETED"
    continue
  fi

  say "START $id (cap ${maxs}s, prompt $pfile)"
  # Session/usage-limit errors are not task failures: wait out the limit and
  # retry the same task instead of marking it complete (up to ~8h of waiting).
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    perl -e 'alarm shift @ARGV; exec @ARGV or die "exec failed: $!"' "$maxs" \
      "$CLAUDE_BIN" --dangerously-skip-permissions -p "$(cat "$ROOT/$pfile")" \
      < /dev/null > "$LOGDIR/$id.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && [ "$(wc -c < "$LOGDIR/$id.log")" -lt 500 ] \
       && grep -Eqi "session limit|usage limit|rate limit" "$LOGDIR/$id.log"; then
      if [ $attempt -ge 24 ]; then
        say "GIVE UP $id: still limited after $attempt attempts"
        break
      fi
      say "LIMITED $id (attempt $attempt): $(head -c 120 "$LOGDIR/$id.log") — sleeping 20m"
      sleep 1200
      continue
    fi
    break
  done
  [ $rc -eq 142 ] && say "NOTE $id hit the ${maxs}s time cap (SIGALRM)"

  # Fallback commit: only paths the guardrails allow, and only this run's
  # results dir (preexisting untracked results/ dirs stay untracked).
  git add harness scripts NOTES.md "results/$RUN_TAG" 2>>"$RUNLOG"
  if ! git diff --cached --quiet; then
    git commit -m "overnight[$id]: runner fallback commit of uncommitted work (task exit $rc)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >>"$RUNLOG" 2>&1
    say "NOTE $id left uncommitted work; fallback-committed"
  fi

  echo "$id" >> "$COMPLETED"
  say "END $id rc=$rc (head $(git rev-parse --short HEAD))"
done < "$TASKS_TSV"

say "queue done: $(grep -c . "$COMPLETED")/$N_TASKS tasks marked complete"
