#!/usr/bin/env bash
set -euo pipefail

BASE_RESULTS_DIR="results"

RUN_DIRS=(
  "webvoyager_mini_30_4_worker_reflection_run4"
  "webvoyager_mini_30_4_worker_reflection_run5"
  "webvoyager_mini_30_4_worker_reflection_run6"
)

# ----- compute common part (longest common prefix) of RUN_DIRS -----
COMMON="${RUN_DIRS[0]}"
for d in "${RUN_DIRS[@]:1}"; do
  while [[ -n "$COMMON" && "${d}" != "$COMMON"* ]]; do
    COMMON="${COMMON%?}"
  done
done

# trim trailing separators to make filename nicer
while [[ "$COMMON" =~ [._-]$ ]]; do
  COMMON="${COMMON%?}"
done

# sanitize spaces (just in case)
COMMON="${COMMON// /_}"

OUT="$BASE_RESULTS_DIR/summary_runs_${COMMON}.txt"
echo "Writing summary to: $OUT"

python3 - "$BASE_RESULTS_DIR" "${RUN_DIRS[@]}" <<'PY' | tee "$OUT"
import sys
from summarize_runs import summarize_runs

base = sys.argv[1]
runs = sys.argv[2:]
summarize_runs(runs, base_results_dir=base)
PY
