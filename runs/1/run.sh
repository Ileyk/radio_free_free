#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$SCRIPT_DIR"

STEP_NAMES=()
STEP_SECONDS=()

run_timed() {
  local name="$1"
  shift
  local start end elapsed status

  start="$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
    -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  end="$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
    -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')"
  elapsed="$(awk -v start="$start" -v end="$end" \
    'BEGIN { printf "%.3f", end - start }')"
  STEP_NAMES+=("$name")
  STEP_SECONDS+=("$elapsed")
  return "$status"
}

print_timing_summary() {
  local i
  printf "\nTiming summary:\n"
  for ((i = 0; i < ${#STEP_NAMES[@]}; i++)); do
    printf "  %-24s %8.3f s\n" "${STEP_NAMES[i]}" "${STEP_SECONDS[i]}"
  done
}

finish() {
  local exit_status=$?
  trap - EXIT
  print_timing_summary
  exit "$exit_status"
}
trap finish EXIT

run_timed "Generate flow" rff-flow wind flow.rff --resolution 64 64 64 \
  --inner-radius 1 --outer-radius 10 \
  --ni_Zi2_per_baryon 1 --eff_Z_i 1.2
run_timed "Add sphere occultation" rff-occult flow.rff tmp.rff \
  --shape sphere --center 0 0 0 --radius 1
run_timed "Add cylinder occultation" rff-occult tmp.rff occulted-flow.rff \
  --shape cylinder \
  --center 0 2 0 --radius 0.8 --height 0.5
run_timed "Compute spectrum" "$REPO_ROOT/build/rff-transfer" \
  --input occulted-flow.rff --output flow-spectrum.csv \
  --n0 1e9 --temperature0 3e5 --length0 70 \
  --frequency-min-hz 1e9 --frequency-max-hz 1e12 --frequency-count 100 \
  --view-theta-deg 53 --view-phi-deg 0 --relative_intensity_threshold 1e-2 \
  --adaptive-longitudinal --integration-tolerance 1e-1
run_timed "Plot spectrum" rff-plot "$SCRIPT_DIR/flow-spectrum.csv"
