#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$SCRIPT_DIR"

rff-flow wind flow.rff --resolution 256 256 256 \
  --inner-radius 1 --outer-radius 10 \
  --ni_Zi2_per_baryon 1 --eff_Z_i 1.2
rff-occult flow.rff tmp.rff --shape sphere --center 0 0 0 --radius 1
rff-occult tmp.rff occulted-flow.rff --shape cylinder \
  --center 0 2 0 --radius 0.8 --height 0.5
time "$REPO_ROOT/build/rff-transfer" \
  --input occulted-flow.rff --output flow-spectrum.csv \
  --n0 1e9 --temperature0 3e5 --length0 70 \
  --frequency-min-hz 1e9 --frequency-max-hz 1e12 --frequency-count 100 \
  --view-theta-deg 53 --view-phi-deg 0 --relative_intensity_threshold 1e-2
rff-plot "$SCRIPT_DIR/flow-spectrum.csv"
