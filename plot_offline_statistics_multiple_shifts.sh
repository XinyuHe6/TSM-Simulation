#!/usr/bin/env bash
set -euo pipefail

# ===== fixed comparison config =====
A=100
I=100
T=100

EDGE_POINTS=21
NUM_GRAPHS_PER_POINT=5
RUNS_PER_GRAPH=3

MC_TRIALS=20
USE_POISSON_LEN=1   # 0 or 1

# Comma-separated try counts to compare. The nth try uses shift n-1.
TRIES="2,3,4,5,6,7,8,9,10"

SEED=0

TRIES_TAG="${TRIES//,/}"
OUT_DIR="out_offstat_multiple_shifts_A${A}_I${I}_T${T}_E${EDGE_POINTS}_MC${MC_TRIALS}_tries${TRIES_TAG}"
mkdir -p "$OUT_DIR"

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
else
  POISSON_FLAG+=(--no_use_poisson_len)
fi

python3 plot_offline_statistics_multiple_shifts.py \
  --A "$A" \
  --I "$I" \
  --T "$T" \
  --edge_points "$EDGE_POINTS" \
  --num_graphs_per_point "$NUM_GRAPHS_PER_POINT" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --mc_trials "$MC_TRIALS" \
  --tries "$TRIES" \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig "$OUT_DIR/comparison.png" \
  "${POISSON_FLAG[@]}"
