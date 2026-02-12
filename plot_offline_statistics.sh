#!/usr/bin/env bash
set -euo pipefail

# ===== change params here =====
A=100
I=100

T_START=0
T_END=100
T_STEP=5

EDGE_POINTS=21

NUM_GRAPHS_PER_CELL=5
RUNS_PER_GRAPH=3

# ===== offline-statistics specific =====
MC_TRIALS=200
USE_POISSON_LEN=0   # 0 or 1

SEED=0

OUT_DIR="out_offstat_A${A}_I${I}_T${T_START}-${T_END}-s${T_STEP}_E${EDGE_POINTS}_MC${MC_TRIALS}"
mkdir -p "$OUT_DIR"

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
fi

python3 driver_offline_statistics.py \
  --A "$A" \
  --I "$I" \
  --T_start "$T_START" \
  --T_end "$T_END" \
  --T_step "$T_STEP" \
  --edge_points "$EDGE_POINTS" \
  --num_graphs_per_cell "$NUM_GRAPHS_PER_CELL" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --mc_trials "$MC_TRIALS" \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig "$OUT_DIR/surface.png" \
  "${POISSON_FLAG[@]}"
