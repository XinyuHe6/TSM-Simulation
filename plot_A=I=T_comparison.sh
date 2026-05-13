#!/usr/bin/env bash
set -euo pipefail

# ===== set A = I = T = N here =====
N=100

# ===== edge_prob sweep =====
EDGE_POINTS=21

# ===== averaging =====
NUM_GRAPHS_PER_POINT=5
RUNS_PER_GRAPH=3

# ===== function_offline_statistics specific =====
MC_TRIALS=20
USE_POISSON_LEN=1   # 0 or 1

SEED=0

OUT_DIR="out_AeqIeqT_N${N}_E${EDGE_POINTS}_G${NUM_GRAPHS_PER_POINT}_R${RUNS_PER_GRAPH}_MC${MC_TRIALS}"
mkdir -p "$OUT_DIR"

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
else
  POISSON_FLAG+=(--no_use_poisson_len)
fi

python3 plot_AeqIeqT_comparison.py \
  --N "$N" \
  --edge_points "$EDGE_POINTS" \
  --num_graphs_per_point "$NUM_GRAPHS_PER_POINT" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --mc_trials "$MC_TRIALS" \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig "$OUT_DIR/comparison.png" \
  "${POISSON_FLAG[@]}"
