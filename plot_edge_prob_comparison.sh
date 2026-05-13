#!/usr/bin/env bash
set -euo pipefail

# ===== random graph experiment config =====
A=100
I=100
T=100

# edge_prob is swept from 0 to 1 with this many points.
EDGE_POINTS=21

# averaging
NUM_GRAPHS_PER_POINT=5
RUNS_PER_GRAPH=3

# Manshadi / offline-statistics specific
MC_TRIALS=20
USE_POISSON_LEN=1   # 0 or 1

# Comma-separated algorithms to compare.
# Options:
#   random_matching, degree_matching, tsm, manshadi2, manshadi3, manshadi4
ALGORITHMS="random_matching,degree_matching,tsm,manshadi2,manshadi3,manshadi4"

SEED=0

ALG_TAG="${ALGORITHMS//,/+}"
OUT_DIR="out_edge_prob_comparison_A${A}_I${I}_T${T}_E${EDGE_POINTS}_G${NUM_GRAPHS_PER_POINT}_R${RUNS_PER_GRAPH}_MC${MC_TRIALS}_${ALG_TAG}"
mkdir -p "$OUT_DIR"

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
else
  POISSON_FLAG+=(--no_use_poisson_len)
fi

python3 plot_edge_prob_comparison.py \
  --A "$A" \
  --I "$I" \
  --T "$T" \
  --edge_points "$EDGE_POINTS" \
  --num_graphs_per_point "$NUM_GRAPHS_PER_POINT" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --mc_trials "$MC_TRIALS" \
  --algorithms "$ALGORITHMS" \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig "$OUT_DIR/comparison.png" \
  "${POISSON_FLAG[@]}"
