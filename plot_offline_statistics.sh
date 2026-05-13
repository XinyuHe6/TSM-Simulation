#!/usr/bin/env bash
set -euo pipefail

# ===== A >> I experiment config =====
A=300
I=20

T_START=0
T_END=100
T_STEP=2

NUM_GRAPHS_PER_CELL=5
RUNS_PER_GRAPH=3

# ===== offline-statistics specific =====
MC_TRIALS=20
USE_POISSON_LEN=1   # 0 or 1

SEED=0

# ===== random graph sweep =====
EDGE_POINTS=31

# ===== k-regular sweep =====
# For A != I, k_regular means each impression type connects to exactly k advertisers.
REGULAR_DEGREE_START=1
REGULAR_DEGREE_END=20
REGULAR_DEGREE_STEP=1

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
fi

run_plot() {
  local graph_mode="$1"
  local graph_tag
  local -a graph_args

  graph_args=(--graph_mode "$graph_mode")

  if [ "$graph_mode" = "random" ]; then
    graph_tag="random_E${EDGE_POINTS}"
    graph_args+=(--edge_points "$EDGE_POINTS")
  elif [ "$graph_mode" = "k_regular" ]; then
    graph_tag="kreg_K${REGULAR_DEGREE_START}-${REGULAR_DEGREE_END}-s${REGULAR_DEGREE_STEP}"
    graph_args+=(
      --regular_degree_start "$REGULAR_DEGREE_START"
      --regular_degree_end "$REGULAR_DEGREE_END"
      --regular_degree_step "$REGULAR_DEGREE_STEP"
    )
  else
    echo "Unsupported graph mode: $graph_mode"
    exit 1
  fi

  local out_dir="out_offstat_${graph_tag}_A${A}_I${I}_T${T_START}-${T_END}-s${T_STEP}_MC${MC_TRIALS}"
  mkdir -p "$out_dir"

  echo "Running $graph_mode -> $out_dir"

  python3 driver_offline_statistics.py \
    --A "$A" \
    --I "$I" \
    --T_start "$T_START" \
    --T_end "$T_END" \
    --T_step "$T_STEP" \
    --num_graphs_per_cell "$NUM_GRAPHS_PER_CELL" \
    --runs_per_graph "$RUNS_PER_GRAPH" \
    --mc_trials "$MC_TRIALS" \
    --seed "$SEED" \
    --out_csv "$out_dir/results.csv" \
    --out_fig "$out_dir/surface.png" \
    "${graph_args[@]}" \
    "${POISSON_FLAG[@]}"
}

run_plot "k_regular"
run_plot "random"
