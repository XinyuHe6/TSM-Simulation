#!/usr/bin/env bash
set -euo pipefail

# ===== fixed experiment config =====
PYTHON_BIN="${PYTHON_BIN:-python3}"

N=100   # A = I = T = N

# Correlated Sampling solves the natural LP at every point,
# so these defaults stay fairly small.
NUM_GRAPHS_PER_POINT=2
RUNS_PER_GRAPH=3

USE_POISSON_LEN=1   # 0 or 1

LP_MAX_ROUNDS=20
LP_SEP_TOL=1e-9
LP_CONSTRAINT_MODE="pair_approx"  # natural or pair_approx
LP_PAIR_CAP=""                    # empty means pair caps use 1 - exp(-(lambda_i + lambda_j))

SEED=0

# ===== random graph sweep =====
EDGE_POINTS=101

# ===== k-regular sweep =====
REGULAR_DEGREE_START=0
REGULAR_DEGREE_END="$N"
REGULAR_DEGREE_STEP=1

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
fi

PAIR_CAP_FLAG=()
if [ -n "$LP_PAIR_CAP" ]; then
  PAIR_CAP_FLAG+=(--lp_pair_cap "$LP_PAIR_CAP")
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

  local mode_tag="$LP_CONSTRAINT_MODE"
  local out_dir="out_corr_${mode_tag}_${graph_tag}_N${N}"
  mkdir -p "$out_dir"

  echo "Running $graph_mode -> $out_dir"

"$PYTHON_BIN" plot_correlated_sampling_2d.py \
  --N "$N" \
  --num_graphs_per_point "$NUM_GRAPHS_PER_POINT" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --lp_max_rounds "$LP_MAX_ROUNDS" \
  --lp_separation_tol "$LP_SEP_TOL" \
  --lp_constraint_mode "$LP_CONSTRAINT_MODE" \
  --skip_failed_points \
  --seed "$SEED" \
  --out_csv "$out_dir/results.csv" \
  --out_fig "$out_dir/plot.png" \
  "${graph_args[@]}" \
  "${POISSON_FLAG[@]}" \
  "${PAIR_CAP_FLAG[@]}"
}

run_plot "k_regular"
run_plot "random"
