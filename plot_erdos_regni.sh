#!/usr/bin/env bash
set -euo pipefail

# ===== Erdos-Renyi bipartite G(n,n,p=k/n) config =====
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

N=100
T="$N"

K_START=0
K_END=10
K_STEP=0.25

NUM_GRAPHS_PER_K=3
RUNS_PER_GRAPH=3

# Manshadi / offline-statistics specific
MC_TRIALS=20

# Correlated Sampling LP options. pair_approx is much faster than natural.
CORR_LP_CONSTRAINT_MODE="pair_approx"  # natural or pair_approx
CORR_LP_MAX_ROUNDS=20
CORR_LP_SEP_TOL=1e-9
CORR_LP_PAIR_CAP=""                    # empty means pair caps use 1 - exp(-(lambda_i + lambda_j))

# 0 means exactly T arrivals per run; 1 means Poisson(T) arrivals per run.
USE_POISSON_LEN=0

# Comma-separated algorithms to compare.
# Options:
#   tsm, correlated_sampling, manshadi2, manshadi3, manshadi4, random, degree_matching
ALGORITHMS="tsm,correlated_sampling,manshadi2,manshadi3,manshadi4,random,degree_matching"

SEED=0

ALG_TAG="${ALGORITHMS//,/+}"
OUT_DIR="out_erdos_regni_n${N}_T${T}_K${K_START}-${K_END}-s${K_STEP}_G${NUM_GRAPHS_PER_K}_R${RUNS_PER_GRAPH}_MC${MC_TRIALS}_${CORR_LP_CONSTRAINT_MODE}_${ALG_TAG}"
mkdir -p "$OUT_DIR"

POISSON_FLAG=()
if [ "$USE_POISSON_LEN" -eq 1 ]; then
  POISSON_FLAG+=(--use_poisson_len)
else
  POISSON_FLAG+=(--no_use_poisson_len)
fi

PAIR_CAP_FLAG=()
if [ -n "$CORR_LP_PAIR_CAP" ]; then
  PAIR_CAP_FLAG+=(--corr_lp_pair_cap "$CORR_LP_PAIR_CAP")
fi

"$PYTHON_BIN" plot_erdos_regni.py \
  --n "$N" \
  --T "$T" \
  --k_start "$K_START" \
  --k_end "$K_END" \
  --k_step "$K_STEP" \
  --num_graphs_per_k "$NUM_GRAPHS_PER_K" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --mc_trials "$MC_TRIALS" \
  --algorithms "$ALGORITHMS" \
  --corr_lp_constraint_mode "$CORR_LP_CONSTRAINT_MODE" \
  --corr_lp_max_rounds "$CORR_LP_MAX_ROUNDS" \
  --corr_lp_separation_tol "$CORR_LP_SEP_TOL" \
  --skip_failed_points \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig_dir "$OUT_DIR" \
  "${POISSON_FLAG[@]}" \
  "${PAIR_CAP_FLAG[@]}"
