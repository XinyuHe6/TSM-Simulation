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

SEED=0

OUT_DIR="out_A${A}_I${I}_T${T_START}-${T_END}-s${T_STEP}_E${EDGE_POINTS}"
mkdir -p "$OUT_DIR"

python3 driver.py \
  --A "$A" \
  --I "$I" \
  --T_start "$T_START" \
  --T_end "$T_END" \
  --T_step "$T_STEP" \
  --edge_points "$EDGE_POINTS" \
  --num_graphs_per_cell "$NUM_GRAPHS_PER_CELL" \
  --runs_per_graph "$RUNS_PER_GRAPH" \
  --seed "$SEED" \
  --out_csv "$OUT_DIR/results.csv" \
  --out_fig "$OUT_DIR/surface.png"
