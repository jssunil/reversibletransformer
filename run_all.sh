#!/usr/bin/env bash
# Full assignment pipeline. Usage: bash run_all.sh [screen|main|all]
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:-all}
B0=32
maxb() { python -c "import json;print(json.load(open('results/maxbatch_$1_$2.json'))['max_batch'])"; }

if [[ $STAGE == screen || $STAGE == all ]]; then
  # Integrator screening: 5M tokens each at the baseline batch
  S="--tokens 5e6 --batch $B0 --out results/screen"
  python train.py --name residual         --trunk residual $S
  python train.py --name midpoint_h0.5    --trunk midpoint_rev --h 0.5  --stream fp64 $S
  python train.py --name midpoint_h0.25   --trunk midpoint_rev --h 0.25 --stream fp64 $S
  python train.py --name midpoint_h0.5_fp32stream --trunk midpoint_rev --h 0.5 --stream fp32 $S
  python train.py --name reveuler_h1.0    --trunk reveuler_rev --h 1.0  --stream fp64 $S
  python train.py --name reveuler_h0.5    --trunk reveuler_rev --h 0.5  --stream fp64 $S
  python train.py --name reveuler_h1.0_fp32stream --trunk reveuler_rev --h 1.0 --stream fp32 $S
  python train.py --name reveuler_h0.5_fp32stream --trunk reveuler_rev --h 0.5 --stream fp32 $S
  python train.py --name midpoint_h0.5_stored     --trunk midpoint     --h 0.5 $S
fi

if [[ $STAGE == main || $STAGE == all ]]; then
  REV=${REV:-reveuler_rev}; H=${H:-0.5}   # winner of the screening stage
  python find_max_batch.py --trunk residual
  python find_max_batch.py --trunk "$REV" --h "$H" --stream fp64 --start 64
  python train.py --name run1_baseline      --trunk residual --batch $B0
  python train.py --name run2_rev_same      --trunk "$REV" --h "$H" --stream fp64 --batch $B0
  python train.py --name run3_rev_max       --trunk "$REV" --h "$H" --stream fp64 --batch "$(maxb "$REV" fp64)" --scale_lr
  python train.py --name run1b_baseline_max --trunk residual --batch "$(maxb residual fp32)" --scale_lr
  # supplementary: fp32 residual stream (faster, approximate reconstruction)
  python find_max_batch.py --trunk "$REV" --h "$H" --stream fp32 --start 64
  python train.py --name run3b_rev_max_fp32stream --trunk "$REV" --h "$H" --stream fp32 --batch "$(maxb "$REV" fp32)" --scale_lr
fi
python make_report.py
