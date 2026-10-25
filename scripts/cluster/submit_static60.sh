#!/bin/bash
# Dynamic decomposition vs static baselines on the 60-frame, 3-camera scene (PRESET=f60).
# All three share every setting except the dynamic branch; score with score_regions.py.
set -euo pipefail
cd "$(dirname "$0")"
X=--exclude=atl1-1-03-013-3-0,atl1-1-02-011-31-0
export PRESET=f60
sbatch $X --time=02:00:00 --job-name=dyn60 train.sbatch dynamic
sbatch $X --time=02:00:00 --job-name=S1_60 train.sbatch S1_static_vanilla \
  features.dynamic=false features.mask_unmodelled_actors=false
sbatch $X --time=02:00:00 --job-name=S2_60 train.sbatch S2_static_masked features.dynamic=false
