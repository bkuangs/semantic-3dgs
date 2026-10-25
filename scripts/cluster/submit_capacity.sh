#!/bin/bash
# Submits the capacity-vs-iterations experiment (one GPU per job (H100/H200/A100, whichever frees first), all run in parallel).
# Run scripts/cluster/smoke.sbatch first so gsplat is compiled once before these start.
# Per-user cap on coe-ice: 32 GPU-hours of *requested* time across running jobs, so keep --time tight.
# Measured on H100: 1.7M Gaussians ~20 it/s, 4M ~12.5 it/s. First job on each node recompiles gsplat (~7-10 min).
set -euo pipefail
cd "$HOME/dev/semantic-3dgs"
S=scripts/cluster/train.sbatch
BUDGET="init.max_background_points=2600000 train.cap_max=4000000"
LONG="train.iterations=120000 train.refine_stop=60000"

sbatch -J base   "$S" base                                     # original settings, same GPU/env as the rest
sbatch -J A      "$S" A_budget4M     $BUDGET
sbatch -J B      --time=06:00:00 "$S" B_iters120k  $LONG
sbatch -J C      --time=06:00:00 "$S" C_budget4M_iters120k $BUDGET $LONG
sbatch -J D      --time=05:00:00 "$S" D_budget6M     init.max_background_points=4000000 train.cap_max=6000000
sbatch -J E      "$S" E_front3       "data.cameras=[FRONT,FRONT_LEFT,FRONT_RIGHT]"
