#!/bin/bash
# Capacity-vs-iterations experiment rerun with scale-corrected MCMC noise (train.noise_lr=210;
# gsplat's 5e5 made budgets >= 3M collapse on the 78 m full log, see README "Known traps").
# Per-user cap on coe-ice: 32 GPU-hours of *requested* time across running jobs.
set -euo pipefail
cd "$HOME/dev/semantic-3dgs"
X=--exclude=atl1-1-03-013-3-0,atl1-1-02-011-31-0
S=scripts/cluster/train.sbatch
N="train.noise_lr=210"
BUDGET="init.max_background_points=2600000 train.cap_max=4000000"
LONG="train.iterations=120000 train.refine_stop=60000"

sbatch $X -J base_n210 --time=03:00:00 $S base_n210 $N
sbatch $X -J A_n210    --time=04:00:00 $S A_budget4M_n210 $N $BUDGET
sbatch $X -J B_n210    --time=05:00:00 $S B_iters120k_n210 $N $LONG
sbatch $X -J C_n210    --time=06:00:00 $S C_budget4M_iters120k_n210 $N $BUDGET $LONG
sbatch $X -J D_n210    --time=05:00:00 $S D_budget6M_n210 $N init.max_background_points=4000000 train.cap_max=6000000
