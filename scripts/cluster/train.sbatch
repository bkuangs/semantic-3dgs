#!/bin/bash
# Usage: sbatch [--time=...] scripts/cluster/train.sbatch JOB_NAME [SECTION.KEY=VALUE ...]
# Trains m4_semantic on the full AV2 scene with the settings of the original full run;
# extra overrides are appended and win over the defaults below.
# PRESET=f60 uses the 60-frame, 3-camera scene with the settings of the original 60-frame run.
# S3DGS_ENTRY="python scripts/cluster/diag_train.py" DIAG_STOP_STEP=N runs the diagnostic trainer instead.
#SBATCH --job-name=s3dgs
#SBATCH --partition=coe-gpu,ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/hice1/bkuang7/scratch/s3dgs/logs/%x-%j.out
set -euo pipefail
JOB="$1"; shift
cd "$HOME/dev/semantic-3dgs"
source scripts/cluster/env.sh
PRESET="${PRESET:-full}"
case "$PRESET" in
  full) OUT="$S3DGS_ROOT/outputs/av2_full/$JOB"; ITERS=60000
        BASE=("data.scene_dir=$SCENE" "data.cameras=[FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT]"
              data.frame_range=null train.iterations=60000 train.lr_means=6.25e-5) ;;
  f60)  OUT="$S3DGS_ROOT/outputs/av2_60/$JOB"; ITERS=30000
        BASE=("data.scene_dir=$SCENE60" "data.cameras=[FRONT,FRONT_LEFT,FRONT_RIGHT]"
              data.frame_range=null train.iterations=30000) ;;
  *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac
if [ -e "$OUT/train_log.jsonl" ]; then echo "refusing to reuse $OUT" >&2; exit 1; fi
# Private, node-local copy of the compiled gsplat: ninja may decide to rebuild on a new node, and
# concurrent rebuilds in the shared cache from different nodes clobber each other.
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/s3dgs-ext-$SLURM_JOB_ID"
mkdir -p "$TORCH_EXTENSIONS_DIR"
cp -a "$XDG_CACHE_HOME/torch_extensions/py311_cu130/." "$TORCH_EXTENSIONS_DIR/"
rm -f "$TORCH_EXTENSIONS_DIR"/gsplat_cuda/lock
trap 'rm -rf "$TORCH_EXTENSIONS_DIR"' EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git --no-pager log --oneline -1

for kv in "$@"; do case "$kv" in train.iterations=*) ITERS="${kv#*=}";; esac; done

uv run --no-sync ${S3DGS_ENTRY:-s3dgs} train configs/waymo/m4_semantic.yaml --set \
  "${BASE[@]}" "output_dir=$OUT" data.cache_images=true "train.checkpoint_every=$ITERS" \
  "$@"

if [ -d "$OUT/eval" ]; then uv run --no-sync python scripts/cluster/score_front.py "$OUT"; fi
uv run --no-sync python scripts/cluster/score_regions.py "$OUT"
