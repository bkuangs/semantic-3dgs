# Source from a PACE ICE shell or sbatch script: keeps the env, caches and outputs on scratch.
export S3DGS_ROOT="${S3DGS_ROOT:-$HOME/scratch/s3dgs}"
export UV_CACHE_DIR="$S3DGS_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$S3DGS_ROOT/venv"
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$S3DGS_ROOT/cache"
# gsplat is JIT-compiled once and shared by every job, so build it for every GPU type we may land on.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
unset CUDA_HOME
export PATH="$HOME/.local/bin:$PATH"
export SCENE="$HOME/dev/semantic-3dgs/data/scenes/av2_full/070bbf42-31d3-3aa9-aca4-c262afc9077d"
export SCENE60="$HOME/dev/semantic-3dgs/data/scenes/av2/070bbf42-31d3-3aa9-aca4-c262afc9077d"
