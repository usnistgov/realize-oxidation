#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --job-name=REALIZE-eval
#SBATCH --output=slurm-%j.out

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "REALIZE evaluation job ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.9}"

if [[ -z "${NEB_MPIRUN:-}" ]]; then
    echo "ERROR: NEB_MPIRUN is not set."
    echo "Set it to the MPI launcher matching data/lmp_neb."
    exit 2
fi

if [[ ! -x "data/lmp_neb" ]]; then
    echo "ERROR: data/lmp_neb is missing or not executable."
    exit 2
fi

if [[ ! -x "$NEB_MPIRUN" ]]; then
    echo "ERROR: NEB_MPIRUN does not point to an executable:"
    echo "  $NEB_MPIRUN"
    exit 2
fi

python scripts/check_environment.py

python - <<'PYCUDA'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available inside this GPU job.")
    sys.exit(2)

print("[OK] CUDA available")
print("[OK] GPU:", torch.cuda.get_device_name(0))
PYCUDA

bash scripts/check_lammps.sh

CHECKPOINT="${CHECKPOINT:-results/checkpoints/checkpoint_ep550.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_runs/ep550}"
N_O_TARGET="${N_O_TARGET:-80}"
MAX_STEPS="${MAX_STEPS:-50}"
SEED="${SEED:-88}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: evaluation checkpoint not found:"
    echo "  $CHECKPOINT"
    echo
    echo "Specify one when submitting, for example:"
    echo
    echo "  CHECKPOINT=results/checkpoints/checkpoint_ep550.pth \\"
    echo "    sbatch examples/slurm_eval.sh"
    exit 2
fi

python -m evaluation.eval_runner \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --n-o-target "$N_O_TARGET" \
    --max-steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --random-seed "$SEED"
