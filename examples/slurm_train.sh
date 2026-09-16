#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --job-name=REALIZE-train
#SBATCH --output=slurm-%j.out

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "REALIZE training job ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"

# Avoid thread oversubscription. NEB uses MPI ranks separately.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Used for the validated REALIZE runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.9}"

# IMPORTANT:
# NEB_MPIRUN must use the same MPI implementation as data/lmp_neb.
# Do not point this to a different conda MPI installation.
if [[ -z "${NEB_MPIRUN:-}" ]]; then
    echo "ERROR: NEB_MPIRUN is not set."
    echo
    echo "Set it to the MPI launcher matching your LAMMPS build."
    echo "For example:"
    echo "  export NEB_MPIRUN=/usr/bin/mpirun"
    echo
    echo "Check compatibility with:"
    echo "  \$NEB_MPIRUN --version"
    echo "  ldd data/lmp_neb | grep -Ei 'mpi|openmpi|mpich'"
    exit 2
fi

required_files=(
    "data/str.dat"
    "data/O2.dat"
    "data/checkpoint.pth"
    "data/neb_lammps.in"
    "data/41524_2023_1112_MOESM1_ESM.txt"
)

for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $f"
        exit 2
    fi
done

if [[ ! -x "data/lmp_neb" ]]; then
    echo "ERROR: data/lmp_neb is missing or not executable."
    echo
    echo "Build LAMMPS with MPI, REAXFF, and REPLICA support, then create"
    echo "a symbolic link such as:"
    echo
    echo "  ln -s /absolute/path/to/lmp data/lmp_neb"
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

echo
echo "LAMMPS executable:"
readlink -f data/lmp_neb
echo
echo "MPI launcher:"
readlink -f "$NEB_MPIRUN"
echo

python realize/train.py --input input.yaml
