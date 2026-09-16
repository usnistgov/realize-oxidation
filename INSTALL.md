# Installing REALIZE-oxidation

REALIZE-oxidation combines an E(3)-equivariant neural-network encoder with LAMMPS/ReaxFF atomistic simulations and MPI-parallel nudged elastic band calculations.

## Reproducibility warning

REALIZE-oxidation is sensitive to dependency versions. In testing, incompatible library versions have produced substantially different scientific behavior without necessarily causing the program to crash.

For manuscript reproduction, use the validated environment below and run the supplied environment checks before starting a production calculation.

In particular, do not use `e3nn==0.5.4`.

## 1. Python environment

The validated environment uses Python 3.9.

Create an environment:

```bash
conda create -n realize python=3.9 -y
conda activate realize
```

Install the validated CUDA 11.8 build of PyTorch 2.6.0:

```bash
pip install torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The validated package versions are:

```text
Python            3.9
torch             2.6.0+cu118
numpy             1.26.4
e3nn              0.5.6
sevenn            0.10.3
ase               3.24.0
torch-geometric   2.6.1
PyYAML            6.0.2
matplotlib        3.9.4
scikit-learn      1.6.0
tqdm              4.67.1
scipy             1.13.1
matscipy          1.1.1
```

Check the Python environment before running REALIZE:

```bash
python scripts/check_environment.py
```

The validation script intentionally exits with an error when the environment differs from the validated configuration.

## 2. SevenNet

REALIZE uses SevenNet as the E(3)-equivariant state encoder.

The validated version is:

```text
SevenNet 0.10.3
```

It is installed by the pinned `requirements.txt`.

To install the corresponding SevenNet source explicitly instead:

```bash
git clone https://github.com/MDIL-SNU/SevenNet.git
cd SevenNet
git checkout v0.10.3
pip install .
cd ..
```

Do not copy the SevenNet source tree into the REALIZE repository.

REALIZE includes its own trained embedding checkpoint:

```text
data/checkpoint.pth
```

This is a custom checkpoint used by this work and is not a stock SevenNet pretrained model.

Its SHA-256 checksum is:

```text
75081175553a7a5ee7a208f90d579dedaa7762b710a5c790d7c4da37b6e3b1f5
```

## 3. LAMMPS requirements

REALIZE uses LAMMPS in two ways:

1. Through the LAMMPS Python interface for relaxation and molecular dynamics.
2. Through an MPI-launched LAMMPS executable for partitioned NEB calculations.

The validated calculations used:

```text
LAMMPS       29 Aug 2024 - Update 1
MPI          Open MPI 4.1.0
Packages     MANYBODY MOLECULE REAXFF REPLICA
```

The essential LAMMPS packages for REALIZE are:

```text
REAXFF
REPLICA
```

The validated build also included `MANYBODY` and `MOLECULE`.

Build LAMMPS as an MPI-enabled shared-library build and enable the required packages. A typical CMake configuration is:

```bash
cmake -S cmake -B build \
    -D BUILD_MPI=yes \
    -D BUILD_SHARED_LIBS=yes \
    -D PKG_REAXFF=yes \
    -D PKG_REPLICA=yes \
    -D PKG_MANYBODY=yes \
    -D PKG_MOLECULE=yes
```

Compile the executable:

```bash
cmake --build build -j
```

The LAMMPS Python module used by REALIZE must correspond to this LAMMPS build. Follow the LAMMPS Python-interface installation procedure for the same shared-library build.

Do not assume that an unrelated `pip install lammps` installation is equivalent to the validated build.

Verify the executable with:

```bash
./build/lmp -h
```

The output should report the intended LAMMPS version and include `REAXFF` and `REPLICA` under installed packages.

## 4. Link the LAMMPS executable

REALIZE currently expects the NEB LAMMPS executable at:

```text
data/lmp_neb
```

Do not distribute or commit the compiled executable.

Instead, create a symbolic link:

```bash
ln -s /absolute/path/to/lammps/build/lmp data/lmp_neb
```

Check it with:

```bash
data/lmp_neb -h | head
```

## 5. MPI launcher

The MPI launcher used for NEB must match the MPI implementation against which the LAMMPS executable was compiled.

The validated setup used:

```text
LAMMPS MPI       Open MPI 4.1.0
MPI launcher     Open MPI 4.1.0
launcher         /usr/bin/mpirun
```

On the system used for the published calculations:

```bash
/usr/bin/mpirun --version
```

reported:

```text
mpirun (Open MPI) 4.1.0
```

and the LAMMPS executable reported:

```text
MPI v3.1: Open MPI v4.1.0
```

Set the launcher explicitly:

```bash
export NEB_MPIRUN=/usr/bin/mpirun
```

Do not use a MPICH/HYDRA `mpirun` with an Open-MPI-linked LAMMPS executable.

For example, a `mpirun` supplied by a conda environment may be a different MPI implementation even if it has the same executable name.

Check the pairing with:

```bash
data/lmp_neb -h | grep -m1 '^MPI '
"$NEB_MPIRUN" --version
ldd data/lmp_neb | grep -Ei 'mpi|openmpi|mpich'
```

Then run:

```bash
bash scripts/check_lammps.sh
```

## 6. ReaxFF parameterization

REALIZE uses the Si/O/H ReaxFF parameterization reported in:

Junichi Noaki, Satoshi Numazawa, Joohyun Jeon, et al.,
"Development of the reactive force field and silicon dry/wet oxidation process modeling,"
npj Computational Materials 9, 161 (2023).

DOI:

```text
10.1038/s41524-023-01112-6
```

Download the ReaxFF parameter file from the supplementary material associated with that publication.

Place it at:

```text
data/41524_2023_1112_MOESM1_ESM.txt
```

REALIZE expects this filename.

The parameter file is not redistributed in this repository.

## 7. Validate the complete installation

Before running REALIZE, the `data/` directory should contain:

```text
data/
├── 41524_2023_1112_MOESM1_ESM.txt
├── checkpoint.pth
├── lmp_neb -> /path/to/lammps/build/lmp
├── neb_lammps.in
├── O2.dat
├── README.md
└── str.dat
```

Run both validation scripts:

```bash
python scripts/check_environment.py
bash scripts/check_lammps.sh
```

Do not start a production or manuscript-reproduction calculation unless both checks pass.

## 8. Training

The primary configuration file is:

```text
input.yaml
```

A direct training run is:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun

python scripts/check_environment.py
bash scripts/check_lammps.sh

python realize/train.py --input input.yaml
```

For a SLURM system, first edit the cluster-specific resource requests at the top of:

```text
examples/slurm_train.sh
```

Then submit:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun
sbatch examples/slurm_train.sh
```

## 9. Evaluation

`data/checkpoint.pth` is the embedding checkpoint. It is not a trained REALIZE PPO policy checkpoint.

Trained REALIZE policy checkpoints are generated under the results directory during training.

For evaluation:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun

CHECKPOINT=results/checkpoints/checkpoint_ep550.pth \
OUTPUT_DIR=eval_runs/ep550 \
sbatch examples/slurm_eval.sh
```

The checkpoint path and output directory can be changed through the corresponding environment variables.
