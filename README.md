# REALIZE-oxidation

E(3)-equivariant deep reinforcement learning for goal-directed atomistic pathway discovery, demonstrated for O2 transport and dissociation at the Si/a-SiO2 interface.

REALIZE treats atomistic rare-event sampling as sequential decision-making. An O2 molecule acts as the agent and proposes continuous rigid-body translations and rotations in an evolving Si/a-SiO2 environment. Proposed moves are relaxed and evaluated with atomistic simulations, including nudged elastic band calculations and molecular-dynamics verification. PPO trains the policy to discover chemically successful pathways while favoring kinetically accessible transitions.

This repository accompanies:

> Wonseok Jeong, Francesca Tavazza, and Brian DeCost,
> **Bridging Atomistic Simulation and Experimental Processing Timescales with Goal-Directed Deep Reinforcement Learning**,
> arXiv:2605.16214 (2026).
> https://arxiv.org/abs/2605.16214

---

## Important reproducibility warning

REALIZE-oxidation is sensitive to software-library versions.

In our testing, incompatible versions have produced substantially different scientific behavior without necessarily causing a Python exception or other obvious failure.

In particular, **do not use `e3nn==0.5.4`**. Runs with that release failed to reproduce the expected training behavior while completing without a software crash.

The validated environment for this release is:

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

Before running REALIZE, use:

```bash
python scripts/check_environment.py
bash scripts/check_lammps.sh
```

See [`INSTALL.md`](INSTALL.md) for the complete installation and validation procedure.

---

## Repository versions

REALIZE-oxidation is distributed in two Git branches.

| Branch | Intended use | Status |
| --- | --- | --- |
| `manuscript` | Reproduction and inspection of the implementation associated with the published study | Validated manuscript implementation |
| `main` | Continued development, corrected accounting behavior, and updated reward formulation | Recommended for new work |

The `manuscript` branch preserves implementation behavior associated with the reported calculations as closely as practical.

The `main` branch contains subsequent corrections and methodological updates. Because these changes can alter training behavior, results from `main` should not be assumed to reproduce the manuscript calculations exactly.

---

## Method overview

At each decision step, the O2 agent proposes:

1. a translation direction on the sphere,
2. a translation magnitude, and
3. a directed O-O molecular axis.

The policy is symmetry-aware and uses an E(3)-equivariant SevenNet-based representation of the atomic environment.

The proposed state is then processed through the atomistic environment. Depending on the stage of the calculation, this includes:

- LAMMPS/ReaxFF structural relaxation,
- NEB barrier evaluation,
- molecular-dynamics verification,
- dissociation detection,
- reward calculation, and
- PPO policy/value updates.

The framework does not require a predefined reaction coordinate for the O2 transport or dissociation pathway.

---

## Repository structure

```text
REALIZE-oxidation/
├── data/
│   ├── checkpoint.pth
│   ├── neb_lammps.in
│   ├── O2.dat
│   ├── str.dat
│   └── README.md
│
├── evaluation/
│   ├── compute_eval_metrics.py
│   ├── deterministic_action.py
│   ├── estimate_pathway_time.py
│   └── eval_runner.py
│
├── examples/
│   ├── slurm_train.sh
│   └── slurm_eval.sh
│
├── realize/
│   ├── config/
│   ├── environment/
│   ├── networks/
│   ├── ppo/
│   ├── training/
│   ├── utils/
│   └── train.py
│
├── scripts/
│   ├── check_environment.py
│   └── check_lammps.sh
│
├── input.yaml
├── requirements.txt
├── INSTALL.md
└── README.md
```

Generated runtime data such as `results/`, `tmp/`, logs, and training outputs are intentionally excluded from version control.

---

## Required external components

### SevenNet

REALIZE uses SevenNet as the E(3)-equivariant state encoder.

The validated release uses:

```text
SevenNet 0.10.3
```

SevenNet is installed as a Python dependency rather than vendored into this repository.

The custom trained embedding checkpoint used by REALIZE is included as:

```text
data/checkpoint.pth
```

SHA-256:

```text
75081175553a7a5ee7a208f90d579dedaa7762b710a5c790d7c4da37b6e3b1f5
```

This file is a checkpoint trained for this work and is not a stock SevenNet pretrained model.

### LAMMPS

REALIZE requires an MPI-enabled LAMMPS build with the `REAXFF` and `REPLICA` packages.

The validated manuscript environment used:

```text
LAMMPS       29 Aug 2024 - Update 1
MPI          Open MPI 4.1.0
Packages     MANYBODY MOLECULE REAXFF REPLICA
```

The compiled LAMMPS executable is intentionally **not distributed** in this repository.

After building LAMMPS, the current implementation expects a symbolic link at:

```text
data/lmp_neb
```

For example:

```bash
ln -s /absolute/path/to/lammps/build/lmp data/lmp_neb
```

### MPI compatibility

Partitioned NEB calculations are launched through `mpirun`.

Set `NEB_MPIRUN` explicitly to the launcher associated with the MPI implementation used to build LAMMPS.

For the validated manuscript environment:

```bash
export NEB_MPIRUN=/usr/bin/mpirun
```

Both the LAMMPS executable and this launcher used Open MPI 4.1.0.

Do not pair an Open-MPI-linked LAMMPS executable with an MPICH/HYDRA launcher merely because both executables are named `mpirun`.

Validate the combination with:

```bash
bash scripts/check_lammps.sh
```

### ReaxFF parameterization

The Si/O/H ReaxFF parameterization used by REALIZE is from:

> Junichi Noaki, Satoshi Numazawa, Joohyun Jeon, et al.
> **Development of the reactive force field and silicon dry/wet oxidation process modeling**
> npj Computational Materials 9, 161 (2023).
> https://doi.org/10.1038/s41524-023-01112-6

The force-field parameter file is intentionally not redistributed here.

Download the supplementary parameter file associated with that publication and place it at:

```text
data/41524_2023_1112_MOESM1_ESM.txt
```

See [`data/README.md`](data/README.md) for details.

---

## Installation

See:

```text
INSTALL.md
```

for the complete procedure.

The short version is:

```bash
conda create -n realize python=3.9 -y
conda activate realize

pip install torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

After installing/building LAMMPS and supplying the ReaxFF parameter file:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun

python scripts/check_environment.py
bash scripts/check_lammps.sh
```

Both validation checks should pass before a production calculation is started.

---

## Running training

The primary user configuration is:

```text
input.yaml
```

Run directly with:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun
python realize/train.py --input input.yaml
```

A SLURM submission example is provided in:

```text
examples/slurm_train.sh
```

Because partitions, GPU resource names, wall-time limits, and task layouts differ between computing systems, edit the `#SBATCH` directives for your cluster before submission.

For example:

```bash
export NEB_MPIRUN=/path/to/matching/mpirun
sbatch examples/slurm_train.sh
```

---

## Evaluation

Evaluation utilities are under:

```text
evaluation/
```

A trained REALIZE PPO checkpoint can be evaluated with:

```bash
python -m evaluation.eval_runner \
    --checkpoint results/checkpoints/checkpoint_ep550.pth \
    --output-dir eval_runs/ep550 \
    --n-o-target 80 \
    --max-steps 50 \
    --device cuda \
    --random-seed 88
```

A SLURM example is provided in:

```text
examples/slurm_eval.sh
```

Note that:

```text
data/checkpoint.pth
```

is the embedding-model checkpoint and is **not** a trained REALIZE PPO checkpoint.

---

## Configuration

Frequently changed parameters are exposed through:

```text
input.yaml
```

The complete configuration definitions and defaults are implemented under:

```text
realize/config/
```

For manuscript reproduction, use the configuration distributed with the `manuscript` branch rather than copying settings from `main`.

---

## Parallel collection

REALIZE supports parallel episode collection. Each worker uses its own temporary LAMMPS/NEB working directory under:

```text
tmp/run{k}/
```

while the primary process performs PPO updates.

When using parallel collection, allocate sufficient CPU resources for:

```text
num_parallel_workers × cores_per_worker
```

### Manuscript-branch warning

The manuscript implementation contains a known parallel-mode cycle-statistics issue. With more than one parallel worker, cycle advancement can occur before all workers have submitted their cycle statistics.

For manuscript-branch calculations intended for quantitative comparison with the paper, use:

```yaml
num_parallel_workers: 1
```

Serial operation is unaffected.

The corresponding logic is corrected in `main`.

---

## Reward formulation and branch differences

The manuscript and main branches do not use identical reward/accounting behavior.

The manuscript implementation uses the reward formulation and midpoint schedule associated with the reported benchmark calculations.

The `main` branch contains subsequent changes including transition-state-referenced effective barriers, commitment-aware barrier weighting, updated midpoint scheduling, and fixes to several accounting edge cases.

For this reason:

- use `manuscript` for reproduction of the reported implementation,
- use `main` for continued development and new calculations.

---

## Known issues

### Dependency sensitivity

Dependency mismatches can alter training behavior without causing a crash. Use the pinned environment and validation scripts.

In particular, do not use `e3nn==0.5.4`.

### Parallel cycle statistics in `manuscript`

With `num_parallel_workers > 1`, the manuscript implementation may advance a cycle after an incomplete set of worker submissions. Use one parallel worker for manuscript-reproduction calculations.

This behavior is corrected in `main`.

### NEB atom-assignment failures

LAMMPS may intermittently report:

```text
ERROR: Did not assign all atoms correctly
```

during NEB structure loading. Repeated failures trigger REALIZE's recovery logic. The frequency can depend on the execution environment and can significantly increase wall-clock time.

### Catastrophic-path barrier accounting in `manuscript`

In the manuscript branch, a rare recovery path following repeated NEB failures can record an unverified zero activation barrier when dissociation has already been detected.

This path was not triggered in the runs reported in the manuscript. It is retained in the manuscript branch for implementation fidelity and corrected in `main`.

### Actor normalizer initialization

At startup, an O-O sanity check used while initializing actor normalization can fail because the final two atoms of the initialization structure are not the agent O2. The normalizer then begins from its default statistics.

This warning is expected in the manuscript implementation and was present in the reported calculations.

---

## Reproducing the manuscript calculations

For calculations intended to reproduce the published implementation:

1. check out the `manuscript` branch,
2. use Python 3.9 and the pinned library versions,
3. use `num_parallel_workers: 1`,
4. use the supplied `input.yaml`,
5. use the supplied `data/checkpoint.pth`,
6. use the documented ReaxFF parameterization,
7. use an MPI-compatible LAMMPS build with REAXFF and REPLICA,
8. run both validation scripts before training.

The method is stochastic and the interfacial environment evolves during training, so individual trajectories should not be expected to reproduce bit-for-bit across hardware and software environments. Aggregate quantities such as dissociation frequency, completed cycles, barrier distributions, and effective activation energies are more appropriate reproduction targets.

---

## Citation

If you use REALIZE-oxidation, please cite:

```text
W. Jeong, F. Tavazza, and B. DeCost,
"Bridging Atomistic Simulation and Experimental Processing Timescales
with Goal-Directed Deep Reinforcement Learning,"
arXiv:2605.16214 (2026).
```

Paper:

https://arxiv.org/abs/2605.16214

---

## License

Licensing information for the NIST software release is provided in `LICENSE.md`.

Third-party software such as SevenNet and LAMMPS is not redistributed as part of this repository and remains subject to its respective license terms.

---

## Contact

- **Principal Investigator:** Brian DeCost
- **Software contact:** Wonseok Jeong, wonseok.jeong@nist.gov
- **Operating Unit:** Material Measurement Laboratory (MML)
- **Division:** Materials Measurement Science Division (MMSD)
- **Group:** Data and AI-Driven Materials Science Group

---

## Support and maturity

REALIZE-oxidation is research software developed to support the accompanying scientific study.

It is provided primarily for reproducibility, inspection, and research use rather than as a production-supported software package.

Bug reports and reproducibility issues may be submitted through the repository's GitHub issue tracker after public release.
