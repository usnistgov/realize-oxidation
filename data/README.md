# REALIZE-oxidation input data

This directory contains the input files required to run the REALIZE silicon-oxidation benchmark.

## Included files

### `str.dat`

Initial Si/a-SiO2 interfacial structure in LAMMPS data format.

### `O2.dat`

O2 molecular structure used when introducing the agent molecule.

### `checkpoint.pth`

Custom trained SevenNet-based embedding checkpoint used by REALIZE.

SHA-256:

```text
75081175553a7a5ee7a208f90d579dedaa7762b710a5c790d7c4da37b6e3b1f5
```

This checkpoint is part of the REALIZE-oxidation release and should not be replaced by a stock SevenNet pretrained checkpoint.

### `neb_lammps.in`

LAMMPS input template used for nudged elastic band calculations.

REALIZE rewrites the ReaxFF parameter-file path in this input at runtime.

## Files supplied separately by the user

Two external files are intentionally not distributed in this repository.

### LAMMPS executable

REALIZE requires an MPI-enabled LAMMPS build with the REAXFF and REPLICA packages.

The manuscript implementation expects the executable at:

```text
data/lmp_neb
```

After building LAMMPS, create a symbolic link:

```bash
ln -s /absolute/path/to/lammps/build/lmp data/lmp_neb
```

Do not commit the compiled `lmp_neb` binary to this repository.

See `../INSTALL.md` for the validated LAMMPS and MPI configuration.

### Si/O ReaxFF parameter file

REALIZE uses the Si/O/H ReaxFF parameterization reported in:

Junichi Noaki, Satoshi Numazawa, Joohyun Jeon, et al.,
"Development of the reactive force field and silicon dry/wet oxidation process modeling,"
npj Computational Materials 9, 161 (2023).

DOI: 10.1038/s41524-023-01112-6

Download the parameter file from the supplementary material associated with that publication and place it here with exactly this filename:

```text
data/41524_2023_1112_MOESM1_ESM.txt
```

The parameter file is not redistributed by REALIZE-oxidation.
