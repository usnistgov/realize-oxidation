#!/bin/bash

set -euo pipefail

LMP="${REALIZE_LAMMPS:-data/lmp_neb}"
MPI="${NEB_MPIRUN:-}"

errors=0

echo "REALIZE LAMMPS/MPI validation"
echo "======================================================"

if [[ ! -x "$LMP" ]]; then
    echo "[ERROR] LAMMPS executable not found or not executable:"
    echo "        $LMP"
    exit 1
fi

if [[ -z "$MPI" ]]; then
    echo "[ERROR] NEB_MPIRUN is not set."
    exit 1
fi

if [[ ! -x "$MPI" ]]; then
    echo "[ERROR] MPI launcher not found or not executable:"
    echo "        $MPI"
    exit 1
fi

LMP_HELP="$("$LMP" -h 2>&1)"
MPI_VERSION="$("$MPI" --version 2>&1 | head -20)"

echo "LAMMPS executable:"
readlink -f "$LMP"
echo

echo "MPI launcher:"
readlink -f "$MPI"
echo

# ----------------------------------------------------------------------
# LAMMPS version
# ----------------------------------------------------------------------

if echo "$LMP_HELP" | grep -q "29 Aug 2024 - Update 1"; then
    echo "[OK] LAMMPS 29 Aug 2024 - Update 1"
else
    echo "[ERROR] Expected LAMMPS 29 Aug 2024 - Update 1."
    echo "        Detected:"
    echo "$LMP_HELP" | grep -m1 "Large-scale Atomic"
    errors=$((errors + 1))
fi

# ----------------------------------------------------------------------
# Required packages
# ----------------------------------------------------------------------

for pkg in REAXFF REPLICA; do
    if echo "$LMP_HELP" | grep -qE "(^|[[:space:]])${pkg}([[:space:]]|$)"; then
        echo "[OK] LAMMPS package: $pkg"
    else
        echo "[ERROR] Required LAMMPS package missing: $pkg"
        errors=$((errors + 1))
    fi
done

# The validated binary also included these packages.
for pkg in MANYBODY MOLECULE; do
    if echo "$LMP_HELP" | grep -qE "(^|[[:space:]])${pkg}([[:space:]]|$)"; then
        echo "[OK] LAMMPS package: $pkg"
    else
        echo "[WARNING] Validated build included $pkg, but this build does not."
    fi
done

# ----------------------------------------------------------------------
# MPI implementation compatibility
# ----------------------------------------------------------------------

LMP_MPI="$(echo "$LMP_HELP" | grep -m1 '^MPI ' || true)"

echo
echo "LAMMPS MPI:"
echo "  ${LMP_MPI:-unknown}"

echo "Launcher:"
echo "$MPI_VERSION" | head -3 | sed 's/^/  /'

if echo "$LMP_MPI" | grep -qi "Open MPI"; then
    if echo "$MPI_VERSION" | grep -qi "Open MPI"; then
        echo "[OK] MPI family matches: Open MPI"
    else
        echo "[ERROR] LAMMPS was built with Open MPI, but NEB_MPIRUN is not Open MPI."
        errors=$((errors + 1))
    fi
elif echo "$LMP_MPI" | grep -Eqi "MPICH|HYDRA"; then
    if echo "$MPI_VERSION" | grep -Eqi "MPICH|HYDRA"; then
        echo "[OK] MPI family matches: MPICH"
    else
        echo "[ERROR] LAMMPS was built with MPICH, but NEB_MPIRUN is not MPICH."
        errors=$((errors + 1))
    fi
else
    echo "[WARNING] Could not determine the MPI implementation used by LAMMPS."
fi

echo

if [[ "$errors" -ne 0 ]]; then
    echo "LAMMPS/MPI validation FAILED with $errors error(s)."
    echo
    echo "Do not run REALIZE until these errors are corrected."
    exit 1
fi

echo "LAMMPS/MPI configuration matches the validated requirements."
