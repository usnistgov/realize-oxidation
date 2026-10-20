#!/usr/bin/env python3
"""Checks installed package versions against the validated environment."""

import sys
from importlib.metadata import PackageNotFoundError, version

EXPECTED = {
    "numpy": "1.26.4",
    "e3nn": "0.5.6",
    "sevenn": "0.10.3",
    "ase": "3.24.0",
    "torch-geometric": "2.6.1",
    "PyYAML": "6.0.2",
    "matplotlib": "3.9.4",
    "scikit-learn": "1.6.0",
    "tqdm": "4.67.1",
    "scipy": "1.13.1",
    "matscipy": "1.1.1",
}

EXPECTED_TORCH = "2.6.0+cu118"

errors = []

print("REALIZE-oxidation environment validation")
print("=" * 55)

python_version = (
    sys.version_info.major,
    sys.version_info.minor,
    sys.version_info.micro,
)

if python_version[:2] != (3, 9):
    errors.append(
        f"Python must be 3.9.x; found "
        f"{python_version[0]}.{python_version[1]}.{python_version[2]}"
    )
else:
    print(
        f"[OK] Python "
        f"{python_version[0]}.{python_version[1]}.{python_version[2]}"
    )

for package, expected in EXPECTED.items():
    try:
        installed = version(package)
    except PackageNotFoundError:
        errors.append(f"{package}: not installed")
        continue

    if installed != expected:
        errors.append(
            f"{package}: expected {expected}, found {installed}"
        )
    else:
        print(f"[OK] {package} {installed}")

try:
    import torch

    installed = torch.__version__

    if installed != EXPECTED_TORCH:
        errors.append(
            f"torch: expected validated build "
            f"{EXPECTED_TORCH}, found {installed}"
        )
    else:
        print(f"[OK] torch {installed}")

    print(f"     torch CUDA version: {torch.version.cuda}")
    print(f"     CUDA available: {torch.cuda.is_available()}")

except Exception as exc:
    errors.append(f"torch import failed: {exc}")

print()

if errors:
    print("ERROR: environment does not match the validated REALIZE setup.")
    print()

    for error in errors:
        print(f"  - {error}")

    print()
    print(
        "REALIZE-oxidation is sensitive to dependency versions and has "
        "exhibited silent behavioral failures under incompatible versions."
    )
    print(
        "Do not use this environment for production or manuscript "
        "reproduction runs."
    )

    sys.exit(1)

print("Environment matches the validated REALIZE-oxidation setup.")
