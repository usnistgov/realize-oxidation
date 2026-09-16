"""Load and resolve the user-facing input.yaml into a validated PPOConfig.

Resolution order (each layer overlaid on the previous):
  1. create_default_config() -- the authoritative dataclass defaults.
  2. config/default_inputs.yaml -- the full default catalog. Overlaid, but each value
     is first checked to equal the current dataclass default, so catalog drift aborts
     at startup. The overlay is a numerically-neutral no-op when in sync.
  3. The user input.yaml at `path` -- only the keys the user set (validated by
     config.schema).
  4. Seed resolution (see _resolve_seed): every run gets a DIFFERENT seed unless one
     is pinned explicitly. meta.seed pins an exact value (reproducible across runs).
     Otherwise seed = meta.seed_base + a SLURM-derived offset -- SLURM_ARRAY_TASK_ID
     for array jobs (reproducible per index; also the path the verification oracles
     force via the env var), else SLURM_JOB_ID for a plain `sbatch` (unique per
     submission, so every run differs without --array), else fresh OS entropy for a
     bare local run. The resolved seed is logged by train.py, so any run can be
     reproduced later by setting meta.seed to it.
  5. PPOConfig.validate() for range / cross-field checks.

The returned device string is kept out of the config object and passed straight to
the trainer by train.py.
"""
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

from realize.config.default_config import PPOConfig, create_default_config
from realize.config.schema import validate_input, META_SECTION

# Full default catalog, shipped next to this loader.
DEFAULTS_FILENAME = "default_inputs.yaml"

# Keep resolved seeds inside numpy's accepted range [0, 2**32 - 1] with headroom for
# the per-worker offsets parallel_collector.py adds on top (base + worker_id * 1000).
SEED_MODULUS = 2 ** 31


@dataclass
class LoadedConfig:
    config: PPOConfig
    device: Optional[str]
    seed: int
    seed_base: int
    task_id: int          # SLURM offset folded into the seed (0 for explicit/local).
    seed_source: str      # how `seed` was derived: explicit|array_task|job_id|random.


def _read_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load input.yaml. Install with `pip install pyyaml`."
        ) from exc
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _defaults_path() -> str:
    """Absolute path to the default catalog beside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULTS_FILENAME)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare a catalog value against a dataclass default (float-tolerant)."""
    # Treat bools strictly (bool is an int subclass in Python).
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _apply_and_verify_defaults(config: PPOConfig) -> None:
    """Overlay config/default_inputs.yaml onto `config` and assert it matches the
    code defaults. Raises on drift. The overlay is a no-op when in sync.
    """
    path = _defaults_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing default catalog: {path}\n"
            f"It must exist and mirror create_default_config() "
            f"(config/default_config.py). Restore it from version control."
        )

    raw = _read_yaml(path)

    # The catalog holds dataclass sections only; device/seed_base are run-level
    # (user input.yaml meta section).
    if META_SECTION in raw:
        raise ValueError(
            f"{DEFAULTS_FILENAME} must not contain a '{META_SECTION}' section; "
            f"device/seed_base are run-level and live in the user input.yaml."
        )

    # Same structural validation the user file gets.
    validate_input(raw)

    mismatches = []
    for section_name, block in raw.items():
        if not block:
            continue
        section = getattr(config, section_name)
        for key, value in block.items():
            current = getattr(section, key)
            if not _values_equal(current, value):
                mismatches.append(
                    f"  {section_name}.{key}: code default={current!r} "
                    f"but {DEFAULTS_FILENAME}={value!r}"
                )
            setattr(section, key, value)

    if mismatches:
        raise ValueError(
            f"{DEFAULTS_FILENAME} has drifted from create_default_config() "
            f"(config/default_config.py).\nThis file MUST mirror the code defaults "
            f"exactly so reproduction is preserved. Divergent keys:\n"
            + "\n".join(mismatches)
            + f"\n\nFix {DEFAULTS_FILENAME} to match the code default, "
            f"or change the code default and update this file to match."
        )


def _slurm_task_id() -> Optional[int]:
    """SLURM_ARRAY_TASK_ID as an int, or None when this is not an array task.

    Returning None (rather than 0) lets the seed resolver tell "array task 0" apart
    from "no array at all", so a plain sbatch can fall through to the job-id offset.
    """
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be an integer, got {raw!r}")


def _slurm_job_id() -> Optional[int]:
    """SLURM_JOB_ID as an int, or None outside a SLURM allocation.

    Slurm assigns a fresh job id to every submission -- including a plain, non-array
    `sbatch submit.sh` -- so folding it into the seed makes each run diverge without
    the user having to remember `--array`.
    """
    raw = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"SLURM_JOB_ID must be an integer, got {raw!r}")


def _resolve_seed(meta: dict, seed_base: int):
    """Resolve the run seed. Returns (seed, offset, source).

    Default policy: every run gets a different seed. Precedence:
      1. meta.seed present  -> used verbatim. The explicit-pin escape hatch for
         reproducibility (the only way to force the SAME seed across runs).
      2. SLURM_ARRAY_TASK_ID -> seed_base + task_id. Reproducible per array index;
         this is the path the verification oracles force via the env var, so their
         seed stays exactly seed_base (offset 0) and reproduction is untouched.
      3. SLURM_JOB_ID        -> seed_base + job_id. Unique per submission, so a plain
         `sbatch submit.sh` differs every run.
      4. neither             -> seed_base + fresh OS entropy, so bare local runs
         differ too.
    All paths are folded mod SEED_MODULUS to stay in numpy's seed range; for the small
    array offsets this is a no-op, preserving byte-identical reproduction.
    """
    explicit = meta.get("seed")
    if explicit is not None:
        return int(explicit) % SEED_MODULUS, 0, "explicit"

    task_id = _slurm_task_id()
    if task_id is not None:
        return (int(seed_base) + task_id) % SEED_MODULUS, task_id, "array_task"

    job_id = _slurm_job_id()
    if job_id is not None:
        return (int(seed_base) + job_id) % SEED_MODULUS, job_id, "job_id"

    entropy = int.from_bytes(os.urandom(4), "little")
    return (int(seed_base) + entropy) % SEED_MODULUS, entropy, "random"


def load_config(path: str) -> LoadedConfig:
    """Parse input.yaml at `path`, validate it, and build a resolved PPOConfig."""
    raw = _read_yaml(path)
    validate_input(raw)

    meta = dict(raw.pop(META_SECTION, {}) or {})
    device = meta.get("device")

    # Layer 1: authoritative defaults. Differs from raw PPOConfig() in the debug/eval
    # flags (log_gradient_norms, enable_equivariance_tests, sampling.debug_equivariance,
    # sampling.debug_frame_encoder forced False; use_cache True) -- a raw PPOConfig()
    # would flip these on.
    config = create_default_config()

    # Layer 2: overlay + verify the default catalog (no-op when in sync; aborts on drift).
    _apply_and_verify_defaults(config)

    seed_base = meta.get("seed_base", config.runtime.seed)

    # Layer 3: user overrides.
    for section_name, block in raw.items():
        if not block:
            continue
        section = getattr(config, section_name)
        for key, value in block.items():
            setattr(section, key, value)

    seed, offset, source = _resolve_seed(meta, seed_base)
    config.runtime.seed = seed

    config.validate()

    return LoadedConfig(
        config=config,
        device=device,
        seed=seed,
        seed_base=int(seed_base),
        task_id=offset,
        seed_source=source,
    )
