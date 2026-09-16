"""REALIZE training entry point.

PPO training of the equivariant O2-dissociation policy. Config is loaded from
input.yaml (validated against config/schema.py) and overlaid on the defaults in
config/default_config.py. Per-task seed = meta.seed_base + SLURM_ARRAY_TASK_ID.

The only import-time side effect is putting the repo root on sys.path (so
`python realize/train.py` resolves `import realize.*` like `python -m realize.train`);
faulthandler, anomaly mode, directory creation, seeding and chdir all happen in main().

    python realize/train.py                      # uses ../input.yaml (repo root)
    python realize/train.py --input my.yaml      # alternate config
    python -m realize.train                      # equivalent module form
"""
import os
import sys

# Put the repo root on sys.path so `import realize.*` resolves whether launched as a
# script (`python realize/train.py`) or as a module (`-m realize.train`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import time
import random

import numpy as np
import torch

from realize.config.input_loader import load_config
from realize.config.paths import (
    PathManager,
    get_input_dir,
    get_neb_executable_path,
    get_results_dir,
    get_temp_dir,
)
from realize.utils.logger import create_logger, print_banner
from realize.utils.log_setup import configure_logging
from realize.training.trainer import PPOTrainer
from realize.training.critic_offline_test import run_offline_critic_test
import logging

# Named explicitly (this runs as "__main__") so the logger stays under the "realize"
# hierarchy that configure_logging() attaches the stdout handler to.
_LOG = logging.getLogger("realize.train")

# input.yaml lives at the repo root, one level above realize/.
_DEFAULT_INPUT = os.path.join(_REPO_ROOT, "input.yaml")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for O2 dissociation (REALIZE)")
    parser.add_argument("--input", type=str, default=_DEFAULT_INPUT,
                        help="Path to input.yaml (default: ./input.yaml)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override meta.device from input.yaml (cuda/cpu)")

    # Offline critic sanity test.
    parser.add_argument("--offline-critic-test", action="store_true",
                        help="Run offline critic sanity test before training")
    parser.add_argument("--offline-test-episodes", type=int, default=100,
                        help="Number of episodes to collect for offline critic test")
    parser.add_argument("--offline-test-epochs", type=int, default=20,
                        help="Number of training epochs for offline critic test")
    return parser.parse_args()


def validate_environment(config):
    """Validate that required input files exist based on configuration."""
    input_dir = get_input_dir()
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist!")

    required_files = ["str.dat", "O2.dat"]

    import shutil
    neb_path = get_neb_executable_path()
    neb_available = os.path.exists(neb_path) or shutil.which(neb_path) is not None
    if not neb_available:
        use_neb = bool(getattr(getattr(config, "reward", None), "use_neb_reward", False))
        if use_neb:
            raise FileNotFoundError(
                f"NEB executable required but not found: {neb_path}\n"
                f"Checked: file existence and PATH"
            )
        _LOG.info(f"NEB executable not found at {neb_path} (not required)")
    else:
        _LOG.info(f"Found NEB executable: {neb_path}")

    for filename in required_files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
        _LOG.info(f"Found: {filename}")

    _LOG.info("All required files validated")


def set_random_seeds(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            _LOG.info(f"Random seed set to {seed} (deterministic mode)")
        else:
            _LOG.info(f"Random seed set to {seed}")
    else:
        _LOG.info(f"Random seed set to {seed}")


def resolve_device(requested):
    """Resolve a device string, falling back to CPU when CUDA is unavailable."""
    if requested and "cuda" in requested.lower():
        if torch.cuda.is_available():
            device = torch.device(requested)
            _LOG.info(f"Using requested device: {device}")
        else:
            device = torch.device("cpu")
            _LOG.info("CUDA requested but not available, using CPU")
    elif requested:
        device = torch.device(requested)
        _LOG.info(f"Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LOG.info(f"Auto-selected device: {device}")
    return device


def main():
    # Route every realize.* logger to stdout with print-identical formatting.
    # Must run before print_banner() or any other logging in this process.
    configure_logging()

    import faulthandler
    faulthandler.enable()
    torch.autograd.set_detect_anomaly(False)  # saves ~50% memory

    start_time = time.time()
    args = parse_arguments()

    print_banner()
    _LOG.info("=" * 60)
    _LOG.info("PPO TRAINING SETUP")
    _LOG.info("=" * 60)

    # Per-task directory tree (scratch isolated per SLURM array task).
    paths = PathManager.from_env().setup()

    # Load and resolve configuration (input.yaml -> validated PPOConfig).
    loaded = load_config(args.input)
    config = loaded.config
    # Switch the stdout level to DEBUG iff verbose (bootstrap call above ran at INFO).
    configure_logging(verbose=config.runtime.verbose)
    _LOG.info(f"Loaded config from {args.input}")
    if loaded.seed_source == "explicit":
        _LOG.info(f"Seed: {loaded.seed} (explicit meta.seed pin -> same every run)")
    elif loaded.seed_source == "array_task":
        _LOG.info(f"Seed: {loaded.seed} (base {loaded.seed_base} + SLURM array task {loaded.task_id})")
    elif loaded.seed_source == "job_id":
        _LOG.info(f"Seed: {loaded.seed} (base {loaded.seed_base} + SLURM job id {loaded.task_id}); "
                  f"pin 'meta.seed: {loaded.seed}' to reproduce this run")
    else:
        _LOG.info(f"Seed: {loaded.seed} (base {loaded.seed_base} + OS entropy); "
                  f"pin 'meta.seed: {loaded.seed}' to reproduce this run")

    validate_environment(config)

    set_random_seeds(config.runtime.seed)

    device_request = args.device if args.device is not None else loaded.device
    device = resolve_device(device_request)

    verbosity = 'DEBUG' if config.runtime.verbose else 'INFO'
    logger = create_logger(get_results_dir(), verbosity=verbosity)
    logger.log_info(f"Results directory: {get_results_dir()}")
    logger.log_info(f"Temp directory: {get_temp_dir()}")
    # work_dir only differs from temp_dir for SLURM per-task scratch; avoid the
    # duplicate line for a single run where the two paths are identical.
    if os.path.abspath(paths.work_dir) != os.path.abspath(get_temp_dir()):
        logger.log_info(f"Work directory: {paths.work_dir}")

    original_dir = os.getcwd()
    work_dir = paths.work_dir
    try:
        logger.log_info(f"Changing to work directory: {work_dir}")
        os.chdir(work_dir)

        trainer = PPOTrainer(config, device, logger)

        if getattr(config.runtime, "resume_path", None):
            logger.log_info(f"Attempting to resume from: {config.runtime.resume_path}")
        elif getattr(trainer, "start_episode", 0) > 0:
            logger.log_info(f"Resumed training from episode {trainer.start_episode}")

        if args.offline_critic_test:
            _LOG.info("\n" + "=" * 70)
            _LOG.info("RUNNING OFFLINE CRITIC SANITY TEST")
            _LOG.info("=" * 70)
            results = run_offline_critic_test(
                trainer,
                num_episodes=args.offline_test_episodes,
                num_epochs=args.offline_test_epochs,
                verbose=True,
            )
            verdict = results["verdict"]
            stats = results["dataset_stats"]
            metrics = results["training_metrics"]
            best_val_ev = metrics.get("best_val_ev", metrics["val_ev"][-1])
            best_val_corr = metrics.get("best_val_corr", metrics["val_correlation"][-1])
            best_epoch = metrics.get("best_epoch", len(metrics["val_ev"]) - 1)
            _LOG.info(f"\n{'=' * 70}")
            _LOG.info(f"OFFLINE TEST VERDICT: {verdict}")
            _LOG.info(f"{'=' * 70}")
            _LOG.info(f"  Dataset: {stats['n_transitions']} transitions "
                  f"({stats['n_train']} train, {stats['n_val']} val)")
            _LOG.info(f"  Returns: mean={stats['returns_mean']:.3f}, "
                  f"std={stats['returns_std']:.3f}")
            _LOG.info(f"  Best val_EV={best_val_ev:.4f} (epoch {best_epoch + 1}), "
                  f"best val_corr={best_val_corr:.4f}")
            _LOG.info(f"  Final val_EV={metrics['val_ev'][-1]:.4f}, "
                  f"final val_corr={metrics['val_correlation'][-1]:.4f}")
            logger.log_info(f"Offline test verdict: {verdict}")

        trainer.train()

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        logger.log_info("Training completed successfully")
        logger.log_info(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    except KeyboardInterrupt:
        logger.log_warning("Training interrupted by user")
        elapsed = time.time() - start_time
        logger.log_info(f"Partial training time: {elapsed / 3600:.2f} hours")

    except Exception as e:
        logger.log_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        os.chdir(original_dir)
        logger.log_info(f"Returned to: {original_dir}")


if __name__ == "__main__":
    main()
