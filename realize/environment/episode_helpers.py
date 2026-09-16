"""Leaf helpers shared by episode_reset and episode_step:
    append_snapshot_to_movie, _log, validate_atom_counts, handle_md_verification.
"""
import os
import gc
import torch
import numpy as np
import time
from ase import io
import shutil
from realize.networks.encoders.embedding_extractor import get_embedding_vector
from realize.utils.file_utils import standardize_file_format, write_structure_file
from realize.config.paths import (
    get_structure_file_path, get_o2_file_path, get_embedding_checkpoint_path,
    get_movie_file_path, get_results_dir, get_neb_results_dir, get_temp_dir,
    get_neb_input_file_path
)
from realize.environment.neb_calculator import NEBCalculator
from typing import Optional, Tuple
from realize.utils.logger import UnifiedLogger
import logging
_LOG = logging.getLogger(__name__)


def append_snapshot_to_movie(snapshot_file, movie_file, logger=None):
    """Append a snapshot to the movie file; raises on any error (no fallbacks)."""
    if not os.path.exists(snapshot_file):
        raise FileNotFoundError(f"Snapshot file does not exist: {snapshot_file}")

    with open(snapshot_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith("ITEM: TIMESTEP"):
            raise ValueError(
                f"Invalid snapshot header. "
                f"Expected 'ITEM: TIMESTEP', got: {first_line[:50]}"
            )

    with open(snapshot_file, 'rb') as src:
        with open(movie_file, 'ab') as dst:
            dst.write(src.read())

    return True

def _log(env, level, message):
    """Route a message to the env logger by level, falling back to module logger."""
    if hasattr(env, 'logger') and env.logger:
        if level == 'info':
            env.logger.log_info(message)
        elif level == 'warning':
            env.logger.log_warning(message)
        elif level == 'error':
            env.logger.log_error(message)
        elif level == 'debug':
            env.logger.log_debug(message)
    else:
        prefix = f"[{level.upper()}] " if level != 'info' else ""
        _LOG.info(f"{prefix}{message}")



def validate_atom_counts(env, stage=""):
    """False on any Si or total atom-count mismatch vs. expected substrate + O2."""
    try:
        atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
        atoms.set_pbc((True, True, True))
        atoms.wrap()

        symbols = atoms.get_chemical_symbols()

        si_count = sum(1 for sym in symbols if sym == 'Si')
        o_count = sum(1 for sym in symbols if sym == 'O')
        total_count = len(atoms)

        expected_si = env.original_si_count
        expected_o_substrate = env.original_o_count
        expected_o_dissociated = getattr(env, 'cumulative_dissociated_o_count', 0)

        # O2 present iff env owns one. Geometric re-detection (d_OO < 1.8 A)
        # is too narrow: a mid-flight O2 stretched to 1.8-2.1 A is neither
        # intact-by-bond-length nor yet verified-dissociated, and would be
        # invisible to a geometric check. The env knows what it owns.
        if not hasattr(env, 'active_o2'):
            raise AttributeError(
                "env.active_o2 not initialized. O2LammpsEnv.__init__ must set "
                "it; every add/remove site must flip it. No silent default."
            )
        has_o2 = bool(env.active_o2)

        expected_total = expected_si + expected_o_substrate + expected_o_dissociated
        if has_o2:
            expected_total += 2

        valid = True

        # Si mismatch is critical (substrate corruption)
        if si_count != expected_si:
            _log(env, 'error', f"{stage}: Si count mismatch! Expected {expected_si}, got {si_count}")
            valid = False

        if total_count != expected_total:
            _log(env, 'error', f"{stage}: Atom count mismatch!")
            _log(env, 'error', f"  Expected: {expected_total}, Got: {total_count}, Diff: {total_count - expected_total}")
            _log(env, 'error', f"  Details: Si={si_count}/{expected_si}, O={o_count}")
            _log(env, 'error', f"  Original substrate: Si={expected_si}, O={expected_o_substrate}")
            _log(env, 'error', f"  Dissociated O: {expected_o_dissociated}, Has O2: {has_o2}")
            return False

        return valid
        
    except Exception as e:
        _log(env, 'error', f"Error in validate_atom_counts: {e}")
        import traceback
        traceback.print_exc()
        return False    

def handle_md_verification(env, episode, movie_file, device, checkpoint_path=None, embedding_extractor=None, logger=None):
    """
    Run MD verification of O2 dissociation and return (md_verified, distance, embedding).

    checkpoint_path is required for embedding extraction. Raises ValueError if device
    is None, NotImplementedError if the env lacks run_dissociation_verification_md.
    """
    if device is None:
        raise ValueError(
            "device parameter is REQUIRED for handle_md_verification(). "
            "Must be torch.device('cuda') or torch.device('cpu')."
        )
    if logger:
        logger.log_debug("Starting MD verification for dissociation")

    # Drives movie file naming
    env.current_episode = episode

    if not hasattr(env, 'run_dissociation_verification_md'):
        raise NotImplementedError(
            "run_dissociation_verification_md is required but not available. "
            "Cannot verify dissociation."
        )

    md_verified = env.run_dissociation_verification_md()

    if md_verified:
        if logger:
            logger.log_debug("MD verification confirmed: O2 remains dissociated")
        env.last_md_verified_dissociation = True
        env.previous_episode_dissociated = True
    else:
        if logger:
            logger.log_debug("MD verification failed: O2 recombined. Continuing episode.")
        env.last_md_verified_dissociation = False
        env.previous_episode_dissociated = False

    # State after MD (restored if recombined)
    distance, _ = env.check_o2_distance()
    # Skip embedding when dissociated or O-O distance is out of valid range
    if md_verified or distance > 2.5:
        embedding = None
    elif embedding_extractor is not None:
        embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
            io.read(env.coo_file, format="lammps-data", atom_style="charge"),
            return_symmetric_only=False
        )
        embedding = embeddings_new_dict['symmetric']
    else:
        embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
        embedding = embeddings_new_dict['symmetric']  # symmetric is used for state construction

    return md_verified, distance, embedding


