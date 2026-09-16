"""Episode reset: clean the substrate, add a fresh O2, return initial embeddings."""
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
from realize.environment.episode_helpers import (
    _log,
    append_snapshot_to_movie,
    validate_atom_counts,
)
import logging
_LOG = logging.getLogger(__name__)


def reset_episode(env, episode_id, checkpoint_path=None, device=None, embedding_extractor=None):
    """
    Reset the environment for a new episode; returns (distance, embeddings_dict).

    Raises ValueError if device is None.
    """
    if device is None:
        raise ValueError(
            "device parameter is REQUIRED for reset_episode(). "
            "Must be torch.device('cuda') or torch.device('cpu'). "
            "No auto-selection allowed."
        )

    if checkpoint_path is None:
        checkpoint_path = get_embedding_checkpoint_path()

    env.global_catastrophic_failures = 0
    env.force_next_step = False
    if hasattr(env, 'neb_failure_tracker'):
        env.neb_failure_tracker.clear()

    _log(env, 'debug', f"[Ep{episode_id}] Reset catastrophic failure counters")

    dist = None
    embedding = None
    movie_file = get_movie_file_path(episode_id + 1)

    # Record original substrate counts once for integrity checks
    if not hasattr(env, 'original_atom_count'):
        try:
            orig_structure_file = get_structure_file_path()
            orig_atoms = io.read(orig_structure_file, format="lammps-data", atom_style="charge")
            env.original_atom_count = len(orig_atoms)

            orig_symbols = orig_atoms.get_chemical_symbols()
            env.original_si_count = sum(1 for sym in orig_symbols if sym == 'Si')
            env.original_o_count = sum(1 for sym in orig_symbols if sym == 'O')

            _log(env, 'info', f"Original substrate has {env.original_atom_count} atoms (Si: {env.original_si_count}, O: {env.original_o_count})")
                

        except Exception as e:
            _log(env, 'error', f"Error determining original atom count: {e}")
            raise ValueError("Cannot determine substrate atom count - cannot continue")

    if not hasattr(env, 'cumulative_dissociated_o_count'):
        env.cumulative_dissociated_o_count = 0
        _log(env, 'info', f"Initialized cumulative_dissociated_o_count to 0")

    # Keep structure_manager counts in sync
    if hasattr(env, 'structure_manager'):
        env.structure_manager.cumulative_dissociated_o_count = env.cumulative_dissociated_o_count
        env.structure_manager.original_atom_count = env.original_atom_count
        env.structure_manager.original_si_count = env.original_si_count
        env.structure_manager.original_o_count = env.original_o_count

    # Capture episode state flags before clearing them
    o2_was_removed = hasattr(env, 'o2_removed') and env.o2_removed

    # Reset action history to avoid cross-episode contamination
    if hasattr(env, 'action_processor'):
        env.action_processor.reset_history()
        _log(env, 'debug', "Reset action processor history for new episode")

    # Previous-episode dissociation state, by source preference
    previous_dissociated = False

    if episode_id > 0:
        if hasattr(env, 'episode_outcomes') and (episode_id - 1) in env.episode_outcomes:
            previous_dissociated = env.episode_outcomes[episode_id - 1].get('dissociated', False)
            _log(env, 'debug', f"Got dissociation state from episode_outcomes: {previous_dissociated}")
        elif hasattr(env, 'previous_episode_dissociated'):
            previous_dissociated = env.previous_episode_dissociated
            _log(env, 'debug', f"Got dissociation state from flag: {previous_dissociated}")
        elif hasattr(env, 'last_md_verified_dissociation'):
            previous_dissociated = env.last_md_verified_dissociation
            _log(env, 'debug', f"Got dissociation state from MD result: {previous_dissociated}")

    _log(env, 'info', f"[Ep{episode_id}] Previous episode dissociation state: {previous_dissociated}")


    # Episode 0 must start with clean substrate
    if episode_id == 0:
        env.cumulative_dissociated_o_count = 0
        if env.structure_manager.cumulative_dissociated_o_count != 0:
            raise ValueError(
                f"Episode 0 started with non-zero dissociated count "
                f"({env.structure_manager.cumulative_dissociated_o_count}). "
                f"Structure must be clean. Verify input structure file."
            )

        shutil.copy2(get_structure_file_path(), env.coo_file)
        standardize_file_format(env.coo_file, target_style="charge")
        env.o2_removed = False
    else:
        # Subsequent episodes: handle based on previous episode outcome
        try:
            atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
        except Exception as e:
            _log(env, 'error', f"Failed to read structure file: {e}")
            raise RuntimeError(f"[Ep{episode_id}] FATAL: Cannot read structure file. Episode cannot continue.") from e

        before_count = len(atoms)
        symbols = atoms.get_chemical_symbols()
        current_si_count = sum(1 for sym in symbols if sym == 'Si')
        current_o_count = sum(1 for sym in symbols if sym == 'O')
        
        _log(env, 'info', f"\n[Ep{episode_id}] Starting with {before_count} atoms")
        _log(env, 'info', f"  Si: {current_si_count}, O: {current_o_count}")
        _log(env, 'info', f"  Previous dissociated: {previous_dissociated}, O2 was removed: {o2_was_removed}")
        _log(env, 'info', f"  Cumulative dissociated O: {env.cumulative_dissociated_o_count}")

        extra_o_atoms = current_o_count - env.original_o_count

        # cumulative_dissociated_o_count is updated at the end of the prior episode;
        # never increment it here.
        if previous_dissociated:
            _LOG.debug(f"[Ep{episode_id}] Previous episode had MD-verified dissociation")
            _LOG.debug(f"  Cumulative dissociated O: {env.cumulative_dissociated_o_count} (already updated)")
            _LOG.debug(f"  Structure should have {env.original_atom_count + env.cumulative_dissociated_o_count} atoms")
        else:
            _LOG.debug(f"[Ep{episode_id}] No MD-verified dissociation in previous episode")
            _LOG.debug(f"  Will remove any O2 beyond the {env.cumulative_dissociated_o_count} previously dissociated O atoms")

        expected_o_substrate = env.original_o_count + env.cumulative_dissociated_o_count
        expected_total = env.original_atom_count + env.cumulative_dissociated_o_count

        _LOG.debug(f"  Expected substrate: {expected_total} atoms (O: {expected_o_substrate})")

        if before_count < expected_total and not o2_was_removed:
            raise RuntimeError(
                f"[Ep{episode_id}] FATAL: Missing atoms. "
                f"Have {before_count}, expected {expected_total}. "
                f"Missing: {expected_total - before_count} atoms"
            )

        if current_si_count != env.original_si_count:
            raise RuntimeError(
                f"[Ep{episode_id}] FATAL: Si atom count corrupted! "
                f"Expected {env.original_si_count}, got {current_si_count}. "
                f"Substrate integrity compromised."
            )

        if previous_dissociated:
            # Keep dissociated O atoms
            _LOG.debug(f"[Ep{episode_id}] Previous episode had dissociation - keeping dissociated O atoms")

            # Remove O that escaped to vacuum during MD: after 1000K MD a dissociated O
            # can fly above the box z top, and boundary p p f makes read_data fail for
            # atoms outside the z bounds.
            positions = atoms.get_positions()
            z_coords = positions[:, 2]
            cell_z = atoms.cell[2, 2]
            symbols = atoms.get_chemical_symbols()

            vacuum_o_indices = []
            for i in range(len(atoms)):
                if symbols[i] == 'O' and z_coords[i] > cell_z - 0.5:
                    vacuum_o_indices.append(i)

            if vacuum_o_indices:
                _LOG.warning(f"[Ep{episode_id}] Found {len(vacuum_o_indices)} O atom(s) near/above z boundary "
                      f"(z > {cell_z - 0.5:.1f}). Removing to prevent LAMMPS read_data failure.")
                for idx in vacuum_o_indices:
                    _LOG.debug(f"  Atom {idx}: z={z_coords[idx]:.3f}, cell_z={cell_z:.3f}")
                del atoms[vacuum_o_indices]
                # These O are lost, not integrated: drop them from the cumulative count
                n_removed = len(vacuum_o_indices)
                env.cumulative_dissociated_o_count -= n_removed
                if env.cumulative_dissociated_o_count < 0:
                    env.cumulative_dissociated_o_count = 0
                if hasattr(env, 'structure_manager'):
                    env.structure_manager.cumulative_dissociated_o_count = env.cumulative_dissociated_o_count
                _LOG.info(f"  Adjusted cumulative_dissociated_o_count to {env.cumulative_dissociated_o_count}")

            write_structure_file(atoms, env.coo_file, atom_style="charge")
            standardize_file_format(env.coo_file, target_style="charge")

        elif o2_was_removed and not previous_dissociated:
            # O2 flew away (removed by check_surrounding_o2); structure already correct
            _LOG.debug(f"[Ep{episode_id}] O2 was removed in previous episode (flew away)")

            write_structure_file(atoms, env.coo_file, atom_style="charge")
            standardize_file_format(env.coo_file, target_style="charge")

        else:
            # No dissociation and O2 stayed: remove the leftover O2 if present
            _LOG.debug(f"[Ep{episode_id}] No dissociation in previous episode - will remove O2 if present")

            expected_atoms = env.original_atom_count + env.cumulative_dissociated_o_count
            expected_o = env.original_o_count + env.cumulative_dissociated_o_count

            extra_atoms = before_count - expected_atoms
            extra_o = current_o_count - expected_o

            if extra_atoms == 2 and extra_o == 2:
                # Stray O2 molecule (the last two O atoms)
                _LOG.debug(f"[Ep{episode_id}] Detected stray O2 molecule ({extra_atoms} extra atoms)")

                o_indices = [i for i, sym in enumerate(symbols) if sym == 'O']

                if len(o_indices) >= 2:
                    o1_idx = o_indices[-2]
                    o2_idx = o_indices[-1]
                    o_distance = atoms.get_distance(o1_idx, o2_idx, mic=True)

                    if o_distance < 1.8:  # within O2 bond cutoff
                        _LOG.debug(f"[Ep{episode_id}] Removing O2 pair at indices {o1_idx}, {o2_idx} (distance={o_distance:.3f}Angstrom)")
                        del atoms[[o2_idx, o1_idx]]
                        write_structure_file(atoms, env.coo_file, atom_style="charge")
                        standardize_file_format(env.coo_file, target_style="charge")
                        env.o2_removed = True
                        _LOG.debug(f"[Ep{episode_id}] O2 removed successfully. New atom count: {len(atoms)}")
                    elif o_distance < 3.5:
                        # Past bond cutoff but still the stray O2 pair (e.g. failed MD
                        # leaves it at 2-3 Angstrom), not integrated into the oxide
                        _LOG.warning(f"[Ep{episode_id}] Stray O2 at {o_distance:.3f}Angstrom > 1.8Angstrom bond cutoff. "
                              f"Likely from failed MD verification. Removing anyway.")
                        del atoms[[o2_idx, o1_idx]]
                        write_structure_file(atoms, env.coo_file, atom_style="charge")
                        standardize_file_format(env.coo_file, target_style="charge")
                        env.o2_removed = True
                        _LOG.debug(f"[Ep{episode_id}] Stray O2 removed. New atom count: {len(atoms)}")
                    else:
                        # O atoms > 3.5 Angstrom apart: one may be substrate O detached
                        # during MD; remove both to restore a clean substrate
                        _LOG.warning(f"[Ep{episode_id}] 2 extra O at d={o_distance:.3f}Angstrom > 3.5Angstrom. "
                              f"Likely substrate O detached during MD. Removing both.")
                        del atoms[[o2_idx, o1_idx]]
                        write_structure_file(atoms, env.coo_file, atom_style='charge')
                        standardize_file_format(env.coo_file, target_style='charge')
                        env.o2_removed = True
                        _LOG.debug(f"[Ep{episode_id}] Extra O atoms removed. New count: {len(atoms)}")
                else:
                    raise RuntimeError(
                        f"[Ep{episode_id}] FATAL: Expected 2 extra O atoms but found {len(o_indices)} total O atoms."
                    )

            elif extra_atoms == 0:
                _LOG.debug(f"[Ep{episode_id}] Structure already correct - no O2 to remove")
                write_structure_file(atoms, env.coo_file, atom_style="charge")
                standardize_file_format(env.coo_file, target_style="charge")

            elif extra_atoms > 0:
                raise RuntimeError(
                    f"[Ep{episode_id}] FATAL: Unexpected extra atoms! "
                    f"Have {before_count}, expected {expected_atoms}. "
                    f"Extra: {extra_atoms} atoms (not an O2 pair). "
                    f"Structure is corrupted."
                )
            else:
                raise RuntimeError(
                    f"[Ep{episode_id}] FATAL: Missing atoms! "
                    f"Have {before_count}, expected {expected_atoms}. "
                    f"Missing: {-extra_atoms} atoms."
                )

    # Clear flags only after all decision logic above
    env.previous_episode_dissociated = False
    env.o2_removed = False

    # Add O2 and build the initial state
    try:
        # Reload structure_manager from the file just written
        if hasattr(env, 'structure_manager'):
            if hasattr(env.structure_manager, 'invalidate_cache'):
                env.structure_manager.invalidate_cache()
            atoms_from_disk = io.read(env.coo_file, format="lammps-data", atom_style="charge")
            env.structure_manager.update_atoms(atoms_from_disk)
            _LOG.debug(f"Forced structure_manager to reload from disk: {len(atoms_from_disk)} atoms")

        env.start_lammps(None)

        # Verify the substrate (no O2 yet)
        pre_o2_atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
        pre_count = len(pre_o2_atoms)
        pre_symbols = pre_o2_atoms.get_chemical_symbols()
        pre_si_count = sum(1 for sym in pre_symbols if sym == 'Si')
        pre_o_count = sum(1 for sym in pre_symbols if sym == 'O')
        
        _LOG.info(f"Structure has {pre_count} atoms before adding O2 (Si: {pre_si_count}, O: {pre_o_count})")

        # Sync cumulative count with the actual extra-O atoms on disk (covers
        # counter resets, checkpoint loads, and any desync)
        actual_atoms = len(pre_o2_atoms)
        actual_extra_o = actual_atoms - env.original_atom_count
        if actual_extra_o >= 0 and actual_extra_o != env.cumulative_dissociated_o_count:
            _LOG.info(f"[SYNC] cumulative_dissociated_o_count out of sync: "
                  f"counter={env.cumulative_dissociated_o_count}, "
                  f"actual extra atoms={actual_extra_o}. Syncing to reality.")
            env.cumulative_dissociated_o_count = actual_extra_o
            if hasattr(env, 'structure_manager'):
                env.structure_manager.cumulative_dissociated_o_count = actual_extra_o

        expected_substrate_atoms = env.original_atom_count + env.cumulative_dissociated_o_count

        if pre_count != expected_substrate_atoms:
            raise RuntimeError(
                f"[Ep{episode_id}] FATAL: Atom count mismatch before adding O2! "
                f"Have {pre_count}, expected {expected_substrate_atoms}"
            )

        if pre_si_count != env.original_si_count:
            _LOG.info(f"ERROR: Si count mismatch before adding O2! Expected {env.original_si_count}, got {pre_si_count}")
            raise RuntimeError("Si atom count error")

        # Save pre-episode substrate for rollback if O2 reforms in vacuum during MD
        import shutil as _shutil
        _pre_ep_state = os.path.join(get_temp_dir(), f'pre_episode_substrate_ep{episode_id}.data')
        _shutil.copy2(env.coo_file, _pre_ep_state)
        env.pre_episode_substrate_state = _pre_ep_state
        _LOG.debug(f'[Ep{episode_id}] Saved pre-episode substrate state.')

        top_z = env.check_top()
        result = env.create_initial_O2(None, top_z)

        if result is False:
            _LOG.info(f"ERROR: create_initial_O2 returned False")
            raise RuntimeError("Failed to create O2")

        # O2 is now in the structure. validate_atom_counts and downstream
        # accounting read this flag to know an env-owned O2 is present.
        env.active_o2 = True

        env.relax_str(None)
        # --- reward-v2: capture absolute initial PE (E_0, eV) of the relaxed start state ---
        try:
            from realize.environment.neb_calculator import KCAL_MOL_TO_EV
            _pe0_raw = env.lammps_interface.get_potential_energy(env.coo_file)
            env.episode_E0 = float(_pe0_raw) * KCAL_MOL_TO_EV
        except Exception as _e_e0:
            _log(env, 'error', f"[reward-v2] E_0 capture failed: {_e_e0}")
            raise
        env.step_emin_list = []
        env.step_commit_list = []
        gc.collect()

        atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
        atoms.set_pbc((True, True, True))
        atoms.wrap()
        write_structure_file(atoms, env.coo_file, atom_style="charge")

        post_count = len(atoms)
        expected_count = pre_count + 2  # added the 2 O atoms of O2
        _LOG.debug(f"Structure has {post_count} atoms after adding O2 (expected {expected_count})")

        if post_count != expected_count:
            raise RuntimeError(
                f"[Ep{episode_id}] FATAL: Atom count mismatch after adding O2! "
                f"Expected {expected_count}, got {post_count}"
            )

        if not validate_atom_counts(env, "[After adding O2]"):
            raise RuntimeError(f"[Ep{episode_id}] FATAL: Atom count validation failed after adding O2")

        dist, _ = env.check_o2_distance()

        if embedding_extractor is not None:
            embeddings_dict = embedding_extractor.get_equivariant_embedding(
                io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                return_symmetric_only=False
            )
        else:
            embeddings_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)

        embedding = embeddings_dict['symmetric']  # symmetric is used for state construction
        embedding_asymmetric = embeddings_dict['asymmetric']

        snapshot_created = env.get_last_snapshot()
        if snapshot_created:
            last_file = os.path.join(get_temp_dir(), "last.lammpstrj")
            append_snapshot_to_movie(last_file, movie_file, env.logger if hasattr(env, 'logger') else None)
        else:
            _log(env, 'debug', "No snapshot available for movie")

    except RuntimeError:
        raise
    except Exception as e:
        _log(env, 'error', f"Unexpected exception in reset_episode: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"[Ep{episode_id}] FATAL: Unhandled exception during reset") from e

    # Detach embeddings so they carry no gradient into the policy
    if embedding is not None:
        if hasattr(embedding, 'detach'):
            embedding = embedding.detach()
        if hasattr(embedding, 'requires_grad_'):
            embedding = embedding.requires_grad_(False)

    if embedding_asymmetric is not None:
        if hasattr(embedding_asymmetric, 'detach'):
            embedding_asymmetric = embedding_asymmetric.detach()
        if hasattr(embedding_asymmetric, 'requires_grad_'):
            embedding_asymmetric = embedding_asymmetric.requires_grad_(False)

    embeddings_dict = {
        'symmetric': embedding,
        'asymmetric': embedding_asymmetric
    }
    return dist, embeddings_dict

