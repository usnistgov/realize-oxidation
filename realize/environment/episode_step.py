"""Environment step functions: step_environment and step_environment_with_rollback."""
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
from realize.environment.neb_calculator_ase import ASENEBCalculator
from typing import Optional, Tuple
from realize.utils.logger import UnifiedLogger
from realize.environment.episode_helpers import (
    _log,
    append_snapshot_to_movie,
    handle_md_verification,
    validate_atom_counts,
)
import logging
_LOG = logging.getLogger(__name__)


def step_environment(env, trans_action, rot_action, episode, checkpoint_path=None):
    """
    Take a step; rot_action is [translation_vector, axis_direction].

    Returns (distance, embedding, reward, done, valid).
    """
    if checkpoint_path is None:
        checkpoint_path = get_embedding_checkpoint_path()

    step_result = env.step(trans_action, rot_action, checkpoint_path)

    if len(step_result) == 5:
        dist, embedding, reward, done, valid = step_result
    elif len(step_result) == 4:
        dist, embedding, reward, done = step_result
        valid = True
    else:
        raise ValueError(f"Unexpected return from env.step: {len(step_result)} values")

    return dist, embedding, reward, done, valid

def step_environment_with_rollback(env, trans_action, rot_action, episode, prev_state=None,
                                  use_neb=True, ea_weight=4.0, dissociation_weight=2.0,
                                  require_dissociation=True, is_final_step=False,
                                  checkpoint_path=None, timeout_penalty=0.5,
                                  invalid_step_penalty=0.005, high_barrier_threshold=2.5,
                                  high_barrier_penalty=0.001, oscillation_penalty=0.005,
                                  step_count=0, max_steps=15, reward_decay_factor=0.99,
                                  agent=None, current_try_num=0, max_trans_distance=3.0,
                                  alpha=0.15, calculate_neb_output=None, logger=None,
                                  pe_reward_weight=0.01, action_dict=None, device=None,
                                  pe_shaping_gamma=0.99, invalid_penalty_cap=-0.3,
                                  embedding_extractor=None):    
    """
    Take a step with rollback for invalid NEB paths; runs NEB and MD verification.

    Raises ValueError if device is None.
    """
    if device is None:
        raise ValueError(
            "device parameter is REQUIRED for step_environment_with_rollback(). "
            "Must be torch.device('cuda') or torch.device('cpu'). "
            "Caller must provide explicit device."
        )

    # Per-episode cumulative invalid penalty (reset on new episode)
    if not hasattr(env, '_episode_invalid_penalty_cumulative'):
        env._episode_invalid_penalty_cumulative = 0.0
    if not hasattr(env, '_current_penalty_episode'):
        env._current_penalty_episode = -1

    if env._current_penalty_episode != episode:
        env._episode_invalid_penalty_cumulative = 0.0
        env._current_penalty_episode = episode

    def _get_capped_invalid_penalty(base_penalty: float) -> float:
        """Cap cumulative invalid penalties so escaping into vacuum stays unattractive."""
        if env._episode_invalid_penalty_cumulative <= invalid_penalty_cap:
            return 0.0

        # Both terms negative
        remaining = invalid_penalty_cap - env._episode_invalid_penalty_cumulative
        actual_penalty = max(-base_penalty, remaining)

        env._episode_invalid_penalty_cumulative += actual_penalty

        return actual_penalty

    if not hasattr(env, 'current_try_num'):
        env.current_try_num = current_try_num
    actual_try_num = env.current_try_num

    if actual_try_num > 0:
        _LOG.debug(f"  try_num: {actual_try_num}")


    from ase import io
    from realize.utils.file_utils import write_structure_file, standardize_file_format
    from realize.environment.neb_calculator import NEBCalculator
    from realize.config.paths import get_embedding_checkpoint_path, get_movie_file_path, get_neb_results_dir, get_temp_dir, get_neb_input_file_path

    if checkpoint_path is None:
        checkpoint_path = get_embedding_checkpoint_path()

    movie_file = get_movie_file_path(episode + 1)

    if prev_state is None:
        raise ValueError(
            "prev_state must be provided for rollback capability. "
            "Cannot auto-create state files."
        )

    caller_state_file = prev_state

    if logger and logger.verbosity == 'DEBUG':
        if isinstance(rot_action, np.ndarray) and rot_action.shape == (2, 3):
            logger.log_debug(f"SH Action - Translation: {rot_action[0]} (magnitude: {np.linalg.norm(rot_action[0]):.3f} Angstrom)")
            logger.log_debug(f"SH Action - Axis direction: {rot_action[1]} (norm: {np.linalg.norm(rot_action[1]):.3f})")

    before_step_state = os.path.join(get_temp_dir(), f"before_step_ep{episode}_step{step_count}_try{actual_try_num}.data")
    curr_state_file = None

    try:
        env.save_current_state(before_step_state)
        standardize_file_format(before_step_state, target_style='charge')

        # Pre-action snapshot serves as the NEB initial endpoint
        orig_state_file = before_step_state

        if hasattr(env.structure_manager, 'verify_substrate_integrity'):
            integrity_ok = env.structure_manager.verify_substrate_integrity()
            if not integrity_ok and logger:
                logger.log_error("Substrate integrity check failed before step!")

        if not validate_atom_counts(env, "[Before step]"):
            _log(env, 'warning', "Atom count validation failed before step")

    except Exception as e:
        if logger:
            logger.log_warning(f"Failed to save before-step state: {e}")

    try:
        step_output = env.step(trans_action, rot_action, checkpoint_path,
                               device=device,
                               oscillation_penalty=oscillation_penalty)

        if len(step_output) == 5:
            dist_new, embedding_new, raw_reward, is_done, is_valid = step_output
        elif len(step_output) == 4:
            dist_new, embedding_new, raw_reward, is_done = step_output
            is_valid = True
        else:
            dist_new, embedding_new, raw_reward = step_output[0], step_output[1], step_output[2]
            is_done = step_output[3] if len(step_output) > 3 else False
            is_valid = True

        _LOG.debug(f"  dist_new: {dist_new:.3f}")
        _LOG.debug(f"  is_done: {is_done}")
        _LOG.debug(f"  is_valid: {is_valid}")

        # Skip embedding when the episode ended (O2 removed/dissociated): O-O
        # distance would be out of range for the extractor
        if is_done:
            embedding_new = None
            embeddings_new_dict = None
        elif embedding_extractor is not None:
            embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                return_symmetric_only=False
            )
            embedding_new = embeddings_new_dict['symmetric']  # symmetric is used for state construction
        else:
            embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
            embedding_new = embeddings_new_dict['symmetric']

    except Exception as e:
        _log(env, 'error', f"env.step() failed -- treating as invalid step with rollback")
        import traceback
        traceback.print_exc()

        # Roll back to pre-step state rather than crash
        try:
            if os.path.exists(orig_state_file):
                env.load_state(orig_state_file)
                _LOG.debug(f"[RECOVERY] Rolled back to: {orig_state_file}")
                if hasattr(env, 'action_processor'):
                    env.action_processor.reset_history()
                env.o2_removed = False
                env.current_try_num = actual_try_num + 1

                dist_new, _ = env.check_o2_distance()
                if embedding_extractor is not None:
                    embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                        io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                        return_symmetric_only=False
                    )
                else:
                    embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                embedding_new = embeddings_new_dict['symmetric']

                _LOG.debug(f"[RECOVERY] After rollback: dist={dist_new:.3f}")
                _LOG.debug(f"[RECOVERY] Returning invalid step (env.step failure: {type(e).__name__})")

                return (dist_new, embedding_new, -0.005, False, False, -1.0, current_try_num, False)
            else:
                raise RuntimeError(
                    f"FATAL: env.step() failed and no rollback state available at "
                    f"episode {episode}, step {step_count}"
                ) from e
        except RuntimeError:
            raise
        except Exception as rollback_e:
            raise RuntimeError(
                f"FATAL: env.step() failed AND rollback failed at "
                f"episode {episode}, step {step_count}. "
                f"Original error: {e}, Rollback error: {rollback_e}"
            ) from e

    # Invalid step (oscillation detected in env.step): roll back
    if not is_valid:
        if logger:
            logger.log_debug("Invalid step detected (oscillation) - rolling back")

        try:
            if os.path.exists(orig_state_file):
                env.load_state(orig_state_file)
                _LOG.debug(f"[DEBUG step_env] Rolled back to: {orig_state_file}")

                if hasattr(env, 'action_processor'):
                    env.action_processor.reset_history()
                    _log(env, 'debug', "Reset action processor history for new episode")

                env.o2_removed = False

                env.current_try_num = actual_try_num + 1

                dist_new, _ = env.check_o2_distance()

                if embedding_extractor is not None:
                    embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                        io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                        return_symmetric_only=False
                    )
                else:
                    embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                    
                embedding_new = embeddings_new_dict['symmetric']
                
                _LOG.debug(f"[DEBUG step_env] After rollback: dist={dist_new:.3f}")
        except Exception as e:
            _LOG.debug(f"[DEBUG step_env] Failed to rollback: {e}")
            if logger:
                logger.log_error(f"Failed to rollback after oscillation: {e}")

        # No snapshot for invalid oscillation steps

        try:
            if os.path.exists(before_step_state):
                os.remove(before_step_state)
        except:
            pass

        _LOG.debug(f" Returning oscillation penalty: {raw_reward:.3f}")
        return (dist_new, embedding_new, raw_reward, False, False, -1.0, env.current_try_num, False)

    # Was O2 removed during the step?
    o2_removed = hasattr(env, 'o2_removed') and env.o2_removed
    _LOG.debug(f" O2 removed flag: {o2_removed}")

    if not o2_removed and hasattr(env, 'original_atom_count'):
        try:
            atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
            current_count = len(atoms)
            cumulative_dissociated = getattr(env, 'cumulative_dissociated_o_count', 0)
            expected_with_o2 = env.original_atom_count + cumulative_dissociated + 2

            if current_count < expected_with_o2:
                # O2 missing -> removed
                o2_removed = True
                env.o2_removed = True
                _LOG.debug(f" O2 removal detected by atom count")
                if logger:
                    logger.log_debug(f"O2 removal detected: {current_count} atoms < {expected_with_o2} expected")
        except:
            pass

    # O2 removed (flew away): terminate episode
    if o2_removed:
        if logger:
            logger.log_debug("O2 molecule has been removed - terminating episode")

        snapshot_created = env.get_last_snapshot()
        if snapshot_created:
            last_file = os.path.join(get_temp_dir(), "last.lammpstrj")
            append_snapshot_to_movie(last_file, movie_file, logger)

        env.o2_removed = True

        try:
            if os.path.exists(before_step_state):
                os.remove(before_step_state)
        except:
            pass

        _LOG.debug(f" Returning removal penalty: -1.0")
        return (np.inf, embedding_new, -1.0, True, True, -1.0, current_try_num, False)

    initial_dissociation = env.is_dissociated(force_check=True) if hasattr(env, 'is_dissociated') else False

    # Not using NEB: accept the step
    if not use_neb:

        snapshot_created = env.get_last_snapshot()
        if snapshot_created:
            last_file = os.path.join(get_temp_dir(), "last.lammpstrj")
            append_snapshot_to_movie(last_file, movie_file, logger)

        if initial_dissociation:
            _LOG.debug(f" Running MD verification...")
            md_verified_dissociation, dist_new, embedding_new = handle_md_verification(
                env, episode, movie_file, device=device, checkpoint_path=checkpoint_path, 
                embedding_extractor=embedding_extractor, logger=logger
            )                    
            if getattr(env, '_scenario3_rollback', False):
                env._scenario3_rollback = False
                _LOG.debug(f'[MD VERIFY] Scenario 3 rollback -- ending episode as invalid.')
                return (np.inf, None, -0.5, True, False, -1.0, current_try_num, False)
            is_done = md_verified_dissociation
        else:
            is_done = False
            md_verified_dissociation = False

        if os.path.exists(before_step_state):
            try:
                os.remove(before_step_state)
            except:
                pass

        _LOG.debug(f" Returning non-NEB result: done={is_done}")
        return (dist_new, embedding_new, 0.0, is_done, True, 0.0, current_try_num, md_verified_dissociation)

    return _run_neb_processing(
        env, episode, step_count, current_try_num, dist_new, embedding_new,
        orig_state_file, before_step_state, movie_file, initial_dissociation,
        calculate_neb_output, high_barrier_threshold, pe_reward_weight,
        pe_shaping_gamma, invalid_step_penalty, timeout_penalty, logger,
        embedding_extractor, checkpoint_path, device, agent,
        _get_capped_invalid_penalty,
    )


def _run_neb_processing(env, episode, step_count, current_try_num, dist_new,
                        embedding_new, orig_state_file, before_step_state,
                        movie_file, initial_dissociation, calculate_neb_output,
                        high_barrier_threshold, pe_reward_weight, pe_shaping_gamma,
                        invalid_step_penalty, timeout_penalty, logger,
                        embedding_extractor, checkpoint_path, device, agent,
                        _get_capped_invalid_penalty):
    # Defaults for the return tuple in case NEB does not run
    pe_reward = 0.0
    actual_try_num = current_try_num
    activation_energy = -1.0
    valid_step = False
    md_verified_dissociation = False
    is_done = False

    try:
        neb_results_dir = get_neb_results_dir()
        os.makedirs(neb_results_dir, exist_ok=True)

        # Authoritative try counter from env to avoid desync
        try_idx = getattr(env, 'current_try_num', current_try_num)
        curr_state_file = os.path.join(neb_results_dir, f"state_ep{episode}_step{step_count}_try{try_idx}.data")
        env.save_current_state(curr_state_file)

        try:
            standardize_file_format(curr_state_file, target_style='charge')
        except Exception as e:
            if logger:
                logger.log_warning(f"Failed to standardize file format: {e}")

        if os.path.exists(curr_state_file):
            file_size = os.path.getsize(curr_state_file)
        else:
            _LOG.error(f"Failed to create state file: {curr_state_file}")
            return (dist_new, embedding_new, -invalid_step_penalty, False, False, -1.0, current_try_num, False)

        activation_energy = -1.0
        valid_step = False
        pe_reward = 0.0

        if calculate_neb_output is None:
            calculate_neb_output = True

        if calculate_neb_output:
            neb_start_time = time.time()

            if not os.path.exists(orig_state_file):
                raise FileNotFoundError(
                    f"Previous state file missing: {orig_state_file}. "
                    f"Episode {episode}, step {step_count}, try {current_try_num}"
                )

            # In parallel worker mode, _neb_num_procs_override caps MPI ranks
            # to this worker's share of CPU cores
            neb_procs = getattr(env, '_neb_num_procs_override', None)
            neb_calc = ASENEBCalculator(
                orig_state_file,
                curr_state_file,
                neb_in_file=get_neb_input_file_path(),
                logger=logger,
                env=env,
                num_procs_override=neb_procs
            )

            image_creation_start = time.time()
            neb_calc.create_images()
            image_creation_time = time.time() - image_creation_start

            _LOG.debug(f" Running NEB calculation...")
            neb_run_start = time.time()
            success = neb_calc.run_neb(episode=episode, step=step_count, try_num=try_idx)
            neb_run_time = time.time() - neb_run_start

            if success:
                activation_energy = neb_calc.get_activation_energy(
                    high_barrier_threshold=high_barrier_threshold,
                    silent=True
                )
                valid_step = (0 <= activation_energy <= high_barrier_threshold)
                _LOG.info(f" NEB result: activation={activation_energy:.3f}, valid={valid_step}")

                # --- reward-v2 capture: absolute E_min,i (eV) + commitment endpoints ---
                from realize.environment.neb_calculator import KCAL_MOL_TO_EV
                _E_min_i = None
                _commit_RP = (None, None)
                if valid_step:
                    try:
                        _pe_raw = env.lammps_interface.get_potential_energy(orig_state_file)
                        _E_min_i = float(_pe_raw) * KCAL_MOL_TO_EV
                    except Exception as _e_emin:
                        _LOG.warning(f"  [reward-v2] E_min capture failed: {_e_emin}")
                        _E_min_i = None
                    try:
                        _commit_RP = neb_calc.get_commitment_endpoints(silent=True)
                    except Exception as _e_cmt:
                        _LOG.warning(f"  [reward-v2] commitment-endpoint capture failed: {_e_cmt}")
                        _commit_RP = (None, None)
                if not hasattr(env, 'step_emin_list'):
                    env.step_emin_list = []
                if not hasattr(env, 'step_commit_list'):
                    env.step_commit_list = []
                env._pending_emin = _E_min_i
                env._pending_commit = _commit_RP

                # Track "success but invalid" runs to escape soft-loops
                if not valid_step:
                    if not hasattr(env, 'invalid_neb_tracker'):
                        env.invalid_neb_tracker = {}
                    failure_key = f"ep{episode}_step{step_count}"
                    env.invalid_neb_tracker[failure_key] = env.invalid_neb_tracker.get(failure_key, 0) + 1
                    consecutive_invalid = env.invalid_neb_tracker[failure_key]
                    _LOG.debug(f"  Consecutive invalid (but successful) NEBs: {consecutive_invalid}")
                else:
                    if hasattr(env, 'invalid_neb_tracker'):
                        failure_key = f"ep{episode}_step{step_count}"
                        env.invalid_neb_tracker.pop(failure_key, None)

                # PE reward only for valid steps. Potential-based shaping
                # F(s,s') = PE(s) - gamma*PE(s') keeps cumulative reward
                # independent of path length.
                pe_reward = 0.0
                if valid_step and pe_reward_weight > 0:
                    pe_initial, pe_final = neb_calc.get_endpoint_energies()

                    if pe_initial is not None and pe_final is not None:
                        from realize.ppo.rewards import calculate_potential_energy_reward

                        delta_pe = pe_initial - pe_final
                        if abs(delta_pe) > 5.0:
                            if logger:
                                logger.log_warning(f"Very large PE change detected: {delta_pe:.4f} eV")
                                logger.log_debug(f"PE initial: {pe_initial:.4f}, PE final: {pe_final:.4f}")

                        pe_reward = calculate_potential_energy_reward(
                            pe_initial,
                            pe_final,
                            weight=pe_reward_weight,
                            gamma=pe_shaping_gamma
                        )

                        # Floor at 0.01 so approaching the surface (PE temporarily up) is not penalized
                        pe_reward = max(0.01, pe_reward)

                        _LOG.debug(f" PE change: {pe_initial:.4f} -> {pe_final:.4f} eV (gamma={pe_shaping_gamma}), reward: {pe_reward:.4f}")

            else:
                # NEB calculation failed (return code != 0)
                _LOG.debug(f"[DEBUG step_env] NEB calculation completely failed")

                _LOG.debug(f"[DEBUG FAILURE] Episode={episode}, Step={step_count}, Try={current_try_num}")
                if hasattr(env, 'global_catastrophic_failures'):
                    _LOG.debug(f"[DEBUG FAILURE] Global catastrophic count: {env.global_catastrophic_failures}")

                # Log O2 position for failure-pattern analysis
                try:
                    o2_dist, _ = env.check_o2_distance()
                    atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
                    o1_pos = atoms.get_positions()[-2]
                    o2_pos = atoms.get_positions()[-1]
                    _LOG.debug(f"[DEBUG FAILURE] O2 distance: {o2_dist:.3f} Angstrom")
                    _LOG.debug(f"[DEBUG FAILURE] O1 position: {o1_pos}")
                    _LOG.debug(f"[DEBUG FAILURE] O2 position: {o2_pos}")
                except:
                    pass

                activation_energy = -1.0
                valid_step = False
                pe_reward = 0.0

                catastrophic_failure = False
                has_nan = False

                if hasattr(neb_calc, 'has_nan_energies'):
                    has_nan = neb_calc.has_nan_energies()
                    if has_nan:
                        catastrophic_failure = True
                        _LOG.info(f"  [CATASTROPHIC] NaN energies detected in NEB output")
                        if logger:
                            logger.log_warning("NaN energies detected in NEB calculation")

                if not hasattr(env, 'neb_failure_tracker'):
                    env.neb_failure_tracker = {}

                failure_key = f"ep{episode}_step{step_count}"
                env.neb_failure_tracker[failure_key] = env.neb_failure_tracker.get(failure_key, 0) + 1
                consecutive_failures = env.neb_failure_tracker[failure_key]

                _LOG.debug(f"  Consecutive NEB failures at this step: {consecutive_failures}")

                # 3+ consecutive failures or any NaN -> catastrophic
                if consecutive_failures >= 3 or has_nan:
                    catastrophic_failure = True
                    _LOG.info(f"  [CATASTROPHIC] Forcing step skip after {consecutive_failures} failures")
                    if logger:
                        logger.log_warning(f"Catastrophic NEB failure at step {step_count}")

                if catastrophic_failure:
                    if failure_key in env.neb_failure_tracker:
                        del env.neb_failure_tracker[failure_key]

                    if not hasattr(env, 'global_catastrophic_failures'):
                        env.global_catastrophic_failures = 0
                    env.global_catastrophic_failures += 1

                    # 5+ catastrophic failures in one episode terminates it
                    if env.global_catastrophic_failures >= 5:
                        _LOG.info(f"  [TERMINAL] {env.global_catastrophic_failures} catastrophic failures in episode - terminating")
                        if logger:
                            logger.log_error(f"Episode terminated: {env.global_catastrophic_failures} catastrophic NEB failures")

                        try:
                            if os.path.exists(before_step_state):
                                os.remove(before_step_state)
                            if os.path.exists(curr_state_file):
                                os.remove(curr_state_file)
                        except:
                            pass

                        return (np.inf, None, -timeout_penalty, True, False, -1.0, current_try_num, False)

                    # force_next_step tells the runner to skip past this problematic state
                    env.force_next_step = True

                    env.current_try_num = 0
                    actual_try_num = 0

                    _LOG.info(f"  [CATASTROPHIC #{env.global_catastrophic_failures}] Set force_next_step=True to skip problematic state")
                    if logger:
                        logger.log_warning(f"Catastrophic failure #{env.global_catastrophic_failures} - will force skip to next step")

                    try:
                        # O2 may have dissociated before rollback; treat that as success
                        pre_rollback_dist, _ = env.check_o2_distance()
                        if pre_rollback_dist > 1.8:
                            _LOG.info(f"  [CATASTROPHIC] O2 dissociated before rollback! "
                                  f"O-O={pre_rollback_dist:.3f} Angstrom -- treating as SUCCESS")
                            env.previous_episode_dissociated = True
                            env.last_md_verified_dissociation = True
                            # NEB failed here, so no saddle was verified: NOT an accepted transition.
                            # activation=-1.0 makes the runner's `>= 0` gate skip the
                            # append, keeping only verified transitions in Ea_eff.
                            return (pre_rollback_dist, None, 0.0, True, True, -1.0, current_try_num, False)
                        if os.path.exists(orig_state_file):
                            env.load_state(orig_state_file)
                            if hasattr(env, 'action_processor'):
                                env.action_processor.reset_history()
                            dist_new, _ = env.check_o2_distance()

                            if embedding_extractor is not None:
                                embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                                    io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                                    return_symmetric_only=False
                                )
                            else:
                                embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                                
                            embedding_new = embeddings_new_dict['symmetric']
                            _LOG.debug(f"  Rolled back, O2 distance={dist_new:.3f}")
                    except Exception as e:
                        _LOG.info(f"  ERROR during rollback: {e}")

                    try:
                        if os.path.exists(before_step_state):
                            os.remove(before_step_state)
                        if os.path.exists(curr_state_file):
                            os.remove(curr_state_file)
                    except:
                        pass

                    _LOG.debug(f" Returning catastrophic penalty: {-timeout_penalty:.3f}")
                    return (dist_new, embedding_new, -timeout_penalty, False, False, -1.0, 0, False)

                # Normal NEB failure (not yet catastrophic)
                else:
                    env.current_try_num = actual_try_num + 1
                    new_try_num = env.current_try_num

                    if logger:
                        logger.log_debug(f"Invalid NEB step (try {current_try_num}). Next try will be {new_try_num}. Applying penalty: -{invalid_step_penalty:.1f}")
                    _LOG.debug(f"  NEB failed, incrementing try_num: {actual_try_num} -> {new_try_num}")
                    _LOG.debug(f"  Next try will be: {new_try_num}")

                    # 20-try rollback limit terminates the episode
                    if new_try_num >= 20:
                        _LOG.info(f"  [TIMEOUT] Reached 20 tries - terminating episode")
                        if logger:
                            logger.log_warning(f"Reached retry limit (20) at step {step_count}")

                        try:
                            if os.path.exists(before_step_state):
                                os.remove(before_step_state)
                            if os.path.exists(curr_state_file):
                                os.remove(curr_state_file)
                        except:
                            pass

                        return (np.inf, None, -timeout_penalty, True, False, -1.0, new_try_num, False)

                    # Roll back to restore the O2 position
                    try:
                        if os.path.exists(orig_state_file):
                            env.load_state(orig_state_file)
                            _LOG.debug(f"  Rolled back O2 position to: {orig_state_file}")

                            # Clear position history to avoid false oscillation detection
                            if hasattr(env, 'action_processor'):
                                env.action_processor.reset_history()
                                if logger:
                                    logger.log_debug("Reset action processor history after rollback")

                            dist_new, _ = env.check_o2_distance()

                            if embedding_extractor is not None:
                                embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                                    io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                                    return_symmetric_only=False
                                )
                            else:
                                embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                                
                            embedding_new = embeddings_new_dict['symmetric']
                            _LOG.debug(f"  After rollback: O2 distance={dist_new:.3f}")
                        else:
                            _LOG.info(f"  WARNING: Previous state file not found for rollback: {orig_state_file}")
                            if logger:
                                logger.log_warning(f"Cannot rollback - previous state file missing: {orig_state_file}")
                    except Exception as e:
                        _LOG.info(f"  ERROR during rollback: {e}")
                        if logger:
                            logger.log_error(f"Failed to rollback after invalid NEB: {e}")

                    # Clean up temporary files
                    try:
                        if os.path.exists(before_step_state):
                            os.remove(before_step_state)
                        if os.path.exists(curr_state_file):
                            os.remove(curr_state_file)
                    except:
                        pass

                    capped_penalty = _get_capped_invalid_penalty(invalid_step_penalty)
                    _LOG.debug(f" Returning invalid step penalty: {capped_penalty:.3f} (cumulative: {env._episode_invalid_penalty_cumulative:.3f})")
                    return (dist_new, embedding_new, capped_penalty, False, False, activation_energy, new_try_num, False)

            neb_total_time = time.time() - neb_start_time

            if not hasattr(env, 'neb_stats'):
                env.neb_stats = {
                    'count': 0,
                    'valid_count': 0,
                    'total_time': 0.0,
                    'image_time': 0.0,
                    'run_time': 0.0,
                    'times': [],
                    'activations': [],
                    'success_rate': 0.0
                }
            
            env.neb_stats['count'] += 1
            env.neb_stats['total_time'] += neb_total_time
            env.neb_stats['image_time'] += image_creation_time
            env.neb_stats['run_time'] += neb_run_time
            env.neb_stats['times'].append(neb_total_time)
            
            if success:
                env.neb_stats['valid_count'] += 1
                if activation_energy >= 0:
                    env.neb_stats['activations'].append(activation_energy)
            
            env.neb_stats['success_rate'] = env.neb_stats['valid_count'] / env.neb_stats['count']

            # Log stats every 10 NEB calls
            if env.neb_stats['count'] % 10 == 0:
                avg_time = env.neb_stats['total_time'] / env.neb_stats['count']
                recent_times = env.neb_stats['times'][-10:]
                recent_avg = sum(recent_times) / len(recent_times)

                if env.neb_stats['activations']:
                    recent_activations = env.neb_stats['activations'][-10:]
                    act_mean = np.mean(recent_activations)
                    act_std = np.std(recent_activations)
                    act_str = f", act_energy={act_mean:.2f}+/-{act_std:.2f} eV"
                else:
                    act_str = ""
                
                if logger:
                    logger.log_debug(
                        f"NEB Stats [{env.neb_stats['count']} calls]: "
                        f"success_rate={env.neb_stats['success_rate']:.1%}, "
                        f"avg_time={avg_time:.2f}s (recent={recent_avg:.2f}s), "
                        f"image/run={image_creation_time:.1f}s/{neb_run_time:.1f}s"
                        f"{act_str}"
                    )
            
            if logger:
                logger.log_neb_result(activation_energy, valid_step)

            if valid_step:
                current_dissociation = env.is_dissociated(force_check=True) if hasattr(env, 'is_dissociated') else False
            else:
                current_dissociation = env.is_dissociated(force_check=False) if hasattr(env, 'is_dissociated') else False

        # NEB ran but result invalid: roll back
        if not valid_step:
            # Soft-loop: repeated "success but invalid" results
            failure_key = f"ep{episode}_step{step_count}"
            invalid_cnt = getattr(env, 'invalid_neb_tracker', {}).get(failure_key, 0)

            if invalid_cnt >= 3:
                # Skip this step to avoid grinding forever
                _LOG.info(f"  [SOFT-LOOP] {invalid_cnt} consecutive invalid NEBs - forcing skip")
                env.force_next_step = True

                if hasattr(env, 'invalid_neb_tracker') and failure_key in env.invalid_neb_tracker:
                    del env.invalid_neb_tracker[failure_key]

                # Roll back and bail with timeout penalty
                try:
                    if os.path.exists(orig_state_file):
                        env.load_state(orig_state_file)
                        if hasattr(env, 'action_processor'):
                            env.action_processor.reset_history()
                        dist_new, _ = env.check_o2_distance()

                        if embedding_extractor is not None:
                            embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                                io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                                return_symmetric_only=False
                            )
                        else:
                            embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                            
                        embedding_new = embeddings_new_dict['symmetric']
                        _LOG.debug(f"  Rolled back, O2 distance={dist_new:.3f}")
                except Exception as e:
                    _LOG.info(f"  ERROR during rollback: {e}")

                try:
                    if os.path.exists(before_step_state):
                        os.remove(before_step_state)
                    if os.path.exists(curr_state_file):
                        os.remove(curr_state_file)
                except:
                    pass

                _LOG.debug(f" Returning soft-loop penalty: {-timeout_penalty:.3f}")
                return (dist_new, embedding_new, -timeout_penalty, False, False, -1.0, 0, False)

            env.current_try_num = actual_try_num + 1
            new_try_num = env.current_try_num

            if logger:
                logger.log_debug(f"Invalid NEB step (try {current_try_num}). Next try will be {new_try_num}. Applying penalty: -{invalid_step_penalty:.1f}")
            _LOG.debug(f"  Next try will be: {new_try_num}")

            # 20-try rollback limit terminates the episode
            if new_try_num >= 20:
                _LOG.info(f"  [TIMEOUT] Reached 20 tries - terminating episode")
                if logger:
                    logger.log_warning(f"Reached retry limit (20) at step {step_count}")

                try:
                    if os.path.exists(before_step_state):
                        os.remove(before_step_state)
                    if os.path.exists(curr_state_file):
                        os.remove(curr_state_file)
                except:
                    pass

                return (np.inf, None, -timeout_penalty, True, False, -1.0, new_try_num, False)

            # Roll back to restore the O2 position
            try:
                if os.path.exists(orig_state_file):
                    env.load_state(orig_state_file)
                    _LOG.debug(f"  Rolled back O2 position to: {orig_state_file}")

                    # Clear position history to avoid false oscillation detection
                    if hasattr(env, 'action_processor'):
                        env.action_processor.reset_history()
                        if logger:
                            logger.log_debug("Reset action processor history after rollback")

                    dist_new, _ = env.check_o2_distance()

                    if embedding_extractor is not None:
                        embeddings_new_dict = embedding_extractor.get_equivariant_embedding(
                            io.read(env.coo_file, format="lammps-data", atom_style="charge"),
                            return_symmetric_only=False
                        )
                    else:
                        embeddings_new_dict = get_embedding_vector(env.coo_file, checkpoint_path, device=device)
                        
                    embedding_new = embeddings_new_dict['symmetric']
                    _LOG.debug(f"  After rollback: O2 distance={dist_new:.3f}")
                else:
                    _LOG.info(f"  WARNING: Previous state file not found for rollback: {orig_state_file}")
                    if logger:
                        logger.log_warning(f"Cannot rollback - previous state file missing: {orig_state_file}")
            except Exception as e:
                _LOG.info(f"  ERROR during rollback: {e}")
                if logger:
                    logger.log_error(f"Failed to rollback after invalid NEB: {e}")

            try:
                if os.path.exists(before_step_state):
                    os.remove(before_step_state)
                if os.path.exists(curr_state_file):
                    os.remove(curr_state_file)
            except:
                pass

            capped_penalty = _get_capped_invalid_penalty(invalid_step_penalty)
            _LOG.debug(f" Returning invalid step penalty: {capped_penalty:.3f} (cumulative: {env._episode_invalid_penalty_cumulative:.3f})")
            return (dist_new, embedding_new, capped_penalty, False, False, activation_energy, new_try_num, False)

        # Valid NEB step: apply PE reward. Add a snapshot now only when MD
        # verification will not run (it handles its own snapshots).
        post_neb_dissociation_check = env.is_dissociated(force_check=False) if hasattr(env, 'is_dissociated') else False
        if not (post_neb_dissociation_check and valid_step):
            snapshot_created = env.get_last_snapshot()
            if snapshot_created:
                last_file = os.path.join(get_temp_dir(), "last.lammpstrj")
                append_snapshot_to_movie(last_file, movie_file, logger)

        # Activation energy accumulates for the episode-end summary
        if not hasattr(env, 'activation_energies'):
            env.activation_energies = []
        env.activation_energies.append(activation_energy)

        post_neb_dissociation = post_neb_dissociation_check

        # Record per-step reward-v2 quantities for ACCEPTED valid steps (parallel to activations)
        # NOTE: the 0.8 eV dissociation gate was removed -- it intercepted dissociations before
        # MD verification, so real (1.5-2.5 eV) dissociations never verified or counted. Barrier
        # preference is handled by the reward's Fermi-Dirac mu mapping, per manuscript design.
        if valid_step:
            if not hasattr(env, 'step_emin_list'):
                env.step_emin_list = []
            if not hasattr(env, 'step_commit_list'):
                env.step_commit_list = []
            env.step_emin_list.append(getattr(env, '_pending_emin', None))
            env.step_commit_list.append(getattr(env, '_pending_commit', (None, None)))
        _LOG.debug(f" Post-NEB dissociation check: {post_neb_dissociation}")

        current_distance, _ = env.check_o2_distance()
        _LOG.debug(f"  initial_dissociation: {initial_dissociation}")
        _LOG.debug(f"  post_neb_dissociation: {post_neb_dissociation}")
        _LOG.debug(f"  valid_step: {valid_step}")
        _LOG.debug(f"  current O2 distance: {current_distance:.3f} Angstrom")
        _LOG.debug(f"  MD will run: {post_neb_dissociation and valid_step}")

        if initial_dissociation and not post_neb_dissociation:
            _LOG.info(f"  WARNING: Dissociation lost after NEB!")
            _LOG.debug(f"  Distance was dissociated, now: {current_distance:.3f} Angstrom")

        md_verified_dissociation = False
        is_done = False

        # Run MD verification if dissociated after a valid NEB step
        if post_neb_dissociation and valid_step:
            _LOG.debug(f"\n=== Before MD Verification ===")
            _LOG.debug(f" Running MD verification for dissociation...")

            # Saved for the fallback decision below
            pre_md_distance = current_distance

            try:
                md_verified_dissociation, dist_new, embedding_new = handle_md_verification(
                    env, episode, movie_file, device=device, checkpoint_path=checkpoint_path,
                    embedding_extractor=embedding_extractor, logger=logger
                )
                if getattr(env, '_scenario3_rollback', False):
                    env._scenario3_rollback = False
                    _LOG.debug(f'[MD VERIFY] Scenario 3 rollback -- ending episode as invalid.')
                    return (np.inf, None, -0.5, True, False, -1.0, current_try_num, False)
            except Exception as md_e:
                _LOG.error(f"MD verification raised exception: {md_e}")
                md_verified_dissociation = False
                # dist_new, embedding_new keep their pre-MD values

            if md_verified_dissociation:
                env.previous_episode_dissociated = True
                is_done = True
                if not hasattr(env, 'cumulative_dissociated_o_count'):
                    env.cumulative_dissociated_o_count = 0
                old_count = env.cumulative_dissociated_o_count
                # +1 if one O flew to vacuum (scenario 2), else +2 for the pair
                increment = 1 if getattr(env, '_scenario2_o_removed', False) else 2
                env._scenario2_o_removed = False
                env.cumulative_dissociated_o_count += increment
                # O2 promoted to substrate; env no longer owns it as active.
                env.active_o2 = False
                if hasattr(env, 'structure_manager'):
                    env.structure_manager.cumulative_dissociated_o_count = env.cumulative_dissociated_o_count
                _LOG.debug(f" MD verified dissociation - updated cumulative count: "
                      f"{old_count} -> {env.cumulative_dissociated_o_count}")
            elif pre_md_distance > 2.5:
                # MD failed (atom loss / LAMMPS error) but O2 was clearly dissociated
                # beforehand: structure is broken, so end as a failed dissociation
                # (timeout penalty, not a dissociation reward).
                _LOG.warning(f"MD verification failed but pre-MD O-O distance was "
                      f"{pre_md_distance:.3f} Angstrom (> 2.5 Angstrom). O2 structure broken -- "
                      f"ending episode as failed dissociation.")
                is_done = True
                md_verified_dissociation = False

                # Restore pre-MD state so the env is clean for reset
                md_initial_state = os.path.join(get_temp_dir(), "md_initial_state.data")
                if os.path.exists(md_initial_state):
                    try:
                        env.load_state(md_initial_state)
                        _LOG.debug(f"  Restored pre-MD state")
                    except Exception as restore_e:
                        _LOG.info(f"  [WARNING] Could not restore pre-MD state: {restore_e}")
                # Update the count now to avoid atom-count validation errors
                if not hasattr(env, 'cumulative_dissociated_o_count'):
                    env.cumulative_dissociated_o_count = 0

                old_count = env.cumulative_dissociated_o_count
                env.cumulative_dissociated_o_count += 2
                # O2 promoted to substrate; env no longer owns it as active.
                env.active_o2 = False

                if hasattr(env, 'structure_manager'):
                    env.structure_manager.cumulative_dissociated_o_count = env.cumulative_dissociated_o_count

                if not hasattr(env, 'episode_outcomes'):
                    env.episode_outcomes = {}
                env.episode_outcomes[episode] = {'dissociated': True}

                _LOG.debug(f" MD verified dissociation - updated cumulative count: {old_count} -> {env.cumulative_dissociated_o_count}")

            if not md_verified_dissociation:
                # Mark the last transition invalid rather than popping parallel lists (avoids misalignment)
                if agent is not None and hasattr(agent, 'buffer_manager'):
                    try:
                        buffer = agent.buffer_manager.buffer

                        if hasattr(buffer, 'valid_flags') and len(buffer.valid_flags) > 0:
                            buffer.valid_flags[-1] = False
                            if logger:
                                logger.log_debug("Marked last transition as invalid (MD verification failed)")
                        elif hasattr(buffer, 'states') and len(buffer.states) > 0:
                            if logger:
                                logger.log_warning(
                                    "Buffer has no valid_flags - cannot safely rollback. "
                                    "Transition will remain in buffer as potentially invalid data."
                                )

                    except Exception as buffer_e:
                        if logger:
                            logger.log_warning(f"Could not mark transition as invalid: {buffer_e}")
        else:
            is_done = False
            md_verified_dissociation = False
            _LOG.debug(f" No dissociation after valid NEB step - continuing")

        actual_try_num = getattr(env, 'current_try_num', current_try_num)

        # Valid step accepted: reset try counter for the next step
        env.current_try_num = 0

        if logger:
            if not validate_atom_counts(env, "[After valid step]"):
                logger.log_warning("Atom count validation failed after valid step")

        try:
            if os.path.exists(before_step_state):
                os.remove(before_step_state)
        except:
            pass

        _LOG.debug(f" Returning valid NEB result: pe_reward={pe_reward:.3f}, done={is_done}")

        return (dist_new, embedding_new, pe_reward, is_done, valid_step, activation_energy, actual_try_num, md_verified_dissociation)

    except Exception as neb_e:
        _LOG.info(f" ERROR during NEB calculation: {neb_e}")
        import traceback
        traceback.print_exc()

        if logger:
            logger.log_error(f"ERROR during NEB calculation: {neb_e}")

        try:
            if os.path.exists(before_step_state):
                os.remove(before_step_state)
        except:
            pass

        capped_penalty = _get_capped_invalid_penalty(invalid_step_penalty)
        _LOG.debug(f" Returning error penalty: {capped_penalty:.3f} (cumulative: {env._episode_invalid_penalty_cumulative:.3f})")
        return (dist_new, embedding_new, capped_penalty, False, False, -1.0,
                getattr(env, 'current_try_num', current_try_num), False)

    finally:
        # Remove only temp files; keep curr_state_file (NEB may still need it)
        temp_files = [before_step_state]
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
