"""N persistent worker processes, each in its own subdirectory (./tmp/run1, ...)
with its own LAMMPS environment. Workers run inference on the device passed from
the main process (the shared GPU when CUDA is available, else CPU) and send
completed episode buffer data back to the main process via mp.Queue. Each worker
is pinned to a disjoint slice of the allocated CPUs so the per-worker NEB mpirun
jobs run in parallel instead of contending for the same cores.

RuntimeConfig knobs: num_parallel_workers (e.g. 6), cores_per_worker (e.g. 7) => 42 total.
Size the SBATCH allocation to num_parallel_workers * cores_per_worker cores.
"""

import os, sys, copy, time, gc, logging, traceback
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.multiprocessing as mp

from realize.utils.log_setup import configure_logging

_LOG = logging.getLogger(__name__)


def _worker_main(
    worker_id, cores_per_worker, config_pickle, initial_policy_state,
    episode_queue, weight_queue, shutdown_event, pause_event, worker_base_dir,
    original_temp_dir, worker_device="cpu",
):
    """Persistent worker process with its own subdirectory and LAMMPS environment.

    Loops: run episode -> extract buffer -> send -> repeat.
    """
    import pickle
    worker_tag = f"[Worker {worker_id}]"

    os.makedirs(worker_base_dir, exist_ok=True)
    os.chdir(worker_base_dir)

    # Redirect stdout/stderr before any import can emit a warning. The per-worker
    # log is a persisted run output (the analysis pipeline parses it for parallel
    # runs), so write it under results/run{k}/ rather than scratch tmp/run{k}/ so
    # it survives tmp/ cleanup. Fall back to the scratch dir if the standard
    # BASE/tmp/run{k} layout cannot be resolved.
    _run_tag = os.path.basename(worker_base_dir.rstrip("/\\"))  # "run{k}"
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(worker_base_dir)))
    try:
        _persist_dir = os.path.join(_repo_root, "results", _run_tag)
        os.makedirs(_persist_dir, exist_ok=True)
        _worker_log_path = os.path.join(_persist_dir, f"worker_{worker_id}.log")
    except OSError:
        _worker_log_path = os.path.join(worker_base_dir, f"worker_{worker_id}.log")
    # Append-mode preserves history across worker restarts so legacy
    # cycle_ea_eff / NEB extractors can replay the full episode stream.
    # A SESSION line marks the boundary so analysis tools can segment.
    _log_existed = os.path.exists(_worker_log_path) and os.path.getsize(_worker_log_path) > 0
    worker_stdout = open(_worker_log_path, "a", buffering=1)
    if _log_existed:
        worker_stdout.write(
            f"\n=== WORKER SESSION RESTART worker_id={worker_id} "
            f"pid={os.getpid()} t={time.time():.3f} ===\n"
        )
    sys.stdout = worker_stdout
    sys.stderr = worker_stdout

    # Attach the stdout handler in this (spawned) worker process. The handler
    # writes through a proxy that resolves sys.stdout at each call, so _LOG
    # output follows the per-worker redirect above into worker_{id}.log, exactly
    # as print(..., flush=True) did before.
    configure_logging()

    import warnings
    warnings.filterwarnings("ignore")

    _LOG.info(f"{worker_tag} PID={os.getpid()}, CWD={os.getcwd()}, "
          f"cores={cores_per_worker}")

    # Single-threaded to avoid oversubscription across workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Workers are CPU-only by design (the trainer holds the only GPU
    # SevenNet). Hiding CUDA here prevents any third-party import
    # from silently allocating on the GPU and OOMing the trainer.
    # Must run before any torch.cuda use in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        config = pickle.loads(config_pickle)

        # Refresh the stdout level now that the worker's config is available. The
        # bootstrap configure_logging() above ran at the normal (INFO) default.
        configure_logging(verbose=getattr(config.runtime, 'verbose', False))

        # Redirect global path constants to worker-specific directories.
        # config.paths derives module-level constants (TEMP_DIR, RESULTS_DIR, ...)
        # from BASE_DIR at import time; left shared, workers would collide on files.
        #
        # Two separate trees per worker:
        #   - scratch / LAMMPS read-write -> tmp/run{k}/      (worker_base_dir == CWD)
        #   - persisted run outputs       -> results/run{k}/  (movies, NEB, per-run logs)
        # Outputs live under the global results/ (not inside scratch tmp/) so they
        # survive even when tmp/ is treated as disposable.
        import realize.config.paths as _paths
        _run_tag = os.path.basename(worker_base_dir)  # "run{k}"
        _paths.TEMP_DIR = worker_base_dir
        # BASE_DIR is file-derived and CWD-independent, so this resolves to the
        # repo-root results/ even after the worker chdirs into its scratch dir.
        _paths.RESULTS_DIR = os.path.join(_paths.BASE_DIR, "results", _run_tag)
        _paths.NEB_RESULTS_DIR = os.path.join(_paths.RESULTS_DIR, "neb_results")
        _paths.CHECKPOINT_DIR = os.path.join(_paths.RESULTS_DIR, "checkpoints")
        _paths.LOG_DIR = os.path.join(_paths.RESULTS_DIR, "logs")
        _paths.SH_DATA_DIR = os.path.join(_paths.RESULTS_DIR, "spherical_harmonics")
        _paths.SH_VISUALIZATION_DIR = os.path.join(_paths.SH_DATA_DIR, "visualizations")
        _paths.SH_COEFFICIENTS_DIR = os.path.join(_paths.SH_DATA_DIR, "coefficients")
        _paths.SH_SPHERE_VISUALIZATION_PATH = os.path.join(_paths.SH_VISUALIZATION_DIR, "sphere_plots")
        # Create the two roots a worker always writes into: scratch (LAMMPS
        # read/write) and run output (movies). Optional sub-trees --
        # neb_results/, spherical_harmonics/, logs/, checkpoints/ -- are created
        # on demand by their getters, leaving no empty placeholders.
        for d in [_paths.TEMP_DIR, _paths.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)

        # Per-worker seed for independent exploration, keyed off the configured
        # base (config.runtime.seed) so parallel and serial runs share a base;
        # fall back to config.random_seed or 42 if unavailable.
        if hasattr(config, 'runtime') and hasattr(config.runtime, 'seed'):
            _base_seed = config.runtime.seed
        elif hasattr(config, 'random_seed'):
            _base_seed = config.random_seed
        else:
            _base_seed = 42
        _worker_seed = _base_seed + worker_id * 1000
        import random
        random.seed(_worker_seed)
        np.random.seed(_worker_seed)
        torch.manual_seed(_worker_seed)
        _LOG.info(f"{worker_tag} Seed set to {_worker_seed} (base={_base_seed}, worker_id={worker_id})")

        # Pin this worker to a disjoint slice of the job's allocated CPUs. Without
        # this, every worker's NEB mpirun defaults to binding its ranks to the same
        # low cores, so the workers' NEB jobs serialize onto a handful of cores
        # instead of running in parallel. Must run before the env/NEB is created so
        # the child mpirun inherits the affinity. Requires the SBATCH allocation to
        # provide num_parallel_workers * cores_per_worker cores.
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            try:
                _num_workers = getattr(config.runtime, "num_parallel_workers", 1)
                _allowed = sorted(os.sched_getaffinity(0))
                _per = len(_allowed) // _num_workers if _num_workers > 0 else 0
                if _per >= 1:
                    _my = _allowed[worker_id * _per:(worker_id + 1) * _per]
                    if _my:
                        os.sched_setaffinity(0, set(_my))
                        _LOG.info(f"{worker_tag} Pinned to CPUs {_my}")
                else:
                    _LOG.info(f"{worker_tag} Not pinning CPUs: only {len(_allowed)} "
                              f"cores for {_num_workers} workers -- allocate "
                              f"num_parallel_workers*cores_per_worker cores in SBATCH.")
            except Exception as _aff_e:
                _LOG.info(f"{worker_tag} CPU affinity pin failed: {_aff_e}")

        # INPUT_DIR stays shared: all workers read the same input files.
        _LOG.info(f"{worker_tag} Paths redirected: scratch(TEMP_DIR)={_paths.TEMP_DIR}, "
                  f"outputs(RESULTS_DIR)={_paths.RESULTS_DIR}")
        _setup_worker_files(worker_base_dir, original_temp_dir)

        from realize.config.paths import get_structure_file_path
        from realize.environment.lammps_env import O2LammpsEnv
        from realize.training.episode_runner import EpisodeRunner
        from realize.ppo.ppo_agent import PPO
        from realize.training.state_normalizer import StateNormalizer
        from realize.utils.logger import UnifiedLogger
        # Each worker gets its own log directory to avoid file conflicts.

        # Inference device: use the GPU the main process passed (all workers share
        # the single allocated GPU) when CUDA is available in this worker; else CPU.
        # Episode payloads are converted to CPU numpy before queuing (see _safe_get),
        # so no CUDA tensors ever cross the process boundary.
        if str(worker_device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda")
            _LOG.info(f"{worker_tag} Inference device: cuda:0")
        else:
            device = torch.device("cpu")
            _LOG.info(f"{worker_tag} Inference device: cpu")
        # Per-run logger writes under the run's output tree (results/run{k}/), not scratch.
        worker_log_base = _paths.RESULTS_DIR
        os.makedirs(worker_log_base, exist_ok=True)
        worker_logger = UnifiedLogger(
            base_dir=worker_log_base,
            verbosity=('DEBUG' if getattr(config.runtime, 'verbose', False) else 'INFO')
        )

        # Environment with its own coo_file in the worker dir.
        env = O2LammpsEnv(
            data_file=get_structure_file_path(),
            coo_file=os.path.join(worker_base_dir, "coo.data"),
            logger=worker_logger,
        )
        # Tag env so episode_manager passes num_procs_override to NEBCalculator.
        env._neb_num_procs_override = cores_per_worker

        # CPU-only inference agent.
        agent = PPO(config=config, env=env, device=device)
        _load_policy_weights(agent, initial_policy_state)
        if hasattr(agent, 'policy'):
            agent.policy.eval()
            # Lock temperature so eval mode does not fall back to gate_temp_final;
            # the main process broadcasts the correct value.
            if hasattr(agent.policy, 'sh_head'):
                init_temp = getattr(config.model, 'gate_temp_init', 1.0)
                agent.policy.sh_head.set_temperature(init_temp)
                _LOG.info(f"{worker_tag} Temperature locked to {init_temp}")

        state_normalizer = StateNormalizer(device)
        episode_runner = EpisodeRunner(
            env=env, agent=agent, config=config,
            state_normalizer=state_normalizer,
            logger=worker_logger, device=device,
        )

        # Per-worker surface-reset tracking.
        local_episode_count = 0
        consecutive_failures = 0
        reset_interval = getattr(config.runtime, 'reset_surface_every_n_episodes', 0)
        max_o = getattr(config.runtime, 'max_cumulative_o_before_reset', 0)

        _LOG.info(f"{worker_tag} Init complete. Entering episode loop.")

        # Main episode loop.
        local_weight_version = 0
        frozen_temperature = None  # Set by main-process broadcast.
        # Fixed-schedule midpoint via the cycle state manager.
        from realize.training.cycle_state_manager import CycleStateManager
        _cycle_state_file = os.path.join(original_temp_dir, 'cycle_state.json')
        if hasattr(config, 'reward'):
            _init_mid = config.reward.ea_sigmoid_midpoint
        elif hasattr(config, 'reward_config'):
            _init_mid = config.reward_config.ea_sigmoid_midpoint
        else:
            raise ValueError(
                "Reward config not found: config must expose 'reward' or "
                "'reward_config' with ea_sigmoid_midpoint. Refusing a silent default."
            )
        # Quorum MUST default to the full worker count. With a smaller quorum a
        # cycle advances before every worker has submitted; the non-submitting
        # workers are then force-advanced, which resets their surface and
        # DISCARDS their accumulated _local_cycle_ea_effs (see the
        # 'external_advance' branch below). That loses most of the per-cycle
        # barrier evidence and can leave a later cycle with no data at all,
        # while the midpoint schedule still advances.
        _quorum = getattr(config.runtime, 'cycle_quorum', None)
        if _quorum is None:
            _quorum = getattr(config.runtime, 'num_parallel_workers', 1)
        _quorum = max(1, int(_quorum))
        _cycle_mgr = CycleStateManager(
            state_file=_cycle_state_file,
            quorum=_quorum,
            initial_midpoint=_init_mid,
            worker_id=f'worker_{worker_id}',
        )
        _local_cycle_ea_effs = []
        consecutive_neb_failures = 0  # NEB death-spiral counter.
        while not shutdown_event.is_set():
            # Wait while the main process runs a PPO update.
            if not pause_event.wait(timeout=5.0):
                continue  # Re-check shutdown while waiting.

            # Non-blocking check for weight updates.
            local_weight_version, new_temp = _drain_weight_updates(
                weight_queue, agent, worker_tag, local_weight_version
            )
            if new_temp is not None:
                frozen_temperature = new_temp
                if hasattr(agent, 'policy') and hasattr(agent.policy, 'sh_head'):
                    agent.policy.sh_head.set_temperature(frozen_temperature)
                _LOG.debug(f"{worker_tag} Temperature set to {frozen_temperature:.6f}")

            # Cycle advance triggered by another worker reaching quorum.
            _cyc_advanced, _cyc_midpoint = _cycle_mgr.check_cycle_advanced()
            if _cyc_advanced:
                _update_worker_midpoint(config, _cyc_midpoint, worker_tag)
                _reset_surface(env)
                _local_cycle_ea_effs = []
                _LOG.info(f"{worker_tag} Cycle advanced externally. Reset + midpoint={_cyc_midpoint:.4f}")

            # Surface reset on the O threshold.
            if _should_reset_surface(local_episode_count, env, reset_interval, max_o):
                if not _cycle_mgr.submitted_this_cycle:
                    # First time over threshold: submit data, do not reset yet.
                    _q_met, _q_mid = _cycle_mgr.submit_cycle_data(_local_cycle_ea_effs)
                    _local_cycle_ea_effs = []
                    if _q_met:
                        # This worker triggered quorum: reset now.
                        _update_worker_midpoint(config, _q_mid, worker_tag)
                        _reset_surface(env)
                        _LOG.info(f"{worker_tag} Quorum met! Reset + midpoint={_q_mid:.4f}")
                    else:
                        _LOG.info(f"{worker_tag} Submitted Ea data, waiting for quorum. "
                              f"Continuing on current surface.")
                # Already submitted: keep running on the current surface, no reset.

            # Run one episode.
            try:
                results = episode_runner.run_episode(
                    episode=local_episode_count,
                    novelty_tracker=None,
                )
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                _LOG.info(f"{worker_tag} Episode {local_episode_count} failed "
                      f"({consecutive_failures} consecutive): {e}\n"
                      f"{traceback.format_exc()}")
                # Hard exit on protracted failure -- prevents spinning forever
                # on a structural bug (e.g. corrupt persisted file). The
                # ParallelEpisodeCollector will restart the worker fresh.
                if consecutive_failures >= 50:
                    _LOG.error(f"{worker_tag} {consecutive_failures} consecutive "
                               f"failures -- exiting worker process.")
                    raise
                if consecutive_failures >= 5:
                    _LOG.info(f"{worker_tag} {consecutive_failures} consecutive failures. "
                          f"Resetting to pristine structure and zeroing counters.")
                    try:
                        from realize.config.paths import get_structure_file_path
                        env.structure_manager.initialize_from_file(get_structure_file_path())
                        env.cumulative_dissociated_o_count = 0
                        if hasattr(env, 'structure_manager'):
                            env.structure_manager.cumulative_dissociated_o_count = 0
                        env.previous_episode_dissociated = False
                        env.last_md_verified_dissociation = False
                        env.global_catastrophic_failures = 0
                        env.force_next_step = False
                        if hasattr(env, 'neb_failure_tracker'):
                            env.neb_failure_tracker.clear()
                        if hasattr(env, 'episode_outcomes'):
                            env.episode_outcomes.clear()
                        consecutive_failures = 0
                        _LOG.info(f"{worker_tag} Reset complete. Resuming.")
                    except Exception as reset_e:
                        _LOG.info(f"{worker_tag} Reset failed: {reset_e}")
                local_episode_count += 1
                continue

            # Accumulate ea_eff for the adaptive midpoint, only before submission.
            if not _cycle_mgr.submitted_this_cycle:
                _ep_ea = results.get('ea_eff')
                if results.get('dissociated') and _ep_ea is not None:
                    _local_cycle_ea_effs.append(_ep_ea)

            # Extract buffer data and send to the main process.
            ep_data = _extract_episode_from_buffer(agent, results, worker_id)
            # Drain weight updates before pushing: if the version changed during
            # this episode the data is stale, so discard it locally.
            pre_push_version = local_weight_version
            local_weight_version, new_temp = _drain_weight_updates(
                weight_queue, agent, worker_tag, local_weight_version
            )
            if new_temp is not None:
                frozen_temperature = new_temp
                if hasattr(agent, 'policy') and hasattr(agent.policy, 'sh_head'):
                    agent.policy.sh_head.set_temperature(frozen_temperature)

            if local_weight_version > pre_push_version:
                _LOG.debug(f"{worker_tag} Discarding own stale episode "
                      f"(ran with v{pre_push_version}, now v{local_weight_version})")
                local_episode_count += 1
                continue

            ep_data['weight_version'] = local_weight_version
            ep_data['termination_type'] = results.get('termination_type', '')
            ep_data['dissociated'] = results.get('dissociated', False)

            # NEB death-spiral detection.
            ep_termination = ep_data.get('termination_type', '')
            ep_dissociated = ep_data.get('dissociated', False)
            if ep_dissociated:
                consecutive_neb_failures = 0  # Success resets the counter.
            elif ep_termination in ('NEB_FAILURE',) and not ep_dissociated:
                consecutive_neb_failures += 1
                if consecutive_neb_failures >= 5:
                    # SPEC item 26: a death-spiral is an EPISODE TERMINATION
                    # condition, NOT a cycle boundary (item 27). It must NOT call
                    # _reset_surface (pristine restore + cumO=0), which is reserved
                    # for the 80-O cycle boundary. Zeroing cumO here silently
                    # destroyed cycle progress (388 dissociations -> 0 cycles).
                    # No last-valid-restore primitive exists, so per the no-fallback
                    # rule we do NOT reset the substrate at all: the episode is
                    # already terminated (NEB_FAILURE); the worker proceeds to the
                    # next episode on the current real substrate, cumO preserved.
                    _LOG.info(f"{worker_tag} NEB DEATH SPIRAL: {consecutive_neb_failures} consecutive "
                          f"failures. Terminating episode; substrate and cumO "
                          f"preserved (NOT a cycle boundary).")
                    consecutive_neb_failures = 0
            else:
                # Non-NEB failure (e.g. vacuum escape): neither count nor reset.
                pass
            local_episode_count += 1

            try:
                episode_queue.put(ep_data, timeout=120)
            except Exception as e:
                _LOG.info(f"{worker_tag} Send failed: {e}")
                continue

            _LOG.info(f"{worker_tag} Ep {local_episode_count}: "
                  f"dissoc={results.get('dissociated', False)}, "
                  f"steps={results.get('steps', 0)}, "
                  f"R={results.get('total_reward', 0):.3f}")

            # Clear the worker's local buffer after extraction.
            buf = getattr(agent, 'buffer', None)
            if buf is None and hasattr(agent, 'buffer_manager'):
                buf = getattr(agent.buffer_manager, 'buffer', None)
            if buf is not None:
                buf.clear()

    except Exception as e:
        _LOG.info(f"{worker_tag} CRASHED: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if 'env' in dir() and env is not None:
                if hasattr(env, 'close'):
                    env.close()
                elif hasattr(env, 'lammps_interface') and hasattr(env.lammps_interface, 'close'):
                    env.lammps_interface.close()
        except Exception:
            pass
        _LOG.info(f"{worker_tag} Exiting.")
        # Restore stdout/stderr and close the log file.
        try:
            if 'worker_stdout' in dir() and worker_stdout is not None:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                worker_stdout.close()
        except Exception:
            pass

def _setup_worker_files(worker_dir, original_temp_dir):
    """Copy essential LAMMPS/NEB files into the worker's CWD."""
    import shutil, glob
    from realize.config.paths import INPUT_DIR

    for pattern in ["*.data", "*.lammps", "*.reax", "*.reaxff",
                    "ffield*", "*.pot", "*.control", "coo.data"]:
        for src in glob.glob(os.path.join(original_temp_dir, pattern)):
            dst = os.path.join(worker_dir, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

    for pattern in ["*.in", "*.reax", "*.reaxff", "ffield*", "*.control"]:
        for src in glob.glob(os.path.join(INPUT_DIR, pattern)):
            dst = os.path.join(worker_dir, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass


def _load_policy_weights(agent, state_dict):
    """Load policy weights into agent, handling key mismatches."""
    if hasattr(agent, 'policy'):
        try:
            agent.policy.load_state_dict(state_dict, strict=False)
        except Exception as e:
            _LOG.info(f"  [WARN] Partial weight load: {e}")
            model_dict = agent.policy.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            agent.policy.load_state_dict(model_dict)
            _LOG.info(f"  [WARN] Loaded {len(filtered)}/{len(state_dict)} keys")


def _drain_weight_updates(weight_queue, agent, tag, current_version=0):
    """Load the latest policy weights from queue (non-blocking).
    Returns the updated weight version number."""
    import io
    latest = None
    latest_version = current_version
    latest_temperature = None
    while True:
        try:
            item = weight_queue.get_nowait()
            if isinstance(item, tuple) and len(item) == 3:
                latest_version, latest, latest_temperature = item
            elif isinstance(item, tuple) and len(item) == 2:
                latest_version, latest = item
            else:
                latest = item
        except Exception:
            break
    if latest is not None:
        if isinstance(latest, bytes):
            latest = torch.load(io.BytesIO(latest), map_location='cpu')
        _load_policy_weights(agent, latest)
        if hasattr(agent, 'policy'):
            agent.policy.eval()
        _LOG.debug(f"{tag} Loaded updated weights (v{latest_version}).")
    return latest_version, latest_temperature

def _should_reset_surface(local_ep, env, reset_interval, max_o):
    """Check if surface reset is needed for this worker."""
    if reset_interval > 0 and local_ep > 0 and local_ep % reset_interval == 0:
        return True
    if max_o > 0 and hasattr(env, 'structure_manager'):
        if getattr(env.structure_manager, 'cumulative_dissociated_o_count', 0) >= max_o:
            return True
    return False


def _reset_surface(env):
    """Reset to pristine substrate. CYCLE BOUNDARY ONLY (spec items 25/27)."""
    from realize.config.paths import get_structure_file_path
    # Fail-fast invariant: cumulative_dissociated_o_count is monotonic within a
    # cycle and may be zeroed ONLY at a genuine 80-O cycle boundary. Any other
    # caller (recovery/error/death-spiral) reaching here is a bug that would
    # silently erase cycle progress -- raise instead (ABSOLUTE RULE: no fallback).
    _sm = getattr(env, 'structure_manager', None)
    _max_o = getattr(getattr(env, 'config', None), 'max_cumulative_o_before_reset', None)
    _cur = getattr(_sm, 'cumulative_dissociated_o_count', None) if _sm is not None else None
    if _max_o is not None and _cur is not None and _cur < _max_o:
        raise RuntimeError(
            f"_reset_surface invoked with cumO={_cur} < threshold {_max_o}. "
            f"This is a CYCLE-BOUNDARY operation (item 27) and must not be called "
            f"from a recovery/termination path (e.g. NEB death-spiral).")
    try:
        env.structure_manager.initialize_from_file(get_structure_file_path())
        if hasattr(env.structure_manager, 'cumulative_dissociated_o_count'):
            env.structure_manager.cumulative_dissociated_o_count = 0
    except Exception as e:
        _LOG.info(f"  [WARN] Surface reset failed: {e}")


def _update_worker_midpoint(config, new_midpoint, tag=""):
    """Update ea_sigmoid_midpoint in worker's local config."""
    if hasattr(config, 'reward'):
        old = config.reward.ea_sigmoid_midpoint
        config.reward.ea_sigmoid_midpoint = new_midpoint
    elif hasattr(config, 'reward_config'):
        old = config.reward_config.ea_sigmoid_midpoint
        config.reward_config.ea_sigmoid_midpoint = new_midpoint
    else:
        old = getattr(config, 'ea_sigmoid_midpoint', 1.2)
        config.ea_sigmoid_midpoint = new_midpoint
    _LOG.info(f"{tag} Midpoint updated: {old:.4f} -> {new_midpoint:.4f} eV")


def _extract_episode_from_buffer(agent, results, worker_id):
    """Extract the current episode's transitions from the agent's buffer as a
    plain dict of numpy arrays, picklable via mp.Queue."""
    buf = getattr(agent, 'buffer', None)
    if buf is None and hasattr(agent, 'buffer_manager'):
        buf = getattr(agent.buffer_manager, 'buffer', None)

    transitions = []
    if buf is not None:
        n = len(buf.states)
        # This episode's slice: run_episode already called mark_episode_end, so
        # the last boundary covers our data.
        start_idx = 0
        if buf.episode_boundaries:
            last_start, last_end = buf.episode_boundaries[-1]
            if last_end == n:
                start_idx = last_start
            else:
                start_idx = last_end

        for i in range(start_idx, n):
            t = {
                'state': _safe_get(buf.states, i),
                'actor_state': _safe_get(buf.actor_states, i),
                'action_rot': _safe_get(buf.actions_rot, i),
                'trans_sample': _safe_get(buf.trans_samples, i),
                'log_prob': _safe_get(buf.log_probs, i),
                'state_value': _safe_get(buf.state_values, i),
                'reward': _safe_get(buf.rewards, i),
                'is_terminal': _safe_get(buf.is_terminals, i),
                'is_truncated': _safe_get(buf.is_truncated, i),
                'log_prob_direction': _safe_get(buf.log_probs_direction, i),
                'log_prob_magnitude': _safe_get(buf.log_probs_magnitude, i),
                'log_prob_axis': _safe_get(buf.log_probs_axis, i),
                'kwargs': {
                    'o1_embedding': _safe_get(buf.o1_embeddings, i),
                    'o2_embedding': _safe_get(buf.o2_embeddings, i),
                    'geometry': _safe_get(buf.geometries, i),
                    'o2_embedding_asymmetric': _safe_get(buf.o2_embeddings_asymmetric, i),
                    'env_multipole': _safe_get(buf.env_multipoles, i),
                    'ea_reward': _safe_get(buf.ea_rewards, i, 0.0),
                    'dissociation_reward': _safe_get(buf.dissociation_rewards, i, 0.0),
                    'diffusion_ea_reward': _safe_get(buf.diffusion_ea_rewards, i, 0.0),
                    'dissociation_ea_reward': _safe_get(buf.dissociation_ea_rewards, i, 0.0),
                    'activation_energy': _safe_get(buf.activation_energies, i, 0.0),
                    'axis_reward': _safe_get(buf.axis_rewards, i, 0.0),
                    'sh_coeffs': _safe_get(buf.sh_coefficients, i),
                    'trans_direction': _safe_get(buf.translation_directions, i),
                    'trans_magnitude': _safe_get(buf.translation_magnitudes, i),
                    'temperature': _safe_get(buf.temperatures, i),
                    'num_channels': _safe_get(buf.num_channels_used, i),
                    'channel_weights': _safe_get(buf.channel_weights, i),
                    'sampled_direction': _safe_get(buf.sampled_directions, i),
                    'conditioned_magnitude': _safe_get(buf.conditioned_magnitudes, i),
                },
            }
            transitions.append(t)

    return {
        'worker_id': worker_id,
        'transitions': transitions,
        'dissociated': results.get('dissociated', False),
        'total_reward': results.get('total_reward', 0.0),
        'reward_components': results.get('reward_components', {}),
        'steps': results.get('steps', 0),
        'stats': results.get('stats', {}),
        'ea_eff': results.get('ea_eff'),
    }


def _safe_get(lst, idx, default=None):
    """Safely get element, converting tensors to numpy for pickling."""
    if lst is None or idx >= len(lst):
        return default
    val = lst[idx]
    if torch.is_tensor(val):
        return val.detach().cpu().numpy()
    return val


def _state_dict_to_cpu(state_dict):
    """Deep-copy a state dict with all tensors on CPU."""
    return {
        k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
        for k, v in state_dict.items()
    }


class ParallelEpisodeCollector:
    """Coordinates parallel episode collection across worker processes.

    Each worker runs in ./tmp/run{k+1}/ with its own LAMMPS + NEB.
    """

    def __init__(self, num_workers, cores_per_worker, config, trainer_ref, device):
        self.num_workers = num_workers
        self.cores_per_worker = cores_per_worker
        self.config = config
        self.trainer_ref = trainer_ref
        self.device = device

        import pickle
        self._config_pickle = pickle.dumps(config)

        # Base temp dir: the cwd before workers chdir into their run dirs.
        self._original_temp_dir = os.getcwd()

        self.episode_queue = None
        self.weight_queues = []
        self.shutdown_event = None
        self.workers = []
        self._started = False
        self.weight_version = 0
        self._current_temperature = 1.0
        self.pause_event = None  # Set in start(); workers wait on it before each episode.

    def start(self):
        """Spawn all worker processes."""
        if self._started:
            return

        policy_state = _state_dict_to_cpu(
            self.trainer_ref.agent.policy.state_dict()
        )

        ctx = mp.get_context("spawn")
        self.episode_queue = ctx.Queue()
        self.shutdown_event = ctx.Event()
        self.pause_event = ctx.Event()
        self.pause_event.set()  # Start unpaused so workers run immediately.

        for i in range(self.num_workers):
            wq = ctx.Queue()
            self.weight_queues.append(wq)
            worker_dir = os.path.join(self._original_temp_dir, f"run{i + 1}")

            p = ctx.Process(
                target=_worker_main,
                args=(
                    i, self.cores_per_worker, self._config_pickle,
                    policy_state, self.episode_queue, wq,
                    self.shutdown_event, self.pause_event,
                    worker_dir, self._original_temp_dir, "cpu",
                ),
                name=f"EpWorker-{i}",
                daemon=True,
            )
            self.workers.append(p)

        for p in self.workers:
            p.start()
            _LOG.info(f"[ParallelCollector] Started {p.name} (PID={p.pid})")

        self._started = True
        _LOG.info(f"[ParallelCollector] All {self.num_workers} workers running.")

    def collect_episodes(self, num_episodes, timeout_per_episode=1800.0):
        """Block until num_episodes collected from workers."""
        if not self._started:
            raise RuntimeError("Workers not started. Call .start() first.")

        collected = []
        deadline = time.time() + num_episodes * timeout_per_episode

        while len(collected) < num_episodes:
            # Restart any dead workers.
            for i, p in enumerate(self.workers):
                if not p.is_alive():
                    _LOG.info(f"[ParallelCollector] Worker {i} died (exit={p.exitcode}). Restarting.")
                    self._restart_worker(i)

            remaining = deadline - time.time()
            if remaining <= 0:
                _LOG.info(f"[ParallelCollector] Timeout: {len(collected)}/{num_episodes}")
                break

            try:
                ep_data = self.episode_queue.get(timeout=min(60.0, remaining))
                ep_ver = ep_data.get('weight_version', 0)
                if ep_ver < self.weight_version:
                    _LOG.info(f"[ParallelCollector] Discarding stale episode "
                          f"(v{ep_ver} < current v{self.weight_version})")
                    continue
                collected.append(ep_data)
            except Exception:
                continue

        return collected

    def broadcast_weights(self, policy_state_dict):
        """Send updated policy weights to all workers.

        Serialize to bytes before queuing to prevent segfaults from
        concurrent tensor access between Queue _feed threads and main thread.
        """
        import io
        cpu_state = {k: v.detach().cpu().clone() for k, v in policy_state_dict.items()}
        buf = io.BytesIO()
        torch.save(cpu_state, buf)
        weight_bytes = buf.getvalue()
        del cpu_state, buf
        self.weight_version += 1
        payload = (self.weight_version, weight_bytes, self._current_temperature)
        for i, wq in enumerate(self.weight_queues):
            try:
                # Drain stale updates.
                while True:
                    try:
                        wq.get_nowait()
                    except Exception:
                        break
                wq.put(payload, timeout=10)
            except Exception as e:
                _LOG.info(f"[ParallelCollector] Weight send to worker {i} failed: {e}")
        # Resume workers after the broadcast.
        self.pause_event.set()
        _LOG.debug(f"[ParallelCollector] Broadcasted weights to {self.num_workers} workers. Workers resumed.")

    def set_temperature(self, temperature):
        """Set the temperature that workers should use for the next accumulation window."""
        self._current_temperature = temperature

    def shutdown(self, timeout=30.0):
        """Gracefully shut down all workers."""
        if not self._started:
            return
        _LOG.info("[ParallelCollector] Shutting down...")
        self.shutdown_event.set()

        for p in self.workers:
            p.join(timeout=timeout)
            if p.is_alive():
                _LOG.info(f"[ParallelCollector] Force-killing {p.name}")
                p.kill()
                p.join(timeout=5)

        for q in [self.episode_queue] + self.weight_queues:
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

        self.workers.clear()
        self.weight_queues.clear()
        self._started = False
        _LOG.info("[ParallelCollector] All workers shut down.")

    def _restart_worker(self, idx):
        """Restart a single failed worker."""
        ctx = mp.get_context("spawn")
        wq = self.weight_queues[idx]
        try:
            while True:
                wq.get_nowait()
        except Exception:
            pass

        policy_state = _state_dict_to_cpu(
            self.trainer_ref.agent.policy.state_dict()
        )
        worker_dir = os.path.join(self._original_temp_dir, f"run{idx + 1}")
        p = ctx.Process(
            target=_worker_main,
            args=(
                idx, self.cores_per_worker, self._config_pickle,
                policy_state, self.episode_queue, wq,
                self.shutdown_event, self.pause_event, worker_dir,
                self._original_temp_dir, "cpu",
            ),
            name=f"EpWorker-{idx}",
            daemon=True,
        )
        self.workers[idx] = p
        p.start()
        _LOG.info(f"[ParallelCollector] Restarted Worker {idx} (PID={p.pid})")

    @property
    def alive_workers(self):
        return sum(1 for p in self.workers if p.is_alive())
