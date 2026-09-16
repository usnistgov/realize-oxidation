import os
import time
import torch
import numpy as np
import pickle
import gc
import shutil
import socket
from typing import Optional
from ase import io as ase_io
from realize.environment.lammps_env import O2LammpsEnv
from realize.ppo.ppo_agent import PPO
from realize.training.episode_runner import EpisodeRunner
from realize.training.parallel_collector import ParallelEpisodeCollector
from realize.training.state_normalizer import StateNormalizer
from realize.training.novelty_tracker import NoveltyTracker
from realize.utils.shared_model_manager import SharedModelManager
from realize.config.paths import get_structure_file_path, get_temp_dir, get_embedding_checkpoint_path, get_results_dir
from realize.training.trainer_distributed import PPOTrainerDistributedMixin
from realize.training.trainer_checkpoint import PPOTrainerCheckpointMixin
from realize.training.trainer_diagnostics import PPOTrainerDiagnosticsMixin
from realize.training.trainer_envreset import PPOTrainerEnvResetMixin
import logging
_LOG = logging.getLogger(__name__)


class PPOTrainer(
    PPOTrainerDistributedMixin,
    PPOTrainerCheckpointMixin,
    PPOTrainerDiagnosticsMixin,
    PPOTrainerEnvResetMixin,
):
    """Main PPO training coordinator with Distributed Collective PPO support."""

    def _to_numpy(self, x):
        """Convert tensor to numpy, handling nested structures"""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, dict):
            return {k: self._to_numpy(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._to_numpy(item) for item in x]
        return x

    def __init__(self, config, device, logger, worker_id: int = 0, world_size: int = 1):
        self.config = config
        self.device = device
        self.logger = logger
        self.state_normalizer = StateNormalizer(device)

        # Resolve log dir early so all components can use it.
        results_dir = get_results_dir()
        log_dir = os.path.join(results_dir, 'logs')
        experiment_name = getattr(config, 'experiment_name', 'collective_ppo')

        # Check config.runtime for use_shared_model (set by --use-shared-model flag)
        use_shared_model = getattr(config.runtime, 'use_shared_model', False) if hasattr(config, 'runtime') else False

        # Get actual worker_id and world_size from SLURM environment if available
        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        slurm_array_size = os.environ.get('SLURM_ARRAY_TASK_COUNT')
        actual_worker_id = int(slurm_task_id) if slurm_task_id is not None else worker_id
        actual_world_size = int(slurm_array_size) if slurm_array_size is not None else world_size


        # Initialize shared model manager if enabled
        self.shared_model_manager = None
        self.global_episodes = 0

        # Distributed Collective PPO tracking
        self.local_model_version = 0
        self.episodes_since_last_update = 0

        # Initialize data collection for visualization
        self.sh_data = []

        # Diagnostic tracking for failure mode analysis
        self._success_by_o_count = {}  # bucket -> [attempts, successes]
        self._cumulative_o_at_episode = []  # Track O count progression

        # Adaptive midpoint tracking across reset cycles
        self._current_cycle = 0
        self._cycle_ea_effs = []        # Ea_eff for successful dissociations in current cycle
        self._cycle_avg_ea = []         # Average Ea per completed cycle

        # Extract config values (no defaults).
        if hasattr(config, 'runtime'):
            # Structured PPOConfig: validate structure.
            if not hasattr(config, 'runtime'):
                raise ValueError("PPOConfig must have 'runtime' attribute")
            if not hasattr(config.runtime, 'total_episodes'):
                raise ValueError("config.runtime.total_episodes is REQUIRED")
            if not hasattr(config.runtime, 'save_interval'):
                raise ValueError("config.runtime.save_interval is REQUIRED")
            if not hasattr(config.runtime, 'checkpoint_dir'):
                raise ValueError("config.runtime.checkpoint_dir is REQUIRED")

            self.total_episodes = config.runtime.total_episodes
            self.save_interval = config.runtime.save_interval

            self.episodes_per_update = getattr(config.runtime, 'episodes_per_update', 4)

            # Model config required if using symmetric embedding
            if not hasattr(config, 'model'):
                raise ValueError("PPOConfig must have 'model' attribute")
            if not hasattr(config.model, 'use_symmetric_embedding'):
                raise ValueError("config.model.use_symmetric_embedding is REQUIRED")

            self.use_symmetric_embedding = config.model.use_symmetric_embedding
            self.load_checkpoint = config.runtime.resume_path is not None
            # RESUME GATE: resume_path is the single source of truth. When None,
            # auto-discovery of on-disk checkpoints is DISABLED (always start fresh).
            _resume_enabled = self.load_checkpoint
            if not _resume_enabled:
                self.logger.log_info(
                    "  resume_path is null -> RESUME DISABLED. Starting fresh at "
                    "episode 0; skipping all checkpoint discovery."
                )

            # Discover checkpoints in priority order. If one exists but cannot
            # load, fail hard (no silent fallback). Discovery only finds
            # checkpoints; loading happens after agent initialization.
            checkpoint_loaded = False
            start_episode = 0
            checkpoint_source = None
            self._pending_checkpoint = None

            self.logger.log_info(
                "Resolving checkpoint to resume from (checked in priority order):"
            )

            #  1: Check shared directory (highest priority)
            if _resume_enabled and config.runtime.use_shared_model:
                # Use spherical checkpoint filename to match SharedModelManager
                shared_checkpoint_path = os.path.join(
                    config.runtime.shared_model_dir,
                    "ppo_spherical_checkpoint.pth"
                )

                if os.path.exists(shared_checkpoint_path):
                    self.logger.log_info(
                        f" 1: Found checkpoint in shared directory: "
                        f"{shared_checkpoint_path}"
                    )

                    try:
                        # Don't apply to agent yet (it doesn't exist). weights_only=False:
                        # checkpoints carry full Python state (numpy RNG, optimizer); trusted
                        # in-house file, required since PyTorch 2.6 changed the default.
                        checkpoint = torch.load(shared_checkpoint_path, map_location=device,
                                                weights_only=False)

                        if checkpoint is None:
                            raise RuntimeError(
                                "torch.load returned None - checkpoint file may be corrupted"
                            )

                        # Validate checkpoint structure
                        if 'policy_state_dict' not in checkpoint:
                            raise RuntimeError("Checkpoint missing 'policy_state_dict'")
                        if 'optimizer_state_dict' not in checkpoint:
                            raise RuntimeError("Checkpoint missing 'optimizer_state_dict'")

                        # Store for loading after agent init
                        self._pending_checkpoint = checkpoint
                        start_episode = checkpoint.get('global_episodes', 0)
                        checkpoint_loaded = True
                        checkpoint_source = "shared_dir"

                        self.logger.log_info(
                            f" 1 SUCCESS: "
                            f"Found valid checkpoint. Will apply after agent initialization. "
                            f"Resuming from episode {start_episode}"
                        )

                    except Exception as e:
                        # Checkpoint exists but cannot load: fail hard.
                        raise RuntimeError(
                            f" 1 FAILED: "
                            f"Shared checkpoint exists at {shared_checkpoint_path} but cannot be loaded. "
                            f"Error: {e}. "
                            f"This is a critical failure - checkpoint may be corrupted. "
                            f"Either fix the checkpoint or manually delete it to start fresh."
                        ) from e
                else:
                    self.logger.log_info(
                        f" 1: No checkpoint in shared directory "
                        f"(checked: {shared_checkpoint_path})"
                    )
            else:
                self.logger.log_info(
                    " 1: Skipped (use_shared_model=False)"
                )

            #  2: Local checkpoint directory. Searches both ./checkpoints/ and
            # results/checkpoints/, accepting ppo_checkpoint_ep{N}.pt and
            # checkpoint_ep{N}.pth filename conventions.
            if _resume_enabled and not checkpoint_loaded:
                import re as _re_ckpt
                try:
                    from realize.config.paths import get_results_dir as _get_results_dir
                except ImportError:
                    _get_results_dir = None

                # Build ordered, deduplicated list of candidate directories
                _ckpt_dirs = []
                _seen = set()
                for _d in (config.runtime.checkpoint_dir,
                           (os.path.join(_get_results_dir(), 'checkpoints')
                            if _get_results_dir is not None else None)):
                    if _d is None:
                        continue
                    _abs = os.path.abspath(_d)
                    if _abs in _seen:
                        continue
                    _seen.add(_abs)
                    if os.path.isdir(_d):
                        _ckpt_dirs.append(_d)

                local_checkpoint_dir = _ckpt_dirs[0] if _ckpt_dirs else config.runtime.checkpoint_dir

                if os.path.exists(local_checkpoint_dir):
                    # Accept ppo_checkpoint_ep{N}.pt and checkpoint_ep{N}.pth
                    # (final_checkpoint.pth excluded).
                    _ckpt_pat = _re_ckpt.compile(
                        r'^(?:ppo_)?checkpoint_(?:ep)?(\d+)\.(?:pt|pth)$'
                    )

                    # Collect (episode, full_path) across all candidate dirs
                    _candidates = []
                    for _d in _ckpt_dirs:
                        for _f in os.listdir(_d):
                            _m = _ckpt_pat.match(_f)
                            if _m is None:
                                continue
                            try:
                                _ep = int(_m.group(1))
                            except ValueError:
                                continue
                            _candidates.append((_ep, os.path.join(_d, _f)))

                    checkpoint_files = [os.path.basename(p) for _e, p in _candidates]

                    if checkpoint_files:
                        _candidates.sort(key=lambda t: t[0], reverse=True)
                        latest_checkpoint = _candidates[0][1]
                        self.logger.log_info(
                            f"  [CKPT-DISCOVERY] Scanned {len(_ckpt_dirs)} dir(s); "
                            f"{len(_candidates)} checkpoint(s) found; "
                            f"selected ep{_candidates[0][0]}: {latest_checkpoint}"
                        )

                        self.logger.log_info(
                            f" 2: Found local checkpoint: {latest_checkpoint}"
                        )

                        try:
                            # weights_only=False: checkpoints carry full Python state
                            # (numpy RNG, optimizer); required since PyTorch 2.6.
                            checkpoint = torch.load(latest_checkpoint, map_location=device,
                                                    weights_only=False)

                            # Validate checkpoint structure
                            if 'policy_state_dict' not in checkpoint:
                                raise RuntimeError("Checkpoint missing 'policy_state_dict'")
                            if 'optimizer_state_dict' not in checkpoint:
                                raise RuntimeError("Checkpoint missing 'optimizer_state_dict'")

                            # Store for loading after agent init
                            self._pending_checkpoint = checkpoint
                            start_episode = checkpoint.get('episode', 0) + 1
                            checkpoint_loaded = True
                            checkpoint_source = "local_checkpoints"

                            self.logger.log_info(
                                f" 2 SUCCESS: "
                                f"Found valid checkpoint. Will apply after agent initialization. "
                                f"Resuming from episode {start_episode}"
                            )

                        except Exception as e:
                            raise RuntimeError(
                                f" 2 FAILED: "
                                f"Local checkpoint exists at {latest_checkpoint} but cannot be loaded. "
                                f"Error: {e}. "
                                f"Either fix the checkpoint or manually delete it to start fresh."
                            ) from e
                    else:
                        self.logger.log_info(
                            f" 2: No checkpoint files in {local_checkpoint_dir}"
                        )
                else:
                    self.logger.log_info(
                        f" 2: Directory does not exist "
                        f"(checked: {local_checkpoint_dir})"
                    )

            #  3: Start from scratch
            if not checkpoint_loaded:
                self.logger.log_info(
                    " 3: No checkpoints found anywhere. "
                    "Starting training from scratch."
                )
                start_episode = 0
                checkpoint_source = "from_scratch"
                self._pending_checkpoint = None

            # Store discovery results
            self.start_episode = start_episode
            self.checkpoint_source = checkpoint_source

        else:
            # Flat config: all attributes must be present (no defaults).
            self.checkpoint_source = "flat_config"
            self._pending_checkpoint = None
            required_attrs = ['total_episodes', 'save_interval', 'use_symmetric_embedding',
                            'load_checkpoint', 'checkpoint_dir']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise ValueError(
                        f"Config missing required attribute '{attr}'. "
                        f"All of {required_attrs} must be explicitly set."
                    )

            self.total_episodes = config.total_episodes
            self.save_interval = config.save_interval
            self.use_symmetric_embedding = config.use_symmetric_embedding
            self.load_checkpoint = config.load_checkpoint
            self.checkpoint_dir = config.checkpoint_dir

        # Validate values
        if self.total_episodes <= 0:
            raise ValueError(f"total_episodes must be > 0, got {self.total_episodes}")
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be > 0, got {self.save_interval}")

        # Initialize novelty tracker (embedding_dim auto-detected on first state)
        self.novelty_tracker = NoveltyTracker(
            threshold=0.85,
            use_equivariant=self.use_symmetric_embedding
        )

        # Initialize environment
        self.env = O2LammpsEnv(
            data_file=get_structure_file_path(),
            coo_file=os.path.join(get_temp_dir(), "coo.data"),
            logger=logger
        )

        # Initialize agent
        self.agent = PPO(config=config, env=self.env, device=device)

        # Now initialize and load shared model if enabled
        use_shared = False
        if hasattr(self.config, 'runtime'):
            use_shared = getattr(self.config.runtime, 'use_shared_model', False)
        else:
            use_shared = getattr(self.config, 'use_shared_model', False)

        if use_shared:
            self.logger.log_info("Initializing SharedModelManager for DISTRIBUTED COLLECTIVE PPO")

            # Get shared_model_dir (NO DEFAULT)
            if hasattr(self.config, 'runtime'):
                if not hasattr(self.config.runtime, 'shared_model_dir'):
                    raise ValueError(
                        "use_shared_model=True requires config.runtime.shared_model_dir to be set"
                    )
                shared_dir = self.config.runtime.shared_model_dir
            else:
                if not hasattr(self.config, 'shared_model_dir'):
                    raise ValueError(
                        "use_shared_model=True requires config.shared_model_dir to be set"
                    )
                shared_dir = self.config.shared_model_dir

            # Validate directory path
            if not isinstance(shared_dir, str) or not shared_dir:
                raise ValueError(f"shared_model_dir must be a non-empty string, got {shared_dir}")

            # Generate unique worker ID to prevent self-loading
            worker_id = f"{socket.gethostname()}-{os.getpid()}-{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"

            self.shared_model_manager = SharedModelManager(
                shared_dir=shared_dir,
                policy_type='spherical',
                worker_id=worker_id,
                eps_per_update=self.episodes_per_update  # Pass eps_per_update
            )

            self.logger.log_info(
                f"[SHARED MODEL] Initialized for DISTRIBUTED COLLECTIVE PPO\n"
                f"  Shared directory: {shared_dir}\n"
                f"  Worker ID: {worker_id}\n"
                f"  Episodes per update: {self.episodes_per_update}"
            )

        # Set logger in agent (method must exist)
        if not hasattr(self.agent, 'set_logger'):
            raise AttributeError(
                "Agent must have 'set_logger' method. "
                "Direct logger assignment is not supported."
            )
        self.agent.set_logger(logger)

        # Episode runner
        self.episode_runner = EpisodeRunner(
            env=self.env,
            agent=self.agent,
            config=config,
            state_normalizer=self.state_normalizer,
            logger=logger,
            device=self.device
        )

        # Attach dual normalizers to agent for PPO updates
        actor_norm, _ = self.episode_runner.get_normalizers()
        self.agent.actor_state_normalizer = actor_norm
        # critic_state_normalizer removed -- CriticEmbedding handles critic inputs directly
        _LOG.info(f"[Trainer] Attached actor normalizer from episode_runner to agent")

        # Wire up episode_runner reference to PPOOptimizer for fail-fast flag
        if hasattr(self.agent, 'optimizer_manager'):
            self.agent.optimizer_manager.episode_runner = self.episode_runner
            _LOG.info(f"[Trainer] Attached episode_runner reference to PPOOptimizer")

        # ---- Parallel episode collection ----
        self.parallel_collector = None
        num_workers = 1
        if hasattr(config, 'runtime'):
            num_workers = getattr(config.runtime, 'num_parallel_workers', 1)
        if num_workers > 1:
            cores_per_worker = getattr(config.runtime, 'cores_per_worker', 7)
            self.parallel_collector = ParallelEpisodeCollector(
                num_workers=num_workers,
                cores_per_worker=cores_per_worker,
                config=config,
                trainer_ref=self,
                device=self.device,
            )
            _LOG.info(f"[Trainer] Parallel collection: {num_workers} workers x "
                  f"{cores_per_worker} cores/worker")

        # Inject dual backbone parameters; must precede any checkpoint/shared load.
        if hasattr(self.episode_runner, 'actor_embedding_extractor'):
            self.agent.inject_backbone_parameters(
                actor_extractor=self.episode_runner.actor_embedding_extractor,
                critic_extractor=self.episode_runner.critic_embedding_extractor
            )
            _LOG.info("[Trainer] Dual backbone parameters injected into agent")
        elif hasattr(self.episode_runner, 'embedding_extractor'):
            self.agent.inject_backbone_parameters(
                actor_extractor=self.episode_runner.embedding_extractor
            )
            _LOG.info("[Trainer] Single backbone injected (single-backbone mode)")

        if self.shared_model_manager is not None:
            self._initialize_or_sync_shared_model()

        # Apply checkpoint after inject_backbone_parameters() so the optimizer
        # has all parameter groups (including the backbone).
        if hasattr(self, '_pending_checkpoint') and self._pending_checkpoint is not None:
            checkpoint = self._pending_checkpoint

            self.logger.log_info(
                f"[CHECKPOINT LOADING] Applying checkpoint from {self.checkpoint_source}..."
            )

            try:
                saved_policy = checkpoint['policy_state_dict']
                self.agent.policy.load_state_dict(saved_policy)
                if hasattr(self.agent, 'policy_old'):
                    self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())
                self.logger.log_info("  \u2713 Policy state loaded (policy_old synced)")
                # Load adaptive reward scaler if present
                if 'reward_scaler_state' in checkpoint and hasattr(self.episode_runner, 'reward_scaler'):
                    self.episode_runner.reward_scaler.load_state_dict(checkpoint['reward_scaler_state'])
                    _LOG.info(f"[CHECKPOINT] Loaded reward scaler: ea_weight={self.episode_runner.reward_scaler.ea_weight:.2f}")

                # Restore trainer cycle bookkeeping.
                if 'cycle_state_full' in checkpoint:
                    _cs = checkpoint['cycle_state_full']
                    self._current_cycle            = int(_cs['current_cycle'])
                    self._cycle_ea_effs            = list(_cs.get('cycle_ea_effs', []))
                    self._cycle_avg_ea             = list(_cs.get('cycle_avg_ea',  []))
                    self.config.ea_sigmoid_midpoint = float(_cs['ea_sigmoid_midpoint'])
                    if 'cumulative_dissociated_o_count' in _cs:
                        _cdo = int(_cs['cumulative_dissociated_o_count'])
                        self.episode_runner.env.cumulative_dissociated_o_count = _cdo
                        if hasattr(self.episode_runner.env, 'structure_manager'):
                            self.episode_runner.env.structure_manager.cumulative_dissociated_o_count = _cdo
                        self.logger.log_info(
                            f"  [RESUME] cumulative_dissociated_o_count restored: {_cdo}"
                        )
                    self.logger.log_info(
                        f"  [RESUME] cycle={self._current_cycle}, "
                        f"midpoint={self.config.ea_sigmoid_midpoint:.4f} eV, "
                        f"in-progress Ea samples={len(self._cycle_ea_effs)}, "
                        f"cycle history len={len(self._cycle_avg_ea)}"
                    )
                else:
                    self.logger.log_info(
                        "  [RESUME] WARNING: checkpoint has no 'cycle_state_full' "
                        "(pre-patch format). Cycle counter and midpoint use config defaults."
                    )

                # Restore RNG state. Non-fatal: RNG state is host-side metadata, so a
                # malformed or device-transferred blob must never abort an otherwise-valid
                # resume. torch.load(map_location=cuda) moves the saved CPU ByteTensor onto
                # CUDA, but torch.set_rng_state requires a CPU uint8 ByteTensor, so coerce.
                if 'rng_state' in checkpoint:
                    import random as _rnd_mod
                    import numpy as _np_mod
                    _rs = checkpoint['rng_state']
                    try:
                        _torch_rng = _rs['torch']
                        if torch.is_tensor(_torch_rng):
                            _torch_rng = _torch_rng.cpu().to(torch.uint8)
                        else:
                            _torch_rng = torch.tensor(_torch_rng, dtype=torch.uint8)
                        torch.set_rng_state(_torch_rng)
                        if _rs.get('torch_cuda') is not None and torch.cuda.is_available():
                            _cuda_rng = [
                                (s.cpu().to(torch.uint8) if torch.is_tensor(s)
                                 else torch.tensor(s, dtype=torch.uint8))
                                for s in _rs['torch_cuda']
                            ]
                            torch.cuda.set_rng_state_all(_cuda_rng)
                        _np_mod.random.set_state(_rs['numpy'])
                        _rnd_mod.setstate(_rs['python'])
                        self.logger.log_info("  [RESUME] RNG state restored (torch/cuda/numpy/python)")
                    except Exception as _rng_err:
                        self.logger.log_info(
                            f"  [RESUME] WARNING: RNG state restore skipped "
                            f"({type(_rng_err).__name__}: {_rng_err}); continuing without RNG "
                            f"resume (sampling will diverge but training is unaffected)."
                        )
                else:
                    self.logger.log_info("  [RESUME] WARNING: no 'rng_state' -- sampling will diverge from original trajectory.")

                # Restore scheduler attributes
                if 'scheduler_state_attrs' in checkpoint:
                    _sched = (getattr(self.agent, 'scheduler', None)
                              or getattr(self.agent, 'temperature_scheduler', None))
                    if _sched is not None:
                        _restored = []
                        for _k, _v in checkpoint['scheduler_state_attrs'].items():
                            if hasattr(_sched, _k):
                                setattr(_sched, _k,
                                        list(_v) if isinstance(_v, list) else _v)
                                _restored.append(_k)
                        self.logger.log_info(
                            f"  [RESUME] Scheduler attrs restored: {_restored}"
                        )

                # Restore substrate. Two layouts:
                #   single-worker : tmp/pre_episode_substrate_ep*.data    -> tmp/coo.data
                #   multi-worker  : tmp/run{N}/pre_episode_substrate_*    -> tmp/run{N}/coo.data
                # No fallback to a pristine structure file.
                import glob as _glob_mod
                import shutil as _shutil_mod
                import re as _re_mod
                _temp_dir = get_temp_dir()
                _resume_target_ep = max(0, self.start_episode - 1)

                def _pick_best_substrate(_files, _max_ep=None):
                    _b, _be = None, -1
                    for _f in _files:
                        _mo = _re_mod.search(r'pre_episode_substrate_ep(\d+)\.data$', _f)
                        if not _mo:
                            continue
                        _ep = int(_mo.group(1))
                        if _max_ep is not None and _ep > _max_ep:
                            continue
                        if _ep > _be:
                            _be, _b = _ep, _f
                    return _b, _be

                _num_parallel = int(getattr(self.config.runtime,
                                            'num_parallel_workers', 1))
                if _num_parallel <= 1:
                    # Single-worker path.
                    _substrate_files = _glob_mod.glob(
                        os.path.join(_temp_dir, 'pre_episode_substrate_ep*.data'))
                    if not _substrate_files:
                        raise RuntimeError(
                            f"[RESUME/single] No pre_episode_substrate_ep*.data in {_temp_dir}. "
                            f"Substrate evolution would be lost on resume."
                        )
                    _best_match, _best_ep = _pick_best_substrate(
                        _substrate_files, _max_ep=_resume_target_ep)
                    if _best_match is None:
                        _avail = sorted(int(_re_mod.search(r'ep(\d+)', _s).group(1))
                                        for _s in _substrate_files
                                        if _re_mod.search(r'ep(\d+)', _s))
                        raise RuntimeError(
                            f"[RESUME/single] Need pre_episode_substrate_ep<={_resume_target_ep}.data "
                            f"but available episodes are: {_avail[:20]}"
                            f"{'...' if len(_avail) > 20 else ''}"
                        )
                    _coo_path = self.episode_runner.env.coo_file
                    _shutil_mod.copy2(_best_match, _coo_path)
                    self.logger.log_info(
                        f"  [RESUME/single] Substrate restored: "
                        f"{os.path.basename(_best_match)} -> coo_file "
                        f"(ep{_best_ep} snapshot, resuming at ep{self.start_episode})"
                    )
                else:
                    # Multi-worker path. Each run{N}/ holds its own substrates.
                    # Workers spawn after this __init__, so seeding run{N}/coo.data
                    # here ensures LAMMPS picks up restored state on first read_data.
                    _run_dirs = sorted(
                        _d for _d in _glob_mod.glob(os.path.join(_temp_dir, 'run*'))
                        if os.path.isdir(_d)
                    )
                    if not _run_dirs:
                        raise RuntimeError(
                            f"[RESUME/parallel] num_parallel_workers={_num_parallel} but "
                            f"no run*/ subdirs in {_temp_dir}. Worker substrates would be lost."
                        )
                    _worker_restores = []
                    _worker_skipped  = []
                    _global_best, _global_best_ep = None, -1
                    for _rd in _run_dirs:
                        _rsubs = _glob_mod.glob(
                            os.path.join(_rd, 'pre_episode_substrate_ep*.data'))
                        if not _rsubs:
                            _worker_skipped.append(os.path.basename(_rd))
                            continue
                        # Per-worker counters may not align with global episode; take
                        # the latest substrate in the worker's dir without a ceiling.
                        _wb, _wb_ep = _pick_best_substrate(_rsubs, _max_ep=None)
                        if _wb is None:
                            _worker_skipped.append(os.path.basename(_rd))
                            continue
                        _worker_coo = os.path.join(_rd, 'coo.data')
                        _shutil_mod.copy2(_wb, _worker_coo)
                        _worker_restores.append(
                            (os.path.basename(_rd), _wb_ep))
                        if _wb_ep > _global_best_ep:
                            _global_best, _global_best_ep = _wb, _wb_ep
                    if not _worker_restores:
                        raise RuntimeError(
                            f"[RESUME/parallel] No substrates found in any run*/ subdir. "
                            f"Cannot resume in parallel mode."
                        )
                    # Seed the main trainer's coo.data from the newest worker
                    # substrate (used for normalizer init / non-rollout env ops,
                    # not rollouts) to avoid stale state.
                    _main_coo = self.episode_runner.env.coo_file
                    if _global_best is not None:
                        _shutil_mod.copy2(_global_best, _main_coo)
                    self.logger.log_info(
                        f"  [RESUME/parallel] Workers restored ({len(_worker_restores)}/{_num_parallel}): "
                        f"{_worker_restores}"
                    )
                    if _worker_skipped:
                        self.logger.log_info(
                            f"  [RESUME/parallel] Workers with no substrate (will use "
                            f"existing coo.data): {_worker_skipped}"
                        )
                    self.logger.log_info(
                        f"  [RESUME/parallel] Main trainer env seeded from ep{_global_best_ep}"
                    )
                # Load optimizer state
                if hasattr(self.agent, 'optimizer_manager'):
                    self.agent.optimizer_manager.optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict']
                    )

                    # Move optimizer state to device
                    for state in self.agent.optimizer_manager.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)

                    self.logger.log_info("  OK Optimizer state loaded")
                elif hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.log_info("  OK Optimizer state loaded")
                else:
                    raise RuntimeError(
                        "Agent has no optimizer to load state into. "
                        "Expected 'optimizer' or 'optimizer_manager.optimizer'"
                    )

                # Load training rewards if present
                if 'training_rewards' in checkpoint and hasattr(self.agent, 'training_rewards'):
                    self.agent.training_rewards = checkpoint['training_rewards']
                    self.logger.log_info(f"  Training history loaded ({len(checkpoint['training_rewards'])} episodes)")

                self.logger.log_info(
                    f"[CHECKPOINT LOADING] Successfully applied checkpoint from {self.checkpoint_source}\n"
                )

            except Exception as e:
                raise RuntimeError(
                    f"[CHECKPOINT LOADING] FAILED to apply checkpoint from {self.checkpoint_source}. "
                    f"Checkpoint loaded but could not be applied. Likely causes: model/optimizer "
                    f"structure mismatch, or a restore step (substrate/scheduler/cycle state) failed. "
                    f"Error: {e}"
                ) from e
            finally:
                del self._pending_checkpoint

        # Flat-config fallback checkpoint load.
        if not hasattr(self.config, 'runtime'):
            self.start_episode = 0
            if self.load_checkpoint:
                self.start_episode = self._load_checkpoint()


    def train(self):
        """Main training loop with Distributed Collective PPO support."""

        reset_interval = getattr(self.config.runtime, 'reset_surface_every_n_episodes', 0)
        max_o = getattr(self.config.runtime, 'max_cumulative_o_before_reset', 0)

        # ---- Parallel mode: start workers ----
        if self.parallel_collector is not None:
            self.parallel_collector.start()
        _LOG.info(f"[TRAIN] Loop range: start_episode={self.start_episode}, "
              f"total_episodes={self.total_episodes}, "
              f"episodes_per_update={self.episodes_per_update}")

        try:
          for episode in range(self.start_episode, self.total_episodes):
            if self.shared_model_manager is not None:
                self._sync_weights_if_needed()

            # Backbone warmup: freeze backbone for first N episodes,
            # letting the critic converge on stable returns before
            # backbone changes make them non-stationary.
            if hasattr(self.config, 'backbone') and self.config.backbone.train_backbone:
                warmup = self.config.backbone.warm_start_episodes
                if warmup > 0 and hasattr(self.episode_runner, 'actor_embedding_extractor'):
                    if episode == self.start_episode and episode < warmup:
                        self.episode_runner.actor_embedding_extractor.freeze_backbone()
                    elif episode == warmup:
                        _LOG.info(f"\n[BACKBONE WARMUP] Episode {episode}: unfreezing actor backbone")
                        self.episode_runner.actor_embedding_extractor.unfreeze_backbone()

            if self.parallel_collector is not None:
                self._run_episode_parallel(episode)
            else:
                self._run_episode(episode)

            if (self.parallel_collector is None
                    and (episode + 1) % 4 == 0
                    and hasattr(self.episode_runner, 'embedding_probe')):
                self.episode_runner.embedding_probe.analyze()
                self.episode_runner.embedding_probe.save()
            if (episode + 1) % self.save_interval == 0:
                self._save_checkpoint(episode)
            if (episode + 1) % 20 == 0:
                self._cleanup_memory()
        finally:
            if self.parallel_collector is not None:
                self.parallel_collector.shutdown()

        # Final save
        self._save_checkpoint(self.total_episodes - 1, final=True)

        # Save SH data for visualization
        self.save_sh_data()

        # Log PPO run summary
        summary_data = {
            'total_episodes': self.total_episodes,
            'episodes_per_update': self.episodes_per_update,
            'expected_updates': self.total_episodes // self.episodes_per_update
        }

        if self.shared_model_manager is not None:
            dist_summary = self.shared_model_manager.get_training_summary()
            summary_data['global_episodes'] = dist_summary.get('total_episodes', 0)
            summary_data['global_updates'] = dist_summary.get('total_updates', 0)
            summary_data['final_model_version'] = self.local_model_version

        if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, '_optimize_complete_count'):
            summary_data['actual_updates'] = self.agent.optimizer_manager._optimize_complete_count
            summary_data['total_optimizer_steps'] = self.agent.optimizer_manager._total_optimizer_steps

    def _run_episode_parallel(self, episode: int):
        # MEM_MIT: per-episode CUDA memory tracking
        if torch.cuda.is_available() and (episode % 10 == 0):
            _alloc = torch.cuda.memory_allocated() / 1e9
            _resv  = torch.cuda.memory_reserved()  / 1e9
            _peak  = torch.cuda.max_memory_allocated() / 1e9
            _LOG.debug(f"[MEM ep{episode}] alloc={_alloc:.2f} GiB "
                  f"reserved={_resv:.2f} GiB peak={_peak:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        # Collect one asynchronously-run worker episode and trigger PPO updates.
        import sys

        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"EPISODE {episode} STARTED")
        _LOG.info(f"{'='*60}\n")

        try:
            # On first episode, broadcast initial temperature to all workers
            if episode == 0 and self.parallel_collector is not None:
                self._update_temperatures(0)
                init_temp = self.agent.policy.get_current_temperature()
                self.parallel_collector.set_temperature(init_temp)
                # Send initial temperature via weight queues
                import io
                cpu_state = {k: v.detach().cpu().clone() for k, v in self.agent.policy.state_dict().items()}
                buf_io = io.BytesIO()
                torch.save(cpu_state, buf_io)
                payload = (0, buf_io.getvalue(), init_temp)
                for wq in self.parallel_collector.weight_queues:
                    try:
                        wq.put(payload, timeout=10)
                    except Exception:
                        pass
                del cpu_state

            # Collect one completed episode from workers
            episode_data_list = self.parallel_collector.collect_episodes(
                num_episodes=1, timeout_per_episode=1800.0
            )

            if not episode_data_list:
                _LOG.warning(f"[PARALLEL] No episode collected for slot {episode}")
                return

            ep_data = episode_data_list[0]
            worker_id = ep_data.get('worker_id', -1)
            dissociated = ep_data.get('dissociated', False)
            total_reward = ep_data.get('total_reward', 0.0)
            steps = ep_data.get('steps', 0)
            n_transitions = len(ep_data.get('transitions', []))

            _LOG.debug(f"ep={episode}: worker={worker_id}, steps={steps}, "
                  f"transitions={n_transitions}, dissoc={dissociated}, R={total_reward:.3f}")

            self.logger.begin_episode(episode)

            # Replay stored transitions into the main buffer
            buf = self.agent.buffer if hasattr(self.agent, 'buffer') else None
            if buf is None and hasattr(self.agent, 'buffer_manager'):
                buf = self.agent.buffer_manager.buffer

            if buf is not None and 'transitions' in ep_data:
                for ti, t in enumerate(ep_data['transitions']):
                    buf.add(
                        state=t['state'],
                        action_rot=t['action_rot'],
                        trans_sample=t['trans_sample'],
                        log_prob=t['log_prob'],
                        state_value=t['state_value'],
                        reward=t['reward'],
                        is_terminal=t['is_terminal'],
                        is_truncated=t['is_truncated'],
                        log_prob_direction=t.get('log_prob_direction'),
                        log_prob_magnitude=t.get('log_prob_magnitude'),
                        log_prob_axis=t.get('log_prob_axis'),
                        **t.get('kwargs', {})
                    )
                    idx = len(buf.rewards) - 1
                    buf.update_reward(
                        index=idx,
                        reward=t['reward'],
                        is_terminal=t['is_terminal'],
                        is_truncated=t['is_truncated'],
                    )
                buf.mark_episode_end(success=dissociated)

            # Log results
            self.logger.finalize_episode(
                dissociated=dissociated,
                total_reward=total_reward,
                reward_components=ep_data.get('reward_components', {}),
                stats=ep_data.get('stats', {})
            )
            self._log_success_by_difficulty(episode, dissociated)

            # Track Ea_eff for adaptive midpoint
            _ep_ea_eff = ep_data.get('ea_eff')
            if dissociated and _ep_ea_eff is not None:
                self._cycle_ea_effs.append(_ep_ea_eff)

            # PPO update logic
            if steps > 0:
                should_update = (episode + 1) % self.episodes_per_update == 0
                is_last_episode = (episode + 1) >= self.total_episodes

                if should_update or is_last_episode:
                    # Pause workers during PPO update to prevent stale episodes
                    if self.parallel_collector is not None and self.parallel_collector.pause_event is not None:
                        self.parallel_collector.pause_event.clear()

                    if hasattr(self.agent, 'policy'):
                        self.agent.policy.train(True)

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Update temperature BEFORE PPO update and broadcast
                    self._update_temperatures(episode)
                    current_temp = self.agent.policy.get_current_temperature()
                    self.parallel_collector.set_temperature(current_temp)

                    self.agent.update(episode)
                    # [WINDOW_NORMALIZER] flush after PPO update
                    _critic_norm = getattr(getattr(self.agent, 'policy', None), 'critic_input_normalizer', None)
                    if _critic_norm is not None and hasattr(_critic_norm, 'flush_pending_updates'):
                        _critic_norm.flush_pending_updates()

                    self.parallel_collector.broadcast_weights(
                        self.agent.policy.state_dict()
                    )

            self._monitor_training_health(episode)

        except Exception as e:
            _LOG.info(f"[PARALLEL] EXCEPTION in _run_episode_parallel(episode={episode}): {e}")
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"EPISODE {episode} COMPLETE (parallel)")
        _LOG.info(f"{'='*60}\n")

    def _run_episode(self, episode: int):
        """Run single training episode."""

        # Check if environment should be reset
        if self._should_reset_environment(episode):
            self._reset_to_fresh_substrate(episode)

        self.logger.begin_episode(episode)

        # Enable distribution capture for visualization
        if hasattr(self.agent.policy, 'set_capture_distributions'):
            self.agent.policy.set_capture_distributions(True)

        # VALIDATION: Check policy outputs on first episode
        if episode == 0 and not hasattr(self, '_policy_validation_done'):
            _LOG.info(f"\n[TRAINER VALIDATION] Checking policy data flow...")
            self._policy_validation_done = True

        # Update temperatures
        self._update_temperatures(episode)

        # Run episode
        try:
            results = self.episode_runner.run_episode(
                episode=episode,
                novelty_tracker=self.novelty_tracker
            )
        except Exception as e:
            _LOG.info(f"  Exception type: {type(e).__name__}")
            _LOG.info(f"  Exception message: {str(e)}")
            _LOG.info(f"\n  Full traceback:")
            import traceback
            traceback.print_exc()
            raise

        # Log results
        self.logger.finalize_episode(
            dissociated=results['dissociated'],
            total_reward=results['total_reward'],
            reward_components=results['reward_components'],
            stats=results['stats']
        )


        # Track success rate by O accumulation
        self._log_success_by_difficulty(episode, results['dissociated'])

        # Track Ea_eff for adaptive midpoint
        if results.get('dissociated') and results.get('ea_eff') is not None:
            self._cycle_ea_effs.append(results['ea_eff'])

        # UPDATE LOGIC: Distributed Collective PPO vs Single-Worker
        if results['steps'] > 0:

            if self.shared_model_manager is not None:
                # Distributed mode: submit to shared buffer, update at threshold.
                self._handle_distributed_update(episode, results)

            else:
                # Single-worker mode.
                should_update = (episode + 1) % self.episodes_per_update == 0
                is_last_episode = (episode + 1) >= self.total_episodes

                episodes_since_last = (episode + 1) % self.episodes_per_update
                if episodes_since_last == 0:
                    episodes_since_last = self.episodes_per_update

                if should_update or is_last_episode:
                    if hasattr(self.agent, 'policy'):
                        self.agent.policy.train(True)

                    buffer_size = self.agent.get_buffer_length() if hasattr(self.agent, 'get_buffer_length') else 0

                    gc.collect()
                    torch.cuda.empty_cache()

                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9

                    self.agent.update(episode)
                    # [WINDOW_NORMALIZER] flush after PPO update
                    _critic_norm = getattr(getattr(self.agent, 'policy', None), 'critic_input_normalizer', None)
                    if _critic_norm is not None and hasattr(_critic_norm, 'flush_pending_updates'):
                        _critic_norm.flush_pending_updates()

                    if hasattr(self.agent, 'get_buffer_length'):
                        buffer_len_after = self.agent.get_buffer_length()
                        if buffer_len_after > 0:
                            self.logger.log_warning(
                                f"Buffer not cleared after update: length={buffer_len_after}. "
                                f"Verify agent.update() calls buffer.clear()"
                            )
                else:
                    buffer_size = self.agent.get_buffer_length() if hasattr(self.agent, 'get_buffer_length') else 0
                    episodes_until_update = self.episodes_per_update - ((episode + 1) % self.episodes_per_update)

        # Monitor health
        self._monitor_training_health(episode)

        # Collect SH and magnitude data for visualization
        if hasattr(self.agent.policy, 'get_last_outputs'):
            outputs = self.agent.policy.get_last_outputs()

            viz = None
            if hasattr(self.agent.policy, "sh_head") and hasattr(self.agent.policy.sh_head, "get_viz_params"):
                viz = self.agent.policy.sh_head.get_viz_params()

            def _to_np(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return x

            log_mass_c = outputs.get("channel_log_mass_normalized", None)
            log_mass_mix = outputs.get("mixture_log_mass_normalized", None)
            per_channel_probs = outputs.get("per_channel_probs", None)
            mixture_probs = outputs.get("mixture_probs", None)

            sh_coeffs_src = (viz.get("sh_coeffs") if viz and viz.get("sh_coeffs") is not None
                            else outputs.get("sh_coeffs"))
            sh_coeffs_np = _to_np(sh_coeffs_src)
            if sh_coeffs_np is not None and sh_coeffs_np.ndim == 3:
                sh_coeffs_np = sh_coeffs_np[0]

            channel_lmax_values = (outputs.get("channel_lmax_values")
                                  or (viz.get("channel_lmax_values") if viz else None))
            if channel_lmax_values is None and hasattr(self.agent.policy, "sh_head"):
                channel_lmax_values = self.agent.policy.sh_head.channel_lmax_values
            channel_lmax_values = np.array(channel_lmax_values, dtype=int)

            _LOG.info(f"  FINAL channel_lmax_values = {channel_lmax_values}")

            frame = (outputs.get("frame") or (viz.get("frame") if viz else None) or "feature")
            grid_xyz = _to_np(outputs.get("grid_points"))
            grid_w = _to_np(outputs.get("grid_weights"))
            grid_log_w = _to_np(outputs.get("grid_log_weights"))
            betas = _to_np(outputs.get("betas"))
            channel_weights = _to_np(outputs.get("channel_weights"))

            if channel_weights is not None:
                if channel_weights.ndim == 2: channel_weights = channel_weights[0]
                if not (np.isfinite(channel_weights).all() and channel_weights.sum() > 1e-12):
                    channel_weights = np.ones(int(len(channel_weights))) / int(len(channel_weights))
                else:
                    channel_weights = channel_weights / channel_weights.sum()

            log_mass_c = _to_np(log_mass_c)
            log_mass_mix = _to_np(log_mass_mix)
            if log_mass_c is not None and log_mass_c.ndim == 3:
                log_mass_c = log_mass_c[0]
            if log_mass_mix is not None and log_mass_mix.ndim == 2:
                log_mass_mix = log_mass_mix[0]

            per_channel_probs = _to_np(per_channel_probs)
            mixture_probs = _to_np(mixture_probs)
            if per_channel_probs is not None and per_channel_probs.ndim == 3:
                per_channel_probs = per_channel_probs[0]
            if mixture_probs is not None and mixture_probs.ndim == 2:
                mixture_probs = mixture_probs[0]

            R_feat_to_world = _to_np(outputs.get("R_feat_to_world"))
            max_channel_lmax = int(channel_lmax_values.max())
            coeff_sh_dim = (max_channel_lmax + 1) ** 2
            translation_direction = _to_np(outputs.get("translation_direction"))
            magnitude = _to_np(outputs.get("translation_magnitude"))
            step = results.get("stats", {}).get("steps", 0)

            sh_entry = {
                "episode": episode,
                "step": step,
                "frame": frame,
                "R_feat_to_world": R_feat_to_world,
                "grid_xyz": grid_xyz,
                "grid_points": grid_xyz,
                "grid_weights": grid_w,
                "grid_log_weights": grid_log_w,
                "channel_lmax_values": channel_lmax_values,
                "max_channel_lmax": max_channel_lmax,
                "coeff_sh_dim": coeff_sh_dim,
                "num_channels": int(len(channel_lmax_values)),
                "sh_coeffs": sh_coeffs_np,
                "per_channel_log_mass_normalized": log_mass_c,
                "mixture_log_mass_normalized": log_mass_mix,
                "per_channel_probs": per_channel_probs,
                "mixture_probs": mixture_probs,
                "betas": betas,
                "channel_weights": channel_weights,
                "translation_direction": translation_direction,
                "magnitude": magnitude,
                "lmax_global": int(getattr(self.agent.policy, "lmax", max_channel_lmax)),
            }

            self.sh_data.append(sh_entry)

        # Disable distribution capture after episode
        if hasattr(self.agent.policy, 'set_capture_distributions'):
            self.agent.policy.set_capture_distributions(False)

    def _update_temperatures(self, episode: int):
        """Update policy temperatures."""
        if hasattr(self.agent.policy, 'update_temperature'):
            temp = self.agent.policy.update_temperature(episode, self.total_episodes)

            if episode % 10 == 0:
                self.logger.log_debug(f"Temperature: {temp:.4f}")

