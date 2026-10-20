"""Runs a single training episode."""
import torch
import numpy as np
import time
import pickle
import os

from realize.utils.embedding_probe import EmbeddingProbe

from ase import io
from typing import Dict, Any, Optional, List

from realize.environment.episode_manager import reset_episode, step_environment_with_rollback
from realize.config.paths import get_neb_results_dir, get_embedding_checkpoint_path, get_results_dir
from realize.ppo.rewards import calculate_pathway_reward, AdaptiveRewardScaler
from realize.utils.file_utils import standardize_file_format
from realize.networks.encoders.embedding_extractor import EmbeddingExtractor
from realize.networks.encoders.critic_embedding import augment_geometry_with_context
from realize.training.state_normalizer import StateNormalizer
from realize.training.episode_runner_diagnostics import EpisodeDiagnosticsMixin
from realize.training.episode_runner_viz import EpisodeVisualizationMixin
from realize.training.episode_runner_rewards import EpisodeRewardMixin
import logging
_LOG = logging.getLogger(__name__)


class EpisodeRunner(EpisodeDiagnosticsMixin, EpisodeVisualizationMixin, EpisodeRewardMixin):
    """Handles single episode execution."""

    def __init__(self, env, agent, config, state_normalizer, logger, device):
        """Initialize EpisodeRunner.

        device (torch.device) is required; raises ValueError if None.
        """
        if device is None:
            raise ValueError(
                "device parameter is REQUIRED for EpisodeRunner. "
                "Must be torch.device('cuda') or torch.device('cpu'). "
                "No auto-selection allowed."
            )

        self.env = env
        self.agent = agent
        self.config = config
        self.state_normalizer = state_normalizer
        self.logger = logger
        self.device = device
        self.embedding_probe = EmbeddingProbe()

        # Environmental multipole: long-range equivariant directional signal
        from realize.networks.encoders.environmental_multipole import EnvironmentalMultipole
        self.env_multipole = EnvironmentalMultipole(lmax=2, r_cut=9.0, device=device)
        _LOG.info(f"[EpisodeRunner] Environmental multipole initialized: "
              f"lmax=2, output={self.env_multipole.output_dim} dims")

        # Dual backbone: actor trainable (adapts to policy), critic frozen (stable value regression).
        from realize.config.paths import get_embedding_checkpoint_path

        train_backbone = False
        unfreeze_n_layers = 2
        if hasattr(config, 'backbone'):
            train_backbone = config.backbone.train_backbone
            unfreeze_n_layers = config.backbone.get_unfreeze_n_layers()

        # Actor embedding extractor (trainable if enabled)
            self.actor_embedding_extractor = EmbeddingExtractor(
                checkpoint_path=get_embedding_checkpoint_path(),
                device=str(device),
                use_sevennet=True,
                train_backbone=train_backbone,
                unfreeze_last_n_layers=unfreeze_n_layers
            )
            self.actor_embedding_extractor.load_checkpoint()
            self.actor_embedding_extractor.build_model()

            # Critic embedding extractor (configurable freeze/unfreeze)
            critic_train = config.backbone.critic_train_backbone
            critic_unfreeze_n = config.backbone.get_critic_unfreeze_n_layers() if critic_train else 0
            self.critic_embedding_extractor = EmbeddingExtractor(
                checkpoint_path=get_embedding_checkpoint_path(),
                device=str(device),
                use_sevennet=True,
                train_backbone=critic_train,
                unfreeze_last_n_layers=critic_unfreeze_n
            )
            self.critic_embedding_extractor.load_checkpoint()
            self.critic_embedding_extractor.build_model()

            # self.embedding_extractor aliases the actor extractor.
            self.embedding_extractor = self.actor_embedding_extractor

            _LOG.info(f"[EpisodeRunner] Dual Backbones Created:")
            _LOG.info(f"  Actor:  train_backbone={train_backbone}, unfreeze_n_layers={unfreeze_n_layers}")
            _LOG.info(f"  Critic: train_backbone={critic_train}, unfreeze_n_layers={critic_unfreeze_n}")

        # Dual normalizers (actor + critic).
        _LOG.info(f"\n[EpisodeRunner] Creating Dual Normalizers:")

        # Actor normalizer is the provided normalizer.
        self.actor_state_normalizer = state_normalizer

        # Seed actor normalizer with a dummy sample so critic can later copy valid stats.
        if self.actor_state_normalizer.get_statistics() is None:
            _LOG.info(f"  Initializing actor normalizer with dummy sample...")
            try:
                # Use clean data_file, not coo_file (which may hold stretched/dissociated O2 from prior runs).
                if hasattr(self.env, 'data_file') and self.env.data_file is not None:
                    clean_structure_path = self.env.data_file
                    _LOG.info(f"  Using clean structure from: {clean_structure_path}")
                else:
                    clean_structure_path = self.env.coo_file
                    _LOG.info(f"  WARNING: No data_file, using coo_file (may be stale): {clean_structure_path}")

                atoms = io.read(clean_structure_path, format='lammps-data', atom_style='charge')
                atoms.set_pbc((True, True, False))  # PBC x,y true / z false, for mic distance

                # Verify O2 is valid before proceeding
                n = len(atoms)
                symbols = atoms.get_chemical_symbols()
                if symbols[-2:] != ['O', 'O']:
                    raise RuntimeError(f"Structure doesn't end with O2: last two = {symbols[-2:]}")
                d_oo = atoms.get_distance(n-2, n-1, mic=True)
                if not (0.9 <= d_oo <= 2.5):
                    raise RuntimeError(f"O-O distance {d_oo:.3f} Angstrom out of valid range [0.9, 2.5]")
                _LOG.info(f"  Structure validated: n_atoms={n}, d_OO={d_oo:.3f} Angstrom OK")

                # Get embedding
                emb_dict = self.actor_embedding_extractor.get_equivariant_embedding(
                    atoms, return_symmetric_only=False
                )
                sample_embedding = emb_dict['symmetric']
                # normalize() updates statistics internally.
                _ = self.actor_state_normalizer.normalize(sample_embedding)
                actor_stats = self.actor_state_normalizer.get_statistics()
                if actor_stats is not None:
                    _LOG.info(f"  Actor normalizer initialized: count={actor_stats['count']:.0f}, "
                          f"feature_dim={len(actor_stats['mean'])}")
                else:
                    _LOG.info(f"  Actor normalizer still uninitialized after dummy sample")
            except Exception as e:
                _LOG.info(f"  Failed to initialize actor normalizer: {e}")
                _LOG.info(f"  Will initialize on first episode step")

        # Critic normalizer starts temp-frozen (no self-accumulated stats); it copies mature
        # actor stats and freezes permanently once actor has seen the threshold samples.
        self._critic_normalizer_frozen = False
        self._critic_freeze_threshold = 100  # Freeze after actor sees this many samples
        _LOG.info(f"  Actor normalizer:  statistics_update=ENABLED  (adapts as backbone trains)")
        _LOG.info(f"  Critic normalizer: statistics_update=DISABLED (temp) until actor_count>={self._critic_freeze_threshold}")

        self.state_normalizer = self.actor_state_normalizer

        self._normalize_config()

        # Tracking for episode-end reward application
        self.last_valid_buffer_idx = -1

        # Per-episode tracking (reset each episode)
        self.magnitude_stats = []
        self.step_timing_stats = {'neb_steps': 0, 'total_time': 0.0, 'times': []}

        self.channel_usage_history = []

        # Asymmetric O2 embedding for axis prediction.
        self.current_o2_embedding_asymmetric = None

        # Diagnostic tracking for failure mode analysis
        self._failure_modes = {
            'vacuum_escape': 0,      # O2 flew away from surface
            'timeout': 0,            # Reached max_steps without dissociation
            'high_barrier_repeated': 0, # Valid steps but Ea too high
            'invalid_step_heavy': 0,     # Too many invalid NEB steps
            'success': 0             # MD-verified dissociation
        }
        self._o2_final_heights = {'success': [], 'fail': []}
        self._direction_history = []
        self._episode_invalid_count = 0  # Invalid steps per episode

        # Success rate by O accumulation (difficulty analysis)
        self._success_by_o_count = {}  # bucket -> [attempts, successes]

        # Valid step indices for distributed credit assignment.
        self._episode_valid_indices = []

        base_ea = self.config.reward_config.ea_weight if hasattr(self.config, 'reward_config') else self.config.reward.ea_weight
        self.reward_scaler = AdaptiveRewardScaler(
            base_ea_weight=base_ea,
            max_ea_weight=base_ea * 3.0,
            variance_threshold=1.0,
            scale_step=0.5,
            window_size=20,
        )

    def _log(self, category: str, message: str, _verbose: bool = False, **kwargs):
        """Emit a "[CATEGORY] message | k=v, ..." line through the logger.

        _verbose=True (or category=='DEBUG') routes to DEBUG (verbose mode only);
        default is INFO. ERROR/CRITICAL/WARNING always route to their handlers.
        """
        full_msg = message
        if kwargs:
            full_msg += " | " + ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if category in ('ERROR', 'CRITICAL'):
            self.logger.log_error(full_msg)
        elif category == 'WARNING':
            self.logger.log_warning(full_msg)
        elif _verbose or category == 'DEBUG':
            self.logger.log_debug(f"[{category}] {full_msg}")
        else:
            self.logger.log_info(f"[{category}] {full_msg}")

    def _get_neighbor_positions(self) -> Optional[torch.Tensor]:
        """
        Extract neighbor atom positions for the policy.
        """
        atoms = None
        method_found = "None"

        # Try in-memory structure manager.
        if hasattr(self.env, 'structure_manager'):
            try:
                if hasattr(self.env.structure_manager, 'get_atoms'):
                    atoms = self.env.structure_manager.get_atoms()
                    method_found = "structure_manager.get_atoms()"
                elif hasattr(self.env.structure_manager, 'atoms'):
                    atoms = self.env.structure_manager.atoms
                    method_found = "structure_manager.atoms"
            except Exception:
                pass

        # Try env-attached embedding extractor.
        if atoms is None:
            if hasattr(self.env, 'embedding_extractor'):
                atoms = self.env.embedding_extractor.atoms
                method_found = "embedding_extractor.atoms"

        # Fallback: read from disk.
        if atoms is None and hasattr(self.env, 'coo_file'):
            try:
                import ase.io
                atoms = ase.io.read(self.env.coo_file, format='lammps-data', atom_style='charge')
                method_found = "disk_read(coo_file)"
            except Exception as e:
                self._log('DEBUG', f"Disk read failed: {e}")

        # Extract positions and tensorize.
        if atoms is not None:
            try:
                pos = atoms.get_positions()
                if len(pos) > 2:
                    neighbor_pos = pos[:-2] # Exclude O2
                    t = torch.tensor(neighbor_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
                    return t
                else:
                    self._log('DEBUG', f"Atoms found via {method_found}, but too few atoms ({len(pos)})")
            except Exception as e:
                self._log('WARNING', f"Failed to extract positions: {e}")
        else:
             self._log('DEBUG', "Could NOT find atoms object")


        return None
    def _to_numpy_safe(self, x, name: str = "tensor"):
        """Safely convert tensor/array to numpy; returns None for None input, raises on failure."""
        if x is None:
            return None

        if isinstance(x, np.ndarray):
            return x

        if torch.is_tensor(x):
            try:
                return x.detach().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Failed to convert torch tensor '{name}' to numpy: {e}")

        try:
            return np.asarray(x)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert '{name}' (type={type(x)}) to numpy array: {e}"
            )

    def _normalize_config(self):
        """Extract config values; raises ValueError if required fields are missing (no defaults)."""
        if hasattr(self.config, 'reward'):
            # PPOConfig structure - validate all required paths.
            missing = []
            config_paths = {
                'reward.use_neb_reward': ('reward', 'use_neb_reward'),
                'reward.timeout_penalty': ('reward', 'timeout_penalty'),
                'reward.dissociation_weight': ('reward', 'dissociation_weight'),
                'reward.ea_weight': ('reward', 'ea_weight'),
                'runtime.max_steps': ('runtime', 'max_steps')
            }

            for path_str, path_parts in config_paths.items():
                obj = self.config
                for part in path_parts:
                    if not hasattr(obj, part):
                        missing.append(path_str)
                        break
                    obj = getattr(obj, part)

            if missing:
                raise ValueError(
                    f"Missing required config fields in PPOConfig: {missing}. "
                    f"All reward and runtime parameters must be explicitly set."
                )

            self.use_neb_reward = self.config.reward.use_neb_reward
            self.timeout_penalty = self.config.reward.timeout_penalty
            self.dissociation_weight = self.config.reward.dissociation_weight
            self.max_steps = self.config.runtime.max_steps
        else:
            # Flat config structure - all fields must be present (no defaults).
            required = ['use_neb_reward', 'timeout_penalty', 'dissociation_weight', 'max_steps']
            missing = [k for k in required if not hasattr(self.config, k)]

            if missing:
                raise ValueError(
                    f"Missing required flat config fields: {missing}. "
                    f"All fields must be explicitly set: {required}"
                )

            self.use_neb_reward = self.config.use_neb_reward
            self.timeout_penalty = self.config.timeout_penalty
            self.dissociation_weight = self.config.dissociation_weight
            self.max_steps = self.config.max_steps

        # Validate values.
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        if self.timeout_penalty < 0:
            raise ValueError(f"timeout_penalty must be >= 0, got {self.timeout_penalty}")

    def run_episode(self, episode: int,
                    novelty_tracker=None) -> Dict[str, Any]:
        """Run a single episode; returns a dict of episode results."""
        # Episode banner so each episode's output is clearly delimited in the log.
        self.logger.log_info(f"\n{'=' * 70}")
        self.logger.log_info(f"EPISODE {episode}")
        self.logger.log_info(f"{'=' * 70}")

        self.last_valid_buffer_idx = -1

        # Tag all transitions in this episode with the correct episode index.
        self.embedding_probe.start_episode()
        if hasattr(self.agent, 'buffer_manager'):
            self.agent.buffer_manager.set_current_episode(episode)
        elif hasattr(self.agent, 'buffer'):
            if hasattr(self.agent.buffer, 'set_current_episode'):
                self.agent.buffer.set_current_episode(episode)

        # Training mode for PPO: stochastic sampling; policy explores via
        # entropy, temperature, and Dirichlet smoothing.
        if hasattr(self.agent, 'policy'):
            self.agent.policy.train(True)

        state = self._reset_episode(episode)

        # state must not be None - _reset_episode should raise on failure
        if state is None:
            raise RuntimeError(
                f"Episode reset failed: _reset_episode returned None for episode {episode}. "
                f"This indicates a critical failure in environment initialization."
            )

        step = 0
        done = False
        dissociated = False

        # Reset soft-loop flag at episode start (fail-safe).
        self._soft_loop_terminated = False

        # Reset per-episode tracking.
        self._episode_invalid_count = 0
        novel_states = 0
        valid_steps = 0

        # Pathway length = sum of accepted-step translation magnitudes. Carried in the
        # episode_outcomes record so the analysis parser reads it directly instead of
        # re-summing per-step [ACTION]/[STEP] lines.
        episode_path_len = 0.0

        activations = []
        pe_rewards = []

        prev_state_file = self._save_initial_state(episode)

        state_uid = 0

        # Expected state shape/dtype for consistency checking.
        expected_state_shape = state.shape
        expected_state_dtype = state.dtype

        # Valid step indices for distributed credit assignment.
        self._episode_valid_indices = []

        # Pre-bind so the truncation V(s') path in _finalize_episode has a defined
        # value even if the loop exits before env_multipole is computed (e.g. the
        # early catastrophic-failure break). Overwritten by the first iteration.
        env_multipole = None

        while step < self.max_steps and not done:
            # Step divider grouping each step's per-forward output.
            self.logger.log_info(f"\n----- Episode {episode} | Step {step} -----")

            # Safety check: terminate if too many catastrophic failures
            if hasattr(self.env, 'global_catastrophic_failures'):
                if self.env.global_catastrophic_failures >= 5:
                    self.logger.log_error(
                        f"Episode {episode} terminated: {self.env.global_catastrophic_failures} "
                        f"catastrophic NEB failures"
                    )
                    _LOG.info(f"\n[TERMINAL] Episode {episode} aborted due to repeated catastrophic failures\n")
                    done = True
                    break

            # Quick normalizer diagnostic (first 3 episodes, step 10).
            if step == 10 and episode < 3:
                actor_stats = self.actor_state_normalizer.get_statistics()
                self._log('NORMALIZER', "Quick check", _verbose=True,
                          episode=episode, step=step,
                          actor_initialized=actor_stats is not None,
                          actor_count=f"{actor_stats['count']:.0f}" if actor_stats else "N/A",
                          actor_frozen=actor_stats.get('frozen') if actor_stats else "N/A")

            # Periodic normalizer divergence check.
            if step == 10 and episode % 5 == 0:
                actor_stats = self.actor_state_normalizer.get_statistics()

                norm_status = {
                    'episode': episode, 'step': step,
                    'actor_initialized': actor_stats is not None
                }

                if actor_stats is not None:
                    norm_status['actor_count'] = f"{actor_stats['count']:.0f}"
                    norm_status['actor_frozen'] = actor_stats['frozen']
                    norm_status['actor_mean_range'] = f"[{actor_stats['mean'].min():.4f}, {actor_stats['mean'].max():.4f}]"
                    norm_status['actor_std_mean'] = f"{actor_stats['std'].mean():.4f}"
                    norm_status['status'] = "OK"
                self._log('NORMALIZER', "Normalizer check", _verbose=True, **norm_status)

            # Validate state consistency.
            if state.shape != expected_state_shape:
                raise RuntimeError(
                    f"State shape mismatch at episode {episode}, step {step}: "
                    f"expected {expected_state_shape}, got {state.shape}. "
                    f"This indicates inconsistent embedding extraction."
                )
            if state.dtype != expected_state_dtype:
                raise RuntimeError(
                    f"State dtype mismatch at episode {episode}, step {step}: "
                    f"expected {expected_state_dtype}, got {state.dtype}"
                )

            # Check novelty.
            if novelty_tracker:
                is_novel, similarity = novelty_tracker.check_novelty(state)
                _LOG.debug(f"  Novelty check: is_novel={is_novel}, max_similarity={similarity:.3f}")
                if is_novel:
                    novel_states += 1

            buffer_idx_before = self._get_buffer_index()

            neighbor_pos = self._get_neighbor_positions()

            # Persistent state file for replay/training.
            replay_state_path = os.path.join(
                get_results_dir(),
                'neb_results',
                f"state_ep{episode}_step{step}_input.data"
            )
            os.makedirs(os.path.dirname(replay_state_path), exist_ok=True)

            # Copy current physical state to this path
            import shutil
            shutil.copy2(self.env.coo_file, replay_state_path)

            # Extract dual-backbone states: actor (already normalized) and critic.
            atoms_current = self.env.structure_manager.get_atoms()

            # Detach during rollout: only forward-pass embeddings are needed here
            # (gradients are recomputed in PPO). Without detach each embedding holds a
            # backbone computation graph (~500MB+) and 100+ transitions OOMs before PPO.
            with torch.no_grad():
                actor_embedding_raw = self.actor_embedding_extractor.get_equivariant_embedding(
                    atoms_current, return_symmetric_only=True
                )
            # Ensure detached even if get_equivariant_embedding did not detach.
            if torch.is_tensor(actor_embedding_raw):
                actor_embedding_raw = actor_embedding_raw.detach()
            elif isinstance(actor_embedding_raw, dict) and 'symmetric' in actor_embedding_raw:
                actor_embedding_raw = actor_embedding_raw['symmetric'].detach()

            # Extract critic embedding: full dict for CriticEmbedding.
            with torch.no_grad():
                critic_emb_dict = self.critic_embedding_extractor.get_equivariant_embedding(
                    atoms_current, return_symmetric_only=False
                )
            # Critic raw inputs: stored in buffer, recomputed in PPO.
            o1_embedding = critic_emb_dict['o1_embedding'].detach()
            o2_embedding = critic_emb_dict['o2_embedding'].detach()
            geometry_base = critic_emb_dict['geometry'].detach()

            actor_state = self.actor_state_normalizer.normalize(actor_embedding_raw)
            # Store raw embedding for direction diagnostic.
            self.agent.policy._last_raw_sym = actor_embedding_raw.detach()
            # Store first-layer symmetric embedding for probe.
            if 'symmetric_first_layer' in critic_emb_dict:
                self._first_layer_sym = critic_emb_dict['symmetric_first_layer'].detach()
                self.agent.policy._first_layer_sym = self._first_layer_sym


            # State for policy input is the actor state.
            state = actor_state

            # Extract O2 positions for frame alignment.
            atoms = self.env.structure_manager.get_atoms()
            positions = atoms.get_positions()
            o2_positions_np = positions[-2:]  # Last two atoms are O2
            o2_positions_tensor = torch.FloatTensor(o2_positions_np).unsqueeze(0).to(self.device)  # [1, 2, 3]

            # Compute environmental multipole (long-range directional signal).
            with torch.no_grad():
                all_positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
                substrate_positions = all_positions[:-2]  # All atoms except O2
                o2_com = all_positions[-2:].mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, 3] -> [1, 3]
                o2_com = o2_com.squeeze(1)  # [1, 3]
                atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
                substrate_species = self.env_multipole.get_species_idx(atomic_numbers[:-2])
                cell_np = np.array(atoms.get_cell())
                pbc_np = np.array(atoms.get_pbc())
                cell_t = torch.tensor(cell_np, dtype=torch.float32, device=self.device)
                pbc_t = torch.tensor(pbc_np, dtype=torch.bool, device=self.device)
                env_multipole = self.env_multipole(
                    o2_com, substrate_positions, substrate_species, cell_t, pbc_t
                )  # [1, 9]
                self.current_env_multipole = env_multipole.detach().cpu().numpy()

            # Augment geometry with time horizon + multipole invariants for critic.
            geometry = augment_geometry_with_context(
                geometry_base, step, self.max_steps, env_multipole
            )

            action_result = self.agent.select_action(
                state, episode, step, state_uid=state_uid,
                o2_positions=o2_positions_tensor,
                env_multipole=env_multipole,
                o2_embedding_asymmetric=self.current_o2_embedding_asymmetric,
                neighbor_positions=neighbor_pos,
                state_file_path=replay_state_path,
                actor_state=actor_state,
                o1_embedding=o1_embedding,
                o2_embedding=o2_embedding,
                geometry=geometry
            )
            # action_result is (trans_vector, o2_config, valid).
            _trans_vec = action_result[0]  # r * v_hat as numpy
            _trans_dir = torch.tensor(_trans_vec, dtype=torch.float32)
            _trans_dir = _trans_dir / (_trans_dir.norm() + 1e-8)  # normalize to v_hat
            self.embedding_probe.collect_step(
                raw_state=actor_embedding_raw,
                norm_state=actor_state,
                direction=_trans_dir,
                o2_positions=torch.tensor(o2_positions_np, dtype=torch.float32),
            )
            # Collect first-layer embedding for directional signal test.
            if hasattr(self, '_first_layer_sym'):
                self.embedding_probe.collect_first_layer(self._first_layer_sym)

            # Verify exactly one transition was added.
            buffer_idx_after = self._get_buffer_index()
            if buffer_idx_after != buffer_idx_before + 1:
                raise RuntimeError(
                    f"Buffer index mismatch at episode {episode}, step {step}: "
                    f"before={buffer_idx_before}, after={buffer_idx_after}. "
                    f"Expected exactly 1 transition added, got {buffer_idx_after - buffer_idx_before}. "
                    f"This indicates agent.select_action() is storing wrong number of transitions."
                )

            # The transition is now at index buffer_idx_before (last inserted).
            buffer_idx = buffer_idx_before

            # Periodic channel monitoring.
            if step % 10 == 0 and hasattr(self.agent, 'policy') and hasattr(self.agent.policy, '_last_outputs'):
                self._log_channel_snapshot(episode, step)

                # Store in history for episode-level summary.
                try:
                    outputs = self.agent.policy._last_outputs
                    cw = outputs.get('channel_weights')
                    if cw is not None:
                        cw_np = self._to_numpy_safe(cw, "channel_weights")
                        if cw_np is not None:
                            cw_np = np.squeeze(cw_np)
                            if cw_np.ndim == 0:
                                cw_np = np.array([float(cw_np)])

                            self.channel_usage_history.append({
                                'episode': episode,
                                'step': step,
                                'weights': cw_np.copy() if isinstance(cw_np, np.ndarray) else cw_np,
                                'dominant': int(np.argmax(cw_np)) if len(cw_np) > 0 else 0,
                                'entropy': float(-np.sum(cw_np * np.log(cw_np + 1e-10)))
                            })
                except Exception as e:
                    _LOG.info(f"  [WARNING] Failed to store channel history: {e}")

            # Unpack action result before using it.
            if len(action_result) != 3:
                raise ValueError(
                    f"agent.select_action returned {len(action_result)} items, expected 3. "
                    f"This indicates a bug in action selection. "
                    f"Expected: (trans_action, o2_config, buffer_success)"
                )

            trans_action, o2_config, buffer_success = action_result

            # Track sampled direction for adaptation analysis.
            if hasattr(self.agent, 'policy') and hasattr(self.agent.policy, '_last_outputs'):
                trans_dir = self.agent.policy._last_outputs.get('translation_direction')
                if trans_dir is not None:
                    if hasattr(trans_dir, 'detach'):
                        trans_dir_np = trans_dir.detach().cpu().numpy().flatten()[:3]
                    else:
                        trans_dir_np = np.array(trans_dir).flatten()[:3]
                    self._direction_history.append(trans_dir_np.copy())

            trans_mag = None
            step_magnitude = None  # this step's translation magnitude (for path_len)

            # Magnitude from policy's last outputs, else from action_dict.
            if hasattr(self.agent, 'policy') and hasattr(self.agent.policy, '_last_outputs'):
                outputs = self.agent.policy._last_outputs
                trans_mag = outputs.get('translation_magnitude')

                if trans_mag is None and hasattr(self.agent, 'last_action_dict'):
                    trans_mag = self.agent.last_action_dict.get('translation_magnitude')

            if trans_mag is not None:
                if hasattr(trans_mag, 'item'):
                    trans_mag_val = trans_mag.item()
                elif isinstance(trans_mag, np.ndarray):
                    trans_mag_val = float(trans_mag)
                elif torch.is_tensor(trans_mag):
                    trans_mag_val = trans_mag.item()
                else:
                    trans_mag_val = float(trans_mag)

                if not hasattr(self, 'magnitude_stats'):
                    self.magnitude_stats = []
                self.magnitude_stats.append(trans_mag_val)
                step_magnitude = trans_mag_val

            # Buffer update must succeed (no silent continue).
            if not buffer_success:
                raise RuntimeError(
                    f"Buffer update failed (buffer_success=False) at episode {episode}, step {step}. "
                    f"This indicates agent.select_action could not store transition in buffer. "
                    f"Check buffer capacity or buffer manager state."
                )

            # Action dict from agent's last action, else built from policy outputs.
            action_dict = None
            if hasattr(self.agent, 'last_action_dict'):
                action_dict = self.agent.last_action_dict
            elif hasattr(self.agent, 'policy') and hasattr(self.agent.policy, '_last_outputs'):
                outputs = self.agent.policy._last_outputs
                action_dict = {
                    'translation_direction': outputs.get('translation_direction'),
                    'translation_magnitude': outputs.get('translation_magnitude'),
                    'axis_direction': outputs.get('axis_direction')
                }

            # Save SH visualization data.
            if action_dict is not None:
                self._save_sh_visualization_data(episode, step, action_dict)

                if 'translation_direction' in action_dict:
                    td = action_dict['translation_direction']
                    if hasattr(td, 'cpu'):
                        td = td.cpu().numpy().flatten()
                    ad = action_dict.get('axis_direction')
                    if ad is not None and hasattr(ad, 'cpu'):
                        ad = ad.cpu().numpy().flatten()
                    mag = action_dict.get('translation_magnitude')
                    if hasattr(mag, 'item'):
                        mag = mag.item()
                    elif hasattr(mag, 'cpu'):
                        mag = mag.cpu().item()

                    self._log('ACTION', "Sampled action", _verbose=False,
                              direction=f"[{td[0]:+.2f},{td[1]:+.2f},{td[2]:+.2f}]",
                              axis=f"[{ad[0]:+.2f},{ad[1]:+.2f},{ad[2]:+.2f}]",
                              magnitude=f"{mag:.2f}")

            else:
                _LOG.debug(f"No action_dict available for SH saving")

            # Take environment step.
            step_result = self._take_step(
                trans_action, o2_config,
                episode, step, prev_state_file
            )

            if step_result is None:
                _LOG.debug(f"  step_result is None - breaking")
                break

            self._log('STEP', "Step result", _verbose=False,
                      valid=step_result.get('valid'),
                      reward=f"{step_result.get('reward', 0):.3f}",
                      done=step_result.get('done'),
                      dissociated=step_result.get('dissociated'),
                      activation=f"{step_result.get('activation_energy', -1):.3f}")

            # Update state for next iteration.
            if step_result['valid']:
                valid_steps += 1
                self.last_valid_buffer_idx = buffer_idx

                # Track for distributed credit assignment.
                self._episode_valid_indices.append(buffer_idx)

                prev_state_file = step_result.get('state_file', prev_state_file)
                self._log('STEP', f"Valid step #{valid_steps}", _verbose=True,
                          state_file=prev_state_file)

                if step_result.get('activation_energy', -1) >= 0:
                    activations.append(step_result['activation_energy'])
                    self._log('NEB', f"Activation energy recorded", _verbose=True,
                              ea=f"{step_result['activation_energy']:.3f}")

                pe_reward = step_result.get('reward', 0.0)
                if pe_reward != 0.0:
                    pe_rewards.append(pe_reward)
                    self._log('STEP', f"PE reward", _verbose=True, reward=f"{pe_reward:.4f}")
            else:
                self._log('STEP', "Invalid step", _verbose=True)
                self._episode_invalid_count += 1  # Track for failure mode analysis

            # Step rewards only for valid steps (invalid steps are discarded by get_batch()).
            if step_result['valid']:
                step_reward = step_result.get('reward', 0.0)  # PE-based reward

                if step_reward != 0.0:
                    self._update_buffer_reward(
                        buffer_idx,
                        step_reward,
                        done=False,  # Not episode terminal.
                        # Ea/dissociation rewards are applied at episode end only.
                    )
            else:
                _LOG.debug(f"  Skipping _update_buffer_reward for invalid step")

            # Soft-loop (3 consecutive invalid NEB results) = computational failure.
            # Treat as truncation (bootstrap from V(s')) because O2 still exists and the
            # MDP hasn't truly ended; earlier steps toward the surface were valid and
            # only the final action(s) broke the NEB evaluation.
            # No synthetic dummy-action transition (would poison PPO): set done=True,
            # set the flag, break; episode-end logic handles truncation.
            if getattr(self.env, 'force_next_step', False):
                self.env.force_next_step = False

                self._log('WARNING', "Soft-loop detected - TERMINATING EPISODE as TERMINAL failure",
                          episode=episode, step=step,
                          reason="3 consecutive invalid NEB results or catastrophic failures")

                done = True
                # Check O-O distance: NEB may have failed but O2 could still have dissociated.
                try:
                    oo_dist, _ = self.env.check_o2_distance()
                    dissociated = (oo_dist > 1.8)
                    if dissociated:
                        self._log('WARNING', "O2 dissociated despite NEB failure!",
                                  episode=episode, step=step, oo_distance=f"{oo_dist:.3f}")
                except Exception:
                    dissociated = False
                self._soft_loop_terminated = True

                # Episode-end logic then classifies as NEB_FAILURE (truncation) via the
                # _soft_loop_terminated flag, penalizes the last valid step, and sets
                # is_truncated=True so GAE bootstraps from V(s').
                break

            # Check termination.
            done = step_result.get('done', False)
            dissociated = step_result.get('dissociated', False)

            if done and dissociated:
                self.env.previous_episode_dissociated = True

            # Get next state.
            state = step_result.get('next_state')
            if state is not None:
                # Validate shape consistency
                if state.shape != expected_state_shape:
                    raise RuntimeError(
                        f"Next state shape mismatch at episode {episode}, step {step}: "
                        f"expected {expected_state_shape}, got {state.shape}. "
                        f"State is: {type(state)}, dtype: {state.dtype}"
                    )
            else:
                _LOG.debug(f"  Next state is None")
                if not done:
                    raise RuntimeError(
                        f"Critical error: Lost state (None) at episode {episode}, step {step} but episode not done. "
                        f"This indicates environment or embedding extraction failure. "
                        f"Cannot continue episode."
                    )

            # Only valid steps advance the step counter.
            if step_result['valid']:
                step += 1
                state_uid += 1  # New state.
                if step_magnitude is not None:
                    episode_path_len += step_magnitude
                _LOG.debug(f"  Valid step completed - moving to step {step}, state_uid={state_uid}")
            else:
                _LOG.debug(f"  Invalid step - remaining at step {step}, state_uid={state_uid}")

        return self._finalize_episode(
            episode, state, step, done, dissociated,
            activations, pe_rewards, valid_steps, novel_states, env_multipole,
            episode_path_len
        )

    def _finalize_episode(self, episode, state, step, done, dissociated,
                          activations, pe_rewards, valid_steps, novel_states,
                          env_multipole, episode_path_len=0.0):
        # Record episode outcome for next episode's reset.
        if hasattr(self.env, 'record_episode_outcome'):
            self.env.record_episode_outcome(
                episode=episode,
                dissociated=dissociated,  # MD-verified result from step_environment_with_rollback
                o2_removed=getattr(self.env, 'o2_removed', False),
                activation_energies=activations,
                path_len=episode_path_len
            )

        self.env.previous_episode_dissociated = dissociated
        self.embedding_probe.end_episode(dissociated=dissociated)

        # Log failure mode and diagnostic information
        self._log_failure_mode(
            episode=episode,
            success=dissociated,
            o2_removed=getattr(self.env, 'o2_removed', False),
            step_count=step,
            max_steps=self.max_steps,
            invalid_step_count=self._episode_invalid_count
        )
        self._log_o2_final_state(episode, dissociated)
        self._log_success_by_difficulty(episode, dissociated)

        if episode % 20 == 0 and episode > 0:
            self._log_direction_adaptation(episode)

        # Mark episode end for success-weighted buffer sampling: oversamples successful
        # (dissociation) episodes during PPO to learn from sparse rewards. The buffer
        # tracks episode boundaries and success flags.
        if hasattr(self.agent, 'buffer_manager'):
            if hasattr(self.agent.buffer_manager, 'mark_episode_end'):
                self.agent.buffer_manager.mark_episode_end(success=dissociated)
        elif hasattr(self.agent, 'buffer'):
            if hasattr(self.agent.buffer, 'mark_episode_end'):
                self.agent.buffer.mark_episode_end(success=dissociated)

        # Log comprehensive episode summary
        if hasattr(self, 'magnitude_stats') and self.magnitude_stats:
            mags = np.array(self.magnitude_stats)
            mag_summary = f"mean={mags.mean():.3f}, std={mags.std():.3f}, max={mags.max():.3f}"
        else:
            mag_summary = "no data"

        if hasattr(self, 'step_timing_stats'):
            timing_summary = (f"avg_time={self.step_timing_stats['total_time']/max(1, step):.2f}s, "
                             f"NEB_rate={self.step_timing_stats['neb_steps']/max(1, step):.1%}")
        else:
            timing_summary = "no data"

        eval_info = ""

        if episode % 10 == 0:
            self._log_episode_channel_summary(episode)

        total_pe_reward = sum(pe_rewards) if pe_rewards else 0.0
        pe_summary = f"{len(pe_rewards)} rewards, total={total_pe_reward:.4f}"

        self.logger.log_info(
            f"\nEpisode {episode} Summary{eval_info}:\n"
            f"  Steps: {step}/{self.max_steps} (valid: {valid_steps})\n"
            f"  Novelty: {novel_states}/{step} = {100*novel_states/max(1,step):.1f}%\n"
            f"  Dissociated: {dissociated}\n"
            f"  Activations: {activations[:5] if activations else 'none'}\n"
            f"  Per-step PE rewards: {pe_summary}\n"
            f"  Translation magnitudes: {mag_summary}\n"
            f"  Step timing: {timing_summary}"
        )

        # Log success statistics periodically.
        if episode % 10 == 0:
            if hasattr(self.agent, 'buffer_manager') and hasattr(self.agent.buffer_manager, 'get_success_statistics'):
                success_stats = self.agent.buffer_manager.get_success_statistics()
                if success_stats['total_episodes'] > 0:
                    self.logger.log_info(
                        f"  Buffer success stats: {success_stats['success_rate']:.1%} "
                        f"({success_stats['success_count']}/{success_stats['total_episodes']} episodes)"
                    )

        self._track_embedding_divergence(episode)

        # Reset stats for next episode.
        self.magnitude_stats = []
        self.step_timing_stats = {'neb_steps': 0, 'total_time': 0.0, 'times': []}


        # Classify how the episode ended:
        #   TRUE TERMINAL (bootstrap=0): MDP absorbing state -- dissociation, O2 removed /
        #     vacuum escape, catastrophic NEB failure, NEB retry exhausted, other done=True.
        #   TRUNCATION (bootstrap from V(s')): step limit reached without a terminal state.
        #   INVALID: transition not used for training (handled via valid_flags in buffer).
        o2_removed = getattr(self.env, 'o2_removed', False)
        # Catastrophic termination (including soft-loop).
        catastrophic_termination = (
            getattr(self.env, 'global_catastrophic_failures', 0) >= 5 or
            getattr(self, '_soft_loop_terminated', False)
        )

        # Clear the soft-loop flag for next episode
        self._soft_loop_terminated = False
        neb_retry_exhausted = getattr(self.env, 'current_try_num', 0) >= 5

        # Timeout is ONLY when we hit max_steps.
        timed_out = (step >= self.max_steps)

        # Terminal vs truncation decision:
        #   TRUE TERMINAL (bootstrap=0): MDP absorbing states --
        #     dissociated (success), o2_removed (escaped reactive zone, MDP boundary),
        #     NEB failure: reward evaluator broken/unreliable (failure).
        #   TRUNCATION (bootstrap from V(s')): timed_out -- compute budget only, could continue.
        # Truncation = "stopped for compute reasons";
        # Terminal = "episode ends by MDP definition or evaluator failure".
        if dissociated:
            # Goal achieved - true terminal success.
            is_true_terminal = True
            is_truncated = False
            termination_type = "SUCCESS"
            if hasattr(self, '_consecutive_neb_failures'):
                self._consecutive_neb_failures = 0
        elif o2_removed:
            # O2 escaped reactive zone - true terminal failure (MDP boundary).
            is_true_terminal = True
            is_truncated = False
            termination_type = "ESCAPED"
        elif timed_out:
            # Compute budget exhausted - truncation (could continue).
            is_true_terminal = False
            is_truncated = True
            termination_type = "TIMEOUT"
            if hasattr(self, '_consecutive_neb_failures'):
                self._consecutive_neb_failures = 0  # Timeout is not NEB failure
        elif done and (catastrophic_termination or neb_retry_exhausted):
            # Evaluator failure - truncation (O2 still exists, MDP not truly ended).
            # Bootstrap from V(s') so earlier good steps aren't punished for a
            # computational failure on the last step.
            is_true_terminal = False
            is_truncated = True
            termination_type = "NEB_FAILURE"
            # Track consecutive NEB failures for death spiral detection.
            if not hasattr(self, '_consecutive_neb_failures'):
                self._consecutive_neb_failures = 0
            self._consecutive_neb_failures += 1
            if self._consecutive_neb_failures >= 5:
                # CANONICAL DEATH-SPIRAL: consistent pristine restore. 5 consecutive
                # catastrophic NEB failures => structure is broken. Restore pristine
                # CONSISTENTLY: write to coo_file (disk) AND reload into memory AND zero
                # cumO, so episode_reset's (atoms == original + cumO) invariant holds.
                _LOG.info(f"[NEB DEATH SPIRAL] {self._consecutive_neb_failures} consecutive NEB "
                      f"failures. Restoring pristine substrate (disk+memory, cumO=0).")
                from realize.config.paths import get_structure_file_path
                import shutil as _shutil
                try:
                    from realize.utils.file_utils import standardize_file_format as _stdfmt
                except Exception:
                    _stdfmt = None
                try:
                    _pristine = get_structure_file_path()
                    _shutil.copy2(_pristine, self.env.coo_file)
                    if _stdfmt is not None:
                        try:
                            _stdfmt(self.env.coo_file, target_style="charge")
                        except Exception:
                            pass
                    self.env.structure_manager.initialize_from_file(_pristine)
                    if hasattr(self.env.structure_manager, "invalidate_cache"):
                        self.env.structure_manager.invalidate_cache()
                    if hasattr(self.env.structure_manager, "cumulative_dissociated_o_count"):
                        self.env.structure_manager.cumulative_dissociated_o_count = 0
                    self.env.cumulative_dissociated_o_count = 0
                    self._consecutive_neb_failures = 0
                except Exception as e:
                    _LOG.info(f"[NEB DEATH SPIRAL] Surface reset failed: {e}")
        elif done:
            # Unknown done - not certain it's MDP terminal; treat as truncation to
            # avoid poisoning returns with uncertain targets.
            is_true_terminal = False
            is_truncated = True
            termination_type = "UNKNOWN_DONE"
        else:
            # Episode continuing normally.
            is_true_terminal = False
            is_truncated = False
            termination_type = None

        # Log termination decisions.
        if done or timed_out:
            term_info = {
                'episode': episode,
                'termination_type': termination_type,
                'is_true_terminal': is_true_terminal,
                'is_truncated': is_truncated,
                'step': step,
                'max_steps': self.max_steps,
                'done': done,
                'dissociated': dissociated,
                'timed_out': timed_out,
                'last_valid_buffer_idx': self.last_valid_buffer_idx
            }

            if termination_type == "NEB_FAILURE":
                reasons = []
                if catastrophic_termination:
                    reasons.append("catastrophic_NEB")
                if neb_retry_exhausted:
                    reasons.append("retry_exhausted")
                term_info['failure_reasons'] = ','.join(reasons)

            # GAE bootstrap behavior.
            if timed_out:
                term_info['gae_behavior'] = "bootstrap from V(s_final)"
            elif done and dissociated:
                term_info['gae_behavior'] = "V=0 bootstrap (true terminal)"
            elif done and not dissociated and not o2_removed:
                term_info['gae_behavior'] = "V=0 bootstrap (evaluator unreliable)"

            self._log('TERMINAL', f"Episode termination: {termination_type}", **term_info)

        if timed_out:
            self._log('TERMINAL', f"TIMEOUT: Episode {episode} reached max_steps={self.max_steps}")
        elif o2_removed and not dissociated:
            self._log('TERMINAL', f"VACUUM ESCAPE: Episode {episode} O2 removed (true terminal)")

        if timed_out and self.last_valid_buffer_idx >= 0:
            self._log('TERMINAL', f"Marking last valid buffer index as TRUNCATED",
                      buffer_idx=self.last_valid_buffer_idx)

        # Apply terminal reward to last valid step.
        buffer_size = self._get_buffer_index()  # calls self.agent.get_buffer_length()

        if self.last_valid_buffer_idx >= 0 and self.last_valid_buffer_idx < buffer_size:
            final_rewards = self._calculate_final_rewards(activations, dissociated, episode)

            terminal_reward_total = final_rewards['total']
            ea_component = final_rewards['components'].get('ea', 0.0)
            dissoc_component = final_rewards['components'].get('dissociation', 0.0)
            kl_component = final_rewards['components'].get('kl', 0.0)

            self._log('CREDIT', "Terminal reward distribution", _verbose=True,
                      total=f"{terminal_reward_total:.4f}",
                      ea_component=f"{ea_component:.4f}",
                      dissoc_component=f"{dissoc_component:.4f}",
                      kl_component=f"{kl_component:.4f}" if kl_component > 0 else "0")

            # Collect all valid step indices for credit distribution.
            if not hasattr(self, '_episode_valid_indices') or not self._episode_valid_indices:
                # Fallback: last step only.
                valid_indices = [self.last_valid_buffer_idx]
                self._log('WARNING', "No valid step tracking - applying to last step only")
            else:
                valid_indices = self._episode_valid_indices
                self._log('CREDIT', f"Distributing across {len(valid_indices)} valid steps", _verbose=True)

            # Compute V(s') for truncation bootstrap.
            truncation_value = None
            if timed_out and state is not None:
                try:
                    with torch.no_grad():
                        # Critic backbone for consistent value estimation.
                        final_critic_dict = self.critic_embedding_extractor.get_equivariant_embedding(
                            self.env.structure_manager.get_atoms(),
                            return_symmetric_only=False
                        )
                        final_o1 = final_critic_dict['o1_embedding'].detach()
                        final_o2 = final_critic_dict['o2_embedding'].detach()
                        final_geom_base = final_critic_dict['geometry'].detach()
                        final_geom = augment_geometry_with_context(
                            final_geom_base, step, self.max_steps, env_multipole
                        )
                        # Ensure batch dimension.
                        if final_o1.dim() == 1:
                            final_o1 = final_o1.unsqueeze(0)
                        if final_o2.dim() == 1:
                            final_o2 = final_o2.unsqueeze(0)
                        if final_geom.dim() == 1:
                            final_geom = final_geom.unsqueeze(0)
                        # Value from policy's critic head.
                        truncation_value = self.agent.policy.get_value(final_o1, final_o2, final_geom)
                        if hasattr(truncation_value, 'item'):
                            truncation_value = truncation_value.item()

                        _LOG.debug(f"  [TRUNCATION] Computed V(s')={truncation_value:.4f} for bootstrap")
                except Exception as e:
                    _LOG.info(f"  [TRUNCATION] Failed to compute V(s'): {e}")
                    truncation_value = None

            self._distribute_episode_reward(
                episode_reward=terminal_reward_total,
                valid_step_indices=valid_indices,
                ea_component=ea_component,
                dissoc_component=dissoc_component,
                lambda_decay=0.95,
                is_truncated=is_truncated,
                is_true_terminal=is_true_terminal,
                truncation_value=truncation_value
            )

            # KL component (if any) goes only to last step, with same terminal/truncation logic.
            if kl_component > 0:
                self.agent.update_rewards(
                    index=valid_indices[-1],
                    reward_total=0.0,  # Already counted in distribution.
                    done=(done and dissociated),  # True terminal only, not truncation.
                    is_truncated=timed_out,
                    kl_divergence=kl_component,
                    additive=True
                )

            _LOG.debug(f"  Episode reward distributed successfully")
        else:
            _LOG.info(f"\n[WARNING] No valid steps in episode {episode} - no terminal reward")
            final_rewards = self._calculate_final_rewards(activations, dissociated, episode)

        # Track reward for monitoring.
        if hasattr(self.agent, 'training_rewards'):
            self.agent.training_rewards.append(final_rewards['total'])
            _LOG.debug(f"  Added to training rewards (now {len(self.agent.training_rewards)} total)")

        # Final verification that dissociation state is properly set.
        if dissociated and not getattr(self.env, 'previous_episode_dissociated', False):
            _LOG.debug(f"Episode ended with dissociation, setting flag")
            self.env.previous_episode_dissociated = True

        pe_total = sum(pe_rewards) if pe_rewards else 0.0
        final_rewards['components']['pe_step_rewards'] = pe_total

        # Compute Ea_eff for adaptive midpoint tracking.
        _ea_eff_for_tracking = None
        if activations and dissociated:
            _kT = 8.617333262e-5 * 1000.0  # k_B * T at 1000K
            _scaled = [e / _kT for e in activations]
            _max_s = max(_scaled)
            _ea_eff_for_tracking = _kT * (_max_s + np.log(sum(np.exp(s - _max_s) for s in _scaled)))

        results = {
            'dissociated': dissociated,
            'total_reward': final_rewards['total'],
            'reward_components': final_rewards['components'],
            'ea_eff': _ea_eff_for_tracking,
            'valid_steps': valid_steps,
            'novel_states': novel_states,
            'steps': step,
            'stats': {
                'steps': step,
                'valid_steps': valid_steps,
                'novelty_rate': novel_states / max(step, 1),
                'max_barrier': max(activations) if activations else 0.0
            }
        }

        _LOG.debug(f"\n Episode {episode} complete. Returning results.")
        _LOG.debug(f"{'='*60}\n")

        return results

    def _take_step(self, trans_action: np.ndarray, o2_config: np.ndarray,
                   episode: int, step: int, prev_state_file: str) -> Dict[str, Any]:
        """Execute single environment step."""
        # trans_action may be a dict from the policy or a numpy translation vector.
        if isinstance(trans_action, dict):
            translation_vector = trans_action['translation_vector']
            axis_direction = trans_action['axis_direction']
            o2_config_packed = np.array([translation_vector, axis_direction])
        else:
            # o2_config contains [translation, axis].
            translation_vector = trans_action
            o2_config_packed = o2_config

        # Take step with NEB validation.
        step_start_time = time.time()

        result = step_environment_with_rollback(
            self.env,
            translation_vector,
            o2_config_packed,
            episode,
            prev_state=prev_state_file,
            use_neb=self.use_neb_reward,
            checkpoint_path=get_embedding_checkpoint_path(),
            step_count=step,
            logger=self.logger,
            device=self.device,
            embedding_extractor=self.embedding_extractor,
            **self._get_step_kwargs()
        )

        step_time = time.time() - step_start_time

        # Track step timing.
        if not hasattr(self, 'step_timing_stats'):
            self.step_timing_stats = {'neb_steps': 0, 'total_time': 0.0, 'times': []}

        self.step_timing_stats['times'].append(step_time)
        self.step_timing_stats['total_time'] += step_time

        # Validate result before accessing
        if result is None or len(result) < 8:
            raise RuntimeError(
                f"Invalid result from step_environment_with_rollback. "
                f"Expected tuple of length 8, got: {result}"
            )

        # Track NEB usage (after validation).
        if self.use_neb_reward and result[4]:  # result[4] is 'valid'
            self.step_timing_stats['neb_steps'] += 1

        # Log timing stats every 10 steps.
        if step % 10 == 0 and len(self.step_timing_stats['times']) > 0:
            recent_times = self.step_timing_stats['times'][-10:]
            avg_time = sum(recent_times) / len(recent_times)
            neb_rate = self.step_timing_stats['neb_steps'] / max(1, len(self.step_timing_stats['times']))

            self.logger.log_debug(
                f"[Step {step}] Timing: avg={avg_time:.2f}s, "
                f"NEB_rate={neb_rate:.1%}, total_time={self.step_timing_stats['total_time']:.1f}s"
            )


        # Unpack and validate result.
        dist, embedding, reward, done, valid, activation, try_num, dissociated = result

        if not isinstance(valid, (bool, np.bool_)):
            raise RuntimeError(
                f"Invalid 'valid' type from step_environment_with_rollback: "
                f"expected bool, got {type(valid)}. Result: {result}"
            )
        if not isinstance(done, (bool, np.bool_)):
            raise RuntimeError(
                f"Invalid 'done' type from step_environment_with_rollback: "
                f"expected bool, got {type(done)}"
            )
        if not isinstance(dissociated, (bool, np.bool_)):
            raise RuntimeError(
                f"Invalid 'dissociated' type from step_environment_with_rollback: "
                f"expected bool, got {type(dissociated)}"
            )

        # Validate numerical fields.
        if reward is not None and not np.isfinite(reward):
            raise RuntimeError(
                f"Non-finite reward from step_environment_with_rollback: {reward}. "
                f"This indicates NaN/Inf propagation in reward calculation."
            )
        if activation is not None:
            if not np.isfinite(activation):
                raise RuntimeError(f"Non-finite activation energy: {activation}")
            if activation < -1.0:  # Activation energies should be >= 0
                raise RuntimeError(
                    f"Negative activation energy: {activation}. "
                    f"Expected activation >= 0 eV."
                )
            if activation > 20.0:  # Sanity check: should not exceed this for Si oxidation
                raise RuntimeError(
                    f"Unreasonably large activation energy: {activation} eV. "
                    f"This may indicate NEB calculation failure."
                )

        # Validate embedding if present (dict form is allowed).
        if embedding is not None:
            if not isinstance(embedding, (np.ndarray, torch.Tensor)):
                if not isinstance(embedding, dict):
                    raise RuntimeError(
                        f"Invalid embedding type: expected ndarray, Tensor or dict, got {type(embedding)}"
                    )
            if isinstance(embedding, torch.Tensor) and embedding.requires_grad:
                # Gradients are allowed here.
                pass

        # Mark MD-verified dissociation.
        if done and dissociated:
            self.env.previous_episode_dissociated = True
            self.env.last_md_verified_dissociation = True

        # Re-extract embeddings using the actor backbone (symmetric + asymmetric, so
        # the policy gets a consistent representation from one backbone).
        next_state = None
        if valid and not done:  # Only re-extract if step was valid and episode continues.
            embeddings_new = self.actor_embedding_extractor.get_equivariant_embedding(
                self.env.structure_manager.get_atoms(),
                return_symmetric_only=False
            )

            if isinstance(embeddings_new, dict):
                embedding_symmetric = embeddings_new['symmetric']
                embedding_asymmetric = embeddings_new['asymmetric']
                # Detach to prevent graph accumulation during rollout.
                if torch.is_tensor(embedding_symmetric):
                    embedding_symmetric = embedding_symmetric.detach()
                if torch.is_tensor(embedding_asymmetric):
                    embedding_asymmetric = embedding_asymmetric.detach()
                # Return raw embedding; normalization happens in run_episode loop.
                next_state = embedding_symmetric
                self.current_o2_embedding_asymmetric = embedding_asymmetric
                _LOG.debug(f"  [_take_step] Re-extracted dual embeddings (RAW, detached)")
            else:
                if torch.is_tensor(embeddings_new):
                    embeddings_new = embeddings_new.detach()
                next_state = embeddings_new
                self.current_o2_embedding_asymmetric = embeddings_new
                _LOG.debug(f"  [_take_step] Re-extracted single embedding (RAW, detached)")

        elif embedding is not None:
            # Fallback to env-provided embedding.
            if isinstance(embedding, dict):
                embedding_symmetric = embedding['symmetric']
                embedding_asymmetric = embedding['asymmetric']
                # Detach to prevent graph accumulation.
                if torch.is_tensor(embedding_symmetric):
                    embedding_symmetric = embedding_symmetric.detach()
                if torch.is_tensor(embedding_asymmetric):
                    embedding_asymmetric = embedding_asymmetric.detach()
                next_state = self.actor_state_normalizer.normalize(embedding_symmetric)
                self.current_o2_embedding_asymmetric = embedding_asymmetric
                _LOG.debug(f"  [_take_step] Using env embedding (fallback, detached)")
            else:
                if torch.is_tensor(embedding):
                    embedding = embedding.detach()
                next_state = self.actor_state_normalizer.normalize(embedding)
                self.current_o2_embedding_asymmetric = embedding
                _LOG.debug(f"  [_take_step] Using single env embedding (fallback, detached)")
        else:
            _LOG.debug(f"  [_take_step] No embedding available")

        # Save state if valid.
        state_file = None
        if valid:
            state_file = self._save_state(episode, step, try_num)

        return {
            'valid': valid,
            'reward': reward,
            'done': done,
            'dissociated': dissociated,
            'activation_energy': activation,
            'next_state': next_state,
            'state_file': state_file
        }
    def _reset_episode(self, episode: int) -> Optional[torch.Tensor]:
        """Reset environment and get initial state."""

        if episode > 0:
            _LOG.debug(f"\n=== Episode {episode} Transition Debug ===")
            _LOG.debug(f"  previous_episode_dissociated: {getattr(self.env, 'previous_episode_dissociated', 'NOT SET')}")
            _LOG.debug(f"  last_md_verified_dissociation: {getattr(self.env, 'last_md_verified_dissociation', 'NOT SET')}")
            _LOG.debug(f"  o2_removed: {getattr(self.env, 'o2_removed', 'NOT SET')}")
            _LOG.debug(f"  cumulative_dissociated_o_count: {getattr(self.env, 'cumulative_dissociated_o_count', 'NOT SET')}")
            # episode_outcomes[N] is the per-episode anchor parsed by analysis
            # (literal_eval of the dict, incl. path_len). Must stay at INFO so the
            # per-cycle metrics pipeline works without a verbose capture.
            if hasattr(self.env, 'episode_outcomes') and episode-1 in self.env.episode_outcomes:
                _LOG.info(f"  episode_outcomes[{episode-1}]: {self.env.episode_outcomes[episode-1]}")
            _LOG.debug("=" * 40 + "\n")

        # Use this runner's embedding_extractor (gradients flow), replacing the env's.
        dist, _ = reset_episode(
            self.env,
            episode,
            checkpoint_path=get_embedding_checkpoint_path(),
            device=self.device,
            embedding_extractor=self.embedding_extractor
        )

        # Extract embeddings using the actor backbone (symmetric + asymmetric).
        embeddings = self.actor_embedding_extractor.get_equivariant_embedding(
            self.env.structure_manager.get_atoms(),
            return_symmetric_only=False
        )

        # Unpack symmetric and asymmetric from the dict (else single tensor for both).
        if isinstance(embeddings, dict):
            embedding_symmetric = embeddings['symmetric']
            embedding_asymmetric = embeddings['asymmetric']
        else:
            embedding_symmetric = embeddings
            embedding_asymmetric = embeddings
            _LOG.info(f"  WARNING: Old embedding format (single tensor), using for both symmetric and asymmetric")

        # Detach embeddings to prevent graph accumulation during rollout.
        if torch.is_tensor(embedding_asymmetric):
            embedding_asymmetric = embedding_asymmetric.detach()
        if torch.is_tensor(embedding_symmetric):
            embedding_symmetric = embedding_symmetric.detach()

        # Asymmetric embedding feeds the policy; symmetric (COM-based) is the state.
        self.current_o2_embedding_asymmetric = embedding_asymmetric
        embedding = embedding_symmetric

        _LOG.debug(f"  Using symmetric embedding for state: shape={embedding.shape}")
        # Validate embedding (must not be None).
        if embedding is None:
            raise RuntimeError(
                f"Failed to get initial embedding for episode {episode}. "
                f"EmbeddingExtractor returned None embedding."
            )

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        # Keep gradients when training the backbone; detach otherwise.
        if hasattr(embedding, 'requires_grad'):
            if self.embedding_extractor.train_backbone:
                _LOG.debug(f"[_reset_episode] Keeping gradients (train_backbone=True)")
            else:
                if embedding.requires_grad:
                    embedding = embedding.detach().requires_grad_(False)

        # Return raw embedding; normalization happens once in run_episode (avoids double-normalizing).
        _LOG.debug(f"[_reset_episode] Returning RAW embedding (normalization in run_episode)")

        return embedding

    def _save_initial_state(self, episode: int) -> str:
        """Save initial state file."""

        path = os.path.join(get_neb_results_dir(), f"initial_state_ep{episode}.data")

        self.env.save_current_state(path)

        standardize_file_format(path, target_style='charge')

        return path

    def _save_state(self, episode: int, step: int, try_num: int) -> str:
        """Save current state file."""

        path = os.path.join(
            get_neb_results_dir(),
            f"state_ep{episode}_step{step}_try{try_num}.data"
        )
        self.env.save_current_state(path)
        standardize_file_format(path, target_style='charge')
        return path

    def _get_step_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for step function (PPOConfig or flat config, no defaults)."""
        if hasattr(self.config, 'reward'):
            if not hasattr(self.config, 'model'):
                raise ValueError("PPOConfig must have 'model' attribute")
            if not hasattr(self.config.model, 'max_trans_distance'):
                raise ValueError("config.model.max_trans_distance is REQUIRED")

            max_trans_distance = self.config.model.max_trans_distance

            return {
                'ea_weight': self.config.reward.ea_weight,
                'dissociation_weight': self.config.reward.dissociation_weight,
                'timeout_penalty': self.config.reward.timeout_penalty,
                'invalid_step_penalty': self.config.reward.invalid_step_penalty,
                'high_barrier_threshold': self.config.reward.high_barrier_threshold,
                'oscillation_penalty': self.config.reward.oscillation_penalty,
                'pe_reward_weight': self.config.reward.pe_reward_weight,
                'max_trans_distance': max_trans_distance,
                'pe_shaping_gamma': getattr(self.config.reward, 'pe_shaping_gamma', 0.99),
                'invalid_penalty_cap': getattr(self.config.reward, 'invalid_penalty_cap', -0.3),
            }
        else:
            return {
                'ea_weight': self.config.ea_weight,
                'dissociation_weight': self.config.dissociation_weight,
                'timeout_penalty': self.config.timeout_penalty,
                'invalid_step_penalty': self.config.invalid_step_penalty,
                'high_barrier_threshold': self.config.high_barrier_threshold,
                'oscillation_penalty': self.config.oscillation_penalty,
                'pe_reward_weight': self.config.pe_reward_weight,
                'max_trans_distance': self.config.max_trans_distance,
                'pe_shaping_gamma': getattr(self.config, 'pe_shaping_gamma', 0.99),
                'invalid_penalty_cap': getattr(self.config, 'invalid_penalty_cap', -0.3),
            }

    def _get_reward_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for reward calculation; raises ValueError if required fields are missing."""
        if hasattr(self.config, 'reward'):
            required_fields = ['require_dissociation', 'Ea_gamma', 'ea_sigmoid_k',
                             'ea_sigmoid_midpoint', 'cap_unsuccessful_reward']
            missing = [f for f in required_fields if not hasattr(self.config.reward, f)]
            if missing:
                raise ValueError(
                    f"Missing required reward config fields: {missing}"
                )

            return {
                'require_dissociation': self.config.reward.require_dissociation,
                'Ea_gamma': self.config.reward.Ea_gamma,
                'ea_sigmoid_k': self.config.reward.ea_sigmoid_k,
                'ea_sigmoid_midpoint': self.config.reward.ea_sigmoid_midpoint,
                'cap_unsuccessful_reward': self.config.reward.cap_unsuccessful_reward
            }
        else:
            required_fields = ['require_dissociation', 'Ea_gamma', 'ea_sigmoid_k',
                             'ea_sigmoid_midpoint', 'cap_unsuccessful_reward']
            missing = [f for f in required_fields if not hasattr(self.config, f)]
            if missing:
                raise ValueError(
                    f"Missing required flat config fields: {missing}"
                )

            return {
                'require_dissociation': self.config.require_dissociation,
                'Ea_gamma': self.config.Ea_gamma,
                'ea_sigmoid_k': self.config.ea_sigmoid_k,
                'ea_sigmoid_midpoint': self.config.ea_sigmoid_midpoint,
                'cap_unsuccessful_reward': self.config.cap_unsuccessful_reward
            }

    def get_normalizers(self):
        """Return (actor_state_normalizer, None) for PPO training; second element is always None."""
        return self.actor_state_normalizer, None
