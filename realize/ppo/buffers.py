"""Rollout buffer with success oversampling."""
import numpy as np
import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

# CriticReplayBuffer lives in its own module; re-exported here so that
# `from realize.ppo.buffers import RolloutBuffer, CriticReplayBuffer` keeps working.
from realize.ppo.critic_replay_buffer import CriticReplayBuffer
from realize.ppo.rollout_buffer_analysis import RolloutBufferAnalysisMixin
import logging
_LOG = logging.getLogger(__name__)


class BufferInterface(ABC):
    """Abstract base class defining the interface for experience buffers"""
    
    @abstractmethod
    def add(self, state, action_rot, trans_sample, log_prob, state_value, 
            reward=0.0, is_terminal=False, **kwargs):
        """Add a transition to the buffer"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored transitions"""
        pass
    
    @abstractmethod
    def update_reward(self, index, reward, is_terminal=False, **kwargs):
        """Update the reward for a specific transition"""
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return the number of transitions stored"""
        pass
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array safely"""
        if tensor is None:
            raise RuntimeError(
                "Cannot convert None tensor to numpy.\n"
                "Verify that data exists before calling tensor_to_numpy()."
            )
        
        if isinstance(tensor, dict):
            # Reject unknown keys rather than silently dropping them
            allowed_keys = {'translation_direction', 'translation_magnitude',
                          'axis_direction', 'axis_frame', 'translation_vector', 'direction_indices',
                          'axis_mean_direction', 'axis_log_concentration',
                          'o1_embedding', 'o2_embedding', 'geometry'}                    
            unknown_keys = set(tensor.keys()) - allowed_keys
            if unknown_keys:
                raise KeyError(
                    f"Unknown action keys: {unknown_keys}. "
                    f"Allowed keys: {allowed_keys}. "
                    f"Unknown keys would be silently dropped, which can cause data loss. "
                    f"Check that action dict only contains expected keys."
                )
            return {k: self._tensor_to_numpy(v) for k, v in tensor.items()}
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, (list, tuple)):
            return [self._tensor_to_numpy(t) for t in tensor]
        return tensor

    def _standardize_action(self, action):
        """Standardize an action to a numpy dict. Dicts must carry the required
        keys; tensors are decomposed into translation_vector/axis_direction. Raises
        on None, missing keys, or unsupported type.
        """
        if action is None:
            raise ValueError("action cannot be None")

        if isinstance(action, dict):
            required_keys = {'translation_direction', 'translation_magnitude',
                           'axis_direction', 'translation_vector'}
            missing_keys = required_keys - set(action.keys())
            if missing_keys:
                raise ValueError(
                    f"Action dict missing required keys: {missing_keys}. "
                    f"Present keys: {set(action.keys())}. "
                    f"All of {required_keys} must be present."
                )

            return {k: self._tensor_to_numpy(v) for k, v in action.items() if v is not None}

        if isinstance(action, (torch.Tensor, np.ndarray)):
            action_np = self._tensor_to_numpy(action)
            if action_np.shape[-1] < 6:
                raise ValueError(
                    f"Tensor action must have at least 6 values [trans(3), axis(3)], "
                    f"got shape {action_np.shape}"
                )
            return {
                'translation_vector': action_np[..., :3],
                'axis_direction': action_np[..., 3:6]
            }

        raise TypeError(
            f"Unsupported action type: {type(action)}. "
            f"Must be dict or tensor/array."
        )


class RolloutBuffer(RolloutBufferAnalysisMixin, BufferInterface):
    """Rollout buffer with success oversampling for sparse rewards.

    Tracks successful (dissociation) episodes and over-represents their steps when
    sampling, maintains episode boundaries for GAE, and crucially distinguishes
    truncation (timeout) from true terminal (dissociation) for correct bootstrapping.
    """

    def __init__(self, debug_mode=False, success_oversample_ratio=3.0):
        """success_oversample_ratio=3.0 makes successful episodes 3x as likely to
        be sampled.
        """
        self.debug_mode = debug_mode
        self.success_oversample_ratio = success_oversample_ratio
        self._initialize_storage()
        self._snapshot_lengths = {}

        # Success tracking
        self.episode_boundaries = []  # (start_idx, end_idx) per episode
        self.episode_success = []     # bool per episode: dissociated?
        self._current_episode_start = 0

        # Per-transition episode index, for consistent PPO log-prob computation
        self.episode_indices = []
        self._current_episode_idx = 0

    def _initialize_storage(self):
        """Initialize all storage lists"""
        # Core trajectory data
        self.states = []
        self.actor_states = []  
        self.actions_rot = []
        self.trans_samples = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = [] 
        self.is_truncated = [] 
        self.truncation_bootstrap_values = {} 
        self.state_values = []
        
        # Reward components
        self.ea_rewards = []
        self.dissociation_rewards = []
        self.diffusion_ea_rewards = []
        self.dissociation_ea_rewards = []
        self.activation_energies = []
        self.axis_rewards = []  # per-step axis reward signal

        # Spherical harmonics data
        self.sh_coefficients = []
        self.translation_directions = []
        self.translation_magnitudes = []

        # Multi-channel data
        self.temperatures = []
        self.num_channels_used = []
        self.channel_weights = []

        # Auto-regressive data
        self.sampled_directions = []
        self.conditioned_magnitudes = []

        # Per-transition validity, to filter placeholder steps before PPO update
        self.valid_flags = []

        # Asymmetric O2 embedding for axis prediction
        self.o2_embeddings_asymmetric = []
        self.env_multipoles = []

        # Critic raw inputs, stored for recomputation through CriticEmbedding (inside
        # PolicyNetwork) during evaluate() so gradients flow; not precomputed features.
        self.o1_embeddings = []    # [T] of [1, D] raw O1 node embeddings
        self.o2_embeddings = []    # [T] of [1, D] raw O2 node embeddings
        self.geometries = []       # [T] of [1, geom_dim] SO(3)-invariant geometry

        # Per-head log-prob contributions for separate policy losses
        self.log_probs_direction = []
        self.log_probs_magnitude = []
        self.log_probs_axis = []
        self.axis_params_stored = []       # axis_params (mean_dir, log_conc) for exact replay

        # Success tracking reset
        self.episode_boundaries = []
        self.episode_success = []
        self._current_episode_start = 0

        # Reset episode indices too; stale indices would mix episodes after clear()
        self.episode_indices = []

    def _snapshot_current_lengths(self):
        """Store current lengths of all lists before adding"""
        self._snapshot_lengths = {
            'states': len(self.states),
            'actor_states': len(self.actor_states),  
            'actions_rot': len(self.actions_rot),        
            'trans_samples': len(self.trans_samples),
            'log_probs': len(self.log_probs),
            'rewards': len(self.rewards),
            'is_terminals': len(self.is_terminals),
            'is_truncated': len(self.is_truncated),
            'state_values': len(self.state_values),
            'ea_rewards': len(self.ea_rewards),
            'dissociation_rewards': len(self.dissociation_rewards),
            'diffusion_ea_rewards': len(self.diffusion_ea_rewards),
            'dissociation_ea_rewards': len(self.dissociation_ea_rewards),
            'activation_energies': len(self.activation_energies),
            'axis_rewards': len(self.axis_rewards),
            'sh_coefficients': len(self.sh_coefficients),
            'translation_directions': len(self.translation_directions),
            'translation_magnitudes': len(self.translation_magnitudes),
            'temperatures': len(self.temperatures),
            'num_channels_used': len(self.num_channels_used),
            'channel_weights': len(self.channel_weights),
            'sampled_directions': len(self.sampled_directions),
            'conditioned_magnitudes': len(self.conditioned_magnitudes),
            'valid_flags': len(self.valid_flags),
            'o2_embeddings_asymmetric': len(self.o2_embeddings_asymmetric),
            'o1_embeddings': len(self.o1_embeddings),
            'o2_embeddings': len(self.o2_embeddings),
            'geometries': len(self.geometries),
            'log_probs_direction': len(self.log_probs_direction),
            'log_probs_magnitude': len(self.log_probs_magnitude),
            'log_probs_axis': len(self.log_probs_axis),
            'episode_indices': len(self.episode_indices)
        }

    def _rollback_to_snapshot(self):
        """Rollback all lists to their snapshot lengths"""
        for attr_name, original_length in self._snapshot_lengths.items():
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                setattr(self, attr_name, attr[:original_length])

        if self.debug_mode:
            _LOG.debug(f"Rolled back buffer addition to snapshot lengths")

    def mark_episode_end(self, success: bool):
        """Record an episode boundary and whether it dissociated. Call on episode
        termination (dissociation or max steps).
        """
        end_idx = len(self.states)
        if end_idx > self._current_episode_start:
            self.episode_boundaries.append((self._current_episode_start, end_idx))
            self.episode_success.append(success)
            
            if self.debug_mode:
                status = "SUCCESS" if success else "FAILURE"
                _LOG.debug(f"[BUFFER] Episode ended: indices [{self._current_episode_start}, {end_idx}), "
                      f"length={end_idx - self._current_episode_start}, status={status}")
        
        self._current_episode_start = end_idx
    
    def add(self, state, action_rot, trans_sample, log_prob, state_value, 
        reward=0.0, is_terminal=False, is_truncated=False,
        log_prob_direction=None, log_prob_magnitude=None, log_prob_axis=None,
        **kwargs):
        """Add a transition, rolling back all lists on any failure. is_terminal is a
        true terminal (dissociation/failure -> GAE bootstrap 0); is_truncated is a
        timeout (GAE bootstraps from V(s')). Returns True on success.
        """
        self._snapshot_current_lengths()

        try:
            state_np = self._tensor_to_numpy(state) if not isinstance(state, np.ndarray) else state
            action_rot_np = self._standardize_action(action_rot)
            trans_sample_np = self._tensor_to_numpy(trans_sample) if trans_sample is not None else None
            log_prob_np = self._tensor_to_numpy(log_prob)
            state_value_np = self._tensor_to_numpy(state_value)

            self.states.append(state_np)

            # Actor state for policy evaluation during PPO update
            actor_state = kwargs.get('actor_state', None)
            if actor_state is not None:
                actor_state_np = self._tensor_to_numpy(actor_state)
                self.actor_states.append(actor_state_np)
            else:
                self.actor_states.append(state_np)

            # Critic raw inputs, recomputed through CriticEmbedding during evaluate()
            # for gradient flow; stored detached (backbone frozen)
            o1_embedding = kwargs.get('o1_embedding', None)
            o2_embedding = kwargs.get('o2_embedding', None)
            geometry = kwargs.get('geometry', None)
            
            if o1_embedding is None or o2_embedding is None or geometry is None:
                raise RuntimeError(
                    "FAIL: buffer.add() requires o1_embedding, o2_embedding, geometry "
                    "for critic value computation. No fallback to old critic_state path. "
                    f"Got: o1={o1_embedding is not None}, o2={o2_embedding is not None}, "
                    f"geom={geometry is not None}"
                )
            
            self.o1_embeddings.append(self._tensor_to_numpy(o1_embedding))
            self.o2_embeddings.append(self._tensor_to_numpy(o2_embedding))
            self.geometries.append(self._tensor_to_numpy(geometry))

            self.actions_rot.append(action_rot_np)            
            self.trans_samples.append(trans_sample_np)
            self.log_probs.append(log_prob_np)
            self.state_values.append(state_value_np)
            self.rewards.append(reward)

            # Asymmetric O2 embedding (optional)
            o2_embedding_asymmetric = kwargs.get('o2_embedding_asymmetric', None)
            if o2_embedding_asymmetric is not None:
                o2_emb_np = self._tensor_to_numpy(o2_embedding_asymmetric)
            else:
                o2_emb_np = None
            self.o2_embeddings_asymmetric.append(o2_emb_np)            
            env_multipole = kwargs.get('env_multipole', None)
            if env_multipole is not None:
                self.env_multipoles.append(self._tensor_to_numpy(env_multipole))
            else:
                self.env_multipoles.append(None)

            self.is_terminals.append(is_terminal)
            self.is_truncated.append(is_truncated)

            # Per-head log-probs; default to total log-prob when not given separately
            self.log_probs_direction.append(log_prob_direction if log_prob_direction is not None else log_prob_np)
            self.log_probs_magnitude.append(log_prob_magnitude if log_prob_magnitude is not None else log_prob_np)
            self.log_probs_axis.append(log_prob_axis if log_prob_axis is not None else log_prob_np)

            ea_reward = kwargs.get('ea_reward', 0.0)
            dissociation_reward = kwargs.get('dissociation_reward', 0.0)
            diffusion_ea_reward = kwargs.get('diffusion_ea_reward', 0.0)
            dissociation_ea_reward = kwargs.get('dissociation_ea_reward', 0.0)
            activation_energy = kwargs.get('activation_energy', 0.0)
            axis_reward = kwargs.get('axis_reward', 0.0)
            
            self.ea_rewards.append(ea_reward)
            self.dissociation_rewards.append(dissociation_reward)
            self.diffusion_ea_rewards.append(diffusion_ea_reward)
            self.dissociation_ea_rewards.append(dissociation_ea_reward)
            self.activation_energies.append(activation_energy)
            self.axis_rewards.append(axis_reward)

            sh_coeffs = kwargs.get('sh_coeffs', None)
            trans_direction = kwargs.get('trans_direction', None)
            trans_magnitude = kwargs.get('trans_magnitude', None)
            
            self.sh_coefficients.append(self._tensor_to_numpy(sh_coeffs))
            self.translation_directions.append(self._tensor_to_numpy(trans_direction))
            self.translation_magnitudes.append(self._tensor_to_numpy(trans_magnitude))

            self.temperatures.append(kwargs.get('temperature', None))
            self.num_channels_used.append(kwargs.get('num_channels', None))
            self.channel_weights.append(self._tensor_to_numpy(kwargs.get('channel_weights', None)))

            # Auto-regressive data: only stored if provided
            sampled_dir = kwargs.get('sampled_direction', None)
            if sampled_dir is not None:
                self.sampled_directions.append(self._tensor_to_numpy(sampled_dir))
            else:
                self.sampled_directions.append(None)
            
            conditioned_mag = kwargs.get('conditioned_magnitude', None)
            if conditioned_mag is not None:
                self.conditioned_magnitudes.append(self._tensor_to_numpy(conditioned_mag))
            else:
                self.conditioned_magnitudes.append(None)

            # Invalid until reward is updated: keeps PPO off placeholder steps
            self.valid_flags.append(False)

            # Episode index for schedule consistency in PPO
            self.episode_indices.append(self._current_episode_idx)

            if not self._check_consistency():
                self._rollback_to_snapshot()
                raise RuntimeError(
                    "Buffer consistency check failed after addition. "
                    "List lengths do not match. Check _check_consistency() output."
                )

            if not self.validate_data_types():
                self._rollback_to_snapshot()
                raise TypeError(
                    "Buffer data type validation failed after addition. "
                    "Non-scalar values in log_probs/state_values/rewards. "
                    "Check validate_data_types() output."
                )

            return True

        except Exception as e:
            self._rollback_to_snapshot()
            raise RuntimeError(
                f"Failed to add to buffer: {str(e)}"
            ) from e

    def _check_consistency(self):
        """True if all buffer lists share one length; logs the mismatch otherwise."""
        # Explicit list of all buffer lists (avoids dir() introspection)
        list_attrs = [
            'states', 'actor_states',
            'actions_rot', 'trans_samples', 'log_probs', 'rewards',
            'is_terminals', 'is_truncated', 'state_values',
            'ea_rewards', 'dissociation_rewards',        
            'diffusion_ea_rewards', 'dissociation_ea_rewards', 'activation_energies',
            'axis_rewards',
            'sh_coefficients', 'translation_directions', 'translation_magnitudes',
            'temperatures', 'num_channels_used', 'channel_weights',
            'sampled_directions', 'conditioned_magnitudes', 'valid_flags',
            'o2_embeddings_asymmetric',
            'o1_embeddings', 'o2_embeddings', 'geometries',
            'log_probs_direction', 'log_probs_magnitude', 'log_probs_axis',
            'env_multipoles'            
        ]

        lengths = {attr: len(getattr(self, attr)) for attr in list_attrs}

        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            _LOG.info(f"Buffer inconsistency detected:")
            for attr, length in lengths.items():
                _LOG.info(f"  {attr}: {length}")
            return False
        
        return True

    def update_reward(self, index, reward, is_terminal=False, is_truncated=False,
                      ea_reward=0.0, dissociation_reward=0.0, axis_reward=0.0, 
                      additive=False, **kwargs):
        """Update reward at index. additive=False replaces (per-step), additive=True adds
        (episode-end bonus). is_terminal -> GAE bootstrap 0; is_truncated -> bootstrap V(s').
        """
        if not (0 <= index < len(self.rewards)):
            raise IndexError(
                f"Invalid index {index} for buffer size {len(self.rewards)}. "
                f"Index must be in range [0, {len(self.rewards)})."
            )
    
        # Coerce to scalar + finite check
        if isinstance(reward, (torch.Tensor, np.ndarray)):
            reward = float(np.array(reward).squeeze())
        if not np.isfinite(reward):
            raise ValueError(
                f"Non-finite reward at index {index}: {reward}. "
                f"Reward must be finite (not NaN or Inf)."
            )

        if additive:
            self.rewards[index] = float(self.rewards[index] + reward)
        else:
            self.rewards[index] = float(reward)

        self.is_terminals[index] = bool(is_terminal)
        self.is_truncated[index] = bool(is_truncated)

        # Now that reward is written, the step is valid for PPO
        self.valid_flags[index] = True

        # Episode components used for per-head reward routing later
        self.ea_rewards[index] = float(ea_reward)
        self.dissociation_rewards[index] = float(dissociation_reward)
        self.axis_rewards[index] = float(axis_reward)

        if 'diffusion_ea_reward' in kwargs:
            self.diffusion_ea_rewards[index] = kwargs['diffusion_ea_reward']
        if 'dissociation_ea_reward' in kwargs:
            self.dissociation_ea_rewards[index] = kwargs['dissociation_ea_reward']
        if 'activation_energy' in kwargs:
            self.activation_energies[index] = kwargs['activation_energy']
        return True

    def _get_success_weighted_indices(self) -> np.ndarray:
        """Per-index sampling probabilities; successful-episode steps weighted higher."""
        buffer_size = len(self.states)
        weights = np.ones(buffer_size)

        for (start, end), success in zip(self.episode_boundaries, self.episode_success):
            if success:
                weights[start:end] = self.success_oversample_ratio

        weights = weights / weights.sum()

        return weights

    def sample(self, batch_size, use_success_weighting: bool = True) -> Dict[str, Any]:
        """Sample a batch (optionally oversampling successful episodes), returning a dict
        of all buffer fields for the chosen indices plus 'sampled_indices'. Raises if empty.
        """
        buffer_size = len(self.states)
        if buffer_size == 0:
            raise RuntimeError(
                "Cannot sample from empty buffer. "
                "Buffer must contain at least one transition."
            )

        batch_size = min(batch_size, buffer_size)

        if batch_size == buffer_size:
            indices = np.arange(buffer_size)
        elif use_success_weighting and self.episode_success and any(self.episode_success):
            weights = self._get_success_weighted_indices()
            indices = np.random.choice(
                buffer_size, 
                batch_size, 
                replace=False,
                p=weights
            )
            
            if self.debug_mode:
                success_sampled = sum(1 for i in indices if self._is_success_index(i))
                _LOG.debug(f"[BUFFER] Sampled {success_sampled}/{batch_size} from successful episodes "
                      f"(ratio={success_sampled/batch_size:.2f})")
        else:
            indices = np.random.choice(buffer_size, batch_size, replace=False)

        return {
            # Dual-backbone core states (actor vs frozen-critic path)
            'states': np.array([self.states[i] for i in indices]),
            'actor_states': np.array([self.actor_states[i] for i in indices]),

            'actions': [self.actions_rot[i] for i in indices],
            'trans_samples': np.array([self.trans_samples[i] for i in indices]),

            # Total and per-head log-probs
            'log_probs': np.array([self.log_probs[i] for i in indices]),
            'log_probs_direction': np.array([self.log_probs_direction[i] for i in indices]),
            'log_probs_magnitude': np.array([self.log_probs_magnitude[i] for i in indices]),
            'log_probs_axis': np.array([self.log_probs_axis[i] for i in indices]),

            'state_values': np.array([self.state_values[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'is_terminals': np.array([self.is_terminals[i] for i in indices]),
            'is_truncated': np.array([self.is_truncated[i] for i in indices]),

            # Reward components for per-head training
            'ea_rewards': np.array([self.ea_rewards[i] for i in indices]),
            'dissociation_rewards': np.array([self.dissociation_rewards[i] for i in indices]),
            'axis_rewards': np.array([self.axis_rewards[i] for i in indices]),

            # Asymmetric embedding for axis head
            'o2_embeddings_asymmetric': [self.o2_embeddings_asymmetric[i] for i in indices],

            # Critic raw inputs, recomputed through CriticEmbedding during evaluate()
            'o1_embeddings': [self.o1_embeddings[i] for i in indices],
            'o2_embeddings': [self.o2_embeddings[i] for i in indices],
            'geometries': [self.geometries[i] for i in indices],

            # Episode index + rollout-time temperature, for schedule consistency
            'episode_indices': np.array([self.episode_indices[i] for i in indices]),
            'temperatures': np.array([self.temperatures[i] for i in indices]),

            # valid_flags filters placeholder steps; sampled_indices keys per-head advantages
            'valid_flags': np.array([self.valid_flags[i] for i in indices]),
            'sampled_indices': indices,
        }

    def _is_success_index(self, idx: int) -> bool:
        """Check if index belongs to a successful episode."""
        for (start, end), success in zip(self.episode_boundaries, self.episode_success):
            if start <= idx < end:
                return success
        return False

    def sample_with_indices(self, batch_size, use_success_weighting: bool = True) -> Dict[str, Any]:
        """DEPRECATED: use sample(), which already returns 'sampled_indices'. Kept for
        backward compatibility; delegates to sample().
        """
        return self.sample(batch_size, use_success_weighting)

    def set_current_episode(self, episode_idx: int):
        """Set current episode index for new transitions."""
        self._current_episode_idx = episode_idx
    
    def clear(self):
        """Clear all stored transitions"""
        self._initialize_storage()
        self._snapshot_lengths.clear()
        gc.collect()

    def __len__(self):
        """Return the number of transitions stored"""
        return len(self.states)
