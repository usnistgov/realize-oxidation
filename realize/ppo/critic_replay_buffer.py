"""Critic replay buffer (CriticReplayBuffer).

buffers.py re-exports this name
(`from realize.ppo.critic_replay_buffer import CriticReplayBuffer`) so imports
`from realize.ppo.buffers import RolloutBuffer, CriticReplayBuffer` keep working,
and the class identity stays a single object (isinstance safe).

CriticReplayBuffer is standalone -- it does not inherit from BufferInterface and
does not reference RolloutBuffer.
"""
import numpy as np
import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import logging
_LOG = logging.getLogger(__name__)


class CriticReplayBuffer:
    """Off-policy replay buffer for critic (value function) training only.

    Stores (state, return, episode_id, is_success) tuples; improves value-function
    sample efficiency without introducing off-policy bias to the policy gradient.
    Circular buffer with success-prioritized sampling and episode-staleness filtering.
    """

    def __init__(
        self,
        capacity: int = 50000,
        success_priority_ratio: float = 3.0,
        max_episode_age: int = 100,
        debug_mode: bool = False
    ):
        """success_priority_ratio: oversampling factor for successful episodes.
        max_episode_age: discard data older than this many episodes.
        """
        self.capacity = capacity
        self.success_priority_ratio = success_priority_ratio
        self.max_episode_age = max_episode_age
        self.debug_mode = debug_mode

        # Storage as lists, converted to arrays on sample
        self.o1_embeddings = []   # raw O1 node embeddings for CriticEmbedding
        self.o2_embeddings = []   # raw O2 node embeddings for CriticEmbedding
        self.geometries = []      # SO(3)-invariant geometry features
        self.returns = []
        self.episode_ids = []
        self.is_success = []

        # Stage 2: policy data for off-policy policy gradient
        self.actions = []
        self.log_probs_direction = []
        self.log_probs_magnitude = []
        self.log_probs_axis = []
        self.advantages = []

        # Circular buffer state
        self.ptr = 0
        self.full = False
        self.current_episode_id = 0

        # Statistics
        self.total_added = 0
        self.total_sampled = 0
        self.success_episodes_added = 0
        self.failure_episodes_added = 0

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.capacity if self.full else len(self.o1_embeddings)    

    def add_batch(
        self,
        o1_embeddings: np.ndarray,
        o2_embeddings: np.ndarray,
        geometries: np.ndarray,
        returns: np.ndarray,
        episode_success: bool,
        actions: List = None,
        log_probs_direction: np.ndarray = None,
        log_probs_magnitude: np.ndarray = None,
        log_probs_axis: np.ndarray = None,
        advantages: np.ndarray = None
    ):            
        """Add one episode's (state, return) pairs. actions and per-head log_probs
        are Stage-2-only. episode_success marks dissociation. Raises ValueError on
        mismatched lengths.
        """
        if isinstance(o1_embeddings, torch.Tensor):
            o1_embeddings = o1_embeddings.detach().cpu().numpy()
        if isinstance(o2_embeddings, torch.Tensor):
            o2_embeddings = o2_embeddings.detach().cpu().numpy()
        if isinstance(geometries, torch.Tensor):
            geometries = geometries.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()

        o1_embeddings = np.asarray(o1_embeddings, dtype=np.float32)
        o2_embeddings = np.asarray(o2_embeddings, dtype=np.float32)
        geometries = np.asarray(geometries, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32).flatten()

        if not (len(o1_embeddings) == len(returns) == len(o2_embeddings) == len(geometries)):
            raise ValueError(
                f"Critic input length mismatch: o1={len(o1_embeddings)}, "
                f"o2={len(o2_embeddings)}, geom={len(geometries)}, returns={len(returns)}. "
                f"All must have same number of transitions."
            )

        if len(o1_embeddings) == 0:
            return

        n_transitions = len(o1_embeddings)

        # Stage 2: convert policy data to numpy if provided
        has_policy_data = actions is not None
        if has_policy_data:
            if log_probs_direction is None or log_probs_magnitude is None or log_probs_axis is None:
                raise ValueError(
                    "Stage 2 replay requires all per-head log probs. "
                    "Provide log_probs_direction, log_probs_magnitude, and log_probs_axis."
                )
            if isinstance(log_probs_direction, torch.Tensor):
                log_probs_direction = log_probs_direction.detach().cpu().numpy()
            if isinstance(log_probs_magnitude, torch.Tensor):
                log_probs_magnitude = log_probs_magnitude.detach().cpu().numpy()
            if isinstance(log_probs_axis, torch.Tensor):
                log_probs_axis = log_probs_axis.detach().cpu().numpy()
            if advantages is not None and isinstance(advantages, torch.Tensor):
                advantages = advantages.detach().cpu().numpy()
            
            log_probs_direction = np.asarray(log_probs_direction, dtype=np.float32).flatten()
            log_probs_magnitude = np.asarray(log_probs_magnitude, dtype=np.float32).flatten()
            log_probs_axis = np.asarray(log_probs_axis, dtype=np.float32).flatten()
            if advantages is not None:
                advantages = np.asarray(advantages, dtype=np.float32).flatten()

        for i in range(n_transitions):
            if len(self.o1_embeddings) < self.capacity:
                # Buffer not full: append
                self.o1_embeddings.append(o1_embeddings[i].copy())
                self.o2_embeddings.append(o2_embeddings[i].copy())
                self.geometries.append(geometries[i].copy())
                self.returns.append(float(returns[i]))
                self.episode_ids.append(self.current_episode_id)
                self.is_success.append(episode_success)

                if has_policy_data:
                    self.actions.append(actions[i] if i < len(actions) else None)
                    self.log_probs_direction.append(float(log_probs_direction[i]))
                    self.log_probs_magnitude.append(float(log_probs_magnitude[i]))
                    self.log_probs_axis.append(float(log_probs_axis[i]))
                    self.advantages.append(float(advantages[i]) if advantages is not None else 0.0)
                else:
                    self.actions.append(None)
                    self.log_probs_direction.append(0.0)
                    self.log_probs_magnitude.append(0.0)
                    self.log_probs_axis.append(0.0)
                    self.advantages.append(0.0)
            else:
                # Buffer full: overwrite at pointer
                self.o1_embeddings[self.ptr] = o1_embeddings[i].copy()
                self.o2_embeddings[self.ptr] = o2_embeddings[i].copy()
                self.geometries[self.ptr] = geometries[i].copy()
                self.returns[self.ptr] = float(returns[i])
                self.episode_ids[self.ptr] = self.current_episode_id
                self.is_success[self.ptr] = episode_success

                if has_policy_data:
                    self.actions[self.ptr] = actions[i] if i < len(actions) else None
                    self.log_probs_direction[self.ptr] = float(log_probs_direction[i])
                    self.log_probs_magnitude[self.ptr] = float(log_probs_magnitude[i])
                    self.log_probs_axis[self.ptr] = float(log_probs_axis[i])
                    self.advantages[self.ptr] = float(advantages[i]) if advantages is not None else 0.0
                else:
                    self.actions[self.ptr] = None
                    self.log_probs_direction[self.ptr] = 0.0
                    self.log_probs_magnitude[self.ptr] = 0.0
                    self.log_probs_axis[self.ptr] = 0.0
                    self.advantages[self.ptr] = 0.0
                    
                self.ptr = (self.ptr + 1) % self.capacity
                self.full = True

        self.total_added += n_transitions
        if episode_success:
            self.success_episodes_added += 1
        else:
            self.failure_episodes_added += 1
        self.current_episode_id += 1

        if self.debug_mode:
            status = "SUCCESS" if episode_success else "FAILURE"
            policy_status = "with policy data" if has_policy_data else "critic only"
            _LOG.debug(f"[CriticReplay] Added {n_transitions} transitions from episode "
                  f"{self.current_episode_id - 1} ({status}, {policy_status}). "
                  f"Buffer: {len(self)}/{self.capacity}")

    def sample(
        self,
        batch_size: int,
        current_episode: int = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch with success-weighted prioritization and staleness
        filtering (current_episode=None disables staleness filtering). Returns a
        dict of 'o1_embeddings'/'o2_embeddings'/'geometries'/'returns' tensors, or
        None if there aren't enough valid samples.
        """
        buffer_size = len(self)

        if buffer_size < batch_size:
            if self.debug_mode:
                _LOG.debug(f"[CriticReplay] Buffer too small: {buffer_size} < {batch_size}")
            return None

        # Valid indices, filtered by staleness
        if current_episode is not None and self.max_episode_age > 0:
            min_episode = current_episode - self.max_episode_age
            valid_mask = np.array(self.episode_ids[:buffer_size]) >= min_episode
            valid_indices = np.where(valid_mask)[0]
        else:
            valid_indices = np.arange(buffer_size)

        if len(valid_indices) < batch_size:
            if self.debug_mode:
                _LOG.debug(f"[CriticReplay] Not enough valid indices: {len(valid_indices)} < {batch_size}")
            return None

        # Sampling weights: success episodes oversampled, then normalized
        weights = np.ones(len(valid_indices), dtype=np.float64)
        if self.success_priority_ratio > 1.0:
            for i, idx in enumerate(valid_indices):
                if self.is_success[idx]:
                    weights[i] = self.success_priority_ratio
        weights = weights / weights.sum()

        sample_idx_in_valid = np.random.choice(
            len(valid_indices),
            size=batch_size,
            replace=False,
            p=weights
        )
        sample_indices = valid_indices[sample_idx_in_valid]

        o1_batch = np.array([self.o1_embeddings[i] for i in sample_indices], dtype=np.float32)
        o2_batch = np.array([self.o2_embeddings[i] for i in sample_indices], dtype=np.float32)
        geom_batch = np.array([self.geometries[i] for i in sample_indices], dtype=np.float32)
        returns_batch = np.array([self.returns[i] for i in sample_indices], dtype=np.float32)

        self.total_sampled += batch_size

        if self.debug_mode:
            n_success = sum(1 for i in sample_indices if self.is_success[i])
            _LOG.debug(f"[CriticReplay] Sampled {batch_size} ({n_success} from successful episodes)")

        return {
            'o1_embeddings': torch.from_numpy(o1_batch),
            'o2_embeddings': torch.from_numpy(o2_batch),
            'geometries': torch.from_numpy(geom_batch),
            'returns': torch.from_numpy(returns_batch)
        }

    def sample_with_policy_data(
        self,
        batch_size: int,
        current_episode: int = None
    ) -> Optional[Dict[str, Any]]:
        """Sample a Stage-2 batch (states, returns, actions, per-head log probs,
        advantages). Returns None if the buffer is too small, no samples survive
        staleness filtering, or no policy data is available.
        """
        buffer_size = len(self)

        if buffer_size < batch_size:
            if self.debug_mode:
                _LOG.debug(f"[CriticReplay] Buffer too small for policy replay: {buffer_size} < {batch_size}")
            return None

        has_policy_data = (
            len(self.actions) >= buffer_size and
            any(a is not None for a in self.actions[:buffer_size])
        )

        if not has_policy_data:
            if self.debug_mode:
                _LOG.debug(f"[CriticReplay] No policy data available for Stage 2 replay")
            return None

        # Valid indices: filter by staleness AND presence of policy data
        valid_indices = []
        for i in range(buffer_size):
            if current_episode is not None and self.max_episode_age > 0:
                min_episode = current_episode - self.max_episode_age
                if self.episode_ids[i] < min_episode:
                    continue

            if self.actions[i] is None:
                continue

            valid_indices.append(i)

        valid_indices = np.array(valid_indices)

        if len(valid_indices) < batch_size:
            if self.debug_mode:
                _LOG.debug(f"[CriticReplay] Not enough valid indices with policy data: "
                      f"{len(valid_indices)} < {batch_size}")
            return None

        # Sampling weights with success priority, normalized
        weights = np.ones(len(valid_indices), dtype=np.float64)
        if self.success_priority_ratio > 1.0:
            for i, idx in enumerate(valid_indices):
                if self.is_success[idx]:
                    weights[i] = self.success_priority_ratio
        weights = weights / weights.sum()

        sample_idx_in_valid = np.random.choice(
            len(valid_indices),
            size=batch_size,
            replace=False,
            p=weights
        )
        sample_indices = valid_indices[sample_idx_in_valid]

        o1_batch = np.array([self.o1_embeddings[i] for i in sample_indices], dtype=np.float32)
        o2_batch = np.array([self.o2_embeddings[i] for i in sample_indices], dtype=np.float32)
        geom_batch = np.array([self.geometries[i] for i in sample_indices], dtype=np.float32)
        returns_batch = np.array([self.returns[i] for i in sample_indices], dtype=np.float32)
        actions_batch = [self.actions[i] for i in sample_indices]
        log_probs_dir_batch = np.array([self.log_probs_direction[i] for i in sample_indices], dtype=np.float32)
        log_probs_mag_batch = np.array([self.log_probs_magnitude[i] for i in sample_indices], dtype=np.float32)
        log_probs_axis_batch = np.array([self.log_probs_axis[i] for i in sample_indices], dtype=np.float32)
        advantages_batch = np.array([self.advantages[i] for i in sample_indices], dtype=np.float32)

        self.total_sampled += batch_size

        if self.debug_mode:
            n_success = sum(1 for i in sample_indices if self.is_success[i])
            _LOG.debug(f"[CriticReplay] Sampled {batch_size} with policy data "
                  f"({n_success} from successful episodes)")

        return {
            'o1_embeddings': torch.from_numpy(o1_batch),
            'o2_embeddings': torch.from_numpy(o2_batch),
            'geometries': torch.from_numpy(geom_batch),
            'returns': torch.from_numpy(returns_batch),
            'actions': actions_batch,
            'old_log_probs_direction': torch.from_numpy(log_probs_dir_batch),
            'old_log_probs_magnitude': torch.from_numpy(log_probs_mag_batch),
            'old_log_probs_axis': torch.from_numpy(log_probs_axis_batch),
            'advantages': torch.from_numpy(advantages_batch)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for logging."""
        buffer_size = len(self)

        if buffer_size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fill_ratio': 0.0,
                'success_ratio': 0.0,
                'total_added': self.total_added,
                'total_sampled': self.total_sampled
            }

        n_success = sum(self.is_success[:buffer_size])

        return {
            'size': buffer_size,
            'capacity': self.capacity,
            'fill_ratio': buffer_size / self.capacity,
            'success_ratio': n_success / buffer_size,
            'success_episodes': self.success_episodes_added,
            'failure_episodes': self.failure_episodes_added,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'current_episode_id': self.current_episode_id
        }

    def clear(self):
        """Clear the buffer (use with caution - loses all replay data)."""
        self.o1_embeddings.clear()
        self.o2_embeddings.clear()
        self.geometries.clear()
        self.returns.clear()        
        self.episode_ids.clear()
        self.is_success.clear()

        # Stage 2 policy data
        self.actions.clear()
        self.log_probs_direction.clear()
        self.log_probs_magnitude.clear()
        self.log_probs_axis.clear()
        self.advantages.clear()
        
        self.ptr = 0
        self.full = False
        # Don't reset episode_id counter or statistics
