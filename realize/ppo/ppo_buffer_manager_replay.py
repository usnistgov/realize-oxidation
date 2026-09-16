"""Critic/policy replay accessors for PPOBufferManager.

The four replay helpers (add_to_critic_replay / sample_critic_replay /
sample_policy_replay / get_replay_statistics) only ever touch
``self.critic_replay_buffer`` and ``self._replay_config`` (both created in
PPOBufferManager.__init__) plus the replay buffer's own methods, so they resolve
entirely through the MRO back to PPOBufferManager (a plain class). Nothing here
creates instance state or runs at construction time.

Because this module has no `from __future__ import annotations`, the signature
annotations (torch and the typing names Dict/Any/Optional/List) are evaluated at
class-body exec time, so those imports must be present here.
"""
import torch
from typing import Dict, Any, Optional, List


class PPOReplayMixin:
    def add_to_critic_replay(
        self, 
        states: torch.Tensor, 
        returns: torch.Tensor, 
        is_success: bool,
        actions: List = None,
        log_probs_direction: torch.Tensor = None,
        log_probs_magnitude: torch.Tensor = None,
        log_probs_axis: torch.Tensor = None,
        advantages: torch.Tensor = None
    ):
        """Add episode data to the critic replay buffer.

        Stage 1 stores states + returns; Stage 2 also stores actions and per-head
        log probs.
        """
        if self.critic_replay_buffer is None:
            return
        
        self.critic_replay_buffer.add_batch(
            states=states,
            returns=returns,
            episode_success=is_success,
            actions=actions,
            log_probs_direction=log_probs_direction,
            log_probs_magnitude=log_probs_magnitude,
            log_probs_axis=log_probs_axis,
            advantages=advantages
        )

    def sample_critic_replay(
        self,
        batch_size: int,
        current_episode: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample from the critic replay buffer (current_episode drives staleness
        filtering). Returns a dict with 'states'/'returns', or None if unavailable.
        """
        if self.critic_replay_buffer is None:
            return None

        min_size = self._replay_config.min_size_before_sampling if self._replay_config else 1000
        if len(self.critic_replay_buffer) < min_size:
            return None
    
        return self.critic_replay_buffer.sample(batch_size, current_episode)

    def sample_policy_replay(
        self,
        batch_size: int = None,
        current_episode: int = None
    ) -> Optional[Dict[str, Any]]:
        """Sample for Stage 2 policy training. Returns None if replay is disabled,
        the buffer is too small, or no policy data is available.
        """
        if self.critic_replay_buffer is None:
            return None

        if self._replay_config is None:
            return None

        if not self._replay_config.enable_policy_replay:
            return None

        if batch_size is None:
            batch_size = self._replay_config.replay_batch_size

        min_size = self._replay_config.min_size_before_sampling
        if len(self.critic_replay_buffer) < min_size:
            return None

        return self.critic_replay_buffer.sample_with_policy_data(
            batch_size=batch_size,
            current_episode=current_episode
        )
    def get_replay_statistics(self) -> Optional[Dict[str, Any]]:
        """Get replay buffer statistics for logging."""
        if self.critic_replay_buffer is None:
            return None
        return self.critic_replay_buffer.get_statistics()
