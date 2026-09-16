import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ase import io
import os
import gc
from typing import Dict, Tuple, Optional, Any

from realize.ppo.ppo_optimizer_setup import PPOSetupMixin
from realize.ppo.ppo_optimizer_advantage import PPOAdvantageMixin
from realize.ppo.ppo_optimizer_loss import PPOLossMixin
from realize.ppo.ppo_optimizer_gradflow import PPOGradientFlowMixin
from realize.ppo.ppo_optimizer_optimize import PPOOptimizeMixin
import logging
_LOG = logging.getLogger(__name__)

class RewardRunningMeanStd:
    """Running mean/std for reward normalization (Welford's online algorithm)."""
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # small initial count for stability

    def update(self, x: np.ndarray):
        """Update running statistics with a new batch of values."""
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 0:
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Merge batch statistics into the running statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        # Parallel variance update
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class PPOOptimizer(
    PPOSetupMixin,
    PPOAdvantageMixin,
    PPOLossMixin,
    PPOGradientFlowMixin,
    PPOOptimizeMixin,
):
    """PPO optimizer for sparse rewards: GAE(lambda) advantages, running-stat
    reward normalization, value clipping, and per-head advantages with success-
    signal routing. A plain orchestrator, NOT an nn.Module.
    """

    def __init__(self, policy, config, device, logger=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.logger = logger

        # Shortcut to the algorithm hyperparameter config
        if hasattr(config, 'algorithm'):
            self.algo_config = config.algorithm
        else:
            self.algo_config = config

        self._setup_optimizer()

        self.grad_norm_history = []
        self._old_sh_coeffs = None

        self.axis_grad_norms = []
        self.axis_param_update_norms = []

        self._grad_clip_count = 0
        self._grad_total_count = 0
        self._large_grad_episodes = []

        # KL baseline, fixed across PPO epochs within one update
        self._kl_baseline = None
        self._kl_baseline_ready = False

        self.axis_reward_fn = None

        self.reward_normalizer = RewardRunningMeanStd()
        self.return_normalizer = RewardRunningMeanStd()
        self.normalize_rewards = getattr(config, 'normalize_rewards', True)

        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)

        self.clip_value_loss = getattr(config, 'clip_value_loss', True)
        self.value_clip_range = getattr(config, 'value_clip_range', 1.0)

        self.replay_config = None
        if hasattr(config, 'replay'):
            self.replay_config = config.replay
            self._log("PPO", "Replay config loaded",
                      critic_replay=config.replay.enable_critic_replay,
                      weight=config.replay.critic_replay_weight)

        self._log("PPO", "Sparse reward improvements ACTIVE",
                  gae_lambda=self.gae_lambda,
                  normalize_rewards=self.normalize_rewards,
                  clip_value_loss=self.clip_value_loss,
                  value_clip_range=self.value_clip_range)

    def _log(self, category: str, message: str, _level: int = logging.INFO, **kwargs):
        """Emit a "[CATEGORY] message | k=v | ..." line.

        _level selects the logging level (normal=INFO is shown always; pass
        logging.DEBUG to demote a line to verbose-only). It is underscore-prefixed
        so it never collides with a display key in **kwargs.
        """
        kv_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_msg = f"[{category}] {message}"
        if kv_str:
            full_msg += f" | {kv_str}"
        _LOG.log(_level, full_msg)

    def _dlog(self, category: str, message: str, **kwargs):
        """Verbose-only sibling of _log(): emits at DEBUG level.

        Use for per-update diagnostic detail that should appear only in
        verbose mode. Equivalent to self._log(..., _level=logging.DEBUG).
        """
        self._log(category, message, _level=logging.DEBUG, **kwargs)

