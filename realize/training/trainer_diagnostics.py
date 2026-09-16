import gc
import torch
import numpy as np
import logging
_LOG = logging.getLogger(__name__)


class PPOTrainerDiagnosticsMixin:
    """Training-health monitoring + memory cleanup + success-by-difficulty
    tracking for PPOTrainer.

    ``_monitor_training_health`` fires every 10 episodes,
    ``_log_success_by_difficulty`` fires per episode, and ``_cleanup_memory`` is
    invoked from the training loop. All resolve agent/config/env/device and the
    per-episode bookkeeping attributes through the MRO back to PPOTrainer (a plain
    orchestrator with no torch Parameters), so this mixin cannot perturb the policy
    fingerprint.
    """
    def _monitor_training_health(self, episode: int):
        """Monitor for issues like channel collapse."""
        if episode % 10 == 0:
            if hasattr(self.agent.policy, 'sh_head') and hasattr(self.agent.policy.sh_head, '_current_betas'):
                if hasattr(self.agent.policy, '_last_outputs'):
                    outputs = self.agent.policy._last_outputs
                    channel_weights = outputs.get('channel_weights', None)

                    if channel_weights is not None:
                        if torch.is_tensor(channel_weights):
                            weights = channel_weights.detach().cpu().numpy()
                        elif isinstance(channel_weights, np.ndarray):
                            weights = channel_weights
                        else:
                            raise TypeError(
                                f"channel_weights must be tensor or numpy array, got {type(channel_weights)}"
                            )

                        if weights.ndim > 2:
                            raise ValueError(f"channel_weights has too many dimensions: {weights.ndim}")

                        if weights.ndim == 2:
                            weights = weights[0]

                        if not np.all(np.isfinite(weights)):
                            raise ValueError("channel_weights contains non-finite values")
                        if not np.all(weights >= 0):
                            raise ValueError("channel_weights contains negative values")

                        weight_sum = weights.sum()
                        if not (0.99 < weight_sum < 1.01):
                            raise ValueError(
                                f"channel_weights do not sum to 1: sum={weight_sum:.6f}. "
                                f"Weights must be properly normalized."
                            )

                        entropy = -(weights * np.log(weights + 1e-10)).sum()
                        max_w = weights.max()

                        if max_w > 0.85:
                            self.logger.log_warning(f"Channel collapse: channel {weights.argmax()} dominates with {max_w:.1%}")

                if len(self.agent.training_rewards) >= 20:
                    recent = self.agent.training_rewards[-20:]
                    if all(r < -0.5 for r in recent):
                        self.logger.log_warning("Agent appears stuck in failure mode")

    def _cleanup_memory(self):
        """Periodic memory cleanup."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _log_success_by_difficulty(self, episode: int, success: bool):
        """Track success rate as function of environment difficulty."""
        cumulative_o = getattr(self.env.structure_manager, 'cumulative_dissociated_o_count', 0)
        self._cumulative_o_at_episode.append((episode, cumulative_o))
        bucket = (cumulative_o // 10) * 10

        if bucket not in self._success_by_o_count:
            self._success_by_o_count[bucket] = [0, 0]

        self._success_by_o_count[bucket][0] += 1
        if success:
            self._success_by_o_count[bucket][1] += 1

        if episode % 10 == 0 and episode > 0:
            reset_interval = getattr(self.config.runtime, 'reset_surface_every_n_episodes', 0)
            max_o_threshold = getattr(self.config.runtime, 'max_cumulative_o_before_reset', 0)

            # Build difficulty stats dict
            difficulty_stats = {}
            for b in sorted(self._success_by_o_count.keys()):
                attempts, successes = self._success_by_o_count[b]
                rate = successes / attempts if attempts > 0 else 0
                difficulty_stats[f"O_{b}-{b+9}"] = f"{successes}/{attempts}={rate:.1%}"

            buckets = sorted(self._success_by_o_count.keys())
            if len(buckets) >= 2:
                rates = []
                for b in buckets:
                    a, s = self._success_by_o_count[b]
                    rates.append(s / a if a > 0 else 0)

                for i in range(1, len(rates)):
                    if rates[i-1] > 0.4 and rates[i] < rates[i-1] - 0.3:
                        _LOG.info(f"  WARNING  CLIFF DROP detected at O_count ~{buckets[i]}: "
                              f"{rates[i-1]:.0%} -> {rates[i]:.0%}")
                        _LOG.info(f"      This suggests surface saturation is blocking dissociation")

            if len(self._cumulative_o_at_episode) >= 10:
                recent_o = [o for (ep, o) in self._cumulative_o_at_episode[-10:]]
                o_increase = recent_o[-1] - recent_o[0]
                _LOG.info(f"  O accumulation (last 10 eps): +{o_increase} atoms")

