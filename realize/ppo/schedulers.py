import math
from typing import Optional, Dict, List
import logging
_LOG = logging.getLogger(__name__)

class UnifiedAnnealingScheduler:
    """Unified scheduler for temperature and entropy annealing"""

    def __init__(self, sampling_config=None, algorithm_config=None):
        """sampling_config drives temperature annealing; algorithm_config drives
        entropy annealing.
        """
        self.sampling_config = sampling_config
        self.algorithm_config = algorithm_config

    def get_temperature(self, episode):
        """Current temperature under cosine annealing."""
        if self.sampling_config is None:
            return 1.0

        total = max(1, int(self.sampling_config.temperature_anneal_episodes))
        if episode >= total:
            return self.sampling_config.temperature_min

        progress = episode / self.sampling_config.temperature_anneal_episodes
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return self.sampling_config.temperature_min + (self.sampling_config.temperature_init - self.sampling_config.temperature_min) * cos_factor

    def get_entropy_coeff(self, episode):
        """Current entropy coefficient."""
        if self.algorithm_config is None:
            return 0.01

        if not getattr(self.algorithm_config, 'entropy_anneal', False):
            return self.algorithm_config.entropy_coeff

        if episode >= self.algorithm_config.entropy_anneal_episodes:
            return self.algorithm_config.final_entropy_coeff

        progress = episode / self.algorithm_config.entropy_anneal_episodes
        return self.algorithm_config.entropy_coeff * (1 - progress) + self.algorithm_config.final_entropy_coeff * progress

class PerChannelTemperatureScheduler:
    """Manages per-channel temperature scheduling for multi-channel SH"""

    def __init__(self, channel_config, num_channels):
        self.channel_config = channel_config
        self.num_channels = num_channels

        # Validate / default bounds
        if not hasattr(self.channel_config, "channel_temp_min"):
            self.channel_config.channel_temp_min = 0.1
        if not hasattr(self.channel_config, "channel_temp_max"):
            self.channel_config.channel_temp_max = 5.0
        if self.channel_config.channel_temp_min <= 0:
            raise ValueError(f"channel_temp_min must be > 0, got {self.channel_config.channel_temp_min}")
        if self.channel_config.channel_temp_min > self.channel_config.channel_temp_max:
            raise ValueError(
                f"channel_temp_min ({self.channel_config.channel_temp_min}) "
                f"> channel_temp_max ({self.channel_config.channel_temp_max})"
            )

        temp_scales = list(getattr(channel_config, 'channel_temp_scales', []) or [])
        if len(temp_scales) == 0:
            temp_scales = [1.0] * num_channels
            _LOG.info(f"WARNING: No channel_temp_scales provided, using 1.0 for all {num_channels} channels")
        elif len(temp_scales) < num_channels:
            # Pad to num_channels by repeating the last value
            temp_scales = temp_scales + [temp_scales[-1]] * (num_channels - len(temp_scales))
        else:
            temp_scales = temp_scales[:num_channels]

        self.base_temperatures = temp_scales[:num_channels]
        self.current_temperatures = list(self.base_temperatures)

    def update(self, episode: int, channel_health_metrics: Optional[Dict] = None):
        """Update per-channel temperatures from episode and optional channel-health
        metrics: underused channels warm up (explore), overused cool down (exploit).
        """
        anneal_factor = 0.995 ** (episode / 100)  # slow decay

        for i in range(self.num_channels):
            self.current_temperatures[i] = self.base_temperatures[i] * anneal_factor

            if channel_health_metrics and 'avg_channel_weights' in channel_health_metrics:
                weights = channel_health_metrics['avg_channel_weights']
                if i < len(weights):
                    usage = weights[i]

                    if usage < 1.0 / (self.num_channels * 2):
                        self.current_temperatures[i] *= 1.1
                    elif usage > 2.0 / self.num_channels:
                        self.current_temperatures[i] *= 0.9

            self.current_temperatures[i] = max(
                self.channel_config.channel_temp_min,
                min(self.current_temperatures[i], self.channel_config.channel_temp_max)
            )

    def get_temperatures(self) -> List[float]:
        """Get current per-channel temperatures (bounded)"""
        tmin = self.channel_config.channel_temp_min
        tmax = self.channel_config.channel_temp_max
        return [max(tmin, min(t, tmax)) for t in self.current_temperatures]


# ============================================================================
# Adaptive reward-midpoint schedule (reward-v2). Data-driven, bounded, floored.
# See manuscript "ADAPTIVE MIDPOINT SCHEDULE" and KEY MATHEMATICAL PROPERTY #25.
# ============================================================================
def adaptive_mu_schedule(
    mu_current,
    cycle_ea_effs,
    recent_medians,
    *,
    T=1000.0,
    nu_0=1e13,
    mu_quantile=0.25,
    mu_margin=0.198,
    mu_floor=0.80,
    max_decades_per_cycle=1.0,
    ratchet_release_cycles=3,
):
    """Adaptive reward-midpoint schedule (physical, bounded, data-driven).

    mu tracks a low quantile of the achieved Ea_eff distribution, tightening at
    most one rate-decade per cycle, floored at mu_floor. Monotone-tightening by
    default; permits bounded loosening only when the achieved median has risen for
    `ratchet_release_cycles` consecutive cycles (environment genuinely harder).

    cycle_ea_effs MUST be the reward-consistent TS-relative Ea_eff values
    (LSE over (E_min + E_barrier - E_0)), NOT bare NEB barriers.

    Returns (mu_next, info_dict).
    """
    import numpy as np
    k_B = 8.617333262e-5
    kT = k_B * T
    delta_max = max_decades_per_cycle * kT * np.log(10.0)
    info = {}

    if not cycle_ea_effs:
        info['reason'] = 'no_data_hold'
        return float(mu_current), info

    arr = np.asarray(cycle_ea_effs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        info['reason'] = 'no_finite_data_hold'
        return float(mu_current), info

    achieved_q = float(np.quantile(arr, mu_quantile))
    achieved_median = float(np.median(arr))
    mu_target = achieved_q - mu_margin
    info.update(achieved_quantile=achieved_q, achieved_median=achieved_median,
                mu_target=mu_target, delta_max=delta_max)

    if mu_target < mu_current:
        mu_next = max(mu_current - delta_max, mu_target, mu_floor)
        info['reason'] = 'tighten'
    else:
        rising = False
        if recent_medians is not None and len(recent_medians) >= ratchet_release_cycles:
            tail = list(recent_medians[-ratchet_release_cycles:]) + [achieved_median]
            rising = all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))
        if rising:
            mu_next = min(mu_current + delta_max, mu_target)
            info['reason'] = 'release_loosen'
        else:
            mu_next = mu_current
            info['reason'] = 'hold_ratchet'

    mu_next = max(mu_next, mu_floor)
    info['mu_floor_active'] = (mu_next <= mu_floor + 1e-9)
    return float(mu_next), info
