"""RolloutBuffer read-only analysis / diagnostics mixin.

Pure read-only reporting helpers: none override a BufferInterface
abstractmethod (add/clear/update_reward/sample/__len__ all stay in RolloutBuffer),
none create instance state (no __init__). They are mixed in via
    class RolloutBuffer(RolloutBufferAnalysisMixin, BufferInterface)
so every `buffer.get_*` / hasattr(buffer, ...) call resolves through the MRO.
"""
import numpy as np
import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import logging
_LOG = logging.getLogger(__name__)


class RolloutBufferAnalysisMixin:
    def get_success_statistics(self) -> Dict[str, Any]:
        """Success/failure episode stats: rate, counts, and step distributions."""
        if not self.episode_success:
            return {
                'total_episodes': 0,
                'success_count': 0,
                'failure_count': 0,
                'success_rate': 0.0,
                'success_steps': 0,
                'failure_steps': 0
            }
        
        success_count = sum(self.episode_success)
        failure_count = len(self.episode_success) - success_count

        success_steps = 0
        failure_steps = 0
        for (start, end), success in zip(self.episode_boundaries, self.episode_success):
            if success:
                success_steps += (end - start)
            else:
                failure_steps += (end - start)
        
        return {
            'total_episodes': len(self.episode_success),
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_count / len(self.episode_success) if self.episode_success else 0.0,
            'success_steps': success_steps,
            'failure_steps': failure_steps,
            'success_step_fraction': success_steps / (success_steps + failure_steps) if (success_steps + failure_steps) > 0 else 0.0
        }
    
    def validate_data_types(self) -> bool:
        """True if stored data has consistent (scalar) types for tensorization."""
        issues = []

        for i, lp in enumerate(self.log_probs):
            if isinstance(lp, (list, np.ndarray)):
                if np.array(lp).size != 1:
                    issues.append(f"log_probs[{i}] is not scalar: shape={np.array(lp).shape}")

        for i, sv in enumerate(self.state_values):
            if isinstance(sv, (list, np.ndarray)):
                if np.array(sv).size != 1:
                    issues.append(f"state_values[{i}] is not scalar: shape={np.array(sv).shape}")

        for i, r in enumerate(self.rewards):
            if isinstance(r, (list, np.ndarray)):
                if np.array(r).size != 1:
                    issues.append(f"rewards[{i}] is not scalar: shape={np.array(r).shape}")

        if issues:
            if self.debug_mode:
                _LOG.info(f"Data type validation failed:")
                for issue in issues[:5]:
                    _LOG.info(f"  {issue}")
                if len(issues) > 5:
                    _LOG.info(f"  ... and {len(issues)-5} more issues")
            return False
        
        return True

    def get_per_head_rewards(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-head reward streams (numpy) for separate policy training.

        Routing (episode_end = ea + dissoc):
        - direction: step PE reward + episode_end
        - magnitude: step PE reward + episode_end
        - axis: per-step axis_reward + episode_end

        Raises RuntimeError if reward component lists differ in length.
        """
        buffer_size = len(self.rewards)

        if not (len(self.ea_rewards) == buffer_size and
                len(self.dissociation_rewards) == buffer_size and
                len(self.axis_rewards) == buffer_size):
            raise RuntimeError(
                f"Reward component mismatch: "
                f"rewards={buffer_size}, ea_rewards={len(self.ea_rewards)}, "
                f"dissociation_rewards={len(self.dissociation_rewards)}, "
                f"axis_rewards={len(self.axis_rewards)}"
            )
    
        rewards_direction = np.zeros(buffer_size)
        rewards_magnitude = np.zeros(buffer_size)
        rewards_axis = np.zeros(buffer_size)

        for i in range(buffer_size):
            step_reward = self.rewards[i]                 # PE reward from env step
            ea_reward = self.ea_rewards[i]                # episode end
            dissoc_reward = self.dissociation_rewards[i]  # episode end
            axis_step_reward = self.axis_rewards[i]       # per-step axis reward
            episode_end_total = ea_reward + dissoc_reward

            rewards_direction[i] = step_reward + episode_end_total
            rewards_magnitude[i] = step_reward + episode_end_total
            rewards_axis[i] = axis_step_reward + episode_end_total

        return rewards_direction, rewards_magnitude, rewards_axis

    def get_per_head_log_probs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-head log-probs (numpy) for separate policy losses. Raises
        RuntimeError if per-head log-probs are not initialized.
        """
        if not (len(self.log_probs_direction) == len(self.log_probs) and
                len(self.log_probs_magnitude) == len(self.log_probs) and
                len(self.log_probs_axis) == len(self.log_probs)):
            raise RuntimeError(
                f"Per-head log-probs not initialized. "
                f"log_probs={len(self.log_probs)}, "
                f"log_probs_direction={len(self.log_probs_direction)}, "
                f"log_probs_magnitude={len(self.log_probs_magnitude)}, "
                f"log_probs_axis={len(self.log_probs_axis)}"
            )
    
        return (
            np.array(self.log_probs_direction),
            np.array(self.log_probs_magnitude),
            np.array(self.log_probs_axis)
        )

    def get_spherical_harmonics_data(self) -> Dict[str, Any]:
        """SH data for visualization, tolerant of ragged shapes."""
        sh_coeffs = [c for c in self.sh_coefficients if c is not None]
        trans_dirs = [d for d in self.translation_directions if d is not None]
        trans_mags = [m for m in self.translation_magnitudes if m is not None]

        data = {
            'sh_coefficients': sh_coeffs,
            'translation_directions': trans_dirs,
            'translation_magnitudes': trans_mags
        }

        data['temperatures'] = [t for t in self.temperatures if t is not None]
        data['num_channels'] = [n for n in self.num_channels_used if n is not None]
        data['channel_weights'] = [w for w in self.channel_weights if w is not None]

        if sh_coeffs:
            shapes = [c.shape for c in sh_coeffs if hasattr(c, 'shape')]
            if shapes and all(s == shapes[0] for s in shapes):
                # Uniform shape: stack and compute aggregate stats
                coeffs_array = np.array(sh_coeffs)
                data['stats'] = self._compute_sh_statistics(coeffs_array)
            else:
                # Ragged shapes: per-item stats
                data['stats'] = {
                    'num_samples': len(sh_coeffs),
                    'shapes': list(set(str(c.shape) for c in sh_coeffs if hasattr(c, 'shape'))),
                    'note': 'Ragged shapes - statistics computed per item'
                }
        
        return data
    
    def _compute_sh_statistics(self, coeffs_array: np.ndarray) -> Dict[str, Any]:
        """Statistics for an SH coefficients array."""
        stats = {'num_samples': len(coeffs_array)}

        if coeffs_array.ndim == 3:  # [batch, channels, sh_dim]
            stats['num_channels'] = coeffs_array.shape[1]
            stats['channel_stats'] = []

            for ch in range(coeffs_array.shape[1]):
                ch_coeffs = coeffs_array[:, ch, :]
                ch_mags = np.linalg.norm(ch_coeffs, axis=-1)
                stats['channel_stats'].append({
                    'mean': float(np.mean(ch_mags)),
                    'max': float(np.max(ch_mags)),
                    'std': float(np.std(ch_mags))
                })
        elif coeffs_array.ndim == 2:  # [batch, sh_dim]
            mags = np.linalg.norm(coeffs_array, axis=-1)
            stats['mean_magnitude'] = float(np.mean(mags))
            stats['max_magnitude'] = float(np.max(mags))
            stats['std_magnitude'] = float(np.std(mags))
        
        return stats
    
    def get_channel_diversity_stats(self) -> Dict[str, Any]:
        """Channel usage and entropy-based diversity score (ragged-tolerant)."""
        stats = {
            'total_steps': len(self.states),
            'channel_usage': {},
            'diversity_score': 0.0
        }

        valid_channels = [n for n in self.num_channels_used if n is not None]
        if valid_channels:
            unique_channels = set(valid_channels)
            for ch in unique_channels:
                stats['channel_usage'][ch] = valid_channels.count(ch) / len(valid_channels)

        valid_weights = [w for w in self.channel_weights if w is not None]
        if valid_weights:
            shapes = [w.shape if hasattr(w, 'shape') else None for w in valid_weights]
            unique_shapes = set(s for s in shapes if s is not None)

            if len(unique_shapes) == 1 and None not in unique_shapes:
                weights_array = np.array(valid_weights)
                if weights_array.ndim == 2:  # [steps, channels]
                    # Normalized-entropy diversity score
                    avg_weights = np.mean(weights_array, axis=0)
                    avg_weights = avg_weights / (np.sum(avg_weights) + 1e-8)
                    entropy = -np.sum(avg_weights * np.log(avg_weights + 1e-8))
                    max_entropy = np.log(len(avg_weights))
                    stats['diversity_score'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            else:
                stats['note'] = f"Inconsistent weight shapes: {unique_shapes}"
        
        return stats

