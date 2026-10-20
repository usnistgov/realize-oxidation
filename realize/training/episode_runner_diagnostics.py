"""Per-episode diagnostics and logging."""
import torch
import numpy as np
import time
from ase import io
import logging
_LOG = logging.getLogger(__name__)


class EpisodeDiagnosticsMixin:
    """Structured per-episode diagnostics/logging for EpisodeRunner.

    ``self._log`` and all instance attributes resolve via the MRO back to
    EpisodeRunner, which is a plain orchestrator (no torch Parameters), so this
    mixin cannot perturb the policy parameter fingerprint.
    """

    def _log_channel_snapshot(self, episode: int, step: int):
        """Log channel statistics periodically (not every step)."""
        outputs = self.agent.policy._last_outputs
        channel_weights = outputs.get('channel_weights', None)

        if channel_weights is None:
            return

        # Convert to numpy
        if hasattr(channel_weights, 'detach'):
            cw = channel_weights.detach().cpu().numpy()
        else:
            cw = channel_weights

        # Validate finiteness (NO NaN/Inf)
        if not np.isfinite(cw).all():
            raise RuntimeError(
                f"channel_weights contains NaN/Inf at episode {episode}, step {step}. "
                f"This indicates numerical instability in channel computation."
            )

        if cw.ndim > 1:
            cw = cw.squeeze()

        # Validate probabilities
        if not np.all(cw >= 0):
            raise RuntimeError(f"channel_weights contains negative values: {cw}")

        cw_sum = cw.sum()
        if not (0.99 < cw_sum < 1.01):
            raise RuntimeError(
                f"channel_weights do not sum to 1: sum={cw_sum:.6f}. "
                f"Weights must be properly normalized."
            )

        # Calculate key metrics
        entropy = -np.sum(cw * np.log(cw + 1e-10))
        max_entropy = np.log(len(cw)) if cw.ndim > 0 else 0.0

        self._log('CHANNEL', f"Channel snapshot", _verbose=True,
                  episode=episode, step=step,
                  weights=str(cw.round(3)),
                  entropy=f"{entropy:.2f}/{max_entropy:.2f}",
                  dominant=f"ch{np.argmax(cw) if cw.ndim > 0 else 0}")

    def _log_episode_channel_summary(self, episode: int):
        """Log episode-level channel statistics (called periodically)."""
        # channel_usage_history must exist (initialized in __init__)
        if not hasattr(self, 'channel_usage_history'):
            raise RuntimeError(
                "channel_usage_history not initialized. "
                "This is a critical initialization error in __init__."
            )

        episode_history = [h for h in self.channel_usage_history if h['episode'] == episode]
        if not episode_history:
            _LOG.debug(f"No channel history recorded for episode {episode}")
            return  # OK to have no data for an episode

        all_weights = np.array([h['weights'] for h in episode_history])
        mean_weights = all_weights.mean(axis=0)

        dominant_counts = np.zeros(len(mean_weights))
        for h in episode_history:
            dominant_counts[h['dominant']] += 1

        entropies = [h['entropy'] for h in episode_history]

        _LOG.debug(f"\n[EPISODE {episode} CHANNEL SUMMARY]")
        _LOG.debug(f"  Steps tracked: {len(episode_history)}")
        _LOG.debug(f"  Avg weights: {mean_weights.round(3)}")
        _LOG.debug(f"  Dominant %: {(dominant_counts/len(episode_history)*100).round(1)}")
        _LOG.debug(f"  Channel gate entropy: mean={np.mean(entropies):.2f}, range=[{np.min(entropies):.2f}, {np.max(entropies):.2f}]")

    def _track_embedding_divergence(self, episode: int):
        """Track how much actor and critic embeddings diverge over time."""
        if episode % 20 != 0:  # Check every 20 episodes
            return

        if not hasattr(self, 'actor_embedding_extractor') or not hasattr(self, 'critic_embedding_extractor'):
            return

        try:
            # Get current atoms
            atoms = self.env.structure_manager.get_atoms()

            # Extract embeddings from both backbones
            with torch.no_grad():
                def _extract_symmetric(x, name: str) -> torch.Tensor:
                    """Extract symmetric embedding, handling both dict and tensor returns."""
                    # Accept dict with 'symmetric' or a raw tensor
                    if isinstance(x, dict):
                        if "symmetric" not in x:
                            raise RuntimeError(f"{name}: dict missing 'symmetric' (keys={list(x.keys())})")
                        x = x["symmetric"]

                    if not torch.is_tensor(x):
                        raise RuntimeError(f"{name}: expected torch.Tensor or dict, got {type(x)}")

                    # Canonicalize shape to [B, D] (strict)
                    if x.dim() == 1:
                        x = x.unsqueeze(0)              # [D] -> [1, D]
                    elif x.dim() == 3:
                        # Only allow [B, 1, D]
                        if x.shape[1] != 1:
                            raise RuntimeError(f"{name}: expected [B,1,D] if 3D, got {tuple(x.shape)}")
                        x = x.squeeze(1)                 # [B,1,D] -> [B,D]

                    if x.dim() != 2:
                        raise RuntimeError(f"{name}: expected 2D [B,D], got dim={x.dim()} shape={tuple(x.shape)}")

                    return x.detach()

                # --- actor embedding ---
                actor_out = self.actor_embedding_extractor.get_equivariant_embedding(
                    atoms, return_symmetric_only=True
                )
                actor_emb = _extract_symmetric(actor_out, "actor_embedding")

                # --- critic embedding ---
                critic_out = self.critic_embedding_extractor.get_equivariant_embedding(
                    atoms, return_symmetric_only=True
                )
                critic_emb = _extract_symmetric(critic_out, "critic_embedding")

            # Convert to numpy for analysis (flatten batch)
            actor_np = actor_emb.cpu().numpy().reshape(-1)
            critic_np = critic_emb.cpu().numpy().reshape(-1)

            # must match size
            if actor_np.shape != critic_np.shape:
                raise RuntimeError(
                    f"Embedding divergence check: shape mismatch actor={actor_np.shape} critic={critic_np.shape}"
                )

            # Compute divergence metrics
            l2_distance = np.linalg.norm(actor_np - critic_np)
            cosine_sim = np.dot(actor_np, critic_np) / (np.linalg.norm(actor_np) * np.linalg.norm(critic_np))
            relative_diff = l2_distance / (np.linalg.norm(actor_np) + 1e-8)

            # Determine status
            if cosine_sim > 0.999:
                status = "CRITICAL: embeddings nearly identical"
            elif relative_diff < 0.001:
                status = "CRITICAL: relative diff < 0.1%"
            else:
                status = "OK: diverging appropriately"

            self._log('BACKBONE', "Embedding divergence check", _verbose=True,
                      episode=episode,
                      l2_distance=f"{l2_distance:.6f}",
                      cosine_sim=f"{cosine_sim:.6f}",
                      relative_diff=f"{relative_diff:.6f}",
                      status=status)

            # Track history
            if not hasattr(self, '_embedding_divergence_history'):
                self._embedding_divergence_history = []

            self._embedding_divergence_history.append({
                'episode': episode,
                'l2_distance': l2_distance,
                'cosine_sim': cosine_sim,
                'relative_diff': relative_diff
            })

            # Trend analysis (if we have 3+ measurements)
            if len(self._embedding_divergence_history) >= 3:
                recent = self._embedding_divergence_history[-3:]
                l2_trend = [h['l2_distance'] for h in recent]

                if all(l2_trend[i] <= l2_trend[i-1] * 1.01 for i in range(1, len(l2_trend))):
                    self._log('WARNING', "L2 distance not increasing - backbone learning stalled?",
                              last_3_l2=str([f"{d:.4f}" for d in l2_trend]))

        except Exception as e:
            self._log('ERROR', f"Embedding divergence check failed: {e}")

    def _log_failure_mode(self, episode: int, success: bool, o2_removed: bool,
                          step_count: int, max_steps: int, invalid_step_count: int):
        """Classify why episodes fail to find the bottleneck."""
        if success:
            self._failure_modes['success'] += 1
        elif o2_removed:
            self._failure_modes['vacuum_escape'] += 1
        elif step_count >= max_steps:
            self._failure_modes['timeout'] += 1
        elif invalid_step_count > 10:
            self._failure_modes['invalid_step_heavy'] += 1
        else:
            self._failure_modes['high_barrier_repeated'] += 1

        # Log every 10 episodes
        if episode % 10 == 0 and episode > 0:
            total = sum(self._failure_modes.values())
            # Build stats dict
            failure_stats = {}
            for mode, count in self._failure_modes.items():
                pct = 100 * count / total if total > 0 else 0
                failure_stats[mode] = f"{count}({pct:.1f}%)"

            self._log('FAILURE', f"Failure mode summary (n={total})", _verbose=True, **failure_stats)

    def _log_o2_final_state(self, episode: int, success: bool):
        """Track O2 final position to see if it's reaching the surface."""
        try:
            from ase import io
            atoms = io.read(self.env.coo_file, format="lammps-data", atom_style="charge")
            positions = atoms.get_positions()

            # Last two atoms are O2
            if len(positions) < 2:
                return

            o1_pos = positions[-2]
            o2_pos = positions[-1]
            o2_com = (o1_pos + o2_pos) / 2

            # Estimate surface z (top of structure excluding O2)
            other_z = positions[:-2, 2]
            if len(other_z) == 0:
                return
            surface_z = np.percentile(other_z, 95)  # Top 5% as "surface"

            height_above_surface = o2_com[2] - surface_z

            key = 'success' if success else 'fail'
            self._o2_final_heights[key].append(height_above_surface)

            # Log every 10 episodes
            if episode % 10 == 0 and episode > 0:
                height_stats = {'episode': episode}
                for k, heights in self._o2_final_heights.items():
                    if len(heights) >= 3:
                        recent = heights[-20:] if len(heights) >= 20 else heights
                        height_stats[f"{k}_mean"] = f"{np.mean(recent):+.2f}Angstrom"
                        height_stats[f"{k}_range"] = f"[{np.min(recent):+.2f}, {np.max(recent):+.2f}]Angstrom"
                        height_stats[f"{k}_n"] = len(recent)

                self._log('HEIGHT', "O2 final height analysis", _verbose=True, **height_stats)

                # Warn if failed episodes have O2 far from surface
                if len(self._o2_final_heights['fail']) >= 5:
                    fail_mean = np.mean(self._o2_final_heights['fail'][-10:])
                    if fail_mean > 3.0:
                        self._log('WARNING', f"Failed episodes have O2 far from surface",
                                  mean_height=f"{fail_mean:.1f}Angstrom",
                                  interpretation="Policy may be avoiding surface")
        except Exception as e:
            pass  # Don't crash on diagnostic failure

    def _log_direction_adaptation(self, episode: int):
        """Track whether policy directions are adapting or stuck."""
        if len(self._direction_history) < 40:
            return

        early_dirs = np.array(self._direction_history[:20])
        recent_dirs = np.array(self._direction_history[-20:])

        # Mean direction change
        early_mean = early_dirs.mean(axis=0)
        early_norm = np.linalg.norm(early_mean)
        if early_norm > 1e-8:
            early_mean = early_mean / early_norm

        recent_mean = recent_dirs.mean(axis=0)
        recent_norm = np.linalg.norm(recent_mean)
        if recent_norm > 1e-8:
            recent_mean = recent_mean / recent_norm

        cos_sim = np.clip(np.dot(early_mean, recent_mean), -1, 1)
        angle_change = np.arccos(cos_sim) * 180 / np.pi

        # Direction variance (exploration measure)
        early_var = np.var(early_dirs, axis=0).sum()
        recent_var = np.var(recent_dirs, axis=0).sum()

        # Build status
        dir_stats = {
            'episode': episode,
            'angle_change': f"{angle_change:.1f}deg",
            'early_mean_dir': f"[{early_mean[0]:+.3f},{early_mean[1]:+.3f},{early_mean[2]:+.3f}]",
            'recent_mean_dir': f"[{recent_mean[0]:+.3f},{recent_mean[1]:+.3f},{recent_mean[2]:+.3f}]",
            'early_var': f"{early_var:.4f}",
            'recent_var': f"{recent_var:.4f}"
        }

        # Determine status
        if angle_change < 5.0 and recent_var < early_var * 0.5:
            dir_stats['status'] = "CRITICAL: stuck AND collapsing"
        elif angle_change < 5.0:
            dir_stats['status'] = "WARNING: direction barely changed"
        elif recent_var < early_var * 0.5:
            exploration_drop = (1 - recent_var/early_var)*100
            dir_stats['status'] = f"WARNING: exploration decreased {exploration_drop:.0f}%"
        else:
            dir_stats['status'] = "OK: adapting normally"

        self._log('DIRECTION', "Direction adaptation analysis", _verbose=True, **dir_stats)

    def _log_success_by_difficulty(self, episode: int, success: bool):
        """Track success rate vs environment difficulty (O accumulation), bucketed
        by tens of cumulative O. A cliff drop flags surface saturation."""
        cumulative_o = getattr(self.env, 'cumulative_dissociated_o_count', 0)

        # Bucket by 10s: 0-9, 10-19, 20-29, etc.
        bucket = (cumulative_o // 10) * 10

        if bucket not in self._success_by_o_count:
            self._success_by_o_count[bucket] = [0, 0]  # [attempts, successes]

        self._success_by_o_count[bucket][0] += 1
        if success:
            self._success_by_o_count[bucket][1] += 1

        # Log every 10 episodes
        if episode % 10 == 0 and episode > 0:
            _LOG.debug(f"\n[DIFFICULTY ANALYSIS] Episode {episode} (current O_count={cumulative_o}):")
            for b in sorted(self._success_by_o_count.keys()):
                attempts, successes = self._success_by_o_count[b]
                rate = successes / attempts if attempts > 0 else 0
                bar = '#' * int(rate * 20)  # 20 chars for 100%
                _LOG.debug(f"  O_count {b:2d}-{b+9:2d}: {successes:2d}/{attempts:2d} = {rate:5.1%} {bar}")

            # Detect cliff drop
            buckets = sorted(self._success_by_o_count.keys())
            if len(buckets) >= 2:
                rates = []
                for b in buckets:
                    a, s = self._success_by_o_count[b]
                    rates.append(s / a if a > 0 else 0)

                # Check for sudden drop (>30% decrease between adjacent buckets)
                for i in range(1, len(rates)):
                    if rates[i-1] > 0.4 and rates[i] < rates[i-1] - 0.3:
                        _LOG.debug(f"  WARNING CLIFF DROP detected at O_count ~{buckets[i]}: "
                              f"{rates[i-1]:.0%} -> {rates[i]:.0%}")
                        _LOG.debug(f"      This suggests surface saturation is blocking dissociation")

