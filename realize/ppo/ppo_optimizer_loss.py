import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any
import logging
_LOG = logging.getLogger(__name__)


class PPOLossMixin:
    """PPO loss computation (value/policy/KL/replay) + log-mixture-norm capture +
    critic overfit test for PPOOptimizer.

    These helpers are invoked from ``PPOOptimizer.optimize`` (which stays in the
    main class) via ``self``; config/device/policy and the per-head bookkeeping
    resolve through the MRO back to PPOOptimizer (not an nn.Module), so this mixin
    cannot perturb the policy fingerprint.
    """
    def _compute_value_loss(self, state_values, returns, old_values=None):
        """Value loss with optional clipping. Clipping bounds the per-update value
        change (analogous to PPO policy clipping), stabilizing training when the
        value function starts very wrong (common with sparse rewards).
        """
        state_values = state_values.reshape(-1)
        returns = returns.reshape(-1)

        if self.clip_value_loss and old_values is not None:
            old_values = old_values.reshape(-1)

            # Huber (delta=10): MSE for small errors, linear for outlier returns,
            # avoiding early value_loss explosions
            value_loss_unclipped = F.huber_loss(
                state_values, returns, reduction='none', delta=10.0
            )

            value_clipped = old_values + torch.clamp(
                state_values - old_values,
                -self.value_clip_range,
                self.value_clip_range
            )
            value_loss_clipped = F.huber_loss(
                value_clipped, returns, reduction='none', delta=10.0
            )

            # Pessimistic max bounds the value update
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            delta = (state_values - old_values).abs()
            clip_frac = (delta > self.value_clip_range).float().mean().item()
            max_delta = delta.max().item()
            target_gap = (returns - old_values).abs()
            needed_range_max = target_gap.max().item()
            needed_range_mean = target_gap.mean().item()
            self._dlog("VALUE", "Value clipping stats",
                      clip_frac=f"{clip_frac:.1%}",
                      mean_delta=f"{delta.mean().item():.4f}",
                      max_delta=f"{max_delta:.4f}",
                      needed_range_mean=f"{needed_range_mean:.4f}",
                      needed_range_max=f"{needed_range_max:.4f}",
                      clip_range=self.value_clip_range,
                      status="CHOKING" if clip_frac > 0.3 else "OK")
        else:
            # Huber (delta=10): MSE when |error|<10, linear above
            value_loss = F.huber_loss(state_values, returns, delta=10.0)

            with torch.no_grad():
                delta = (state_values - returns).abs()
                self._dlog("VALUE", "Value MSE stats",
                          mae=f"{delta.mean().item():.4f}",
                          maxae=f"{delta.max().item():.4f}",
                          v_mean=f"{state_values.mean().item():.4f}",
                          r_mean=f"{returns.mean().item():.4f}",
                          v_std=f"{state_values.std().item():.4f}",
                          r_std=f"{returns.std().item():.4f}",
                          scale_gap=f"{(returns.std() / (state_values.std() + 1e-8)).item():.1f}x")

        # Value loss tracking over recent calls
        if not hasattr(self, '_h5_loss_history'):
            self._h5_loss_history = []
        self._h5_loss_history.append(float(value_loss.item()) if hasattr(value_loss, 'item') else float(value_loss))
        if len(self._h5_loss_history) % 10 == 0:
            _recent = self._h5_loss_history[-10:]
            _trend = _recent[-1] - _recent[0] if len(_recent) > 1 else 0.0
            self._dlog('VALUE', 'Value loss tracking (H5)',
                      call_count=len(self._h5_loss_history),
                      recent_mean=f"{(sum(_recent)/len(_recent)):.4f}",
                      recent_min=f"{min(_recent):.4f}",
                      recent_max=f"{max(_recent):.4f}",
                      trend=f"{_trend:.4f}",
                      status='DECREASING' if _trend < -0.001 else 'FLAT_OR_UP')

        return value_loss * self.algo_config.value_loss_coeff

    def _critic_overfit_test(self, policy, batch_data, returns, episode_idx):
        """Diagnostic: run 200 critic-only gradient steps on a fixed batch with the
        policy frozen. If MSE doesn't drop substantially, there's a representation
        or graph issue. Restores critic params afterward.
        """
        self._dlog("CRITIC_TEST", "=== CRITIC OVERFIT TEST START ===", episode=episode_idx)

        o1_emb = batch_data['o1_embeddings'].to(self.device)
        o2_emb = batch_data['o2_embeddings'].to(self.device)
        geom = batch_data['geometries'].to(self.device)
        target_returns = returns.to(self.device).reshape(-1).detach()
        
        self._dlog("CRITIC_TEST", "Batch stats",
                  batch_size=target_returns.numel(),
                  returns_mean=f"{target_returns.mean().item():.4f}",
                  returns_std=f"{target_returns.std().item():.4f}",
                  returns_min=f"{target_returns.min().item():.4f}",
                  returns_max=f"{target_returns.max().item():.4f}")

        # Snapshot critic params for restoration after the test
        critic_params = {}
        for name, param in policy.named_parameters():
            if 'critic' in name or 'value' in name:
                critic_params[name] = param.data.clone()

        if not critic_params:
            self._dlog("CRITIC_TEST", "ERROR: No critic/value params found!", status="FAIL")
            return

        self._dlog("CRITIC_TEST", f"Found {len(critic_params)} critic params")

        # Temporary critic-only optimizer
        critic_param_list = [p for n, p in policy.named_parameters()
                            if 'critic' in n or 'value' in n]
        test_optimizer = torch.optim.Adam(critic_param_list, lr=1e-3)

        with torch.no_grad():
            values_init = policy._compute_value(o1_emb, o2_emb, geom).reshape(-1)
            mse_init = F.mse_loss(values_init, target_returns).item()
            ev_init = 1 - (target_returns - values_init).var().item() / (target_returns.var().item() + 1e-8)
        
        self._dlog("CRITIC_TEST", "Initial",
                  mse=f"{mse_init:.4f}",
                  ev=f"{ev_init:.4f}",
                  v_mean=f"{values_init.mean().item():.4f}",
                  v_std=f"{values_init.std().item():.4f}")

        mse_history = []
        for step in range(200):
            test_optimizer.zero_grad()
            values = policy._compute_value(o1_emb, o2_emb, geom).reshape(-1)
            loss = F.mse_loss(values, target_returns)
            loss.backward()
            test_optimizer.step()
            
            if step % 50 == 0 or step == 199:
                with torch.no_grad():
                    v = policy._compute_value(o1_emb, o2_emb, geom).reshape(-1)
                    mse_now = F.mse_loss(v, target_returns).item()
                    ev_now = 1 - (target_returns - v).var().item() / (target_returns.var().item() + 1e-8)
                    mse_history.append(mse_now)
                    self._dlog("CRITIC_TEST", f"Step {step}",
                              mse=f"{mse_now:.4f}",
                              ev=f"{ev_now:.4f}",
                              v_mean=f"{v.mean().item():.4f}",
                              v_std=f"{v.std().item():.4f}")

        mse_final = mse_history[-1]
        reduction = (mse_init - mse_final) / (mse_init + 1e-8)
        
        if reduction > 0.8:
            status = "PASS - critic CAN overfit (representation OK, problem is non-stationarity)"
        elif reduction > 0.3:
            status = "PARTIAL - critic partially overfits (representation marginal)"
        else:
            status = "FAIL - critic CANNOT overfit (broken graph, wrong inputs, or dead head)"
        
        self._dlog("CRITIC_TEST", "=== RESULT ===",
                  mse_init=f"{mse_init:.4f}",
                  mse_final=f"{mse_final:.4f}",
                  reduction=f"{reduction:.1%}",
                  status=status)

        # Restore original critic params (discard the overfit weights)
        for name, param in policy.named_parameters():
            if name in critic_params:
                param.data.copy_(critic_params[name])

        self._dlog("CRITIC_TEST", "Critic params restored to pre-test values")

    def _compute_critic_replay_loss(
        self,
        replay_batch: Dict[str, torch.Tensor],
        policy: nn.Module
    ) -> torch.Tensor:
        """Value loss on replay samples: extra value-function signal from past
        experience, without affecting the policy gradient.
        """
        # CriticReplayBuffer stores raw critic inputs
        if 'o1_embeddings' not in replay_batch:
            raise RuntimeError(
                "FAIL: critic replay batch missing o1_embeddings. "
                "CriticReplayBuffer must store o1_embeddings/o2_embeddings/geometries."
            )
        o1_emb = replay_batch['o1_embeddings'].to(self.device)
        o2_emb = replay_batch['o2_embeddings'].to(self.device)
        geom = replay_batch['geometries'].to(self.device)
        returns = replay_batch['returns'].to(self.device)

        # Value predictions via CriticEmbedding -> value_mlp (gradients flow)
        with torch.enable_grad():
            values = policy._compute_value(o1_emb, o2_emb, geom)

        values = values.reshape(-1)
        returns = returns.reshape(-1)

        # MSE, no clipping (old values unavailable for replay)
        replay_value_loss = F.mse_loss(values, returns)

        return replay_value_loss

    def _compute_policy_replay_loss(
        self,
        replay_batch: Dict[str, Any],
        policy: nn.Module,
        current_episode: int, 
        clip_ratio: float = 0.25 
    ) -> Dict[str, torch.Tensor]:            
        """Importance-sampled policy loss on replay samples. clip_ratio is tighter
        than on-policy because replay comes from older policies (larger shift).
        Returns per-head and total policy replay losses.
        """
        if 'o1_embeddings' not in replay_batch:
            raise RuntimeError(
                "FAIL: policy replay batch missing o1_embeddings. "
                "CriticReplayBuffer must store o1_embeddings/o2_embeddings/geometries."
            )
        o1_emb = replay_batch['o1_embeddings'].to(self.device)
        o2_emb = replay_batch['o2_embeddings'].to(self.device)
        geom = replay_batch['geometries'].to(self.device)

        if 'actor_states' in replay_batch:
            actor_states = replay_batch['actor_states'].to(self.device)
        else:
            actor_states = replay_batch['states'].to(self.device)

        actions = replay_batch['actions']
        old_log_dir = replay_batch['old_log_probs_direction'].to(self.device)
        old_log_mag = replay_batch['old_log_probs_magnitude'].to(self.device)
        old_log_axis = replay_batch['old_log_probs_axis'].to(self.device)

        # Use stored advantages if present, else compute returns - V
        if 'advantages' in replay_batch and replay_batch['advantages'] is not None:
            advantages = replay_batch['advantages'].to(self.device)
        else:
            returns = replay_batch['returns'].to(self.device)
            with torch.no_grad():
                values = policy._compute_value(o1_emb, o2_emb, geom)
                values = values.reshape(-1)
            advantages = returns - values

        # Reuse stored temperature so recomputed log probs match
        stored_temperatures = replay_batch.get('temperatures', None)

        # Stored O2 positions for frame alignment
        replay_o2_positions_list = []
        for act in actions:
            o2_pos = act.get('o2_positions', None)
            if o2_pos is not None:
                if isinstance(o2_pos, np.ndarray):
                    o2_pos = torch.from_numpy(o2_pos)
                replay_o2_positions_list.append(o2_pos.to(self.device).float())
        if replay_o2_positions_list and len(replay_o2_positions_list) == len(actions):
            replay_o2_positions = torch.stack(replay_o2_positions_list)
        else:
            replay_o2_positions = None

        env_multipoles_batch = batch_data.get('env_multipoles', None)
        _, state_values, _, per_head_new = policy.evaluate(
            actor_states,
            episode_idx=current_episode,
            old_actions=actions,
            state_uid=None,
            o1_embeddings=o1_emb,
            o2_embeddings=o2_emb,
            geometries=geom,
            stored_temperatures=stored_temperatures,
            o2_positions=replay_o2_positions,
            env_multipoles=env_multipoles_batch,
        )

        new_log_dir = per_head_new['direction'].to(self.device).reshape(-1)
        new_log_mag = per_head_new['magnitude'].to(self.device).reshape(-1)
        new_log_axis = per_head_new['axis'].to(self.device).reshape(-1)

        # Sampling-vs-recomputed logprob identity: must hold or PPO is incorrect
        if not hasattr(self, '_identity_test_count'):
            self._identity_test_count = 0

        if self._identity_test_count < 5:  # first 5 calls only
            self._identity_test_count += 1

            diff_dir = (new_log_dir - old_log_dir).abs()
            diff_mag = (new_log_mag - old_log_mag).abs()
            diff_axis = (new_log_axis - old_log_axis).abs()

            _LOG.debug(f"\n[TEST 0 - LOGPROB IDENTITY] Check #{self._identity_test_count}")
            _LOG.debug(f"  Direction: max_diff={diff_dir.max().item():.6f}, mean_diff={diff_dir.mean().item():.6f}")
            _LOG.debug(f"  Magnitude: max_diff={diff_mag.max().item():.6f}, mean_diff={diff_mag.mean().item():.6f}")
            _LOG.debug(f"  Axis:      max_diff={diff_axis.max().item():.6f}, mean_diff={diff_axis.mean().item():.6f}")

            # tol 1e-4 absorbs floating-point and temperature variation
            tol = 1e-4
            dir_pass = diff_dir.max().item() < tol
            mag_pass = diff_mag.max().item() < tol
            axis_pass = diff_axis.max().item() < tol

            _LOG.debug(f"  PASS: direction={dir_pass}, magnitude={mag_pass}, axis={axis_pass}")

            if not (dir_pass and mag_pass and axis_pass):
                # Log worst offender per head
                worst_dir_idx = diff_dir.argmax().item()
                _LOG.debug(f"  [DIRECTION MISMATCH] idx={worst_dir_idx}: stored={old_log_dir[worst_dir_idx].item():.6f}, recomputed={new_log_dir[worst_dir_idx].item():.6f}")

                worst_mag_idx = diff_mag.argmax().item()
                _LOG.debug(f"  [MAGNITUDE MISMATCH] idx={worst_mag_idx}: stored={old_log_mag[worst_mag_idx].item():.6f}, recomputed={new_log_mag[worst_mag_idx].item():.6f}")

                worst_axis_idx = diff_axis.argmax().item()
                _LOG.debug(f"  [AXIS MISMATCH] idx={worst_axis_idx}: stored={old_log_axis[worst_axis_idx].item():.6f}, recomputed={new_log_axis[worst_axis_idx].item():.6f}")

        # Truncate all to a common length
        min_len = min(
            old_log_dir.numel(), old_log_mag.numel(), old_log_axis.numel(),
            new_log_dir.numel(), new_log_mag.numel(), new_log_axis.numel(),
            advantages.numel()
        )

        old_log_dir = old_log_dir[:min_len]
        old_log_mag = old_log_mag[:min_len]
        old_log_axis = old_log_axis[:min_len]
        new_log_dir = new_log_dir[:min_len]
        new_log_mag = new_log_mag[:min_len]
        new_log_axis = new_log_axis[:min_len]
        advantages = advantages[:min_len]

        def clipped_ratio_loss(new_lp, old_lp, adv, clip):
            """PPO clipped surrogate with tight clipping for off-policy data."""
            new_lp = new_lp.clamp(min=-25.0, max=25.0)

            log_ratio = (new_lp - old_lp).clamp(-3.0, 3.0)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            return -torch.min(surr1, surr2).mean()

        loss_dir = clipped_ratio_loss(new_log_dir, old_log_dir, advantages, clip_ratio)
        loss_mag = clipped_ratio_loss(new_log_mag, old_log_mag, advantages, clip_ratio)
        loss_axis = clipped_ratio_loss(new_log_axis, old_log_axis, advantages, clip_ratio)

        # Ratio sanity check
        if not hasattr(self, '_ratio_test_count'):
            self._ratio_test_count = 0
        
        if self._ratio_test_count < 5:
            self._ratio_test_count += 1
            
            with torch.no_grad():
                ratio_dir = torch.exp((new_log_dir - old_log_dir).clamp(-3.0, 3.0))
                ratio_mag = torch.exp((new_log_mag - old_log_mag).clamp(-3.0, 3.0))
                ratio_axis = torch.exp((new_log_axis - old_log_axis).clamp(-3.0, 3.0))
                
                _LOG.debug(f"\n[TEST 1 - RATIO SANITY] Check #{self._ratio_test_count}")
                _LOG.debug(f"  Direction ratio: mean={ratio_dir.mean().item():.4f}, std={ratio_dir.std().item():.4f}, range=[{ratio_dir.min().item():.4f}, {ratio_dir.max().item():.4f}]")
                _LOG.debug(f"  Magnitude ratio: mean={ratio_mag.mean().item():.4f}, std={ratio_mag.std().item():.4f}, range=[{ratio_mag.min().item():.4f}, {ratio_mag.max().item():.4f}]")
                _LOG.debug(f"  Axis ratio:      mean={ratio_axis.mean().item():.4f}, std={ratio_axis.std().item():.4f}, range=[{ratio_axis.min().item():.4f}, {ratio_axis.max().item():.4f}]")
                
                if ratio_dir.std().item() > 1.0 or ratio_dir.max().item() > 5.0:
                    _LOG.info(f"  WARNING: Direction ratios have large variance/tails - likely mismatch!")

        # Head-importance weighting (same as on-policy)
        dir_weight = 1.0
        mag_weight = 0.5
        axis_weight = 0.05

        total = dir_weight * loss_dir + mag_weight * loss_mag + axis_weight * loss_axis

        return {
            'direction': loss_dir,
            'magnitude': loss_mag,
            'axis': loss_axis,
            'total': total
        }

    def _compute_losses(
        self,
        per_head_new: Dict[str, torch.Tensor],
        per_head_old: Dict[str, torch.Tensor],
        per_head_adv: Dict[str, torch.Tensor],
        state_values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor],
        entropy: torch.Tensor,
        entropy_coeff: float,
        episode_idx: int,
        ke: int = 0,
        k_epochs: int = 1,
        should_log: bool = False, 
    ):
        losses = {}

        # Detailed diagnostics only on first/last epoch of logged episodes
        should_print = should_log and (ke == 0 or ke == k_epochs - 1)

        def ppo_clip_loss(new_logprob, old_logprob, advantages, clip_ratio=0.25):
            with torch.no_grad():
                _lp_diff = (new_logprob - old_logprob)
                _ratio = torch.exp(_lp_diff.clamp(-3.0, 3.0))
                _LOG.debug(f'[ADV4_CLIP_LOSS] adv_std={advantages.std():.6f} '
                      f'adv_mean={advantages.mean():.4f} '
                      f'ratio_mean={_ratio.mean():.4f} '
                      f'ratio_std={_ratio.std():.4f} '
                      f'lp_diff_std={_lp_diff.std():.4f}')
            """Compute PPO clipped surrogate loss."""

            if not torch.isfinite(new_logprob).all():
                bad_idx = (~torch.isfinite(new_logprob)).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"new_logprob contains NaN/Inf at indices {bad_idx.tolist()}\n"
                    f"Values: {new_logprob[~torch.isfinite(new_logprob)].tolist()}"
                )
            
            if not torch.isfinite(old_logprob).all():
                bad_idx = (~torch.isfinite(old_logprob)).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"old_logprob contains NaN/Inf at indices {bad_idx.tolist()}\n"
                    f"Values: {old_logprob[~torch.isfinite(old_logprob)].tolist()}"
                )
            
            if not torch.isfinite(advantages).all():
                bad_idx = (~torch.isfinite(advantages)).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"advantages contains NaN/Inf at indices {bad_idx.tolist()}\n"
                    f"Values: {advantages[~torch.isfinite(advantages)].tolist()}"
                )
            
            # log-prob ratio; keep raw for diagnostics, clamp for stability
            log_ratios_raw = new_logprob - old_logprob
            log_ratios = log_ratios_raw.clamp(min=-3.0, max=3.0)

            # log_ratio std distinguishes frozen vs modestly-changing policy
            if not hasattr(self, '_logratio_diagnostic_printed'):
                self._logratio_diagnostic_printed = set()

            if episode_idx not in self._logratio_diagnostic_printed:
                self._logratio_diagnostic_printed.add(episode_idx)
                if len(self._logratio_diagnostic_printed) > 50:
                    min_keep = episode_idx - 50
                    self._logratio_diagnostic_printed = {e for e in self._logratio_diagnostic_printed if e >= min_keep}

                frac_clamped_low = (log_ratios_raw < -3.0).float().mean().item()
                frac_clamped_high = (log_ratios_raw > 3.0).float().mean().item()

                log_ratio_std = log_ratios_raw.std().item()
                if log_ratio_std < 0.001:
                    interp_status = "FROZEN"
                elif log_ratio_std < 0.05:
                    interp_status = "MODEST"
                elif log_ratio_std < 0.20:
                    interp_status = "NORMAL"
                else:
                    interp_status = "AGGRESSIVE"
                
                self._dlog("PPO", "Log-ratio diagnostic",
                          episode=episode_idx,
                          logratio_mean=f"{log_ratios_raw.mean():.6f}",
                          logratio_std=f"{log_ratios_raw.std():.6f}",
                          logratio_min=f"{log_ratios_raw.min():.6f}",
                          logratio_max=f"{log_ratios_raw.max():.6f}",
                          clamped_low=f"{frac_clamped_low:.1%}",
                          clamped_high=f"{frac_clamped_high:.1%}",
                          interpretation=interp_status)

            # Safety guard: should never trigger after the clamp above
            if (log_ratios > 100).any():
                bad_idx = (log_ratios > 100).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"log_ratios too large (>100) at indices {bad_idx.tolist()}\n"
                    f"This causes exp(log_ratio) = inf. Check old_logprob computation."
                )
            
            ratios = torch.exp(log_ratios)

            if not hasattr(self, '_printed_ratio_debug'):
                self._printed_ratio_debug = set()

            should_print_ratio = (episode_idx % 10 == 0 and episode_idx not in self._printed_ratio_debug)
            if should_print_ratio:
                self._printed_ratio_debug.add(episode_idx)
                if len(self._printed_ratio_debug) > 100:
                    min_to_keep = episode_idx - 100
                    self._printed_ratio_debug = {e for e in self._printed_ratio_debug if e >= min_to_keep}

                ratio_mean = ratios.mean().item()
                ratio_std = ratios.std().item()
                ratio_min = ratios.min().item()
                ratio_max = ratios.max().item()

                below_clip = (ratios < 1.0 - clip_ratio).sum().item()
                above_clip = (ratios > 1.0 + clip_ratio).sum().item()
                total = len(ratios)
                clip_frac = (below_clip + above_clip) / total

                if clip_frac > 0.5:
                    clip_status = "HIGH_CLIP"
                elif ratio_max > 100.0:
                    clip_status = "RATIO_EXPLOSION"
                else:
                    clip_status = "OK"

                self._dlog("PPO", "Ratio check",
                          episode=episode_idx,
                          ratio_mean=f"{ratio_mean:.3f}",
                          ratio_std=f"{ratio_std:.3f}",
                          ratio_range=f"[{ratio_min:.3f}, {ratio_max:.3f}]",
                          below_clip=f"{below_clip}/{total}",
                          above_clip=f"{above_clip}/{total}",
                          clip_frac=f"{clip_frac:.1%}",
                          status=clip_status)

            if not torch.isfinite(ratios).all():
                bad_idx = (~torch.isfinite(ratios)).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"ratios contains NaN/Inf at indices {bad_idx.tolist()}\n"
                    f"log_ratios at those indices: {log_ratios[~torch.isfinite(ratios)].tolist()}"
                )
            
            surr1 = ratios * advantages

            if not torch.isfinite(surr1).all():
                bad_idx = (~torch.isfinite(surr1)).nonzero(as_tuple=False).squeeze()
                raise RuntimeError(
                    f"surr1 (ratios * advantages) contains NaN/Inf at indices {bad_idx.tolist()}\n"
                    f"ratios: {ratios[~torch.isfinite(surr1)].tolist()}\n"
                    f"advantages: {advantages[~torch.isfinite(surr1)].tolist()}"
                )
            
            surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            loss = -torch.min(surr1, surr2).mean()
            
            return loss

        # Verify direction log_probs change between old/new (delta ~0 => index mismatch)
        if should_print:
            delta_dir = per_head_new['direction'] - per_head_old['direction']
            delta_axis = per_head_new['axis'] - per_head_old['axis']
            
            dir_status = "OK" if delta_dir.abs().max() >= 1e-5 else "IDENTICAL"
            if delta_dir.std() < 0.001 and dir_status == "OK":
                dir_status = "LOW_VARIANCE"
            
            self._dlog("PPO", "Direction delta check",
                      dir_delta_mean=f"{delta_dir.mean():.6f}",
                      dir_delta_std=f"{delta_dir.std():.6f}",
                      dir_delta_max_abs=f"{delta_dir.abs().max():.6f}",
                      axis_delta_mean=f"{delta_axis.mean():.6f}",
                      axis_delta_std=f"{delta_axis.std():.6f}",
                      axis_delta_max_abs=f"{delta_axis.abs().max():.6f}",
                      status=dir_status)

        # Per-head policy losses (new vs old per head)
        loss_policy_dir  = ppo_clip_loss(per_head_new['direction'], per_head_old['direction'], per_head_adv['direction'])
        loss_policy_mag  = ppo_clip_loss(per_head_new['magnitude'], per_head_old['magnitude'], per_head_adv['magnitude'])
        loss_policy_axis = ppo_clip_loss(per_head_new['axis'],      per_head_old['axis'],      per_head_adv['axis'])

        # Direct axis gradient test: do axis params get gradients from axis loss alone?
        if should_print:
            axis_params_list = list(self.policy.axis_from_frame.parameters())
            saved_grads = [p.grad.clone() if p.grad is not None else None for p in axis_params_list]

            for p in axis_params_list:
                if p.grad is not None:
                    p.grad.zero_()

            if loss_policy_axis.requires_grad:
                test_grad = torch.autograd.grad(
                    loss_policy_axis, 
                    axis_params_list, 
                    retain_graph=True,
                    allow_unused=True
                )
                param_names = ['frame_proj', 'o2_proj', 'combiner.0', 'combiner.1', 'combiner.2']
                for i, (name, g) in enumerate(zip(param_names, test_grad)):
                    if g is None:
                        self._dlog("AXIS", f"Direct gradient test: {name}", status="NOT_CONNECTED")
                    elif g.abs().max() == 0:
                        self._dlog("AXIS", f"Direct gradient test: {name}", status="DEAD")
                    else:
                        self._dlog("AXIS", f"Direct gradient test: {name}",
                                  grad_norm=f"{g.norm():.6e}",
                                  status="OK")
            else:
                self._dlog("AXIS", "Direct gradient test: loss_policy_axis has no grad_fn", status="ERROR")

            for p, saved in zip(axis_params_list, saved_grads):
                p.grad = saved

        if should_print:
            axis_diff = (per_head_new['axis'] - per_head_old['axis']).abs()
            
            if per_head_adv['axis'].abs().max() < 1e-4:
                adv_status = "NEAR_ZERO"
            else:
                adv_status = "OK"
            
            if axis_diff.max() < 1e-6:
                axis_status = "IDENTICAL"
            else:
                axis_status = "VARYING"
            
            self._dlog("AXIS", "Loss gradient debug",
                      new_requires_grad=per_head_new['axis'].requires_grad,
                      new_grad_fn=str(per_head_new['axis'].grad_fn)[:30] if per_head_new['axis'].grad_fn else "None",
                      old_requires_grad=per_head_old['axis'].requires_grad,
                      loss_requires_grad=loss_policy_axis.requires_grad,
                      axis_diff_mean=f"{axis_diff.mean():.6f}",
                      axis_diff_max=f"{axis_diff.max():.6f}",
                      adv_mean=f"{per_head_adv['axis'].mean():.6f}",
                      adv_std=f"{per_head_adv['axis'].std():.6f}",
                      adv_status=adv_status,
                      axis_status=axis_status)

        # Log_prob ranges (helps diagnose high losses)
        if should_print:
            self._dlog("PPO", "Log_prob ranges",
                      episode=episode_idx,
                      epoch=f"{ke}/{k_epochs}",
                      dir_old=f"[{per_head_old['direction'].min().item():.1f}, {per_head_old['direction'].max().item():.1f}]",
                      dir_new=f"[{per_head_new['direction'].min().item():.1f}, {per_head_new['direction'].max().item():.1f}]",
                      mag_old=f"[{per_head_old['magnitude'].min().item():.1f}, {per_head_old['magnitude'].max().item():.1f}]",
                      mag_new=f"[{per_head_new['magnitude'].min().item():.1f}, {per_head_new['magnitude'].max().item():.1f}]",
                      axis_old=f"[{per_head_old['axis'].min().item():.1f}, {per_head_old['axis'].max().item():.1f}]",
                      axis_new=f"[{per_head_new['axis'].min().item():.1f}, {per_head_new['axis'].max().item():.1f}]")
            
            # Ratios should sit ~0.5-2.0 for stable PPO
            dir_ratio = torch.exp(per_head_new['direction'] - per_head_old['direction'])
            self._dlog("PPO", "Direction ratios",
                      range=f"[{dir_ratio.min().item():.4f}, {dir_ratio.max().item():.4f}]")

        # Regularize vMF kappa via its entropy: high kappa = sharp = low entropy,
        # penalized to discourage overconfidence
        if hasattr(self.policy, '_last_outputs') and 'log_concentration' in self.policy._last_outputs:
            log_concentration = self.policy._last_outputs['log_concentration']
            concentration = torch.exp(log_concentration)

            # entropy ~ -log(kappa) for kappa > 1; subtract it (favor high entropy = low kappa)
            entropy_kappa = torch.log(concentration.clamp(min=1e-6))

            entropy_reg_strength = 0.02  # controls kappa sharpness
            loss_kappa_entropy_reg = entropy_reg_strength * entropy_kappa.mean()

            # Hard safety ceiling on kappa for numerical stability
            max_kappa_hard = 35.0
            kappa_excess = torch.clamp(concentration - max_kappa_hard, min=0.0)
            # Only enforce during early exploration (first 50 episodes)
            if episode_idx < 50:
                loss_kappa_safety = 100.0 * kappa_excess.mean()
            else:
                loss_kappa_safety = torch.tensor(0.0, device=self.device)

            loss_kappa_reg = loss_kappa_entropy_reg + loss_kappa_safety

            if not torch.isfinite(loss_kappa_reg):
                raise RuntimeError(
                    f"kappa regularization loss is non-finite: {loss_kappa_reg}\n"
                    f"concentration values: {concentration.tolist()}\n"
                    f"Check concentration computation in policy."
                )

            loss_policy_axis = loss_policy_axis + loss_kappa_reg
        
        losses['policy_direction'] = loss_policy_dir
        losses['policy_magnitude'] = loss_policy_mag
        losses['policy_axis']      = loss_policy_axis

        dir_weight = 1.0
        mag_weight = 0.5
        axis_weight = 0.05

        losses['policy'] = (dir_weight * loss_policy_dir +
                           mag_weight * loss_policy_mag +
                           axis_weight * loss_policy_axis)

        # Policy loss coefficient balances policy vs value loss
        policy_loss_coeff = getattr(self.algo_config, 'policy_loss_coeff', 1.0)
        losses['policy'] = policy_loss_coeff * losses['policy']

        # Channel entropy regularization to encourage learned specialization
        channel_weights = self.policy._last_outputs.get('channel_weights', None)

        if channel_weights is not None:
            # H = -sum p_c log p_c; high = uniform, low = specialized
            channel_entropy = -torch.sum(
                channel_weights * torch.log(channel_weights.clamp(min=1e-12)),
                dim=-1
            ).mean()

            # Negative sign minimizes entropy (favors non-uniform channel weights)
            losses['channel_entropy_reg'] = -0.1 * channel_entropy
            
            if episode_idx % 50 == 0:
                self._dlog("CHANNEL", "Channel entropy regularization",
                          entropy=f"{channel_entropy.item():.4f}",
                          reg_loss=f"{losses['channel_entropy_reg'].item():.6f}")
        else:
            losses['channel_entropy_reg'] = torch.tensor(0.0, device=self.device)

        # Channel repulsion: penalize cos^2 similarity between channel pairs,
        # preventing mode collapse (sim ~+1) and cancellation collapse (sim ~-1).
        # Softmax weighting lets the worst pairs dominate the loss.
        sh_coeffs = self.policy._last_outputs.get('sh_coeffs', None) if hasattr(self.policy, '_last_outputs') else None
        if sh_coeffs is not None and sh_coeffs.dim() == 3 and sh_coeffs.shape[1] > 1:
            num_ch = sh_coeffs.shape[1]
            channel_sh_dims = None
            try:
                channel_sh_dims = self.policy.sh_head.channel_sh_dims
            except AttributeError:
                pass
            pair_cos2_list = []
            pair_sims = []
            for i in range(num_ch):
                for j in range(i + 1, num_ch):
                    if channel_sh_dims is not None:
                        trim = channel_sh_dims[min(i, j)]
                    else:
                        trim = sh_coeffs.shape[2]
                    ci = sh_coeffs[:, i, :trim]
                    cj = sh_coeffs[:, j, :trim]
                    cos_sim = F.cosine_similarity(ci, cj, dim=-1)
                    cos2 = cos_sim.pow(2).mean()
                    pair_cos2_list.append(cos2)
                    pair_sims.append(cos_sim.mean().item())
            if pair_cos2_list:
                pair_cos2 = torch.stack(pair_cos2_list)
                # Temperature 0.1 focuses the softmax on the worst (highest cos^2) pairs
                weights = torch.softmax(pair_cos2 / 0.1, dim=0)
                repulsion_loss = (weights * pair_cos2).sum()
            else:
                repulsion_loss = torch.tensor(0.0, device=self.device)
            repulsion_coeff = 0.4
            losses['channel_repulsion'] = repulsion_coeff * repulsion_loss
            if should_print:
                abs_sims = [abs(s) for s in pair_sims]
                self._dlog("CHANNEL", "Repulsion loss",
                          mean_cos2=f"{sum(s**2 for s in pair_sims)/len(pair_sims):.4f}",
                          max_abs_sim=f"{max(abs_sims):.4f}",
                          pair_sims=f"[{', '.join(f'{s:.3f}' for s in pair_sims)}]",
                          loss=f"{losses['channel_repulsion'].item():.6f}")
        else:
            losses['channel_repulsion'] = torch.tensor(0.0, device=self.device)

        # Value loss with optional clipping
        losses['value'] = self._compute_value_loss(state_values, returns, old_values)

        if should_print:
            with torch.no_grad():
                self._log("VALUE", "Returns/Values diagnostic",
                          batch_size=returns.numel(),
                          returns_mean=f"{returns.mean().item():.4f}",
                          returns_std=f"{returns.std().item():.4f}",
                          returns_min=f"{returns.min().item():.4f}",
                          returns_max=f"{returns.max().item():.4f}",
                          values_mean=f"{state_values.mean().item():.4f}",
                          values_std=f"{state_values.std().item():.4f}",
                          mse=f"{F.mse_loss(state_values, returns).item():.4f}")

                if old_values is not None:
                    delta = (state_values - old_values)
                    clip_frac = (delta.abs() > self.value_clip_range).float().mean().item()
                    self._dlog("VALUE", "Value clip diagnostic",
                              clip_range=self.value_clip_range,
                              clip_frac=f"{clip_frac:.3f}",
                              delta_mean=f"{delta.mean().item():.4f}",
                              delta_std=f"{delta.std().item():.4f}",
                              delta_max=f"{delta.abs().max().item():.4f}")

                # EV inline, using the same tensors as the value loss
                _ret_var = returns.var().item()
                if _ret_var > 1e-6:
                    _ev_inline = 1 - (returns - state_values).var().item() / _ret_var
                    self._log("VALUE", "EV inline (same tensors as loss)",
                              ev_inline=f"{_ev_inline:.4f}",
                              returns_std=f"{returns.std().item():.4f}",
                              values_std=f"{state_values.std().item():.4f}")

        # Entropy must be a proper tensor with gradient connection
        if not isinstance(entropy, torch.Tensor):
            raise RuntimeError(
                f"entropy must be a tensor, got {type(entropy)}. "
                f"policy.evaluate() is returning wrong type."
            )
        
        if entropy.grad_fn is None and not entropy.requires_grad:
            raise RuntimeError(
                f"entropy has no grad_fn and requires_grad=False. "
                f"Entropy is detached from computation graph - entropy bonus won't train anything! "
                f"Check policy.evaluate() entropy computation."
            )
        
        entropy = entropy.to(self.device)

        entropy_mean = entropy.mean()

        # Entropy floor: quadratic penalty below the floor during early exploration,
        # then off so the policy can converge toward deterministic optimal.
        entropy_floor = 3.0
        entropy_floor_active = False

        if entropy_floor_active and entropy_mean < entropy_floor:
            entropy_deficit = entropy_floor - entropy_mean
            entropy_penalty = 10.0 * (entropy_deficit ** 2)

            losses['entropy'] = -entropy_coeff * entropy_mean + entropy_penalty

            if episode_idx % 10 == 0:
                self._dlog("PPO", "Entropy floor triggered",
                          entropy=f"{entropy_mean.item():.2f}",
                          floor=entropy_floor,
                          penalty=f"{entropy_penalty.item():.4f}")
        else:
            # Standard entropy bonus
            losses['entropy'] = -entropy_coeff * entropy_mean
            if episode_idx == 100:
                self._dlog("PPO", "Entropy floor DISABLED (episode 100 reached)")

        # KL regularization (directional mixture)
        losses['kl'] = self._compute_kl_loss(episode_idx) if self.algo_config.use_kl_regularization else torch.tensor(0.0, device=self.device)

        # Loss dominance: is the entropy term overwhelming policy/value?
        if should_print:
            with torch.no_grad():
                denom = (losses['policy'].abs() + losses['value'].abs() + 1e-8).item()
                self._dlog("PPO", "Loss dominance diagnostic",
                          policy=f"{losses['policy'].item():.4f}",
                          value=f"{losses['value'].item():.4f}",
                          entropy=f"{losses['entropy'].item():.4f}",
                          entropy_over_pv=f"{(losses['entropy'].abs().item() / denom):.2f}")

        # Total includes channel entropy reg and repulsion terms
        losses['total'] = (
            losses['policy'] +
            losses['value'] +
            losses['entropy'] +
            losses['kl'] +
            losses['channel_entropy_reg'] +
            losses['channel_repulsion']
        )

        if should_print:
            self._dlog("PPO", "Loss breakdown",
                      episode=episode_idx,
                      epoch=f"{ke}/{k_epochs}",
                      policy_dir=f"{loss_policy_dir.item():.4f}",
                      policy_mag=f"{loss_policy_mag.item():.4f}",
                      policy_axis=f"{loss_policy_axis.item():.4f}",
                      policy_total=f"{losses['policy'].item():.4f}",
                      value=f"{losses['value'].item():.4f}",
                      entropy_val=f"{entropy.mean().item():.4f}",
                      entropy_bonus=f"{losses['entropy'].item():.4f}",
                      kl=f"{losses['kl'].item():.4f}" if losses['kl'].item() != 0.0 else "0",
                      chan_entropy_reg=f"{losses['channel_entropy_reg'].item():.6f}" if losses['channel_entropy_reg'].item() != 0.0 else "0",
                      total=f"{losses['total'].item():.4f}")

            for head_name in ['direction', 'magnitude', 'axis']:
                adv = per_head_adv[head_name]
                self._dlog("PPO", f"Advantage ({head_name})",
                          mean=f"{adv.mean():.4f}",
                          std=f"{adv.std():.4f}",
                          min=f"{adv.min():.4f}",
                          max=f"{adv.max():.4f}")

            adv_all = torch.cat([per_head_adv['direction'], per_head_adv['magnitude'], per_head_adv['axis']])
            adv_status = "WEAK" if adv_all.std().item() < 0.1 else "OK"
            self._dlog("PPO", "Advantage stats (combined)",
                      mean=f"{adv_all.mean().item():.4f}",
                      std=f"{adv_all.std().item():.4f}",
                      min=f"{adv_all.min().item():.4f}",
                      max=f"{adv_all.max().item():.4f}",
                      status=adv_status)

            # Clip fraction: how often PPO clips
            with torch.no_grad():
                dir_ratio = torch.exp(per_head_new['direction'] - per_head_old['direction'])
                clip_frac_dir = ((dir_ratio < 0.95) | (dir_ratio > 1.05)).float().mean().item()
                clip_status = "HIGH" if clip_frac_dir > 0.3 else "OK"
                self._log("PPO", "Clip fraction (direction)",
                          clip_frac=f"{clip_frac_dir:.1%}",
                          status=clip_status)

            # Value prediction quality: EV = 1 - Var(returns - values) / Var(returns)
            with torch.no_grad():
                returns_std = returns.std().item()
                
                if returns_std < 1e-3:
                    self._dlog("VALUE", "Explained variance SKIPPED",
                              returns_std=f"{returns_std:.2e}")
                else:
                    returns_var = returns_std ** 2
                    residual_var = (returns - state_values).var().item()
                    explained_var = 1 - residual_var / returns_var

                    # Compare EV from stored rollout values vs current network
                    # (old_values = rollout-time critic predictions)
                    if old_values is not None:
                        _ret_flat_h4 = returns.reshape(-1)
                        _old_vals_h4 = old_values.reshape(-1)
                        
                        if _old_vals_h4.numel() == _ret_flat_h4.numel():
                            _stored_resid = (_ret_flat_h4 - _old_vals_h4).var().item()
                            _ev_stored = 1 - _stored_resid / returns_var
                            _ev_improv = explained_var - _ev_stored
                            self._dlog("VALUE", "EV comparison (H4)",
                                      episode=episode_idx,
                                      ev_current=f"{explained_var:.4f}",
                                      ev_stored=f"{_ev_stored:.4f}",
                                      improvement=f"{_ev_improv:.4f}",
                                      status="LEARNING" if _ev_improv > 0.01 else "FLAT")
                        else:
                            self._dlog("VALUE", "EV comparison (H4) SKIPPED",
                                      episode=episode_idx,
                                      old_vals_n=_old_vals_h4.numel(),
                                      returns_n=_ret_flat_h4.numel())

                    # EV component breakdown when EV is bad
                    if explained_var < 0.1:
                        _sv = state_values.detach().reshape(-1).float()
                        _ret = returns.detach().reshape(-1).float()
                        _sv_mean = _sv.mean()
                        _ret_mean = _ret.mean()
                        _sv_std = _sv.std(unbiased=False)
                        _ret_std = _ret.std(unbiased=False)

                        # Correlation (torch-only)
                        _sv_z = _sv - _sv_mean
                        _ret_z = _ret - _ret_mean
                        _den = (_sv_z.pow(2).mean().sqrt() * _ret_z.pow(2).mean().sqrt()) + 1e-8
                        _corr = (_sv_z * _ret_z).mean() / _den
                        
                        self._dlog("VALUE", "EV components (bad EV)",
                                  episode=episode_idx,
                                  returns_mean=f"{_ret_mean.item():.4f}",
                                  returns_std=f"{_ret_std.item():.4f}",
                                  values_mean=f"{_sv_mean.item():.4f}",
                                  values_std=f"{_sv_std.item():.4f}",
                                  scale_ratio=f"{(_ret_std / (_sv_std + 1e-8)).item():.2f}",
                                  correlation=f"{_corr.item():.4f}",
                                  residual_var=f"{residual_var:.6g}",
                                  returns_var=f"{returns_var:.6g}")

                    

                    
                    if explained_var < -10:
                        ev_status = "EXTREME"
                    elif explained_var < 0:
                        ev_status = "NEGATIVE"
                    elif explained_var < 0.1:
                        ev_status = "LOW"
                    else:
                        ev_status = "OK"
                    
                    self._dlog("VALUE", "Explained variance",
                              ev=f"{explained_var:.4f}",
                              status=ev_status)

                # Value prediction stats
                self._dlog("VALUE", "Prediction stats",
                          value_mean=f"{state_values.mean().item():.3f}",
                          value_std=f"{state_values.std().item():.3f}",
                          returns_mean=f"{returns.mean().item():.3f}",
                          returns_std=f"{returns_std:.3f}")

        return losses

    def _capture_current_log_mixture_norm(self):
        """Current policy's normalized log-mixture on S^2 [batch, P], detached.
        Raises RuntimeError if required components are missing.
        """
        if not hasattr(self.policy, 'sh_head'):
            raise RuntimeError("Policy missing sh_head for KL computation")

        head = self.policy.sh_head

        if not hasattr(head, '_channel_log_mass_normalized'):
            raise RuntimeError("sh_head missing _channel_log_mass_normalized")

        current_log_mass_normalized = head._channel_log_mass_normalized

        # Channel weights live in one of several caches
        current_channel_weights = None

        if hasattr(head, '_last_outputs') and 'channel_weights' in head._last_outputs:
            current_channel_weights = head._last_outputs['channel_weights']
        elif hasattr(self.policy, '_last_outputs') and 'channel_weights' in self.policy._last_outputs:
            current_channel_weights = self.policy._last_outputs['channel_weights']
        elif hasattr(head, '_channel_weights'):
            current_channel_weights = head._channel_weights
        
        if current_channel_weights is None:
            raise RuntimeError(
                "channel_weights cache not found in any expected location. "
                "Ensure policy/head stores channel_weights in one of: "
                "head._last_outputs['channel_weights'], "
                "policy._last_outputs['channel_weights'], "
                "or head._channel_weights"
            )

        # Mix channels in probability space (same as sampling)
        log_w_gate = torch.log(current_channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        current_log_mixture = torch.logsumexp(
            log_w_gate + current_log_mass_normalized,
            dim=1
        )  # [batch, P]

        current_log_mixture_norm = current_log_mixture - torch.logsumexp(
            current_log_mixture, dim=-1, keepdim=True
        )

        return current_log_mixture_norm

    def _compute_kl_loss(self, episode):
        """KL divergence regularization (when enabled). Uses the fixed baseline
        captured at the start of optimize(), held constant across all PPO epochs in
        the update. Raises if enabled but required components are missing.
        """
        if not self.algo_config.use_kl_regularization:
            return torch.zeros((), device=self.device)

        # Use the fixed baseline; updating it here would collapse KL to zero
        if not self._kl_baseline_ready:
            # First ever call, before any optimize() run
            self._dlog("PPO", f"KL baseline not ready at episode {episode} - skipping")
            return torch.zeros((), device=self.device)

        if not hasattr(self.policy, 'sh_head'):
            raise RuntimeError(
                "KL regularization enabled but policy.sh_head is missing. "
                "Policy must have spherical harmonics head for KL computation."
            )
        
        head = self.policy.sh_head

        if not hasattr(head.s2_block, 'log_w_vec'):
            raise RuntimeError("s2_block missing log_w_vec (log grid weights)")

        grid_log_weights = head.s2_block.log_w_vec

        # Current mixture (after gradient updates)
        current_log_mixture_norm = self._capture_current_log_mixture_norm()

        # KL(current || baseline) = sum_p new_mass_p * (log new_density - log old_density),
        # with density = mass / quadrature_weight
        baseline_log_density = self._kl_baseline - grid_log_weights.unsqueeze(0)
        current_log_density = current_log_mixture_norm - grid_log_weights.unsqueeze(0)

        current_mass = torch.exp(current_log_mixture_norm)
        kl_div = (current_mass * (current_log_density - baseline_log_density)).sum(dim=-1).mean()

        if not torch.isfinite(kl_div):
            raise FloatingPointError(
                f"Non-finite KL divergence: {kl_div.item()}. "
                f"Check channel weights and log masses for NaN/Inf."
            )

        return self.config.kl_coeff * kl_div
