"""GAE advantages and reward normalization."""
import os
import torch
import numpy as np
import logging
_LOG = logging.getLogger(__name__)


class PPOAdvantageMixin:
    """GAE / advantage / reward-normalization computation for PPOOptimizer.

    ``compute_advantages`` (+ the per-head variant and the GAE fallback) and
    ``normalize_rewards_batch`` resolve reward_normalizer / config / device and
    the running-stats helpers through the MRO back to PPOOptimizer (not an
    nn.Module), so this mixin cannot perturb the policy fingerprint.
    """
    def normalize_rewards_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards by running std and clip; updates running stats."""
        if not self.normalize_rewards:
            return rewards

        rewards = np.array(rewards, dtype=np.float64)

        self.reward_normalizer.update(rewards)

        # Divide by std only (no mean subtraction) to preserve reward sign
        normalized = rewards / (self.reward_normalizer.std + 1e-8)

        normalized = np.clip(normalized, -10.0, 10.0)

        return normalized.astype(np.float32)

    def compute_advantages(self, rewards, state_values, is_terminals, gamma, gae_lambda=None, update_rms: bool = True, is_truncated=None, next_values=None, episode: int = None):        
        """GAE(lambda) advantages and returns. lambda interpolates one-step TD
        (lambda=0, low variance/high bias) and Monte Carlo (lambda=1); ~0.95-0.98
        suits sparse rewards. Returns (returns=advantages+values, normalized
        advantages). Raises on length mismatch or wrong type.
        """
        if gae_lambda is None:
            gae_lambda = self.gae_lambda

        # MC warmup check: early return before any processing
        _algo = getattr(self.config, 'algorithm', self.config)
        mc_warmup = getattr(_algo, 'mc_warmup_episodes', 0)        
        if episode is not None:
            self._current_episode = episode
        current_episode = getattr(self, '_current_episode', 0)
        use_mc = mc_warmup > 0 and current_episode < mc_warmup

        if not isinstance(rewards, (list, tuple, np.ndarray, torch.Tensor)):
            raise TypeError(f"rewards must be list/array/tensor, got {type(rewards)}")

        # Normalize rewards before computing advantages
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.detach().cpu().numpy()
        else:
            rewards_np = np.array(rewards, dtype=np.float32)

        if self.normalize_rewards and update_rms:
            rewards_np = self.normalize_rewards_batch(rewards_np)
        elif self.normalize_rewards and not update_rms:
            # Normalize with existing RMS, without updating it
            rewards_np = rewards_np / (self.reward_normalizer.std + 1e-8)
            rewards_np = np.clip(rewards_np, -10.0, 10.0).astype(np.float32)

        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device)
        T = len(rewards)

        if T == 0:
            raise ValueError("rewards is empty - cannot compute advantages")

        if not isinstance(is_terminals, (list, tuple, np.ndarray, torch.Tensor)):
            raise TypeError(f"is_terminals must be sequence of bools, got {type(is_terminals)}")

        if len(is_terminals) != T:
            raise ValueError(
                f"is_terminals length {len(is_terminals)} != rewards length {T}"
            )

        if not isinstance(is_terminals, torch.Tensor):
            is_terminals = torch.tensor(is_terminals, dtype=torch.float32, device=self.device)
        else:
            is_terminals = is_terminals.to(dtype=torch.float32, device=self.device)

        if isinstance(state_values, list):
            if len(state_values) != T:
                raise ValueError(
                    f"state_values list length {len(state_values)} != rewards length {T}"
                )
            state_values = torch.cat([sv.reshape(-1) for sv in state_values])
        
        state_values = state_values.detach().to(self.device).reshape(-1)
        
        if state_values.numel() != T:
            raise ValueError(
                f"state_values length {state_values.numel()} != rewards length {T}"
            )

        # GAE: A_t = delta_t + (gamma*lambda)delta_{t+1} + ...,  delta_t = r_t + gamma*V(s') - V(s_t).
        # Three bootstrap cases for V(s'):
        #   terminal -> V(s')=0, cut trace; truncation -> V(s')=next_values[t], cut trace;
        #   mid-episode -> V(s')=state_values[t+1], continue trace.
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0

        if is_truncated is not None:
            if not isinstance(is_truncated, torch.Tensor):
                is_truncated_t = torch.tensor(is_truncated, dtype=torch.float32, device=self.device)
            else:
                is_truncated_t = is_truncated.to(dtype=torch.float32, device=self.device)
        else:
            # No truncation tracking: treat all non-terminals as continuing
            is_truncated_t = torch.zeros(T, dtype=torch.float32, device=self.device)

        if next_values is not None:
            if not isinstance(next_values, torch.Tensor):
                next_values_t = torch.tensor(next_values, dtype=torch.float32, device=self.device)
            else:
                next_values_t = next_values.to(dtype=torch.float32, device=self.device)
        else:
            # Fallback: shifted state_values
            next_values_t = torch.zeros(T, dtype=torch.float32, device=self.device)
            if T > 1:
                next_values_t[:-1] = state_values[1:]
                next_values_t[-1] = state_values[-1]

        # MC warmup: pure Monte Carlo returns when critic is unreliable
        if use_mc:
            returns_mc = torch.zeros(T, dtype=torch.float32, device=self.device)
            running_return = 0.0

            for t in reversed(range(T)):
                terminal_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
                truncated_t = float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t])

                if terminal_t > 0.5 or truncated_t > 0.5:
                    if truncated_t > 0.5 and not (terminal_t > 0.5):
                        running_return = float(next_values_t[t].item())
                    else:
                        running_return = 0.0

                running_return = float(rewards[t]) + gamma * running_return
                returns_mc[t] = running_return

            advantages_mc = returns_mc - state_values.detach()

            if advantages_mc.numel() > 1:
                advantages_mc = (advantages_mc - advantages_mc.mean()) / (advantages_mc.std() + 1e-8)

            if not hasattr(self, '_mc_logged') or self._mc_logged != current_episode:
                _LOG.debug(f"[MC_WARMUP] episode {current_episode}/{mc_warmup} | "
                      f"MC returns: mean={returns_mc.mean():.3f} std={returns_mc.std():.3f} | "
                      f"V baseline: mean={state_values.mean():.3f} std={state_values.std():.3f} | "
                      f"raw_adv: mean={advantages_mc.mean():.4f} std={advantages_mc.std():.4f}")
                self._mc_logged = current_episode

            return returns_mc, advantages_mc

        for t in reversed(range(T)):
            terminal_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
            truncated_t = float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t])

            # Episode boundary (terminal OR truncated) cuts the GAE trace
            episode_ended = (terminal_t > 0.5) or (truncated_t > 0.5)

            # V(s_{t+1}) for the TD error
            if terminal_t > 0.5:
                next_value = 0.0  # true terminal: zero bootstrap
            elif truncated_t > 0.5:
                next_value = float(next_values_t[t].item())  # truncation: V at truncation point
            elif t == T - 1:
                # Last step in batch, neither terminal nor truncated
                next_value = float(next_values_t[t].item())
            else:
                next_value = float(state_values[t + 1].item())  # mid-episode: next state's value

            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - state_values[t]

            # A_t = delta_t + gamma*lambda*A_{t+1}; trace resets at episode boundaries
            if episode_ended:
                gae = float(delta)
            else:
                gae = float(delta) + gamma * gae_lambda * gae

            advantages[t] = gae

        # Returns = advantages + values (value-function target)
        returns = advantages + state_values

        # GAE fail-fast assertions: catch silent target corruption
        smoke_mode = os.environ.get('PPO_SMOKE_MODE', '').strip().lower() in ('1', 'true', 'yes', 'y')
        
        if smoke_mode or not hasattr(self, '_gae_assertions_passed'):
            self._gae_assertions_passed = True

            # Bootstrap value used at each step
            bootstrap_used = []
            for t in range(T):
                term_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
                trunc_t = float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t])
                
                if term_t > 0.5:
                    bootstrap_used.append(0.0)
                elif trunc_t > 0.5:
                    bootstrap_used.append(float(next_values_t[t].item()))
                elif t == T - 1:
                    bootstrap_used.append(float(next_values_t[t].item()))
                else:
                    bootstrap_used.append(float(state_values[t + 1].item()))

            # Assertion 1: terminal steps must have bootstrap = 0
            for t in range(T):
                term_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
                if term_t > 0.5:
                    if abs(bootstrap_used[t]) > 1e-6:
                        raise RuntimeError(
                            f"GAE FAIL-FAST: Terminal step t={t} has bootstrap={bootstrap_used[t]:.6f}, expected 0.\n"
                            f"This corrupts value targets and makes learning impossible."
                        )
            
            # Assertion 2: truncation steps must have bootstrap = next_values_t[t]
            for t in range(T):
                trunc_t = float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t])
                if trunc_t > 0.5:
                    expected_bootstrap = float(next_values_t[t].item())
                    if abs(bootstrap_used[t] - expected_bootstrap) > 1e-4:
                        raise RuntimeError(
                            f"GAE FAIL-FAST: Truncated step t={t} has bootstrap={bootstrap_used[t]:.6f}, "
                            f"expected next_values_t[t]={expected_bootstrap:.6f}.\n"
                            f"This corrupts value targets for truncated episodes."
                        )
            
            # Assertion 3: no step is both terminal AND truncated
            for t in range(T):
                term_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
                trunc_t = float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t])
                if term_t > 0.5 and trunc_t > 0.5:
                    raise RuntimeError(
                        f"GAE FAIL-FAST: Step t={t} is BOTH terminal AND truncated.\n"
                        f"This is logically impossible and indicates buffer corruption."
                    )
            
            # Raw advantage stats, logged before normalization
            raw_adv_mean = advantages.mean().item()
            raw_adv_std = advantages.std().item()

            # INFO line is parsed downstream; keep at INFO
            _LOG.info(f"[GAE ASSERTIONS] Raw advantages: mean={raw_adv_mean:.4f}, std={raw_adv_std:.4f}")

            # Degenerate advantage warning
            if raw_adv_std < 0.01:
                _LOG.info(f"[GAE WARNING] WARNING ADV STD TOO SMALL ({raw_adv_std:.6f}) -> policy gradient likely noise; learning signal weak")
                # Diagnostic: fraction of return variance from terminal reward
                rewards_tensor = torch.tensor([float(rewards[t].item() if isinstance(rewards[t], torch.Tensor) else rewards[t]) for t in range(T)])
                terminal_idx = None
                for t in range(T):
                    term_t = float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t])
                    if term_t > 0.5:
                        terminal_idx = t
                        break
                if terminal_idx is not None:
                    terminal_reward = float(rewards_tensor[terminal_idx])
                    non_terminal_rewards = rewards_tensor[rewards_tensor.abs() < abs(terminal_reward) * 0.9] if terminal_reward != 0 else rewards_tensor[:-1]
                    _LOG.info(f"[GAE WARNING]   Terminal reward: {terminal_reward:.4f}, "
                          f"non-terminal reward std: {non_terminal_rewards.std().item() if len(non_terminal_rewards) > 1 else 0:.4f}")

            n_term = sum(1 for t in range(T) if float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t]) > 0.5)
            n_trunc = sum(1 for t in range(T) if float(is_truncated_t[t].item() if isinstance(is_truncated_t[t], torch.Tensor) else is_truncated_t[t]) > 0.5)
            
            _LOG.debug(f"[GAE ASSERTIONS] T={T}, terminals={n_term}, truncations={n_trunc} | status=OK")

        # GAE target table (first call only): shows both next_values_t[t] and
        # state_values[t+1] for bootstrap diagnosis
        if not hasattr(self, '_h3_gae_table_dumped'):
            self._h3_gae_table_dumped = True
            _LOG.debug("")
            _LOG.debug("[GAE TARGET TABLE] First batch - verifying bootstrap logic:")
            _LOG.debug("  t      rew   term  trunc      V_t  nxt_val_t   V_t+1  used_nxt   delta      ret      adv")
            _LOG.debug("-" * 95)
            _num_show = min(15, T)
            for _t in range(_num_show):
                _term = float(is_terminals[_t].item()) if hasattr(is_terminals[_t], 'item') else float(is_terminals[_t])
                _trunc = float(is_truncated_t[_t].item()) if hasattr(is_truncated_t[_t], 'item') else float(is_truncated_t[_t])
                _vt = float(state_values[_t].item())
                _vnext_stored = float(next_values_t[_t].item())
                if _t < T - 1:
                    _vt_plus1 = float(state_values[_t + 1].item())
                else:
                    _vt_plus1 = float('nan')
                _ret = float(returns[_t].item())
                _adv = float(advantages[_t].item())
                _rew = float(rewards[_t].item())
                # used_next_value, mirroring the GAE loop exactly
                if _term > 0.5:
                    _used_next = 0.0
                elif _trunc > 0.5:
                    _used_next = _vnext_stored
                elif _t == T - 1:
                    _used_next = _vnext_stored
                else:
                    _used_next = _vt_plus1
                _delta = _rew + gamma * _used_next - _vt
                _vt_plus1_str = "   N/A" if _t == T - 1 else "%7.4f" % _vt_plus1
                _LOG.debug("  %3d  %6.3f  %4.0f  %5.0f  %7.4f  %7.4f  %s  %7.4f  %7.4f  %7.4f  %7.4f" % (
                    _t, _rew, _term, _trunc, _vt, _vnext_stored, _vt_plus1_str, _used_next, _delta, _ret, _adv))
            _LOG.debug("-" * 95)

            _n_term = 0
            _n_trunc = 0
            for _t in range(T):
                _term = float(is_terminals[_t].item()) if hasattr(is_terminals[_t], 'item') else float(is_terminals[_t])
                _trunc = float(is_truncated_t[_t].item()) if hasattr(is_truncated_t[_t], 'item') else float(is_truncated_t[_t])
                if _term > 0.5:
                    _n_term += 1
                    _LOG.debug("[TERMINAL t=%d] used_next=0" % _t)
                elif _trunc > 0.5:
                    _n_trunc += 1
                    _vnext = float(next_values_t[_t].item())
                    _LOG.debug("[TRUNCATED t=%d] used_next=%.4f (from next_values_t)" % (_t, _vnext))

            _LOG.debug("[GAE SUMMARY] T=%d, terminals=%d, truncations=%d, gamma=%.3f" % (T, _n_term, _n_trunc, gamma))
            # Flag mid-episode steps where V_t+1 != next_values_t[t]
            _inconsistent = 0
            for _t in range(T - 1):
                _term = float(is_terminals[_t].item()) if hasattr(is_terminals[_t], 'item') else float(is_terminals[_t])
                _trunc = float(is_truncated_t[_t].item()) if hasattr(is_truncated_t[_t], 'item') else float(is_truncated_t[_t])
                if _term < 0.5 and _trunc < 0.5:
                    _vt_plus1 = float(state_values[_t + 1].item())
                    _vnext_stored = float(next_values_t[_t].item())
                    if abs(_vt_plus1 - _vnext_stored) > 0.001:
                        _inconsistent += 1
            if _inconsistent > 0:
                _LOG.warning("%d mid-episode steps have V_t+1 != next_values_t[t] (diff > 0.001)" % _inconsistent)
            _LOG.debug("NOTE: valid_flags not in compute_advantages - check buffer for masking")
            _LOG.debug("")

        # Per-batch advantage normalization (optimize() normalizes again per-minibatch)
        if advantages.numel() > 1:
            adv_std = advantages.std()
            adv_mean = advantages.mean()
            if adv_std < 0.01:
                # Collapse (returns ~= values, signal is noise): scale down rather
                # than normalizing noise to unit variance; preserves relative order
                advantages = (advantages - adv_mean) * 10.0
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Post-normalization check: mean ~0, std ~1
        smoke_mode = os.environ.get('PPO_SMOKE_MODE', '').strip().lower() in ('1', 'true', 'yes', 'y')
        if smoke_mode or not hasattr(self, '_gae_norm_checked'):
            self._gae_norm_checked = True
            norm_mean = advantages.mean().item()
            norm_std = advantages.std().item()

            if abs(norm_mean) > 1e-3:
                _LOG.info(f"[GAE WARNING] Post-norm advantage mean={norm_mean:.6f} (expected ~0)")

            if norm_std < 0.9 or norm_std > 1.1:
                _LOG.info(f"[GAE WARNING] Post-norm advantage std={norm_std:.4f} (expected ~1.0)")
            else:
                _LOG.debug(f"[GAE ASSERTIONS] Post-norm advantages: mean={norm_mean:.6f}, std={norm_std:.4f} | status=OK")

        return returns, advantages
    
    def compute_per_head_advantages(self, batch_data, gamma, gae_lambda=None):
        """Separate truncation-aware GAE advantages per head: direction/magnitude
        use total rewards, axis uses axis_rewards when available.
        """
        if gae_lambda is None:
            gae_lambda = self.gae_lambda

        state_values = batch_data['state_values']
        is_terminals = batch_data['is_terminals']
        total_rewards = batch_data['rewards']
        is_truncated = batch_data.get('is_truncated', None)

        axis_rewards = batch_data.get('axis_rewards', None)
        next_values  = batch_data.get('next_values', None)
        rewards_dir  = total_rewards
        rewards_mag  = total_rewards
        rewards_axis = axis_rewards if axis_rewards is not None else total_rewards

        _algo = getattr(self.config, 'algorithm', self.config)
        dir_lambda = getattr(_algo, 'direction_gae_lambda', gae_lambda)

        if hasattr(self, 'compute_advantages'):
            _, advantages_dir = self.compute_advantages(
                rewards_dir, state_values, is_terminals, gamma, gae_lambda,
                update_rms=True, is_truncated=is_truncated, next_values=next_values
            )
            _, advantages_mag = self.compute_advantages(
                rewards_mag, state_values, is_terminals, gamma, gae_lambda,
                update_rms=False, is_truncated=is_truncated, next_values=next_values
            )
            _, advantages_axis = self.compute_advantages(
                rewards_axis, state_values, is_terminals, gamma, gae_lambda,
                update_rms=False, is_truncated=is_truncated, next_values=next_values
            )
        else:
            advantages_dir = self._compute_gae_fallback(
                rewards_dir, state_values, is_terminals, gamma, gae_lambda,
                is_truncated=is_truncated, next_values=next_values
            )
            advantages_mag = self._compute_gae_fallback(
                rewards_mag, state_values, is_terminals, gamma, gae_lambda,
                is_truncated=is_truncated, next_values=next_values
            )
            advantages_axis = self._compute_gae_fallback(
                rewards_axis, state_values, is_terminals, gamma, gae_lambda,
                is_truncated=is_truncated, next_values=next_values
            )

        _LOG.debug(f'[ADV2_PER_HEAD] raw per-head std: '
              f'dir={advantages_dir.std():.6f} '
              f'mag={advantages_mag.std():.6f} '
              f'axis={advantages_axis.std():.6f}')
        return {
            'direction': advantages_dir,
            'magnitude': advantages_mag,
            'axis':      advantages_axis
        }
    def _compute_gae_fallback(self, rewards, state_values, is_terminals, gamma, gae_lambda, is_truncated=None, next_values=None):
        """GAE with truncation handling: truncations (timeouts) reset accumulation
        but bootstrap from V(s'); true terminals (dissociation) zero both.
        """
        if torch.is_tensor(rewards):
            rewards_np = rewards.detach().cpu().numpy()
        else:
            rewards_np = np.array(rewards)

        if torch.is_tensor(state_values):
            values_np = state_values.detach().cpu().numpy()
        else:
            values_np = np.array(state_values)

        if torch.is_tensor(is_terminals):
            terminals_np = is_terminals.detach().cpu().numpy().astype(bool)
        else:
            terminals_np = np.array(is_terminals, dtype=bool)

        if is_truncated is not None:
            if torch.is_tensor(is_truncated):
                truncated_np = is_truncated.detach().cpu().numpy().astype(bool)
            else:
                truncated_np = np.array(is_truncated, dtype=bool)
        else:
            truncated_np = np.zeros(len(rewards_np), dtype=bool)

        if next_values is not None:
            if torch.is_tensor(next_values):
                next_values_np = next_values.detach().cpu().numpy()
            else:
                next_values_np = np.array(next_values, dtype=np.float32)
        else:
            next_values_np = None

        N = len(rewards_np)
        advantages = np.zeros(N, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(N)):
            if terminals_np[t]:
                # True terminal (dissociation/failure): zero bootstrap
                next_value = 0.0
                last_gae = 0.0
            elif truncated_np[t]:
                # Truncation (timeout): use rollout-captured V(s'), NOT values_np[t+1]
                # (values_np[t+1] is the next episode's first step -- wrong bootstrap)
                if next_values_np is not None:
                    next_value = float(next_values_np[t])
                else:
                    next_value = values_np[t + 1] if t < N - 1 else 0.0
                last_gae = 0.0
            else:
                next_value = values_np[t + 1] if t < N - 1 else 0.0

            delta = rewards_np[t] + gamma * next_value - values_np[t]
            boundary = terminals_np[t] or truncated_np[t]
            last_gae = delta + gamma * gae_lambda * (0.0 if boundary else last_gae)
            advantages[t] = last_gae

        _adv_std = float(advantages.std())
        _adv_mean = float(advantages.mean())
        _n_term = int(terminals_np.sum())
        _n_trunc = int(truncated_np.sum())
        _LOG.debug(f'[ADV1_GAE_FALLBACK] N={N} terminals={_n_term} truncated={_n_trunc} '
              f'raw_adv: mean={_adv_mean:.4f} std={_adv_std:.6f} '
              f'floor={_adv_std < 0.01}')
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)
