"""Manages the rollout and critic-replay buffers."""
import torch
import numpy as np
from typing import Dict, Any, Optional, Literal, List
from realize.ppo.buffers import RolloutBuffer, CriticReplayBuffer
from realize.ppo.ppo_buffer_manager_replay import PPOReplayMixin
import logging
_LOG = logging.getLogger(__name__)


class PPOBufferManager(PPOReplayMixin):
    """Owns the RolloutBuffer and optional CriticReplayBuffer; delegates episode-end
    marking and success-weighted sampling, exposes success statistics.
    """

    def __init__(self, config, device, success_oversample_ratio: float = 3.0):
        """config is PPOAlgorithmConfig or full PPOConfig; success_oversample_ratio
        oversamples successful episodes (default 3.0).
        """
        self.config = config
        self.device = device

        if hasattr(config, 'success_oversample_ratio'):
            success_oversample_ratio = config.success_oversample_ratio

        # verbose may live on config or on config.runtime depending on config type
        debug_mode = False
        if hasattr(config, 'verbose'):
            debug_mode = config.verbose
        elif hasattr(config, 'runtime') and hasattr(config.runtime, 'verbose'):
            debug_mode = config.runtime.verbose
        
        try:
            self.buffer = RolloutBuffer(
                debug_mode=debug_mode,
                success_oversample_ratio=success_oversample_ratio
            )
        except TypeError:
            self.buffer = RolloutBuffer(debug_mode=debug_mode)
            _LOG.info("[PPOBufferManager] Warning: Buffer doesn't support success_oversample_ratio. "
                  "Using uniform sampling.")

        self.critic_replay_buffer = None
        self._replay_config = None

        # replay config only present on full PPOConfig, not PPOAlgorithmConfig
        if hasattr(config, 'replay'):
            self._replay_config = config.replay
            if config.replay.enable_critic_replay:
                self.critic_replay_buffer = CriticReplayBuffer(
                    capacity=config.replay.capacity,
                    success_priority_ratio=config.replay.success_priority_ratio,
                    max_episode_age=config.replay.max_episode_age,
                    debug_mode=debug_mode
                )
                _LOG.info(f"[PPOBufferManager] Critic replay buffer initialized "
                      f"(capacity={config.replay.capacity})")

    def _tensor_to_numpy(self, tensor):
        """Delegate tensor->numpy conversion to the buffer."""
        return self.buffer._tensor_to_numpy(tensor)

    def mark_episode_end(self, success: bool):
        """Record episode end (success=True if O2 dissociated) for success-weighted
        sampling. Call on episode termination (dissociation or max steps).
        """
        if hasattr(self.buffer, 'mark_episode_end'):
            self.buffer.mark_episode_end(success)
        else:
            if not hasattr(self, '_warned_no_success_tracking'):
                _LOG.info("[PPOBufferManager] Warning: Buffer doesn't support mark_episode_end(). "
                      "Success oversampling disabled.")
                self._warned_no_success_tracking = True
    
    def get_success_statistics(self) -> Dict[str, Any]:
        """Success rate, counts, and step distributions for episodes in the buffer."""
        if hasattr(self.buffer, 'get_success_statistics'):
            return self.buffer.get_success_statistics()
        else:
            return {
                'total_episodes': 0,
                'success_count': 0,
                'failure_count': 0,
                'success_rate': 0.0,
                'note': 'Buffer does not support success tracking'
            }

    def add(self, state, action, log_prob, state_value, temperature=None, channel_weights=None,
            log_prob_direction=None, log_prob_magnitude=None, log_prob_axis=None,
            sh_coeffs=None, trans_direction=None, trans_magnitude=None,
            actor_state=None,
            o1_embedding=None, o2_embedding=None, geometry=None,
            env_multipole=None):            
        """Validate and convert one step's data, then forward to buffer.add(). sh_coeffs,
        trans_direction, trans_magnitude and o1/o2/geometry are required; per-head
        log-probs and actor_state are optional. One step per call.
        """
        try:
            # Single-step adds only: a batch dimension would mix scalar/array types
            if isinstance(state, (torch.Tensor, np.ndarray)):
                if state.ndim > 1 and state.shape[0] > 1:
                    raise ValueError(f"PPOBufferManager.add() expects single steps, got batch size {state.shape[0]}. "
                                   "Call add() once per step.")

            # Multi-component log_prob is summed below; only a true batch (dim 0 > 1) is rejected
            if isinstance(log_prob, (torch.Tensor, np.ndarray)):
                if hasattr(log_prob, 'ndim') and log_prob.ndim > 1:
                    if log_prob.shape[0] > 1:
                        raise ValueError(
                            f"log_prob has batch dimension > 1: {log_prob.shape}. "
                            f"PPOBufferManager.add() expects single steps. "
                            f"Call add() once per step, not in batches."
                        )
            if isinstance(action, dict):
                # All required keys must be present; action is never reconstructed
                required_keys = {'translation_direction', 'translation_magnitude',
                               'axis_direction', 'translation_vector'}
                missing = required_keys - set(action.keys())
                if missing:
                    raise ValueError(
                        f"Action missing required keys: {missing}. "
                        f"Present keys: {set(action.keys())}. "
                        f"Policy must provide all required action components. "
                        f"No reconstruction is performed."
                    )
                
                # direction_indices is the grid index from categorical sampling; without it
                # evaluate() cannot reproduce act()'s exact log-prob (PPO ratio would be wrong)
                if 'direction_indices' not in action or action['direction_indices'] is None:
                    if 'translation_direction' not in action or action['translation_direction'] is None:
                        raise RuntimeError(                
                            "Action missing 'direction_indices' required for exact PPO evaluation. "
                            "Policy must store the grid index used during categorical sampling. "
                            "Without this, log-probs in act() and evaluate() will not match exactly."
                        )
            
            trans_sample = None
            if isinstance(action, dict) and 'translation_vector' in action:
                trans_sample = action['translation_vector']
                if isinstance(trans_sample, torch.Tensor):
                    trans_sample = trans_sample.cpu()

            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
            if state_np.ndim > 1 and state_np.shape[0] == 1:
                state_np = state_np.squeeze(0)

            if isinstance(log_prob, torch.Tensor):
                log_prob_np = log_prob.detach().cpu().numpy()
            else:
                log_prob_np = log_prob
            
            if isinstance(log_prob_np, np.ndarray):
                # Multi-component log-prob is summed into a single scalar
                if log_prob_np.size > 1:
                    log_prob_np = float(log_prob_np.sum())
                else:
                    log_prob_np = float(log_prob_np.squeeze())
            else:
                log_prob_np = float(log_prob_np)

            if not np.isfinite(log_prob_np):
                raise FloatingPointError(
                    f"Non-finite log_prob: {log_prob_np}. "
                    f"log_prob must be finite (not NaN or Inf). "
                    f"Check policy.act() output."
                )

            if isinstance(state_value, torch.Tensor):
                state_value_np = state_value.detach().cpu().numpy()
            else:
                state_value_np = state_value

            if isinstance(state_value_np, np.ndarray):
                state_value_np = float(state_value_np.squeeze())
            else:
                state_value_np = float(state_value_np)

            if not np.isfinite(state_value_np):
                raise FloatingPointError(
                    f"Non-finite state_value: {state_value_np}. "
                    f"state_value must be finite (not NaN or Inf). "
                    f"Check policy value head output."
                )

            log_prob_dir_np = None
            log_prob_mag_np = None
            log_prob_axis_np = None

            if log_prob_direction is not None:
                if isinstance(log_prob_direction, torch.Tensor):
                    log_prob_dir_np = float(log_prob_direction.detach().cpu().numpy())
                else:
                    log_prob_dir_np = float(log_prob_direction)
                if not np.isfinite(log_prob_dir_np):
                    raise FloatingPointError(f"Non-finite log_prob_direction: {log_prob_dir_np}")
            
            if log_prob_magnitude is not None:
                if isinstance(log_prob_magnitude, torch.Tensor):
                    log_prob_mag_np = float(log_prob_magnitude.detach().cpu().numpy())
                else:
                    log_prob_mag_np = float(log_prob_magnitude)
                if not np.isfinite(log_prob_mag_np):
                    raise FloatingPointError(f"Non-finite log_prob_magnitude: {log_prob_mag_np}")
            
            if log_prob_axis is not None:
                if isinstance(log_prob_axis, torch.Tensor):
                    log_prob_axis_np = float(log_prob_axis.detach().cpu().numpy())
                else:
                    log_prob_axis_np = float(log_prob_axis)
                if not np.isfinite(log_prob_axis_np):
                    raise FloatingPointError(f"Non-finite log_prob_axis: {log_prob_axis_np}")

            # Asymmetric O2 embedding (axis head), if the action dict carries it
            o2_embedding_asymmetric_for_buffer = None
            if isinstance(action, dict) and 'o2_embedding_asymmetric' in action:
                o2_emb = action['o2_embedding_asymmetric']
                if o2_emb is not None:
                    if isinstance(o2_emb, torch.Tensor):
                        o2_embedding_asymmetric_for_buffer = o2_emb.detach().cpu().numpy()
                    else:
                        o2_embedding_asymmetric_for_buffer = np.asarray(o2_emb)

            if sh_coeffs is None:
                raise RuntimeError(
                    "sh_coeffs cannot be None in buffer.add(). "
                    "ppo_agent.select_action() must extract and pass sh_coeffs."
                )
            if isinstance(sh_coeffs, torch.Tensor):
                sh_coeffs_np = sh_coeffs.detach().cpu().numpy()
            else:
                sh_coeffs_np = np.asarray(sh_coeffs)

            if trans_direction is None:
                raise RuntimeError(
                    "trans_direction cannot be None in buffer.add(). "
                    "ppo_agent.select_action() must extract and pass trans_direction."
                )
            if isinstance(trans_direction, torch.Tensor):
                trans_direction_np = trans_direction.detach().cpu().numpy()
            else:
                trans_direction_np = np.asarray(trans_direction)

            if trans_magnitude is None:
                raise RuntimeError(
                    "trans_magnitude cannot be None in buffer.add(). "
                    "ppo_agent.select_action() must extract and pass trans_magnitude."
                )
            if isinstance(trans_magnitude, torch.Tensor):
                trans_magnitude_np = trans_magnitude.detach().cpu().numpy()
            else:
                trans_magnitude_np = np.asarray(trans_magnitude)

            # Actor-path state (dual backbone)
            actor_state_np = None

            if actor_state is not None:
                if isinstance(actor_state, torch.Tensor):
                    actor_state_np = actor_state.detach().cpu().numpy()
                else:
                    actor_state_np = np.asarray(actor_state)
                if actor_state_np.ndim > 1 and actor_state_np.shape[0] == 1:
                    actor_state_np = actor_state_np.squeeze(0)

            # Critic raw inputs (o1/o2/geometry) -> numpy for buffer storage
            def _to_np_squeeze(t):
                if t is None:
                    return None
                if isinstance(t, torch.Tensor):
                    arr = t.detach().cpu().numpy()
                else:
                    arr = np.asarray(t)
                if arr.ndim > 1 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                return arr

            o1_embedding_np = _to_np_squeeze(o1_embedding)
            o2_embedding_np = _to_np_squeeze(o2_embedding)
            geometry_np = _to_np_squeeze(geometry)

            env_multipole_np = _to_np_squeeze(env_multipole)

            success = self.buffer.add(
                state=state_np,
                action_rot=action,  # whole action dict/tensor
                trans_sample=trans_sample,
                log_prob=log_prob_np,
                log_prob_direction=log_prob_dir_np,
                log_prob_magnitude=log_prob_mag_np,
                log_prob_axis=log_prob_axis_np,
                state_value=state_value_np,
                reward=0.0,
                is_terminal=False,
                temperature=float(temperature) if temperature is not None else None,
                channel_weights=channel_weights,
                o2_embedding_asymmetric=o2_embedding_asymmetric_for_buffer,
                sh_coeffs=sh_coeffs_np,
                trans_direction=trans_direction_np,
                trans_magnitude=trans_magnitude_np,
                actor_state=actor_state_np,
                o1_embedding=o1_embedding_np,
                o2_embedding=o2_embedding_np,
                geometry=geometry_np,
                env_multipole=env_multipole_np
            )

            return success

        except Exception as e:
            _LOG.info(f"Error adding to buffer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_reward(self, index, reward, is_terminal, mode: Literal['set', 'add'] = 'set', **kwargs):
        """Update reward at index. mode='set' replaces (per-step); mode='add' accumulates
        terminal bonuses onto the last valid step. is_truncated and truncation_value
        (V(s') bootstrap) are read from kwargs for GAE.
        """
        if mode not in ('set', 'add'):
            raise ValueError(f"mode must be 'set' or 'add', got {mode}")

        is_truncated = kwargs.pop('is_truncated', False)
        truncation_value = kwargs.pop('truncation_value', None)

        if mode == 'add':
            result = self.buffer.update_reward(
                index,
                reward,
                is_terminal,
                is_truncated=is_truncated,
                additive=True,
                **kwargs
            )
        else:
            result = self.buffer.update_reward(
                index,
                reward,
                is_terminal,
                is_truncated=is_truncated,
                additive=False,
                **kwargs
            )

        if is_truncated and truncation_value is not None:
            self.buffer.truncation_bootstrap_values[index] = float(truncation_value)
            _LOG.debug(f"  [BUFFER] Stored truncation V(s')={truncation_value:.4f} at idx {index}")
        
        return result

    def compute_advantages(self, rewards: torch.Tensor, state_values: torch.Tensor,
                          is_terminals: torch.Tensor, gamma: float = 0.99,
                          gae_lambda: float = 0.95,
                          is_truncated: torch.Tensor = None,
                          next_values: torch.Tensor = None) -> tuple:
        """GAE returns/advantages. is_terminals (dissociation/failure) bootstrap 0;
        is_truncated (timeouts) bootstrap from next_values=V(s'). gae_lambda 0.95
        (raise to 0.97-0.98 for very sparse rewards).
        """

        N = len(rewards)

        # Callers without separate is_truncated imply is_terminals holds only true terminals
        if is_truncated is None:
            is_truncated = torch.zeros(N, dtype=torch.bool)

        # next_values defaults to state_values shifted by one (approximate; last step bootstraps
        # from own value and is overridden when terminal)
        if next_values is None:
            next_values = torch.zeros(N, dtype=torch.float32)
            if N > 1:
                next_values[:-1] = state_values[1:]

        # EPISODE BOUNDARY DIAGNOSTIC
        _LOG.debug(f"\n{'='*60}")
        _LOG.debug(f"[GAE EPISODE BOUNDARY DIAGNOSTIC]")
        _LOG.debug(f"{'='*60}")
        
        if not isinstance(is_terminals, torch.Tensor):
            is_terminals_t = torch.tensor(is_terminals)
        else:
            is_terminals_t = is_terminals
            
        if not isinstance(is_truncated, torch.Tensor):
            is_truncated_t = torch.tensor(is_truncated)
        else:
            is_truncated_t = is_truncated
        
        terminal_indices = torch.where(is_terminals_t)[0].tolist()
        truncated_indices = torch.where(is_truncated_t)[0].tolist()
        terminal_count = len(terminal_indices)
        truncated_count = len(truncated_indices)
        
        _LOG.debug(f"  Total steps in batch: {N}")
        _LOG.debug(f"  TRUE terminal steps: {terminal_count} (bootstrap=0)")
        _LOG.debug(f"  Truncated steps: {truncated_count} (bootstrap from V(s'))")
        _LOG.debug(f"  Terminal indices: {terminal_indices[:20]}{'...' if len(terminal_indices) > 20 else ''}")
        _LOG.debug(f"  Truncated indices: {truncated_indices[:20]}{'...' if len(truncated_indices) > 20 else ''}")
        
        all_boundary_indices = sorted(set(terminal_indices + truncated_indices))
        if all_boundary_indices:
            episode_lengths = []
            prev_idx = -1
            for idx in all_boundary_indices:
                episode_lengths.append(idx - prev_idx)
                prev_idx = idx
            _LOG.debug(f"  Episode lengths: {episode_lengths[:10]}{'...' if len(episode_lengths) > 10 else ''}")
            _LOG.debug(f"  Avg episode length: {sum(episode_lengths)/len(episode_lengths):.1f}")
        
        if terminal_count == 0 and truncated_count == 0:
            _LOG.info(f"\n  WARNING NO BOUNDARIES IN BATCH!")
            _LOG.info(f"     GAE will treat entire batch as single trajectory.")
        elif terminal_count == N:
            _LOG.info(f"\n  ALERT ALL STEPS ARE TRUE TERMINAL!")
            _LOG.info(f"     This is almost certainly a bug.")

        if truncated_count > 0:
            _LOG.debug(f"\n  OK TRUNCATION HANDLING ACTIVE")
            _LOG.debug(f"     {truncated_count} steps will bootstrap from V(s') instead of 0")

        if N > 0:
            last_is_terminal = bool(is_terminals_t[-1].item()) if torch.is_tensor(is_terminals_t[-1]) else bool(is_terminals[-1])
            last_is_truncated = bool(is_truncated_t[-1].item()) if torch.is_tensor(is_truncated_t[-1]) else bool(is_truncated[-1])
            _LOG.debug(f"\n  Last step: terminal={last_is_terminal}, truncated={last_is_truncated}")
            if last_is_truncated and not last_is_terminal:
                _LOG.debug(f"     Will bootstrap from V(s_final) = {next_values[-1].item():.4f}")
            elif last_is_terminal:
                _LOG.debug(f"     True terminal - bootstrap = 0")
        
        rewards_np = rewards.cpu().numpy() if torch.is_tensor(rewards) else rewards
        _LOG.debug(f"\n  Reward stats:")
        _LOG.debug(f"    mean={float(np.mean(rewards_np)):.4f}, std={float(np.std(rewards_np)):.4f}")
        _LOG.debug(f"    min={float(np.min(rewards_np)):.4f}, max={float(np.max(rewards_np)):.4f}")
        _LOG.debug(f"    nonzero: {np.sum(np.abs(rewards_np) > 1e-6)}/{N}")

        values_np = state_values.cpu().numpy() if torch.is_tensor(state_values) else state_values
        _LOG.debug(f"\n  Value stats:")
        _LOG.debug(f"    mean={float(np.mean(values_np)):.4f}, std={float(np.std(values_np)):.4f}")
        _LOG.debug(f"    min={float(np.min(values_np)):.4f}, max={float(np.max(values_np)):.4f}")
        
        _LOG.debug(f"{'='*60}\n")

        # Accumulate into lists, stack once at the end (no inplace tensor mods)
        gae_list = []
        returns_list = []

        # Backward pass through the trajectory
        running_next_value = 0.0
        running_next_gae = 0.0

        for t in reversed(range(N)):
            if is_terminals[t]:
                # True terminal: no future value, bootstrap 0
                running_next_value = 0.0
                running_next_gae = 0.0
            elif is_truncated[t]:
                # Truncation: value bootstraps from V(s'), GAE trace resets
                running_next_value = float(next_values[t])
                running_next_gae = 0.0

            # TD residual: r + gamma * V(s') - V(s)
            delta = rewards[t] + gamma * running_next_value - state_values[t]

            gae_t = delta + gamma * gae_lambda * running_next_gae

            return_t = gae_t + state_values[t]

            gae_list.insert(0, gae_t)
            returns_list.insert(0, return_t)

            running_next_value = state_values[t]
            running_next_gae = gae_t

        gae = torch.stack(gae_list, dim=0) if gae_list else torch.zeros(N, dtype=torch.float32)
        returns = torch.stack(returns_list, dim=0) if returns_list else torch.zeros(N, dtype=torch.float32)

        advantages = gae

        # GAE OUTPUT DIAGNOSTIC
        returns_np = returns.cpu().numpy() if torch.is_tensor(returns) else returns
        advantages_np = advantages.cpu().numpy() if torch.is_tensor(advantages) else advantages

        _LOG.debug(f"\n[GAE OUTPUT DIAGNOSTIC]")
        _LOG.debug(f"  Returns: mean={float(np.mean(returns_np)):.4f}, std={float(np.std(returns_np)):.4f}")
        _LOG.debug(f"  Advantages: mean={float(np.mean(advantages_np)):.4f}, std={float(np.std(advantages_np)):.4f}")

        if float(np.std(advantages_np)) < 0.01:
            _LOG.info(f"  ALERT ADVANTAGE COLLAPSE: std={float(np.std(advantages_np)):.6f}")
            _LOG.info(f"     Policy gradient will be near zero!")
        
        return returns, advantages

    def has_data(self):
        """Check if buffer has data"""
        return len(self.buffer.states) > 0
    
    def get_batch(self, use_success_weighting: bool = True) -> Dict[str, Any]:
        """Build the PPO batch from valid (reward-updated) transitions only, keeping PPO
        off placeholder steps. Raises if the buffer is empty or has no valid transitions.
        """
        if not self.has_data():
            raise RuntimeError(
                "Cannot get batch from empty buffer. "
                "Buffer must contain data before calling get_batch()."
            )

        # valid_flags excludes placeholder steps (reward not yet written)
        valid = np.array(self.buffer.valid_flags, dtype=bool)
        n_total = len(valid)
        n_valid = valid.sum()

        if n_valid == 0:
            raise RuntimeError(
                f"No valid transitions in buffer (0/{n_total} valid). "
                f"All transitions are placeholders (reward not updated). "
                f"Check that update_reward() is called for collected steps."
            )

        _LOG.debug(f"[BUFFER] Returning {n_valid}/{n_total} valid transitions for PPO")

        success_stats = self.get_success_statistics()
        if success_stats['total_episodes'] > 0:
            _LOG.debug(f"[BUFFER] Success rate: {success_stats['success_rate']:.1%} "
                  f"({success_stats['success_count']}/{success_stats['total_episodes']} episodes)")

        valid_indices = np.where(valid)[0]

        # Keep every reward component (PE, ea, dissoc, axis), then filter to valid steps
        all_rewards = np.array(self.buffer.rewards)
        all_ea_rewards = np.array(self.buffer.ea_rewards)
        all_dissoc_rewards = np.array(self.buffer.dissociation_rewards)
        all_axis_rewards = np.array(self.buffer.axis_rewards)

        rewards_valid = all_rewards[valid_indices]
        ea_rewards_valid = all_ea_rewards[valid_indices]
        dissoc_rewards_valid = all_dissoc_rewards[valid_indices]
        axis_rewards_valid = all_axis_rewards[valid_indices]

        # Confirm PE rewards survived filtering
        pe_total_all = float(all_rewards.sum())
        pe_total_valid = float(rewards_valid.sum())
        
        _LOG.debug(f"[BUFFER/REWARD DEBUG] Total PE rewards:")
        _LOG.debug(f"  All transitions: {pe_total_all:.6f}")
        _LOG.debug(f"  Valid transitions: {pe_total_valid:.6f}")
        _LOG.debug(f"  EA rewards (valid): {ea_rewards_valid.sum():.6f}")
        _LOG.debug(f"  Dissoc rewards (valid): {dissoc_rewards_valid.sum():.6f}")
        _LOG.debug(f"  Axis rewards (valid): {axis_rewards_valid.sum():.6f}")

        if abs(pe_total_valid) < 1e-10:
            _LOG.warning(f"PE rewards in valid transitions are effectively ZERO!")
            _LOG.info(f"  This means either: (1) No PE reward was computed, or (2) All PE steps were filtered out")

        rewards_tensor = torch.as_tensor(rewards_valid, dtype=torch.float32)
        ea_rewards_tensor = torch.as_tensor(ea_rewards_valid, dtype=torch.float32)
        dissoc_rewards_tensor = torch.as_tensor(dissoc_rewards_valid, dtype=torch.float32)
        axis_rewards_tensor = torch.as_tensor(axis_rewards_valid, dtype=torch.float32)

        try:
            states = torch.as_tensor(np.array(self.buffer.states)[valid], dtype=torch.float32)
        except Exception as e:
            raise TypeError(
                f"Cannot convert states to tensor. States must be uniform numeric arrays. "
                f"Error: {e}"
            ) from e

        # Actor path states (dual backbone); fall back to states if unavailable
        try:
            actor_states = torch.as_tensor(
                np.array(self.buffer.actor_states)[valid],
                dtype=torch.float32
            )
            _LOG.debug(f"[BUFFER] actor_states extracted: shape={actor_states.shape}")
        except Exception as e:
            _LOG.info(f"[BUFFER] WARNING: actor_states not available ({e}), using states")
            actor_states = states

        # Critic raw inputs, recomputed through CriticEmbedding in evaluate()
        o1_embeddings = torch.as_tensor(
            np.array(self.buffer.o1_embeddings)[valid],
            dtype=torch.float32
        )
        o2_embeddings = torch.as_tensor(
            np.array(self.buffer.o2_embeddings)[valid],
            dtype=torch.float32
        )
        geometries = torch.as_tensor(
            np.array(self.buffer.geometries)[valid],
            dtype=torch.float32
        )
        _LOG.debug(f"[BUFFER] critic inputs: o1={o1_embeddings.shape}, "
              f"o2={o2_embeddings.shape}, geom={geometries.shape}")

        if actor_states.shape[-1] != 480:
            _LOG.info(f"[BUFFER] WARNING: actor_states dim={actor_states.shape[-1]}, expected 480")

        actions = [a for a, v in zip(self.buffer.actions_rot, valid) if v]

        # log_probs must reduce to scalars
        log_probs_list = []
        for lp, v in zip(self.buffer.log_probs, valid):
            if not v:
                continue
            lp_np = self._tensor_to_numpy(lp)
            if isinstance(lp_np, np.ndarray):
                if lp_np.size != 1:
                    raise ValueError(f"log_prob must be scalar, got shape {lp_np.shape}")
                log_probs_list.append(float(lp_np.flatten()[0]))
            else:
                log_probs_list.append(float(lp_np))

        log_probs_tensor = torch.as_tensor(np.array(log_probs_list), dtype=torch.float32)

        # True terminals (GAE bootstrap 0)
        all_is_terminals = np.array(self.buffer.is_terminals, dtype=bool)
        is_terminals_valid = all_is_terminals[valid_indices]
        is_terminals_tensor = torch.as_tensor(is_terminals_valid, dtype=torch.bool)

        # Truncations (GAE bootstrap from V(s')); absent buffer field -> all False
        if hasattr(self.buffer, 'is_truncated'):
            all_is_truncated = np.array(self.buffer.is_truncated, dtype=bool)
            is_truncated_valid = all_is_truncated[valid_indices]
        else:
            is_truncated_valid = np.zeros(len(valid_indices), dtype=bool)

        is_truncated_tensor = torch.as_tensor(is_truncated_valid, dtype=torch.bool)

        # Episode indices for schedule consistency
        all_episode_indices = np.array(self.buffer.episode_indices)
        episode_indices_valid = all_episode_indices[valid_indices]
        episode_indices_tensor = torch.as_tensor(episode_indices_valid, dtype=torch.long)

        # Rollout temperatures: must match rollout value or the PPO ratio breaks
        all_temperatures = np.array(self.buffer.temperatures, dtype=np.float32)
        temperatures_valid = all_temperatures[valid_indices]
        temperatures_tensor = torch.as_tensor(temperatures_valid, dtype=torch.float32)

        # state_values must reduce to scalars
        state_values_list = []
        for sv, v in zip(self.buffer.state_values, valid):
            if not v:
                continue
            sv_np = self._tensor_to_numpy(sv)
            if isinstance(sv_np, np.ndarray):
                if sv_np.size != 1:
                    raise ValueError(f"state_value must be scalar, got shape {sv_np.shape}")
                state_values_list.append(float(sv_np.flatten()[0]))
            else:
                state_values_list.append(float(sv_np))

        state_values_tensor = torch.as_tensor(np.array(state_values_list), dtype=torch.float32)

        # next_values = V(s') for GAE truncation bootstrap: state_values shifted by 1
        # (repeat last), then overridden by stored values at truncated steps
        if len(state_values_list) > 1:
            next_values = state_values_list[1:] + [state_values_list[-1]]
        else:
            next_values = state_values_list.copy()
        next_values_tensor = torch.as_tensor(next_values, dtype=torch.float32)

        if hasattr(self.buffer, 'truncation_bootstrap_values'):
            for original_idx, bootstrap_val in self.buffer.truncation_bootstrap_values.items():
                # Map original index to its position within valid_indices
                matches = np.where(valid_indices == original_idx)[0]
                if len(matches) > 0:
                    filtered_pos = matches[0]
                    next_values_tensor[filtered_pos] = float(bootstrap_val)

        log_probs_direction_list = []
        log_probs_magnitude_list = []
        log_probs_axis_list = []

        for lp_dir, lp_mag, lp_axis, v in zip(
            self.buffer.log_probs_direction,
            self.buffer.log_probs_magnitude,
            self.buffer.log_probs_axis,
            valid
        ):
            if not bool(v):
                continue

            lp_dir_np = self._tensor_to_numpy(lp_dir)
            lp_mag_np = self._tensor_to_numpy(lp_mag)
            lp_axis_np = self._tensor_to_numpy(lp_axis)

            if isinstance(lp_dir_np, np.ndarray) and lp_dir_np.size == 1:
                lp_dir_np = float(lp_dir_np.flatten()[0])
            if isinstance(lp_mag_np, np.ndarray) and lp_mag_np.size == 1:
                lp_mag_np = float(lp_mag_np.flatten()[0])
            if isinstance(lp_axis_np, np.ndarray) and lp_axis_np.size == 1:
                lp_axis_np = float(lp_axis_np.flatten()[0])

            log_probs_direction_list.append(float(lp_dir_np))
            log_probs_magnitude_list.append(float(lp_mag_np))
            log_probs_axis_list.append(float(lp_axis_np))

        log_probs_direction_tensor = torch.as_tensor(log_probs_direction_list, dtype=torch.float32)
        log_probs_magnitude_tensor = torch.as_tensor(log_probs_magnitude_list, dtype=torch.float32)
        log_probs_axis_tensor = torch.as_tensor(log_probs_axis_list, dtype=torch.float32)

        # Asymmetric O2 embedding (axis head): encodes O1 vs O2 ordering; zeros if missing
        o2_embeddings_asymmetric_list = []
        for o2_emb, v in zip(self.buffer.o2_embeddings_asymmetric, valid):
            if not v:
                continue
            if o2_emb is not None:
                if isinstance(o2_emb, np.ndarray):
                    o2_emb_tensor = torch.from_numpy(o2_emb).float()
                else:
                    o2_emb_tensor = o2_emb if isinstance(o2_emb, torch.Tensor) else torch.tensor(o2_emb, dtype=torch.float32)
                o2_embeddings_asymmetric_list.append(o2_emb_tensor)
            else:
                o2_embeddings_asymmetric_list.append(torch.zeros(480, dtype=torch.float32))

        if o2_embeddings_asymmetric_list:
            o2_embeddings_asymmetric_tensor = torch.stack(o2_embeddings_asymmetric_list, dim=0)  # [n_valid, 480]
        else:
            o2_embeddings_asymmetric_tensor = None
            _LOG.info(f"[BUFFER] No asymmetric embeddings found")

        batch = {
            'states': states,                   # [n_valid, state_dim]
            'actor_states': actor_states,       # [n_valid, 480] actor path
            'o1_embeddings': o1_embeddings,     # [n_valid, D] for CriticEmbedding
            'o2_embeddings': o2_embeddings,     # [n_valid, D] for CriticEmbedding
            'geometries': geometries,           # [n_valid, 6] SO(3)-invariant geometry
            'actions': actions,                 # list of action dicts
            'log_probs': log_probs_tensor,      # [n_valid] combined
            'state_values': state_values_tensor,  # [n_valid]
            'is_terminals': is_terminals_tensor,   # [n_valid] true terminals only
            'is_truncated': is_truncated_tensor,   # [n_valid] timeouts, bootstrap from V
            'next_values': next_values_tensor,     # [n_valid] V(s') truncation bootstrap

            # [n_valid, 480] O1-vs-O2 ordering for axis head
            'o2_embeddings_asymmetric': o2_embeddings_asymmetric_tensor,

            'rewards': rewards_tensor,          # [n_valid] total per-step reward
            'ea_rewards': ea_rewards_tensor,    # [n_valid]
            'dissoc_rewards': dissoc_rewards_tensor,  # [n_valid]
            'axis_rewards': axis_rewards_tensor,      # [n_valid]

            # Per-head log-probs for separate policy losses
            'log_probs_direction': log_probs_direction_tensor,  # [n_valid]
            'log_probs_magnitude': log_probs_magnitude_tensor,  # [n_valid]
            'log_probs_axis': log_probs_axis_tensor,            # [n_valid]

            # Schedule consistency: episode index + rollout temperature
            'episode_indices': episode_indices_tensor,  # [n_valid]
            'temperatures': temperatures_tensor,        # [n_valid]
            'valid_count': n_valid,
            'total_count': n_total,
            'success_stats': success_stats,
        }

        # Environment multipole, for consistent rollout/evaluate injection
        _mp_len = len(self.buffer.env_multipoles) if hasattr(self.buffer, 'env_multipoles') else 0
        _mp_nones = sum(1 for x in self.buffer.env_multipoles if x is None) if _mp_len > 0 else 0
        if hasattr(self.buffer, 'env_multipoles') and self.buffer.env_multipoles:        
            env_mp_list = []
            for mp, v in zip(self.buffer.env_multipoles, valid):
                if not v:
                    continue
                if mp is not None:
                    if isinstance(mp, np.ndarray):
                        env_mp_list.append(torch.from_numpy(mp).float())
                    else:
                        env_mp_list.append(torch.tensor(mp, dtype=torch.float32))
                else:
                    env_mp_list.append(torch.zeros(9, dtype=torch.float32))
            if env_mp_list:
                batch['env_multipoles'] = torch.stack(env_mp_list, dim=0)

        return batch

    def set_current_episode(self, episode_idx: int):
        """Set current episode for new transitions."""
        self.buffer.set_current_episode(episode_idx)

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

