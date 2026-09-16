"""PPO update machinery (PPOUpdateMixin).

The large optimization-step methods kept out of ppo_agent.py:
    PPO.update, PPO.update_from_aggregated_batch, PPO._track_backbone_updates.
They are mixed into PPO via `class PPO(PPOUpdateMixin)` in ppo_agent.py.

The mixin holds no state and defines no __init__/dunders -- all attributes
(self.policy, self.optimizer, self.buffer_manager, self.scheduler, ...) are
created by PPO.__init__ in ppo_agent.py and reached through `self` at call time.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import inspect
from typing import Optional, Dict, Any, Tuple

from realize.ppo.ppo_optimizer import PPOOptimizer
from realize.ppo.ppo_buffer_manager import PPOBufferManager
from realize.ppo.schedulers import UnifiedAnnealingScheduler
from realize.networks.policy_network import PolicyNetwork
from realize.config.default_config import PPOConfig
import logging
_LOG = logging.getLogger(__name__)


class PPOUpdateMixin:
    def update(self, episode: int, quick_update: bool = False):
        """Run one PPO update: GAE advantages, success-weighted sampling, critic replay,
        then optimize(). quick_update does a single epoch. Raises on non-int episode or
        empty buffer.
        """
        if not isinstance(episode, int):
            raise TypeError(f"Episode must be an integer, got {type(episode)}: {episode}")

        if not self.buffer_manager.has_data():
            raise RuntimeError(
                f"PPO.update() called with empty buffer at episode {episode}. "
                f"This indicates no experiences were collected during episode. "
                f"Check that select_action() was called and added data to buffer."
            )
        
        self.policy.train()
        self.current_episode = episode

        # Success-weighted batch: with ratio 3.0 and ~10% success, ~25% of steps are
        # from successful episodes. "No valid transitions" is handled gracefully.
        try:
            if hasattr(self.buffer_manager, 'get_batch'):
                sig = inspect.signature(self.buffer_manager.get_batch)
                if 'use_success_weighting' in sig.parameters:
                    batch_data = self.buffer_manager.get_batch(use_success_weighting=True)
                    if episode == 0 or episode % 50 == 0:
                        _LOG.debug(f"[PPO VERIFY] OK Success-weighted sampling ACTIVE")
                else:
                    batch_data = self.buffer_manager.get_batch()
                    if episode == 0:
                        _LOG.debug(f"[PPO VERIFY] FAIL Success-weighted sampling FALLBACK (old buffer_manager)")
            else:
                batch_data = self.buffer_manager.get_batch()            
        except RuntimeError as e:
            if "No valid transitions" in str(e):
                _LOG.info(f"[PPO] Skipping update at episode {episode}: {e}")
                _LOG.info(f"[PPO] This typically means all steps in recent episodes failed NEB.")
                self._clear_rollout_window("skip_no_valid_transitions")
                return
            raise

        if batch_data is None:
            _LOG.info(f"[PPO] Skipping update: get_batch() returned None at episode {episode}")
            self._clear_rollout_window("skip_get_batch_none")
            return

        if 'actions' not in batch_data:
            raise RuntimeError(
                f"Batch missing 'actions' key at episode {episode}. "
                f"Buffer must store actions. "
                f"Batch keys: {list(batch_data.keys())}"
            )
        
        if not isinstance(batch_data['actions'], list):
            raise TypeError(
                f"batch_data['actions'] must be list, got {type(batch_data['actions'])} "
                f"at episode {episode}"
            )
        
        if not batch_data['actions']:
            raise RuntimeError(
                f"batch_data['actions'] is empty list at episode {episode}. "
                f"Buffer must contain at least one action."
            )

        # GAE lambda (bias-variance tradeoff): 0=one-step TD, 1=MC, 0.95 default,
        # 0.97-0.98 for very sparse rewards (propagates signal further)
        if hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'gae_lambda'):
            gae_lambda = self.config.algorithm.gae_lambda
        elif hasattr(self.config, 'gae_lambda'):
            gae_lambda = self.config.gae_lambda
        else:
            gae_lambda = 0.95

        sig = inspect.signature(self.optimizer_manager.compute_advantages)
        if 'gae_lambda' in sig.parameters:
            is_truncated = batch_data.get('is_truncated', None)
            next_values = batch_data.get('next_values', None)
            
            returns, advantages = self.optimizer_manager.compute_advantages(
                batch_data['rewards'],
                batch_data['state_values'],
                batch_data['is_terminals'],
                self.gamma,
                gae_lambda=gae_lambda,
                is_truncated=is_truncated,
                next_values=next_values,
                episode=episode
            )

            if episode == 0 or episode % 50 == 0:
                _LOG.debug(f"[PPO VERIFY] OK GAE ACTIVE (lambda={gae_lambda})")
                if is_truncated is not None:
                    trunc_count = is_truncated.sum().item() if hasattr(is_truncated, 'sum') else sum(is_truncated)
                    term_count = batch_data['is_terminals'].sum().item() if hasattr(batch_data['is_terminals'], 'sum') else sum(batch_data['is_terminals'])
                    _LOG.debug(f"[PPO VERIFY] OK Truncation handling: {term_count} terminals, {trunc_count} truncations")
                else:
                    _LOG.debug(f"[PPO VERIFY] WARNING No truncation data in batch")
        else:
            returns, advantages = self.optimizer_manager.compute_advantages(
                batch_data['rewards'],
                batch_data['state_values'],
                batch_data['is_terminals'],
                self.gamma,
                episode=episode
            )
            if episode == 0:
                _LOG.debug(f"[PPO VERIFY] FAIL GAE FALLBACK (old optimizer, using MC returns)")

        # Variance-collapse check, run every update
        returns_std = returns.std().item()
        returns_mean = returns.mean().item()

        if returns_std < 0.1:
            _LOG.info(f"\n[WARNING RETURN VARIANCE LOW] Episode {episode}")
            _LOG.info(f"  returns: mean={returns_mean:.4f}, std={returns_std:.6f}")
            _LOG.info(f"  This makes EV unreliable! Need more diverse outcomes.")

            returns_np = returns.cpu().numpy()
            _LOG.info(f"  returns histogram: min={returns_np.min():.3f}, max={returns_np.max():.3f}")
            unique_returns = len(set(returns_np.round(2)))
            _LOG.info(f"  unique returns (rounded): {unique_returns}")

        if episode % 10 == 0:
            with torch.no_grad():
                v_pred = batch_data['state_values']
                if not isinstance(v_pred, torch.Tensor):
                    v_pred = torch.tensor(v_pred, dtype=torch.float32)
                v_pred = v_pred.to(self.device)

                v_actual = returns.to(self.device)

                mse = ((v_pred - v_actual) ** 2).mean().item()
                mae = (v_pred - v_actual).abs().mean().item()

                if len(v_pred) > 1:
                    corr = np.corrcoef(v_pred.cpu().numpy(), v_actual.cpu().numpy())[0, 1]
                else:
                    corr = 0.0

                adv_mean = advantages.mean().item()
                adv_std = advantages.std().item()

                # Explained variance EV = 1 - Var(returns - values) / Var(returns);
                # undefined when Var(returns) is tiny
                return_var = v_actual.var().item()
                residual_var = (v_actual - v_pred).var().item()

                if return_var < 1e-6:
                    explained_variance = float('nan')
                    ev_status = "UNDEFINED (return_var too small)"
                else:
                    explained_variance = 1.0 - residual_var / return_var
                    if explained_variance < -10:
                        ev_status = f"ALERT EXTREME NEGATIVE ({explained_variance:.1f})"
                    elif explained_variance < 0:
                        ev_status = f"WARNING NEGATIVE ({explained_variance:.3f})"
                    elif explained_variance < 0.3:
                        ev_status = f"LOW ({explained_variance:.3f})"
                    else:
                        ev_status = f"OK OK ({explained_variance:.3f})"

                _LOG.debug(f"[VALUE FUNC] Ep {episode}:")
                _LOG.debug(f"  V(s) error: MSE={mse:.4f}, MAE={mae:.4f}")
                _LOG.debug(f"  V(s) corr with returns: {corr:.3f}")
                _LOG.debug(f"  Advantages: mean={adv_mean:.4f}, std={adv_std:.4f}")
                _LOG.debug(f"  V(s) range: [{v_pred.min():.3f}, {v_pred.max():.3f}]")
                _LOG.debug(f"  Returns range: [{v_actual.min():.3f}, {v_actual.max():.3f}]")
                _LOG.debug(f"  Explained Variance: {ev_status}")
                _LOG.debug(f"    return_var={return_var:.6f}, residual_var={residual_var:.6f}")

                # Collapse detection
                v_pred_std = v_pred.std().item()
                v_actual_std = v_actual.std().item()
                
                if v_pred_std < 0.01:
                    _LOG.info(f"  ALERT VALUE COLLAPSE: V(s) std={v_pred_std:.6f} - Critic outputting near-constant!")
                if v_actual_std < 0.1:
                    _LOG.info(f"  WARNING LOW RETURN VARIANCE: std={v_actual_std:.4f} - EV will be unstable!")
                if adv_std < 0.01:
                    _LOG.info(f"  ALERT ADVANTAGE COLLAPSE: std={adv_std:.6f} - No learning signal!")

        # Store this rollout in the critic replay buffer (after GAE, before optimize):
        # Stage 1 = states+returns, Stage 2 = +actions+per-head log-probs+advantages
        if self._replay_enabled and hasattr(self.buffer_manager, 'add_to_critic_replay'):
            success_stats = self.get_buffer_success_stats()
            episode_success = False
            if success_stats and success_stats.get('success_rate', 0) > 0:
                episode_success = True

            states_for_replay = batch_data['states']

            actions_for_replay = batch_data.get('actions', None)
            log_probs_dir_for_replay = batch_data.get('log_probs_direction', None)
            log_probs_mag_for_replay = batch_data.get('log_probs_magnitude', None)
            log_probs_axis_for_replay = batch_data.get('log_probs_axis', None)

            self.buffer_manager.add_to_critic_replay(
                states=states_for_replay,
                returns=returns,
                is_success=episode_success,
                actions=actions_for_replay,
                log_probs_direction=log_probs_dir_for_replay,
                log_probs_magnitude=log_probs_mag_for_replay,
                log_probs_axis=log_probs_axis_for_replay,
                advantages=advantages
            )

            if episode % 10 == 0:
                replay_stats = self.buffer_manager.get_replay_statistics()
                if replay_stats:
                    _LOG.debug(f"[REPLAY] Buffer: {replay_stats['size']}/{replay_stats['capacity']} "
                          f"({replay_stats['fill_ratio']:.1%}), "
                          f"success_ratio={replay_stats['success_ratio']:.1%}")

        # Sample a replay batch for critic training once the buffer is warm enough
        replay_batch = None
        if self._replay_enabled and hasattr(self.buffer_manager, 'sample_critic_replay'):
            replay_batch_size = 256
            if self._replay_config is not None:
                replay_batch_size = self._replay_config.replay_batch_size

            min_size = 1000
            if self._replay_config is not None:
                min_size = self._replay_config.min_size_before_sampling

            replay_stats = self.buffer_manager.get_replay_statistics()
            if replay_stats and replay_stats['size'] >= min_size:
                replay_batch = self.buffer_manager.sample_critic_replay(
                    batch_size=replay_batch_size,
                    current_episode=episode
                )
                
                if replay_batch is not None and episode % 20 == 0:
                    _LOG.debug(f"[REPLAY] Sampled {replay_batch_size} transitions for critic training")

        if hasattr(self.optimizer_manager, 'set_replay_batch'):
            self.optimizer_manager.set_replay_batch(replay_batch)

        policy_replay_batch = None
        if self._replay_enabled and hasattr(self.buffer_manager, 'sample_policy_replay'):
            policy_replay_batch = self.buffer_manager.sample_policy_replay(
                current_episode=episode
            )
            if policy_replay_batch is not None and episode % 20 == 0:
                _LOG.debug(f"[REPLAY] Sampled policy replay batch for Stage 2")

        if hasattr(self.optimizer_manager, 'set_policy_replay_batch'):
            self.optimizer_manager.set_policy_replay_batch(policy_replay_batch)

        # Per-head advantages (direction/magnitude/axis) from the filtered batch
        if hasattr(self.optimizer_manager, 'compute_per_head_advantages'):
            per_head_advantages = self.optimizer_manager.compute_per_head_advantages(
                batch_data, self.gamma, gae_lambda
            )
            batch_data['advantages_direction'] = per_head_advantages.get('direction', advantages)
            batch_data['advantages_magnitude'] = per_head_advantages.get('magnitude', advantages)
            batch_data['advantages_axis'] = per_head_advantages.get('axis', advantages)
            
            if episode == 0 or episode % 50 == 0:
                _LOG.debug(f"[PPO VERIFY] OK Per-head advantages ACTIVE (dir/mag/axis)")
                _LOG.debug(f"  Batch size: {len(batch_data['states'])}")
                _LOG.debug(f"  Dir: {batch_data['advantages_direction'].shape}, "
                      f"mean={batch_data['advantages_direction'].mean():.4f}")
                _LOG.debug(f"  Mag: {batch_data['advantages_magnitude'].shape}, "
                      f"mean={batch_data['advantages_magnitude'].mean():.4f}")
                _LOG.debug(f"  Axis: {batch_data['advantages_axis'].shape}, "
                      f"mean={batch_data['advantages_axis'].mean():.4f}")
                _LOG.debug(f"  Dir: {batch_data['advantages_direction'].shape}, "
                      f"Mag: {batch_data['advantages_magnitude'].shape}, "
                      f"Axis: {batch_data['advantages_axis'].shape}")
        else:
            if episode == 0:
                _LOG.debug(f"[PPO VERIFY] FAIL Per-head advantages FALLBACK (using shared advantages)")

        entropy_coeff = self.scheduler.get_entropy_coeff(episode)

        # Backbone fine-tuning setup: locate the embedding extractor and per-step state
        # file paths used to recompute embeddings during the update
        backbone_state_paths = None
        state_normalizer = getattr(self, 'state_normalizer', None)

        # embedding_extractor is injected via inject_backbone_parameters
        embedding_extractor = getattr(self, 'embedding_extractor', None)

        _LOG.debug(f"\n[BACKBONE PATHS Ep{episode}]")
        _LOG.debug(f"  embedding_extractor present: {embedding_extractor is not None}")

        if embedding_extractor is not None:
            actions = batch_data.get('actions', [])
            paths = []

            _LOG.debug(f"\n[PATH EXTRACTION Ep{episode}]")
            _LOG.debug(f"  Total actions: {len(actions)}")
            _LOG.debug(f"  Total states: {len(batch_data['states'])}")

            if isinstance(actions, list):
                for i, a in enumerate(actions[:3]):
                    if isinstance(a, dict):
                        path = a.get('state_file_path')
                        _LOG.debug(f"    Action[{i}]: has state_file_path = {path is not None}, value = {path}")
                        paths.append(path)
                    else:
                        _LOG.debug(f"    Action[{i}]: not a dict, type = {type(a)}")
                        paths.append(None)
        state_normalizer = getattr(self, 'state_normalizer', None)

        embedding_extractor = getattr(self, 'embedding_extractor', None)

        if embedding_extractor is None:
            # Fall back to the extractor on episode_runner if attached there
            if hasattr(self, 'episode_runner'):
                embedding_extractor = getattr(self.episode_runner, 'embedding_extractor', None)

        if embedding_extractor is not None:
            actions = batch_data.get('actions', [])
            paths = []

            if isinstance(actions, list):
                for a in actions:
                    if isinstance(a, dict):
                        path = a.get('state_file_path')
                        paths.append(path)
                        if len(paths) <= 3:
                            _LOG.debug(f"    Action[{len(paths)-1}] state_file_path: {path}")
                    else:
                        paths.append(None)
            elif isinstance(actions, dict) and 'state_file_path' in actions:
                # Batched dict input
                p = actions['state_file_path']
                if isinstance(p, (list, tuple, np.ndarray)):
                    paths = list(p)

            _LOG.debug(f"  Extracted {len(paths)} paths")

            if len(paths) != len(batch_data['states']):
                _LOG.debug(f"  FAIL Length mismatch: {len(paths)} paths vs {len(batch_data['states'])} states")
                backbone_state_paths = None
            else:
                none_count = sum(1 for p in paths if p is None)
                missing_count = sum(1 for p in paths if p is not None and not os.path.exists(p))
                
                _LOG.debug(f"  None paths: {none_count}")
                _LOG.debug(f"  Missing files: {missing_count}")
                
                if none_count > 0 or missing_count > 0:
                    _LOG.debug(f"  FAIL Invalid paths detected")
                    if none_count > 0:
                        none_indices = [i for i, p in enumerate(paths) if p is None][:5]
                        _LOG.debug(f"     First None indices: {none_indices}")
                    if missing_count > 0:
                        missing_paths = [p for p in paths if p is not None and not os.path.exists(p)][:3]
                        _LOG.debug(f"     First missing paths: {missing_paths}")
                    backbone_state_paths = None
                else:
                    _LOG.debug(f"  OK All paths valid")
                    backbone_state_paths = paths
            
            if backbone_state_paths is not None and len(backbone_state_paths) == len(batch_data['states']) and all(p is not None and os.path.exists(p) for p in backbone_state_paths):
                pass
            else:
                _LOG.debug(f"  Final: backbone_state_paths = {'None' if backbone_state_paths is None else f'{len(backbone_state_paths)} paths'}")                

        _LOG.debug(f"\n[PASSING TO OPTIMIZER Ep{episode}]")
        _LOG.debug(f"  embedding_extractor: {embedding_extractor is not None}")
        _LOG.debug(f"  state_file_paths: {backbone_state_paths is not None}")
        if backbone_state_paths is not None:
            _LOG.debug(f"  Number of paths: {len(backbone_state_paths)}")
        
        actor_extractor = getattr(self, 'actor_embedding_extractor', embedding_extractor)
        critic_extractor = getattr(self, 'critic_embedding_extractor', embedding_extractor)

        # Dual normalizers attached by the trainer from episode_runner
        actor_state_normalizer = getattr(self, 'actor_state_normalizer', state_normalizer)
        losses, metrics = self.optimizer_manager.optimize(
            batch_data=batch_data,
            returns=returns,
            advantages=advantages,
            entropy_coeff=entropy_coeff,
            episode_idx=episode,
            quick_update=quick_update,
            policy=self.policy,
            policy_old=self.policy_old,
            actor_embedding_extractor=actor_extractor,
            critic_embedding_extractor=critic_extractor,
            state_file_paths=backbone_state_paths,
            actor_state_normalizer=actor_state_normalizer,
            critic_state_normalizer=None
        )

        # Log metrics (+ SH viz params if available)
        if self.logger:
            self.logger.log_update(episode, losses, metrics)
            if hasattr(self.policy, 'sh_head') and hasattr(self.policy.sh_head, 'get_viz_params'):
                try:
                    viz = self.policy.sh_head.get_viz_params()
                    self.logger.log_debug(f"SH viz: temps={viz.get('channel_temps')}, scales={viz.get('channel_scales')}")
                except Exception:
                    pass

            if replay_batch is not None and 'value_replay' in losses:
                self.logger.log_debug(f"[REPLAY] value_loss_replay={losses['value_replay']:.4f}")

        if episode % 10 == 0:
            success_stats = self.get_buffer_success_stats()
            if success_stats and success_stats['total_episodes'] > 0:
                _LOG.debug(f"[PPO] Buffer success rate: {success_stats['success_rate']:.1%} "
                      f"({success_stats['success_count']}/{success_stats['total_episodes']} episodes)")

        self._track_backbone_updates(episode)

        # policy_old now matches policy, ready for the next rollout cycle
        self.sync_policy_old()

        # Replay buffer persists across updates; only the rollout window is cleared
        if not quick_update:
            self._clear_rollout_window("post_update")

    def update_from_aggregated_batch(self, batch_data: dict, episode: int) -> tuple:
        """PPO update from a pre-aggregated batch (distributed collective PPO, batch shared
        across workers). Backbone fine-tuning is disabled here since state_file_paths
        cannot be serialized across workers. Returns (losses, metrics).
        """
        required_fields = ['states', 'log_probs', 'state_values', 'rewards', 'is_terminals']
        for field in required_fields:
            if field not in batch_data:
                raise ValueError(f"batch_data missing required field: {field}")

        self.policy.train()
        self.current_episode = episode

        gae_lambda = getattr(self.config.algorithm, 'gae_lambda', 0.95)

        sig = inspect.signature(self.optimizer_manager.compute_advantages)
        if 'gae_lambda' in sig.parameters:
            is_truncated = batch_data.get('is_truncated', None)
            next_values = batch_data.get('next_values', None)
            
            returns, advantages = self.optimizer_manager.compute_advantages(
                batch_data['rewards'],
                batch_data['state_values'],
                batch_data['is_terminals'],
                self.gamma,
                gae_lambda=gae_lambda,
                is_truncated=is_truncated,
                next_values=next_values,
                episode=episode                
            )        
        else:
            returns, advantages = self.optimizer_manager.compute_advantages(
                batch_data['rewards'],
                batch_data['state_values'],
                batch_data['is_terminals'],
                self.gamma,
                episode=episode
            )
        # GAE truncation handling + value-function accuracy diagnostic
        if episode % 10 == 0:
            term_count = batch_data['is_terminals'].sum().item() if hasattr(batch_data['is_terminals'], 'sum') else sum(batch_data['is_terminals'])
            trunc_count = is_truncated.sum().item() if is_truncated is not None and hasattr(is_truncated, 'sum') else 0
            _LOG.debug(f"\n[GAE DIAGNOSTIC] Episode {episode}")
            _LOG.debug(f"  Batch size: {len(batch_data['rewards'])}")
            _LOG.debug(f"  Terminals: {term_count}, Truncations: {trunc_count}")
            _LOG.debug(f"  Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
            _LOG.debug(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}") 
            with torch.no_grad():
                v_pred = batch_data['state_values']
                if not isinstance(v_pred, torch.Tensor):
                    v_pred = torch.tensor(v_pred, dtype=torch.float32)
                v_pred = v_pred.to(self.device)
                v_actual = returns.to(self.device)
    
                mse = ((v_pred - v_actual) ** 2).mean().item()
                mae = (v_pred - v_actual).abs().mean().item()
    
                _LOG.debug(f"[COLLECTIVE UPDATE] Value error: MSE={mse:.4f}, MAE={mae:.4f}")
                _LOG.debug(f"  Advantages: mean={advantages.mean().item():.4f}, "
                      f"std={advantages.std().item():.4f}")
    
        # Per-head advantages (direction/magnitude/axis)
        if hasattr(self.optimizer_manager, 'compute_per_head_advantages'):
            per_head_advantages = self.optimizer_manager.compute_per_head_advantages(
                batch_data, self.gamma, gae_lambda
            )
            batch_data['advantages_direction'] = per_head_advantages.get('direction', advantages)
            batch_data['advantages_magnitude'] = per_head_advantages.get('magnitude', advantages)
            batch_data['advantages_axis'] = per_head_advantages.get('axis', advantages)

        entropy_coeff = self.scheduler.get_entropy_coeff(episode)

        # Backbone fine-tuning stays disabled in distributed mode
        losses, metrics = self.optimizer_manager.optimize(
            batch_data=batch_data,
            returns=returns,
            advantages=advantages,
            entropy_coeff=entropy_coeff,
            episode_idx=episode,
            quick_update=False,
            policy=self.policy,
            policy_old=self.policy_old,
            # File paths cannot be serialized across workers, so disabled here
            actor_embedding_extractor=None,
            critic_embedding_extractor=None,
            state_file_paths=None,
            actor_state_normalizer=getattr(self, 'actor_state_normalizer', None),
            critic_state_normalizer=None
        )

        if self.logger:
            self.logger.log_update(episode, losses, metrics)

        self.sync_policy_old()

        return losses, metrics

    def _track_backbone_updates(self, episode: int):
        """Every 20 episodes, log the actor-backbone parameter-norm change; a near-zero
        relative change means the backbone is stuck.
        """
        if episode % 20 != 0:
            return

        if not hasattr(self, 'actor_embedding_extractor'):
            return

        extractor = self.actor_embedding_extractor
        if extractor is None or not hasattr(extractor, 'model'):
            return

        current_norm = 0.0
        trainable_params = 0

        for param in extractor.model.parameters():
            if param.requires_grad:
                trainable_params += 1
                current_norm += param.data.norm().item() ** 2

        current_norm = np.sqrt(current_norm) if current_norm > 0 else 0.0

        if hasattr(self, '_backbone_param_norm_history'):
            prev_entry = self._backbone_param_norm_history[-1]
            prev_norm = prev_entry['norm']
            prev_episode = prev_entry['episode']
            
            update_magnitude = abs(current_norm - prev_norm)
            relative_change = update_magnitude / (prev_norm + 1e-8)
            episodes_elapsed = episode - prev_episode
            
            _LOG.debug(f"\n[BACKBONE UPDATES] Episode {episode}:")
            _LOG.debug(f"  Trainable params: {trainable_params}")
            _LOG.debug(f"  Current norm: {current_norm:.6f}")
            _LOG.debug(f"  Previous norm: {prev_norm:.6f} (Ep {prev_episode})")
            _LOG.debug(f"  Change: {update_magnitude:.6f} ({relative_change*100:.4f}%)")
            _LOG.debug(f"  Per-episode rate: {update_magnitude/episodes_elapsed:.6f}")

            if relative_change < 0.0001:
                _LOG.info(f"  Parameters barely changed (<0.01%) - backbone stuck!")
            elif relative_change < 0.001:
                _LOG.info(f"  WARNING WARNING: Small parameter change (<0.1%) - learning slow?")
            else:
                _LOG.debug(f"  OK Parameters updating normally")
        else:
            _LOG.debug(f"\n[BACKBONE UPDATES] Episode {episode}: First snapshot")
            _LOG.debug(f"  Trainable params: {trainable_params}")
            _LOG.debug(f"  Initial norm: {current_norm:.6f}")
            self._backbone_param_norm_history = []

        self._backbone_param_norm_history.append({
            'episode': episode,
            'norm': current_norm
        })

