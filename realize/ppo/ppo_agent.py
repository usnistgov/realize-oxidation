"""PPO agent: owns the policy, optimizer, and buffers."""
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
from realize.networks.validate_architecture import validate_architecture
from realize.config.default_config import PPOConfig
from realize.ppo.ppo_agent_update import PPOUpdateMixin
import logging
_LOG = logging.getLogger(__name__)


class PPO(PPOUpdateMixin):
    """Main PPO agent orchestrator: owns the policy, optimizer manager, buffer manager,
    scheduler, and optional critic replay buffer for O2-dissociation sparse rewards.
    """

    def __init__(self,
                 config: Optional[PPOConfig] = None,
                 env=None,
                 device: Optional[torch.device] = None):
        """Build the PPO agent. config (PPOConfig) is required; device defaults to CUDA
        when available. Raises ValueError if config is None.
        """
        if config is None:
            raise ValueError(
                "PPO requires an explicit PPOConfig. "
                "Cannot create agent without configuration. "
                "Create config using create_default_config() or custom config."
            )
        self.config = config
        self.env = env

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.gamma = config.algorithm.gamma
        self.current_episode = 0

        self.policy = PolicyNetwork(
            config=config,
            device=self.device,
            total_episodes=config.runtime.total_episodes,
            zero_backbone_l1l2=True,
            geometry_only_critic=True
        ).to(self.device)

        # policy_old: frozen snapshot for old_logprob via the same code path as
        # new_logprob (only weights differ), so ratio = exp(new - old) is correct
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.eval()
        for p in self.policy_old.parameters():
            p.requires_grad = False
        _LOG.info(f"[PPO] Created policy_old snapshot for consistent PPO ratios")

        # Architecture-invariant check at construction: irreps, lmax/SH dim, S2 grid,
        # RQS bins, vMF axis head, frozen critic. Non-fatal here (an introspection bug
        # must never block a run); the strict form runs in tests/smoke_test.py --full.
        try:
            arch_results = validate_architecture(self.policy, config, strict=False)
            arch_failed = [r for r in arch_results if not r.ok]
            if arch_failed:
                _LOG.info(f"[ARCH] validate_architecture: {len(arch_failed)}/{len(arch_results)} "
                      f"invariant check(s) FAILED - possible architectural drift:")
                for r in arch_failed:
                    _LOG.info(f"[ARCH]   {r}")
            else:
                _LOG.info(f"[ARCH] validate_architecture: {len(arch_results)}/{len(arch_results)} "
                      f"checks OK")
        except Exception as arch_exc:
            _LOG.info(f"[ARCH] validate_architecture skipped (introspection error: {arch_exc})")

        self.logger = None  # set by trainer

        self.optimizer_manager = PPOOptimizer(
            self.policy,
            config,
            device=self.device,
            logger=None  # set via set_logger
        )

        # Full config so replay settings are available
        self.buffer_manager = PPOBufferManager(
            config,
            device=self.device
        )

        # All transitions in one PPO accumulation window share one temperature: frozen
        # on the first select_action() after a buffer clear, reused until the next clear
        self._rollout_temperature = None

        self.scheduler = UnifiedAnnealingScheduler(
            sampling_config=config.sampling,
            algorithm_config=config.algorithm
        )

        self.logger = None

        self.training_rewards = []
        self.recent_dissociations = []
        self.dissociation_window = 100

        self._replay_enabled = False
        self._replay_config = None
        if hasattr(config, 'replay'):
            self._replay_config = config.replay
            self._replay_enabled = config.replay.enable_critic_replay
            if self._replay_enabled:
                _LOG.info(f"[PPO] Critic replay ENABLED "
                      f"(capacity={config.replay.capacity}, "
                      f"weight={config.replay.critic_replay_weight})")
        
        _LOG.info(f"PPO initialized on {self.device} with {config.model.num_channels} channels")

    def sync_policy_old(self):
        """Copy current policy weights into policy_old. Call after each PPO update so
        policy_old matches the policy used for the next rollout.
        """
        # Diagnostic: weight diff before the sync
        diff_norm = 0.0
        param_count = 0
        for (name, p_old), (_, p_new) in zip(
            self.policy_old.named_parameters(), 
            self.policy.named_parameters()
        ):
            diff_norm += (p_old - p_new).abs().sum().item()
            param_count += p_old.numel()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if param_count > 0:
            avg_diff = diff_norm / param_count
            _LOG.debug(f"[PPO] Synced policy_old <- policy | avg_weight_diff={avg_diff:.6f}")
            if avg_diff < 1e-10:
                _LOG.info(f"  WARNING WARNING: policy weights unchanged from policy_old!")
        else:
            _LOG.debug(f"[PPO] Synced policy_old <- policy")

    def _reset_rollout_window(self):
        """Reset rollout window state. Call after every buffer clear."""
        self._rollout_temperature = None

    def _clear_rollout_window(self, reason: str = ""):
        """Clear buffer AND reset rollout temperature atomically.

        Always use this instead of buffer_manager.clear() directly.
        This ensures temperature is always reset when buffer is cleared.
        """
        self.buffer_manager.clear()
        self._reset_rollout_window()
        if reason:
            _LOG.debug(f"[PPO] Buffer cleared ({reason}); rollout window reset")

    def set_logger(self, logger):
        """Set the logger and propagate it to optimizer/buffer managers and policy."""
        self.logger = logger

        if hasattr(self, 'optimizer_manager'):
            self.optimizer_manager.logger = logger

        if hasattr(self, 'buffer_manager') and hasattr(self.buffer_manager, 'set_logger'):
            self.buffer_manager.set_logger(logger)

        if hasattr(self.policy, 'set_logger'):
            self.policy.set_logger(logger)

    def inject_backbone_parameters(self, actor_extractor, critic_extractor=None):
        """Register the dual backbone extractors with the optimizer. actor_extractor is
        the trainable actor backbone; critic_extractor (required) is frozen or trainable
        per config. Passing the same object twice selects single-backbone mode.
        """
        if actor_extractor is None:
            _LOG.info("[PPO] No actor_extractor provided - skipping backbone injection")
            return

        # Dual backbone requires an explicit critic_extractor
        if critic_extractor is None:
            raise ValueError(
                "DUAL BACKBONE MODE REQUIRES critic_extractor!\n"
                "Pass both actor and critic extractors:\n"
                "  agent.inject_backbone_parameters(actor_extractor, critic_extractor)\n"
                "To use single backbone, pass the same extractor twice:\n"
                "  agent.inject_backbone_parameters(extractor, extractor)"
            )
    
        self.actor_embedding_extractor = actor_extractor
        self.critic_embedding_extractor = critic_extractor

        if actor_extractor is critic_extractor:
            _LOG.info("[PPO] WARNING: Actor and critic extractors are the SAME object (single backbone mode)")
            _LOG.info("     For true dual backbone, create separate EmbeddingExtractor instances")
        else:
            # Dual backbone needs separate model instances, not just separate wrappers
            if id(actor_extractor.model) == id(critic_extractor.model):
                raise RuntimeError(
                    "DUAL BACKBONE BROKEN: Actor and critic share the same SevenNet model!\n"
                    "They must have separate model instances."
                )
            # Freeze status is controlled by config, only reported here
            critic_model_frozen = all(not p.requires_grad for p in critic_extractor.model.parameters())
            critic_trainable_count = sum(
                p.numel() for p in critic_extractor.model.parameters() if p.requires_grad
            )
    
            _LOG.info("[PPO] Dual backbone validation passed")
            _LOG.info(f"  Actor and critic are different objects: OK")
            _LOG.info(f"  Critic backbone frozen: {critic_model_frozen} "
                  f"(trainable params: {critic_trainable_count})")
    
        self.embedding_extractor = actor_extractor

        self.optimizer_manager.inject_backbone_parameters(actor_extractor)

        # Inject critic backbone params only when it is trainable
        if critic_extractor is not None and critic_extractor.train_backbone:
            self.optimizer_manager.inject_critic_backbone_parameters(critic_extractor)
    
        actor_param_count = sum(p.numel() for p in actor_extractor.parameters() if p.requires_grad)
        critic_param_count = sum(p.numel() for p in critic_extractor.parameters() if p.requires_grad)
        critic_status = "trainable" if critic_extractor.train_backbone else "frozen"
    
        _LOG.info("[PPO] Dual backbone parameters injected:")
        _LOG.info(f"  Actor:  {actor_param_count} trainable params")
        _LOG.info(f"  Critic: {critic_param_count} params ({critic_status})")

    def select_action(self, state: np.ndarray, episode: int = 0, step: int = 0,
                      exploration_phase: bool = False, state_uid: Optional[int] = None,
                      o2_embedding_asymmetric: Optional[torch.Tensor] = None,
                      neighbor_positions: Optional[torch.Tensor] = None,
                      state_file_path: Optional[str] = None,
                      actor_state: Optional[torch.Tensor] = None,
                      o1_embedding: Optional[torch.Tensor] = None,
                      o2_embedding: Optional[torch.Tensor] = None,
                      geometry: Optional[torch.Tensor] = None,
                      o2_positions: Optional[torch.Tensor] = None,
                      env_multipole: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray, bool]:

        """Sample an action from the policy and store the transition in the buffer."""
        # Stay in train() during rollouts (only gradients are disabled): sampling must
        # use the same mode as optimize(), so do NOT call policy.eval()
        prev_training = self.policy.training

        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)

            state = state.to(self.device)

            if state.ndim == 1:
                state = state.unsqueeze(0)

            # Freeze the temperature for the whole accumulation window so PPO log-probs
            # stay consistent across multi-episode batches
            if self._rollout_temperature is None:
                self._rollout_temperature = float(self.scheduler.get_temperature(episode))
                _LOG.debug(f"[ROLLOUT] Freezing temperature for accumulation window: {self._rollout_temperature:.6f} (episode {episode})")
            
            temperature = self._rollout_temperature
            self.policy.set_temperature(temperature)

            # Asymmetric O2 embedding (axis head) -> tensor with batch dim, if provided
            o2_embedding_asymmetric_tensor = None
            if o2_embedding_asymmetric is not None:
                if not isinstance(o2_embedding_asymmetric, torch.Tensor):
                    o2_embedding_asymmetric_tensor = torch.FloatTensor(o2_embedding_asymmetric).to(self.device)
                else:
                    o2_embedding_asymmetric_tensor = o2_embedding_asymmetric.to(self.device)

                if o2_embedding_asymmetric_tensor.dim() == 1:
                    o2_embedding_asymmetric_tensor = o2_embedding_asymmetric_tensor.unsqueeze(0)

            if episode == 0 and step == 0:
                if o1_embedding is not None and o2_embedding is not None and geometry is not None:
                    _LOG.debug(f"[ROLLOUT VALUE CHECK] OK critic inputs provided: "
                          f"o1={o1_embedding.shape}, o2={o2_embedding.shape}, geom={geometry.shape}")
                else:
                    _LOG.debug(f"[ROLLOUT VALUE CHECK] WARNING  critic inputs missing -- "
                          f"o1={o1_embedding is not None}, o2={o2_embedding is not None}, "
                          f"geom={geometry is not None}")

            action, log_prob, state_value, entropy = self.policy.act(
                state,
                exploration_phase=exploration_phase,
                episode=episode,
                step=step,
                state_uid=state_uid,
                o2_embedding_asymmetric=o2_embedding_asymmetric_tensor,
                neighbor_positions=neighbor_positions,
                o1_embedding=o1_embedding,
                o2_embedding=o2_embedding,
                geometry=geometry,
                o2_positions=o2_positions,
                env_multipole=env_multipole,                
            )

            if state_uid is not None and state_uid % 10 == 0:
                pass

            # Translation and axis vectors to CPU numpy, shaped [3]
            trans_vector = action['translation_vector'][0].cpu().numpy()
            axis_direction = action['axis_direction'][0].cpu().numpy()

            trans_vector = trans_vector.reshape(3)
            axis_direction = axis_direction.reshape(3)

            o2_config_np = np.stack([trans_vector, axis_direction], axis=0)

            if step % 5 == 0:
                _LOG.debug(f"  [select_action] state_file_path: {state_file_path}")

            buffer_action = {
                'state_file_path': state_file_path, 
                'translation_vector': action['translation_vector'][0].cpu(),
                'translation_direction': action['translation_direction'][0].cpu(),            
                'translation_magnitude': action['translation_magnitude'][0].cpu() if action['translation_magnitude'].dim() > 0 else action['translation_magnitude'].cpu(),
                'axis_direction': action['axis_direction'][0].cpu(),
                'axis_frame': action['axis_frame'][0].cpu(),                
                'axis_mean_direction': action['axis_mean_direction'][0].cpu(),
                'axis_log_concentration': action['axis_log_concentration'][0].cpu(),
                'o2_embedding_asymmetric': o2_embedding_asymmetric_tensor[0].cpu() if o2_embedding_asymmetric_tensor is not None else None,
                'o2_positions': o2_positions[0].cpu() if o2_positions is not None else None,                
            }

            # direction_indices is required for exact PPO log-prob parity (unless continuous)
            if 'direction_indices' not in action or action['direction_indices'] is None:
                if not getattr(self.config.model, 'use_continuous_direction', False):
                    raise RuntimeError("Policy.act() must return 'direction_indices'")

            direction_indices = action['direction_indices']
            if isinstance(direction_indices, torch.Tensor):
                buffer_action['direction_indices'] = direction_indices[0].cpu() if direction_indices.dim() > 0 else direction_indices.cpu()
            else:
                buffer_action['direction_indices'] = direction_indices

            if hasattr(self.policy, '_last_outputs') and 'channel_weights' in self.policy._last_outputs:
                channel_weights = self.policy._last_outputs['channel_weights']
                if channel_weights is not None:
                    buffer_action['channel_weights'] = channel_weights.cpu() if isinstance(channel_weights, torch.Tensor) else channel_weights

            log_prob_direction = self.policy._last_outputs.get('log_prob_direction')
            log_prob_magnitude = self.policy._last_outputs.get('log_prob_magnitude')
            log_prob_axis = self.policy._last_outputs.get('log_prob_axis')
            
            if log_prob_direction is None or log_prob_magnitude is None or log_prob_axis is None:
                raise RuntimeError("Per-head log-probs missing")

            sh_coeffs = self.policy._last_outputs.get('sh_coeffs', None)
            if sh_coeffs is None: raise RuntimeError("sh_coeffs missing")
            sh_coeffs = sh_coeffs[0].cpu() if sh_coeffs.dim() > 2 else sh_coeffs.cpu()

            trans_direction = self.policy._last_outputs.get('translation_direction', None)
            trans_magnitude = self.policy._last_outputs.get('translation_magnitude', None)

            if trans_direction is None or trans_magnitude is None:
                 raise RuntimeError("Translation components missing")

            trans_direction = trans_direction[0].cpu() if trans_direction.dim() > 1 else trans_direction.cpu()
            trans_magnitude = trans_magnitude[0].cpu() if trans_magnitude.dim() > 0 else trans_magnitude.cpu()

            channel_weights_for_buffer = buffer_action.get('channel_weights', None)

            # Fall back to the single state when no separate actor_state is given
            state_for_buffer = state
            actor_state_for_buffer = actor_state if actor_state is not None else state
            buffer_add_success = self.buffer_manager.add(
                state=state_for_buffer,
                actor_state=actor_state_for_buffer,
                o1_embedding=o1_embedding,
                o2_embedding=o2_embedding,
                geometry=geometry,            
                action=buffer_action,
                log_prob=log_prob,
                state_value=state_value,
                temperature=float(temperature),
                channel_weights=channel_weights_for_buffer,
                log_prob_direction=log_prob_direction[0].cpu() if log_prob_direction.dim()>0 else log_prob_direction.cpu(),
                log_prob_magnitude=log_prob_magnitude[0].cpu() if log_prob_magnitude.dim()>0 else log_prob_magnitude.cpu(),
                log_prob_axis=log_prob_axis[0].cpu() if log_prob_axis.dim()>0 else log_prob_axis.cpu(),
                sh_coeffs=sh_coeffs,
                trans_direction=trans_direction,
                trans_magnitude=trans_magnitude,
                env_multipole=env_multipole                
            )

            if not buffer_add_success:
                raise RuntimeError("Buffer add failed")

            return trans_vector, o2_config_np, True

    def update_rewards(self, index: int, reward_total: float, done: bool,
                       ea_reward: float = 0.0,
                       dissociation_reward: float = 0.0,
                       additive: bool = False,
                       kl_divergence: float = 0.0,
                       is_truncated: bool = False,
                       **kwargs):
        """Reward update entrypoint. additive=False replaces the placeholder 0.0 (per-step);
        additive=True accumulates an episode-end terminal bonus. ea/dissociation components
        feed per-head reward streams. GAE bootstrap: done -> V=0; is_truncated -> V(s');
        neither -> normal TD.
        """
        if self.buffer_manager is None:
            raise RuntimeError("buffer_manager is None - cannot update rewards")
    
        mode = 'add' if additive else 'set'
        ok = self.buffer_manager.update_reward(
            index=index,
            reward=reward_total,
            is_terminal=done,
            is_truncated=is_truncated,
            mode=mode,
            ea_reward=ea_reward,
            dissociation_reward=dissociation_reward,
            kl_divergence=kl_divergence,
            **kwargs
        )
        if not ok:
            raise RuntimeError("buffer_manager.update_reward(...) returned False")
        if self.buffer_manager is None:
            raise RuntimeError("buffer_manager is None - cannot update rewards")

    def get_buffer_success_stats(self):
        """Get success statistics from buffer for monitoring."""
        if hasattr(self.buffer_manager, 'get_success_statistics'):
            return self.buffer_manager.get_success_statistics()
        return None

    def get_replay_statistics(self) -> Optional[Dict[str, Any]]:
        """Get replay buffer statistics for monitoring."""
        if hasattr(self.buffer_manager, 'get_replay_statistics'):
            return self.buffer_manager.get_replay_statistics()
        return None

    def save_checkpoint(self, episode: int):
        """Save checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer_manager.optimizer.state_dict(),
            'episode': episode,
            'config': self.config,
            'training_rewards': self.training_rewards,
            'recent_dissociations': self.recent_dissociations
        }

        if hasattr(self, 'embedding_extractor') and self.embedding_extractor is not None:
            checkpoint['embedding_extractor_state_dict'] = self.embedding_extractor.state_dict()
            _LOG.info(f"[CHECKPOINT] Saved embedding_extractor state_dict")
        else:
            _LOG.info(f"[CHECKPOINT] WARNING: No embedding_extractor to save!")

        if self._replay_enabled and hasattr(self.buffer_manager, 'critic_replay_buffer'):
            replay_buffer = self.buffer_manager.critic_replay_buffer
            if replay_buffer is not None:
                checkpoint['replay_buffer'] = {
                    'states': replay_buffer.states.copy() if isinstance(replay_buffer.states, list) else replay_buffer.states,
                    'returns': replay_buffer.returns.copy() if isinstance(replay_buffer.returns, list) else replay_buffer.returns,
                    'episode_ids': replay_buffer.episode_ids.copy() if isinstance(replay_buffer.episode_ids, list) else replay_buffer.episode_ids,
                    'is_success': replay_buffer.is_success.copy() if isinstance(replay_buffer.is_success, list) else replay_buffer.is_success,
                    # Stage 2 policy data
                    'actions': replay_buffer.actions.copy() if isinstance(replay_buffer.actions, list) else replay_buffer.actions,
                    'log_probs_direction': replay_buffer.log_probs_direction.copy() if isinstance(replay_buffer.log_probs_direction, list) else replay_buffer.log_probs_direction,
                    'log_probs_magnitude': replay_buffer.log_probs_magnitude.copy() if isinstance(replay_buffer.log_probs_magnitude, list) else replay_buffer.log_probs_magnitude,
                    'log_probs_axis': replay_buffer.log_probs_axis.copy() if isinstance(replay_buffer.log_probs_axis, list) else replay_buffer.log_probs_axis,
                    'advantages': replay_buffer.advantages.copy() if isinstance(replay_buffer.advantages, list) else replay_buffer.advantages,
                    'ptr': replay_buffer.ptr,
                    'full': replay_buffer.full,
                    'current_episode_id': replay_buffer.current_episode_id,
                    'total_added': replay_buffer.total_added,
                    'total_sampled': replay_buffer.total_sampled,
                    'success_episodes_added': replay_buffer.success_episodes_added,
                    'failure_episodes_added': replay_buffer.failure_episodes_added
                }                
                _LOG.info(f"[CHECKPOINT] Saved replay buffer ({len(replay_buffer)} transitions)")

        checkpoint_dir = self.config.runtime.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        path = os.path.join(checkpoint_dir, f"ppo_checkpoint_{episode}.pt")
        torch.save(checkpoint, path)
        _LOG.info(f"Saved checkpoint to {path}")

        if self.logger:
            self.logger.save_logs()

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        """Load checkpoint and return starting episode"""
        if path is None:
            path = self.config.runtime.resume_path

        if path and os.path.exists(path):
            # weights_only=False: checkpoints carry full Python state (numpy RNG,
            # optimizer state); trusted in-house file, required since PyTorch 2.6
            # flipped the weights_only default
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer_manager.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'embedding_extractor_state_dict' in checkpoint:
                if hasattr(self, 'embedding_extractor') and self.embedding_extractor is not None:
                    self.embedding_extractor.load_state_dict(checkpoint['embedding_extractor_state_dict'])
                    _LOG.info(f"[CHECKPOINT] Restored embedding_extractor weights")
                else:
                    _LOG.info(f"[CHECKPOINT] WARNING: embedding_extractor not initialized yet!")
            else:
                _LOG.info(f"[CHECKPOINT] WARNING: No embedding_extractor_state_dict in checkpoint")

            # Move optimizer state tensors to the active device
            for state in self.optimizer_manager.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.training_rewards = checkpoint.get('training_rewards', [])
            self.recent_dissociations = checkpoint.get('recent_dissociations', [])

            if self._replay_enabled and 'replay_buffer' in checkpoint:
                if hasattr(self.buffer_manager, 'critic_replay_buffer'):
                    replay_buffer = self.buffer_manager.critic_replay_buffer
                    if replay_buffer is not None:
                        saved_replay = checkpoint['replay_buffer']
                        replay_buffer.states = list(saved_replay['states'])
                        replay_buffer.returns = list(saved_replay['returns'])
                        replay_buffer.episode_ids = list(saved_replay['episode_ids'])
                        replay_buffer.is_success = list(saved_replay['is_success'])
                        # Stage 2 policy data
                        if 'actions' in saved_replay:
                            replay_buffer.actions = list(saved_replay['actions'])
                            replay_buffer.log_probs_direction = list(saved_replay['log_probs_direction'])
                            replay_buffer.log_probs_magnitude = list(saved_replay['log_probs_magnitude'])
                            replay_buffer.log_probs_axis = list(saved_replay['log_probs_axis'])
                            replay_buffer.advantages = list(saved_replay['advantages'])
                        replay_buffer.ptr = saved_replay['ptr']                        
                        replay_buffer.full = saved_replay['full']
                        replay_buffer.current_episode_id = saved_replay['current_episode_id']
                        replay_buffer.total_added = saved_replay['total_added']
                        replay_buffer.total_sampled = saved_replay['total_sampled']
                        replay_buffer.success_episodes_added = saved_replay['success_episodes_added']
                        replay_buffer.failure_episodes_added = saved_replay['failure_episodes_added']
                        _LOG.info(f"[CHECKPOINT] Loaded replay buffer ({len(replay_buffer)} transitions)")
            
            start_episode = checkpoint.get('episode', 0) + 1
            _LOG.info(f"Resumed from episode {start_episode}")
            return start_episode
        
        return 0

    def get_buffer_length(self) -> int:
        """Get current buffer length for indexing"""
        if self.buffer_manager and hasattr(self.buffer_manager.buffer, 'states'):
            return len(self.buffer_manager.buffer.states)
        return 0
