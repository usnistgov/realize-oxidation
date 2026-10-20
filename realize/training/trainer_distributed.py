"""Distributed collective PPO."""
import gc
import torch
import numpy as np
import logging
_LOG = logging.getLogger(__name__)


class PPOTrainerDistributedMixin:
    """Distributed collective-PPO methods for PPOTrainer.

    These methods implement the multi-worker weight-sync / shared-buffer
    aggregation / collective-update path. They are invoked from PPOTrainer.train /
    _run_episode via ``self``: the shared_model_manager / agent /
    local_model_version / device attributes plus the sibling helpers resolve
    through the MRO back to PPOTrainer (a plain orchestrator with no torch
    Parameters of its own), so this mixin cannot perturb the policy fingerprint.

    NOTE: in single-worker runs shared_model_manager is None and the collective
    update path never fires.
    """
    def _prepare_trajectory_for_submission(self, results: dict) -> dict:
        """Package an episode batch (via buffer_manager.get_batch()) for shared
        buffer submission, converting all tensors to CPU numpy for pickling."""

        def to_cpu(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x.copy()  # Ensure we don't share memory
            elif isinstance(x, list):
                return [to_cpu(item) for item in x]
            elif isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            return x

        # Check if buffer has data
        if not self.agent.buffer_manager.has_data():
            return {
                'episode_reward': results['total_reward'],
                'dissociated': results['dissociated'],
                'model_version': self.local_model_version,
                'worker_id': self.shared_model_manager.worker_id,
                'empty': True,
                'num_steps': 0
            }

        # Get processed batch from buffer_manager
        # This applies validation, filtering for valid_flags, etc.
        try:
            batch_data = self.agent.buffer_manager.get_batch(use_success_weighting=False)
        except Exception as e:
            _LOG.info(f"[DISTRIBUTED] WARNING: get_batch failed: {e}")
            return {
                'episode_reward': results['total_reward'],
                'dissociated': results['dissociated'],
                'model_version': self.local_model_version,
                'worker_id': self.shared_model_manager.worker_id,
                'empty': True,
                'error': str(e)
            }

        if batch_data is None:
            return {
                'episode_reward': results['total_reward'],
                'dissociated': results['dissociated'],
                'model_version': self.local_model_version,
                'worker_id': self.shared_model_manager.worker_id,
                'empty': True
            }

        # Build trajectory data dict with ALL fields needed for PPO
        trajectory_data = {
            # Episode metadata
            'episode_reward': float(results['total_reward']),
            'dissociated': bool(results['dissociated']),
            'model_version': self.local_model_version,
            'worker_id': self.shared_model_manager.worker_id,
            'num_steps': len(batch_data.get('states', [])),
            'empty': False,

            # Core trajectory data (REQUIRED for PPO)
            'states': to_cpu(batch_data.get('states')),
            'actor_states': to_cpu(batch_data.get('actor_states')),
            'o1_embeddings': to_cpu(batch_data.get('o1_embeddings')),
            'o2_embeddings': to_cpu(batch_data.get('o2_embeddings')),
            'geometries': to_cpu(batch_data.get('geometries')),
            'actions': to_cpu(batch_data.get('actions')),  # List of action dicts
            'log_probs': to_cpu(batch_data.get('log_probs')),
            'state_values': to_cpu(batch_data.get('state_values')),
            'rewards': to_cpu(batch_data.get('rewards')),
            'is_terminals': to_cpu(batch_data.get('is_terminals')),

            # Per-head log probs (for separate policy losses)
            'log_probs_direction': to_cpu(batch_data.get('log_probs_direction')),
            'log_probs_magnitude': to_cpu(batch_data.get('log_probs_magnitude')),
            'log_probs_axis': to_cpu(batch_data.get('log_probs_axis')),

            # Reward components (for per-head advantage computation)
            'ea_rewards': to_cpu(batch_data.get('ea_rewards')),
            'dissociation_rewards': to_cpu(batch_data.get('dissociation_rewards')),
            'axis_rewards': to_cpu(batch_data.get('axis_rewards')),

            # Asymmetric embedding for axis head
            # Handle tensor or list of tensors
            'o2_embeddings_asymmetric': to_cpu(batch_data.get('o2_embeddings_asymmetric')),

            # Episode tracking
            'episode_indices': to_cpu(batch_data.get('episode_indices')),
            'temperatures': to_cpu(batch_data.get('temperatures')),
            'valid_flags': to_cpu(batch_data.get('valid_flags')),
        }

        # Clear local buffer after extraction
        self.agent.buffer_manager.clear()

        _LOG.debug(f"[DISTRIBUTED] Prepared trajectory: {trajectory_data['num_steps']} steps, "
              f"reward={trajectory_data['episode_reward']:.3f}, "
              f"dissociated={trajectory_data['dissociated']}")

        return trajectory_data


    def _manual_ppo_update(self, batch_data: dict, episode: int) -> tuple:
        """Fallback PPO update calling optimizer_manager directly, used when the
        agent lacks update_from_aggregated_batch."""

        # Ensure policy is in training mode
        self.agent.policy.train()

        # Get config values
        gamma = self.agent.gamma
        gae_lambda = getattr(self.config.algorithm, 'gae_lambda', 0.95)

        # PRE-GAE DIAGNOSTIC
        _LOG.debug(f"\n[PRE-GAE DIAGNOSTIC] Episode {episode}")
        rewards_t = batch_data['rewards']
        values_t = batch_data['state_values']
        terminals_t = batch_data['is_terminals']

        if torch.is_tensor(rewards_t):
            r_np = rewards_t.cpu().numpy()
        else:
            r_np = np.array(rewards_t)
        if torch.is_tensor(values_t):
            v_np = values_t.cpu().numpy()
        else:
            v_np = np.array(values_t)

        _LOG.debug(f"  Batch size: {len(r_np)}")
        _LOG.debug(f"  Rewards: mean={np.mean(r_np):.4f}, std={np.std(r_np):.4f}")
        _LOG.debug(f"  Values:  mean={np.mean(v_np):.4f}, std={np.std(v_np):.4f}")

        # BUFFER CONSISTENCY CHECK
        batch_size = batch_data['actor_states'].shape[0]

        # Check episode IDs if available
        if 'episode_ids' in batch_data:
            unique_episodes = torch.unique(batch_data['episode_ids']).tolist()
            _LOG.debug(f"[BUFFER CHECK] Batch contains {batch_size} transitions from {len(unique_episodes)} episodes: {unique_episodes}")

        # Verify rewards match expected structure
        rewards = batch_data.get('rewards')
        if rewards is not None:
            _LOG.debug(f"[BUFFER CHECK] Rewards: sum={rewards.sum().item():.4f}, mean={rewards.mean().item():.4f}, "
                  f"min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")

        # Check terminal/truncation flags
        is_terminals = batch_data.get('is_terminals')
        is_truncated = batch_data.get('is_truncated')
        if is_terminals is not None:
            n_terminals = is_terminals.sum().item() if is_terminals.dtype == torch.bool else (is_terminals > 0.5).sum().item()
            _LOG.debug(f"[BUFFER CHECK] Terminals: {int(n_terminals)}/{batch_size}")
        if is_truncated is not None:
            n_truncated = is_truncated.sum().item() if is_truncated.dtype == torch.bool else (is_truncated > 0.5).sum().item()
            _LOG.debug(f"[BUFFER CHECK] Truncations: {int(n_truncated)}/{batch_size}")

        # Compute advantages
        # Set current episode for MC warmup check
        self.agent.optimizer_manager._current_episode = episode
        returns, advantages = self.agent.optimizer_manager.compute_advantages(
            batch_data['rewards'],
            batch_data['state_values'],
            batch_data['is_terminals'],
            gamma,
            gae_lambda=gae_lambda,
            is_truncated=batch_data.get('is_truncated', None),
            next_values=batch_data.get('next_values', None),
            episode=episode
        )

        # POST-GAE DIAGNOSTIC
        if torch.is_tensor(returns):
            ret_np = returns.cpu().numpy()
        else:
            ret_np = np.array(returns)
        if torch.is_tensor(advantages):
            adv_np = advantages.cpu().numpy()
        else:
            adv_np = np.array(advantages)

        _LOG.debug(f"\n[POST-GAE DIAGNOSTIC] Episode {episode}")
        _LOG.debug(f"  Returns: mean={np.mean(ret_np):.4f}, std={np.std(ret_np):.4f}")
        _LOG.debug(f"  Advantages: mean={np.mean(adv_np):.4f}, std={np.std(adv_np):.4f}")

        # Scale alignment check
        if np.std(ret_np) > 0 and np.std(v_np) > 0:
            scale_ratio = np.std(ret_np) / np.std(v_np)
            _LOG.debug(f"  returns_std / values_std = {scale_ratio:.4f}")
            if scale_ratio > 5.0 or scale_ratio < 0.2:
                _LOG.info(f"  ALERT SCALE MISMATCH: returns and values on different scales!")

        # Advantage collapse check
        if np.std(adv_np) < 0.1:
            _LOG.info(f"  ALERT ADVANTAGE COLLAPSE: std={np.std(adv_np):.6f} - weak learning signal!")

        # Compute per-head advantages if supported
        if hasattr(self.agent.optimizer_manager, 'compute_per_head_advantages'):
            per_head_advantages = self.agent.optimizer_manager.compute_per_head_advantages(
                batch_data, gamma, gae_lambda
            )
            batch_data['advantages_direction'] = per_head_advantages.get('direction', advantages)
            batch_data['advantages_magnitude'] = per_head_advantages.get('magnitude', advantages)
            batch_data['advantages_axis'] = per_head_advantages.get('axis', advantages)

        # Get entropy coefficient from scheduler
        entropy_coeff = self.agent.scheduler.get_entropy_coeff(episode)

        # Backbone fine-tuning is disabled in distributed mode because
        # state_file_paths cannot be serialized across workers.
        losses, metrics = self.agent.optimizer_manager.optimize(
            batch_data=batch_data,
            returns=returns,
            advantages=advantages,
            entropy_coeff=entropy_coeff,
            episode_idx=episode,
            quick_update=False,
            policy=self.agent.policy,
            policy_old=self.agent.policy_old,
            # Backbone fine-tuning disabled in distributed mode.
            actor_embedding_extractor=None,
            critic_embedding_extractor=None,
            state_file_paths=None,
            actor_state_normalizer=None,
            critic_state_normalizer=None
        )

        # Sync policy_old
        self.agent.sync_policy_old()

        return losses, metrics


    def _handle_distributed_update(self, episode: int, results: dict):
        """Submit this episode's trajectory to the shared buffer; when it is full,
        this worker runs the collective PPO update and commits new weights."""

        trajectory_data = self._prepare_trajectory_for_submission(results)

        should_update, batch_data_list, buffer_size = \
            self.shared_model_manager.submit_episode(trajectory_data)

        status = self.shared_model_manager.get_buffer_status()

        if should_update:
            # This worker performs the collective PPO update.
            gc.collect()
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9

            try:
                new_policy_dict, new_optimizer_dict, training_info = \
                    self._perform_collective_ppo_update(batch_data_list, episode)

                new_version = self.shared_model_manager.finish_update_and_clear_buffer(
                    new_policy_dict,
                    new_optimizer_dict,
                    training_info
                )

                self.local_model_version = new_version

            except Exception as e:
                _LOG.info(f"[DISTRIBUTED] ERROR during collective update: {e}")
                import traceback
                traceback.print_exc()
                raise



    def _sync_weights_if_needed(self):
        """Reload weights from other workers if updated. Call before each episode."""
        if self.shared_model_manager is None:
            return

        was_reloaded, new_policy, new_optimizer, new_version = \
            self.shared_model_manager.check_and_reload_if_needed(str(self.device))

        if was_reloaded:
            self.agent.policy.load_state_dict(new_policy)

            # Keep policy_old in sync for consistent PPO ratios.
            if hasattr(self.agent, 'policy_old'):
                self.agent.policy_old.load_state_dict(new_policy)

            if new_optimizer is not None:
                if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, 'optimizer'):
                    self.agent.optimizer_manager.optimizer.load_state_dict(new_optimizer)
                    # Move optimizer state to device.
                    for state in self.agent.optimizer_manager.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                elif hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(new_optimizer)

            old_version = self.local_model_version
            self.local_model_version = new_version


    def _initialize_or_sync_shared_model(self):
        """Ensure all workers start with identical weights: the first worker saves
        initial weights as master, the rest load them."""
        if self.shared_model_manager is None:
            return

        self.logger.log_info("=" * 60)
        self.logger.log_info("DISTRIBUTED COLLECTIVE PPO: Synchronizing weights...")
        self.logger.log_info("=" * 60)

        optimizer_state = None
        if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, 'optimizer'):
            optimizer_state = self.agent.optimizer_manager.optimizer.state_dict()
        elif hasattr(self.agent, 'optimizer'):
            optimizer_state = self.agent.optimizer.state_dict()

        version, loaded_policy, loaded_optimizer = self.shared_model_manager.initialize_or_sync(
            policy_state_dict=self.agent.policy.state_dict(),
            optimizer_state_dict=optimizer_state,
            device=str(self.device)
        )

        if loaded_policy is not None:
            self.agent.policy.load_state_dict(loaded_policy)
            if hasattr(self.agent, 'policy_old'):
                self.agent.policy_old.load_state_dict(loaded_policy)

            self.logger.log_info(f"  Loaded policy weights from shared model v{version}")

            if loaded_optimizer is not None:
                if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, 'optimizer'):
                    self.agent.optimizer_manager.optimizer.load_state_dict(loaded_optimizer)
                    # Move optimizer state to device
                    for state in self.agent.optimizer_manager.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                elif hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(loaded_optimizer)
                self.logger.log_info(f"  Loaded optimizer state from shared model")

            self.local_model_version = version
            self.logger.log_info(f"  JOINED existing training at version {version}")
        else:
            # First worker: keep our initialized weights.
            self.local_model_version = 0
            self.logger.log_info(f"  INITIALIZED as MASTER worker (version 0)")

        self.logger.log_info("=" * 60)

    def _perform_collective_ppo_update(self, batch_data_list: list, episode: int) -> tuple:
        """Run a PPO update over trajectories aggregated from all workers.

        Returns (new_policy_state_dict, new_optimizer_state_dict, training_info).
        """
        valid_trajectories = [t for t in batch_data_list if not t.get('empty', True)]

        if not valid_trajectories:
            raise RuntimeError(
                f"No valid trajectories in batch_data_list. "
                f"All {len(batch_data_list)} submissions were empty."
            )

        _LOG.debug(f"[COLLECTIVE PPO] Aggregating {len(valid_trajectories)} trajectories "
              f"from {len(batch_data_list)} submissions")

        # Define fields to aggregate
        array_fields = [
            'states', 'actor_states', 'o1_embeddings', 'o2_embeddings', 'geometries',
            'log_probs', 'state_values', 'rewards', 'is_terminals',
            'log_probs_direction', 'log_probs_magnitude', 'log_probs_axis',
            'ea_rewards', 'dissociation_rewards', 'axis_rewards',
            'episode_indices', 'temperatures', 'valid_flags'
        ]

        list_fields = ['actions', 'o2_embeddings_asymmetric']

        # Initialize aggregated data
        aggregated = {field: [] for field in array_fields + list_fields}

        # Aggregate trajectories
        total_steps = 0
        for traj in valid_trajectories:
            num_steps = traj.get('num_steps', 0)
            if num_steps == 0:
                continue

            total_steps += num_steps

            # Aggregate array fields
            for field in array_fields:
                data = traj.get(field)
                if data is not None:
                    if isinstance(data, np.ndarray):
                        aggregated[field].extend(data.tolist())
                    elif isinstance(data, list):
                        aggregated[field].extend(data)

            # Aggregate list fields (actions are list of dicts)
            for field in list_fields:
                data = traj.get(field)
                if data is not None:
                    if isinstance(data, list):
                        aggregated[field].extend(data)
                    elif isinstance(data, np.ndarray):
                        # Handle 2D arrays (e.g., o2_embeddings_asymmetric [n_steps, dim])
                        if data.ndim == 2:
                            aggregated[field].extend([data[i] for i in range(len(data))])
                        else:
                            aggregated[field].append(data)

        _LOG.debug(f"[COLLECTIVE PPO] Total steps: {total_steps}")

        if total_steps == 0:
            raise RuntimeError("Aggregated batch has 0 steps")

        # Convert to numpy arrays / proper format
        batch_data = {}
        for field in array_fields:
            if aggregated[field]:
                batch_data[field] = np.array(aggregated[field])

        for field in list_fields:
            if aggregated[field]:
                batch_data[field] = aggregated[field]

        # Handle o2_embeddings_asymmetric specially - convert list to tensor
        if aggregated['o2_embeddings_asymmetric']:
            # Filter out None values
            valid_embeddings = [e for e in aggregated['o2_embeddings_asymmetric'] if e is not None]
            if valid_embeddings:
                # Stack into tensor
                if isinstance(valid_embeddings[0], np.ndarray):
                    batch_data['o2_embeddings_asymmetric'] = np.stack(valid_embeddings, axis=0)
                elif isinstance(valid_embeddings[0], list):
                    batch_data['o2_embeddings_asymmetric'] = np.array(valid_embeddings)
                else:
                    batch_data['o2_embeddings_asymmetric'] = valid_embeddings
            else:
                batch_data['o2_embeddings_asymmetric'] = None
        else:
            batch_data['o2_embeddings_asymmetric'] = None

        # Validate required fields exist
        required_fields = ['states', 'log_probs', 'state_values', 'rewards', 'is_terminals']
        for field in required_fields:
            if field not in batch_data or batch_data[field] is None:
                raise RuntimeError(f"Missing required field '{field}' in aggregated batch")

        # Move data to device
        device = self.device

        def to_tensor(x, dtype=torch.float32):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.tensor(x, dtype=dtype, device=device)

        # Convert core fields to tensors
        batch_data['states'] = to_tensor(batch_data['states'])
        batch_data['actor_states'] = to_tensor(batch_data.get('actor_states'))
        # Convert critic raw inputs to tensors
        for _key in ('o1_embeddings', 'o2_embeddings', 'geometries'):
            _raw = batch_data.get(_key)
            if _raw is not None:
                batch_data[_key] = to_tensor(_raw)
        batch_data['log_probs'] = to_tensor(batch_data['log_probs'])
        batch_data['state_values'] = to_tensor(batch_data['state_values'])
        batch_data['rewards'] = to_tensor(batch_data['rewards'])
        batch_data['is_terminals'] = to_tensor(batch_data['is_terminals'], dtype=torch.bool)

        # Convert per-head log probs
        batch_data['log_probs_direction'] = to_tensor(batch_data.get('log_probs_direction'))
        batch_data['log_probs_magnitude'] = to_tensor(batch_data.get('log_probs_magnitude'))
        batch_data['log_probs_axis'] = to_tensor(batch_data.get('log_probs_axis'))

        # Convert reward components
        batch_data['ea_rewards'] = to_tensor(batch_data.get('ea_rewards'))
        batch_data['dissociation_rewards'] = to_tensor(batch_data.get('dissociation_rewards'))
        batch_data['axis_rewards'] = to_tensor(batch_data.get('axis_rewards'))

        # Convert o2_embeddings_asymmetric
        batch_data['o2_embeddings_asymmetric'] = to_tensor(batch_data.get('o2_embeddings_asymmetric'))

        # Call PPO update using aggregated batch
        if hasattr(self.agent, 'update_from_aggregated_batch'):
            losses, metrics = self.agent.update_from_aggregated_batch(batch_data, episode)
        else:
            losses, metrics = self._manual_ppo_update(batch_data, episode)

        # Get updated state dicts
        new_policy_dict = self.agent.policy.state_dict()

        if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, 'optimizer'):
            new_optimizer_dict = self.agent.optimizer_manager.optimizer.state_dict()
        elif hasattr(self.agent, 'optimizer'):
            new_optimizer_dict = self.agent.optimizer.state_dict()
        else:
            new_optimizer_dict = {}

        # Prepare training info
        avg_reward = sum(t.get('episode_reward', 0) for t in valid_trajectories) / len(valid_trajectories)
        success_rate = sum(1 for t in valid_trajectories if t.get('dissociated', False)) / len(valid_trajectories)

        training_info = {
            'num_episodes': len(valid_trajectories),
            'total_steps': total_steps,
            'avg_episode_reward': avg_reward,
            'success_rate': success_rate,
            'policy_loss': losses.get('policy', None),
            'value_loss': losses.get('value', None),
            'entropy': metrics.get('entropy', None),
        }

        return new_policy_dict, new_optimizer_dict, training_info
