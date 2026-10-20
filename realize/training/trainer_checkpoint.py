"""Checkpoint save and load."""
import os
import torch
import pickle

from realize.config.paths import get_results_dir, get_temp_dir
import logging
_LOG = logging.getLogger(__name__)


class PPOTrainerCheckpointMixin:
    """Checkpoint save/load + SH-data persistence for PPOTrainer.

    ``_save_checkpoint`` and ``save_sh_data`` fire during training (periodic +
    final), ``_load_checkpoint`` fires on resume. All resolve
    config/agent/episode_runner/logger/device through the MRO back to PPOTrainer
    (a plain orchestrator with no torch Parameters), so this mixin cannot perturb
    the policy fingerprint.
    """
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint (failures raise)."""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.agent.policy.state_dict(),
        }

        # Save DUAL backbones from episode_runner
        if hasattr(self.episode_runner, 'actor_embedding_extractor'):
            checkpoint['actor_backbone_state_dict'] = self.episode_runner.actor_embedding_extractor.state_dict()
            checkpoint['actor_backbone_config'] = {
                'train_backbone': self.episode_runner.actor_embedding_extractor.train_backbone,
                'unfreeze_last_n_layers': self.episode_runner.actor_embedding_extractor.unfreeze_last_n_layers
            }
            _LOG.info(f"[Trainer] Saved actor backbone state_dict")

            if hasattr(self.episode_runner, 'critic_embedding_extractor'):
                checkpoint['critic_backbone_state_dict'] = self.episode_runner.critic_embedding_extractor.state_dict()
                _LOG.info(f"[Trainer] Saved critic backbone state_dict")
        elif hasattr(self.episode_runner, 'embedding_extractor'):
            checkpoint['backbone_state_dict'] = self.episode_runner.embedding_extractor.state_dict()
            checkpoint['backbone_config'] = {
                'train_backbone': self.episode_runner.embedding_extractor.train_backbone,
                'unfreeze_last_n_layers': self.episode_runner.embedding_extractor.unfreeze_last_n_layers
            }
            _LOG.info(f"[Trainer] Saved single backbone (single-backbone format)")

        if not hasattr(self.agent, 'training_rewards'):
            raise AttributeError("Agent must have 'training_rewards' attribute")
        checkpoint['training_rewards'] = self.agent.training_rewards

        # Save adaptive reward scaler state
        if hasattr(self.episode_runner, 'reward_scaler'):
            checkpoint['reward_scaler_state'] = self.episode_runner.reward_scaler.state_dict()

        # Trainer's in-memory cycle bookkeeping (not in reward_scaler_state).
        checkpoint['cycle_state_full'] = {
            'current_cycle':       getattr(self, '_current_cycle', 0),
            'cycle_ea_effs':       list(getattr(self, '_cycle_ea_effs', []) or []),
            'cycle_avg_ea':        list(getattr(self, '_cycle_avg_ea',  []) or []),
            'ea_sigmoid_midpoint': float(getattr(self.config, 'ea_sigmoid_midpoint', 1.2)),
            'cumulative_dissociated_o_count': int(getattr(
                getattr(self.episode_runner, 'env', None),
                'cumulative_dissociated_o_count', 0) or 0),
        }

        # Full RNG state: required for bit-consistent resume of PPO sampling,
        # Langevin noise, vMF draws, and MD thermostat seeding.
        import random as _rnd_mod
        import numpy as _np_mod
        checkpoint['rng_state'] = {
            'torch':      torch.get_rng_state(),
            'torch_cuda': (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            'numpy':      _np_mod.random.get_state(),
            'python':     _rnd_mod.getstate(),
        }

        # Scheduler state (annealed temperatures), serialized by attribute name.
        _sched = (getattr(self.agent, 'scheduler', None)
                  or getattr(self.agent, 'temperature_scheduler', None))
        if _sched is not None:
            _sched_attrs = {}
            for _attr in ('current_temperatures', 'current_temperature',
                          'current_step', 'step_count', 'steps_since_update'):
                if hasattr(_sched, _attr):
                    _v = getattr(_sched, _attr)
                    _sched_attrs[_attr] = (list(_v)
                                            if isinstance(_v, (list, tuple))
                                            else _v)
            if _sched_attrs:
                checkpoint['scheduler_state_attrs'] = _sched_attrs

        if hasattr(self.agent, 'optimizer_manager'):
            if not hasattr(self.agent.optimizer_manager, 'optimizer'):
                raise AttributeError(
                    "optimizer_manager must have 'optimizer' attribute"
                )
            checkpoint['optimizer_state_dict'] = self.agent.optimizer_manager.optimizer.state_dict()
        elif hasattr(self.agent, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.agent.optimizer.state_dict()
        else:
            raise RuntimeError(
                "Agent must have either 'optimizer' or 'optimizer_manager.optimizer'. "
                "Cannot save checkpoint without optimizer state."
            )

        checkpoint_dir = os.path.join(get_results_dir(), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        if final:
            save_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
        else:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')

        # Save state normalizer statistics for resume consistency
        actor_norm, critic_norm = self.episode_runner.get_normalizers()
        if actor_norm is not None and actor_norm.running_stats is not None:
            checkpoint['normalizer_state'] = {
                'mean': actor_norm.running_stats.mean.copy(),
                'var': actor_norm.running_stats.var.copy(),
                'count': float(actor_norm.running_stats.count),
                'frozen': actor_norm._frozen,
            }
        if critic_norm is not None and critic_norm.running_stats is not None:
            checkpoint['critic_normalizer_state'] = {
                'mean': critic_norm.running_stats.mean.copy(),
                'var': critic_norm.running_stats.var.copy(),
                'count': float(critic_norm.running_stats.count),
                'frozen': critic_norm._frozen,
            }

        torch.save(checkpoint, save_path)
        self.logger.log_info(f"Saved checkpoint: {save_path}")

        # Save SH data for visualization (periodic, not just at end)
        self.save_sh_data()

    def _load_checkpoint(self) -> int:
        """Load training checkpoint (failures raise)."""
        checkpoint_path = None
        if hasattr(self.config, 'runtime') and hasattr(self.config.runtime, 'resume_path'):
            checkpoint_path = self.config.runtime.resume_path
        else:
            checkpoint_dir = os.path.join(get_results_dir(), 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    checkpoints.sort()
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])

        if checkpoint_path is None:
            raise ValueError(
                "load_checkpoint=True but no checkpoint found. "
                "Set config.runtime.resume_path or ensure checkpoints exist in results/checkpoints/"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )

        # weights_only=False: checkpoints carry full Python state (numpy RNG,
        # optimizer); required since PyTorch 2.6.
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        if 'policy_state_dict' not in checkpoint:
            raise RuntimeError(f"Checkpoint missing 'policy_state_dict': {checkpoint_path}")
        if 'episode' not in checkpoint:
            raise RuntimeError(f"Checkpoint missing 'episode': {checkpoint_path}")

        saved_policy = checkpoint['policy_state_dict']
        self.agent.policy.load_state_dict(saved_policy)

        # Load adaptive reward scaler if present
        if 'reward_scaler_state' in checkpoint and hasattr(self.episode_runner, 'reward_scaler'):
            self.episode_runner.reward_scaler.load_state_dict(checkpoint['reward_scaler_state'])
            _LOG.info(f"[CHECKPOINT] Loaded reward scaler: ea_weight={self.episode_runner.reward_scaler.ea_weight:.2f}")

        if 'actor_backbone_state_dict' in checkpoint:
            _LOG.info(f"[CHECKPOINT] Loading dual backbone state dicts...")

            if hasattr(self.episode_runner, 'actor_embedding_extractor'):
                try:
                    self.episode_runner.actor_embedding_extractor.load_state_dict(
                        checkpoint['actor_backbone_state_dict']
                    )
                    _LOG.info(f"[CHECKPOINT] OK Actor backbone loaded")
                except Exception as e:
                    _LOG.info(f"[CHECKPOINT] FAIL Failed to load actor backbone: {e}")

            if 'critic_backbone_state_dict' in checkpoint:
                if hasattr(self.episode_runner, 'critic_embedding_extractor'):
                    try:
                        self.episode_runner.critic_embedding_extractor.load_state_dict(
                            checkpoint['critic_backbone_state_dict']
                        )
                        _LOG.info(f"[CHECKPOINT] OK Critic backbone loaded")
                    except Exception as e:
                        _LOG.info(f"[CHECKPOINT] FAIL Failed to load critic backbone: {e}")

        elif 'backbone_state_dict' in checkpoint:
            _LOG.info(f"[CHECKPOINT] Loading single backbone...")
            single_state = checkpoint['backbone_state_dict']

            if hasattr(self.episode_runner, 'actor_embedding_extractor'):
                self.episode_runner.actor_embedding_extractor.load_state_dict(single_state)
                _LOG.info(f"[CHECKPOINT] OK Loaded into actor backbone")

            if hasattr(self.episode_runner, 'critic_embedding_extractor'):
                self.episode_runner.critic_embedding_extractor.load_state_dict(single_state)
                _LOG.info(f"[CHECKPOINT] OK Loaded into critic backbone")

        if 'optimizer_state_dict' in checkpoint:
            if hasattr(self.agent, 'optimizer_manager') and hasattr(self.agent.optimizer_manager, 'optimizer'):
                self.agent.optimizer_manager.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif hasattr(self.agent, 'optimizer'):
                self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                raise RuntimeError(
                    "Checkpoint contains optimizer_state_dict but agent has no optimizer to load it into"
                )

        if 'training_rewards' in checkpoint:
            self.agent.training_rewards = checkpoint['training_rewards']

        # Restore state normalizer statistics
        if 'normalizer_state' in checkpoint:
            actor_norm, _ = self.episode_runner.get_normalizers()
            if actor_norm is not None:
                ns = checkpoint['normalizer_state']
                from realize.training.state_normalizer import RunningMeanStd
                if actor_norm.running_stats is None:
                    actor_norm.running_stats = RunningMeanStd(shape=ns['mean'].shape)
                actor_norm.running_stats.mean = ns['mean']
                actor_norm.running_stats.var = ns['var']
                actor_norm.running_stats.count = ns['count']
                actor_norm._initialized = True
                if ns.get('frozen', False):
                    actor_norm._frozen = True
                self.agent.actor_state_normalizer = actor_norm
                _LOG.info(f"[CHECKPOINT] Restored normalizer (count={actor_norm.running_stats.count:.0f})")
        else:
            _LOG.info(f"[CHECKPOINT] WARNING: No normalizer_state in checkpoint. "
                  f"First PPO update may have logprob mismatch.")

        # Restore critic normalizer
        if 'critic_normalizer_state' in checkpoint:
            _, critic_norm = self.episode_runner.get_normalizers()
            if critic_norm is not None:
                ns = checkpoint['critic_normalizer_state']
                from realize.training.state_normalizer import RunningMeanStd
                if critic_norm.running_stats is None:
                    critic_norm.running_stats = RunningMeanStd(shape=ns['mean'].shape)
                critic_norm.running_stats.mean = ns['mean']
                critic_norm.running_stats.var = ns['var']
                critic_norm.running_stats.count = ns['count']
                critic_norm._initialized = True
                if ns.get('frozen', False):
                    critic_norm._frozen = True
                _LOG.info(f"[CHECKPOINT] Restored critic normalizer "
                      f"(count={critic_norm.running_stats.count:.0f}, "
                      f"frozen={critic_norm._frozen})")
        else:
            _LOG.info(f"[CHECKPOINT] No critic_normalizer_state -- critic normalizer starts fresh")

        # Sync policy_old to loaded policy (needed for exact resume)
        if hasattr(self.agent, 'policy_old'):
            self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())
            _LOG.info(f"[CHECKPOINT] Synced policy_old to loaded policy weights")

        start_episode = checkpoint['episode']
        self.logger.log_info(f"Loaded checkpoint from episode {start_episode}")
        return start_episode

    def save_sh_data(self, filename=None):
        """Save collected SH data for visualization.
        In parallel mode, merges worker sphere_params files."""
        if filename is None:
            filename = os.path.join(get_results_dir(), 'sphere_params.pkl')
        # In parallel mode, merge from worker directories
        import glob
        worker_files = sorted(glob.glob(os.path.join(get_temp_dir(), 'run*', 'results', 'sphere_params.pkl')))
        if worker_files:
            merged = {}
            for wf in worker_files:
                try:
                    with open(wf, 'rb') as f:
                        data = pickle.load(f)
                    for ep, steps in data.items():
                        if ep not in merged:
                            merged[ep] = {}
                        merged[ep].update(steps)
                except Exception:
                    pass
            if merged:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    pickle.dump(merged, f)
                _LOG.info(f"Merged {len(worker_files)} worker sphere_params -> {filename} ({len(merged)} episodes)")
                return
        if self.sh_data:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            nested_data = {}
            for entry in self.sh_data:
                episode = entry['episode']
                step = entry['step']
                if episode not in nested_data:
                    nested_data[episode] = {}
                nested_data[episode][step] = entry

            with open(filename, 'wb') as f:
                pickle.dump(nested_data, f)
            _LOG.info(f"Saved {len(self.sh_data)} SH data entries to {filename} (nested format)")

