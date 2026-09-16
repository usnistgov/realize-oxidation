"""PPO optimize() driver.

The single ``optimize`` method (the PPO update entry point) is the bulk of the
optimizer logic, kept here as a mixin so ppo_optimizer.py stays a thin assembly
point alongside its sibling mixins (_setup / _advantage / _loss / _gradflow).
The method overrides nothing and creates no instance state: every
``self.<attr>`` / ``self.<helper>()`` it touches resolves through the MRO back to
PPOOptimizer (a plain class, NOT an nn.Module), so this cannot perturb the policy
parameter fingerprint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ase import io
import os
import gc
import logging
_LOG = logging.getLogger(__name__)


class PPOOptimizeMixin:
    def _diag_ev_collapse(self, batch_data, episode_idx, policy, policy_old):
        self._dlog("PPO", f"{'='*30} EV COLLAPSE DIAGNOSTIC {'='*30}",
                  episode=episode_idx,
                  update_num=self._optimize_complete_count + 1)
        
        # 1) Terminal handling analysis
        is_terms = batch_data['is_terminals']
        if torch.is_tensor(is_terms):
            terminal_count = is_terms.sum().item()
            total_steps = len(is_terms)
        else:
            terminal_count = sum(is_terms)
            total_steps = len(is_terms)
        
        terminal_frac = terminal_count / total_steps
        if terminal_count == 0:
            term_status = "NO_TERMINALS"
        elif terminal_count == total_steps:
            term_status = "ALL_TERMINALS"
        elif terminal_count > total_steps * 0.3:
            term_status = "HIGH_FRACTION"
        else:
            term_status = "OK"
        
        self._dlog("TERMINAL", "Terminal handling analysis",
                  total_steps=total_steps,
                  terminal_count=terminal_count,
                  terminal_frac=f"{terminal_frac:.1%}",
                  status=term_status)
        
        # 2) Reward distribution
        rewards = batch_data['rewards']
        if torch.is_tensor(rewards):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = np.array(rewards)
        
        nonzero_count = np.sum(np.abs(rewards_np) > 1e-6)
        rew_std = np.std(rewards_np)
        rew_status = "COLLAPSED" if rew_std < 0.01 else "OK"
        
        self._dlog("REWARD", "Reward distribution",
                  mean=f"{np.mean(rewards_np):.4f}",
                  std=f"{rew_std:.4f}",
                  min=f"{np.min(rewards_np):.4f}",
                  max=f"{np.max(rewards_np):.4f}",
                  nonzero=f"{nonzero_count}/{len(rewards_np)}",
                  status=rew_status)
        
        # 3) Value prediction analysis
        state_values = batch_data['state_values']
        if torch.is_tensor(state_values):
            values_np = state_values.cpu().numpy()
        else:
            values_np = np.array(state_values)
        
        val_std = np.std(values_np)
        val_status = "COLLAPSED" if val_std < 0.01 else "OK"
        
        self._dlog("VALUE", "Value predictions",
                  mean=f"{np.mean(values_np):.4f}",
                  std=f"{val_std:.4f}",
                  min=f"{np.min(values_np):.4f}",
                  max=f"{np.max(values_np):.4f}",
                  status=val_status)
        
        # 4) Scale alignment check
        reward_std = np.std(rewards_np)
        value_std = np.std(values_np)
        if reward_std > 0 and value_std > 0:
            scale_ratio = reward_std / value_std
            scale_status = "MISMATCH" if (scale_ratio > 10 or scale_ratio < 0.1) else "OK"
            self._dlog("VALUE", "Scale alignment",
                      reward_std=f"{reward_std:.4f}",
                      value_std=f"{value_std:.4f}",
                      scale_ratio=f"{scale_ratio:.4f}",
                      status=scale_status)
        else:
            self._dlog("VALUE", "Scale alignment - cannot compute (zero std)")
        
        # 5) Per-head log prob check
        for head in ['direction', 'magnitude', 'axis']:
            key = f'log_probs_{head}'
            if key in batch_data:
                lp = batch_data[key]
                if torch.is_tensor(lp):
                    lp_np = lp.cpu().numpy()
                else:
                    lp_np = np.array(lp)
                self._dlog("PPO", f"Per-head log_prob ({head})",
                          head=head,
                          mean=f"{np.mean(lp_np):.4f}",
                          std=f"{np.std(lp_np):.4f}",
                          min=f"{np.min(lp_np):.4f}",
                          max=f"{np.max(lp_np):.4f}")
        
        # 6) policy_old sanity check
        if policy_old is not None:
            same_object = policy is policy_old
            p1 = list(policy.parameters())[0]
            p2 = list(policy_old.parameters())[0]
            shared_storage = p1.data_ptr() == p2.data_ptr()
            
            if self._optimize_complete_count > 0:
                param_diff = (p1.data - p2.data).abs().max().item()
                params_identical = param_diff < 1e-10
            else:
                param_diff = None
                params_identical = None
            
            status = "CRITICAL" if (same_object or shared_storage) else "OK"
            self._dlog("PPO", "policy_old sanity check",
                      same_object=same_object,
                      shared_storage=shared_storage,
                      param_diff=f"{param_diff:.6f}" if param_diff is not None else "N/A",
                      params_identical=params_identical,
                      status=status)
        else:
            self._dlog("PPO", "policy_old is None - using buffer log-probs", status="WARN")

    def optimize(self, batch_data, returns, advantages, entropy_coeff,
                 episode_idx, quick_update, policy, policy_old=None,
                 old_values=None,
                 actor_embedding_extractor=None, critic_embedding_extractor=None,
                 embedding_extractor=None, state_file_paths=None,
                 actor_state_normalizer=None, critic_state_normalizer=None,
                 state_normalizer=None):
        """Run a PPO update with per-head (new vs old) ratios.

        policy_old is a frozen pre-update snapshot used to recompute old_logprob
        through the same code path as new_logprob; if None, stored logprobs are used.
        """

        # Run counters for the end-of-training summary
        if not hasattr(self, '_optimize_call_count'):
            self._optimize_call_count = 0
            self._optimize_complete_count = 0
            self._total_optimizer_steps = 0
        self._optimize_call_count += 1

        if not policy.training:
            raise RuntimeError(
                f"Policy in EVAL mode during PPO at episode {episode_idx}!\n"
                f"  policy.training = {policy.training}\n"
                f"  Call policy.train(True) before optimize()."
            )
        _should_log_update = (self._optimize_complete_count < 5 or
                              self._optimize_complete_count % 1 == 0)

        # PPO invariant is temperature homogeneity, not single-episode batches;
        # multi-episode batches are allowed if temperatures match.
        episode_indices = batch_data.get('episode_indices', None)
        if episode_indices is not None:
            if torch.is_tensor(episode_indices):
                ep_min = int(episode_indices.min().item())
                ep_max = int(episode_indices.max().item())
                n_episodes = len(episode_indices.unique())
            else:
                ep_min = int(min(episode_indices))
                ep_max = int(max(episode_indices))
                n_episodes = len(set(episode_indices))
            
            if ep_min != ep_max:
                if _should_log_update:
                    self._dlog("PPO", f"Multi-episode batch: episodes {ep_min}-{ep_max} ({n_episodes} unique)",
                              status="WARN")

                # Smoke mode fails fast on multi-episode batches (buffer isolation broken)
                env_flag = os.environ.get('PPO_SMOKE_MODE', '').strip().lower()
                smoke_mode = getattr(self, '_smoke_mode', False) or env_flag in ('1', 'true', 'yes', 'y')

                self._dlog("PPO", "SMOKE_MODE_CHECK",
                          env_var=os.environ.get('PPO_SMOKE_MODE', '<not set>'),
                          attr_smoke_mode=getattr(self, '_smoke_mode', None),
                          computed_smoke_mode=smoke_mode)
                
                if smoke_mode:
                    raise RuntimeError(
                        f"FAIL-FAST (smoke mode): Multi-episode batch detected!\n"
                        f"  Expected: 1 episode per batch\n"
                        f"  Got: {n_episodes} episodes ({ep_min}-{ep_max})\n"
                        f"  This indicates buffer isolation is broken.\n"
                        f"  Set PPO_SMOKE_MODE=0 to disable this check."
                    )

        # Temperature must exist, be tensor, match batch, and be homogeneous
        temps = batch_data.get('temperatures', None)
        if temps is None:
            raise RuntimeError(
                "batch_data missing 'temperatures'. "
                "Buffer must store per-transition temperature for PPO consistency."
            )
        
        if not torch.is_tensor(temps):
            raise RuntimeError(
                f"temperatures must be a torch.Tensor, got {type(temps)}. "
                f"Ensure buffer stores temperatures as tensors."
            )
        
        # Length must match batch
        batch_len = len(batch_data.get('actions', []))
        if temps.numel() != batch_len:
            raise RuntimeError(
                f"temperatures length {temps.numel()} != batch_len {batch_len}. "
                f"Temperature array is misaligned with actions."
            )
        
        temp_min = temps.min().item()
        temp_max = temps.max().item()
        if (temp_max - temp_min) > 1e-5:
            raise RuntimeError(
                f"mixed temperatures in minibatch: range=[{temp_min:.6f}, {temp_max:.6f}]. "
                f"PPO requires temperature-homogeneous batches. "
                f"Ensure batching groups transitions by episode/temperature."
            )

        has_actor_states = 'actor_states' in batch_data
        has_critic_inputs = 'o1_embeddings' in batch_data
        if has_actor_states and not has_critic_inputs:
            raise RuntimeError(
                "buffer has actor_states but missing o1_embeddings/o2_embeddings/geometries. "
                f"Keys present: {list(batch_data.keys())}"
            )

        # Freeze normalizer stats during PPO epochs so state representation stays
        # consistent. Pre-init all keys so the finally block never KeyErrors.
        normalizer_prev_temp = {'actor': None, 'critic': None, 'main': None}

        for norm_name, norm in [('actor', actor_state_normalizer),
                                 ('critic', critic_state_normalizer),
                                 ('main', state_normalizer),
                                 ('critic_input', getattr(policy, 'critic_input_normalizer', None))]:
            if norm is not None and hasattr(norm, 'set_frozen_temporarily'):
                normalizer_prev_temp[norm_name] = getattr(norm, '_temp_frozen', False)
                norm.set_frozen_temporarily(True)
                if episode_idx % 10 == 0:
                    self._dlog("NORMALIZER", f"{norm_name}: Temporarily frozen for PPO (was_temp={normalizer_prev_temp[norm_name]})")

        try:
            if quick_update:
                k_epochs = 1
            else:
                if hasattr(self.algo_config, 'k_epochs'):
                    k_epochs = int(self.algo_config.k_epochs)
                elif hasattr(self.algo_config, 'n_epochs'):
                    k_epochs = int(self.algo_config.n_epochs)
                else:
                    raise ValueError(f"Missing k_epochs (or n_epochs) in config: {self.algo_config}")
                if k_epochs <= 0:
                    raise ValueError(f"k_epochs must be > 0, got {k_epochs}")
        


            # Dual backbone mode: actor states + critic raw inputs.
            if 'actor_states' in batch_data and 'o1_embeddings' in batch_data:
                actor_states_all = batch_data['actor_states'].to(self.device)
                o1_embeddings_all = batch_data['o1_embeddings'].to(self.device)
                o2_embeddings_all = batch_data['o2_embeddings'].to(self.device)
                geometries_all = batch_data['geometries'].to(self.device)
                env_multipoles_all = batch_data['env_multipoles'].to(self.device) if 'env_multipoles' in batch_data else None                
                dual_mode = True
                if _should_log_update:
                    self._dlog("BACKBONE", "DUAL BACKBONE MODE ACTIVE (CriticEmbedding)",
                              update_num=self._optimize_complete_count + 1,
                              actor_shape=str(actor_states_all.shape),
                              o1_shape=str(o1_embeddings_all.shape),
                              o2_shape=str(o2_embeddings_all.shape),
                              geom_shape=str(geometries_all.shape))
            else:
                # Single backbone mode: no separate critic raw inputs.
                states_all = batch_data['states'].to(self.device)
                actor_states_all = states_all
                o1_embeddings_all = None
                o2_embeddings_all = None
                geometries_all = None
                env_multipoles_all = None                
                dual_mode = False
                if _should_log_update:
                    self._dlog("BACKBONE", "SINGLE BACKBONE MODE",
                              state_shape=str(actor_states_all.shape))


            if _should_log_update:
                self._diag_ev_collapse(batch_data, episode_idx, policy, policy_old)

            if 'actions' not in batch_data:
                raise RuntimeError("batch_data missing required key 'actions'")
            old_actions = batch_data['actions']
            batch_len = len(old_actions)
            
            # Rollout V(s), pinned once and reused everywhere.
            old_values_rollout = None
            if 'state_values' in batch_data:
                old_values_raw = batch_data['state_values']
                if torch.is_tensor(old_values_raw):
                    old_values_rollout = old_values_raw.detach().to(self.device)
                else:
                    old_values_rollout = torch.tensor(old_values_raw, device=self.device, dtype=torch.float32)
                
                # Shape normalization: accept [B] or [B,1], reject others
                if old_values_rollout.ndim == 2 and old_values_rollout.shape[1] == 1:
                    old_values_rollout = old_values_rollout.squeeze(1)
                if old_values_rollout.ndim != 1:
                    raise RuntimeError(f"old_values_rollout must be 1D [B], got shape={tuple(old_values_rollout.shape)}")
                
                if old_values_rollout.shape[0] != batch_len:
                    raise RuntimeError(
                        f"old_values_rollout length mismatch: {old_values_rollout.shape[0]} vs batch_len={batch_len}"
                    )
                
                if not torch.isfinite(old_values_rollout).all():
                    raise RuntimeError("old_values_rollout contains NaN/Inf")
                
                # Store hash for consistency check across epochs
                self._old_values_rollout_hash = old_values_rollout.detach().cpu().numpy().tobytes()
                
            elif self.clip_value_loss:
                raise RuntimeError(
                    "value clipping is enabled but batch_data has no 'state_values' "
                    "(rollout-time V(s) required for clipping)."
                )

            # PPO requires old_logprob and new_logprob on IDENTICAL states, so the
            # frozen-backbone path uses the exact state tensors stored during rollout.
            # Re-extracting would drift (new tensor object, changed normalizer stats,
            # or any backbone parameter change).

            if _should_log_update:
                self._dlog("PPO", "PPO STATE SOURCE",
                          episode=episode_idx,
                          update_num=self._optimize_complete_count + 1)

            # Backbone training re-extracts actor embeddings so gradients flow back
            # through the actor backbone. PPO ratio = exp(new_logprob - old_logprob):
            # old_logprob came from rollout (old backbone, in buffer), new_logprob from
            # the current backbone (fresh extraction), so the ratio captures backbone
            # change too. The critic keeps stored states (frozen backbone, no gradients).

            backbone_training_enabled = (
                actor_embedding_extractor is not None and
                hasattr(actor_embedding_extractor, 'train_backbone') and
                actor_embedding_extractor.train_backbone and
                state_file_paths is not None and
                len(state_file_paths) == batch_len
            )

            if backbone_training_enabled:
                if _should_log_update:
                    self._dlog("BACKBONE", "Backbone training enabled - will re-extract embeddings",
                              num_paths=len(state_file_paths),
                              batch_len=batch_len)
            else:
                if _should_log_update:
                    self._dlog("BACKBONE", "Using stored buffer states (backbone frozen or unavailable)")

            losses_tracker = []

            for ke in range(k_epochs):
                _should_log = (ke == 0 and _should_log_update)

                # Gradient accumulation: run each mini-batch through a full
                # forward/backward so all computation graphs aren't held at once
                # (the backbone's nested autograd.grad() calls would OOM otherwise).
                if backbone_training_enabled:
                    # One SevenNet graph alive at a time. Gradient accumulation
                    # outside the mb loop (zero_grad once per epoch, backward()
                    # per mb) makes the final gradient identical to the batched
                    # version, so the math is preserved. Raise this only after
                    # confirming GPU headroom -- each unit adds one retained
                    # SevenNet activation graph.
                    EMBED_MINI_BATCH_SIZE = 32
                    num_paths = len(state_file_paths)
                    num_mini_batches = (num_paths + EMBED_MINI_BATCH_SIZE - 1) // EMBED_MINI_BATCH_SIZE

                    if _should_log:
                        mem_info = {}
                        if torch.cuda.is_available():
                            mem_info['mem_alloc_gb'] = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
                            mem_info['mem_reserved_gb'] = f"{torch.cuda.memory_reserved() / 1e9:.2f}"
                        self._dlog("PPO", f"Epoch {ke} GRADIENT ACCUMULATION MODE",
                                  num_paths=num_paths,
                                  mini_batches=num_mini_batches,
                                  batch_size=EMBED_MINI_BATCH_SIZE,
                                  **mem_info)

                    # Zero gradients once at start of epoch
                    self.optimizer.zero_grad()

                    # Accumulators for metrics
                    accumulated_losses = {}
                    accumulated_entropy = 0.0
                    accumulated_state_values = []
                    accumulated_returns = []
                    accumulated_stored_values = []
                    self._cf_dir_sum = 0.0
                    self._cf_mag_sum = 0.0
                    self._cf_axis_sum = 0.0
                    self._cf_n = 0
                    total_samples = 0

                    # For pre-update consistency check (first mini-batch only)
                    first_mb_old_dir_check = None
                    first_mb_new_dir_check = None
                    first_mb_old_axis_check = None
                    first_mb_new_axis_check = None

                    for mb_idx in range(num_mini_batches):
                        mb_start = mb_idx * EMBED_MINI_BATCH_SIZE
                        mb_end = min(mb_start + EMBED_MINI_BATCH_SIZE, num_paths)
                        mb_size = mb_end - mb_start
                        mb_paths = state_file_paths[mb_start:mb_end]

                        if _should_log and mb_idx == 0:
                            self._dlog("PPO", f"Processing mini-batch {mb_idx + 1}/{num_mini_batches}",
                                      samples_start=mb_start,
                                      samples_end=mb_end)

                        # Extract mini-batch embeddings
                        mb_symmetric = []
                        mb_asymmetric = []

                        for path in mb_paths:
                            atoms = io.read(path, format="lammps-data", atom_style="charge")
                            embeddings = actor_embedding_extractor.get_equivariant_embedding(
                                atoms=atoms,
                                return_symmetric_only=False
                            )
                            mb_symmetric.append(embeddings['symmetric'])
                            mb_asymmetric.append(embeddings['asymmetric'])

                        mb_actor_states_raw = torch.cat(mb_symmetric, dim=0).to(self.device)
                        mb_o2_asymmetric_reextracted = torch.cat(mb_asymmetric, dim=0).to(self.device)
                        del mb_symmetric, mb_asymmetric

                        if _should_log and mb_idx == 0:
                            self._dlog("EMBEDDING", "Extracted mini-batch",
                                      shape=str(mb_actor_states_raw.shape),
                                      requires_grad=mb_actor_states_raw.requires_grad)

                        # Normalize mini-batch
                        if actor_state_normalizer is not None:
                            stats = actor_state_normalizer.get_statistics()
                            if stats is not None:
                                mean = torch.tensor(stats['mean'], dtype=mb_actor_states_raw.dtype, device=self.device)
                                std = torch.tensor(stats['std'], dtype=mb_actor_states_raw.dtype, device=self.device)
                                mb_actor_states = (mb_actor_states_raw - mean) / (std + 1e-8)
                            else:
                                mb_actor_states = mb_actor_states_raw
                                if _should_log and mb_idx == 0:
                                    self._dlog("NORMALIZER", "Normalizer not initialized, using raw embeddings", status="WARN")
                        else:
                            mb_actor_states = mb_actor_states_raw
                        del mb_actor_states_raw

                        # Mini-batch actions (needed for stored o2)
                        mb_old_actions = old_actions[mb_start:mb_end]

                        # Use stored o2_embedding_asymmetric: re-extraction can differ
                        # and cause axis log_prob mismatch.
                        mb_o2_asymmetric_stored = []
                        use_stored_o2 = True
                        for act in mb_old_actions:
                            o2_emb = act.get('o2_embedding_asymmetric')
                            if o2_emb is None:
                                use_stored_o2 = False
                                break
                            if isinstance(o2_emb, np.ndarray):
                                o2_emb = torch.from_numpy(o2_emb)
                            mb_o2_asymmetric_stored.append(o2_emb.to(self.device).float())

                        if use_stored_o2 and len(mb_o2_asymmetric_stored) == len(mb_old_actions):
                            mb_o2_asymmetric = torch.stack(mb_o2_asymmetric_stored)
                            if _should_log and mb_idx == 0 and ke == 0:
                                self._dlog("PPO", "Using STORED o2_embedding_asymmetric for axis consistency", status="OK")
                        else:
                            mb_o2_asymmetric = mb_o2_asymmetric_reextracted
                            if _should_log and mb_idx == 0 and ke == 0:
                                self._dlog("PPO", "Using RE-EXTRACTED o2_embedding_asymmetric (stored not available)", status="WARN")
                        
                        # Compare stored vs re-extracted o2_embedding
                        if ke == 0 and mb_idx == 0 and episode_idx <= 3:
                            _LOG.debug(f"\n[O2_EMB DEBUG] Episode {episode_idx}")
                            _LOG.debug(f"  use_stored_o2: {use_stored_o2}")
                            _LOG.debug(f"  len(mb_o2_asymmetric_stored): {len(mb_o2_asymmetric_stored)}")
                            _LOG.debug(f"  len(mb_old_actions): {len(mb_old_actions)}")
                            if use_stored_o2 and len(mb_o2_asymmetric_stored) > 0:
                                stored_stack = torch.stack(mb_o2_asymmetric_stored)
                                diff = (stored_stack - mb_o2_asymmetric_reextracted).abs().max().item()
                                _LOG.debug(f"  max diff (stored vs reextracted): {diff:.6f}")
                                if diff > 1e-4:
                                    _LOG.info(f"  WARNING: stored and reextracted o2_embedding differ!")
                            _LOG.debug(f"  mb_o2_asymmetric shape: {mb_o2_asymmetric.shape}")

                        # Rest of mini-batch data from buffer
                        mb_o1_embeddings = o1_embeddings_all[mb_start:mb_end] if o1_embeddings_all is not None else None
                        mb_o2_embeddings = o2_embeddings_all[mb_start:mb_end] if o2_embeddings_all is not None else None
                        mb_geometries = geometries_all[mb_start:mb_end] if geometries_all is not None else None
                        mb_env_multipoles = env_multipoles_all[mb_start:mb_end] if env_multipoles_all is not None else None
                        mb_o2_positions_list = []
                        for act in mb_old_actions:
                            o2_pos = act.get('o2_positions')
                            if o2_pos is None:
                                o2_pos = act.get('o2_position')
                            if o2_pos is not None:
                                if isinstance(o2_pos, np.ndarray):
                                    o2_pos = torch.from_numpy(o2_pos)
                                mb_o2_positions_list.append(o2_pos.to(self.device).float())
                        if mb_o2_positions_list and len(mb_o2_positions_list) == len(mb_old_actions):
                            mb_o2_positions = torch.stack(mb_o2_positions_list)
                        else:
                            mb_o2_positions = None
                        mb_returns = returns[mb_start:mb_end].to(self.device).reshape(-1)                        
                        mb_advantages = advantages[mb_start:mb_end].to(self.device).reshape(-1)

                        # Per-head old log probs from buffer
                        mb_old_dir_buffer = batch_data['log_probs_direction'][mb_start:mb_end].to(self.device).reshape(-1)
                        mb_old_mag_buffer = batch_data['log_probs_magnitude'][mb_start:mb_end].to(self.device).reshape(-1)
                        mb_old_axis = batch_data['log_probs_axis'][mb_start:mb_end].to(self.device).reshape(-1)

                        # Per-head advantages
                        mb_adv_dir = batch_data.get('advantages_direction', advantages)[mb_start:mb_end].to(self.device).reshape(-1).detach()
                        mb_adv_mag = batch_data.get('advantages_magnitude', advantages)[mb_start:mb_end].to(self.device).reshape(-1).detach()
                        mb_adv_axis = batch_data.get('advantages_axis', advantages)[mb_start:mb_end].to(self.device).reshape(-1).detach()

                        mb_stored_temps = None
                        if 'temperatures' in batch_data and batch_data['temperatures'] is not None:
                            mb_stored_temps = batch_data['temperatures'][mb_start:mb_end]

                        # Old values for value clipping
                        mb_old_values = None
                        if old_values is not None:
                            mb_old_values = old_values[mb_start:mb_end].reshape(-1)

                        # Recompute old log-probs with policy_old on the same fresh
                        # embeddings, so old and new log-probs share inputs.
                        if policy_old is not None:
                            with torch.no_grad():
                                mb_o2_asymmetric_stored_old = []
                                for act in mb_old_actions:
                                    o2_emb = act.get('o2_embedding_asymmetric')
                                    if o2_emb is not None:
                                        if isinstance(o2_emb, np.ndarray):
                                            o2_emb = torch.from_numpy(o2_emb)
                                        mb_o2_asymmetric_stored_old.append(o2_emb.to(self.device).float())
                                
                                if mb_o2_asymmetric_stored_old:
                                    mb_o2_asymmetric_for_old = torch.stack(mb_o2_asymmetric_stored_old)
                                else:
                                    mb_o2_asymmetric_for_old = mb_o2_asymmetric
                                
                                _, _, _, mb_per_head_old = policy_old.evaluate(
                                    mb_actor_states,
                                    episode_idx=episode_idx,
                                    old_actions=mb_old_actions,
                                    state_uid=None,
                                    o2_embeddings_asymmetric=mb_o2_asymmetric_for_old,
                                    o1_embeddings=mb_o1_embeddings,
                                    o2_embeddings=mb_o2_embeddings,
                                    geometries=mb_geometries,                             
                                    stored_temperatures=mb_stored_temps,
                                    o2_positions=mb_o2_positions,
                                    env_multipoles=mb_env_multipoles,
                                )                                        
                            mb_old_dir = mb_per_head_old['direction'].reshape(-1)
                            mb_old_mag = mb_per_head_old['magnitude'].reshape(-1)
                            mb_old_axis = mb_per_head_old['axis'].reshape(-1)  
                        else:
                            mb_old_dir = mb_old_dir_buffer
                            mb_old_mag = mb_old_mag_buffer
                            if _should_log and mb_idx == 0:
                                self._dlog("PPO", "policy_old not available - using buffer log-probs", status="WARN")

                        # Forward pass on mini-batch; stored o2_embedding_asymmetric
                        # keeps axis log_prob consistent with rollout.
                        mb_o2_asymmetric_stored = []
                        for act in mb_old_actions:
                            o2_emb = act.get('o2_embedding_asymmetric')
                            if o2_emb is not None:
                                if isinstance(o2_emb, np.ndarray):
                                    o2_emb = torch.from_numpy(o2_emb)
                                mb_o2_asymmetric_stored.append(o2_emb.to(self.device).float())
                        
                        if mb_o2_asymmetric_stored:
                            mb_o2_asymmetric_for_eval = torch.stack(mb_o2_asymmetric_stored)
                        else:
                            mb_o2_asymmetric_for_eval = mb_o2_asymmetric  # fall back to re-extracted

                        _, mb_state_values, mb_entropy, mb_per_head_new = policy.evaluate(
                            mb_actor_states,
                            episode_idx=episode_idx,
                            old_actions=mb_old_actions,
                            state_uid=None,
                            o2_embeddings_asymmetric=mb_o2_asymmetric_for_eval,
                            o1_embeddings=mb_o1_embeddings,
                            o2_embeddings=mb_o2_embeddings,
                            geometries=mb_geometries,
                            stored_temperatures=mb_stored_temps,
                            o2_positions=mb_o2_positions,
                            env_multipoles=mb_env_multipoles,
                        )

                        mb_new_dir = mb_per_head_new['direction'].reshape(-1)
                        mb_new_mag = mb_per_head_new['magnitude'].reshape(-1)
                        mb_new_axis = mb_per_head_new['axis'].reshape(-1)
                        mb_sv_flat = mb_state_values.reshape(-1)


                        # Compare buffer vs recomputed
                        if ke == 0 and mb_idx == 0:
                            self._dlog("AXIS_DEBUG", "Buffer vs Recomputed comparison",
                                      episode=episode_idx,
                                      buffer_axis_0=f"{mb_old_axis[0].item():.4f}",
                                      buffer_axis_1=f"{mb_old_axis[1].item():.4f}" if len(mb_old_axis) > 1 else "N/A",
                                      recomp_axis_0=f"{mb_new_axis[0].item():.4f}",
                                      recomp_axis_1=f"{mb_new_axis[1].item():.4f}" if len(mb_new_axis) > 1 else "N/A",
                                      diff_0=f"{abs(mb_old_axis[0].item() - mb_new_axis[0].item()):.4f}",
                                      diff_1=f"{abs(mb_old_axis[1].item() - mb_new_axis[1].item()):.4f}" if len(mb_old_axis) > 1 else "N/A")

                        if ke == 0 and mb_idx == 0 and _should_log:                        
                            first_mb_old_dir_check = mb_old_dir.detach().clone()                        
                            first_mb_new_dir_check = mb_new_dir.detach().clone()
                            first_mb_old_axis_check = mb_old_axis.detach().clone()
                            first_mb_new_axis_check = mb_new_axis.detach().clone()

                            log_ratio_dir = first_mb_new_dir_check - first_mb_old_dir_check
                            ratio_dir = torch.exp(log_ratio_dir.clamp(-10, 10))
                            dir_ratio_ok = ratio_dir.min() >= 0.8 and ratio_dir.max() <= 1.25

                            log_ratio_axis = first_mb_new_axis_check - first_mb_old_axis_check
                            ratio_axis = torch.exp(log_ratio_axis.clamp(-10, 10))
                            axis_ratio_ok = ratio_axis.min() >= 0.8 and ratio_axis.max() <= 1.25
                            # Identity test: fail if old/new share storage (not recomputed)
                            if hasattr(mb_old_dir, 'data_ptr') and hasattr(mb_new_dir, 'data_ptr'):
                                same_ptr_dir = mb_old_dir.data_ptr() == mb_new_dir.data_ptr()
                                same_ptr_axis = mb_old_axis.data_ptr() == mb_new_axis.data_ptr()
                                max_diff_dir = float((mb_new_dir - mb_old_dir).abs().max())
                                max_diff_axis = float((mb_new_axis - mb_old_axis).abs().max())
                                if same_ptr_dir or same_ptr_axis:
                                    raise RuntimeError(
                                        f"IDENTITY TEST FAILED: Comparing tensor to itself!\n"
                                        f"  dir same_ptr={same_ptr_dir}, axis same_ptr={same_ptr_axis}\n"
                                        f"  This means evaluate() is not truly recomputing log_prob."
                                    )
                                self._dlog("PPO", "Pre-update consistency (policy_old on fresh embeddings)",
                                          episode=episode_idx,
                                          dir_logratio_mean=f"{log_ratio_dir.mean():.3f}",
                                          dir_logratio_std=f"{log_ratio_dir.std():.3f}",
                                          dir_ratio_range=f"[{ratio_dir.min():.3f}, {ratio_dir.max():.3f}]",
                                          dir_status="OK" if dir_ratio_ok else "MISMATCH",
                                          axis_logratio_mean=f"{log_ratio_axis.mean():.3f}",
                                          axis_logratio_std=f"{log_ratio_axis.std():.3f}",
                                          axis_ratio_range=f"[{ratio_axis.min():.3f}, {ratio_axis.max():.3f}]",
                                          axis_status="OK" if axis_ratio_ok else "MISMATCH")
                                # A ratio far from 1.0 before any optimizer step means
                                # misaligned states/actions, temperature mismatch, or
                                # non-deterministic evaluate().
                                if not dir_ratio_ok or not axis_ratio_ok:
                                    raise RuntimeError(
                                        f"Pre-update logprob mismatch at episode {episode_idx}. "
                                        f"dir_ratio_range=[{ratio_dir.min().item():.4f}, {ratio_dir.max().item():.4f}], "
                                        f"axis_ratio_range=[{ratio_axis.min().item():.4f}, {ratio_axis.max().item():.4f}]. "
                                        f"This means new_logprob != old_logprob before any update. "                            
                                    f"Check: temperature consistency, state/action alignment, evaluate() determinism."
                                )
    
                            # AXIS FROZEN ROOT CAUSE CHECK
                            if axis_ratio_ok and log_ratio_axis.abs().max().item() < 1e-6:
                                self._dlog("AXIS", "Axis ratio is EXACTLY 1.000",
                                          episode=episode_idx,
                                          axis_logratio_max=f"{log_ratio_axis.abs().max().item():.8f}",
                                          diagnosis="axis_params likely not recomputed from current policy")
    
                        # VALUE GRADIENT CHECK (first mini-batch)
                        if _should_log and mb_idx == 0:
                            grad_status = "CRITICAL_DETACHED" if not mb_state_values.requires_grad else "OK"
                            self._dlog("GRADIENT", "Value gradient sanity check",
                                      episode=episode_idx,
                                      requires_grad=mb_state_values.requires_grad,
                                      grad_fn=str(mb_state_values.grad_fn)[:50] if mb_state_values.grad_fn else "None",
                                      status=grad_status)
    
                        # VALIDATE TENSOR SIZES
                        if mb_sv_flat.numel() != mb_size:
                            raise RuntimeError(f"state_values {mb_sv_flat.numel()} != mb_size {mb_size}")
                        if mb_new_dir.numel() != mb_size:
                            raise RuntimeError(f"new_dir {mb_new_dir.numel()} != mb_size {mb_size}")
    
                        # Check for NaNs
                        for name, tensor in [('mb_sv', mb_sv_flat), ('mb_new_dir', mb_new_dir),
                                             ('mb_new_axis', mb_new_axis), ('mb_returns', mb_returns)]:
                            if not torch.isfinite(tensor).all():
                                nan_count = (~torch.isfinite(tensor)).sum().item()
                                raise RuntimeError(f"{name} contains {nan_count} NaN/Inf!")
    
                        # COMPUTE MINI-BATCH LOSSES
                        mb_losses = self._compute_losses(                    
                            per_head_new={'direction': mb_new_dir, 'magnitude': mb_new_mag, 'axis': mb_new_axis},
                            per_head_old={'direction': mb_old_dir, 'magnitude': mb_old_mag, 'axis': mb_old_axis},
                            per_head_adv={'direction': mb_adv_dir, 'magnitude': mb_adv_mag, 'axis': mb_adv_axis},
                            state_values=mb_sv_flat,
                            returns=mb_returns,
                            old_values=mb_old_values,
                            entropy=mb_entropy,
                            entropy_coeff=entropy_coeff,
                            episode_idx=episode_idx,
                            ke=ke,
                            k_epochs=k_epochs,
                            should_log=(_should_log and mb_idx == 0),
                        )
    
                        # Scale by mini-batch fraction so accumulated gradients equal
                        # the full-batch gradient.
                        scale = mb_size / num_paths
                        scaled_loss = mb_losses['total'] * scale
                        scaled_loss.backward()
    
                        # ACCUMULATE METRICS (inside minibatch loop)
                        for k, v in mb_losses.items():
                            if k not in accumulated_losses:
                                accumulated_losses[k] = 0.0
                            accumulated_losses[k] += v.item() * mb_size
                        accumulated_entropy += mb_entropy.mean().item() * mb_size
                        accumulated_state_values.append(mb_sv_flat.detach())
                        accumulated_returns.append(mb_returns.detach())
                        if old_values_rollout is not None:
                            _mb_stored_h4 = old_values_rollout[mb_start:mb_end].detach()
                            accumulated_stored_values.append(_mb_stored_h4)
                        total_samples += mb_size

                        # Accumulate clip fraction (before del)
                        if ke == k_epochs - 1:
                            with torch.no_grad():
                                _dir_r = torch.exp((mb_new_dir - mb_old_dir).clamp(-3.0, 3.0))
                                _mag_r = torch.exp((mb_new_mag - mb_old_mag).clamp(-3.0, 3.0))
                                _axis_r = torch.exp((mb_new_axis - mb_old_axis).clamp(-3.0, 3.0))
                                self._cf_dir_sum += ((_dir_r < 0.7) | (_dir_r > 1.3)).float().sum().item()
                                self._cf_mag_sum += ((_mag_r < 0.7) | (_mag_r > 1.3)).float().sum().item()
                                self._cf_axis_sum += ((_axis_r < 0.7) | (_axis_r > 1.3)).float().sum().item()
                                self._cf_n += mb_size
                                if mb_idx == 0:
                                    _LOG.debug(f"[DEBUG-CF] ke={ke} mb={mb_idx} cf_dir_sum={self._cf_dir_sum:.1f} cf_n={self._cf_n} dir_ratio=[{_dir_r.min().item():.6f},{_dir_r.max().item():.6f}] mag_ratio=[{_mag_r.min().item():.6f},{_mag_r.max().item():.6f}] axis_ratio=[{_axis_r.min().item():.6f},{_axis_r.max().item():.6f}]")

                        # FREE MINI-BATCH MEMORY
                        del mb_actor_states, mb_o2_asymmetric, mb_state_values, mb_entropy
                        del mb_per_head_new, mb_losses, scaled_loss
                        del mb_new_dir, mb_new_mag, mb_new_axis, mb_sv_flat
                        del mb_old_dir, mb_old_mag, mb_old_axis
                        del mb_old_dir_buffer, mb_old_mag_buffer
                        del mb_adv_dir, mb_adv_mag, mb_adv_axis
                        del mb_returns, mb_advantages
                        if mb_old_values is not None:
                            del mb_old_values
                        torch.cuda.empty_cache()

                    # GRAD NORM DIAGNOSTIC (after all minibatches in this epoch)
                    if ke == 0 and _should_log:
                        value_grad_norm = 0.0
                        axis_grad_norm = 0.0
                        value_param_count = 0
                        axis_param_count = 0
    
                        for name, param in policy.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if 'value' in name.lower():
                                    value_grad_norm += grad_norm ** 2
                                    value_param_count += 1
                                if 'axis' in name.lower():
                                    axis_grad_norm += grad_norm ** 2
                                    axis_param_count += 1
    
                        value_grad_norm = value_grad_norm ** 0.5
                        axis_grad_norm = axis_grad_norm ** 0.5
    
                        self._dlog("GRADIENT", "Head grad norms (epoch 0)",
                                  episode=episode_idx,
                                  value_grad_norm=f"{value_grad_norm:.4e}",
                                  value_params=value_param_count,
                                  axis_grad_norm=f"{axis_grad_norm:.4e}",
                                  axis_params=axis_param_count)
    
                        if value_grad_norm < 1e-8:
                            self._log("GRADIENT", "WARNING: value_head has ~zero gradients", status="CRITICAL")
                        if axis_grad_norm < 1e-8:
                            self._log("GRADIENT", "WARNING: axis_net has ~zero gradients", status="CRITICAL")
    
                    # Memory logging (periodic)
                    if _should_log :
                        if torch.cuda.is_available():
                            self._dlog("MEMORY", "Memory after all mini-batches",
                                      alloc_gb=f"{torch.cuda.memory_allocated() / 1e9:.2f}",
                                      reserved_gb=f"{torch.cuda.memory_reserved() / 1e9:.2f}")
    
                    # AVERAGE ACCUMULATED METRICS
                    for k in accumulated_losses:
                        accumulated_losses[k] /= total_samples
                    avg_entropy = accumulated_entropy / total_samples
    
                    all_sv_flat = torch.cat(accumulated_state_values, dim=0)
                    all_ret_flat = torch.cat(accumulated_returns, dim=0)

                    losses = {k: torch.tensor(v, device=self.device) for k, v in accumulated_losses.items()}
                    entropy = torch.tensor(avg_entropy, device=self.device)
    
                    # CRITIC QUALITY METRICS
                    if _should_log:
                        with torch.no_grad():
                            value_error = (all_sv_flat - all_ret_flat).abs().mean().item()
                            value_mse = ((all_sv_flat - all_ret_flat) ** 2).mean().item()
                            returns_std = all_ret_flat.std().item()
                                
                            if returns_std < 1e-3:
                                self._dlog("VALUE", "Critic quality (batch aggregated)",
                                          episode=episode_idx,
                                          mae=f"{value_error:.4f}",
                                          mse=f"{value_mse:.4f}",
                                          ev="SKIPPED",
                                          returns_std=f"{returns_std:.2e}")
                            else:
                                returns_var = returns_std ** 2
                                residual_var = (all_ret_flat - all_sv_flat).var().item()
                                explained_var = 1 - residual_var / returns_var
                                    
                                # EV from stored rollout values vs current critic
                                if len(accumulated_stored_values) > 0:
                                    all_stored_h4 = torch.cat(accumulated_stored_values, dim=0)
                                    if all_stored_h4.numel() == all_ret_flat.numel():
                                        stored_resid = (all_ret_flat - all_stored_h4).var().item()
                                        ev_stored = 1 - stored_resid / returns_var
                                        ev_improv = explained_var - ev_stored
                                        self._dlog("VALUE", "EV comparison (H4)",
                                                  episode=episode_idx,
                                                  ev_current=f"{explained_var:.4f}",
                                                  ev_stored=f"{ev_stored:.4f}",
                                                  improvement=f"{ev_improv:.4f}",
                                                  status="LEARNING" if ev_improv > 0.01 else "FLAT")
                                    
                                self._dlog("VALUE", "Critic quality (batch aggregated)",
                                          episode=episode_idx,
                                          mae=f"{value_error:.4f}",
                                          mse=f"{value_mse:.4f}",
                                          ev=f"{explained_var:.4f}")
    
                    del accumulated_state_values, accumulated_returns, accumulated_stored_values, all_sv_flat, all_ret_flat

                    if _should_log:
                        self._dlog("GRADIENT", f"Gradient check after backward (update #{self._optimize_complete_count + 1})",
                                  episode=episode_idx)

                        # Value head gradients
                        for name in ['critic_embedder', 'value_mlp']:
                            if hasattr(policy, name):
                                layer = getattr(policy, name)
                                if hasattr(layer, 'weight') and layer.weight.grad is not None:                                
                                    gn = layer.weight.grad.norm().item()
                                    status = "OK" if gn > 1e-8 else "ZERO"
                                    self._dlog("GRADIENT", f"Value head {name}",
                                              grad_norm=f"{gn:.3e}",
                                              status=status)
                                else:
                                    self._dlog("GRADIENT", f"Value head {name}", status="NO_GRADIENT")
    
                        # Channel networks
                        if hasattr(policy, 'sh_head') and hasattr(policy.sh_head, 'channel_linears'):
                            channel_nets = policy.sh_head.channel_linears
                            if hasattr(channel_nets, 'module'):
                                channel_nets = channel_nets.module
                            if hasattr(channel_nets, '__len__'):
                                for ch_idx in range(min(len(channel_nets), 4)):
                                    ch_net = channel_nets[ch_idx]
                                    ch_grad_norm = sum(p.grad.norm().item() ** 2 for p in ch_net.parameters() if p.grad is not None) ** 0.5
                                    ch_has_grad = sum(1 for p in ch_net.parameters() if p.grad is not None and p.grad.norm().item() > 0)
                                    ch_params = sum(1 for _ in ch_net.parameters())
                                    status = "OK" if ch_has_grad > 0 else "DEAD"
                                    self._dlog("CHANNEL", f"Channel[{ch_idx}] gradients",
                                              grad_norm=f"{ch_grad_norm:.3e}",
                                              params_with_grad=f"{ch_has_grad}/{ch_params}",
                                              status=status)
    
                        # Gate network
                        gate_net = None
                        for attr in ['gate_network', 'channel_gate', 'gate_input_proj']:
                            if hasattr(policy.sh_head, attr):
                                gate_net = getattr(policy.sh_head, attr)
                                break
                        if gate_net is not None:
                            gate_grad_norm = sum(p.grad.norm().item() ** 2 for p in gate_net.parameters() if p.grad is not None) ** 0.5
                            gate_has_grad = sum(1 for p in gate_net.parameters() if p.grad is not None and p.grad.norm().item() > 0)
                            gate_params = sum(1 for _ in gate_net.parameters())
                            status = "OK" if gate_has_grad > 0 else "DEAD"
                            self._dlog("GRADIENT", "Gate network",
                                      grad_norm=f"{gate_grad_norm:.3e}",
                                      params_with_grad=f"{gate_has_grad}/{gate_params}",
                                      status=status)
    
                        # Full gradient flow audit
                        self._check_all_gradient_flow(policy, episode_idx)
    
                        # Backbone gradient check
                        if actor_embedding_extractor is not None and critic_embedding_extractor is not None:
                            self._check_backbone_gradient_flow(actor_embedding_extractor, critic_embedding_extractor, episode_idx)
    
                    # Clip direction (SH) params tighter so global scaling doesn't drown
                    # magnitude/axis.
                    max_grad_norm = getattr(self.algo_config, 'max_grad_norm', 5.0)
                    sh_clip = 1.0  # tight for direction head
                    other_clip = max_grad_norm

                    grad_norm_before = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5

                    sh_params = []
                    other_params = []
                    for group in self.optimizer.param_groups:
                        name = group.get('name', '')
                        for p in group['params']:
                            if p.grad is not None:
                                if 'sh_' in name or 'cross_tp' in name:
                                    sh_params.append(p)
                                else:
                                    other_params.append(p)

                    if sh_params:
                        nn.utils.clip_grad_norm_(sh_params, sh_clip)
                    if other_params:
                        nn.utils.clip_grad_norm_(other_params, other_clip)

                    grad_norm_after = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5
                    grad_norm = grad_norm_after
                    self._grad_total_count += 1
                    was_clipped = grad_norm_before > max_grad_norm
                    if was_clipped:
                        self._grad_clip_count += 1
                        if grad_norm_before > max_grad_norm * 3:
                            self._large_grad_episodes.append(episode_idx)
                            self._dlog("GRADIENT", "Large gradient clipped (per-group)",
                                      before=f"{grad_norm_before:.2f}",
                                      sh_clip=f"{sh_clip}",
                                      other_clip=f"{other_clip}")

                    # Periodic clipping summary
                    if episode_idx % 50 == 0 and ke == 0 and self._grad_total_count > 0:
                        clip_rate = 100 * self._grad_clip_count / self._grad_total_count
                        self._dlog("GRADIENT", "Gradient clip summary",
                                  episode=episode_idx,
                                  clip_count=self._grad_clip_count,
                                  total_count=self._grad_total_count,
                                  clip_rate=f"{clip_rate:.1f}%")
    
                    # Check for non-finite loss
                    if not torch.isfinite(losses['total']):
                        raise FloatingPointError(f"Non-finite total loss at episode {episode_idx}, epoch {ke}")
    
                    # Store weight before step for delta check
                    _w0_value_mlp = None
                    if _should_log and hasattr(policy, "value_mlp"):
                        _w0_value_mlp = policy.value_mlp[0].weight.detach().clone()

                    self.optimizer.step()

                    if ke == k_epochs - 1 and _should_log_update and self._cf_n > 0:
                        cf_dir = self._cf_dir_sum / self._cf_n
                        cf_mag = self._cf_mag_sum / self._cf_n
                        cf_axis = self._cf_axis_sum / self._cf_n
                        _LOG.debug(f"[DEBUG-CF] LOGGING: cf_dir={cf_dir:.4f} cf_mag={cf_mag:.4f} cf_axis={cf_axis:.4f} n={self._cf_n}")
                        _LOG.info(f"[PPO] Clip fraction (LAST EPOCH ke={ke}) | "
                              f"dir={cf_dir:.1%} mag={cf_mag:.1%} axis={cf_axis:.1%}")
                        del self._cf_dir_sum, self._cf_mag_sum, self._cf_axis_sum, self._cf_n

                    if _w0_value_mlp is not None:
                        dw = (policy.value_mlp[0].weight.detach() - _w0_value_mlp).abs().mean().item()
                        status = "UPDATING" if dw > 1e-8 else "NOT_UPDATING"
                        self._dlog("VALUE", "Critic step check",
                                  mean_delta_weight=f"{dw:.3e}",
                                  status=status)

                    # Axis step diagnostic
                    if hasattr(self.policy, 'axis_from_frame'):
                        axis_param = list(self.policy.axis_from_frame.parameters())[0]
                        if hasattr(self, '_axis_param_prev'):
                            delta = (axis_param - self._axis_param_prev).abs().mean().item()
                            status = "UPDATING" if delta > 1e-8 else "STUCK"
                            self._dlog("AXIS", "Axis step check",
                                      mean_delta_weight=f"{delta:.3e}",
                                      status=status)
                        self._axis_param_prev = axis_param.detach().clone()
    
                    # AXIS HEALTH CHECK (epoch 0 only)
                    if _should_log and ke == 0 and hasattr(self.policy, 'axis_from_frame'):
                        # A) Check policy_old independence
                        if policy_old is not None:
                            policy_axis_params = list(self.policy.axis_from_frame.parameters())
                            policy_old_axis_params = list(policy_old.axis_from_frame.parameters())
                            same_storage = any(p.data_ptr() == p_old.data_ptr() 
                                             for p, p_old in zip(policy_axis_params, policy_old_axis_params))
                            if same_storage:
                                self._dlog("AXIS", "Health check A: policy/policy_old SHARE axis param storage",
                                          status="CRITICAL")
                            else:
                                param_diff = sum((p.data - p_old.data).abs().max().item() 
                                               for p, p_old in zip(policy_axis_params, policy_old_axis_params))
                                if param_diff < 1e-10:
                                    self._dlog("AXIS", "Health check A: policy/policy_old axis params identical",
                                              status="WARN")
                                else:
                                    self._dlog("AXIS", "Health check A: policy_old independent",
                                              param_diff=f"{param_diff:.6f}",
                                              status="OK")
    
                        # B) Check axis params in optimizer
                        axis_param_ids = {id(p) for p in self.policy.axis_from_frame.parameters()}
                        optimizer_param_ids = set()
                        for group in self.optimizer.param_groups:
                            optimizer_param_ids.update(id(p) for p in group['params'])
                        if axis_param_ids.issubset(optimizer_param_ids):
                            self._dlog("AXIS", f"Health check B: All {len(axis_param_ids)} axis params in optimizer",
                                      status="OK")
                        else:
                            missing = len(axis_param_ids - optimizer_param_ids)
                            self._log("AXIS", f"Health check B: {missing} axis params NOT in optimizer",
                                      status="CRITICAL")
    
                        # C) Check axis gradients
                        axis_grad_norms = [(name, p.grad.norm().item() if p.grad is not None else None) 
                                          for name, p in self.policy.axis_from_frame.named_parameters()]
                        has_grads = [g for _, g in axis_grad_norms if g is not None and g > 0]
                        if len(has_grads) == len(axis_grad_norms):
                            self._dlog("AXIS", "Health check C: All axis params have gradients",
                                      status="OK")
                        else:
                            dead_params = [name for name, g in axis_grad_norms if g is None or g == 0]
                            self._log("AXIS", "Health check C: Axis gradient issues",
                                      dead_params=str(dead_params),
                                      status="CRITICAL")
    
                    # Track losses
                    losses_tracker.append({k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
                    self.grad_norm_history.append(float(grad_norm))
    
                    if _should_log:
                        self._dlog("PPO", f"Epoch {ke} complete (gradient accumulation)",
                                  total_loss=f"{losses['total'].item():.4f}",
                                  policy_loss=f"{losses['policy'].item():.4f}",
                                  value_loss=f"{losses['value'].item():.4f}",
                                  entropy=f"{avg_entropy:.4f}",
                                  grad_norm=f"{grad_norm:.4f}",
                                  grad_norm_before=f"{grad_norm_before:.4f}")
    
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Backbone path handled the whole epoch; skip the rest.
                    continue

                # Non-backbone path: use stored states (frozen backbone).
                actor_states = actor_states_all

                if _should_log:
                    if ke == 0:
                        self._state_hash_epoch0 = actor_states.detach().cpu().numpy().tobytes()
                        o2_emb = batch_data.get('o2_embeddings_asymmetric', None)
                        if o2_emb is not None:
                            self._o2_emb_hash_epoch0 = o2_emb.detach().cpu().numpy().tobytes()
                        else:
                            self._o2_emb_hash_epoch0 = None
                        self._dlog("PPO", "State consistency check - Epoch 0: Stored state hash",
                                  episode=episode_idx)

                # Verification (first epoch only)
                if _should_log:
                    self._dlog("PPO", "State verification",
                              batch_size=actor_states.shape[0],
                              state_shape=str(actor_states.shape),
                              actor_requires_grad=actor_states.requires_grad,
                              critic_inputs_present=(o1_embeddings_all is not None)) 

                # Use a consistent episode for all transitions in the schedule
                episode_indices = batch_data.get('episode_indices', None)
                if episode_indices is not None:
                    if isinstance(episode_indices, torch.Tensor):
                        ep_min, ep_max = episode_indices.min().item(), episode_indices.max().item()
                        ep_unique = episode_indices.unique().tolist()
                    else:
                        ep_min, ep_max = min(episode_indices), max(episode_indices)
                        ep_unique = list(set(episode_indices))

                    if _should_log:
                        self._dlog("PPO", "Episode tracking",
                                  batch_episodes=str(ep_unique),
                                  original_episode_idx=episode_idx)

                    if ep_min != ep_max:
                        episode_idx_for_eval = ep_min
                        if _should_log:
                            self._dlog("PPO", f"Multi-episode batch: using ep_min={ep_min} for schedule",
                                      status="WARN")
                    else:
                        episode_idx_for_eval = ep_min
                        if _should_log:
                            self._dlog("PPO", f"Single-episode batch: using ep={ep_min}",
                                      status="OK")
                else:
                    episode_idx_for_eval = episode_idx
                    if _should_log:
                        self._dlog("PPO", f"episode_indices not available, using {episode_idx}")

                # EVALUATE POLICY ON STORED STATES
                stored_temperatures = batch_data.get('temperatures', None)

                log_probs_new_agg, state_values, entropy, per_head_new = policy.evaluate(
                    actor_states,
                    episode_idx=episode_idx_for_eval,
                    old_actions=old_actions,
                    state_uid=None,
                    o2_embeddings_asymmetric=batch_data.get('o2_embeddings_asymmetric', None),
                    o1_embeddings=o1_embeddings_all,
                    o2_embeddings=o2_embeddings_all,
                    geometries=geometries_all,
                    stored_temperatures=stored_temperatures,
                    o2_positions=batch_data.get('o2_positions', None),
                    env_multipoles=env_multipoles_all,
                )

                # Pre-update consistency check (epoch 0 only)
                if ke == 0 and _should_log:
                    # Rollout vs update value consistency
                    stored_values_h2 = batch_data.get('state_values', None)
                    if stored_values_h2 is not None:
                        stored_flat_h2 = stored_values_h2.to(self.device).reshape(-1)
                        current_flat_h2 = state_values.detach().reshape(-1)

                        if stored_flat_h2.numel() != current_flat_h2.numel():
                            raise RuntimeError(
                                f"H2 SHAPE MISMATCH: stored={stored_flat_h2.numel()}, "
                                f"current={current_flat_h2.numel()}. "
                                f"Rollout and update use different state tensors!")
                        
                        v_diff_h2 = (current_flat_h2 - stored_flat_h2).abs()
                        diff_mean_h2 = v_diff_h2.mean().item()
                        diff_max_h2 = v_diff_h2.max().item()
                        match_ok_h2 = diff_max_h2 < 0.01

                        self._dlog("VALUE", "Rollout vs Update consistency (H2)",
                                  episode=episode_idx,
                                  stored_mean=f"{stored_flat_h2.mean().item():.4f}",
                                  current_mean=f"{current_flat_h2.mean().item():.4f}",
                                  diff_mean=f"{diff_mean_h2:.6f}",
                                  diff_max=f"{diff_max_h2:.6f}",
                                  status="OK" if match_ok_h2 else "MISMATCH")
                        
                        # Fail on mismatch (relaxed threshold for parallel workers)
                        h2_threshold = 1.0 if getattr(self.config.runtime, 'num_parallel_workers', 1) > 1 else 0.01
                        if diff_max_h2 >= h2_threshold:
                            raise RuntimeError(
                                f"H2 VALUE MISMATCH: diff_max={diff_max_h2:.6f} >= {h2_threshold}. "
                                f"Rollout values don't match evaluate() output!")

                    old_dir_check = batch_data['log_probs_direction'].to(self.device).reshape(-1)
                    old_axis_check = batch_data['log_probs_axis'].to(self.device).reshape(-1)

                    self._dlog("AXIS", "Buffer vs Evaluate comparison",
                              episode=episode_idx,
                              buffer_axis_vals=f"[{old_axis_check[0].item():.4f}, {old_axis_check[1].item():.4f}]" if len(old_axis_check) > 1 else f"[{old_axis_check[0].item():.4f}]",
                              eval_axis_vals=f"[{per_head_new['axis'].reshape(-1)[0].item():.4f}, {per_head_new['axis'].reshape(-1)[1].item():.4f}]" if len(per_head_new['axis'].reshape(-1)) > 1 else f"[{per_head_new['axis'].reshape(-1)[0].item():.4f}]")

                    new_dir_check = per_head_new['direction'].reshape(-1)                
                    log_ratio_dir = new_dir_check - old_dir_check
                    ratio_dir = torch.exp(log_ratio_dir.clamp(-10, 10))
                    dir_ratio_ok = ratio_dir.min() >= 0.8 and ratio_dir.max() <= 1.25

                    new_axis_check = per_head_new['axis'].reshape(-1)
                    log_ratio_axis = new_axis_check - old_axis_check
                    ratio_axis = torch.exp(log_ratio_axis.clamp(-10, 10))
                    axis_ratio_ok = ratio_axis.min() >= 0.8 and ratio_axis.max() <= 1.25

                    # Identity test: fail if old/new share storage (not recomputed)
                    same_ptr_dir = old_dir_check.data_ptr() == new_dir_check.data_ptr()
                    same_ptr_axis = old_axis_check.data_ptr() == new_axis_check.data_ptr()
                    max_diff_dir = float((new_dir_check - old_dir_check).abs().max())
                    max_diff_axis = float((new_axis_check - old_axis_check).abs().max())

                    if same_ptr_dir or same_ptr_axis:
                        raise RuntimeError(
                            f"IDENTITY TEST FAILED: Comparing tensor to itself!\n"
                            f"  dir same_ptr={same_ptr_dir}, axis same_ptr={same_ptr_axis}\n"
                            f"  old_dir.data_ptr={old_dir_check.data_ptr()}, new_dir.data_ptr={new_dir_check.data_ptr()}\n"
                            f"  old_axis.data_ptr={old_axis_check.data_ptr()}, new_axis.data_ptr={new_axis_check.data_ptr()}\n"
                            f"  This means evaluate() is not truly recomputing log_prob."
                        )

                    if episode_idx <= 20:
                        _LOG.debug(f"  [IDENTITY TEST] episode={episode_idx}: "
                              f"max_diff_dir={max_diff_dir:.8f}, max_diff_axis={max_diff_axis:.8f}")
                        # max_diff == 0.0 is fine (deterministic recompute); the pointer
                        # check above is the real fail-fast condition.

                    self._dlog("PPO", "Pre-update consistency (using buffer)",
                              episode=episode_idx,
                              dir_logratio_mean=f"{log_ratio_dir.mean():.3f}",
                              dir_logratio_std=f"{log_ratio_dir.std():.3f}",
                              dir_ratio_range=f"[{ratio_dir.min():.3f}, {ratio_dir.max():.3f}]",
                              dir_status="OK" if dir_ratio_ok else "MISMATCH",
                              axis_logratio_mean=f"{log_ratio_axis.mean():.3f}",
                              axis_logratio_std=f"{log_ratio_axis.std():.3f}",
                              axis_ratio_range=f"[{ratio_axis.min():.3f}, {ratio_axis.max():.3f}]",
                              axis_status="OK" if axis_ratio_ok else "MISMATCH")

                # Verify gradients flow to the value head
                if _should_log:
                    grad_status = "CRITICAL_DETACHED" if not state_values.requires_grad else "OK"
                    self._dlog("GRADIENT", "Value gradient sanity check",
                              episode=episode_idx,
                              requires_grad=state_values.requires_grad,
                              grad_fn=str(state_values.grad_fn)[:50] if state_values.grad_fn else "None",
                              status=grad_status)

                # One-time KL baseline capture
                if self.algo_config.use_kl_regularization and not self._kl_baseline_ready:
                    self._kl_baseline = self._capture_current_log_mixture_norm().detach()
                    self._kl_baseline_ready = True

                # Entropy checks
                if entropy is None or not isinstance(entropy, torch.Tensor) or entropy.numel() == 0:
                    raise RuntimeError(f"policy.evaluate() returned invalid entropy at episode {episode_idx}.")

                # Flatten everything
                sv_flat = state_values.reshape(-1)
                ret_flat = returns.reshape(-1)
                adv_flat = advantages.reshape(-1)

                # Old per-head log-probs from buffer (frozen backbone)
                old_dir = batch_data['log_probs_direction'].to(self.device).reshape(-1)
                old_mag = batch_data['log_probs_magnitude'].to(self.device).reshape(-1)
                old_axis = batch_data['log_probs_axis'].to(self.device).reshape(-1)

                if _should_log:
                    self._dlog("PPO", "Using stored buffer log-probs (frozen backbone)")

                # New per-head log-probs
                new_dir = per_head_new['direction'].to(self.device).reshape(-1)
                new_mag = per_head_new['magnitude'].to(self.device).reshape(-1)
                new_axis = per_head_new['axis'].to(self.device).reshape(-1)

                # Per-head advantages
                adv_dir = batch_data.get('advantages_direction', advantages).to(self.device).reshape(-1).detach()
                adv_mag = batch_data.get('advantages_magnitude', advantages).to(self.device).reshape(-1).detach()
                adv_axis = batch_data.get('advantages_axis', advantages).to(self.device).reshape(-1).detach()


                # VALIDATION
                batch_len = len(old_actions)

                if sv_flat.numel() != batch_len:
                    raise RuntimeError(f"state_values {sv_flat.numel()} != batch_len {batch_len}")
                if ret_flat.numel() != batch_len:
                    raise RuntimeError(f"returns {ret_flat.numel()} != batch_len {batch_len}")
                if old_dir.numel() != batch_len:
                    raise RuntimeError(f"old_dir {old_dir.numel()} != batch_len {batch_len}")
                if new_dir.numel() != batch_len:
                    raise RuntimeError(f"new_dir {new_dir.numel()} != batch_len {batch_len}")

                old_val_sliced = None
                if old_values is not None:
                    old_val_flat = old_values.reshape(-1)
                    if old_val_flat.numel() != batch_len:
                        raise RuntimeError(f"old_values {old_val_flat.numel()} != batch_len {batch_len}")
                    old_val_sliced = old_val_flat

                # Check for NaNs
                for name, tensor in [('sv_flat', sv_flat), ('ret_flat', ret_flat), ('old_dir', old_dir), 
                                     ('new_dir', new_dir), ('new_axis', new_axis)]:
                    if not torch.isfinite(tensor).all():
                        nan_count = (~torch.isfinite(tensor)).sum().item()
                        raise RuntimeError(f"{name} contains {nan_count} NaN/Inf!")

                # COMPUTE LOSSES
                losses = self._compute_losses(
                    per_head_new={'direction': new_dir, 'magnitude': new_mag, 'axis': new_axis},
                    per_head_old={'direction': old_dir, 'magnitude': old_mag, 'axis': old_axis},
                    per_head_adv={'direction': adv_dir, 'magnitude': adv_mag, 'axis': adv_axis},
                    state_values=sv_flat,
                    returns=ret_flat,
                    old_values=old_values_rollout,                    
                    entropy=entropy,
                    entropy_coeff=entropy_coeff,
                    episode_idx=episode_idx,
                    ke=ke,
                    k_epochs=k_epochs,
                    should_log=_should_log,
                )


                self.optimizer.zero_grad()
                losses['total'].backward()

                # Mark first update complete to enable episode_runner fail-fast checks
                if not getattr(self, '_first_update_marked', False):
                    self._first_update_marked = True
                    if hasattr(self, 'episode_runner'):
                        self.episode_runner._has_completed_first_update = True
                    elif hasattr(self, 'trainer') and hasattr(self.trainer, 'episode_runner'):
                        self.trainer.episode_runner._has_completed_first_update = True
                    _LOG.debug(f"[PPO] First update completed - fail-fast checks now active")

                # Head grad norms (non-backbone path)
                if _should_log and ke == 0:
                    value_grad_norm = 0.0
                    value_param_count = 0
                    axis_grad_norm = 0.0
                    axis_param_count = 0
                    for name, param in policy.named_parameters():
                        if param.grad is not None:
                            gn = param.grad.norm().item()
                            if name.startswith('critic_embedder.') or name.startswith('value_mlp.'):
                                value_grad_norm += gn ** 2
                                value_param_count += 1
                            if name.startswith('axis_from_frame.') or name.startswith('axis_net.'):
                                axis_grad_norm += gn ** 2
                                axis_param_count += 1
                    value_grad_norm = value_grad_norm ** 0.5
                    axis_grad_norm = axis_grad_norm ** 0.5
                    self._dlog("GRADIENT", "Head grad norms (epoch 0, non-backbone)",
                              episode=episode_idx,
                              value_grad_norm=f"{value_grad_norm:.4e}",
                              value_params=value_param_count,
                              axis_grad_norm=f"{axis_grad_norm:.4e}",
                              axis_params=axis_param_count)
                    if value_grad_norm < 1e-8 and value_param_count > 0:
                        self._log("GRADIENT", "WARNING: value_head has ~zero gradients", status="CRITICAL")
                    if axis_grad_norm < 1e-8 and axis_param_count > 0:
                        self._log("GRADIENT", "WARNING: axis_net has ~zero gradients", status="CRITICAL")

                # Value grad check
                if _should_log:
                    with torch.no_grad():
                        total_grad = 0.0
                        count = 0
                        for name in ['critic_embedder', 'value_mlp']:
                            if hasattr(policy, name):
                                layer = getattr(policy, name)
                                for p in layer.parameters():                        
                                    if p.grad is not None:
                                        total_grad += p.grad.data.norm(2).item() ** 2
                                        count += 1
                        self._dlog("VALUE", "Value grad diagnostic",
                                  grad_norm=f"{(total_grad ** 0.5):.4e}",
                                  num_tensors=count)                

                # GRADIENT CHECKS
                if _should_log:
                    self._dlog("GRADIENT", f"Gradient check after backward (episode {episode_idx})")
                    for name in ['critic_embedder', 'value_mlp']:
                        if hasattr(policy, name):
                            layer = getattr(policy, name)
                            if hasattr(layer, 'weight') and layer.weight.grad is not None:                        
                                gn = layer.weight.grad.norm().item()
                                status = "OK" if gn > 1e-8 else "ZERO"
                                self._dlog("GRADIENT", f"Value head {name}",
                                          grad_norm=f"{gn:.3e}",
                                          status=status)

                    self._check_all_gradient_flow(policy, episode_idx)

                # PER-GROUP GRADIENT CLIPPING
                max_grad_norm = getattr(self.algo_config, 'max_grad_norm', 5.0)
                sh_clip = 1.0
                other_clip = max_grad_norm
                grad_norm_before = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5
                sh_params = []
                other_params = []
                for group in self.optimizer.param_groups:
                    name = group.get('name', '')
                    for p in group['params']:
                        if p.grad is not None:
                            if 'sh_' in name or 'cross_tp' in name:
                                sh_params.append(p)
                            else:
                                other_params.append(p)
                if sh_params:
                    nn.utils.clip_grad_norm_(sh_params, sh_clip)
                if other_params:
                    nn.utils.clip_grad_norm_(other_params, other_clip)
                grad_norm = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5
                self._grad_total_count += 1
                if grad_norm_before > max_grad_norm:
                    self._grad_clip_count += 1

                if not torch.isfinite(losses['total']):
                    raise FloatingPointError(f"Non-finite total loss at episode {episode_idx}, epoch {ke}")

                # Store weight for delta check (first Linear in value_mlp Sequential)
                _w0_value_mlp = None
                if _should_log and hasattr(policy, "value_mlp"):
                    _w0_value_mlp = policy.value_mlp[0].weight.detach().clone()

                # Pre-step MSE (tiny-batch diagnostic)
                if _should_log:
                    with torch.no_grad():
                        pre_step_mse = F.mse_loss(state_values, returns).item()
                        signed_bias = (returns.mean() - state_values.mean()).item()
                
                # OPTIMIZER STEP
                self.optimizer.step()
                # CLIP FRACTION AT LAST EPOCH
                if ke == k_epochs - 1 and _should_log_update:
                    with torch.no_grad():
                        dir_ratio = torch.exp((new_dir - old_dir).clamp(-3.0, 3.0))
                        mag_ratio = torch.exp((new_mag - old_mag).clamp(-3.0, 3.0))
                        axis_ratio = torch.exp((new_axis - old_axis).clamp(-3.0, 3.0))
                        cf_dir = ((dir_ratio < 0.7) | (dir_ratio > 1.3)).float().mean().item()
                        cf_mag = ((mag_ratio < 0.7) | (mag_ratio > 1.3)).float().mean().item()
                        cf_axis = ((axis_ratio < 0.7) | (axis_ratio > 1.3)).float().mean().item()
                        _LOG.info(f"[PPO] Clip fraction (LAST EPOCH ke={ke}) | "
                              f"dir={cf_dir:.1%} mag={cf_mag:.1%} axis={cf_axis:.1%} | "
                              f"dir_ratio=[{dir_ratio.min().item():.6f}, {dir_ratio.max().item():.6f}] | "
                              f"mag_ratio=[{mag_ratio.min().item():.6f}, {mag_ratio.max().item():.6f}] | "
                              f"axis_ratio=[{axis_ratio.min().item():.6f}, {axis_ratio.max().item():.6f}]")

                # Mark first update complete to enable fail-fast checks
                if not getattr(self, '_first_update_marked', False):
                    self._first_update_marked = True
                    if hasattr(self, 'episode_runner'):
                        self.episode_runner._has_completed_first_update = True
                    elif hasattr(self, 'trainer') and hasattr(self.trainer, 'episode_runner'):
                        self.trainer.episode_runner._has_completed_first_update = True
                    _LOG.debug(f"[PPO] First update completed - fail-fast checks now active")
                    
                # Post-step MSE (verify critic is fitting)
                if _should_log:
                    with torch.no_grad():
                        # Re-evaluate values after step (value head only)
                        post_values = policy._compute_value(o1_embeddings_all, o2_embeddings_all, geometries_all)
                        post_step_mse = F.mse_loss(post_values.reshape(-1), returns).item()
                        mse_improved = pre_step_mse > post_step_mse
                        self._dlog("VALUE", "Pre/Post step MSE",
                                  pre_mse=f"{pre_step_mse:.4f}",
                                  post_mse=f"{post_step_mse:.4f}",
                                  signed_bias=f"{signed_bias:.4f}",
                                  improved="YES" if mse_improved else "NO")
                        
                        # Post-update EV/correlation, first epoch only to avoid flooding
                        if ke == 0:
                            _post_flat = post_values.reshape(-1).float()
                            _ret_flat_post = returns.reshape(-1).float()

                            if _post_flat.numel() != _ret_flat_post.numel():
                                self._dlog("VALUE", "POST-UPDATE EV (Test A) SKIPPED",
                                          episode=episode_idx,
                                          reason="shape_mismatch",
                                          post_n=_post_flat.numel(),
                                          ret_n=_ret_flat_post.numel())
                            else:
                                # unbiased=False matches the batch-statistics convention
                                _ret_var_post = _ret_flat_post.var(unbiased=False).item()
                                _ret_std_post = _ret_flat_post.std(unbiased=False).item()

                                if _ret_var_post > 1e-6:
                                    _post_resid = (_ret_flat_post - _post_flat).var(unbiased=False).item()
                                    _post_ev = 1 - _post_resid / _ret_var_post

                                    _post_mean = _post_flat.mean()
                                    _ret_mean_post = _ret_flat_post.mean()
                                    _post_z = _post_flat - _post_mean
                                    _ret_z_post = _ret_flat_post - _ret_mean_post
                                    _denom = (_post_z.pow(2).mean().sqrt() * _ret_z_post.pow(2).mean().sqrt()) + 1e-8
                                    _post_corr = (_post_z * _ret_z_post).mean() / _denom
                                    
                                    # Pre-update EV for comparison (if shape matches)
                                    _pre_ev_str = "N/A"
                                    _ev_improv_str = "N/A"
                                    _learning_status = "NO_BASELINE"
                                    _baseline_std_str = "N/A"
                                    
                                    if old_values_rollout is not None:
                                        _old_flat = old_values_rollout.reshape(-1).float()
                                        if _old_flat.numel() == _ret_flat_post.numel():
                                            _pre_resid = (_ret_flat_post - _old_flat).var(unbiased=False).item()
                                            _pre_ev = 1 - _pre_resid / _ret_var_post
                                            _pre_ev_str = f"{_pre_ev:.4f}"
                                            _baseline_std_str = f"{_old_flat.std().item():.4f}"
                                            _ev_improv = _post_ev - _pre_ev
                                            _ev_improv_str = f"{_ev_improv:.4f}"
                                            _learning_status = "LEARNING" if _ev_improv > 0.01 else "FLAT"
                                        else:
                                            _pre_ev_str = "SHAPE_MISMATCH"
                                    
                                    self._dlog("VALUE", "POST-UPDATE EV (Test A)",
                                              episode=episode_idx,
                                              pre_ev=_pre_ev_str,
                                              post_ev=f"{_post_ev:.4f}",
                                              ev_improvement=_ev_improv_str,
                                              post_corr=f"{_post_corr.item():.4f}",
                                              post_values_std=f"{_post_flat.std().item():.4f}",
                                              baseline_src="old_values_rollout",
                                              baseline_std=_baseline_std_str,
                                              status=_learning_status)
                                              
                                    if _post_corr.item() < -0.3:
                                        self._log("VALUE", "POST-UPDATE WARNING: Negative correlation",
                                                  episode=episode_idx,
                                                  post_corr=f"{_post_corr.item():.4f}",
                                                  hint="Early training noise OR critic scale/sign issue")
                                else:
                                    self._dlog("VALUE", "POST-UPDATE EV (Test A) SKIPPED",
                                              episode=episode_idx,
                                              reason="returns_var_too_small",
                                              returns_var=f"{_ret_var_post:.2e}",
                                              returns_std=f"{_ret_std_post:.4f}",
                                              returns_mean=f"{_ret_flat_post.mean().item():.4f}",
                                              post_values_mean=f"{_post_flat.mean().item():.4f}",
                                              post_values_std=f"{_post_flat.std().item():.4f}")
                        
                        # Value clip diagnostic (is clipping binding?) + rollout-value
                        # consistency check.
                        if _should_log and old_values_rollout is not None:
                            current_hash = old_values_rollout.detach().cpu().numpy().tobytes()
                            if current_hash != self._old_values_rollout_hash:
                                raise RuntimeError("old_values_rollout was modified during epoch loop!")
                            delta = (post_values.reshape(-1) - old_values_rollout)
                            clip_frac = (delta.abs() > self.value_clip_range).float().mean().item()
                            self._dlog("VALUE", f"Value clip (epoch {ke})",
                                      clip_range=self.value_clip_range,
                                      clip_frac=f"{clip_frac:.3f}",
                                      delta_mean=f"{delta.mean().item():.4f}",
                                      delta_std=f"{delta.std().item():.4f}",
                                      delta_max=f"{delta.abs().max().item():.4f}")

                # Post-step weight delta tracking
                if _w0_value_mlp is not None:
                    _w_after = policy.value_mlp[0].weight.detach()
                    _dw_abs = (_w_after - _w0_value_mlp).abs().mean().item()
                    _delta_norm = (_w_after - _w0_value_mlp).norm().item()                    
                    _w_norm = _w_after.norm().item()
                    _rel_delta = _delta_norm / (_w_norm + 1e-8)
                    _status = "UPDATING" if _delta_norm > 1e-6 else "FROZEN"
                    self._dlog("VALUE", "Weight delta (H5b)",
                              episode=episode_idx,
                              delta_norm=f"{_delta_norm:.6f}",
                              weight_norm=f"{_w_norm:.4f}",
                              relative_delta=f"{_rel_delta:.6f}",
                              mean_abs_delta=f"{_dw_abs:.3e}",
                              status=_status)

                if hasattr(self.policy, 'axis_from_frame'):
                    axis_param = list(self.policy.axis_from_frame.parameters())[0]
                    if hasattr(self, '_axis_param_prev'):
                        delta = (axis_param - self._axis_param_prev).abs().mean().item()
                        status = "UPDATING" if delta > 1e-8 else "STUCK"
                        self._dlog("AXIS", "Axis step check",
                                  mean_delta_weight=f"{delta:.3e}",
                                  status=status)
                    self._axis_param_prev = axis_param.detach().clone()

                # AXIS HEALTH CHECK (non-backbone path)
                if _should_log and ke == 0 and hasattr(self.policy, 'axis_from_frame'):
                    # A) Check policy_old independence
                    if policy_old is not None:
                        policy_axis_params = list(self.policy.axis_from_frame.parameters())
                        policy_old_axis_params = list(policy_old.axis_from_frame.parameters())
                        same_storage = any(p.data_ptr() == p_old.data_ptr() 
                                         for p, p_old in zip(policy_axis_params, policy_old_axis_params))
                        if same_storage:
                            self._dlog("AXIS", "Health check A: policy/policy_old SHARE axis param storage",
                                      status="CRITICAL")
                        else:
                            param_diff = sum((p.data - p_old.data).abs().max().item() 
                                           for p, p_old in zip(policy_axis_params, policy_old_axis_params))
                            if param_diff < 1e-10:
                                self._dlog("AXIS", "Health check A: policy/policy_old axis params identical",
                                          status="WARN")
                            else:
                                self._dlog("AXIS", "Health check A: policy_old independent",
                                          param_diff=f"{param_diff:.6f}",
                                          status="OK")
                    else:
                        self._dlog("AXIS", "Health check A: policy_old is None - using buffer log-probs",
                                  status="WARN")

                    # B) Check axis params in optimizer
                    axis_param_ids = {id(p) for p in self.policy.axis_from_frame.parameters()}
                    optimizer_param_ids = set()
                    for group in self.optimizer.param_groups:
                        optimizer_param_ids.update(id(p) for p in group['params'])
                    if axis_param_ids.issubset(optimizer_param_ids):
                        self._dlog("AXIS", f"Health check B: All {len(axis_param_ids)} axis params in optimizer",
                                  status="OK")
                    else:
                        missing = len(axis_param_ids - optimizer_param_ids)
                        self._log("AXIS", f"Health check B: {missing} axis params NOT in optimizer",
                                  status="CRITICAL")

                    # C) Check axis gradients exist
                    axis_grad_norms = [(name, p.grad.norm().item() if p.grad is not None else None) 
                                      for name, p in self.policy.axis_from_frame.named_parameters()]
                    has_grads = [g for _, g in axis_grad_norms if g is not None and g > 0]
                    if len(has_grads) == len(axis_grad_norms):
                        self._dlog("AXIS", "Health check C: All axis params have gradients",
                                  status="OK")
                    else:
                        dead_params = [name for name, g in axis_grad_norms if g is None or g == 0]
                        self._log("AXIS", "Health check C: Axis gradient issues",
                                  dead_params=str(dead_params),
                                  status="CRITICAL")

                # Track
                losses_tracker.append({k: v.item() for k, v in losses.items()})                
                self.grad_norm_history.append(float(grad_norm))

                if _should_log:
                    self._dlog("PPO", f"Epoch {ke} complete (non-backbone path)",
                              total_loss=f"{losses['total'].item():.4f}",
                              grad_norm=f"{grad_norm:.4f}")

            # Refresh KL baseline after the full update
            if self.algo_config.use_kl_regularization:                
                self._kl_baseline = self._capture_current_log_mixture_norm().detach()
        
            if not losses_tracker:
                raise RuntimeError(
                    f"No valid optimization steps completed at episode {episode_idx}."
                )
        
            avg_losses = {k: np.mean([l[k] for l in losses_tracker]) for k in losses_tracker[0]}
            metrics = {
                'grad_norm': np.mean(self.grad_norm_history[-k_epochs:]) if self.grad_norm_history else 0.0,
                'entropy': float(entropy.mean().item())
            }

            # Extra critic-only epochs: joint PPO gives the critic only k_epochs steps,
            # too few for high-variance returns. Extra MSE-only steps push online EV up.
            critic_extra_epochs = getattr(self.algo_config, 'critic_extra_epochs', 12)
            if critic_extra_epochs > 0 and o1_embeddings_all is not None:
                critic_param_set = set()
                for comp_name in ['critic_embedder', 'value_mlp']:
                    if hasattr(policy, comp_name):
                        for p in getattr(policy, comp_name).parameters():
                            critic_param_set.add(id(p))

                critic_extra_losses = []
                for ce in range(critic_extra_epochs):
                    # Minibatch the extra epochs to cap peak memory: N/MB optimizer steps
                    # per epoch, matching the actor minibatch envelope to avoid OOM.
                    EXTRA_EPOCHS_MB_SIZE = 32
                    _ce_mb_losses = []
                    _ce_N = o1_embeddings_all.shape[0]
                    _ce_MB = min(EXTRA_EPOCHS_MB_SIZE, _ce_N)
                    for _mb_s in range(0, _ce_N, _ce_MB):
                        _mb_e = min(_mb_s + _ce_MB, _ce_N)
                        state_values_mb = policy._compute_value(
                            o1_embeddings_all[_mb_s:_mb_e],
                            o2_embeddings_all[_mb_s:_mb_e],
                            geometries_all[_mb_s:_mb_e],
                        ).squeeze(-1)
                        value_loss_mb = F.mse_loss(state_values_mb, returns[_mb_s:_mb_e])
                        self.optimizer.zero_grad()
                        value_loss_mb.backward()
                        for param in policy.parameters():
                            if param.grad is not None and id(param) not in critic_param_set:
                                param.grad.zero_()
                        critic_params_list = [p for p in policy.parameters()
                                             if id(p) in critic_param_set and p.grad is not None]
                        nn.utils.clip_grad_norm_(critic_params_list, max_grad_norm)
                        self.optimizer.step()
                        _ce_mb_losses.append(value_loss_mb.item())
                    value_loss_ce_val = sum(_ce_mb_losses) / max(1, len(_ce_mb_losses))
                    critic_extra_losses.append(value_loss_ce_val)
                with torch.no_grad():
                    v_final = policy._compute_value(
                        o1_embeddings_all, o2_embeddings_all, geometries_all
                    ).squeeze(-1)
                    ev_after = 1 - (returns - v_final).var().item() / (returns.var().item() + 1e-8)

                self._dlog("CRITIC_EXTRA", "Extra critic epochs complete",
                          extra_epochs=critic_extra_epochs,
                          loss_first=f"{critic_extra_losses[0]:.4f}",
                          loss_last=f"{critic_extra_losses[-1]:.4f}",
                          ev_after=f"{ev_after:.4f}")

            # UPDATE RUN COUNTERS
            self._optimize_complete_count += 1
            self._total_optimizer_steps += k_epochs

            # One-time critic overfit test after first update
            if self._optimize_complete_count == 1:
                self._critic_overfit_test(policy, batch_data, returns, episode_idx)

            # PPO UPDATE SUMMARY
            self._log("PPO", "PPO UPDATE SUMMARY",
                      episode=episode_idx,
                      epochs=k_epochs,
                      batch_size=len(old_actions),
                      policy_loss=f"{avg_losses['policy']:.4f}",
                      value_loss=f"{avg_losses['value']:.4f}",
                      entropy_bonus=f"{avg_losses['entropy']:.4f}",
                      total_loss=f"{avg_losses['total']:.4f}",
                      grad_norm=f"{metrics['grad_norm']:.4f}",
                      entropy=f"{metrics['entropy']:.4f}")
            
            return avg_losses, metrics

        finally:
            # Restore temp-freeze to its pre-PPO state. Never raise here; it would
            # mask the real exception.
            for norm_name, norm in [('actor', actor_state_normalizer),
                                     ('critic', critic_state_normalizer),
                                     ('main', state_normalizer),
                                     ('critic_input', getattr(policy, 'critic_input_normalizer', None))]:
                if norm is not None and hasattr(norm, 'set_frozen_temporarily'):
                    prev = normalizer_prev_temp.get(norm_name)
                    if prev is None:
                        self._log("NORMALIZER", f"missing prev temp for {norm_name} (save block didn't run)",
                                  status="CRITICAL")
                    else:
                        norm.set_frozen_temporarily(prev)
                        if episode_idx % 10 == 0:
                            self._dlog("NORMALIZER", f"{norm_name}: Restored temp_frozen={prev}")

