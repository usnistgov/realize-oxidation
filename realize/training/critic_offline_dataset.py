"""Offline critic dataset collection: compute_returns_with_bootstrap, collect_offline_dataset."""
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from realize.config.paths import get_embedding_checkpoint_path
import logging
_LOG = logging.getLogger(__name__)


def compute_returns_with_bootstrap(
    rewards: List[float],
    terminals: List[bool],
    truncations: List[bool],
    next_values: List[float],
    gamma: float = 0.99
) -> List[float]:
    """Returns with truncation bootstrap, matching PPO GAE.

    At truncation: G_t = r_t + gamma * V(s_{t+1}). At terminal: G_t = r_t (running return resets).
    """
    n = len(rewards)
    returns = [0.0] * n
    running = 0.0

    for t in reversed(range(n)):
        if terminals[t]:
            # True terminal: no bootstrap.
            running = 0.0
        elif truncations[t]:
            # Truncation: bootstrap from V(s_{t+1}).
            running = float(next_values[t])

        running = rewards[t] + gamma * running
        returns[t] = running

    return returns


def collect_offline_dataset(
    trainer,
    num_episodes: int = 100,
    gamma: float = 0.99,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """Collect (o1_embedding, o2_embedding, geometry, return) tuples with a frozen actor.

    Reuses episode_runner's episode-collection logic; handles truncation with bootstrap (matches PPO).
    """
    if verbose:
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"OFFLINE CRITIC TEST: Collecting {num_episodes} episodes...")
        _LOG.info(f"{'='*60}")

    episode_runner = trainer.episode_runner
    agent = trainer.agent
    device = agent.device

    # Actor normalizer is frozen during collection. CriticEmbedding handles
    # critic inputs directly, so there is no separate critic normalizer.
    actor_norm_prev = None

    # Freeze is deferred until the first embedding seeds running stats: the
    # offline test may run before any training episodes.
    actor_norm_frozen = False

    policy_prev_training = agent.policy.training if hasattr(agent, 'policy') else True
    if hasattr(agent, 'policy'):
        agent.policy.eval()

    # Storage for critic raw inputs (o1, o2, geometry).
    all_o1_embeddings = []
    all_o2_embeddings = []
    all_geometries = []
    all_returns = []
    all_rewards = []
    all_episode_ids = []
    all_is_train = []

    # Train/val split: 80/20 by episode, shuffled.
    rng = np.random.default_rng(seed=42)
    episode_indices = np.arange(num_episodes)
    rng.shuffle(episode_indices)
    train_episodes = set(episode_indices[:int(num_episodes * 0.8)])

    max_steps = getattr(trainer.config.runtime, 'max_steps', 50)

    skipped_episodes = 0
    max_skipped = num_episodes // 2  # Skip at most half before bailing.

    env = episode_runner.env

    try:
        ep_idx = 0
        collected_episodes = 0

        while collected_episodes < num_episodes and ep_idx < num_episodes * 2:
            # Force a clean structure reset.
            try:
                if hasattr(env, 'structure_manager') and hasattr(env, 'data_file'):
                    env.structure_manager.initialize_from_file(env.data_file)
                    if verbose and ep_idx == 0:
                        _LOG.info(f"  [OFFLINE] Forcing clean reset from {env.data_file}")

                env.previous_episode_dissociated = False
                env.o2_removed = False
                env.cumulative_dissociated_o_count = 0

            except Exception as e:
                _LOG.info(f"  [WARNING] Failed to force reset structure: {e}")

            # Reset via episode_runner.
            try:
                state = episode_runner._reset_episode(ep_idx)

                if state is None:
                    _LOG.info(f"  [WARNING] Episode {ep_idx} reset returned None, skipping")
                    skipped_episodes += 1
                    ep_idx += 1
                    continue

            except RuntimeError as e:
                if "corrupted" in str(e).lower() or "FATAL" in str(e):
                    _LOG.info(f"  [WARNING] Episode {ep_idx} reset failed (corrupted structure), skipping: {e}")
                    skipped_episodes += 1
                    ep_idx += 1
                    if skipped_episodes > max_skipped:
                        raise RuntimeError(f"Too many skipped episodes ({skipped_episodes}). Environment unstable.")
                    continue
                else:
                    raise

            episode_rewards = []
            episode_terminals = []
            episode_truncations = []
            episode_next_values = []
            # Per-step critic raw inputs.
            episode_o1_embs = []
            episode_o2_embs = []
            episode_geoms = []

            done = False
            step = 0

            while not done and step < max_steps:
                with torch.no_grad():
                    # Get embeddings via the same pipeline as training.
                    atoms_current = episode_runner.env.structure_manager.get_atoms()

                    # Diagnostic: O1/O2 stability check.
                    if step == 0 or (ep_idx < 5 and step < 10):
                        symbols = atoms_current.get_chemical_symbols()
                        n_atoms = len(symbols)
                        last_two = symbols[-2:]
                        d_oo = atoms_current.get_distance(n_atoms-2, n_atoms-1, mic=True)
                        _LOG.info(f"  [DIAG ep{ep_idx} step{step}] n_atoms={n_atoms}, "
                              f"last_two={last_two}, d_OO={d_oo:.3f} Angstrom")

                    # Actor embedding for the policy forward pass.
                    if hasattr(episode_runner, 'actor_embedding_extractor'):
                        actor_embedding_raw = episode_runner.actor_embedding_extractor.get_equivariant_embedding(
                            atoms_current, return_symmetric_only=True
                        )
                        if isinstance(actor_embedding_raw, dict) and 'symmetric' in actor_embedding_raw:
                            actor_embedding_raw = actor_embedding_raw['symmetric']
                    else:
                        if not isinstance(state, torch.Tensor):
                            actor_embedding_raw = torch.tensor(state, device=device).float()
                        else:
                            actor_embedding_raw = state.to(device).float()

                    if actor_embedding_raw.dim() == 1:
                        actor_embedding_raw = actor_embedding_raw.unsqueeze(0)

                    # Critic raw inputs (o1, o2, geometry) for CriticEmbedding.
                    # Match the training path: augment geom_base with step fraction
                    # plus multipole invariants.
                    if hasattr(episode_runner, 'critic_embedding_extractor'):
                        critic_emb_dict = episode_runner.critic_embedding_extractor.get_equivariant_embedding(
                            atoms_current, return_symmetric_only=False
                        )
                        o1_emb = critic_emb_dict['o1_embedding'].detach()
                        o2_emb = critic_emb_dict['o2_embedding'].detach()
                        geom_base = critic_emb_dict['geometry'].detach()

                        # Compute env_multipole. The O2 pair is the last two atoms;
                        # substrate is all positions except those. Signature:
                        # forward(com, substrate_positions, substrate_species_idx, cell=None, pbc=None) -> [1, 9].
                        from realize.networks.encoders.critic_embedding import augment_geometry_with_context as _aug_geom
                        import numpy as _np_aug
                        env_multipole = None
                        if hasattr(episode_runner, 'env_multipole') and episode_runner.env_multipole is not None:
                            _all_positions = torch.as_tensor(
                                atoms_current.get_positions(), dtype=torch.float32, device=device
                            )  # [N_total, 3]
                            _substrate_positions = _all_positions[:-2]                      # [N_sub, 3]
                            _o2_com = _all_positions[-2:].mean(dim=0, keepdim=True)         # [1, 3]
                            _atomic_numbers = torch.as_tensor(
                                atoms_current.get_atomic_numbers(), dtype=torch.long, device=device
                            )  # [N_total]
                            _substrate_species = episode_runner.env_multipole.get_species_idx(
                                _atomic_numbers[:-2]
                            )  # [N_sub]
                            _cell_np = _np_aug.array(atoms_current.get_cell())
                            _pbc_np = _np_aug.array(atoms_current.get_pbc())
                            _cell_t = torch.as_tensor(_cell_np, dtype=torch.float32, device=device)
                            _pbc_t = torch.as_tensor(_pbc_np, dtype=torch.bool, device=device)
                            env_multipole = episode_runner.env_multipole(
                                _o2_com, _substrate_positions, _substrate_species, _cell_t, _pbc_t
                            ).detach()  # [1, 9]

                        # Augment: [B, 6] -> [B, 10].
                        max_steps_for_aug = getattr(trainer.config.runtime, 'max_steps', 50)
                        geom = _aug_geom(geom_base, step, max_steps_for_aug, env_multipole)

                        if geom.shape[-1] != 10:
                            raise RuntimeError(
                                f"[OFFLINE_TEST_FIX] geometry augmentation produced shape {list(geom.shape)}, "
                                f"expected last dim = 10. base_shape={list(geom_base.shape)}, "
                                f"env_multipole_shape={None if env_multipole is None else list(env_multipole.shape)}"
                            )

                        # Diagnostic: first few augmentations.
                        if ep_idx < 3 and step < 3:
                            _LOG.info(f"  [OFFLINE_AUG ep{ep_idx} step{step}] "
                                  f"geom_base.shape={list(geom_base.shape)} -> geom.shape={list(geom.shape)}")
                            _LOG.info(f"    geom_base = {geom_base.squeeze().tolist()}")
                            _LOG.info(f"    geom_full = {geom.squeeze().tolist()}")
                            if env_multipole is not None:
                                _LOG.info(f"    env_multipole = shape={list(env_multipole.shape)}, "
                                      f"l0={env_multipole[0,0].item():.4f}, "
                                      f"l1_norm={env_multipole[0,1:4].norm().item():.4f}, "
                                      f"l2_norm={env_multipole[0,4:9].norm().item():.4f}")
                            else:
                                _LOG.info(f"    env_multipole = None (environmental_multipole not available)")
                    else:
                        raise RuntimeError(
                            "FAIL: critic_embedding_extractor not found on episode_runner. "
                            "Cannot run offline critic test without dual backbone."
                        )
                    if ep_idx < 3 and step < 5:
                        _LOG.info(f"  [DIAG EMBEDDING ep{ep_idx} step{step}]")
                        _LOG.info(f"    o1_emb: shape={o1_emb.shape}, mean={o1_emb.mean():.4f}, std={o1_emb.std():.4f}")
                        _LOG.info(f"    o2_emb: shape={o2_emb.shape}, mean={o2_emb.mean():.4f}, std={o2_emb.std():.4f}")
                        _LOG.info(f"    geom:   shape={geom.shape}, values={geom.squeeze().tolist()}")

                        # Check dead dimensions.
                        o1_dead = (o1_emb.std(dim=0) < 1e-6).float().mean().item() if o1_emb.dim() > 1 else 0
                        o2_dead = (o2_emb.std(dim=0) < 1e-6).float().mean().item() if o2_emb.dim() > 1 else 0
                        _LOG.info(f"    Dead dims (std<1e-6): o1={o1_dead:.1%}, o2={o2_dead:.1%}")

                    # Ensure batch dimensions.
                    if o1_emb.dim() == 1:
                        o1_emb = o1_emb.unsqueeze(0)
                    if o2_emb.dim() == 1:
                        o2_emb = o2_emb.unsqueeze(0)
                    if geom.dim() == 1:
                        geom = geom.unsqueeze(0)

                    # Normalize actor embedding for action selection.
                    if hasattr(episode_runner, 'actor_state_normalizer'):
                        norm = episode_runner.actor_state_normalizer
                        # If frozen but uninitialized, temporarily unfreeze so normalize()
                        # can seed running stats on the first call.
                        need_temp_unfreeze = (
                            not getattr(norm, '_initialized', False) and
                            (getattr(norm, '_frozen', False) or getattr(norm, '_temp_frozen', False))
                        )
                        if need_temp_unfreeze:
                            orig_frozen = norm._frozen
                            orig_temp_frozen = getattr(norm, '_temp_frozen', False)
                            norm._frozen = False
                            if hasattr(norm, '_temp_frozen'):
                                norm._temp_frozen = False

                        actor_state = norm.normalize(actor_embedding_raw)

                        if need_temp_unfreeze:
                            # Restore frozen state after initialization.
                            norm._frozen = orig_frozen
                            if hasattr(norm, '_temp_frozen'):
                                norm._temp_frozen = orig_temp_frozen
                    else:
                        actor_state = actor_embedding_raw


                    # Store critic raw inputs.
                    episode_o1_embs.append(o1_emb.detach().cpu())
                    episode_o2_embs.append(o2_emb.detach().cpu())
                    episode_geoms.append(geom.detach().cpu())

                    # O2 asymmetric embedding if available.
                    o2_emb_asym = getattr(episode_runner, 'current_o2_embedding_asymmetric', None)

                    # Select action, passing critic raw inputs for value computation.
                    action_dict, log_prob, value, entropy = agent.policy.act(
                        actor_state,
                        episode=ep_idx,
                        o2_embedding_asymmetric=o2_emb_asym,
                        o1_embedding=o1_emb.to(device),
                        o2_embedding=o2_emb.to(device),
                        geometry=geom.to(device),
                        # Pass env_multipole to the actor to match the training path.
                        env_multipole=env_multipole if env_multipole is not None else None,
                    )

                # Step the environment.
                translation_direction = action_dict['translation_direction']
                translation_magnitude = action_dict['translation_magnitude']
                axis_direction = action_dict.get('axis_direction')

                if isinstance(translation_direction, torch.Tensor):
                    translation_direction = translation_direction[0].cpu().numpy().flatten()
                if isinstance(translation_magnitude, torch.Tensor):
                    if translation_magnitude.dim() > 0:
                        translation_magnitude = translation_magnitude[0].cpu().numpy()
                    else:
                        translation_magnitude = translation_magnitude.cpu().numpy()
                if axis_direction is not None and isinstance(axis_direction, torch.Tensor):
                    axis_direction = axis_direction[0].cpu().numpy().flatten()

                trans_vector = translation_direction * float(translation_magnitude)

                if axis_direction is None:
                    axis_direction = np.array([0.0, 0.0, 1.0])

                o2_config = np.stack([trans_vector, axis_direction], axis=0)

                try:
                    step_result = episode_runner.env.step(
                        trans_action=trans_vector,
                        o2_config=o2_config,
                        checkpoint_path=get_embedding_checkpoint_path(),
                        device=device
                    )
                    distance, next_embedding, reward, done, valid = step_result
                    info = {
                        'distance': distance,
                        'valid': valid
                    }

                except Exception as e:
                    _LOG.info(f"  [WARNING] Step failed at ep={ep_idx}, step={step}: {e}")
                    reward = -0.5
                    done = True
                    valid = False
                    info = {'error': str(e)}
                    next_embedding = actor_embedding_raw

                # Detect truncation.
                known_trunc_keys = ("truncated", "TimeLimit.truncated")
                has_trunc_flag = any(k in info for k in known_trunc_keys) if isinstance(info, dict) else False
                truncated_from_info = bool(
                    info.get("truncated", False) or info.get("TimeLimit.truncated", False)
                )
                truncated_from_steps = (step == max_steps - 1) and (not done)
                truncated = bool(truncated_from_info or truncated_from_steps)

                is_terminal = bool(done and (not truncated))

                # Bootstrap value for truncation.
                next_value = 0.0
                if truncated and next_embedding is not None:
                    with torch.no_grad():
                        # Critic raw inputs from the post-step atoms.
                        atoms_next = episode_runner.env.structure_manager.get_atoms()

                        # Match the training path: augment the truncation-state geom
                        # with step fraction plus multipole invariants.
                        if hasattr(episode_runner, 'critic_embedding_extractor'):
                            next_critic_dict = episode_runner.critic_embedding_extractor.get_equivariant_embedding(
                                atoms_next, return_symmetric_only=False
                            )
                            next_o1 = next_critic_dict['o1_embedding'].detach()
                            next_o2 = next_critic_dict['o2_embedding'].detach()
                            next_geom_base = next_critic_dict['geometry'].detach()

                            # Compute env_multipole for the truncation state (same signature as above).
                            from realize.networks.encoders.critic_embedding import augment_geometry_with_context as _aug_geom_t
                            import numpy as _np_aug_t
                            next_env_multipole = None
                            if hasattr(episode_runner, 'env_multipole') and episode_runner.env_multipole is not None:
                                _ap = torch.as_tensor(atoms_next.get_positions(), dtype=torch.float32, device=device)
                                _sp = _ap[:-2]                                  # [N_sub, 3]
                                _com = _ap[-2:].mean(dim=0, keepdim=True)       # [1, 3]
                                _an = torch.as_tensor(atoms_next.get_atomic_numbers(), dtype=torch.long, device=device)
                                _ss = episode_runner.env_multipole.get_species_idx(_an[:-2])
                                _c = torch.as_tensor(_np_aug_t.array(atoms_next.get_cell()),
                                                      dtype=torch.float32, device=device)
                                _p = torch.as_tensor(_np_aug_t.array(atoms_next.get_pbc()),
                                                      dtype=torch.bool, device=device)
                                next_env_multipole = episode_runner.env_multipole(
                                    _com, _sp, _ss, _c, _p
                                ).detach()  # [1, 9]

                            # Augment: [B, 6] -> [B, 10].
                            _max_steps = getattr(trainer.config.runtime, 'max_steps', 50)
                            next_geom = _aug_geom_t(next_geom_base, step, _max_steps, next_env_multipole)

                            if next_geom.shape[-1] != 10:
                                raise RuntimeError(
                                    f"[OFFLINE_TEST_FIX truncation] geometry augmentation produced shape "
                                    f"{list(next_geom.shape)}, expected last dim = 10."
                                )
                        else:
                            raise RuntimeError(
                                "FAIL: critic_embedding_extractor not found. "
                                "Cannot bootstrap truncation value."
                            )
                        if next_o1.dim() == 1:
                            next_o1 = next_o1.unsqueeze(0)
                        if next_o2.dim() == 1:
                            next_o2 = next_o2.unsqueeze(0)
                        if next_geom.dim() == 1:
                            next_geom = next_geom.unsqueeze(0)

                        next_value = agent.policy.get_value(next_o1, next_o2, next_geom)
                        if hasattr(next_value, 'item'):
                            next_value = next_value.item()

                episode_rewards.append(float(reward))
                episode_terminals.append(is_terminal)
                episode_truncations.append(truncated)
                episode_next_values.append(next_value)

                # Update state for the next step.
                if next_embedding is None:
                    if not done:
                        _LOG.info(f"  [WARNING] next_embedding is None but done=False at ep={ep_idx}, step={step}")
                        done = True
                elif not isinstance(next_embedding, torch.Tensor):
                    state = torch.tensor(next_embedding, device=device).float()
                else:
                    state = next_embedding.to(device).float()

                step += 1

            # Compute returns with bootstrap.
            episode_returns = compute_returns_with_bootstrap(
                episode_rewards, episode_terminals, episode_truncations,
                episode_next_values, gamma
            )

            # Store, only if data was collected.
            if episode_o1_embs:
                is_train_ep = collected_episodes in train_episodes
                for o1, o2, gm, ret, rew in zip(
                    episode_o1_embs, episode_o2_embs, episode_geoms,
                    episode_returns, episode_rewards
                ):
                    all_o1_embeddings.append(o1)
                    all_o2_embeddings.append(o2)
                    all_geometries.append(gm)
                    all_returns.append(ret)
                    all_rewards.append(rew)
                    all_episode_ids.append(collected_episodes)
                    all_is_train.append(is_train_ep)

                collected_episodes += 1

                if verbose and collected_episodes % 10 == 0:
                    n_terms = sum(episode_terminals)
                    n_truncs = sum(episode_truncations)
                    mean_ret = np.mean(episode_returns) if episode_returns else 0.0
                    _LOG.info(f"  Collected {collected_episodes}/{num_episodes}: steps={step}, "
                          f"terminals={n_terms}, truncs={n_truncs}, mean_return={mean_ret:.3f}, "
                          f"skipped={skipped_episodes}")

            ep_idx += 1

    finally:
        # Restore normalizer state.
        def _restore_norm(norm, prev, name: str):
            if prev is None:
                return
            if prev["method"] == "api":
                if not prev.get("had__frozen", False):
                    raise RuntimeError(
                        f"{name}: freeze()/unfreeze() exists but no _frozen flag to restore prior state."
                    )
                was_frozen = bool(prev["_frozen"])
                if was_frozen:
                    norm.freeze()
                else:
                    norm.unfreeze()
                norm._frozen = was_frozen
            elif prev["method"] == "attr":
                norm._frozen = prev["_frozen"]
            else:
                raise RuntimeError(f"{name}: unknown restore method {prev['method']}")

        if hasattr(episode_runner, "actor_state_normalizer"):
            _restore_norm(episode_runner.actor_state_normalizer, actor_norm_prev, "actor_state_normalizer")

        # Restore policy training mode.
        if hasattr(agent, 'policy'):
            agent.policy.train(policy_prev_training)

    # Stack into tensors.
    if not all_o1_embeddings:
        raise RuntimeError("No data collected - all episodes may have failed")

    o1_embeddings = torch.cat(all_o1_embeddings, dim=0)
    o2_embeddings = torch.cat(all_o2_embeddings, dim=0)
    geometries = torch.cat(all_geometries, dim=0)
    returns = torch.tensor(all_returns, dtype=torch.float32)
    rewards = torch.tensor(all_rewards, dtype=torch.float32)
    episode_ids = torch.tensor(all_episode_ids, dtype=torch.long)
    is_train = torch.tensor(all_is_train, dtype=torch.bool)

    if verbose:
        n_train = is_train.sum().item()
        n_val = (~is_train).sum().item()
        _LOG.info(f"\nDataset collected: {o1_embeddings.shape[0]} transitions")
        _LOG.info(f"  o1: {o1_embeddings.shape}, o2: {o2_embeddings.shape}, geom: {geometries.shape}")
        _LOG.info(f"  Train: {n_train}, Val: {n_val}")
        _LOG.info(f"  Returns: mean={returns.mean():.3f}, std={returns.std():.3f}")

    # Dataset diagnostic statistics.
    if verbose:
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"DATASET DIAGNOSTIC SUMMARY")
        _LOG.info(f"{'='*60}")

        # Check embedding variance.
        o1_std = o1_embeddings.std(dim=0)
        o2_std = o2_embeddings.std(dim=0)
        geom_std = geometries.std(dim=0)

        _LOG.info(f"  o1_embeddings: shape={o1_embeddings.shape}")
        _LOG.info(f"    mean={o1_embeddings.mean():.4f}, std={o1_embeddings.std():.4f}")
        _LOG.info(f"    per-dim std: min={o1_std.min():.6f}, max={o1_std.max():.4f}, mean={o1_std.mean():.4f}")
        _LOG.info(f"    dead dims (std<1e-6): {(o1_std < 1e-6).sum().item()}/{o1_std.numel()}")

        _LOG.info(f"  o2_embeddings: shape={o2_embeddings.shape}")
        _LOG.info(f"    mean={o2_embeddings.mean():.4f}, std={o2_embeddings.std():.4f}")
        _LOG.info(f"    per-dim std: min={o2_std.min():.6f}, max={o2_std.max():.4f}, mean={o2_std.mean():.4f}")
        _LOG.info(f"    dead dims (std<1e-6): {(o2_std < 1e-6).sum().item()}/{o2_std.numel()}")

        _LOG.info(f"  geometries: shape={geometries.shape}")
        _LOG.info(f"    per-dim stats:")
        for i in range(geometries.shape[1]):
            g = geometries[:, i]
            _LOG.info(f"      dim{i}: mean={g.mean():.4f}, std={g.std():.4f}, min={g.min():.4f}, max={g.max():.4f}")

        # Check return variance.
        _LOG.info(f"  returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
        _LOG.info(f"    unique values: {len(returns.unique())}/{len(returns)}")

        # Check value-head expected input dims.
        if hasattr(trainer.agent.policy, 'critic_embedder'):
            ce = trainer.agent.policy.critic_embedder
            expected_o_dim = getattr(ce, 'o_irreps', None)
            expected_geom_dim = getattr(ce, 'geom_dim', None)
            _LOG.info(f"  CriticEmbedding expects: o_irreps={expected_o_dim}, geom_dim={expected_geom_dim}")
            _LOG.info(f"  Actual: o1={o1_embeddings.shape[1]}, o2={o2_embeddings.shape[1]}, geom={geometries.shape[1]}")

    return {
        'o1_embeddings': o1_embeddings,
        'o2_embeddings': o2_embeddings,
        'geometries': geometries,
        'returns': returns,
        'rewards': rewards,
        'episode_ids': episode_ids,
        'is_train': is_train
    }
