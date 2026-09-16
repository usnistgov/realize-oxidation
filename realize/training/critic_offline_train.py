"""Offline critic training + explained-variance helpers:
compute_baseline_ev, train_critic_offline, compute_short_horizon_ev.
"""
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from realize.config.paths import get_embedding_checkpoint_path
import logging
_LOG = logging.getLogger(__name__)


def compute_baseline_ev(train_returns, val_returns):
    """EV for the constant predictor (mean of training returns); ~0 by
    construction. Wildly negative means the EV calculation is broken."""
    mean_pred = train_returns.mean().item()

    val_targets = val_returns
    val_preds = torch.full_like(val_targets, mean_pred)

    targets_var = val_targets.var(unbiased=False).item()

    if targets_var < 1e-6:
        return float('nan'), mean_pred

    residual_var = (val_targets - val_preds).var(unbiased=False).item()
    baseline_ev = 1.0 - residual_var / (targets_var + 1e-8)

    return baseline_ev, mean_pred


def train_critic_offline(
    trainer,
    dataset: Dict[str, torch.Tensor],
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train critic (CriticEmbedding + value_mlp) on a fixed dataset with a
    train/val split. Saves and restores value-head weights so the test does not
    contaminate the policy."""
    if verbose:
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"OFFLINE CRITIC TRAINING: {num_epochs} epochs")
        _LOG.info(f"{'='*60}")

    device = trainer.agent.device
    policy = trainer.agent.policy

    # Find value-head layers.
    value_layer_names = ['critic_embedder', 'value_mlp']
    value_layers = {}

    for name in value_layer_names:
        if hasattr(policy, name):
            value_layers[name] = getattr(policy, name)

    if not value_layers:
        raise RuntimeError(
            f"Cannot find value head layers on policy. "
            f"Expected: {value_layer_names}. "
            f"Available: {[n for n, _ in policy.named_children()]}"
        )

    if verbose:
        _LOG.info(f"  Found value layers: {list(value_layers.keys())}")

    # Save value-head weights for restoration.
    original_state_dicts = {
        name: copy.deepcopy(layer.state_dict())
        for name, layer in value_layers.items()
    }

    try:
        # Unpack critic raw inputs.
        o1_embs = dataset['o1_embeddings'].to(device)
        o2_embs = dataset['o2_embeddings'].to(device)
        geoms = dataset['geometries'].to(device)
        returns = dataset['returns'].to(device)
        is_train = dataset['is_train'].to(device)

        train_o1 = o1_embs[is_train]
        train_o2 = o2_embs[is_train]
        train_geom = geoms[is_train]
        train_returns_raw = returns[is_train]

        val_o1 = o1_embs[~is_train]
        val_o2 = o2_embs[~is_train]
        val_geom = geoms[~is_train]
        val_returns_raw = returns[~is_train]

        n_train = train_o1.shape[0]
        n_val = val_o1.shape[0]

        # Normalize returns by train-set mean/std.
        returns_mean = train_returns_raw.mean()
        returns_std = train_returns_raw.std(unbiased=False) + 1e-8

        train_returns = (train_returns_raw - returns_mean) / returns_std
        val_returns_normalized = (val_returns_raw - returns_mean) / returns_std

        if verbose:
            _LOG.info(f"  Return normalization: mean={returns_mean.item():.4f}, std={returns_std.item():.4f}")

        def _stats(x, name):
            x = x.detach().float().cpu()
            _LOG.info(f"  {name}: n={x.numel()} mean={x.mean():.4f} std={x.std(unbiased=False):.6f} "
                  f"min={x.min():.4f} max={x.max():.4f}")

        if verbose:
            _stats(train_returns, "train_returns")
            _stats(val_returns_normalized, "val_returns_normalized")

            val_std = val_returns_normalized.std(unbiased=False).item()
            if val_std < 0.01:
                _LOG.info(f"  WARNING WARNING: val_returns has very low variance ({val_std:.6f}). EV metrics will be unreliable!")

        baseline_ev, mean_return = compute_baseline_ev(train_returns, val_returns_normalized)
        if verbose:
            _LOG.info(f"  Baseline (constant predictor) val_EV: {baseline_ev:.4f}")
            _LOG.info(f"  Mean return: {mean_return:.4f}")
            if baseline_ev < -0.5:
                _LOG.info(f"  WARNING WARNING: Baseline EV is very negative. Check val split or EV calculation.")

        # Fresh optimizer over value-head params only.
        value_params = []
        for layer in value_layers.values():
            value_params.extend(layer.parameters())
        optimizer = torch.optim.Adam(value_params, lr=learning_rate)

        metrics = {
            'train_mse': [],
            'train_ev': [],
            'val_mse': [],
            'val_ev': [],
            'val_correlation': [],
            'val_values_std': [],
            'returns_std': float(returns.std().item())
        }

        def compute_metrics(o1, o2, gm, targets_normalized, ret_mean, ret_std, targets_raw=None):
            """MSE, EV, and correlation for a dataset, using policy._compute_value for predictions."""
            was_training = policy.training
            policy.eval()
            try:
                with torch.no_grad():
                    values_normalized = policy._compute_value(o1, o2, gm)
                    values_normalized = values_normalized.reshape(-1)
                    targets_normalized = targets_normalized.reshape(-1)

                    if values_normalized.shape != targets_normalized.shape:
                        raise RuntimeError(
                            f"Offline critic metrics: values={values_normalized.shape}, "
                            f"targets={targets_normalized.shape}"
                        )

                mse_normalized = F.mse_loss(values_normalized, targets_normalized).item()

                values_raw = values_normalized * ret_std + ret_mean
                if targets_raw is None:
                    targets_raw = targets_normalized * ret_std + ret_mean
                else:
                    targets_raw = targets_raw.reshape(-1)

                targets_var = targets_raw.var(unbiased=False).item()

                if targets_var < 1e-6:
                    extra_stats = {
                        'values_mean': values_raw.mean().item(),
                        'values_std': values_raw.std(unbiased=False).item(),
                        'targets_mean': targets_raw.mean().item(),
                        'targets_std': 0.0,
                        'slope': float('nan'),
                        'intercept': float('nan'),
                    }
                    return mse_normalized, float('nan'), float('nan'), values_raw.std(unbiased=False).item(), extra_stats

                residual_var = (targets_raw - values_raw).var(unbiased=False).item()
                ev = 1.0 - residual_var / (targets_var + 1e-8)

                values_centered = values_raw - values_raw.mean()
                targets_centered = targets_raw - targets_raw.mean()
                corr = (values_centered * targets_centered).mean() / (
                    values_centered.std(unbiased=False) * targets_centered.std(unbiased=False) + 1e-8
                )

                V = values_raw.cpu().numpy()
                G = targets_raw.cpu().numpy()
                V_mean, G_mean = V.mean(), G.mean()
                V_var = ((V - V_mean) ** 2).mean()
                if V_var > 1e-8:
                    slope = ((V - V_mean) * (G - G_mean)).mean() / V_var
                    intercept = G_mean - slope * V_mean
                else:
                    slope, intercept = float('nan'), float('nan')

                values_std = values_raw.std(unbiased=False).item()
                targets_std = targets_raw.std(unbiased=False).item()

                extra_stats = {
                    'values_mean': values_raw.mean().item(),
                    'values_std': values_std,
                    'targets_mean': targets_raw.mean().item(),
                    'targets_std': targets_std if targets_var > 1e-6 else 0.0,
                    'slope': float(slope),
                    'intercept': float(intercept),
                }

                return mse_normalized, ev, corr.item(), values_std, extra_stats

            finally:
                policy.train(was_training)

        for epoch in range(num_epochs):
            perm = torch.randperm(n_train)
            train_o1_s = train_o1[perm]
            train_o2_s = train_o2[perm]
            train_geom_s = train_geom[perm]
            train_returns_shuffled = train_returns[perm]

            for i in range(0, n_train, batch_size):
                b_o1 = train_o1_s[i:i+batch_size]
                b_o2 = train_o2_s[i:i+batch_size]
                b_gm = train_geom_s[i:i+batch_size]
                batch_returns = train_returns_shuffled[i:i+batch_size]

                optimizer.zero_grad()

                # Forward through CriticEmbedding + value_mlp.
                values = policy._compute_value(b_o1, b_o2, b_gm)
                values = values.reshape(-1)
                batch_returns = batch_returns.reshape(-1)

                if values.shape != batch_returns.shape:
                    raise RuntimeError(
                        f"Offline critic minibatch: values={values.shape}, "
                        f"targets={batch_returns.shape}"
                    )

                loss = F.mse_loss(values, batch_returns)
                loss.backward()
                optimizer.step()

            train_mse, train_ev, _, _, train_extra = compute_metrics(
                train_o1, train_o2, train_geom, train_returns,
                returns_mean, returns_std, targets_raw=train_returns_raw
            )
            val_mse, val_ev, val_corr, val_std, val_extra = compute_metrics(
                val_o1, val_o2, val_geom, val_returns_normalized,
                returns_mean, returns_std, targets_raw=val_returns_raw
            )

            metrics['train_mse'].append(train_mse)
            metrics['train_ev'].append(train_ev)
            metrics['val_mse'].append(val_mse)
            metrics['val_ev'].append(val_ev)
            metrics['val_correlation'].append(val_corr)
            metrics['val_values_std'].append(val_std)

            if verbose:
                scale_gap = val_extra.get('targets_std', 1.0) / (val_std + 1e-8) if val_extra else float('nan')
                slope = val_extra.get('slope', float('nan')) if val_extra else float('nan')
                intercept = val_extra.get('intercept', float('nan')) if val_extra else float('nan')

                if not np.isfinite(val_ev):
                    status = "DEGENERATE (val targets have no variance)"
                elif val_ev > 0.1:
                    status = "LEARNING"
                elif val_ev > 0:
                    status = "LOW"
                else:
                    status = "NEGATIVE"

                _LOG.info(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                      f"train_EV={train_ev:.4f}, val_EV={val_ev:.4f} [{status}], "
                      f"val_corr={val_corr:.4f}, scale_gap={scale_gap:.1f}x, "
                      f"fit: G~={slope:.2f}*V+{intercept:.2f}")

        if verbose:
            _LOG.info(f"\n{'='*60}")
            _LOG.info(f"OFFLINE CRITIC TEST RESULTS (VALIDATION SET)")
            _LOG.info(f"{'='*60}")

            valid_evs = [ev for ev in metrics['val_ev'] if np.isfinite(ev)]
            if valid_evs:
                best_epoch = int(np.nanargmax(metrics['val_ev']))
                best_val_ev = metrics['val_ev'][best_epoch]
                best_val_corr = metrics['val_correlation'][best_epoch]
            else:
                best_epoch = -1
                best_val_ev = float('nan')
                best_val_corr = float('nan')

            final_val_ev = metrics['val_ev'][-1]
            final_val_corr = metrics['val_correlation'][-1]

            _LOG.info(f"  Initial val EV: {metrics['val_ev'][0]:.4f}")
            _LOG.info(f"  Best val EV:    {best_val_ev:.4f} (epoch {best_epoch + 1})")
            _LOG.info(f"  Final val EV:   {final_val_ev:.4f}")
            _LOG.info(f"  Best val corr:  {best_val_corr:.4f}")
            _LOG.info(f"  Final val corr: {final_val_corr:.4f}")

            if not np.isfinite(best_val_ev) or not np.isfinite(best_val_corr):
                _LOG.info(f"\n  WARNING VERDICT: INCONCLUSIVE - validation set is degenerate (near-zero variance).")
                _LOG.info(f"     Run with more episodes or check reward distribution.")
            elif best_val_ev > 0.2 and best_val_corr > 0.4:
                _LOG.info(f"\n  OK VERDICT: Critic CAN learn. Issue is online nonstationarity/PPO coupling.")
            elif best_val_ev > 0.05 or best_val_corr > 0.25:
                _LOG.info(f"\n  WARNING VERDICT: Critic learns weakly. Signal exists but noisy.")
            else:
                _LOG.info(f"\n  FAIL VERDICT: Critic CANNOT learn. Returns not predictable from state.")

        metrics['best_epoch'] = best_epoch
        metrics['best_val_ev'] = best_val_ev
        metrics['best_val_corr'] = best_val_corr

        return metrics

    finally:
        # Restore value-head weights so offline scoring leaves the policy untouched.
        for name, layer in value_layers.items():
            layer.load_state_dict(original_state_dicts[name])

        if verbose:
            _LOG.info(f"\n  [Restored value head weights - no training contamination]")


def compute_short_horizon_ev(dataset: Dict[str, torch.Tensor], gamma: float = 0.99,
                              k_steps: List[int] = [1, 5, 10], verbose: bool = True) -> Dict[str, float]:
    """Explained variance at several horizon targets, to diagnose credit assignment.

    Uses concatenated [o1, o2, geometry] as the feature vector for linear predictability.
    """
    # Flat feature vector for linear analysis.
    o1 = dataset['o1_embeddings']
    o2 = dataset['o2_embeddings']
    geom = dataset['geometries']
    features = torch.cat([o1, o2, geom], dim=-1)  # [N, D_o1 + D_o2 + D_geom]

    returns = dataset['returns']
    rewards = dataset['rewards']
    episode_ids = dataset['episode_ids']
    is_train = dataset['is_train']

    n_total = features.shape[0]

    def compute_linear_ev(states, targets, is_train_mask):
        """EV via simple correlation, a proxy for linear predictability."""
        train_states = states[is_train_mask]
        train_targets = targets[is_train_mask]
        val_states = states[~is_train_mask]
        val_targets = targets[~is_train_mask]

        if val_targets.numel() < 2:
            return float('nan'), float('nan')

        val_var = val_targets.var(unbiased=False).item()
        if val_var < 1e-8:
            return float('nan'), float('nan')

        state_mean = val_states.mean(dim=1)
        if state_mean.std() < 1e-8:
            return float('nan'), float('nan')

        corr = torch.corrcoef(torch.stack([state_mean, val_targets]))[0, 1].item()
        if not np.isfinite(corr):
            corr = 0.0

        r_squared = corr ** 2

        return r_squared, corr

    results = {}

    if verbose:
        _LOG.info(f"\n{'='*60}")
        _LOG.info("P1: SHORT-HORIZON PREDICTABILITY TEST")
        _LOG.info(f"{'='*60}")
        _LOG.info(f"  Dataset: {n_total} transitions, {is_train.sum().item()} train, {(~is_train).sum().item()} val")
        _LOG.info(f"  Feature dim: {features.shape[1]} (o1={o1.shape[1]} + o2={o2.shape[1]} + geom={geom.shape[1]})")

    # 1. Immediate reward (k=1)
    ev_reward, corr_reward = compute_linear_ev(features, rewards, is_train)
    results['ev_reward'] = ev_reward
    results['corr_reward'] = corr_reward
    if verbose:
        _LOG.info(f"  EV(immediate reward):  R^2={ev_reward:.4f}, corr={corr_reward:.4f}")

    # 2. k-step returns
    for k in k_steps:
        k_returns = torch.zeros_like(returns)
        unique_episodes = episode_ids.unique()

        for ep_id in unique_episodes:
            ep_mask = episode_ids == ep_id
            ep_indices = torch.where(ep_mask)[0]
            ep_rewards = rewards[ep_mask]
            ep_len = len(ep_rewards)

            for i, idx in enumerate(ep_indices):
                k_return = 0.0
                discount = 1.0
                for j in range(min(k, ep_len - i)):
                    k_return += discount * ep_rewards[i + j].item()
                    discount *= gamma
                k_returns[idx] = k_return

        ev_k, corr_k = compute_linear_ev(features, k_returns, is_train)
        results[f'ev_{k}step'] = ev_k
        results[f'corr_{k}step'] = corr_k
        if verbose:
            _LOG.info(f"  EV({k:2d}-step return):    R^2={ev_k:.4f}, corr={corr_k:.4f}")

    # 3. Full returns
    ev_full, corr_full = compute_linear_ev(features, returns, is_train)
    results['ev_full'] = ev_full
    results['corr_full'] = corr_full
    if verbose:
        _LOG.info(f"  EV(full return):       R^2={ev_full:.4f}, corr={corr_full:.4f}")

    if verbose:
        _LOG.info(f"\n  INTERPRETATION:")
        if ev_reward > 0.1 and ev_full < 0.05:
            _LOG.info(f"  -> Immediate reward predictable but full return not -> CREDIT ASSIGNMENT issue")
            _LOG.info(f"    (sparse terminal rewards dominate; consider reward shaping)")
        elif ev_reward < 0.05 and ev_full < 0.05:
            _LOG.info(f"  -> Neither reward nor return predictable -> REPRESENTATION issue")
            _LOG.info(f"    (critic state may not contain enough information)")
        elif ev_full > 0.1:
            _LOG.info(f"  -> Full return is predictable -> Signal exists, check optimization")
        else:
            _LOG.info(f"  -> Weak signal across all horizons -> May need more data or better features")

    return results
