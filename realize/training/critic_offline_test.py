import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from realize.config.paths import get_embedding_checkpoint_path

# Re-export dataset-building and training helpers from sibling modules.
from realize.training.critic_offline_dataset import (
    compute_returns_with_bootstrap,
    collect_offline_dataset,
)
from realize.training.critic_offline_train import (
    compute_baseline_ev,
    train_critic_offline,
    compute_short_horizon_ev,
)
import logging
_LOG = logging.getLogger(__name__)


def run_offline_critic_test(
    trainer,
    num_episodes: int = 100,
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    verbose: bool = True
) -> Dict:
    """
    Full offline critic test pipeline.

    Returns:
        Dictionary with:
        - 'dataset_stats': info about collected data
        - 'training_metrics': per-epoch metrics (train and val)
        - 'verdict': string summarizing result
    """
    # 1. Collect dataset
    dataset = collect_offline_dataset(trainer, num_episodes, gamma, verbose)
    # Save the collected dataset to disk for post-hoc analysis.
    try:
        import os as _os_save
        # '..','..' climbs from training/ to repo root so output lands in results/.
        _save_dir = _os_save.path.join(_os_save.path.dirname(__file__), '..', '..', 'results')
        _save_dir = _os_save.path.abspath(_save_dir)
        _os_save.makedirs(_save_dir, exist_ok=True)
        _save_path = _os_save.path.join(_save_dir, 'offline_test_dataset.pt')
        _save_payload = {}
        for _k, _v in dataset.items():
            if hasattr(_v, 'detach'):
                _save_payload[_k] = _v.detach().cpu()
            else:
                _save_payload[_k] = _v
        torch.save(_save_payload, _save_path)
        _LOG.info(f"\n[SAVE_DATASET] Saved {len(_save_payload)} keys to {_save_path}")
        for _k, _v in _save_payload.items():
            if hasattr(_v, 'shape'):
                _LOG.info(f"  {_k}: shape={list(_v.shape)} dtype={_v.dtype}")
            else:
                _LOG.info(f"  {_k}: type={type(_v).__name__}")
    except Exception as _save_err:
        _LOG.info(f"\n[SAVE_DATASET] WARN: save failed: {_save_err}")
        # Don't propagate -- saving is auxiliary

    n_train = dataset['is_train'].sum().item()
    n_val = (~dataset['is_train']).sum().item()

    dataset_stats = {
        'n_transitions': dataset['o1_embeddings'].shape[0],
        'n_train': n_train,
        'n_val': n_val,
        'n_episodes': num_episodes,
        'returns_mean': float(dataset['returns'].mean()),
        'returns_std': float(dataset['returns'].std())
    }

    # 2. Short-horizon predictability test (P1)
    if 'rewards' in dataset:
        short_horizon_results = compute_short_horizon_ev(dataset, gamma, verbose=verbose)
    else:
        short_horizon_results = {}

    # 3. Train critic (weights restored after)
    metrics = train_critic_offline(
        trainer, dataset, num_epochs, batch_size, learning_rate, verbose
    )

    # 4. Determine verdict
    best_val_ev = metrics.get('best_val_ev', metrics['val_ev'][-1])
    best_val_corr = metrics.get('best_val_corr', metrics['val_correlation'][-1])

    if not np.isfinite(best_val_ev) or not np.isfinite(best_val_corr):
        verdict = "INCONCLUSIVE: Validation set is degenerate. Run with more episodes."
    elif best_val_ev > 0.2 and best_val_corr > 0.4:
        verdict = "PASS: Critic can learn. Online nonstationarity is the issue."
    elif best_val_ev > 0.05 or best_val_corr > 0.25:
        verdict = "WEAK_PASS: Critic learns weakly but signal exists."
    else:
        verdict = "FAIL: Critic cannot learn. Returns not predictable from state representation."

    return {
        'dataset_stats': dataset_stats,
        'training_metrics': metrics,
        'short_horizon': short_horizon_results,
        'verdict': verdict
    }
