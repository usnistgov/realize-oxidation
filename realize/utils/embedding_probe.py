#!/usr/bin/env python3
"""Embedding information probe.

Tests whether the policy embedding predicts episode outcome (classification)
and useful action directions (regression).

Wiring (see INJECTION_GUIDE at the bottom for the full version): call
start_episode() at episode start, collect_step(...) after each action, and
end_episode(dissociated) at episode end; call analyze() after ~100 episodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Optional
import logging
_LOG = logging.getLogger(__name__)


class EmbeddingProbe:
    def __init__(self, save_path: str = "probe_data.pt"):
        self.save_path = save_path
        self._current_episode = {
            'raw_states': [],      # pre-normalization [480]
            'norm_states': [],     # post-normalization [480]
            'directions': [],      # sampled direction [3]
            'o2_positions': [],    # O2 COM z-coordinate
        }
        self.episodes = []
        self.n_episodes = 0

    def start_episode(self):
        """Call at the start of each episode."""
        self._current_episode = {
            'raw_states': [],
            'norm_states': [],
            'directions': [],
            'o2_positions': [],
        }

    def collect_step(self,
                     raw_state: torch.Tensor,    # [1, 480] or [480] pre-norm SYM
                     norm_state: torch.Tensor,    # [1, 480] or [480] post-norm SYM
                     direction: torch.Tensor,     # [1, 3] or [3] sampled world direction
                     o2_positions: Optional[torch.Tensor] = None,  # [2, 3]
                     ):
        """Call at each step after action is sampled."""
        with torch.no_grad():
            raw = raw_state.detach().cpu().flatten()[:480]
            norm = norm_state.detach().cpu().flatten()[:480]
            d = direction.detach().cpu().flatten()[:3]

            self._current_episode['raw_states'].append(raw)
            self._current_episode['norm_states'].append(norm)
            self._current_episode['directions'].append(d)

            if o2_positions is not None:
                com_z = o2_positions.detach().cpu().float().mean(dim=0)[2].item()
            else:
                com_z = 0.0
            self._current_episode['o2_positions'].append(com_z)

    def collect_first_layer(self, first_layer_sym: torch.Tensor):
        """Collect first-layer SYM embedding for directional signal test."""
        with torch.no_grad():
            fl = first_layer_sym.detach().cpu().flatten()[:480]
            if 'first_layer' not in self._current_episode:
                self._current_episode['first_layer'] = []
            self._current_episode['first_layer'].append(fl)

    def end_episode(self, dissociated: bool):
        """Call at the end of each episode."""
        if len(self._current_episode['raw_states']) == 0:
            return

        ep = {
            'raw_states': torch.stack(self._current_episode['raw_states']),
            'norm_states': torch.stack(self._current_episode['norm_states']),
            'directions': torch.stack(self._current_episode['directions']),
            'o2_com_z': self._current_episode['o2_positions'],
            'dissociated': dissociated,
            'n_steps': len(self._current_episode['raw_states']),
        }
        if 'first_layer' in self._current_episode and self._current_episode['first_layer']:
            ep['first_layer'] = torch.stack(self._current_episode['first_layer'])                
        self.episodes.append(ep)
        self.n_episodes += 1

        if self.n_episodes % 20 == 0:
            _LOG.debug(f"[PROBE] Collected {self.n_episodes} episodes "
                  f"({sum(1 for e in self.episodes if e['dissociated'])}/{self.n_episodes} dissociated)")

    def analyze(self):
        """Run all analyses. Call after ~100+ episodes."""
        if self.n_episodes < 30:
            _LOG.debug(f"[PROBE] Only {self.n_episodes} episodes. Need 30+. Keep collecting.")
            return

        n_dissoc = sum(1 for e in self.episodes if e['dissociated'])
        n_fail = self.n_episodes - n_dissoc
        _LOG.debug(f"\n{'='*70}")
        _LOG.debug(f"EMBEDDING INFORMATION PROBE -- {self.n_episodes} episodes "
              f"({n_dissoc} dissoc, {n_fail} fail)")
        _LOG.debug(f"{'='*70}")

        if n_dissoc < 5 or n_fail < 5:
            _LOG.debug("[PROBE] Need at least 5 of each outcome. Skipping.")
            return

        self._test_outcome_prediction()
        self._test_direction_prediction()
        self._test_embedding_separability()
        self._test_l_block_importance()
        self._test_first_layer_direction()        

    def _test_outcome_prediction(self):
        """Test 1: Can a linear model predict dissociation from step-0 embedding?"""
        _LOG.debug(f"\n--- TEST 1: Outcome Prediction (step-0 embedding -> dissociation) ---")

        for state_key, label in [('raw_states', 'RAW'), ('norm_states', 'NORMALIZED')]:
            X = []
            y = []
            for ep in self.episodes:
                X.append(ep[state_key][0])  # step-0 embedding
                y.append(1.0 if ep['dissociated'] else 0.0)

            X = torch.stack(X)  # [N, 480]
            y = torch.tensor(y)  # [N]

            acc, auc = self._train_classifier(X, y)
            _LOG.debug(f"  {label}: accuracy={acc:.1%} | AUC={auc:.3f} "
                  f"{'OK PREDICTIVE' if auc > 0.6 else 'FAIL NOT PREDICTIVE'}")

        # Also test with step-0 O2 COM z alone (baseline: does height predict outcome?)
        X_z = torch.tensor([[ep['o2_com_z'][0]] for ep in self.episodes])
        y = torch.tensor([1.0 if ep['dissociated'] else 0.0 for ep in self.episodes])
        acc_z, auc_z = self._train_classifier(X_z, y)
        _LOG.debug(f"  O2_z alone: accuracy={acc_z:.1%} | AUC={auc_z:.3f} "
              f"(baseline -- does height predict outcome?)")

    def _test_direction_prediction(self):
        """Test 2: Can a linear model predict 'good directions' from embedding?"""
        _LOG.debug(f"\n--- TEST 2: Direction Prediction (embedding -> successful direction) ---")

        # Collect (embedding, direction) from successful episodes
        X_succ, d_succ = [], []
        X_fail, d_fail = [], []

        for ep in self.episodes:
            for t in range(ep['n_steps']):
                if ep['dissociated']:
                    X_succ.append(ep['raw_states'][t])
                    d_succ.append(ep['directions'][t])
                else:
                    X_fail.append(ep['raw_states'][t])
                    d_fail.append(ep['directions'][t])

        if len(X_succ) < 10:
            _LOG.debug("  Too few successful steps. Skipping.")
            return

        X_succ = torch.stack(X_succ)
        d_succ = torch.stack(d_succ)
        X_fail = torch.stack(X_fail)
        d_fail = torch.stack(d_fail)

        # Train linear map: embedding -> direction (on successful episodes only)
        cos_succ = self._train_direction_regressor(X_succ, d_succ, "Successful episodes")
        cos_fail = self._train_direction_regressor(X_fail, d_fail, "Failed episodes")

        _LOG.debug(f"\n  Interpretation:")
        if cos_succ > 0.3:
            _LOG.debug(f"  OK Embedding predicts successful directions (cos={cos_succ:.3f})")
            _LOG.debug(f"    -> The representation ENCODES actionable directional information")
        else:
            _LOG.debug(f"  FAIL Embedding does NOT predict successful directions (cos={cos_succ:.3f})")
            _LOG.debug(f"    -> Either the embedding lacks directional info, or successful")
            _LOG.debug(f"       directions vary too much across placements")

        # Compare mean direction z between successful and failed
        z_succ = d_succ[:, 2].mean().item()
        z_fail = d_fail[:, 2].mean().item()
        _LOG.debug(f"\n  Mean sampled direction z: dissociated={z_succ:.3f} | failed={z_fail:.3f}")
        if z_succ < z_fail - 0.1:
            _LOG.debug(f"  OK Successful episodes sample MORE DOWNWARD (Deltaz={z_succ-z_fail:.3f})")
        elif abs(z_succ - z_fail) < 0.1:
            _LOG.debug(f"  FAIL No z-direction difference between success/failure -> directions are random")
        else:
            _LOG.debug(f"  ? Successful episodes sample MORE UPWARD (unexpected)")

    def _test_embedding_separability(self):
        """Test 3: Are embeddings from successful vs failed episodes distinguishable?"""
        _LOG.debug(f"\n--- TEST 3: Embedding Separability (success vs failure clusters) ---")

        for state_key, label in [('raw_states', 'RAW'), ('norm_states', 'NORMALIZED')]:
            emb_succ = []
            emb_fail = []
            for ep in self.episodes:
                mean_emb = ep[state_key].mean(dim=0)  # episode-average embedding
                if ep['dissociated']:
                    emb_succ.append(mean_emb)
                else:
                    emb_fail.append(mean_emb)

            succ = torch.stack(emb_succ)  # [N_succ, 480]
            fail = torch.stack(emb_fail)  # [N_fail, 480]

            # Compute centroid distance
            centroid_succ = succ.mean(dim=0)
            centroid_fail = fail.mean(dim=0)
            centroid_dist = (centroid_succ - centroid_fail).norm().item()

            # Within-class spread
            spread_succ = (succ - centroid_succ).norm(dim=-1).mean().item()
            spread_fail = (fail - centroid_fail).norm(dim=-1).mean().item()
            avg_spread = (spread_succ + spread_fail) / 2

            # Fisher criterion: between / within
            fisher = centroid_dist / (avg_spread + 1e-8)

            _LOG.debug(f"  {label}: centroid_dist={centroid_dist:.4f} | "
                  f"avg_spread={avg_spread:.4f} | Fisher={fisher:.4f} "
                  f"{'OK SEPARABLE' if fisher > 0.3 else 'FAIL OVERLAPPING'}")

    def _test_l_block_importance(self):
        """Test 4: Which irreps blocks (l=0, l=1, l=2) are most predictive?"""
        _LOG.debug(f"\n--- TEST 4: Per-Block Outcome Prediction ---")

        blocks = {
            'l=0 (scalars 0:128)':   (0, 128),
            'l=1 (vectors 128:320)':  (128, 320),
            'l=2 (tensors 320:480)':  (320, 480),
            'l=0+l=1 (0:320)':        (0, 320),
            'full (0:480)':            (0, 480),
        }

        y = torch.tensor([1.0 if ep['dissociated'] else 0.0 for ep in self.episodes])

        for name, (start, end) in blocks.items():
            X = torch.stack([ep['raw_states'][0, start:end] for ep in self.episodes])
            acc, auc = self._train_classifier(X, y)
            _LOG.debug(f"  {name:30s}: AUC={auc:.3f} "
                  f"{'OK' if auc > 0.6 else '*'}")

        _LOG.debug(f"\n  If l=0 AUC >> l=1 AUC: scalars (chemistry) predict outcome, not direction")
        _LOG.debug(f"  If l=1 AUC >> l=0 AUC: directional info predicts outcome")
        _LOG.debug(f"  If both low: step-0 embedding doesn't predict outcome (later steps matter)")

    def _train_classifier(self, X: torch.Tensor, y: torch.Tensor,
                          n_epochs: int = 300) -> tuple:
        """Train logistic regression, return (accuracy, AUC)."""
        N = X.shape[0]
        n_train = max(int(0.7 * N), N - 10)

        # Shuffle
        perm = torch.randperm(N)
        X, y = X[perm], y[perm]

        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        if len(y_te) < 3:
            return 0.5, 0.5

        # Normalize
        X_mean = X_tr.mean(dim=0)
        X_std = X_tr.std(dim=0).clamp(min=1e-6)
        X_tr = (X_tr - X_mean) / X_std
        X_te = (X_te - X_mean) / X_std

        model = nn.Linear(X.shape[-1], 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for _ in range(n_epochs):
            logits = model(X_tr).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            test_logits = model(X_te).squeeze(-1)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).float()
            accuracy = (test_preds == y_te).float().mean().item()

            # Simple AUC approximation
            auc = self._compute_auc(test_probs.numpy(), y_te.numpy())

        return accuracy, auc

    def _train_direction_regressor(self, X: torch.Tensor, d: torch.Tensor,
                                    label: str, n_epochs: int = 500) -> float:
        """Train linear regressor: X -> direction, return test cosine similarity."""
        N = X.shape[0]
        n_train = max(int(0.7 * N), N - 20)

        perm = torch.randperm(N)
        X, d = X[perm], d[perm]

        X_tr, X_te = X[:n_train], X[n_train:]
        d_tr, d_te = d[:n_train], d[n_train:]

        X_mean = X_tr.mean(dim=0)
        X_std = X_tr.std(dim=0).clamp(min=1e-6)
        X_tr = (X_tr - X_mean) / X_std
        X_te = (X_te - X_mean) / X_std

        model = nn.Linear(X.shape[-1], 3)
        nn.init.normal_(model.weight, std=0.01)
        nn.init.zeros_(model.bias)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)

        best_cos = -1.0
        for epoch in range(n_epochs):
            pred = F.normalize(model(X_tr), dim=-1)
            cos = (pred * d_tr).sum(dim=-1)
            loss = -cos.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    test_pred = F.normalize(model(X_te), dim=-1)
                    test_cos = (test_pred * d_te).sum(dim=-1).mean().item()
                    if test_cos > best_cos:
                        best_cos = test_cos

        _LOG.debug(f"  {label}: cosine={best_cos:.3f} "
              f"({'predictive' if best_cos > 0.3 else 'not predictive'})")
        return best_cos

    def _test_first_layer_direction(self):
        """Test 5: Does the first-layer SYM l=1 encode substrate direction?"""
        _LOG.debug(f"\n--- TEST 5: First-Layer l=1 Directional Signal ---")

        has_first = any('first_layer' in ep for ep in self.episodes)
        if not has_first:
            _LOG.debug("  No first-layer data collected. Skipping.")
            return

        # Check if first-layer l=1 has consistent direction
        # and if it predicts outcome better than last-layer l=1
        l1_start, l1_end = 128, 320  # l=1 block

        z_components = []
        l1_norms = []
        outcomes = []
        for ep in self.episodes:
            if 'first_layer' not in ep:
                continue
            fl = ep['first_layer'][0]  # step-0
            l1_block = fl[l1_start:l1_end].reshape(64, 3)
            l1_mean = l1_block.mean(dim=0)  # [3]
            l1_norm = l1_mean.norm().item()
            if l1_norm > 1e-6:
                z_dir = (l1_mean[2] / l1_mean.norm()).item()
            else:
                z_dir = 0.0
            z_components.append(z_dir)
            l1_norms.append(l1_norm)
            outcomes.append(1.0 if ep['dissociated'] else 0.0)

        if not z_components:
            _LOG.debug("  No data. Skipping.")
            return

        z_arr = np.array(z_components)
        norm_arr = np.array(l1_norms)
        out_arr = np.array(outcomes)

        _LOG.debug(f"  First-layer l=1 mean norm: {norm_arr.mean():.4f} "
              f"(vs last-layer ~0.02)")
        _LOG.debug(f"  First-layer l=1 mean z-direction: {z_arr.mean():.3f} "
              f"({'toward substrate' if z_arr.mean() < -0.2 else 'NOT consistent'})")

        # Compare z-direction between success/failure
        z_succ = z_arr[out_arr > 0.5]
        z_fail = z_arr[out_arr < 0.5]
        if len(z_succ) > 0 and len(z_fail) > 0:
            _LOG.debug(f"  z-dir dissociated={z_succ.mean():.3f} | failed={z_fail.mean():.3f}")

        # Outcome prediction from first-layer embedding
        X_fl = []
        y_fl = []
        for ep in self.episodes:
            if 'first_layer' not in ep:
                continue
            X_fl.append(ep['first_layer'][0])
            y_fl.append(1.0 if ep['dissociated'] else 0.0)

        X_fl = torch.stack(X_fl)
        y_fl = torch.tensor(y_fl)

        # Full first layer
        acc, auc = self._train_classifier(X_fl, y_fl)
        _LOG.debug(f"  Full first-layer AUC: {auc:.3f} "
              f"{'OK PREDICTIVE' if auc > 0.6 else 'FAIL'}")

        # l=1 only from first layer
        acc_l1, auc_l1 = self._train_classifier(X_fl[:, l1_start:l1_end], y_fl)
        _LOG.debug(f"  First-layer l=1 only AUC: {auc_l1:.3f} "
              f"{'OK PREDICTIVE' if auc_l1 > 0.6 else 'FAIL'}")

        # Compare with last-layer l=1
        X_last = torch.stack([ep['raw_states'][0, l1_start:l1_end] for ep in self.episodes
                              if 'first_layer' in ep])
        acc_last, auc_last = self._train_classifier(X_last, y_fl)
        _LOG.debug(f"  Last-layer l=1 only AUC: {auc_last:.3f} (comparison)")

        _LOG.debug(f"\n  If first-layer l=1 AUC >> last-layer l=1 AUC:")
        _LOG.debug(f"    -> First layer encodes direction, last layer lost it")
        _LOG.debug(f"    -> Use first-layer SYM as directional input to policy")

    @staticmethod
    def _compute_auc(probs: np.ndarray, labels: np.ndarray) -> float:
        """Simple AUC computation without sklearn."""
        pos = probs[labels > 0.5]
        neg = probs[labels < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        auc = 0.0
        for p in pos:
            auc += (neg < p).sum() + 0.5 * (neg == p).sum()
        return float(auc) / (len(pos) * len(neg))

    def save(self):
        """Save collected data."""
        torch.save({
            'episodes': self.episodes,
            'n_episodes': self.n_episodes,
        }, self.save_path)
        _LOG.debug(f"[PROBE] Saved {self.n_episodes} episodes to {self.save_path}")


INJECTION_GUIDE = """
=== ADD THESE TO YOUR CODE ===

1. In episode_runner.py __init__ (or trainer.py), create the probe:

    from embedding_probe import EmbeddingProbe
    self.embedding_probe = EmbeddingProbe()

2. In episode_runner.py, at episode start (near line ~415):

    FIND:
        if hasattr(self.agent, 'buffer_manager'):
    ADD BEFORE:
        if hasattr(self, 'embedding_probe'):
            self.embedding_probe.start_episode()

3. In episode_runner.py, right after action is sampled (after select_action):

    FIND the block where actor_state and action_result are both available.
    The key variables you need are:
      - actor_embedding_raw (pre-normalization SYM, [1, 480])
      - actor_state (post-normalization SYM, [1, 480])
      - action_result['translation_direction'] ([1, 3] or [3])
      - o2_positions ([2, 3])

    ADD:
        if hasattr(self, 'embedding_probe'):
            self.embedding_probe.collect_step(
                raw_state=actor_embedding_raw,
                norm_state=actor_state,
                direction=action_result['translation_direction'],
                o2_positions=torch.tensor(o2_positions_np, dtype=torch.float32),
            )

4. In episode_runner.py, at episode end (where dissociation result is known):

    ADD:
        if hasattr(self, 'embedding_probe'):
            self.embedding_probe.end_episode(dissociated=episode_dissociated)

5. In trainer.py, after every ~100 episodes:

    ADD:
        if episode % 100 == 99 and hasattr(self.episode_runner, 'embedding_probe'):
            self.episode_runner.embedding_probe.analyze()
            self.episode_runner.embedding_probe.save()
"""


if __name__ == "__main__":
    print(INJECTION_GUIDE)

    # Demo with synthetic data
    print("\n=== DEMO WITH SYNTHETIC DATA ===\n")
    probe = EmbeddingProbe()

    for ep_idx in range(80):
        probe.start_episode()
        # Simulate: high z-component in l=1 -> more likely to dissociate
        base_emb = torch.randn(480)

        # Make dissociation correlate with l=1 z-component
        dissociated = (base_emb[128 + 2] > 0.0)  # l=1 first vector z > 0

        for step in range(10):
            raw = base_emb + 0.1 * torch.randn(480)
            norm = raw * 0.01  # simulate crushing
            direction = F.normalize(torch.randn(3), dim=0)
            probe.collect_step(raw, norm, direction)

        probe.end_episode(bool(dissociated))

    probe.analyze()
