import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict
from realize.networks.distributions.policy_utils.policy_helpers import von_mises_log_prob
from realize.networks.distributions.policy_utils.vMF_utils import von_mises_fisher_logprob 
import logging
_LOG = logging.getLogger(__name__)

class AxisSamplingManager:
    """Manager for O-O axis prediction and stochastic sampling."""
    
    def __init__(self, policy_network):
        self.policy = policy_network
        self.sample_call_count = 0
        self.eval_call_count = 0
    
    def predict_axis_distribution_params(
        self,
        frame_features: torch.Tensor,
        o2_embedding_asymmetric: Optional[torch.Tensor] = None,
        o2_dir_aligned: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Predict axis distribution parameters using geometric residual vMF.

        Architecture: single directional vMF whose mean direction is obtained by
        applying the spherical exponential map to a zero-initialized equivariant
        tangent correction projected onto a geometric spear-prior base point.
        """
        if not hasattr(self.policy, 'axis_from_frame'):
            raise RuntimeError(
                "axis_from_frame head not found in PolicyNetwork.\n"
                "Must define self.axis_from_frame outputting 1x1e+1x0e (4D total)\n"
                "for von Mises-Fisher axis sampling."
            )
        if o2_embedding_asymmetric is None:
            raise RuntimeError(
                "o2_embedding_asymmetric must be provided for axis prediction."
            )
        if o2_dir_aligned is None:
            raise RuntimeError(
                "o2_dir_aligned must be provided for geometric spear prior."
            )
        if o2_embedding_asymmetric.shape[0] != frame_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: frame_features batch={frame_features.shape[0]}, "
                f"o2_embedding_asymmetric batch={o2_embedding_asymmetric.shape[0]}"
            )

        # Input diagnostics: only log first 3 calls to avoid I/O overhead
        if self.sample_call_count < 3:
            _LOG.debug(f"[AXIS_MANAGER] frame_features: shape={frame_features.shape}, "
                  f"mean={frame_features.mean().item():.4f}, std={frame_features.std().item():.4f}, "
                  f"requires_grad={frame_features.requires_grad}")
            _LOG.debug(f"[AXIS_MANAGER] o2_embedding_asymmetric: shape={o2_embedding_asymmetric.shape}, "
                  f"mean={o2_embedding_asymmetric.mean().item():.4f}, std={o2_embedding_asymmetric.std().item():.4f}, "
                  f"requires_grad={o2_embedding_asymmetric.requires_grad}")

        # Note: When backbone is frozen, input embeddings won't have requires_grad=True.
        # This is EXPECTED and does NOT prevent learning - gradients flow through axis head weights.
        if torch.is_grad_enabled():
            if not hasattr(self, '_frozen_backbone_warning_shown'):
                self._frozen_backbone_warning_shown = False
            if not frame_features.requires_grad and not o2_embedding_asymmetric.requires_grad:
                if not self._frozen_backbone_warning_shown:
                    _LOG.debug(f"  [INFO] Embeddings detached (backbone frozen) - axis head weights still trainable")
                    self._frozen_backbone_warning_shown = True

        axis_outputs = self.policy.axis_from_frame(
            frame_features,
            o2_embedding_asymmetric
        )

        # FAIL-FAST: Catch broken gradient graph
        if torch.is_grad_enabled():
            if not axis_outputs.requires_grad:
                raise RuntimeError(
                    f"axis_outputs has requires_grad=False inside grad-enabled context!\n"
                    f"This breaks PPO learning. Check:\n"
                    f"  1. torch.no_grad() wrapping axis forward\n"
                    f"  2. frame_features or o2_embedding_asymmetric are detached\n"
                    f"  frame_features.requires_grad={frame_features.requires_grad}\n"
                    f"  o2_embedding_asymmetric.requires_grad={o2_embedding_asymmetric.requires_grad}"
                )
        if self.sample_call_count < 3:
            _LOG.debug(f"[AXIS_MANAGER] axis_outputs: shape={axis_outputs.shape}, "
                  f"values={axis_outputs[0].detach().cpu().numpy()}")

        if axis_outputs.shape[-1] != 4:
            raise RuntimeError(
                f"axis_from_frame must output 4 values (1x1e+1x0e), "
                f"got {axis_outputs.shape[-1]}."
            )

        batch_size = axis_outputs.shape[0]
        device = axis_outputs.device

        # --- Unpack raw outputs ---
        delta_z = axis_outputs[:, 0:3]              # [B, 3] correction vector (1e, zero at init)
        log_concentration_raw = axis_outputs[:, 3]   # [B] log_kappa_raw (0e)

        # --- Geometric spear prior: base point construction ---
        # b_raw = d_z * e_z + eps * u_OO
        #       = [eps*u_x, eps*u_y, (1+eps)*d_z]
        # At init (delta_z=0): mu = b = spear direction (+/-v_hat)
        # After training: exp map can rotate mu anywhere on S2
        EPS_PRIOR = 0.01
        d_z = o2_dir_aligned[:, 2]                              # [B]
        b_raw = EPS_PRIOR * o2_dir_aligned.clone()              # [B, 3]
        b_raw[:, 2] = b_raw[:, 2] + d_z                        # amplify z-component
        b = F.normalize(b_raw, dim=-1)                          # [B, 3] unit base point

        # --- Tangent projection: remove radial component of delta_z ---
        dot = (delta_z * b).sum(dim=-1, keepdim=True)           # [B, 1]
        tau = delta_z - dot * b                                 # [B, 3] tangent vector

        # --- Spherical exponential map: geodesic perturbation ---
        # mu = cos(r) * b + sinc(r) * tau,  where r = ||tau||
        # At r->0: sinc(r)->1, so mu->b (exactly the prior)
        r = tau.norm(dim=-1, keepdim=True)                      # [B, 1]
        r_safe = r.clamp(min=1e-8)
        sinc_r = torch.sin(r_safe) / r_safe                    # sin(r)/r, ~1 when r~0
        mean_direction = torch.cos(r) * b + sinc_r * tau        # [B, 3] on S2

        # Safety: ensure unit norm (protect against float drift)
        mean_direction = F.normalize(mean_direction, dim=-1)

        # --- Bounded concentration parameterization (sigmoid mapping) ---
        log_kappa_min = 0.0    # kappa_min = 1.0 (~57 deg spread)
        log_kappa_max = 4.0    # kappa_max ~ 55 (~8 deg spread)
        log_concentration = (
            log_kappa_min
            + (log_kappa_max - log_kappa_min) * torch.sigmoid(log_concentration_raw)
        )
        log_concentration = log_concentration.clamp(min=log_kappa_min, max=log_kappa_max)

        # --- Per-step logging (parseable by extract.sh) ---
        kappa_val = torch.exp(log_concentration[0]).item()
        mu_np = mean_direction[0].detach().cpu().numpy()
        dz_norm = delta_z[0].norm().item()
        tau_norm = r[0].item()
        _LOG.info(
            f"[AXIS] kappa={kappa_val:.4f} "
            f"axis_mu=[{mu_np[0]:.6f}, {mu_np[1]:.6f}, {mu_np[2]:.6f}] "
            f"log_conc_raw={log_concentration_raw[0].item():.4f} "
            f"delta_z_norm={dz_norm:.4f} tau_norm={tau_norm:.4f} "
            f"d_z={d_z[0].item():.4f}")
        self.sample_call_count += 1

        axis_params = {
            'mean_direction': mean_direction,
            'log_concentration': log_concentration,
        }

        return axis_params

    def sample_axis_stochastic(
        self,
        axis_params: dict,
        sampled_direction: torch.Tensor,
        training: bool = True,
        R_feat_to_world: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample an axis from a single directional vMF distribution.
        # log_concentration is clamped at max=4.0 (kappa <= ~54.6) before exp to
        # keep kappa numerically well-behaved; log_prob is left unclamped so
        # gradients survive when kappa*(mu.x) is very negative.
        import os as _os_sa

        mean_direction = axis_params['mean_direction']
        log_concentration = axis_params['log_concentration']

        # Bound kappa at head level instead of clamping log_prob later
        log_concentration_bounded = log_concentration.clamp(max=4.0)
        concentration = torch.exp(log_concentration_bounded)

        # Single directional vMF sampling
        # Mean direction already encodes correct sign via geometric spear prior
        axis_sample = self._sample_vmf_householder(mean_direction, concentration)
        axis_aligned = axis_sample

        # Single vMF log-prob (no clamp)
        log_prob = von_mises_fisher_logprob(axis_aligned, mean_direction, concentration)

        # DIAGNOSTICS
        if _os_sa.environ.get('AXIS_FIX_DEBUG', '1') == '1':
            if not hasattr(self, '_axis_sample_call_count'):
                self._axis_sample_call_count = 0
                self._axis_sample_log_every = int(_os_sa.environ.get('AXIS_FIX_DEBUG_EVERY', '200'))
            self._axis_sample_call_count += 1

            if self._axis_sample_call_count % self._axis_sample_log_every == 0:
                with torch.no_grad():
                    kappa = concentration
                    lc_raw = log_concentration
                    lc_clamped_frac = (lc_raw > 4.0).float().mean().item()
                    mu_dot_x = (mean_direction * axis_aligned).sum(dim=-1)
                    _LOG.debug(f"[AXIS_FIX SAMPLE call={self._axis_sample_call_count}] "
                          f"kappa: min={kappa.min().item():.2f} max={kappa.max().item():.2f} "
                          f"mean={kappa.mean().item():.2f} "
                          f"| log_conc_capped_frac={lc_clamped_frac:.2%} "
                          f"| mu.x: min={mu_dot_x.min().item():.3f} "
                          f"max={mu_dot_x.max().item():.3f} mean={mu_dot_x.mean().item():.3f} "
                          f"| log_prob: min={log_prob.min().item():.2f} "
                          f"max={log_prob.max().item():.2f} mean={log_prob.mean().item():.2f}")

        # Return axis in ALIGNED frame -- world conversion in rotate_axis_to_global()
        return axis_aligned, log_prob

    def _sample_vmf_householder(self, mu, kappa):
        """
        Robust vMF sampling using Householder reflection.
        Guarantees sample is rotated from e1 to mu correctly.
        
        Args:
            mu: [B, 3] Normalized mean directions
            kappa: [B] Concentration parameters
        Returns:
            x: [B, 3] Sampled vectors on S2
        """
        batch_size = mu.shape[0]
        device = mu.device
        
        # 1. Sample w (cosine of angle from mean) from marginal p(w|kappa)
        # For 3D vMF, p(w) ~ exp(kappa*w)
        w = self._sample_vmf_cosine(kappa) # [B, 1]
        
        # 2. Sample v (uniform on circle S1) orthogonal to w
        phi = 2 * math.pi * torch.rand(batch_size, 1, device=device)
        v = torch.cat([torch.cos(phi), torch.sin(phi)], dim=1) # [B, 2]
        
        # 3. Construct sample 'z' aligned with basis e1 = [1, 0, 0]
        # z = [w, v * sqrt(1-w^2)]
        sin_theta = torch.sqrt((1 - w**2).clamp(min=0.0))
        z = torch.cat([w, v * sin_theta], dim=1) # [B, 3]
        
        # 4. Rotate 'z' to align with 'mu' using Householder reflection
        # Basis vector e1
        e1 = torch.zeros_like(mu)
        e1[:, 0] = 1.0
        
        # Householder vector u = (e1 - mu) / ||e1 - mu||
        u = e1 - mu
        u_norm = torch.norm(u, dim=-1, keepdim=True)
        
        # Handle cases where mu is parallel/anti-parallel to e1
        mask_no_rot = u_norm < 1e-6  # mu approx e1 (no rotation needed)
        mask_singularity = (e1 + mu).norm(dim=-1, keepdim=True) < 1e-6 # mu approx -e1
        
        # Normalize u safely
        u = u / (u_norm + 1e-9)
        
        # Reflection: x = z - 2(z . u)u
        z_dot_u = (z * u).sum(dim=-1, keepdim=True)
        x = z - 2 * z_dot_u * u
        
        # Apply edge case fixes
        x = torch.where(mask_no_rot, z, x)
        # If mu = -e1, we just flip z
        x = torch.where(mask_singularity, -z, x)
        
        return x

    def _sample_vmf_cosine(self, kappa):
        """
        Numerically stable Inverse CDF sampling for vMF cosine (w) in 3D.
        For 3D specifically, the CDF is invertible analytically.
        
        p(w) propto exp(kappa * w)
        w = 1 + (1/kappa) * log(y + (1-y)e^(-2*kappa))
        """
        batch_size = kappa.shape[0]
        y = torch.rand(batch_size, device=kappa.device)
        
        # Stable log1p implementation for full range of kappa
        # w = 1 + log(y + (1-y)e^-2k) / k
        #   = 1 + log(y(1 - e^-2k) + e^-2k) / k
        
        exp_minus_2k = torch.exp(-2 * kappa)
        term = y * (1 - exp_minus_2k) + exp_minus_2k
        
        # Epsilon for log stability
        w = 1.0 + torch.log(term) / (kappa + 1e-6)
        
        # Fix kappa ~ 0 case (Uniform sampling on sphere => Uniform w in [-1, 1])
        mask_zero = kappa < 1e-4
        w_uniform = 1.0 - 2.0 * y # Uniform in [-1, 1]
        
        w = torch.where(mask_zero, w_uniform, w)
        return w.unsqueeze(-1)

    def rotate_axis_to_global(
        self,
        axis_frame: torch.Tensor,
        R_feat_to_world: torch.Tensor
    ) -> torch.Tensor:
        """
        Rotate sampled axis from feature frame to world frame.
        """
        axis_world = torch.bmm(
            axis_frame.unsqueeze(1), 
            R_feat_to_world.transpose(1, 2) 
        ).squeeze(1)
        
        return axis_world
    
    def evaluate_axis_log_prob(
        self,
        axis_params: dict,
        old_axes: torch.Tensor,
        sampled_directions: torch.Tensor = None,
        R_feat_to_world: torch.Tensor = None
    ) -> torch.Tensor:
        # Evaluate log-prob of old axes under the current axial vMF distribution.
        # kappa is bounded at the head level; log_prob is left unclamped.
        import os as _os_ea

        self.eval_call_count += 1

        # GRADIENT FLOW DIAGNOSTIC
        if self.eval_call_count <= 5:
            _LOG.debug(f"\n[AXIS LOG_PROB GRADIENT CHECK] call #{self.eval_call_count}")
            mean_dir = axis_params['mean_direction']
            log_conc = axis_params['log_concentration']
            _LOG.debug(f"  mean_direction: requires_grad={mean_dir.requires_grad}, grad_fn={mean_dir.grad_fn}")
            _LOG.debug(f"  log_concentration: requires_grad={log_conc.requires_grad}, grad_fn={log_conc.grad_fn}")
            _LOG.debug(f"  old_axes: requires_grad={old_axes.requires_grad}")

            if not mean_dir.requires_grad:
                _LOG.debug(f"  mean_direction has no gradient! Axis network won't learn!")
            if not log_conc.requires_grad:
                _LOG.debug(f"  log_concentration has no gradient! Axis network won't learn!")

        batch_size = old_axes.shape[0]
        device = old_axes.device

        mean_direction = axis_params['mean_direction']
        log_concentration_bounded = axis_params['log_concentration'].clamp(max=4.0)
        concentration = torch.exp(log_concentration_bounded)

        old_axes = old_axes / (torch.norm(old_axes, dim=-1, keepdim=True) + 1e-8)

        # Single directional vMF log-prob (no clamp)
        log_probs = von_mises_fisher_logprob(old_axes, mean_direction, concentration)

        # DIAGNOSTICS
        if _os_ea.environ.get('AXIS_FIX_DEBUG', '1') == '1':
            if not hasattr(self, '_axis_eval_call_count'):
                self._axis_eval_call_count = 0
                self._axis_eval_log_every = int(_os_ea.environ.get('AXIS_FIX_DEBUG_EVERY', '50'))
            self._axis_eval_call_count += 1

            if self._axis_eval_call_count % self._axis_eval_log_every == 0:
                with torch.no_grad():
                    mu_dot_x = (mean_direction * old_axes).sum(dim=-1)
                    saturated_neg = (log_probs < -10).float().mean().item()
                    saturated_pos = (log_probs > 10).float().mean().item()
                    _LOG.debug(f"[AXIS_FIX EVAL call={self._axis_eval_call_count}] "
                          f"kappa: mean={concentration.mean().item():.2f} max={concentration.max().item():.2f} "
                          f"| mu.old_axis: mean={mu_dot_x.mean().item():.3f} min={mu_dot_x.min().item():.3f} "
                          f"| log_prob: mean={log_probs.mean().item():.2f} min={log_probs.min().item():.2f} max={log_probs.max().item():.2f} "
                          f"| saturated: neg_frac={saturated_neg:.2%} pos_frac={saturated_pos:.2%} "
                          f"| log_prob.requires_grad={log_probs.requires_grad}")

        return log_probs

    def compute_entropy(self, axis_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute differential entropy of directional von Mises-Fisher distribution.

        For vMF on S^2 with concentration kappa:
        The exact entropy involves modified Bessel functions and is expensive to compute.
        We use a well-established approximation:
        - High kappa (concentrated): H ~= -log(kappa) + log(2pi) + 1
        - Low kappa (diffuse): H ~= log(4pi) ~= 2.53 (uniform on sphere)

        Args:
            axis_params: Dict with 'log_concentration' key

        Returns:
            entropy: [batch] entropy in nats
        """
        log_kappa = axis_params['log_concentration']
        kappa = torch.exp(log_kappa).clamp(min=1e-6, max=100.0)

        # Single-mode vMF entropy approximation
        high_kappa_entropy = -torch.log(kappa + 1e-9) + np.log(2 * np.pi) + 1.0
        low_kappa_entropy = torch.full_like(kappa, np.log(4 * np.pi))
        blend_weight = torch.sigmoid(kappa - 3.0)
        vmf_entropy = blend_weight * high_kappa_entropy + (1 - blend_weight) * low_kappa_entropy

        # Single vMF -- no bipolar Bernoulli entropy term
        entropy = vmf_entropy

        # Clamp to reasonable range
        entropy = torch.clamp(entropy, min=0.0, max=3.0)

        return entropy
