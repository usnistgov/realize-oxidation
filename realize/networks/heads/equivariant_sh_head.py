"""Equivariant head that outputs spherical-harmonic coefficients."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple, List

from e3nn.o3 import Irreps, Linear, spherical_harmonics, FullyConnectedTensorProduct, TensorProduct
from e3nn import o3

from realize.networks.heads.sh_head_utils.sh_blocks import S2Grid, CrossMultiplicityTP

import logging
_LOG = logging.getLogger(__name__)


class EquivariantSHHead(nn.Module):
    """
    Equivariant SH head generating spherical-harmonic coefficients for the
    translation direction. Uses cross-multiplicity tensor products
    TP(features_A, features_B) to extract directional correlations between
    different "views" of the local environment, rather than a plain Linear
    that would preserve z-bias.

    Zonal output emerges when neighbor directions are uniform (vacuum);
    non-zonal output emerges when they are diverse (e.g. SiO2 channel).
    """
    
    @staticmethod
    def _soft_clip_equivariant(coeffs: torch.Tensor, lmax: int, limit: float = 12.0) -> torch.Tensor:
        """Clip SH coefficients by per-l norm (preserves equivariance)."""
        result = coeffs.clone()
        offset = 0
        for l in range(lmax + 1):
            dim = 2 * l + 1
            if offset + dim > coeffs.shape[-1]:
                break
            c_l = coeffs[..., offset:offset+dim]
            norm_l = torch.norm(c_l, dim=-1, keepdim=True)
            clipped_norm = limit * torch.tanh(norm_l / limit)
            scale = clipped_norm / (norm_l + 1e-8)
            result[..., offset:offset+dim] = c_l * scale
            offset += dim
        return result

    def __init__(self, model_config, channel_config, device, reward_config=None):
        super().__init__()
        
        self.model_config = model_config
        self.channel_config = channel_config
        self.device = device
        
        self.num_channels = model_config.num_channels
        self.lmax = model_config.lmax
        self.sh_dim = (self.lmax + 1) ** 2
        
        _LOG.info(f"[EquivariantSHHead] Initializing with lmax={self.lmax}, channels={self.num_channels}")
        _LOG.info(f"[EquivariantSHHead] Using CROSS-MULTIPLICITY TENSOR PRODUCTS")
        
        encoder_irreps_str = getattr(model_config, "encoder_output_irreps", "16x0e+8x1e+4x2e")
        self.encoder_irreps = Irreps(encoder_irreps_str)

        self._input_max_l = 0
        for mul, ir in self.encoder_irreps:
            if ir.l > self._input_max_l:
                self._input_max_l = ir.l
        _LOG.info(f"[EquivariantSHHead] Input irreps: {self.encoder_irreps}")
        _LOG.info(f"[EquivariantSHHead] Input max_l = {self._input_max_l}")
        
        # Detect l=1 parity for output
        self._l1_parity = 'e'
        for mul, ir in self.encoder_irreps:
            if ir.l == 1:
                self._l1_parity = 'e' if ir.p == 1 else 'o'
                break
        
        # Channel lmax values
        # TP can produce l up to l1+l2, so max achievable = 2 * input_max_l
        # e.g., input l=3 -> 2(x)3->l=5, 3(x)3->l=6. Don't cap at input_max_l.
        if self.num_channels == 4:
            raw_lmax = [3, 3, 2, 2]
        elif self.num_channels == 3:
            raw_lmax = [3, 3, 2]
        else:
            raw_lmax = [self.lmax]
        max_achievable_l = 2 * self._input_max_l
        self.channel_lmax_values = [min(lm, max_achievable_l) for lm in raw_lmax]

        self.max_channel_lmax = max(self.channel_lmax_values)
        self.channel_sh_dims = [(lmax + 1) ** 2 for lmax in self.channel_lmax_values]
        
        _LOG.info(f"[EquivariantSHHead] Channel lmax values: {self.channel_lmax_values}")
        
        # Reward config for beta scaling
        reward = reward_config if reward_config is not None else getattr(model_config, "reward", None)
        self.target_mass_fraction = float(getattr(reward, "target_mass_fraction", 0.7))
        area_list = getattr(reward, "area_at_mass_targets", [0.10, 0.20, 0.35, 0.50])
        self.area_at_mass_targets = list(map(float, area_list))
        
        # Beta bounds per channel
        self.register_buffer('beta_max_per_channel', 
                            torch.tensor([25.0, 20.0, 14.0, 10.0], dtype=torch.float32)[:self.num_channels])

        # Learnable beta (used in single-channel mode when use_learnable_beta=True)
        self.use_learnable_beta = getattr(reward, "use_learnable_beta", False)
        if self.use_learnable_beta:
            beta_init = float(getattr(reward, "learnable_beta_init", 2.0))
            self.learnable_beta_min = float(getattr(reward, "learnable_beta_min", 0.5))
            self.learnable_beta_max = float(getattr(reward, "learnable_beta_max", 25.0))
            # nn.Parameter: gradient flows through beta -> policy learns sharpness
            self.log_beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32, device=device))
            _LOG.info(f"[EquivariantSHHead] Using LEARNABLE beta: init log_beta={beta_init:.2f} "
                  f"(beta~={math.exp(beta_init):.1f}), bounds=[{self.learnable_beta_min}, {self.learnable_beta_max}]")

        # S2 Grid
        self.s2_block = S2Grid(
            lmax=self.max_channel_lmax,
            res_beta=64,
            res_alpha=64,
            device=device
        )
        
        # Per-channel SH evaluation matrices
        self._sh_to_grid_by_lmax = {}
        for L in set(self.channel_lmax_values):
            dim = (L + 1) ** 2
            self._sh_to_grid_by_lmax[L] = self.s2_block.sh_to_grid[:dim, :].contiguous()
        
        # Temperature scheduling for gates
        self.gate_temp_init = 5.0
        self.gate_temp_final = 1.5
        self.gate_temp_episodes = 120
        self.current_temperature = self.gate_temp_init
        self._external_temperature = None
        
        # State buffers
        self.register_buffer('_last_steering_dir', None, persistent=False)
        self.register_buffer('_last_sh_coeffs', None, persistent=False)
        self.register_buffer('_last_channel_weights', None, persistent=False)
        self.register_buffer('_last_channel_rotations', None, persistent=False)
        self.register_buffer('_last_coeffs_are_world', torch.tensor(True), persistent=False)

        self._build_networks()
        self.to(device)

    def set_temperature(self, t):
        if t is None:
            self._external_temperature = None
        else:
            self._external_temperature = float(t)
            self.current_temperature = float(t)

    def _build_networks(self):
        """Build cross-multiplicity TP and channel projection networks."""
        
        # Cross-Multiplicity Tensor Product
        # This extracts directional correlations from multi-copy features
        
        self.cross_tp = CrossMultiplicityTP(
            input_irreps=self.encoder_irreps,
            output_lmax=self.max_channel_lmax,
            device=self.device
        )
        
        # The cross-TP outputs irreps up to max_channel_lmax
        enriched_irreps = self.cross_tp.output_irreps
        _LOG.info(f"[EquivariantSHHead] Cross-TP output: {enriched_irreps}")
        
        # Per-Channel Linear Projection
        # Projects enriched features to per-channel SH coefficients
        
        self.channel_linears = nn.ModuleList()
        
        for i in range(self.num_channels):
            ch_lmax = self.channel_lmax_values[i]
            out_spec = [f"1x{l}e" for l in range(ch_lmax + 1)]
            out_irreps = Irreps("+".join(out_spec))
            
            linear = Linear(enriched_irreps, out_irreps).to(self.device)
            # Init std=0.01 for all params; per-l decay applied below via l_decay buffers
            with torch.no_grad():
                for param in linear.parameters():
                    nn.init.normal_(param, mean=0.0, std=0.01)
            self.channel_linears.append(linear)

            _LOG.info(f"[EquivariantSHHead] Channel {i}: {enriched_irreps} -> {out_irreps}")

        # Per-l init decay: l0 suppressed, l1 (dipole "go toward surface") boosted,
        # l2+ decayed, so the dipole signal dominates from the start.
        decay_map = {0: 1.0, 1: 0.3, 2: 0.1, 3: 0.03, 4: 0.01, 5: 0.003}
        for i in range(self.num_channels):
            ch_lmax = self.channel_lmax_values[i]
            scales = []
            for l in range(ch_lmax + 1):
                scales.extend([decay_map.get(l, 0.01)] * (2 * l + 1))
            self.register_buffer(
                f'l_decay_ch{i}',
                torch.tensor(scales, dtype=torch.float32, device=self.device)
            )
            _LOG.info(f"[EquivariantSHHead] Channel {i} l-decay scales (l1-dominant init): "
                  + ", ".join(f"l{l}={decay_map.get(l, 0.01)}" for l in range(ch_lmax + 1)))            

        # Gate Network (Invariant)
        # Learns channel mixing weights from scalar features
        
        self._build_gate_network()
        
        _LOG.info(f"[EquivariantSHHead] Built {self.num_channels} channels with cross-multiplicity TP")

    def _build_gate_network(self):
        """Build invariant gate network for channel weights."""
        # Extract invariants (scalars + norms) from input
        num_scalars = 0
        num_vectors = 0
        
        for mul, ir in self.encoder_irreps:
            if ir.l == 0:
                num_scalars += mul
            else:
                num_vectors += mul  # Will use norm
        
        gate_in_dim = num_scalars + num_vectors
        self._gate_in_dim = gate_in_dim
        self._num_scalars_input = num_scalars
        
        gate_hidden = getattr(self.model_config, "gate_hidden_dim", 64)
        
        self.gate_network = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_channels),
        ).to(self.device)
        
        # Initialize gate network with diversity-encouraging init
        with torch.no_grad():
            for m in self.gate_network.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)  # Larger gain
                    if m.bias is not None:
                        m.bias.zero_()
            
            # Final layer: moderate scale allows non-uniform initial gates
            # This prevents vanishing gradients at the softmax
            final_linear = None
            for m in reversed(list(self.gate_network.modules())):
                if isinstance(m, nn.Linear):
                    final_linear = m
                    break
            
            if final_linear is not None:
                # Scale down but not too much (0.1 instead of 0.01)
                final_linear.weight.data.mul_(0.1)
                # Add small initial bias differences to break symmetry
                # This gives initial softmax != [0.25, 0.25, 0.25, 0.25]
                final_linear.bias.data = torch.randn(self.num_channels, device=self.device) * 0.1

    def _extract_invariants(self, features: torch.Tensor) -> torch.Tensor:
        """Extract invariant features (scalars + norms) for gate network."""
        batch_size = features.shape[0]
        invariants = []
        offset = 0
        
        for mul, ir in self.encoder_irreps:
            dim = mul * (2 * ir.l + 1)
            block = features[:, offset:offset+dim]
            
            if ir.l == 0:
                # Keep scalars directly
                invariants.append(block)
            else:
                # Take norm of each multiplicity copy
                block_reshaped = block.reshape(batch_size, mul, 2 * ir.l + 1)
                norms = torch.norm(block_reshaped, dim=-1)
                invariants.append(norms)
            
            offset += dim
        
        return torch.cat(invariants, dim=-1)

    def _find_beta_for_target_batch(self, g_batch: torch.Tensor, target_area: float, 
                                     beta_hi: float, target_mass: float = 0.70,
                                     beta_lo: float = 0.1, n_iters: int = 20) -> torch.Tensor:
        """Binary search for beta achieving target area at mass fraction."""
        batch_size = g_batch.shape[0]
        weights = self.s2_block.w_vec
        log_w = self.s2_block.log_w_vec
        w_sum = weights.sum()

        # Check for degenerate (constant) distributions
        g_var = ((g_batch - (g_batch * weights).sum(-1, keepdim=True) / w_sum) ** 2 * weights).sum(-1) / w_sum
        degenerate = g_var < 1e-10
        
        beta_lo_vec = torch.full((batch_size,), beta_lo, device=g_batch.device)
        beta_hi_vec = torch.full((batch_size,), beta_hi, device=g_batch.device)
        beta_lo_vec[degenerate] = 0.0
        beta_hi_vec[degenerate] = 0.0
        
        for _ in range(n_iters):
            beta_mid = 0.5 * (beta_lo_vec + beta_hi_vec)
            
            # Compute normalized mass distribution
            log_mass = beta_mid.unsqueeze(-1) * g_batch + log_w.unsqueeze(0)
            log_mass_norm = log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)
            mass = torch.exp(log_mass_norm)
            
            # Sort by mass (descending) and compute cumulative
            idx_sorted = torch.argsort(mass, dim=-1, descending=True)
            mass_sorted = torch.gather(mass, 1, idx_sorted)
            weights_sorted = torch.gather(weights.unsqueeze(0).expand(batch_size, -1), 1, idx_sorted)
            
            cum_mass = torch.cumsum(mass_sorted, dim=-1)
            threshold = torch.full((batch_size, 1), target_mass, device=g_batch.device)
            exceeds = cum_mass >= threshold
            k_idx = exceeds.float().argmax(dim=-1)
            
            # Compute area for top-k points
            mask = torch.arange(weights.shape[0], device=g_batch.device).unsqueeze(0) <= k_idx.unsqueeze(1)
            area_current = (weights_sorted * mask).sum(dim=-1) / (4.0 * math.pi)
            
            # Binary search update
            too_broad = (area_current > target_area) & (~degenerate)
            beta_lo_vec = torch.where(too_broad, beta_mid, beta_lo_vec)
            beta_hi_vec = torch.where(too_broad, beta_hi_vec, beta_mid)
        
        return 0.5 * (beta_lo_vec + beta_hi_vec)

    def _normalize_per_l(self, features):
        """Normalize each l-block of cross-TP output to unit RMS (Fix B).

        CrossMultiplicityTP produces structurally unbalanced l-block magnitudes
        because different l values receive contributions from different numbers
        of TP paths (e.g., l=0 gets 155 cross-pairs, l=5 gets 9). Despite fewer
        paths, high-l blocks accumulate more total energy due to having more
        dimensions (l=5: 11 dims vs l=0: 1 dim).

        Per-l normalization ensures channel_linears receive balanced input,
        so their weights directly control the frequency balance of SH output.
        Direction information is preserved; only magnitude is normalized.
        """
        output_lmax = self.cross_tp.output_lmax
        normalized_blocks = []
        offset = 0
        for l in range(output_lmax + 1):
            dim = 2 * l + 1
            block = features[:, offset:offset + dim]  # [B, 2l+1]
            rms = (block.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()            
            normalized_blocks.append(block / rms)
            offset += dim
        return torch.cat(normalized_blocks, dim=-1)

    def forward(self, features: torch.Tensor, episode: int = 0,
                state_uid: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate SH distribution using cross-multiplicity tensor products.
        
        The key innovation is in the cross_tp module which extracts directional
        correlations by computing tensor products between different multiplicity
        copies of the l=1 (and higher) features.
        """
        batch_size = features.shape[0]
        w = self.s2_block.w_vec
        log_w = self.s2_block.log_w_vec
        
        # Cross-Multiplicity Tensor Product
        # This creates non-zonal components from directional correlations
        
        enriched_features = self.cross_tp(features)

        # Per-l RMS normalization (Fix B): balance l-block magnitudes
#        enriched_features = self._normalize_per_l(enriched_features)

        # Per-Channel SH Coefficient Generation

        actual_max_sh_dim = max(self.channel_sh_dims)
        channel_coeffs_list = []

        for i in range(self.num_channels):
            # Linear projection to SH coefficients
            sh_coeffs_raw = self.channel_linears[i](enriched_features)
            # Per-l frequency decay 
            sh_coeffs_raw = sh_coeffs_raw * getattr(self, f'l_decay_ch{i}')
            ch_lmax = self.channel_lmax_values[i]

            # Soft clip per-l norm for stability
            sh_coeffs_stable = self._soft_clip_equivariant(sh_coeffs_raw, ch_lmax, limit=12.0)

            # Pad to max dimension
            if sh_coeffs_stable.shape[-1] < actual_max_sh_dim:
                padding = torch.zeros(batch_size, actual_max_sh_dim - sh_coeffs_stable.shape[-1],
                                      device=self.device, dtype=features.dtype)
                sh_coeffs_stable = torch.cat([sh_coeffs_stable, padding], dim=-1)
            
            channel_coeffs_list.append(sh_coeffs_stable)

        sh_coeffs = torch.stack(channel_coeffs_list, dim=1)  # [B, C, D]

        # Identity rotations (coefficients are in world frame)
        self._last_channel_rotations = torch.eye(3, device=self.device, dtype=features.dtype
            ).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_channels, 3, 3)
        self._last_coeffs_are_world = torch.tensor(True, device=self.device)

        # Beta Scaling for Each Channel
        
        channel_betas = []
        channel_logits_list = []
        channel_coeffs_scaled_list = []
        
        for i in range(self.num_channels):
            ch_lmax = self.channel_lmax_values[i]
            ch_sh_dim = (ch_lmax + 1) ** 2
            ch_coeffs = sh_coeffs[:, i, :ch_sh_dim]
            
            # Evaluate on grid
            if ch_lmax == self.max_channel_lmax:
                vals = torch.matmul(ch_coeffs, self.s2_block.sh_to_grid)
            else:
                vals = torch.matmul(ch_coeffs, self._sh_to_grid_by_lmax[ch_lmax])
            
            # RMS of grid values (no mean subtraction -- l=0 constant cancels
            # exactly in log p = u(x) - logZ)
            w_sum = self.s2_block.w_sum
            weighted_var = (w * (vals ** 2)).sum(dim=-1, keepdim=True) / (w_sum + 1e-8)
            with torch.no_grad():
                degenerate = weighted_var.squeeze(-1) < 1e-10

            # RMS normalization
            rms = torch.sqrt(weighted_var + 1e-8)
            rms_safe = torch.where(degenerate.unsqueeze(-1), torch.ones_like(rms), rms)
            vals_norm = vals / rms_safe
            
            # ---- Beta: learnable vs solver ----
            if self.use_learnable_beta:
                # Learnable beta: gradient flows through to policy via PPO
                # log_beta is a scalar parameter shared across batch
                beta_val = self.log_beta.exp().clamp(
                    min=self.learnable_beta_min, max=self.learnable_beta_max
                )
                beta = beta_val.expand(batch_size)
                channel_betas.append(beta)
            else:
                # Original: detached beta solver targeting area-at-mass
                beta_max_i = self.beta_max_per_channel[i].item()
                beta_hi = beta_max_i
                if self.training and i == 0:
                    # Ramp up beta gradually
                    ramp = min(1.0, episode / 50.0)
                    beta_hi = beta_hi * (0.6 + 0.4 * ramp)

                with torch.no_grad():
                    beta_raw = torch.zeros(batch_size, device=self.device, dtype=g.dtype)
                    nondeg = ~degenerate
                    if nondeg.any():
                        beta_raw[nondeg] = self._find_beta_for_target_batch(
                            vals_norm[nondeg].detach(), self.area_at_mass_targets[i],
                            beta_hi, self.target_mass_fraction, 0.0, 15
                        )

                # Ensure all channels have meaningful concentration (beta >= 1.5)
                beta_floor = 1.5
                beta = beta_raw.clamp(min=beta_floor, max=beta_hi)
                channel_betas.append(beta)

            # Scale coefficients
            scale = (beta / torch.clamp(rms.squeeze(-1), min=1e-8)).unsqueeze(-1)
            ch_coeffs_scaled = scale * sh_coeffs[:, i, :ch_sh_dim]
            
            if ch_coeffs_scaled.shape[-1] < actual_max_sh_dim:
                padding = torch.zeros(batch_size, actual_max_sh_dim - ch_coeffs_scaled.shape[-1],
                                      device=self.device, dtype=ch_coeffs_scaled.dtype)
                ch_coeffs_scaled = torch.cat([ch_coeffs_scaled, padding], dim=-1)
            channel_coeffs_scaled_list.append(ch_coeffs_scaled)
            
            # Compute logits
            beta_floor = 0.5
            beta_eff = torch.where(degenerate, torch.full_like(beta, beta_floor), beta)
            logits = beta_eff.unsqueeze(-1) * vals_norm
            channel_logits_list.append(logits)
        
        sh_coeffs_scaled = torch.stack(channel_coeffs_scaled_list, dim=1)
        self._current_betas = torch.stack(channel_betas, dim=1)
        self._channel_logits = torch.stack(channel_logits_list, dim=1)

        # Directional signal enters only through multipole_injector in the
        # actor state; no post-hoc directional bias is added here.

        # Log beta values periodically
        if not hasattr(self, '_beta_log_counter'):
            self._beta_log_counter = 0
        self._beta_log_counter += 1
        if self._beta_log_counter % 10 == 1:
            beta_vals = self._current_betas[0].detach().cpu().numpy()
            if self.use_learnable_beta:
                log_b = self.log_beta.item()
                _LOG.info(f"[Beta] LEARNABLE log_beta={log_b:.3f} beta={math.exp(log_b):.2f}, "
                      f"effective={np.array2string(beta_vals, precision=2, suppress_small=True)}")
            else:
                _LOG.debug(f"[Beta] betas={np.array2string(beta_vals, precision=2, suppress_small=True)}, "
                      f"targets={self.area_at_mass_targets}")

        # Per-channel normalized log mass
        logZ_per_ch = torch.logsumexp(self._channel_logits + log_w.unsqueeze(0).unsqueeze(0), dim=-1)
        self._channel_log_mass_normalized = (
            self._channel_logits + log_w.unsqueeze(0).unsqueeze(0) - logZ_per_ch.unsqueeze(-1)
        )
        
        # Gate Network for Channel Mixing
        
        invariants = self._extract_invariants(features)
        gate_logits = self.gate_network(invariants)
        
        # Temperature scheduling
        if self._external_temperature is not None:
            temp = float(self._external_temperature)
        elif self.training:
            progress = min(episode / self.gate_temp_episodes, 1.0)
            temp = self.gate_temp_init + (self.gate_temp_final - self.gate_temp_init) * progress
        else:
            temp = self.gate_temp_final
        self.current_temperature = temp
        
        if self.num_channels == 1:
            channel_weights = torch.ones_like(gate_logits[:, :1])
        else:
            channel_weights = F.softmax(gate_logits / temp, dim=-1)
        
        # Dirichlet smoothing during training
        if self.training:
            eps = 0.02
            uniform = torch.full_like(channel_weights, 1.0 / self.num_channels)
            channel_weights = (1.0 - eps) * channel_weights + eps * uniform
            channel_weights = channel_weights / channel_weights.sum(dim=-1, keepdim=True)
        
        # Compute Mixture Distribution
        
        log_cw = torch.log(channel_weights + 1e-10).unsqueeze(-1)
        mixture_log_probs = torch.logsumexp(
            self._channel_log_mass_normalized + log_cw, dim=1
        )
        
        # Steering direction (expected direction under mixture)
        mixture_probs = torch.exp(mixture_log_probs)
        mixture_probs = mixture_probs / mixture_probs.sum(dim=-1, keepdim=True)
        
        grid_xyz = self.s2_block.grid_xyz
        steering_dir = torch.einsum('bp,pd->bd', mixture_probs, grid_xyz)
        steering_dir = F.normalize(steering_dir, dim=-1)
        
        # Store state and return
        
        self._last_steering_dir = steering_dir.detach()
        self._last_channel_weights = channel_weights[0].detach() if batch_size > 0 else channel_weights.detach()
        self._last_sh_coeffs_scaled = sh_coeffs_scaled  # keep full batch with grad for continuous eval

        # Aggregate coefficients for compatibility
        common_sh_dim = self.sh_dim
        sh_coeffs_padded = []
        for i in range(self.num_channels):
            ch_sh_dim = self.channel_sh_dims[i]
            ch = sh_coeffs_scaled[:, i, :ch_sh_dim]
            if ch_sh_dim < common_sh_dim:
                ch = torch.cat([ch, torch.zeros(batch_size, common_sh_dim - ch_sh_dim, device=self.device)], dim=-1)
            else:
                ch = ch[:, :common_sh_dim]
            sh_coeffs_padded.append(ch)
        sh_coeffs_padded = torch.stack(sh_coeffs_padded, dim=1)
        aggregated = (channel_weights.unsqueeze(-1) * sh_coeffs_padded).sum(dim=1)

        # Diagnostic: Check non-zonal content and channel diversity
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1

        if self._diag_counter % 50 == 1:
            with torch.no_grad():
                _LOG.debug(f"\n[SH DIAGNOSTIC] Forward #{self._diag_counter}:")

                # Check non-zonal ratio for l=1 (indices 1,2,3)
                # m=-1 is index 1, m=0 is index 2, m=+1 is index 3
                for ch_idx in range(self.num_channels):
                    ch_coeffs = sh_coeffs_scaled[0, ch_idx, :4]  # l=0 and l=1
                    if ch_coeffs.shape[0] >= 4:
                        l0 = ch_coeffs[0].item()
                        l1_m_minus = ch_coeffs[1].item()
                        l1_m_zero = ch_coeffs[2].item()
                        l1_m_plus = ch_coeffs[3].item()

                        zonal_power = l1_m_zero**2
                        non_zonal_power = l1_m_minus**2 + l1_m_plus**2
                        ratio = non_zonal_power / (zonal_power + 1e-8)

                        _LOG.debug(f"  Ch{ch_idx}: l=0={l0:.3f}, l=1=[{l1_m_minus:.3f}, {l1_m_zero:.3f}, {l1_m_plus:.3f}], "
                              f"non-zonal/zonal={ratio:.3f}")

                # Check channel similarity (cosine similarity of coefficient vectors)
                sims = []
                for i in range(self.num_channels):
                    for j in range(i+1, self.num_channels):
                        ci = sh_coeffs_scaled[0, i, :self.channel_sh_dims[min(i,j)]]
                        cj = sh_coeffs_scaled[0, j, :self.channel_sh_dims[min(i,j)]]
                        sim = F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0)).item()
                        sims.append(sim)
                _LOG.debug(f"  Channel similarities: {[f'{s:.3f}' for s in sims]}")
                # Warn on BOTH positive AND negative high correlation
                abs_sims = [abs(s) for s in sims]
                max_abs_sim = max(abs_sims) if abs_sims else 0.0
                if max_abs_sim > 0.9:
                    collapse_type = "MODE" if max(sims) > 0.9 else "CANCELLATION"
                    _LOG.info(f"  WARNING {collapse_type} COLLAPSE WARNING: max |similarity|={max_abs_sim:.3f}")

        return {
            'mixture_log_mass_normalized': mixture_log_probs,
            'grid_n_beta': self.s2_block.res_beta,
            'grid_n_alpha': self.s2_block.res_alpha,
            'channel_log_mass_normalized': self._channel_log_mass_normalized,
            'channel_weights': channel_weights,
            'channel_logits': self._channel_logits,
            'sh_coeffs': sh_coeffs_scaled,
            'sh_coeffs_raw': sh_coeffs,
            'aggregated_coeffs': aggregated,
            'grid_points': self.s2_block.grid_xyz,
            'grid_weights': self.s2_block.w_vec,
            'grid_log_weights': self.s2_block.log_w_vec,
            'channel_rotations': self._last_channel_rotations,
            'steering_dir': steering_dir,
            'betas': self._current_betas,
        }

    def compute_mixture_log_mass(self, channel_log_mass_normalized: torch.Tensor,
                                  channel_weights: torch.Tensor) -> torch.Tensor:
        """Compute mixture log probability over channels."""
        log_w = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        log_mix = torch.logsumexp(log_w + channel_log_mass_normalized, dim=1)
        return log_mix - torch.logsumexp(log_mix, dim=-1, keepdim=True)

    def sample_direction(self, channel_log_mass_normalized: torch.Tensor,
                         channel_weights: torch.Tensor,
                         return_log_prob: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Sample direction from mixture distribution.
        
        Always returns exact grid index from multinomial.
        This ensures PPO rollout and evaluate use identical discrete actions.
        
        Returns:
            sampled_dirs: [batch, 3] direction vectors
            log_prob: [batch] log-probability (None if return_log_prob=False)
            idx: [batch] exact grid indices from multinomial
        """
        batch_size = channel_log_mass_normalized.shape[0]
        log_mix = self.compute_mixture_log_mass(channel_log_mass_normalized, channel_weights)

        probs = torch.exp(log_mix).clamp(min=0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        idx = torch.multinomial(probs, num_samples=1).squeeze(-1).long()
        sampled_dirs = self.s2_block.grid_xyz[idx]

        if not return_log_prob:
            return sampled_dirs, None, idx

        b_idx = torch.arange(batch_size, device=self.device)
        log_prob = log_mix[b_idx, idx]
        return sampled_dirs, log_prob, idx

    def compute_entropy(self, channel_log_mass_normalized: torch.Tensor,
                        channel_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of mixture distribution."""
        log_mix = self.compute_mixture_log_mass(channel_log_mass_normalized, channel_weights)
        mass = torch.exp(log_mix)
        return -(mass * log_mix).sum(dim=-1)

    def evaluate_sh_at_points(self, directions, sh_coeffs_scaled=None):
        """
        Evaluate SH field at arbitrary S^2 points (not grid-locked).

        Args:
            directions: [B, 3] unit vectors on S^2
            sh_coeffs_scaled: [B, C, D] scaled SH coefficients (if None, use last computed)
        Returns:
            vals: [B, C] field values per channel at each direction
        """
        if sh_coeffs_scaled is None:
            sh_coeffs_scaled = self._last_sh_coeffs_scaled

        # Evaluate Y_lm at arbitrary directions using e3nn
        irreps = Irreps.spherical_harmonics(self.max_channel_lmax)
        Y = spherical_harmonics(irreps, directions.to(sh_coeffs_scaled.dtype), normalize=True)  # [B, D_max]

        # Per-channel dot product: u_c(x) = Sigma c_lm^scaled x Y_lm(x)
        D = Y.shape[-1]
        vals = torch.einsum('bd,bcd->bc', Y, sh_coeffs_scaled[:, :, :D])  # [B, C]
        return vals

    def continuous_log_prob(self, directions, channel_weights, sh_coeffs_scaled=None):
        """
        Compute log p(x) = u(x) - logZ for arbitrary directions.

        u(x) = logsumexp over channels of per-channel field values
        logZ computed via grid quadrature (cached from forward pass)
        """
        # Per-channel field values at the query point
        vals_per_channel = self.evaluate_sh_at_points(directions, sh_coeffs_scaled)  # [B, C]

        # Channel mixing via logsumexp
        log_w_c = torch.log(channel_weights.clamp(min=1e-12))  # [B, C]
        u_x = torch.logsumexp(log_w_c + vals_per_channel, dim=-1)  # [B]

        # logZ from grid quadrature (cached during forward)
        # _channel_logits: [B, C, N_grid], log_w: quadrature log weights [N_grid]
        log_w = self.s2_block.log_w_vec  # [N_grid]
        log_w_c_expand = log_w_c.unsqueeze(-1)  # [B, C, 1]
        # Per-grid-point mixture energy
        mixture_energy = torch.logsumexp(
            log_w_c_expand + self._channel_logits, dim=1  # [B, N_grid]
        )
        logZ = torch.logsumexp(mixture_energy + log_w.unsqueeze(0), dim=-1)  # [B]

        log_prob = u_x - logZ
        return log_prob.clamp(-20, 20)

    def perturb_on_tangent_plane(self, directions, std=0.1):
        """
        Add Gaussian noise on the tangent plane of S^2, then project back.

        Args:
            directions: [B, 3] unit vectors
            std: noise standard deviation in radians
        Returns:
            perturbed: [B, 3] perturbed unit vectors
        """
        B = directions.shape[0]
        device = directions.device

        # Random tangent vector via rejection: sample 3D Gaussian, subtract radial component
        noise = torch.randn(B, 3, device=device) * std
        radial = (noise * directions).sum(dim=-1, keepdim=True) * directions
        tangent_noise = noise - radial  # project onto tangent plane

        # Apply exponential map (for small noise, just add and normalize)
        perturbed = F.normalize(directions + tangent_noise, dim=-1, eps=1e-8)
        return perturbed

    def langevin_sample_direction(self, channel_weights, sh_coeffs_scaled=None,
                                   n_steps=10, step_size=0.1, noise_scale=0.05,
                                   deterministic=False):
        """
        Sample direction via Langevin dynamics on S^2 guided by the SH energy field.
        
        Score function: grad_x u(x) projected onto tangent plane of S^2 at x.
        Update: x <- normalize(x + step_size x score + noise)
        
        Args:
            channel_weights: [B, C]
            sh_coeffs_scaled: [B, C, D] (if None, use cached)
            n_steps: number of MCMC steps
            step_size: Langevin step size
            noise_scale: noise magnitude
            deterministic: if True, return mode (no noise)
        Returns:
            x: [B, 3] sampled directions
            log_prob: [B] log probabilities
        """
        if sh_coeffs_scaled is None:
            sh_coeffs_scaled = self._last_sh_coeffs_scaled
        
        B = sh_coeffs_scaled.shape[0]
        device = sh_coeffs_scaled.device
        
        # Initialize from grid categorical (warm start)
        with torch.no_grad():
            log_mix = self.compute_mixture_log_mass(
                self._channel_log_mass_normalized, channel_weights)
            probs = torch.exp(log_mix).clamp(min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1).long()
            x = self.s2_block.grid_xyz[idx].clone()  # [B, 3]
        
        if deterministic:
            log_prob = self.continuous_log_prob(x, channel_weights, sh_coeffs_scaled)
            return x, log_prob
        
        # Langevin dynamics on S^2 (need grad even during rollout's no_grad context)
        irreps = Irreps.spherical_harmonics(self.max_channel_lmax)
        log_w_c = torch.log(channel_weights.clamp(min=1e-12))
        D = sh_coeffs_scaled.shape[-1]

        with torch.enable_grad():
            for step in range(n_steps):
                x_param = x.detach().requires_grad_(True)

                # Compute energy u(x) = logsumexp_c(Sigma c_lm^scaled x Y_lm(x))
                Y = spherical_harmonics(irreps, x_param, normalize=True)
                vals_per_ch = torch.einsum('bd,bcd->bc', Y[:, :D], sh_coeffs_scaled[:, :, :D].detach())
                u = torch.logsumexp(log_w_c + vals_per_ch, dim=-1)  # [B]

                # Score: gradient of u w.r.t. x
                score = torch.autograd.grad(u.sum(), x_param, create_graph=False)[0]  # [B, 3]

                # Project score onto tangent plane of S^2 at x
                radial = (score * x).sum(dim=-1, keepdim=True) * x
                tangent_score = score - radial  # [B, 3]

                # Langevin update + noise
                noise = torch.randn_like(x) * noise_scale
                noise_radial = (noise * x).sum(dim=-1, keepdim=True) * x
                tangent_noise = noise - noise_radial

                # Update and project back to sphere
                x = x + step_size * tangent_score + tangent_noise
                x = F.normalize(x, dim=-1, eps=1e-8)

        # Final log_prob via continuous evaluation
        log_prob = self.continuous_log_prob(x.detach(), channel_weights, sh_coeffs_scaled)
        
        return x.detach(), log_prob

    def rotate_distribution_to_world(self, sh_coeffs_input: torch.Tensor,
                                      channel_rotations: torch.Tensor,
                                      steering_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rotate SH coefficients to world frame if needed."""
        # Currently coefficients are already in world frame
        return sh_coeffs_input

    def set_feature_to_world_transform(self, R: torch.Tensor):
        """Store frame transform."""
        if R.dim() == 3:
            R = R[0]
        if hasattr(self, '_R_feat_to_world'):
            self._R_feat_to_world = R.detach().clone()
        else:
            self.register_buffer('_R_feat_to_world', R.detach().clone(), persistent=False)

    def clear_feature_to_world_transform(self):
        """Clear stored frame transform."""
        if hasattr(self, '_R_feat_to_world'):
            delattr(self, '_R_feat_to_world')

    def has_world_frame(self) -> bool:
        return hasattr(self, '_R_feat_to_world')

    def get_viz_params(self) -> Dict:
        """Get parameters for visualization."""
        with torch.no_grad():
            params = {
                'lmax': self.lmax,
                'num_channels': self.num_channels,
                'channel_lmax_values': self.channel_lmax_values,
                'area_at_mass_targets': self.area_at_mass_targets,
                'target_mass_fraction': self.target_mass_fraction,
            }
            
            if hasattr(self, '_current_betas'):
                params['betas'] = self._current_betas[0].cpu().numpy()
            else:
                params['betas'] = None
            
            params['channel_temps'] = np.full(self.num_channels, self.current_temperature)
            params['channel_scales'] = np.ones(self.num_channels)
            
            if hasattr(self, '_last_sh_coeffs') and self._last_sh_coeffs is not None:
                params['sh_coeffs'] = self._last_sh_coeffs
                params['frame'] = 'world'
            
            return params
