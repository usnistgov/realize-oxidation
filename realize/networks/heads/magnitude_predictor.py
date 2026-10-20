"""Translation-magnitude prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from e3nn.o3 import Irreps, Linear
from realize.networks.irrep_sequential import IrrepSequential, IrrepActivation
from realize.networks.heads.rqs_flow import BoundedRQSFlow

class MagnitudePredictor(nn.Module):
    def __init__(self, model_config, device, sampling_config):        
        super().__init__()
        self.model_config = model_config
        self.sampling_config = sampling_config
        self.device = device
        
        self.r_min = self.model_config.r_min
        self.r_max = self.model_config.r_max
        self.max_trans_distance = self.model_config.max_trans_distance
        
        self.rqs_flow_enabled = self.model_config.rqs_flow_enabled
        self.use_clamped_magnitude = self.model_config.use_clamped_magnitude
        self.use_bessel = getattr(self.model_config, 'use_radial_conditioning', False)
        self.use_bessel_norm = True
        self.use_rqs_flow = self.rqs_flow_enabled
        self._last_magnitude_data = None
        
        # Track exploration schedule
        self._current_episode = 0
        self._exploration_warmup_episodes = 100  # Reduce exploration after warmup

        if self.use_bessel:
            self.cutoff_radius = getattr(self.model_config, 'radial_max_radius', 10.0)
            self.num_bessel = getattr(self.model_config, 'radial_num_basis', 32)
            bessel_n = torch.arange(1, self.num_bessel + 1, dtype=torch.float32)
            self.register_buffer('bessel_n_values', bessel_n)
            self.register_buffer('bessel_scale_factor', torch.tensor(math.pi / self.cutoff_radius))
        else:
            self.num_bessel = 0
        
        # Scalar dim from frame features in aligned frame:
        self.actual_scalar_dim = 28
        total_input_dim = self.actual_scalar_dim + self.num_bessel

        if self.rqs_flow_enabled:
            self.rqs_flow = BoundedRQSFlow(
                scalar_dim=total_input_dim,
                r_min=self.r_min, 
                r_max=self.r_max,
                num_bins=getattr(self.model_config, 'rqs_num_bins', 8),
                num_layers=getattr(self.model_config, 'rqs_num_layers', 2)
            )
        elif self.use_clamped_magnitude:
            self.clamped_head = nn.Sequential(
                nn.Linear(total_input_dim, self.model_config.mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.model_config.mlp_hidden_dim, 1)
            )
            with torch.no_grad():
                self.clamped_head[-1].weight.mul_(0.01)
                self.clamped_head[-1].bias.fill_(1.0)

    def set_episode(self, episode: int):
        """Update current episode for exploration scheduling."""
        self._current_episode = episode

    def _get_exploration_alpha(self, temperature: float) -> float:
        """Exploration mixing coefficient: high uniform fraction early (anti-collapse),
        annealed down later so the flow takes over."""
        # Base exploration from temperature
        temp_scalar = float(temperature) if temperature > 0 else 1.0
        
        if temp_scalar > 1.0:
            temp_exploration = max(0.0, 1.0 - (1.0 / temp_scalar))
        else:
            temp_exploration = 0.0
        
        # Annealed exploration schedule
        # Start with 20% uniform, anneal to 2% over warmup period
        warmup_progress = min(1.0, self._current_episode / self._exploration_warmup_episodes)
        
        # Cosine annealing from 0.20 to 0.02
        alpha_init = 0.15
        alpha_final = 0.05
        alpha_schedule = alpha_final + 0.5 * (alpha_init - alpha_final) * (1.0 + math.cos(math.pi * warmup_progress))
        
        # Combine with temperature-based exploration
        alpha = min(0.5, alpha_schedule + temp_exploration)
        
        return alpha

    def forward(self, frame_features, direction, neighbor_positions, o2_com,
                temperature=0, return_log_prob=False, deterministic=False):
        """
        Predict translation magnitude conditioned on direction.
        """                     
        batch_size = frame_features.shape[0]
        device = frame_features.device
        
        # Extract directional scalars
        directional_scalars = self._extract_directional_scalars(frame_features)

        # Bessel Features
        if self.use_bessel:
            if neighbor_positions is None or o2_com is None:
                bessel_coeffs = torch.zeros(batch_size, self.num_bessel, device=device, dtype=frame_features.dtype)
            else:
                if direction.shape != (batch_size, 3): 
                    raise ValueError(f"direction shape {direction.shape}")
                if neighbor_positions.dim() != 3: 
                    raise ValueError(f"neighbor_positions shape {neighbor_positions.shape}")
                if o2_com.shape != (batch_size, 3): 
                    raise ValueError(f"o2_com shape {o2_com.shape}")

                bessel_coeffs = self._compute_bessel_embedding(direction, neighbor_positions, o2_com)
            
            rqs_features = torch.cat([directional_scalars, bessel_coeffs], dim=-1)
        else:
            rqs_features = directional_scalars

        # RQS Flow Path
        if self.use_rqs_flow:
            # Dimension validation/padding
            if rqs_features.shape[-1] != self.rqs_flow.scalar_dim:
                current_dim = rqs_features.shape[-1]
                target_dim = self.rqs_flow.scalar_dim
                
                if current_dim > target_dim:
                    rqs_features = rqs_features[:, :target_dim]
                elif current_dim < target_dim:
                    padding = torch.zeros(batch_size, target_dim - current_dim, device=device)
                    rqs_features = torch.cat([rqs_features, padding], dim=-1)

            # Get exploration coefficient
            alpha = self._get_exploration_alpha(temperature)

            # Cache MIXTURE distribution for visualization
            with torch.no_grad():
                n_grid = 500
                mag_grid = torch.linspace(self.r_min, self.r_max, n_grid, device=device)
                feat_expanded = rqs_features[0].unsqueeze(0).expand(n_grid, -1)                
                
                # Flow log-prob
                log_prob_flow = self.rqs_flow.log_prob(mag_grid.unsqueeze(-1), feat_expanded)
                
                # Uniform log-prob (as tensor)
                log_prob_uniform = torch.full_like(log_prob_flow, -math.log(self.r_max - self.r_min))
                
                # Mixture log-prob (what's actually used for training)
                log_prob_mixture = torch.logaddexp(
                    math.log(1.0 - alpha + 1e-10) + log_prob_flow,
                    math.log(alpha + 1e-10) + log_prob_uniform
                )
                
                prob_mixture = torch.exp(log_prob_mixture)
                
                # Also store flow-only for comparison
                prob_flow = torch.exp(log_prob_flow)
                
                self._last_magnitude_data = {
                    'mag_grid': mag_grid.cpu().numpy().tolist(),
                    'pdf': prob_mixture.cpu().numpy().tolist(),  # Mixture (actual)
                    'pdf_flow_only': prob_flow.cpu().numpy().tolist(),  # Flow only
                    'alpha': alpha,  # Current exploration rate
                }

            if deterministic:
                torch.manual_seed(42)
                device_str = str(self.device)
                if 'cuda' in device_str:
                    torch.cuda.manual_seed_all(42)
                    
                magnitude_sample = self.rqs_flow.sample(rqs_features)
                magnitude = magnitude_sample.view(batch_size)
                log_prob = self.rqs_flow.log_prob(magnitude_sample, rqs_features).view(batch_size)
            else:
                from torch.distributions import Bernoulli, Uniform
                
                take_uniform = Bernoulli(probs=torch.full((batch_size,), alpha, device=device)).sample().bool()
                mag_flow = self.rqs_flow.sample(rqs_features)
                mag_flow_flat = mag_flow.view(batch_size)
                mag_uniform = Uniform(self.r_min, self.r_max).sample((batch_size,)).to(device)
                magnitude = torch.where(take_uniform, mag_uniform, mag_flow_flat)
                
                # Compute mixture log-prob
                logp_flow = self.rqs_flow.log_prob(magnitude.unsqueeze(-1), rqs_features).view(batch_size)
                logp_uniform = torch.full_like(logp_flow, -math.log(self.r_max - self.r_min))
                
                log_prob = torch.logaddexp(
                    torch.log(torch.tensor(1.0 - alpha, device=device)) + logp_flow,
                    torch.log(torch.tensor(alpha, device=device)) + logp_uniform
                )

            self.last_magnitude_log_prob = log_prob
            return (magnitude, log_prob) if return_log_prob else magnitude
        
        # Clamped-MLP path (non-RQS)
        else:
            self._last_magnitude_data = None 
            combined_features = rqs_features
            raw_magnitude = self.clamped_head(combined_features).squeeze(-1)
            magnitude = torch.clamp(raw_magnitude, min=self.r_min, max=self.r_max)
            
            if return_log_prob:
                return magnitude, torch.zeros_like(magnitude)
            return magnitude

    def _extract_directional_scalars(self, frame_features):
        """
        Extract direction-conditioned 0e scalars from v_hat-aligned features:
        - l=0: all scalars (invariant)
        - l=1: z'-components (prop_to cos(theta))
        - l=2: m=0 components (prop_to 3cos^2theta-1)
        All are 0e in the aligned frame (magnitude head consumes 0e only).

        Frame features layout (74 dims): 16x0e + 8x1e + 4x2e + 2x3e
        Returns [batch, 28]: 16 scalars + 8 l1_z + 4 l2_m0.
        """
        batch_size = frame_features.shape[0]
        input_dim = frame_features.shape[-1]
        
        if input_dim == 60:
            # 60-dim layout: 16x0e + 8x1e + 4x2e
            scalars = frame_features[:, :16]  # 16x0e
            l1_vectors = frame_features[:, 16:40].reshape(batch_size, 8, 3)
            l1_z = l1_vectors[:, :, 2]  # [B, 8]
            l2_tensors = frame_features[:, 40:60].reshape(batch_size, 4, 5)
            l2_m0 = l2_tensors[:, :, 2]  # [B, 4]
            # Total: 16 + 8 + 4 = 28
            return torch.cat([scalars, l1_z, l2_m0], dim=-1)
        elif input_dim == 74:
            # 74-dim layout: 16x0e + 8x1e + 4x2e + 2x3e
            scalars = frame_features[:, :16]
            l1_vectors = frame_features[:, 16:40].reshape(batch_size, 8, 3)
            l1_z = l1_vectors[:, :, 2]
            l2_tensors = frame_features[:, 40:60].reshape(batch_size, 4, 5)
            l2_m0 = l2_tensors[:, :, 2]
            l3_tensors = frame_features[:, 60:74].reshape(batch_size, 2, 7)
            l3_m0 = l3_tensors[:, :, 3]
            return torch.cat([scalars, l1_z, l2_m0, l3_m0], dim=-1)

        elif input_dim == 64:
            # 64-dim layout: 16 scalars + 16x3 vectors
            scalars = frame_features[:, :16]
            vectors = frame_features[:, 16:].reshape(batch_size, 16, 3)
            z_components = vectors[:, :, 2]
            return torch.cat([scalars, z_components], dim=-1)  # [B, 32]
        
        else:
            raise ValueError(
                f"Unexpected frame feature dim {input_dim}. "
                f"Expected 74 (16x0e+8x1e+4x2e+2x3e) or 64 (16 scalars + 16x3 vectors)."
            )

    def _compute_bessel_embedding(self, direction, neighbor_positions, o2_com):
        v_hat = F.normalize(direction, dim=-1)
        rel = neighbor_positions - o2_com.unsqueeze(1)
        dists = torch.norm(rel, dim=-1).clamp(min=1e-10)
        cos = torch.sum((rel / dists.unsqueeze(-1)) * v_hat.unsqueeze(1), dim=-1)
        
        n = self.bessel_n_values.to(dists.device).unsqueeze(0).unsqueeze(0)
        x = n * self.bessel_scale_factor * dists.unsqueeze(-1)
        j0 = torch.where(x.abs() < 1e-4, torch.ones_like(x), torch.sin(x)/x)
        basis = j0 * math.sqrt(2.0/self.cutoff_radius)
        cutoff = 0.5 * (torch.cos(math.pi * dists / self.cutoff_radius) + 1.0) * (dists < self.cutoff_radius).float()
        
        weighted = basis * cutoff.unsqueeze(-1) * (torch.clamp(cos, min=0)**4).unsqueeze(-1)
        return weighted.sum(dim=1) / (neighbor_positions.shape[1] + 1e-10)

    def get_visualization_data(self, frame_features, direction=None, sampled_magnitude=None):
        """Get visualization data for magnitude distribution."""

        if self._last_magnitude_data is not None:
            mag_grid = np.array(self._last_magnitude_data['mag_grid'])
            pdf = np.array(self._last_magnitude_data['pdf'])
            log_p_grid = np.log(np.clip(pdf, 1e-10, None))

            # Compute statistics
            dr = mag_grid[1] - mag_grid[0] if len(mag_grid) > 1 else 0.025
            pdf_norm = pdf / (np.sum(pdf) * dr + 1e-10)
            r_mean = np.sum(mag_grid * pdf_norm) * dr
            r_var = np.sum((mag_grid - r_mean)**2 * pdf_norm) * dr
            r_std = np.sqrt(max(r_var, 1e-6))

            magnitude_sample = float(sampled_magnitude.flatten()[0].item()) if sampled_magnitude is not None else r_mean

            result = {
                # Primary keys expected by visualization
                'mag_grid': mag_grid,
                'log_p_grid': log_p_grid,
                'magnitude_sample': magnitude_sample,
                'r_min': float(mag_grid[0]),
                'r_max': float(mag_grid[-1]),
                # Also include alternate names for compatibility
                'r_grid': mag_grid,
                'log_p_r': log_p_grid,
                'r_sample': magnitude_sample,
                'r_mean': float(r_mean),
                'r_std': float(r_std),
            }
            
            # Add flow-only data if available (for debugging)
            if 'pdf_flow_only' in self._last_magnitude_data:
                result['pdf_flow_only'] = np.array(self._last_magnitude_data['pdf_flow_only'])
                result['exploration_alpha'] = self._last_magnitude_data.get('alpha', 0.05)
            
            return result
        return None

    # Accessors
    def get_distribution_data(self, *args, **kwargs): pass 
    def set_temperature(self, t): pass 
    def get_magnitude_log_data(self): return None
    def verify_direction_dependence(self, frame_features, direction): return {}
    def get_diagnostic_info(self, frame_features, direction=None, neighbor_positions=None, o2_com=None):
        with torch.no_grad():
            info = {}
            magnitude = self.forward(frame_features, direction, neighbor_positions, o2_com, temperature=0)
            info['magnitude'] = magnitude[0].item()
            return info

    def compute_entropy(self, context_features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute entropy of the magnitude distribution.

        The distribution is a mixture: p(r) = (1-alpha)*p_flow(r) + alpha*p_uniform(r)
        where alpha is the exploration coefficient.

        Mixture entropy approximation:
        H_mix ~= (1-alpha)*H_flow + alpha*H_uniform + H_mixing

        Args:
            context_features: [batch, scalar_dim] features for RQS conditioning
            temperature: Sampling temperature (affects alpha)

        Returns:
            entropy: [batch] entropy in nats
        """
        batch_size = context_features.shape[0]
        device = context_features.device

        if not (hasattr(self, 'rqs_flow') and self.use_rqs_flow):
            # If not using RQS, return small fixed entropy
            return torch.zeros(batch_size, device=device)

        # Get exploration coefficient
        alpha = self._get_exploration_alpha(temperature)

        # Flow entropy (from RQS flow's entropy method)
        flow_entropy = self.rqs_flow.entropy(context_features)  # [batch]

        # Uniform entropy: H[Uniform(a,b)] = log(b-a)
        uniform_entropy = np.log(self.r_max - self.r_min)

        # Mixing entropy: -p*log(p) - (1-p)*log(1-p)
        # This captures additional uncertainty from not knowing which component
        if alpha > 0.001 and alpha < 0.999:
            mixing_entropy = -(1-alpha)*np.log(1-alpha) - alpha*np.log(alpha)
        else:
            mixing_entropy = 0.0

        # Total mixture entropy (approximate)
        # H_mix ~= (1-alpha)*H_flow + alpha*H_uniform + H_mixing
        mixture_entropy = (1 - alpha) * flow_entropy + alpha * uniform_entropy + mixing_entropy

        return mixture_entropy
