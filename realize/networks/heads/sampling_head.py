"""Equivariant sampling head for the translation direction."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from e3nn.o3 import Irreps, Linear, spherical_harmonics
from realize.networks.irrep_sequential import IrrepSequential, IrrepActivation
from realize.networks.distributions.spherical_distribution import MultiChannelSphericalDistribution
from realize.networks.heads.sampling_head_sampling import SamplingHeadSamplingMixin
import logging
_LOG = logging.getLogger(__name__)

class SamplingHead(SamplingHeadSamplingMixin, nn.Module):
    """
    E(3)-equivariant sampling head for direction distribution.
    Maps irrep features to spherical harmonic coefficients maintaining proper rotation behavior.
    """
    
    def __init__(self, model_config, sampling_config, device):
        super().__init__()

        self.model_config = model_config
        self.sampling_config = sampling_config
        self.device = device
        
        # Configuration
        self.lmax = model_config.lmax
        self.num_channels = model_config.num_channels
        self.temperature_init = sampling_config.temperature_init        
        self.temperature_min = sampling_config.temperature_min
        self.temperature_anneal_episodes = getattr(sampling_config, 'temperature_anneal_episodes', 1000)        
        self.current_temperature = self.temperature_init

        # Per-channel temperatures (NO AUTO-PAD - exact length required)
        if self.num_channels > 1:
            # Initialize channel-specific temperatures
            if not hasattr(self.model_config, 'channel_temp_scales'):
                raise ValueError(
                    f"model_config.channel_temp_scales is REQUIRED for multi-channel (num_channels={self.num_channels})"
                )
            channel_temp_scales = self.model_config.channel_temp_scales
            
            # Validate exact length
            if len(channel_temp_scales) != self.num_channels:
                raise ValueError(
                    f"channel_temp_scales length ({len(channel_temp_scales)}) must exactly match "
                    f"num_channels ({self.num_channels}); no auto-padding allowed"
                )
            
            initial_temps = torch.tensor(channel_temp_scales, device=device, dtype=torch.float32)
            
            # Validate values
            if not torch.all(torch.isfinite(initial_temps)):
                raise ValueError(f"channel_temp_scales contains non-finite values: {channel_temp_scales}")
            if torch.any(initial_temps <= 0):
                raise ValueError(f"channel_temp_scales must be positive: {channel_temp_scales}")
            
            self.channel_log_temps = nn.Parameter(torch.log(initial_temps))        
        else:
            self.channel_log_temps = None

        self.use_coeff_norm = getattr(model_config, 'normalize_sh_coeffs', False)
        
        # Input and output irreps - NO DEFAULT (must be in config)
        if not hasattr(model_config, 'embedding_irreps'):
            raise ValueError(
                "model_config.embedding_irreps is REQUIRED; no default allowed. "
                "This must match the output irreps of the frame encoder."
            )
        self.input_irreps = Irreps(model_config.embedding_irreps)

        # Build irreps for spherical harmonics up to lmax with correct parity
        # Even l -> 'e' (even parity), odd l -> 'o' (odd parity)
        sh_irreps_list = []
        for l in range(self.lmax + 1):
            parity = 'e' if l % 2 == 0 else 'o'
            sh_irreps_list.append(f"1x{l}{parity}")
        self.sh_irreps = Irreps("+".join(sh_irreps_list))
        self.sh_dim = sum(2*l + 1 for l in range(self.lmax + 1))
        
        _LOG.info(f"SH irreps structure: {self.sh_irreps}")
        _LOG.info(f"Total SH coefficients: {self.sh_dim}")

        # Compute and cache l-block boundaries
        self._compute_l_boundaries()
        
        # Simple validation
        self._validate_irreps_ordering()
        
        # Learned per-l scaling factors for normalization (if enabled)
        if self.use_coeff_norm:
            for l in range(self.lmax + 1):
                scale_init = 1.0 if l == 0 else 0.5
                self.register_parameter(f'l{l}_scale', 
                                       nn.Parameter(torch.tensor(scale_init)))

        # Precompute temperature schedule (REQUIRED - no on-the-fly fallback)
        if not hasattr(sampling_config, 'total_episodes'):
            raise ValueError(
                "sampling_config.total_episodes is REQUIRED to precompute temperature schedule. "
                "On-the-fly computation is not allowed."
            )
        if sampling_config.total_episodes <= 0:
            raise ValueError(f"total_episodes must be positive, got {sampling_config.total_episodes}")
        
        self._precompute_temperature_schedule(sampling_config.total_episodes)

        # Multi-channel architecture with proper equivariance
        if self.num_channels > 1:
            # Channel-specific equivariant networks
            self.channel_nets = nn.ModuleList([
                self._build_equivariant_sh_network() for _ in range(self.num_channels)
            ])
            
            # Gating network (scalar output for mixture weights)
            self.gate_net = IrrepSequential(
                Linear(self.input_irreps, Irreps("16x0e+4x1e")),
                IrrepActivation(Irreps("16x0e+4x1e"), acts=[F.silu, None]),
                Linear(Irreps("16x0e+4x1e"), Irreps("8x0e")),
                IrrepActivation(Irreps("8x0e"), acts=[F.silu]),
                Linear(Irreps("8x0e"), Irreps(f"{self.num_channels}x0e"))
            )
            
            _LOG.info(f"Initialized {self.num_channels} equivariant channel networks")
        else:
            # Single channel
            self.sh_net = self._build_equivariant_sh_network()
            _LOG.info(f"Initialized single equivariant SH network")
        
        # Optional: Coefficient stabilization via LayerNorm on scalars only
        if self.use_coeff_norm:
            # Only normalize l=0 (scalar) component for stability
            self.coeff_norm = nn.LayerNorm(1)
        
        # Grid configuration for sampling (NO DEFAULTS - must be explicit)
        if not hasattr(sampling_config, 'grid_type'):
            raise ValueError("sampling_config.grid_type is REQUIRED (e.g., 'gauss_legendre', 'lebedev')")
        self.grid_type = sampling_config.grid_type
        
        if hasattr(sampling_config, 'get_grid_points'):
            self.num_grid_points = sampling_config.get_grid_points()
        else:
            if not hasattr(sampling_config, 'num_grid_points'):
                raise ValueError(
                    "sampling_config.num_grid_points is REQUIRED "
                    "(or provide get_grid_points() method)"
                )
            self.num_grid_points = sampling_config.num_grid_points
        
        # Validate grid points is positive
        if self.num_grid_points <= 0:
            raise ValueError(f"num_grid_points must be positive, got {self.num_grid_points}")

        self._setup_grid()

        self.to(device)

    def _compute_l_boundaries(self):
        """Compute and cache the start/end indices for each l-block in SH coefficients."""
        self._l_boundaries = {}
        idx = 0
        for l in range(self.lmax + 1):
            size = 2 * l + 1
            self._l_boundaries[l] = (idx, idx + size)
            idx += size
        
        # Register as buffer for model saving
        boundaries_tensor = torch.tensor([[start, end] for start, end in self._l_boundaries.values()])
        self.register_buffer('l_boundaries', boundaries_tensor)
    
    def _validate_irreps_ordering(self):
        """Simple validation that SH irreps match expected structure."""
        # Since we construct sh_irreps ourselves, just verify the dimension
        expected_dim = sum(2*l + 1 for l in range(self.lmax + 1))
        actual_dim = self.sh_irreps.dim
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"SH irreps dimension mismatch. Expected {expected_dim}, got {actual_dim}. "
                "This indicates an irreps construction bug."
            )

        _LOG.info(f"Validated SH dimension: {actual_dim} for lmax={self.lmax}")

    def _precompute_temperature_schedule(self, total_episodes):
        """Precompute cosine annealing schedule for efficiency."""
        schedule = torch.zeros(total_episodes)
        for ep in range(total_episodes):
            if ep >= self.temperature_anneal_episodes:
                schedule[ep] = self.temperature_min
            else:
                progress = ep / self.temperature_anneal_episodes
                schedule[ep] = self.temperature_min + \
                    0.5 * (self.temperature_init - self.temperature_min) * \
                    (1 + math.cos(math.pi * progress))
        
        self.register_buffer('temperature_schedule', schedule)
        _LOG.info(f"Precomputed temperature schedule for {total_episodes} episodes")

    def _build_equivariant_sh_network(self):
        """
        Build an equivariant network that maps input irreps to SH coefficients.
        This ensures the output SH coefficients transform properly under rotations.
        """
        # Progressive expansion to target SH structure
        return IrrepSequential(
            # First layer: expand features
            Linear(self.input_irreps, Irreps("64x0e+32x1e+16x2e")),
            IrrepActivation(Irreps("64x0e+32x1e+16x2e"), acts=[F.silu, None, None]),
            
            # Second layer: start building higher-l components
            Linear(Irreps("64x0e+32x1e+16x2e"), 
                   Irreps("32x0e+16x1e+8x2e+4x3e") if self.lmax >= 3 else Irreps("32x0e+16x1e+8x2e")),
            IrrepActivation(
                Irreps("32x0e+16x1e+8x2e+4x3e") if self.lmax >= 3 else Irreps("32x0e+16x1e+8x2e"),
                acts=[F.silu, None, None, None] if self.lmax >= 3 else [F.silu, None, None]
            ),
            
            # Output layer: map to exact SH structure
            Linear(
                Irreps("32x0e+16x1e+8x2e+4x3e") if self.lmax >= 3 else Irreps("32x0e+16x1e+8x2e"),
                self.sh_irreps
            )
        )
    
    def _compute_normalized_masses(self, sh_coeffs):
        """
        Compute normalized per-channel masses on S^2 grid.
        
        Each channel is projected to the grid, beta-scaled for concentration,
        and normalized to integrate to 1 with quadrature weights.
        
        Args:
            sh_coeffs: [batch, num_channels, sh_dim] SH coefficients
            
        Returns:
            channel_log_mass_normalized: [batch, num_channels, P] normalized log-mass
        """
        from e3nn.o3 import Irreps, spherical_harmonics
        
        batch_size, num_channels, sh_dim = sh_coeffs.shape
        device = sh_coeffs.device
        
        # Validate SH basis exists (NO AUTO-REBUILD)
        if not hasattr(self, '_sh_grid_basis'):
            raise RuntimeError(
                "SH grid basis not initialized. Call _setup_grid() first. "
                "This indicates improper initialization order."
            )
        
        # Validate grid hasn't changed
        grid_fingerprint = self._compute_grid_fingerprint()
        
        if not hasattr(self, '_sh_grid_basis_fingerprint'):
            raise RuntimeError(
                "SH grid basis fingerprint not found. "
                "_setup_grid() may not have completed properly."
            )
        
        if not self._grids_match(self._sh_grid_basis_fingerprint, grid_fingerprint):
            raise RuntimeError(
                f"Grid identity violation: SH basis was computed on a different grid. "
                f"Expected fingerprint: {self._sh_grid_basis_fingerprint}\n"
                f"Current fingerprint:  {grid_fingerprint}\n"
                "Call _setup_grid() again to rebuild basis, or ensure grid is not modified."
            )

        Y_grid = self._sh_grid_basis  # [P, sh_dim]
        P = Y_grid.shape[0]
        
        # Get quadrature weights
        w = self.grid_config.weights
        log_w = torch.log(w.clamp(min=1e-10))
        w_sum = w.sum()
        
        # Project each channel to grid
        channel_log_mass_list = []
        
        for c in range(num_channels):
            # Project SH coefficients to grid: [batch, P]
            ch_coeffs = sh_coeffs[:, c, :]  # [batch, sh_dim]
            vals = torch.matmul(ch_coeffs, Y_grid.T)  # [batch, P]
            
            # Mean-center per batch element (weighted)
            mu = (vals * w).sum(dim=-1, keepdim=True) / w_sum  # [batch, 1]
            g = vals - mu  # [batch, P]
            
            # RMS-normalize for scale-invariance
            rms = torch.sqrt((w * (g ** 2)).sum(dim=-1, keepdim=True) / w_sum + 1e-8)
            rms = rms.clamp(min=1e-3)  # Floor to prevent extreme beta
            g_norm = g / rms  # [batch, P]
            
            # Solve beta for channel-specific concentration target (NO DEFAULTS)
            if not hasattr(self.sampling_config, 'area_targets'):
                raise ValueError(
                    "sampling_config.area_targets is REQUIRED for multi-channel sampling. "
                    "Must be a list of target areas (e.g., [0.05, 0.12, 0.25, 0.50])"
                )
            
            area_targets = torch.as_tensor(
                self.sampling_config.area_targets, 
                device=device, 
                dtype=sh_coeffs.dtype
            )
            
            # Validate exact length
            if area_targets.numel() != num_channels:
                raise ValueError(
                    f"area_targets length ({area_targets.numel()}) must exactly match "
                    f"num_channels ({num_channels})"
                )
            
            # Validate values are in valid range
            if torch.any(area_targets <= 0) or torch.any(area_targets > 1):
                raise ValueError(f"area_targets must be in (0, 1], got: {area_targets}")
            
            target_area = area_targets[c].item()            
            target_mass_fraction = 0.70
            
            # Bisection solver (vectorized over batch)
            beta_c = self._solve_beta_batch(
                g_norm.detach(),  # Detach for solver only
                w, 
                target_area, 
                target_mass_fraction
            )  # [batch]

            # Apply beta to normalized field
            logits = beta_c.unsqueeze(-1) * g_norm  # [batch, P]
            
            # Normalize to valid mass: add log(w), then normalize
            log_mass_unnorm = logits + log_w.unsqueeze(0)  # [batch, P]
            log_Z = torch.logsumexp(log_mass_unnorm, dim=-1, keepdim=True)
            log_mass_norm = log_mass_unnorm - log_Z  # [batch, P]
            
            channel_log_mass_list.append(log_mass_norm)
        
        # Stack channels: [batch, num_channels, P]
        channel_log_mass_normalized = torch.stack(channel_log_mass_list, dim=1)
   
        # Store grid fingerprint for validation
        self._grid_fingerprint = self._compute_grid_fingerprint()

        # Verify normalization (debug check)
        if self.training and hasattr(self, '_normalization_check_counter'):
            self._normalization_check_counter = getattr(self, '_normalization_check_counter', 0) + 1
            if self._normalization_check_counter % 100 == 0:
                with torch.no_grad():
                    for c in range(num_channels):
                        integral = torch.exp(channel_log_mass_normalized[:, c, :]).sum(dim=-1)
                        max_error = (integral - 1.0).abs().max().item()
                        if max_error > 0.01:
                            _LOG.warning(f"Channel {c} integration error: {max_error:.6f}")
        
        return channel_log_mass_normalized

    def _solve_beta_batch(self, g_batch, weights, target_area, target_mass_fraction):
        """Vectorized bisection to find beta for Area@Mass target."""
        batch_size = g_batch.shape[0]
        device = g_batch.device
        
        beta_lo = torch.zeros(batch_size, device=device)
        beta_hi = torch.full((batch_size,), 64.0, device=device)
        
        for iteration in range(15):  # 15 bisection iterations
            beta_mid = 0.5 * (beta_lo + beta_hi)
            
            # Compute mass at beta_mid
            logits = beta_mid.unsqueeze(-1) * g_batch
            log_mass = logits + torch.log(weights.clamp(min=1e-10)).unsqueeze(0)
            log_mass_norm = log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)
            mass = torch.exp(log_mass_norm)
            
            # Find area containing target_mass_fraction
            idx_sorted = torch.argsort(logits, dim=-1, descending=True)
            mass_sorted = torch.gather(mass, 1, idx_sorted)
            cum_mass = torch.cumsum(mass_sorted, dim=-1)
            
            # Index where cumulative mass >= target
            exceeds = cum_mass >= target_mass_fraction
            k_indices = exceeds.float().argmax(dim=-1)  # [batch]
            
            # Compute area
            P = weights.shape[0]
            mask = torch.arange(P, device=device).unsqueeze(0) <= k_indices.unsqueeze(1)
            weights_sorted = torch.gather(weights.unsqueeze(0).expand(batch_size, -1), 1, idx_sorted)
            area_current = (weights_sorted * mask).sum(dim=-1) / (4.0 * torch.pi)
            
            # Bisection update
            too_broad = area_current > target_area
            beta_lo = torch.where(too_broad, beta_mid, beta_lo)
            beta_hi = torch.where(too_broad, beta_hi, beta_mid)
        
        return 0.5 * (beta_lo + beta_hi)


    def forward(self, state_features, episode=0):
        """
        Generate spherical harmonic coefficients from state features.

        Args:
            state_features: Input features with irrep structure [batch, feature_dim]
            episode: Current episode for temperature scheduling

        Returns:
            sh_coeffs: SH coefficients [batch, num_channels, sh_dim] or [batch, sh_dim]
            channel_weights: Channel mixture weights [batch, num_channels] or None
        """
        batch_size = state_features.shape[0]

        # Ensure input is on correct device
        if state_features.device != self.device:
            state_features = state_features.to(self.device)

        # Validate input dimension (EVERY forward, not just once)
        if state_features.shape[-1] != self.input_irreps.dim:
            raise ValueError(
                f"Input features dimension ({state_features.shape[-1]}) does not match "
                f"expected irreps dimension ({self.input_irreps.dim}). "
                f"Expected irreps: {self.input_irreps}"
            )

        # Update temperature BEFORE using it for gates
        self.update_temperature(episode)
        
        if self.num_channels > 1:
            # Multi-channel: compute coefficients for each channel
            channel_coeffs = []
            for i, net in enumerate(self.channel_nets):
                coeffs = net(state_features)
                channel_coeffs.append(coeffs)
            
            # Stack channels
            sh_coeffs = torch.stack(channel_coeffs, dim=1)  # [batch, num_channels, sh_dim]
            
            # Compute gating weights with UPDATED temperature
            gate_logits = self.gate_net(state_features)  # [batch, num_channels]
            if gate_logits.dim() > 2:
                gate_logits = gate_logits.squeeze(1)
            
            # Use softmax with current (updated) temperature (NO FLOOR)
            if self.current_temperature <= 0:
                raise RuntimeError(
                    f"Current temperature is non-positive: {self.current_temperature}. "
                    "update_temperature() should have caught this."
                )
            channel_weights = F.softmax(gate_logits / self.current_temperature, dim=-1)

            # Apply coefficient normalization if enabled
            if self.use_coeff_norm:
                sh_coeffs = self._normalize_coefficients(sh_coeffs)

            # Probability-space normalization: each channel integrates to 1 on
            # the grid, so channel weights mix fairly for PPO.
            self._channel_log_mass_normalized = self._compute_normalized_masses(sh_coeffs)
            
            # Cache for reuse in sampling
            self._cached_sh_coeffs = sh_coeffs.detach()
            self._cached_channel_weights = channel_weights  

            # FAIRNESS DIAGNOSTIC (verify equal weights -> equal influence)
            if episode % 50 == 0 and self.training:
                with torch.no_grad():
                    # Compute mixture
                    log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
                    log_mix_mass = torch.logsumexp(
                        log_w_c + self._channel_log_mass_normalized, dim=1
                    )

                    # Responsibilities
                    log_resp = (
                        log_w_c + self._channel_log_mass_normalized - log_mix_mass.unsqueeze(1)
                    )

                    # Mass-weighted influence
                    mass_mix = torch.exp(log_mix_mass)
                    resp = torch.exp(log_resp)
                    influence = (resp * mass_mix.unsqueeze(1)).sum(dim=-1)  # [batch, channels]

                    # Check fairness (FAIL HARD on violations)
                    max_diff = (channel_weights - influence).abs().max().item()

                    if max_diff > 0.05:
                        raise RuntimeError(
                            f"Channel fairness violation at episode {episode}: "
                            f"max|influence - weights| = {max_diff:.4f} > 0.05\n"
                            f"  Weights:    {channel_weights[0].cpu().numpy()}\n"
                            f"  Influence:  {influence[0].cpu().numpy()}\n"
                            "This indicates the normalization is not working correctly."
                        )

            return sh_coeffs, channel_weights

        else:
            # Single channel
            sh_coeffs = self.sh_net(state_features)  # [batch, sh_dim]
        
            if self.use_coeff_norm:
                sh_coeffs = self._normalize_coefficients(sh_coeffs)
            
            # Add channel dimension for consistency
            sh_coeffs_expanded = sh_coeffs.unsqueeze(1)  # [batch, 1, sh_dim]
            
            # Probability-space normalization (single channel)
            self._channel_log_mass_normalized = self._compute_normalized_masses(sh_coeffs_expanded)
            
            # Cache
            self._cached_sh_coeffs = sh_coeffs_expanded
            self._cached_channel_weights = None



            return sh_coeffs, None

    def _normalize_coefficients(self, sh_coeffs):
        """
        Apply per-l block RMS normalization to SH coefficients while preserving equivariance.
        This prevents explosion without destroying the DC term.
        """
        device = sh_coeffs.device
        
        # Get l-block boundaries if not cached
        if not hasattr(self, '_l_boundaries'):
            self._compute_l_boundaries()
        
        if sh_coeffs.dim() == 3:  # Multi-channel [batch, channels, sh_dim]
            sh_coeffs_normalized = sh_coeffs.clone()
            
            for l, (start, end) in self._l_boundaries.items():
                # Extract l-block
                l_block = sh_coeffs[:, :, start:end]
                
                # RMS normalization without centering (preserves DC for l=0)
                rms = torch.sqrt(torch.mean(l_block**2, dim=-1, keepdim=True)).clamp(min=1e-8)
                
                # Apply learned scaling factor per l
                if hasattr(self, f'l{l}_scale'):
                    scale = getattr(self, f'l{l}_scale')
                    sh_coeffs_normalized[:, :, start:end] = scale * l_block / rms
                else:
                    sh_coeffs_normalized[:, :, start:end] = l_block / rms
            
            return sh_coeffs_normalized
        else:  # Single channel [batch, sh_dim]
            sh_coeffs_normalized = sh_coeffs.clone()
            
            for l, (start, end) in self._l_boundaries.items():
                # Extract l-block
                l_block = sh_coeffs[:, start:end]
                
                # RMS normalization
                rms = torch.sqrt(torch.mean(l_block**2, dim=-1, keepdim=True)).clamp(min=1e-8)
                
                # Apply learned scaling
                if hasattr(self, f'l{l}_scale'):
                    scale = getattr(self, f'l{l}_scale')
                    sh_coeffs_normalized[:, start:end] = scale * l_block / rms
                else:
                    sh_coeffs_normalized[:, start:end] = l_block / rms
            
            return sh_coeffs_normalized

    def _setup_grid(self):
        """Setup sampling grid configuration"""
        from realize.networks.grids.grid_factory import get_spherical_grid

        # Get appropriate number of points for testing vs training
        if self.grid_type == 'lebedev':
            # For Lebedev, use fixed points based on degree
            self.num_grid_points = 170  # degree 11
        elif hasattr(self.sampling_config, 'gl_n_beta'):
            # For GL, check if we're in test mode (small grid)
            self.num_grid_points = self.sampling_config.gl_n_beta * self.sampling_config.gl_n_alpha

        self.grid_config = get_spherical_grid(
            self.grid_type,  # Use cached grid type
            self.lmax,
            self.num_grid_points,
            device=self.device,
            gl_n_beta=getattr(self.sampling_config, 'gl_n_beta', 64),
            gl_n_alpha=getattr(self.sampling_config, 'gl_n_alpha', 128)
        )

        # BUILD SH BASIS HERE (so it exists before _compute_normalized_masses)
        grid_points_norm = F.normalize(self.grid_config.points, dim=-1)
        irreps = Irreps("+".join(f"{l}{'e' if l%2==0 else 'o'}" for l in range(self.lmax+1)))
        self._sh_grid_basis = spherical_harmonics(irreps, grid_points_norm, normalize=True)
        self._sh_grid_basis_fingerprint = self._compute_grid_fingerprint()
        _LOG.info(f"Built SH grid basis for grid with {self.num_grid_points} points")

        # Validate grid is on correct device (NO AUTO-MOVE)
        if self.grid_config.points.device != self.device:
            raise RuntimeError(
                f"Grid points on wrong device: expected {self.device}, "
                f"got {self.grid_config.points.device}. "
                "Grid factory must return tensors on the requested device."
            )
        if self.grid_config.weights.device != self.device:
            raise RuntimeError(
                f"Grid weights on wrong device: expected {self.device}, "
                f"got {self.grid_config.weights.device}. "
                "Grid factory must return tensors on the requested device."
            )

        # ALWAYS check for duplicates (not just in debug mode)
        distances = torch.cdist(self.grid_config.points, self.grid_config.points)
        distances.fill_diagonal_(float('inf'))
        min_dist = distances.min().item()
        
        if min_dist < 1e-6:
            duplicates = (distances < 1e-6).nonzero()
            num_duplicates = len(duplicates) // 2  # Symmetric matrix
            raise RuntimeError(
                f"Grid has {num_duplicates} duplicate points (min distance: {min_dist:.2e} < 1e-6). "
                f"This will cause numerical issues. Check grid construction. "
                f"First duplicate pair: {duplicates[0].tolist()}"
            )

    def _compute_grid_fingerprint(self):
        """Compute grid fingerprint for identity verification."""
        with torch.no_grad():
            return {
                'num_points': self.grid_config.points.shape[0],
                'weight_sum': self.grid_config.weights.sum().item(),
                'weight_mean': self.grid_config.weights.mean().item(),
                'first_point': tuple(self.grid_config.points[0].cpu().numpy()),
                'grid_type': self.grid_type,  # Use cached grid type
            }

    def _grids_match(self, fp1, fp2):
        """Check if two grid fingerprints match."""
        return (
            fp1['num_points'] == fp2['num_points'] and
            abs(fp1['weight_sum'] - fp2['weight_sum']) < 1e-6 and
            abs(fp1['weight_mean'] - fp2['weight_mean']) < 1e-6 and
            all(abs(a - b) < 1e-6 for a, b in zip(fp1['first_point'], fp2['first_point']))
        )

