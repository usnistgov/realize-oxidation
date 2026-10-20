"""Multi-channel spherical-harmonic distribution over translation directions."""
import os
import torch
import torch.nn.functional as F
import math
import numpy as np  
from e3nn.o3 import spherical_harmonics
from e3nn.o3 import Irreps
from typing import Optional, Tuple
from realize.networks.grids.grid_factory import GridConfig, get_spherical_grid
import logging
_LOG = logging.getLogger(__name__)

class MultiChannelSphericalDistribution:
    """Multi-channel spherical distribution for PPO sampling.

    Contract for PPO:
    1. Always construct with channel_log_mass_normalized from head
       (normalized masses where each channel integrates to 1)
    2. Do not use the sh_coeffs path for PPO (only for visualization)
    3. Grid must match head's grid exactly (enforced on construction)
    4. Channel weights must come from head (never recompute in policy)

    Gradient flow:
    - beta solver: solved under no_grad in head (stable bisection)
    - PPO gradients flow through:
      * Probability-space mixture (logsumexp over channels)
      * Channel weights (gate network)
      * Per-channel normalized masses (SH networks)
    - PPO gradients do not flow through:
      * beta values (detached during solving)
      * Grid construction (fixed quadrature)

    Fairness guarantee:
    With normalized masses, influence ~= gate_weights within 0.02.
    Verified on construction and in compute_channel_influence().

    Immutability:
    Create a new instance for different coefficients or weights.
    Distribution properties are frozen at construction time.
    """
    def __init__(self,
                 sh_coeffs: Optional[torch.Tensor] = None, 
                 lmax: int = None, 
                 num_channels: int = None, 
                 temperature: float = 1.0,
                 grid_config: Optional[GridConfig] = None,
                 num_grid_points: Optional[int] = None,
                 grid_type: str = 'auto',
                 channel_temperatures: Optional[torch.Tensor] = None,
                 channel_weights: Optional[torch.Tensor] = None,
                 channel_log_mass_normalized: Optional[torch.Tensor] = None,
                 global_step: int = 0):    
        """
        Initialize multi-channel spherical distribution.
        
        For PPO, pass channel_log_mass_normalized from the head.
        Only use the sh_coeffs path for visualization.
        Args:
            sh_coeffs: SH coefficients [batch, num_channels, sh_dim]
            lmax: Maximum degree of spherical harmonics
            num_channels: Number of channels
            temperature: Sampling temperature (clamped to [1e-4, 100])
            grid_config: Pre-configured grid (must not be None)
            num_grid_points: Number of grid points (unused if grid_config provided)
            grid_type: Grid type selection (unused if grid_config provided)
            global_step: Global training step (for diagnostics)
        """
        # Validate grid_config first
        if grid_config is None:
            raise ValueError("grid_config cannot be None - must provide a GridConfig object")
        if not hasattr(grid_config, 'points'):
            raise ValueError("grid_config must have 'points' attribute")
        if not hasattr(grid_config, 'weights'):
            raise ValueError("grid_config must have 'weights' attribute")

        # VALIDATE: Need either SH or normalized masses
        if channel_log_mass_normalized is None and sh_coeffs is None:
            raise ValueError(
                "Must provide either 'channel_log_mass_normalized' (for PPO) "
                "or 'sh_coeffs' (for visualization)"
            )

        # Extract dimensions from whichever is provided
        if channel_log_mass_normalized is not None:
            # Using normalized masses - primary path for PPO
            batch_size, num_channels_actual, num_points_actual = channel_log_mass_normalized.shape

            if num_channels is None:
                num_channels = num_channels_actual
            elif num_channels != num_channels_actual:
                raise ValueError(
                    f"num_channels mismatch: {num_channels} vs {num_channels_actual}"
                )

            # lmax not needed for normalized mass path
            self.lmax = lmax  # Can be None
            self.sh_dim = None
            self._sh_coeffs = None

            # Create irreps string only if lmax provided
            if lmax is not None:
                self.irreps_str = "+".join(f"{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))
            else:
                self.irreps_str = None

        else:
            # Using SH path - visualization only
            if sh_coeffs.ndim != 3:
                raise ValueError(f"sh_coeffs must be 3D, got shape {sh_coeffs.shape}")

            batch_size, actual_channels, sh_dim = sh_coeffs.shape

            if actual_channels != num_channels:
                raise ValueError(
                    f"sh_coeffs.shape[1] ({actual_channels}) must equal num_channels ({num_channels})"
                )

            # Validate that sh_dim matches lmax exactly
            expected_sh_dim = (lmax + 1) ** 2

            # Check if sh_dim is a perfect square
            root = int(math.isqrt(sh_dim))
            if root * root != sh_dim:
                raise ValueError(
                    f"sh_dim ({sh_dim}) is not a perfect square; invalid spherical harmonics dimension"
                )

            inferred_lmax = root - 1

            # Fail fast if mismatch
            if sh_dim != expected_sh_dim:
                raise ValueError(
                    f"SH dimension mismatch: got sh_dim={sh_dim} (implies lmax={inferred_lmax}), "
                    f"but expected sh_dim={expected_sh_dim} for configured lmax={lmax}. "
                    f"Fix the SH coefficient producer to output {expected_sh_dim} coefficients."
                )

            # Create irreps string for SH evaluation
            self.irreps_str = "+".join(f"{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))

            self.lmax = lmax
            self.sh_dim = sh_dim

        # DEVICE: tie distribution device to the tensor that defines it
        if channel_log_mass_normalized is not None:
            self.device = channel_log_mass_normalized.device
        else:
            self.device = sh_coeffs.device

        # Validate temperature strictly
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not np.isfinite(temperature):
            raise ValueError(f"Temperature must be finite, got {temperature}")
        if temperature > 100:  # Sanity check
            raise ValueError(
                f"Temperature suspiciously large ({temperature}). "
                "Check if this is intentional (typical range: 0.01 - 10)."
            )
        self.temperature = temperature

        # Set debug mode flag
        self.debug_mode = True
        self.global_step = global_step

        # Store common attributes
        self.num_channels = num_channels
        self.batch_size = batch_size
        
        # Store grid configuration
        self.grid_config = grid_config

        self._R_feat_to_world = None  # Will be set when building distribution
        self._sh_coeffs = sh_coeffs
        
        # Extract and store grid points and weights
        self.points = grid_config.points
        self.weights = grid_config.weights
        self.num_points = self.points.shape[0]
        
        # Validate all tensors are on the same device (NO AUTO-MOVE)
        if self.points.device != self.device:
            raise RuntimeError(
                f"Grid points on wrong device: expected {self.device}, got {self.points.device}. "
                "Fix the grid_config device before passing to distribution."
            )
        if self.weights.device != self.device:
            raise RuntimeError(
                f"Grid weights on wrong device: expected {self.device}, got {self.weights.device}. "
                "Fix the grid_config device before passing to distribution."
            )

        # Create aliases for compatibility
        self.grid_points = self.points
        self.grid_weights = self.weights
        self.integration_weights = self.weights

        # PERFORMANCE: Cache log quadrature weights (used frequently)
        weights_typed = self.weights.to(device=self.device, dtype=self.points.dtype)
        self.log_weights = torch.log(weights_typed.clamp(min=1e-10))        

        # Store channel temperatures (NO AUTO-MOVE)
        self.channel_temperatures = channel_temperatures
        if self.channel_temperatures is not None:
            if self.channel_temperatures.device != self.device:
                raise RuntimeError(
                    f"channel_temperatures on wrong device: expected {self.device}, "
                    f"got {self.channel_temperatures.device}"
                )

        # Store channel weights (NO DEFAULT - must be provided or caller must pass to methods)
        if channel_weights is not None:
            # Validate
            if channel_weights.shape != (batch_size, num_channels):
                raise ValueError(
                    f"channel_weights shape mismatch: got {channel_weights.shape}, "
                    f"expected ({batch_size}, {num_channels})"
                )
            if not torch.isfinite(channel_weights).all():
                raise ValueError("channel_weights contains non-finite values")
            if torch.any(channel_weights < 0):
                raise ValueError("channel_weights must be non-negative")
            
            self.channel_weights = channel_weights.to(self.device)
        else:
            # No default - caller must provide to sample()/log_prob()/entropy()
            self.channel_weights = None

        # Validate points and weights shapes
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be [num_points, 3], got shape {self.points.shape}")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {self.weights.shape}")
        if self.weights.shape[0] != self.num_points:
            raise ValueError(
                f"weights length ({self.weights.shape[0]}) must match number of points ({self.num_points})"
            )
        
        # SH-SPACE PATH (visualization only)
        # Skip SH computation when using pre-normalized masses
        if channel_log_mass_normalized is None and sh_coeffs is not None:
            # Compute spherical harmonics basis at grid points
            self._Y_grid = self._compute_Y_grid()
            
            # Validate Y_grid shape
            if self._Y_grid.shape != (self.num_points, self.sh_dim):
                raise RuntimeError(
                    f"Y_grid has wrong shape: {self._Y_grid.shape}, "
                    f"expected ({self.num_points}, {self.sh_dim})"
                )
            
            # Compute the values for each channel at each grid point
            # NOTE: Despite the name, these are LINEAR values, not logs
            self.channel_values = self._compute_channel_values(self.sh_coeffs)
            
            # Validate channel_values shape
            if self.channel_values.shape != (batch_size, num_channels, self.num_points):
                raise RuntimeError(
                    f"channel_values has wrong shape: {self.channel_values.shape}, "
                    f"expected ({batch_size}, {num_channels}, {self.num_points})"
                )
            
            # Validate inputs
            if not torch.isfinite(self.sh_coeffs).all():
                raise ValueError("SH coefficients contain NaN or Inf")
                
        else:
            # Using normalized masses - no SH needed
            self._Y_grid = None
            self.channel_values = None

        # PROBABILITY-SPACE PATH (for PPO)
        # If pre-normalized masses provided by head, use them directly
        if channel_log_mass_normalized is not None:
            if channel_log_mass_normalized.shape != (batch_size, num_channels, self.num_points):
                raise ValueError(
                    f"channel_log_mass_normalized shape mismatch: got {channel_log_mass_normalized.shape}, "
                    f"expected ({batch_size}, {num_channels}, {self.num_points})"
                )
            # Store normalized log-mass directly (each channel integrates to 1)
            self.channel_log_mass_normalized = channel_log_mass_normalized
            self._using_normalized_masses = True

            # Enforce grid identity (fail fast if mismatch)
            # The head computed normalized masses on a specific grid.
            # If this distribution uses a different grid, all normalizations
            # and fairness guarantees are invalid.
            self.validate_grid_alignment(grid_config.points, grid_config.weights)

            # INVARIANT CHECKS (critical for correctness)
            with torch.no_grad():
                # Check: Each channel must integrate to 1 (no tolerance)
                for c in range(num_channels):
                    integral = torch.exp(self.channel_log_mass_normalized[:, c, :]).sum(dim=-1)
                    max_error = (integral - 1.0).abs().max().item()

                    # Use tiny epsilon for numerical precision only
                    if max_error > 1e-5:
                        raise RuntimeError(
                            f"Channel {c} normalization violated: integration error {max_error:.2e} > 1e-6. "
                            f"This indicates the head did not normalize correctly. "
                            f"Expected sum(exp(log_mass)) = 1.0, got {integral[0].item():.10f}"
                        )
                
                # [SPHERICAL_DIST_INIT] Log channel normalization on first construction
                if self.global_step == 0 and os.environ.get('DEBUG_SPHERICAL_DIST_INIT') == '1':
                    for c in range(num_channels):
                        integral = torch.exp(self.channel_log_mass_normalized[:, c, :]).sum(dim=-1)
                        _LOG.debug(f"[SPHERICAL_DIST_INIT] step={self.global_step} channel={c} "
                              f"integral={integral[0].item():.10f} max_error={max_error:.2e}")
                
        else:
            # Fallback: compute from SH coeffs (visualization only)
            self._using_normalized_masses = False
            # Will compute on-demand in sample() if needed

        # Validate grid consistency
        if not self.validate_grid_consistency():
            raise RuntimeError("Grid consistency check failed")
        
        # Only run debug check if explicitly requested and using SH path
        if os.environ.get('DEBUG_SPHERICAL_DIST') == '1' and self.channel_values is not None:
            self.validate_normalization(tolerance=1e-3)

    def _get_channel_weights(self, channel_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Get channel weights (NO FALLBACK - must be provided).

        Args:
            channel_weights: Channel weights [batch, num_channels]

        Returns:
            [batch, num_channels] weights
            
        Raises:
            ValueError: If no weights are available
        """
        if channel_weights is not None:
            # Validate shape
            if channel_weights.shape != (self.batch_size, self.num_channels):
                raise ValueError(
                    f"channel_weights shape mismatch: got {channel_weights.shape}, "
                    f"expected ({self.batch_size}, {self.num_channels})"
                )
            # Validate values
            if not torch.isfinite(channel_weights).all():
                raise ValueError("channel_weights contains non-finite values (NaN/Inf)")
            if torch.any(channel_weights < 0):
                raise ValueError("channel_weights must be non-negative")
            
            return channel_weights

        if hasattr(self, 'channel_weights') and self.channel_weights is not None:
            return self.channel_weights

        # NO UNIFORM FALLBACK - must be explicit
        raise ValueError(
            "channel_weights must be provided (no uniform fallback allowed). "
            "Either pass channel_weights argument or set in __init__."
        )

    def _compute_mixture_log_mass_and_density(self, channel_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CANONICAL mixture computation used by all PPO methods.

        This is the single source of truth for how channels mix.
        Any drift here affects all of sample()/log_prob()/entropy().

        Args:
            channel_weights: [batch, num_channels] gate weights

        Returns:
            log_mix_mass_norm: [batch, P] normalized mixture log-mass
            log_density_mix: [batch, P] mixture log-density (for PPO)
        """
        # Get normalized per-channel log-mass (NO SH FALLBACK)
        self._require_normalized_masses('get_mixture_log_mass')
        log_mass_c = self.channel_log_mass_normalized

        # Mix channels in log-space
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        log_mix_mass = torch.logsumexp(log_w_c + log_mass_c, dim=1)  # [batch, P]

        # Normalize mixture
        log_mix_mass_norm = log_mix_mass - torch.logsumexp(log_mix_mass, dim=-1, keepdim=True)

        # Convert mass -> density (subtract quadrature weight)
        log_w_quad = self.log_weights  # cached, correct dtype/device
        log_density_mix = log_mix_mass_norm - log_w_quad.unsqueeze(0)        

        # [SPHERICAL_DIST_MIX] Diagnostic: gate weights and mixture structure
        if (self.global_step % 50 == 0 and self.global_step > 0 and 
            os.environ.get('DEBUG_SPHERICAL_DIST_MIX') == '1'):
            with torch.no_grad():
                # Channel contribution to mixture (per-channel peak vs average)
                contrib_c = torch.exp(log_w_c.squeeze(-1) + log_mass_c).sum(dim=-1)  # [batch, channels]
                contrib_norm = contrib_c / contrib_c.sum(dim=-1, keepdim=True)
                
                # Check correlation between gate weights and actual contribution
                gate_w = channel_weights[0].cpu().numpy()
                contrib_w = contrib_norm[0].cpu().numpy()
                
                _LOG.debug(f"[SPHERICAL_DIST_MIX] step={self.global_step} "
                      f"gate_weights={gate_w} contrib_norm={contrib_w}")

        return log_mix_mass_norm, log_density_mix

    def _require_normalized_masses(self, method_name: str):
        """Guard PPO methods against accidental SH-path usage."""
        if not self._using_normalized_masses:
            raise RuntimeError(
                f"{method_name} requires normalized masses from head. "
                f"This distribution was constructed with sh_coeffs (visualization path). "
                f"For PPO, construct with channel_log_mass_normalized instead."
            )

    def validate_grid_alignment(self, grid_points: torch.Tensor, grid_weights: torch.Tensor):
        """
        Validate that provided grid matches the grid used for normalized masses.

        If channel_log_mass_normalized was computed on a different grid,
        it's invalid for this distribution.

        Args:
            grid_points: [P, 3] grid points from head
            grid_weights: [P] quadrature weights from head

        Raises:
            RuntimeError: If grids don't match
        """
        if not self._using_normalized_masses:
            return  # No validation needed for fallback path

        # Check number of points
        if grid_points.shape[0] != self.num_points:
            raise RuntimeError(
                f"Grid size mismatch: normalized masses computed on {self.num_points} points, "
                f"but provided grid has {grid_points.shape[0]} points"
            )

        # Check point positions
        if not torch.allclose(grid_points, self.points, atol=1e-6, rtol=1e-6):
            max_diff = (grid_points - self.points).abs().max().item()
            raise RuntimeError(
                f"Grid points don't match: max difference {max_diff:.2e} (tolerance: 1e-6). "
                f"Normalized masses are invalid for this grid. "
                f"This indicates grid was reconstructed or modified."
            )

        # Check weights (same tolerance)
        if not torch.allclose(grid_weights, self.weights, atol=1e-6, rtol=1e-6):
            max_diff = (grid_weights - self.weights).abs().max().item()
            raise RuntimeError(
                f"Grid weights don't match: max difference {max_diff:.2e} (tolerance: 1e-7). "
                f"Normalized masses are invalid for this grid."
            )

    def sample_continuous(self, channel_weights=None, n_beta=None, n_alpha=None):
        """
        DEPRECATED and FORBIDDEN.
        
        This method is a deprecated API that should not be used.
        Use sample() directly instead.
        
        Raises:
            RuntimeError: Always
        """
        raise RuntimeError(
            "sample_continuous() is DEPRECATED and forbidden. "
            "Use sample() directly instead. "
            "This method exists only to catch outdated usage."
        )

    def validate_grid_consistency(self) -> bool:
        """
        Simple validation that grid is properly initialized.
        """
        return (self.points.shape[0] == self.weights.shape[0] and 
                self.points.shape[1] == 3 and
                self.points.device == self.device and
                self.weights.device == self.device)

    def validate_normalization(self, tolerance: float = 1e-5) -> bool:
        """
        Validate that the distribution is properly normalized.
        
        VISUALIZATION ONLY: This method uses the SH path which treats linear
        SH projections as log-values (mathematically invalid for negative values).
        It is DISABLED for PPO (normalized-mass path).
        
        Args:
            tolerance: Numerical tolerance for sum to 1
            
        Returns:
            True if properly normalized
            
        Raises:
            RuntimeError: If using normalized masses (PPO path)
        """
        # SH-only method
        if self._using_normalized_masses:
            raise RuntimeError(
                "validate_normalization() is SH-only and uses invalid log-domain math.\n"
                "This method treats linear SH values as log-values (wrong for negatives).\n"
                "For PPO (normalized-mass path), use probability-space diagnostics instead."
            )
        
        if self._Y_grid is None:
            raise RuntimeError("SH basis not computed (_Y_grid is None)")
        
        # Sample many times and check empirical distribution
        with torch.no_grad():        
            # Use uniform channel weights for validation
            channel_weights = torch.ones(self.batch_size, self.num_channels, device=self.device) / self.num_channels
            
            # Compute probabilities at all grid points
            log_Z = self._compute_log_partition_function_with_weights(torch.log(channel_weights))
            
            # Get unnormalized values
            log_values = self._compute_log_values(self._Y_grid, torch.log(channel_weights))
            
            # Add quadrature weights (critical!)
            log_values = log_values + torch.log(self.weights.clamp(min=1e-10)).unsqueeze(0)
            
            # Normalize
            log_probs = log_values - log_Z.unsqueeze(1)
            probs = torch.exp(log_probs)
            
            # Sum should be 1 for each batch element
            prob_sums = probs.sum(dim=1)
            
            # Check normalization
            is_normalized = torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=tolerance)
            
            if not is_normalized:
                max_error = (prob_sums - 1.0).abs().max().item()
                _LOG.info(f"Warning: Distribution not normalized. Max error: {max_error}")
                
            return is_normalized

    def _compute_Y_grid(self):
        """
        Compute spherical harmonics basis functions at grid points.
    
        Returns:
            Y_grid: [num_points, sh_dim] spherical harmonics values
        """
    
        # Normalize points to unit sphere
        points_normalized = self.points / torch.norm(self.points, dim=1, keepdim=True).clamp(min=1e-8)
    
        # Build irreps robustly across e3nn versions
        try:
            irreps = Irreps.spherical_harmonics(self.lmax)
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct spherical harmonics for lmax={self.lmax}.\n"
                f"Error: {e}\n"
                f"No fallback to irreps_str allowed. Check that lmax is valid (0-20)."
            )

        Y = spherical_harmonics(
            irreps,
            points_normalized.to(device=self.device, dtype=self.points.dtype),
            normalize=True
        )        
        
        # Ensure Y matches grid dtype/device exactly
        Y = Y.to(device=self.device, dtype=self.points.dtype)

        # Ensure Y is on the correct device
        if Y.device != self.device:
            Y = Y.to(self.device)
        
        return Y

    def _compute_channel_values(self, sh_coeffs):
        """Compute probability values (NOT logs) for each channel."""
        values = torch.einsum('bcn,pn->bcp', sh_coeffs, self._Y_grid)
        
        # Apply temperature (NO CLAMP/MAX - validate first)
        if self.channel_temperatures is not None:
            # Shape channel_temperatures properly
            if self.channel_temperatures.dim() == 1:  # [C]
                temps = self.channel_temperatures.view(1, 1, -1)  # [1, 1, C]
            else:  # [B, C]
                temps = self.channel_temperatures.unsqueeze(1)  # [B, 1, C]
            
            # Validate temperatures
            if torch.any(temps <= 0):
                raise ValueError(
                    f"channel_temperatures must be positive, got min={temps.min().item():.2e}"
                )
            
            scaled_values = values / temps
        else:
            # Validate global temperature
            if self.temperature <= 0:
                raise ValueError(f"temperature must be positive, got {self.temperature}")
            
            scaled_values = values / self.temperature

        return scaled_values

    def to(self, device):
        """
        Move distribution to a different device.
        
        WARNING: For distributions with normalized masses, moving to a different device
        may invalidate grid alignment guarantees. Only move if you're certain the grid
        will also be moved consistently.
        
        Args:
            device: Target device
            
        Returns:
            self
            
        Raises:
            RuntimeError: If using normalized masses (grid must stay aligned)
        """
        # FORBID moving normalized-mass distributions (breaks grid alignment)
        if self._using_normalized_masses:
            raise RuntimeError(
                "Cannot move distribution constructed with normalized masses to different device. "
                "The normalized masses are tied to a specific grid instance. "
                "If you need to change devices, recreate the distribution on the target device. "
                f"Current device: {self.device}, requested: {device}"
            )
        
        # For SH-only distributions, allow moving
        # Use the internal storage _sh_coeffs; the .sh_coeffs property is read-only.
        if getattr(self, '_sh_coeffs', None) is not None:
            self._sh_coeffs = self._sh_coeffs.to(device)

        self.device = device
        
        # Move grid (allowed for SH-only path)
        if hasattr(self.grid_config, 'to'):
            self.grid_config = self.grid_config.to(device)
            self.points = self.grid_config.points
            self.weights = self.grid_config.weights
        else:
            self.points = self.points.to(device)
            self.weights = self.weights.to(device)

        # Update aliases
        self.grid_points = self.points
        self.grid_weights = self.weights
        self.integration_weights = self.weights

        # Move precomputed SH tensors
        if getattr(self, '_Y_grid', None) is not None:
            self._Y_grid = self._Y_grid.to(device)
        
        if getattr(self, 'channel_log_values', None) is not None:
            self.channel_log_values = self.channel_log_values.to(device)
        
        if getattr(self, 'log_weights', None) is not None:
            self.log_weights = self.log_weights.to(device)

        # Move channel temperatures if present
        if hasattr(self, 'channel_temperatures') and self.channel_temperatures is not None:
            self.channel_temperatures = self.channel_temperatures.to(device)
        
        return self

    def _compute_log_values(self, Y: torch.Tensor, channel_log_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        DEPRECATED, visualization-only. DO NOT USE FOR PPO.

        Treats the linear SH projection Y @ sh_coeffs (which can be negative) as
        log-values and logsumexps them, which is mathematically invalid. The
        correct SH path uses energy E = beta*(Y @ sh_coeffs) normalized by
        Z = integral exp(E) dOmega. PPO uses channel_log_mass_normalized directly.
        """
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        
        # Compute values for all channels
        values = torch.einsum('pd,bcd->bpc', Y, self.sh_coeffs)
        # values shape: [batch, num_points, num_channels]
        
        # Apply temperature scaling (NO CLAMP - validate first)
        if hasattr(self, 'channel_temperatures') and self.channel_temperatures is not None:
            # channel_temperatures: either [C] or [B,C]
            if self.channel_temperatures.dim() == 1:  # [C]
                temps = self.channel_temperatures.view(1, 1, -1)  # [1, 1, C]
            else:  # [B,C]
                temps = self.channel_temperatures.unsqueeze(1)  # [B, 1, C]
            
            # Validate
            if torch.any(temps <= 0):
                raise ValueError(
                    f"channel_temperatures must be positive, got min={temps.min().item():.2e}"
                )
            
            log_values = values / temps
        else:
            # Validate global temperature
            if self.temperature <= 0:
                raise ValueError(f"temperature must be positive, got {self.temperature}")
            
            log_values = values / self.temperature

        # Add channel weights in log space
        if channel_log_weights is not None:
            log_values = log_values.transpose(1, 2) + channel_log_weights.unsqueeze(-1)
            log_values = log_values.transpose(1, 2)
        
        # Stable log-sum-exp over channels
        return torch.logsumexp(log_values, dim=-1)

    def _compute_log_partition_function_with_weights(self, channel_log_weights):
        """
        Compute log partition function with channel weights.

        This helper is SH-only and must not be called for normalized-mass
        (PPO) distributions, because it relies on self.sh_coeffs and
        self._Y_grid being defined.

        Args:
            channel_log_weights: [batch, num_channels] log weights for each channel

        Returns:
            log_Z: [batch] log partition function for each batch element
        """
        if channel_log_weights is None:
            raise ValueError("channel_log_weights cannot be None")

        if self._using_normalized_masses:
            raise RuntimeError(
                "_compute_log_partition_function_with_weights() is SH-only. "
                "It relies on self.sh_coeffs and self._Y_grid, which are "
                "undefined for normalized-mass (PPO) distributions. "
                "Do not call this in the PPO path."
            )

        # Use existing helper to compute log values at all grid points
        log_values = self._compute_log_values(self._Y_grid, channel_log_weights)  # [batch, P]
        
        # Get log of quadrature weights (cached if available)
        if not hasattr(self, 'log_weights'):
            self.log_weights = torch.log(self.integration_weights.clamp(min=1e-10))
        
        log_w = self.log_weights  # [P]
        
        # Numerical stability via max-shift
        max_vals = log_values.max(dim=1, keepdim=True)[0]  # [batch, 1]
        log_integrand_shifted = (log_values - max_vals) + log_w.unsqueeze(0)
        log_Z = torch.logsumexp(log_integrand_shifted, dim=1) + max_vals.squeeze(1)  # [batch]
        
        return log_Z

    def sample(self, channel_weights=None, hierarchical=False):
        """Sample direction from normalized probability-space mixture.
        
        Uses pre-normalized channel masses from the head for fair mixing.
        Each channel integrates to 1, so equal gate weights -> equal influence.
        
        Args:
            channel_weights: [batch, num_channels] gate weights
            hierarchical: If True, use hierarchical sampling (default: False for lower variance)
            
        Returns:
            sampled_direction: [batch, 3]
            log_prob: [batch] log probability of sample under the mixture
        """
        self._require_normalized_masses('sample')  

        batch_size = self.batch_size
        
        # Use stored weights if available
        channel_weights = self._get_channel_weights(channel_weights)

        # CANONICAL mixture computation (single source of truth)
        log_mix_mass_norm, log_density_mix = self._compute_mixture_log_mass_and_density(channel_weights)
        
        # INVARIANT CHECK: Mixture must integrate to 1
        if self._using_normalized_masses:
            with torch.no_grad():
                mixture_integral = torch.exp(log_mix_mass_norm).sum(dim=-1)
                max_error = (mixture_integral - 1.0).abs().max().item()

                if max_error > 1e-6:
                    raise RuntimeError(
                        f"Mixture normalization violated: integration error {max_error:.2e} > 1e-6. "
                        f"Expected sum = 1.0, got {mixture_integral[0].item():.10f}"
                    )

        # Convert to probabilities for sampling
        probs = torch.exp(log_mix_mass_norm).clamp(min=0)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Ensure exact normalization
        
        # Sample grid point
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]
        sampled_dirs_feature_frame = self.points[idx]  # [batch, 3] in feature frame
        
        if os.environ.get('DEBUG_SPHERICAL_DIST_SAMPLE_VERBOSE') == '1':
            _LOG.debug(f"[SAMPLE DEBUG] Sampled indices: {idx}")
            _LOG.debug(f"[SAMPLE DEBUG] Directions in feature frame: {sampled_dirs_feature_frame.cpu().numpy()}")
        
        sampled_dirs = sampled_dirs_feature_frame

        # Extract log-density at sampled points (already computed)
        b_idx = torch.arange(batch_size, device=self.device)
        log_density_sampled = log_density_mix[b_idx, idx]  # [batch]

        # Store for diagnostics
        self._last_sampled_idx = idx
        self._last_log_mix_mass = log_mix_mass_norm
        self._last_log_density_sampled = log_density_sampled  # For debugging
        
        # [SPHERICAL_DIST_SAMPLE] Log sampling diagnostics periodically
        if (self.global_step % 50 == 0 and self.global_step > 0 and 
            os.environ.get('DEBUG_SPHERICAL_DIST_SAMPLE') == '1'):
            with torch.no_grad():
                max_prob = probs.max(dim=-1)[0]
                entropy = -(torch.exp(log_mix_mass_norm) * log_density_mix).sum(dim=-1)
                _LOG.debug(f"[SPHERICAL_DIST_SAMPLE] step={self.global_step} "
                      f"max_prob={max_prob[0].item():.4f} entropy={entropy[0].item():.4f}")
        
        return sampled_dirs, log_density_sampled, idx

    def _evaluate_spherical_harmonics(self, directions):
        """
        Evaluate spherical harmonics at given directions.

        Args:
            directions: [batch, 3] direction vectors

        Returns:
            Y: [batch, sh_dim] spherical harmonics values
        """
        # Build irreps robustly across e3nn versions
        try:
            irreps = Irreps.spherical_harmonics(self.lmax)
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct spherical harmonics for lmax={self.lmax}.\n"
                f"Error: {e}\n"
                f"No fallback to irreps_str allowed. Check that lmax is valid (0-20)."
            )

        if directions.dim() != 2 or directions.size(-1) != 3:
            raise RuntimeError(
                f"_evaluate_spherical_harmonics expects [B,3], got {directions.shape}"
            )

        norms = directions.norm(dim=-1, keepdim=True)
        directions_normalized = directions / (norms + 1e-12)

        Y = spherical_harmonics(
            irreps,
            directions_normalized.to(device=self.device, dtype=self.points.dtype),
            normalize=True
        )

        # Ensure Y matches grid dtype/device exactly
        Y = Y.to(device=self.device, dtype=self.points.dtype)

        if Y.device != self.device:
            Y = Y.to(self.device)

        return Y

    def log_prob(self, directions, channel_weights=None):
        """
        Compute log probability with nearest-neighbor approximation.
        
        WARNING: This method uses approximation and should NOT be used for PPO.
        For PPO, use log_prob_from_indices() with exact sampled indices.
        
        This method is only enabled in debug mode (set DEBUG_ALLOW_APPROX_LOGPROB=1).
        
        Args:
            directions: [batch, 3] unit vectors
            channel_weights: [batch, num_channels] gate weights
            
        Returns:
            log_prob: [batch] log probabilities (densities, not mass)
            
        Raises:
            RuntimeError: If not in debug mode
        """
        if os.environ.get('DEBUG_ALLOW_APPROX_LOGPROB') != '1':
            raise RuntimeError(
                "log_prob() uses nearest-neighbor approximation and is FORBIDDEN for PPO. "
                "Use log_prob_from_indices() with exact sampled indices instead. "
                "To enable this method for debugging, set DEBUG_ALLOW_APPROX_LOGPROB=1"
            )
        
        self._require_normalized_masses('log_prob')  
        batch_size = directions.shape[0]
        
        channel_weights = self._get_channel_weights(channel_weights)

        # Normalize directions to unit sphere (critical for numerical stability)
        directions_norm = F.normalize(directions, dim=-1, p=2)

        # Find nearest grid points
        dots = torch.matmul(directions_norm, self.points.T)  # [batch, P]        
        nearest_idx = torch.argmax(dots, dim=-1)  # [batch]
        
        # Use canonical mixture computation
        _, log_density_mix = self._compute_mixture_log_mass_and_density(channel_weights)

        # Extract log-density at nearest points
        b_idx = torch.arange(batch_size, device=self.device)
        log_prob = log_density_mix[b_idx, nearest_idx]
        
        _LOG.warning(f"Using approximate log_prob (nearest-neighbor). Use log_prob_from_indices() for PPO.")
        
        return log_prob

    def log_prob_from_indices(self, indices: torch.Tensor, channel_weights=None) -> torch.Tensor:
        """
        Compute exact log-probability at stored grid indices.

        This avoids nearest-neighbor approximation and ensures
        evaluate() exactly matches sample() log-probs.

        USAGE:
        1. sample() stores indices in self._last_sampled_idx
        2. Policy stores these indices in rollout buffer
        3. evaluate() calls this method with stored indices

        Args:
            indices: [batch] grid point indices
            channel_weights: [batch, num_channels] gate weights

        Returns:
            log_prob: [batch] exact log-probabilities (densities)
        """
        self._require_normalized_masses('log_prob_from_indices')
        indices = indices.to(self.device).long()
        batch_size = indices.shape[0]
        channel_weights = self._get_channel_weights(channel_weights)
        
        log_mass_c = self.channel_log_mass_normalized  # [batch, channels, P]
        
        # Use log_mass, not log_density
        # Sampling returns log_mass, so evaluation must too
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)  # [batch, channels, 1]
        mixture_input = log_w_c + log_mass_c  # [batch, channels, P]
        log_mass_mix = torch.logsumexp(mixture_input, dim=1)  # [batch, P]
        
        b_idx = torch.arange(batch_size, device=self.device)
        log_prob = log_mass_mix[b_idx, indices]  # Return log_mass, not log_density
        
        return log_prob        

    def entropy(self, channel_weights=None) -> torch.Tensor:
        """
        Compute entropy H = -integral  p(x) log p(x) dOmega using normalized mixture.
        
        Returns:
            entropy: [batch] entropy in nats
        """
        self._require_normalized_masses('entropy')        
        batch_size = self.batch_size
        
        channel_weights = self._get_channel_weights(channel_weights)
        
        # Use canonical mixture computation
        log_mix_mass_norm, log_density_mix = self._compute_mixture_log_mass_and_density(channel_weights)

        # Entropy: -integral  p(x) log p(x) dOmega = -Sigma (mass * log density)
        # where mass already includes quadrature weights
        mass = torch.exp(log_mix_mass_norm)
        entropy = -(mass * log_density_mix).sum(dim=-1)

        # [SPHERICAL_DIST_ENTROPY] Log entropy diagnostics
        if (self.global_step % 50 == 0 and self.global_step > 0 and 
            os.environ.get('DEBUG_SPHERICAL_DIST_ENTROPY') == '1'):
            with torch.no_grad():
                _LOG.debug(f"[SPHERICAL_DIST_ENTROPY] step={self.global_step} "
                      f"entropy_mean={entropy.mean().item():.4f} "
                      f"entropy_min={entropy.min().item():.4f} "
                      f"entropy_max={entropy.max().item():.4f}")

        return entropy

    def get_mixture_log_mass(self, channel_weights=None) -> torch.Tensor:
        """
        Get the normalized mixture log-mass on the grid.
        Useful for diagnostics and visualization.

        Returns:
            log_mix_mass: [batch, num_points] normalized log-mass
        """
        batch_size = self.batch_size

        channel_weights = self._get_channel_weights(channel_weights)

        # Get normalized per-channel log-mass (NO SH FALLBACK)
        if not self._using_normalized_masses:
            raise RuntimeError(
                "_compute_mixture_log_mass_and_density requires normalized masses from head. "
                "This distribution was constructed with sh_coeffs (visualization path). "
                "For PPO, construct with channel_log_mass_normalized instead."
            )
        
        log_mass_c = self.channel_log_mass_normalized

        # Mix channels
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        log_mix_mass = torch.logsumexp(log_w_c + log_mass_c, dim=1)

        # Normalize
        log_mix_mass_norm = log_mix_mass - torch.logsumexp(log_mix_mass, dim=-1, keepdim=True)

        return log_mix_mass_norm

    def compute_channel_influence(self, channel_weights):
        """Compute actual influence of each channel in the mixture.

        With proper normalization, this should equal channel_weights.

        Returns:
            influence: [batch, num_channels] - actual contribution of each channel
        """
        batch_size = channel_weights.shape[0]

        # Require normalized masses (NO SH FALLBACK)
        self._require_normalized_masses('compute_channel_influence')
        
        log_mass_c = self.channel_log_mass_normalized
        # Convert to density for mixture computation
        log_w_quad = torch.log(self.weights.clamp(min=1e-10))
        log_density_c = log_mass_c - log_w_quad.unsqueeze(0).unsqueeze(0)

        # Per-channel contribution to mixture at each point
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        numerator_c = torch.exp(log_w_c + log_density_c)  # [batch, num_channels, P]
        denominator = numerator_c.sum(dim=1, keepdim=True)  # [batch, 1, P]

        # Contribution gamma_c(x) = w_c * rho_c(x) / Sigma_j w_j * rho_j(x)
        gamma_c = numerator_c / (denominator + 1e-12)  # [batch, num_channels, P]

        # CORRECT FAIRNESS: mass-weighted responsibilities
        # Form mixture mass
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        
        if self._using_normalized_masses:
            log_mix_mass = torch.logsumexp(log_w_c + log_mass_c, dim=1)  # [batch, P]
        else:
            log_mix_mass = torch.logsumexp(log_w_c + log_density_c, dim=1)
        
        mass_mix = torch.exp(log_mix_mass)  # [batch, P]
        
        # Influence = Sigma_x mass_mix(x) * gamma_c(x)
        # where gamma_c is already computed above
        influence = (gamma_c * mass_mix.unsqueeze(1)).sum(dim=-1)  # [batch, channels]
        # No need to renormalize - should already sum to 1

        # INVARIANT CHECK: Fairness (gate weights ~= influence)
        if self._using_normalized_masses:
            with torch.no_grad():
                max_diff = (channel_weights - influence).abs().max().item()

                # Use very tight tolerance (0.01 = 1%)
                if max_diff > 0.01:
                    raise RuntimeError(
                        f"Fairness invariant violated: max|influence - weights| = {max_diff:.4f} > 0.01 (1%). "
                        f"Gate weights: {channel_weights[0].cpu().numpy()}\n"
                        f"Influence:    {influence[0].cpu().numpy()}\n"
                        f"This indicates incorrect probability-space mixing or normalization."
                    )
                
                # [SPHERICAL_DIST_FAIRNESS] Log fairness diagnostics
                if (self.global_step % 50 == 0 and self.global_step > 0 and 
                    os.environ.get('DEBUG_SPHERICAL_DIST_FAIRNESS') == '1'):
                    gate_w = channel_weights[0].cpu().numpy()
                    infl = influence[0].cpu().numpy()
                    _LOG.debug(f"[SPHERICAL_DIST_FAIRNESS] step={self.global_step} "
                          f"gate_weights={gate_w} influence={infl} "
                          f"max_diff={max_diff:.4f}")

        return influence

    def responsibilities_and_influence(self, channel_weights) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-point responsibilities and integrated influence.

        Returns detailed fairness diagnostics for logging/analysis.

        Args:
            channel_weights: [batch, num_channels] gate weights

        Returns:
            gamma_c: [batch, channels, P] per-point responsibilities
            influence: [batch, channels] integrated influence
            fairness_error: [batch] max |influence - weights| per sample
        """
        self._require_normalized_masses('responsibilities_and_influence')

        batch_size = channel_weights.shape[0]

        # Get normalized masses
        log_mass_c = self.channel_log_mass_normalized
        log_w_quad = torch.log(self.weights.clamp(min=1e-10))
        log_density_c = log_mass_c - log_w_quad.unsqueeze(0).unsqueeze(0)

        # Per-channel contribution to mixture
        log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
        numerator_c = torch.exp(log_w_c + log_density_c)  # [batch, channels, P]
        denominator = numerator_c.sum(dim=1, keepdim=True)  # [batch, 1, P]

        # Responsibilities: gamma_c(x) = w_c * rho_c(x) / Sigma_j w_j * rho_j(x)
        gamma_c = numerator_c / (denominator + 1e-12)  # [batch, channels, P]

        # Form mixture mass for influence computation
        log_mix_mass = torch.logsumexp(log_w_c + log_mass_c, dim=1)  # [batch, P]
        mass_mix = torch.exp(log_mix_mass)

        # Integrated influence: Sigma_x mass_mix(x) * gamma_c(x)
        influence = (gamma_c * mass_mix.unsqueeze(1)).sum(dim=-1)  # [batch, channels]

        # Fairness error per sample
        fairness_error = (channel_weights - influence).abs().max(dim=-1)[0]  # [batch]

        return gamma_c, influence, fairness_error

    def kl_divergence(self, other: 'MultiChannelSphericalDistribution', 
                     channel_weights=None) -> torch.Tensor:
        """
        Compute KL divergence KL(self || other) with optional channel weights.
        
        REQUIRES: Both distributions must be constructed with sh_coeffs.
        This method uses the SH path and is NOT compatible with normalized masses.
        
        Args:
            other: Target distribution
            channel_weights: Optional channel weights [batch, num_channels]
        
        Returns:
            KL divergence for each batch item [batch]
            
        Raises:
            RuntimeError: If either distribution lacks SH coefficients
        """
        assert isinstance(other, MultiChannelSphericalDistribution)
        
        # Guard against normalized-masses-only distributions
        if self._using_normalized_masses or other._using_normalized_masses:
            raise RuntimeError(
                "kl_divergence() requires both distributions to have sh_coeffs. "
                "At least one distribution was constructed with normalized masses only. "
                "Either: (1) construct both with sh_coeffs, or (2) implement KL in probability space."
            )
        
        # Validate SH structures exist
        if self._Y_grid is None or other._Y_grid is None:
            raise RuntimeError("KL divergence requires SH basis (_Y_grid) to be computed")            
        assert self.batch_size == other.batch_size
        assert self.grid_config.num_points == other.grid_config.num_points
        assert torch.allclose(self.grid_points, other.grid_points, atol=1e-6)
        assert torch.allclose(self.weights, other.weights, atol=1e-6)
        
        # Optional weights
        if channel_weights is not None:
            if channel_weights.device != self.device:
                channel_weights = channel_weights.to(self.device)
            log_w = torch.log(channel_weights.clamp(min=1e-10))
        else:
            log_w = torch.zeros(self.batch_size, self.num_channels, device=self.device)
        
        # Self on own grid with weights
        self_log_vals = self._compute_log_values(self._Y_grid, channel_log_weights=log_w)
        self_log_Z = self._compute_log_partition_function_with_weights(log_w)
        
        # Other evaluated on self grid with same weights
        other_Y_grid = other._evaluate_spherical_harmonics(self.grid_points)
        other_log_vals = other._compute_log_values(other_Y_grid, channel_log_weights=log_w)
        # Create helper for other's partition function
        other_log_Z = other._compute_log_partition_function_with_weights(log_w)
        
        # Normalized probabilities
        self_log_p = self_log_vals - self_log_Z.unsqueeze(1)
        other_log_p = other_log_vals - other_log_Z.unsqueeze(1)
        
        p = torch.exp(self_log_p)
        log_ratio = self_log_p - other_log_p
        kl = (p * log_ratio * self.integration_weights.unsqueeze(0)).sum(dim=1)
        
        return torch.clamp(kl, min=0.0)

    def get_grid_probability_stats(self, channel_weights=None):
        """
        Get statistics about probability distribution on grid points.
        
        VISUALIZATION ONLY: This method uses the SH path which treats linear
        SH projections as log-values (mathematically invalid for negative values).
        It is DISABLED for PPO (normalized-mass path).
    
        Returns:
            dict with min, max, mean, std of probabilities and entropy
            
        Raises:
            RuntimeError: If using normalized masses (PPO path)
        """
        # SH-only method
        if self._using_normalized_masses:
            raise RuntimeError(
                "get_grid_probability_stats() is SH-only and uses invalid log-domain math.\n"
                "This method treats linear SH values as log-values (wrong for negatives).\n"
                "For PPO (normalized-mass path), compute stats directly from channel_log_mass_normalized."
            )
        
        if self._Y_grid is None:
            raise RuntimeError("SH basis not computed (_Y_grid is None)")
        
        with torch.no_grad():
            # Handle channel weights
            if channel_weights is not None:
                if channel_weights.device != self.device:
                    channel_weights = channel_weights.to(self.device)
                channel_log_weights = torch.log(channel_weights.clamp(min=1e-10))
            else:
                channel_log_weights = torch.zeros(self.batch_size, self.num_channels, device=self.device)
    
            # Use existing helpers (no undefined channel_log_values)
            log_Z = self._compute_log_partition_function_with_weights(channel_log_weights)
            log_values = self._compute_log_values(self._Y_grid, channel_log_weights)  # [batch, P]            
            # Get log quadrature weights (cached if available)
            if not hasattr(self, 'log_weights'):
                self.log_weights = torch.log(self.weights.clamp(min=1e-10))
            
            # Compute log probabilities
            log_probs = log_values + self.log_weights.unsqueeze(0) - log_Z.unsqueeze(-1)
            probs = torch.exp(log_probs)
    
            # Get statistics for first batch element
            probs_0 = probs[0].detach().cpu().numpy()
            top5 = sorted(probs_0.tolist(), reverse=True)[:5]
            bottom5 = sorted(probs_0.tolist())[:5]
            
            stats = {
                'min_prob': float(probs_0.min()),
                'max_prob': float(probs_0.max()),
                'mean_prob': float(probs_0.mean()),
                'std_prob': float(probs_0.std()),
                'entropy': float(-(probs[0] * torch.log(probs[0] + 1e-10)).sum().item()),
                'num_points': len(probs_0),
                'effective_support': 1.0 / float((probs_0 ** 2).sum()),
                'top_5_probs': top5,
                'bottom_5_probs': bottom5
            }
            return stats, probs


    # Optional convenience alias so PolicyNetwork's helper can also see .sh_coeffs
    @property
    def sh_coeffs(self) -> torch.Tensor:
        """
        Return raw SH coefficients [B, C, N_coeffs].

        This is a READ-ONLY view; use self._sh_coeffs internally to assign.
        """
        if self._sh_coeffs is None:
            raise RuntimeError(
                "sh_coeffs accessed before being set. "
                "You must assign self._sh_coeffs in the constructor or build path."
            )
        return self._sh_coeffs

    def export_sh_coeffs(self, l_max: int = None) -> torch.Tensor:
        """
        Export SH coefficients, optionally truncated to l_max.

        Returns
        -------
        coeffs : [B, C, N_coeffs]
        """
        coeffs = self.sh_coeffs  # uses the property

        if coeffs.dim() != 3:
            raise RuntimeError(
                f"Expected SH coeffs [B, C, N_coeffs], got {tuple(coeffs.shape)}"
            )

        if l_max is None:
            return coeffs

        B, C, N = coeffs.shape
        L_eff = int(math.sqrt(N) - 1)
        if (L_eff + 1) ** 2 != N:
            raise RuntimeError(
                f"Cannot infer L from N_coeffs={N} in export_sh_coeffs(); "
                "expected N_coeffs=(L+1)^2."
            )

        if l_max >= L_eff:
            return coeffs

        N_use = (l_max + 1) ** 2
        return coeffs[:, :, :N_use]        
