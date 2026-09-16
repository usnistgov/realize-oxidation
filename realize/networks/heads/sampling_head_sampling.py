"""SamplingHead sampling / diagnostics mixin.

None of these methods run during SamplingHead.__init__ and none creates an
nn.Parameter / submodule / buffer, so the parameter set and registration order
(the 44-tensor / 75978-scalar forward fingerprint) are untouched. The mixin
defines no __init__/__setattr__, so nn.Module.__init__ and its Parameter/submodule
interception still apply; all self.* calls resolve through the MRO.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from e3nn.o3 import Irreps, Linear, spherical_harmonics
from realize.networks.irrep_sequential import IrrepSequential, IrrepActivation
from realize.networks.distributions.spherical_distribution import MultiChannelSphericalDistribution
import logging
_LOG = logging.getLogger(__name__)


class SamplingHeadSamplingMixin:
    def _build_mixture_with_tempering(self, channel_log_mass_normalized, channel_weights, tau=1.0):
        """
        Build mixture with optional per-channel tempering to preserve fairness.
        
        Args:
            channel_log_mass_normalized: [batch, channels, P] pre-normalized masses
            channel_weights: [batch, channels] gate weights
            tau: Exploration temperature (tau=1 preserves fairness)
            
        Returns:
            log_mix_mass_norm: [batch, P] normalized mixture log-mass
            log_density_mix: [batch, P] mixture log-density
        """
        if tau == 1.0:
            # No tempering - use standard mixing
            log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
            log_mix_mass = torch.logsumexp(log_w_c + channel_log_mass_normalized, dim=1)
            log_mix_mass_norm = log_mix_mass - torch.logsumexp(log_mix_mass, dim=-1, keepdim=True)
        else:
            # Per-channel tempering then mix (preserves fairness)
            # Temper each channel's mass separately
            log_mass_tempered_c = channel_log_mass_normalized / max(tau, 1e-8)
            
            # Re-normalize each channel (stays integrated to 1)
            log_mass_tempered_norm_c = log_mass_tempered_c - torch.logsumexp(log_mass_tempered_c, dim=-1, keepdim=True)
            
            # Now mix with same gate weights
            log_w_c = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
            log_mix_mass = torch.logsumexp(log_w_c + log_mass_tempered_norm_c, dim=1)
            log_mix_mass_norm = log_mix_mass - torch.logsumexp(log_mix_mass, dim=-1, keepdim=True)
        
        return log_mix_mass_norm, None

    def sample_direction(self, state_features, episode=0, use_langevin=None):
        """Sample direction from the learned distribution."""
        
        # Grid validation
        if hasattr(self, '_channel_log_mass_normalized'):
            expected_points = self._channel_log_mass_normalized.shape[-1]
            actual_points = self.grid_config.points.shape[0]
            
            if expected_points != actual_points:
                raise RuntimeError(
                    f"Grid size mismatch: normalized masses computed on {expected_points} points, "
                    f"but current grid has {actual_points} points."
                )
            
            # Check grid values (not just count)
            if hasattr(self, '_grid_fingerprint'):
                current_fingerprint = self._compute_grid_fingerprint()
                
                if not self._grids_match(self._grid_fingerprint, current_fingerprint):
                    raise RuntimeError(
                        f"Grid identity violated: normalized masses computed on different grid. "
                        f"Grid fingerprint mismatch. Recompute with forward()."
                    )
    
        # Forward pass (always recompute for fresh normalized masses)
        sh_coeffs, channel_weights = self.forward(state_features, episode)
    
        # Store channel weights for evaluation reuse
        if channel_weights is not None:
            self.last_channel_weights = channel_weights.detach()
        else:
            self.last_channel_weights = None
    
        # Determine sampling method (NO AUTO-DECIDE - must be explicit)
        if use_langevin is None:
            raise ValueError(
                "use_langevin must be explicitly True or False; no automatic decision allowed. "
                "Caller must decide the sampling method."
            )
        
        # Validate it's a bool
        if not isinstance(use_langevin, bool):
            raise TypeError(f"use_langevin must be bool, got {type(use_langevin)}")

        # Create distribution
        # Ensure channel dimension for shape inference only
        if sh_coeffs.dim() == 2:
            sh_coeffs = sh_coeffs.unsqueeze(1)  # Add channel dimension
        
        # Get per-channel temperatures if multi-channel
        channel_temperatures = None
        if self.num_channels > 1 and hasattr(self, 'channel_log_temps') and self.channel_log_temps is not None:
            channel_temperatures = torch.exp(self.channel_log_temps).clamp(
                min=getattr(self.model_config, 'channel_temp_min', 0.01),
                max=getattr(self.model_config, 'channel_temp_max', 5.0)
            )
        
        # Create distribution with PRE-NORMALIZED masses (probability-space path)
        # Pass None for sh_coeffs since we're using normalized masses directly
        dist = MultiChannelSphericalDistribution(
            sh_coeffs=None,  # Not needed - using probability-space path
            lmax=self.lmax,
            num_channels=sh_coeffs.shape[1],
            temperature=self.current_temperature,
            grid_config=self.grid_config,
            channel_temperatures=channel_temperatures,
            channel_log_mass_normalized=self._channel_log_mass_normalized
        )

        # Sample to get directions, log_probs, and indices
        use_gumbel = self.training and getattr(self.sampling_config, 'use_gumbel_softmax', False)
    
        if use_gumbel and not use_langevin:
            # Gumbel-Softmax sampling
            batch_size = sh_coeffs.shape[0]
            tau = getattr(self.sampling_config, 'gumbel_temperature', 1.0)
            
            # Build tempered mixture
            log_mix_mass_norm, _ = self._build_mixture_with_tempering(
                self._channel_log_mass_normalized,
                channel_weights,
                tau=tau
            )
            
            if self.training:
                # Add Gumbel noise for exploration over the discrete grid
                u = torch.rand_like(log_mix_mass_norm)
                g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
                noisy_logits = log_mix_mass_norm + g
                hard_indices = noisy_logits.argmax(dim=-1)  # [batch]
            else:
                # Pure categorical sampling over discrete probabilities
                probs = torch.exp(log_mix_mass_norm)
                hard_indices = torch.multinomial(probs, 1).squeeze(-1)
            
            # Get sampled directions
            grid_points = self.grid_config.points
            directions = grid_points[hard_indices]  # [batch, 3]
            
            # Discrete log-prob at chosen grid points (NO density conversion here)
            b_idx = torch.arange(batch_size, device=self.device)
            log_probs = log_mix_mass_norm[b_idx, hard_indices]  # [batch]
            sampled_indices = hard_indices
            
            # mark for debug tolerance downstream
            self._last_gumbel_log_probs = True

        elif use_langevin:
            # Langevin sampling (not recommended for PPO)
            _LOG.warning("Langevin sampling is not PPO-safe (uses non-differentiable nearest-neighbor lookup)")
            
            directions = self._langevin_sample(dist, sh_coeffs, channel_weights)
            
            # For off-grid Langevin samples, cannot compute exact log-prob
            raise NotImplementedError(
                "Langevin sampling requires off-grid SH evaluation with beta calibration. "
                "Use grid-based sampling instead."
            )
            
        else:
            # Standard grid-based sampling
            directions, log_probs, sampled_indices = dist.sample(channel_weights)
    
        # Store exact densities for visualization
        # (AFTER sampling so sampled_indices is defined)
        with torch.no_grad():
            # Convert normalized mass to density (mass / quadrature_weight)
            w = self.grid_config.weights  # [P]
            
            # Per-channel densities
            per_channel_density = torch.exp(self._channel_log_mass_normalized) / w.unsqueeze(0).unsqueeze(0)  # [B, C, P]
            
            # Mixture density (exact same as used for sampling)
            if channel_weights is not None:
                log_w_gate = torch.log(channel_weights.clamp(min=1e-12)).unsqueeze(-1)
                log_mix_mass = torch.logsumexp(
                    log_w_gate + self._channel_log_mass_normalized, dim=1
                )  # [B, P]
                mix_density = torch.exp(log_mix_mass) / w.unsqueeze(0)  # [B, P]
            else:
                # Single channel
                # Use the channel's normalized log-mass directly as the mixture (discrete)
                log_mix_mass = self._channel_log_mass_normalized[:, 0, :]  # [B, P]
                mix_density = torch.exp(log_mix_mass) / w.unsqueeze(0)      # [B, P]

            # Store for plotting (use _plotting_* names to match get_plotting_params)
            self._plotting_per_channel_density = per_channel_density[0].cpu().numpy()  # [C, P]
            self._plotting_mixture_density = mix_density[0].cpu().numpy()  # [P]
            self._plotting_grid_points = self.grid_config.points.cpu().numpy()  # [P, 3]
            
            # Store sampled index and diagnostic info
            if sampled_indices is not None:
                self._plotting_sampled_index = int(sampled_indices[0].item())

                # Diagnostic: Check if sample landed in high-probability region
                mass = torch.exp(log_mix_mass)
                b_idx = torch.arange(mass.shape[0], device=mass.device)
                sel_mass = mass[b_idx, sampled_indices]  # [B]
                
                # Percentile: what fraction of grid has lower mass?
                percentile = (mass <= sel_mass.unsqueeze(-1)).float().mean(dim=-1)  # [B]
                
                if episode % 10 == 0:
                    _LOG.debug(f"[Sample Diagnostic] Episode {episode}: "
                          f"picked-mass percentile = {percentile.mean().item():.3f} "
                          f"(should be >0.7 for good sampling)")
                
                # Store percentile for plotting
                self._plotting_sample_percentile = percentile[0].item()

        # Debug validation (run occasionally)
        if episode % 10 == 0 and self.training:
            is_consistent = self.validate_log_prob_consistency(dist, directions, channel_weights, log_probs)
            if not is_consistent:
                _LOG.debug(f"[Episode {episode}] Log-prob validation failed!")
    
        # Ensure log_probs has correct shape [batch] (NO RESCUE)
        if log_probs.dim() > 1:
            # We have per-channel log probs that need mixture reduction
            if log_probs.shape[-1] > 1 and channel_weights is not None:
                # Mixture reduction: log p(x) = log Sigma_k w_k p_k(x)
                log_probs = torch.logsumexp(
                    torch.log(channel_weights.clamp(min=1e-10)) + log_probs, 
                    dim=-1
                )
            elif log_probs.shape[-1] == 1:
                # Single channel, just squeeze
                log_probs = log_probs.squeeze(-1)
            else:
                # NO RESCUE - fail immediately
                raise RuntimeError(
                    f"Unexpected log_probs shape {log_probs.shape}. "
                    f"Expected [batch] or [batch, 1] or [batch, {self.num_channels}]. "
                    "This indicates a sampling bug."
                )

        # Final shape validation
        if log_probs.dim() != 1 or log_probs.shape[0] != directions.shape[0]:
            raise RuntimeError(
                f"log_probs must be [batch] scalars, got shape {log_probs.shape}. "
                f"Expected [{directions.shape[0]}]. This indicates a sampling bug."
            )

        # Compute entropy from SAME distribution used for sampling
        if use_gumbel and self.training and not use_langevin:
            # Gumbel uses tempered distribution - entropy must match
            tau = getattr(self.sampling_config, 'gumbel_temperature', 1.0)
            
            # Get untempered mixture mass
            log_mix_mass_untempered = dist.get_mixture_log_mass(channel_weights)
            
            # Temper it
            log_mass_tempered = log_mix_mass_untempered / max(tau, 1e-8)
            log_mass_tempered_norm = log_mass_tempered - torch.logsumexp(log_mass_tempered, dim=-1, keepdim=True)
            
            # Convert to density
            log_w = torch.log(self.grid_config.weights.clamp(min=1e-10))
            log_density_tempered = log_mass_tempered_norm - log_w.unsqueeze(0)
            
            # Entropy: -Sum mass * log_density
            mass_tempered = torch.exp(log_mass_tempered_norm)
            entropy = -(mass_tempered * log_density_tempered).sum(dim=-1)
        else:
            # Standard path: use distribution's entropy (untempered)
            entropy = dist.entropy(channel_weights)
    
        return directions, log_probs, entropy, sampled_indices

    def validate_log_prob_consistency(self, dist, directions, channel_weights, log_probs_returned):
        """
        Validate that returned log-probs match exact computation.
        This is a debug check to ensure all sampling methods return consistent density.
        
        This method calls dist.log_prob() which uses nearest-neighbor approximation.
        It's ONLY enabled when DEBUG_ALLOW_APPROX_LOGPROB=1 to prevent accidental use during PPO.
        """
        # Skip validation unless explicitly enabled (prevents approximate path during training)
        if os.environ.get('DEBUG_ALLOW_APPROX_LOGPROB') != '1':
            return True  # Assume valid, skip check
        
        with torch.no_grad():
            # Compute approximate log-prob using nearest-neighbor
            # (This is why we guard this method - it's not PPO-safe)
            log_probs_exact = dist.log_prob(directions, channel_weights)
            
            # Both should be densities now
            max_diff = (log_probs_returned - log_probs_exact).abs().max().item()
            
            # For Gumbel-Softmax, allow larger tolerance due to grid discretization
            tolerance = 0.1 if hasattr(self, '_last_gumbel_log_probs') else 1e-4
            
            if max_diff > tolerance:
                _LOG.debug(f"Log-prob inconsistency detected: max diff = {max_diff:.6f}")
                _LOG.debug(f"  Returned: {log_probs_returned[:3]}")
                _LOG.debug(f"  Exact: {log_probs_exact[:3]}")
                _LOG.debug(f"  Tolerance: {tolerance}")
            
            return max_diff < tolerance

    def _langevin_sample(self, dist, sh_coeffs, channel_weights=None, num_steps=None, step_size=None):
        """
        Langevin sampling on sphere with proper tangent space operations.
        """
        if num_steps is None:
            num_steps = getattr(self.sampling_config, 'langevin_num_steps', 50)
        if step_size is None:
            step_size = getattr(self.sampling_config, 'langevin_init_step_size', 0.01)
        
        batch_size = sh_coeffs.shape[0]
        device = sh_coeffs.device
        
        # Initialize uniformly on sphere
        samples = F.normalize(torch.randn(batch_size, 3, device=device), dim=-1)
        
        # NO GUARD - temperature should already be validated
        if self.current_temperature <= 0:
            raise RuntimeError(
                f"Current temperature is non-positive: {self.current_temperature} for Langevin sampling"
            )
        beta = 1.0 / self.current_temperature

        for step in range(num_steps):
            # Compute score (gradient of log probability)
            samples.requires_grad_(True)
            log_probs = dist.log_prob(samples, channel_weights)  # Pass channel_weights
            score = torch.autograd.grad(log_probs.sum(), samples)[0]
            samples = samples.detach()

            # Project to tangent space at current point on sphere
            score_tangent = score - torch.sum(score * samples, dim=-1, keepdim=True) * samples
            
            # Langevin update in tangent space
            noise = torch.randn_like(samples) * math.sqrt(2 * step_size / beta)
            noise_tangent = noise - torch.sum(noise * samples, dim=-1, keepdim=True) * samples
            
            update = step_size * score_tangent + noise_tangent
            
            # Apply exponential map to stay on sphere
            update_norm = torch.norm(update, dim=-1, keepdim=True).clamp(min=1e-8)
            samples = torch.cos(update_norm) * samples + torch.sin(update_norm) * update / update_norm
            samples = F.normalize(samples, dim=-1)
            
            # Decay step size
            step_size *= (1 - 0.5/num_steps)
        
        return samples
    
    def update_temperature(self, episode):
        """Update sampling temperature with cosine annealing (precomputed schedule REQUIRED)."""
        # NO FALLBACK - schedule must exist
        if not hasattr(self, 'temperature_schedule'):
            raise RuntimeError(
                "temperature_schedule not found; _precompute_temperature_schedule() was not called. "
                "This indicates __init__ failed to setup properly."
            )
        
        if episode < 0:
            raise ValueError(f"episode must be non-negative, got {episode}")
        
        # Use precomputed schedule
        if episode < len(self.temperature_schedule):
            self.current_temperature = self.temperature_schedule[episode].item()
        else:
            # Beyond schedule: use minimum (this is correct behavior, not a fallback)
            self.current_temperature = self.temperature_min
        
        # Validate result (NO SILENT FLOOR - fail if something is wrong)
        if self.current_temperature <= 0:
            raise RuntimeError(
                f"Temperature became non-positive: {self.current_temperature} at episode {episode}. "
                "Check temperature_init and temperature_min configuration."
            )
        
        return self.current_temperature

    def get_distribution_entropy(self, sh_coeffs, channel_weights=None):
        """
        Compute entropy using normalized probability-space mixture.
        """
        # Ensure channel dimension
        if sh_coeffs.dim() == 2:
            sh_coeffs = sh_coeffs.unsqueeze(1)
        
        # Normalized masses MUST exist (NO SILENT RECOMPUTE)
        if not hasattr(self, '_channel_log_mass_normalized'):
            raise RuntimeError(
                "Cannot compute entropy: normalized masses not available. "
                "Call forward() before get_distribution_entropy()."
            )
        
        # Validate shape matches
        if self._channel_log_mass_normalized.shape[0] != sh_coeffs.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch: normalized masses have batch size "
                f"{self._channel_log_mass_normalized.shape[0]}, but sh_coeffs has "
                f"{sh_coeffs.shape[0]}. Call forward() with current batch."
            )

        # Get per-channel temperatures (kept for compatibility)
        channel_temperatures = None
        if self.num_channels > 1 and self.channel_log_temps is not None:
            channel_temperatures = torch.exp(self.channel_log_temps).clamp(
                min=getattr(self.model_config, 'channel_temp_min', 0.01),
                max=getattr(self.model_config, 'channel_temp_max', 5.0)
            )
        
        # Create distribution with normalized masses (probability-space path)
        # Pass None for sh_coeffs since we're using normalized masses directly
        dist = MultiChannelSphericalDistribution(
            sh_coeffs=None,  # Not needed - using probability-space path
            lmax=self.lmax,
            num_channels=sh_coeffs.shape[1],
            temperature=self.current_temperature,
            grid_config=self.grid_config,
            channel_temperatures=channel_temperatures,
            channel_log_mass_normalized=self._channel_log_mass_normalized
        )

        # Compute entropy from normalized mixture
        entropy = dist.entropy(channel_weights)

        return entropy        

    def get_plotting_params(self, R_feat_to_world=None):
        """
        Get exact parameters for plotting the distribution used for sampling.

        Args:
            R_feat_to_world: [3, 3] rotation matrix (optional)

        Returns:
            dict with keys for visualization
        """
        if not hasattr(self, '_plotting_per_channel_density'):
            raise RuntimeError("Must call sample_direction() before getting plotting params")

        params = {
            'lmax': self.lmax,
            'per_channel_probs': self._plotting_per_channel_density,  # [C, P] - densities
            'mixture_probs': self._plotting_mixture_density,  # [P] - density
            'grid_points': self._plotting_grid_points,  # [P, 3]
            'grid_type': self.grid_type,
            'num_channels': self.num_channels,
        }

        # Add sampled point info if available
        if hasattr(self, '_plotting_sampled_index'):
            params['sampled_index'] = self._plotting_sampled_index
            params['sample_percentile'] = self._plotting_sample_percentile

            # Get exact grid point (in feature frame)
            sampled_point_feat = self._plotting_grid_points[self._plotting_sampled_index]  # [3]

            # Rotate to world if transform provided
            if R_feat_to_world is not None:
                sampled_point_world = R_feat_to_world @ sampled_point_feat
                params['sampled_point_world'] = sampled_point_world
            else:
                params['sampled_point_world'] = sampled_point_feat

        # Add channel weights if available
        if hasattr(self, 'last_channel_weights') and self.last_channel_weights is not None:
            params['channel_weights'] = self.last_channel_weights.cpu().numpy()

        return params
