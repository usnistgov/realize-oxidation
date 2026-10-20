"""Builds the translation-direction distribution."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from realize.networks.grids.grid_factory import GridConfig
from realize.networks.distributions.spherical_distribution import MultiChannelSphericalDistribution
from realize.networks.distributions.policy_utils.policy_helpers import extract_o2_positions

class DirectionDistributionManager:
    """Manager for translation direction distribution."""
    
    def __init__(self, policy_network, device):
        self.policy = policy_network
        self.device = device
        
    def get_channel_temperatures(self) -> torch.Tensor:
        if hasattr(self.policy.sh_head, 'current_temperature'):
            temp = self.policy.sh_head.current_temperature
            return torch.full((self.policy.num_channels,), float(temp), device=self.policy.device, dtype=torch.float32)
        else:
            return torch.ones(self.policy.num_channels, device=self.policy.device)
    
    def build_distribution(
        self,
        states: torch.Tensor,
        episode: int,
        for_sampling: bool = True,
        state_uid: Optional[str] = None,
        o2_positions: Optional[torch.Tensor] = None,
        override_temperature: Optional[float] = None
    ) -> Tuple[MultiChannelSphericalDistribution, torch.Tensor, torch.Tensor, torch.Tensor]:

        if o2_positions is None:
            o2_positions = extract_o2_positions(states)
        
        unaligned_features = self.policy.frame_encoder.extract_unaligned_features(states)

        # The multipole's directional signal enters the SH head exclusively
        # through the actor state (multipole_injector), not as a post-hoc logit
        # bias. This keeps the energy field u(x) consistent between grid logits,
        # Langevin score, and log-prob.
        head_outputs = self.policy.sh_head(unaligned_features, episode=episode, state_uid=state_uid)

        mixture_log_mass_normalized = head_outputs['mixture_log_mass_normalized']
        channel_log_mass_normalized = head_outputs['channel_log_mass_normalized']
        channel_weights = head_outputs['channel_weights']
        sh_coeffs = head_outputs['sh_coeffs']
        grid_points = head_outputs['grid_points']        
        grid_weights = head_outputs['grid_weights']
        grid_log_weights = head_outputs['grid_log_weights']

        grid_config = GridConfig(
            grid_type='gauss_legendre',
            num_points=grid_points.shape[0],
            points=grid_points,
            weights=grid_weights,
            log_weights=grid_log_weights
        )

        # Use override_temperature if provided
        # This fixes schedule mismatch where evaluate() uses different temp than rollout
        if override_temperature is not None:
            effective_temperature = override_temperature
        else:
            effective_temperature = self.policy.sh_head.current_temperature
        
        dist = MultiChannelSphericalDistribution(
            sh_coeffs,
            lmax=self.policy.model_config.lmax,
            num_channels=self.policy.num_channels,
            temperature=effective_temperature,
            grid_config=grid_config,
            channel_temperatures=self.get_channel_temperatures(),
            channel_log_mass_normalized=channel_log_mass_normalized
        )

        dist._o2_positions = o2_positions
        dist.validate_grid_alignment(grid_points, grid_weights)

        return dist, channel_weights, sh_coeffs, head_outputs    

    def sample_direction(self, dist, channel_weights, deterministic=False):
        if deterministic:
            # Deterministic ArgMax Sampling
            log_mix_mass = self.policy.sh_head.compute_mixture_log_mass(dist.channel_log_mass_normalized, channel_weights)
            idx = torch.argmax(log_mix_mass, dim=-1)
            sampled_dirs = self.policy.sh_head.s2_block.grid_xyz[idx]
            
            batch_size = log_mix_mass.shape[0]
            b_idx = torch.arange(batch_size, device=self.device)
            log_prob = log_mix_mass[b_idx, idx]
            return sampled_dirs, log_prob, idx
        else:
            # Stochastic Sampling
            sampled_dirs, log_prob, idx = self.policy.sh_head.sample_direction(
                dist.channel_log_mass_normalized, channel_weights, return_log_prob=True
            )

            return sampled_dirs, log_prob, idx

    def compute_entropy(self, dist, channel_weights):
        return self.policy.sh_head.compute_entropy(dist.channel_log_mass_normalized, channel_weights)

    def sample_direction_deterministic(
        self, states, direction_indices, episode, o2_positions
    ):
        # Re-build distribution to get exact probability state
        dist, channel_weights, _, _ = self.build_distribution(
            states, episode=episode, for_sampling=False, state_uid=None, o2_positions=o2_positions
        )

        device = self.device
        if not isinstance(direction_indices, torch.Tensor):
            direction_indices = torch.as_tensor(direction_indices, device=device)
        direction_indices = direction_indices.to(device=device, dtype=torch.long)
        
        # Allow [B, 1] -> [B]
        if direction_indices.dim() == 2 and direction_indices.shape[1] == 1:
            direction_indices = direction_indices.squeeze(-1)

        # Retrieve grid points directly from head
        grid_points = self.policy.sh_head.s2_block.grid_xyz
        sampled_direction_feat = grid_points[direction_indices]

        # Compute exact log prob for these indices
        log_mix_mass = self.policy.sh_head.compute_mixture_log_mass(
            dist.channel_log_mass_normalized, channel_weights
        )
        b_idx = torch.arange(states.shape[0], device=device)
        log_prob = log_mix_mass[b_idx, direction_indices]

        return sampled_direction_feat, log_prob, direction_indices, direction_indices

