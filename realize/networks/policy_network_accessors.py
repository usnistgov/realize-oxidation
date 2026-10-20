"""Accessor methods for PolicyNetwork."""
import math
import torch
from typing import Dict, Any, Tuple

from realize.networks.distributions.policy_utils.policy_helpers import validate_temperature


class PolicyAccessorsMixin:
    """Getters/setters/diagnostic accessors for PolicyNetwork.

    Create no nn.Parameters and run outside construction, so they do not affect the
    forward fingerprint (44 tensors / 75978 scalars). Submodules and config resolve
    through the MRO back to PolicyNetwork.
    """
    def get_last_outputs(self) -> Dict[str, Any]:
        return self._last_outputs.copy()

    def set_temperature(self, temperature: float):
        temperature = validate_temperature(temperature)
        if hasattr(self, 'sh_head'):
            self.sh_head.set_temperature(temperature)
        if hasattr(self, 'magnitude_predictor') and hasattr(self.magnitude_predictor, 'set_temperature'):
            self.magnitude_predictor.set_temperature(temperature)

    def get_current_temperature(self) -> float:
        if hasattr(self.sh_head, 'current_temperature'):
            return float(self.sh_head.current_temperature)
        return 1.0

    def check_channel_health(self, episode: int):
        pass

    def get_lmax(self):
        return self.model_config.lmax if hasattr(self, 'model_config') else self.config.lmax

    def get_sampling_diagnostics(self) -> Dict[str, Any]:
        return {
            'channel_weights': self._last_outputs.get('channel_weights'),
            'direction_entropy': self._last_outputs.get('direction_entropy'),
        }

    def set_capture_distributions(self, capture: bool):
        self._capture_distributions = capture

    @torch.no_grad()
    def get_direction_distribution_sh_coeffs(
        self,
        state_sym: torch.Tensor,
        o2_embedding_asymmetric: torch.Tensor,
        o2_positions: torch.Tensor,
        l_max: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            dist, channel_weights, sh_coeffs, _ = self.direction_manager.build_distribution(
                state_sym,
                episode=0,
                for_sampling=False,
                state_uid=None,
                o2_positions=o2_positions,
            )
            
            if hasattr(self.sh_head, '_last_channel_rotations'):
                rotations = self.sh_head._last_channel_rotations
                sh_coeffs_world = self.sh_head.rotate_distribution_to_world(sh_coeffs, rotations)
            else:
                sh_coeffs_world = sh_coeffs

            if sh_coeffs_world.dim() == 2:
                sh_coeffs_world = sh_coeffs_world.unsqueeze(0)

            if l_max is not None:
                B, C, N = sh_coeffs_world.shape
                L_eff = int(math.isqrt(N) - 1)
                if l_max < L_eff:
                    N_use = (l_max + 1) ** 2
                    sh_coeffs_world = sh_coeffs_world[:, :, :N_use]

        return sh_coeffs_world, channel_weights
