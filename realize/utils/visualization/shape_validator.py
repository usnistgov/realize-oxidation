import numpy as np
import torch
from typing import Union, Tuple, Optional
import logging
_LOG = logging.getLogger(__name__)


class ShapeValidator:
    """Validates and cleans shapes for visualization data."""
    
    @staticmethod
    def clean_sh_coeffs(coeffs: Union[np.ndarray, torch.Tensor], 
                        expected_channels: Optional[int] = None,
                        strict: bool = False) -> np.ndarray:
        """Clean SH coeffs to (sh_dim,) single-channel or (num_channels, sh_dim).

        strict=True raises on a channel-count mismatch instead of warning.
        """
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.detach().cpu().numpy()

        coeffs = np.asarray(coeffs, dtype=np.float32)

        while coeffs.ndim > 1 and coeffs.shape[0] == 1:
            coeffs = coeffs.squeeze(0)

        if coeffs.ndim == 1:
            if expected_channels and expected_channels != 1:
                msg = f"Expected {expected_channels} channels, got single channel"
                if strict:
                    raise ValueError(msg)
                else:
                    _LOG.info(f"WARNING: {msg}")
            return coeffs
        elif coeffs.ndim == 2:
            if expected_channels and coeffs.shape[0] != expected_channels:
                msg = f"Expected {expected_channels} channels, got {coeffs.shape[0]}"
                if strict:
                    raise ValueError(msg)
                else:
                    _LOG.info(f"WARNING: {msg}")
            return coeffs
        else:
            raise ValueError(f"Invalid SH coeffs shape after cleaning: {coeffs.shape}")

    @staticmethod
    def clean_o2_positions(positions: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Clean O2 positions to (2, 3) shape."""
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()

        positions = np.asarray(positions, dtype=np.float32)

        while positions.ndim > 2 and positions.shape[0] == 1:
            positions = positions.squeeze(0)

        if positions.shape != (2, 3):
            raise ValueError(f"Invalid O2 positions shape: {positions.shape}, expected (2, 3)")
        
        return positions
    
    @staticmethod
    def clean_channel_weights(weights: Union[np.ndarray, torch.Tensor], 
                            num_channels: int) -> np.ndarray:
        """Clean channel weights to (num_channels,); None means uniform weights."""
        if weights is None:
            return np.ones(num_channels, dtype=np.float32) / num_channels

        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        weights = np.asarray(weights, dtype=np.float32)

        while weights.ndim > 1 and weights.shape[0] == 1:
            weights = weights.squeeze(0)

        weights = weights.flatten()

        if len(weights) != num_channels:
            raise ValueError(f"Invalid channel weights length: {len(weights)}, expected {num_channels}")
        
        return weights
    
    @staticmethod
    def clean_direction(direction: Union[np.ndarray, torch.Tensor, None]) -> Optional[np.ndarray]:
        """Clean direction vector to (3,); returns None if absent or malformed."""
        if direction is None:
            return None

        if isinstance(direction, torch.Tensor):
            direction = direction.detach().cpu().numpy()

        direction = np.asarray(direction, dtype=np.float32)

        direction = direction.flatten()

        if len(direction) != 3:
            _LOG.info(f"WARNING: Invalid direction shape: {direction.shape}, expected (3,)")
            return None
        
        return direction
