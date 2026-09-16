import torch
import math
import numpy as np
from typing import Dict, Tuple


def von_mises_log_prob(theta: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Stable log p(theta | mu, kappa) for circular von Mises.
    Uses log I0 via torch.special.i0e: log I0(kappa) = log(i0e(kappa)) + |kappa|
    log p = kappa * cos(theta - mu) - log(2pi) - log I0(kappa)
    
    Args:
        theta: Sampled angles [batch]
        mu: Mean angle [batch]
        kappa: Concentration parameter [batch]
    
    Returns:
        log_prob: [batch] log probability values
    """
    # clamp kappa for stability
    kappa = kappa.clamp(min=1e-6)
    # i0e is exponentially scaled I0 so log I0 = log(i0e) + |kappa|
    log_i0 = torch.log(torch.special.i0e(kappa)) + torch.abs(kappa)
    return kappa * torch.cos(theta - mu) - math.log(2 * math.pi) - log_i0


def extract_o2_positions(state: torch.Tensor) -> torch.Tensor:
    """
    Extract O2 atom positions from state for world frame computation.
    
    Assumes state structure: [O1_x, O1_y, O1_z, O2_x, O2_y, O2_z, ...]
    
    Args:
        state: State tensor [batch, state_dim]
    
    Returns:
        o2_positions: [batch, 2, 3] positions of two O atoms in world frame
        
    Raises:
        RuntimeError: If state dimension is invalid
    """
    batch_size = state.shape[0]
    
    # State must contain O2 positions and be divisible by 3
    if state.shape[1] < 6:
        raise RuntimeError(
            f"State dimension {state.shape[1]} < 6.\n"
            f"Cannot extract O2 positions - state must contain at least 6 values."
        )
    
    if state.shape[1] % 3 != 0:
        raise RuntimeError(
            f"State dimension {state.shape[1]} not divisible by 3.\n"
            f"State must contain packed (x,y,z) positions only.\n"
            f"Got {state.shape[1]} values, expected multiple of 3."
        )
    
    # Extract first 6 values as O2 positions
    o1_pos = state[:, 0:3]  # [batch, 3]
    o2_pos = state[:, 3:6]  # [batch, 3]
    
    # Stack to [batch, 2, 3]
    o2_positions = torch.stack([o1_pos, o2_pos], dim=1)
    
    return o2_positions


def rotate_state(state: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate positions in the state by rotation matrix R.
    
    Args:
        state: State tensor [batch, state_dim]
        R: Rotation matrices [batch, 3, 3]
        
    Returns:
        Rotated state [batch, state_dim]
        
    Raises:
        RuntimeError: If state dimension is invalid
    """
    B, D = state.shape
    
    # Assuming state is packed as [O1_pos, O2_pos, neighbor_positions...]
    # Each position is 3 values (x, y, z)
    if D < 6:
        raise RuntimeError(
            f"Cannot rotate state with dimension {D} < 6.\n"
            "State must contain at least O2 positions (6 values) for rotation testing."
        )
    
    # Reshape to [batch, num_atoms, 3]
    assert D % 3 == 0, f"State dimension {D} must be divisible by 3 for position rotation"
    num_atoms = D // 3
    positions = state.view(B, num_atoms, 3)
    
    # Rotate all positions
    positions_rotated = torch.bmm(positions, R.transpose(1, 2))
    
    return positions_rotated.view(B, D)


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature value is finite and positive.
    
    Args:
        temperature: Temperature value to validate
    
    Returns:
        Temperature as float
        
    Raises:
        ValueError: If temperature is invalid
    """
    if temperature is not None:
        t = float(temperature)
        if not math.isfinite(t) or t <= 0.0:
            raise ValueError(
                f"Invalid temperature {temperature}.\n"
                "Temperature must be finite and > 0."
            )
        return t
    return temperature

