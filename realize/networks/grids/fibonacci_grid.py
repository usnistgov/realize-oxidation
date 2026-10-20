"""Fibonacci spiral grid on the unit sphere."""
import torch
import math

def generate_fibonacci_grid(n: int = 1024) -> torch.Tensor:
    """Uniformly distributed points on the unit sphere via Fibonacci spiral; returns [n, 3]."""
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    phi = torch.acos(1 - 2 * indices / n)  # Polar angle [0, pi]
    theta = torch.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden angle)
    theta = theta % (2 * torch.pi)  # Keep angles bounded

    # Convert to Cartesian.
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=1)
