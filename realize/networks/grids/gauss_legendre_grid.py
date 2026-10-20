"""Gauss-Legendre quadrature grid on S^2."""
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
import math

def generate_gauss_legendre_grid(n_beta: int, n_alpha: int, device: torch.device):
    """Generate a Gauss-Legendre quadrature grid over S^2.

    Args:
      - n_beta: Number of polar angle points (> 0)
      - n_alpha: Number of azimuthal angle points (> 0)
      - device: torch.device

    Returns:
      points:  [n_beta*n_alpha, 3] unit vectors on S^2
      weights: [n_beta*n_alpha] quadrature weights summing to 4pi

    Raises:
      TypeError: If device is not a torch.device
      ValueError: If n_beta or n_alpha are invalid
      RuntimeError: If generated grid fails validation
    """
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be a torch.device, got {type(device)}")

    if n_beta is None or n_alpha is None:
        raise ValueError("n_beta and n_alpha must be provided explicitly (no None allowed)")

    if not isinstance(n_beta, int) or not isinstance(n_alpha, int):
        raise TypeError(f"n_beta and n_alpha must be integers, got {type(n_beta)}, {type(n_alpha)}")

    if n_beta <= 0 or n_alpha <= 0:
        raise ValueError(
            f"n_beta and n_alpha must be > 0, got n_beta={n_beta}, n_alpha={n_alpha}"
        )

    # GL nodes/weights; polar angle beta = arccos(x).
    x, w_beta = leggauss(n_beta)  # numpy float64 arrays
    beta = torch.tensor(np.arccos(x), device=device, dtype=torch.float32)
    alpha = torch.linspace(0.0, 2.0 * math.pi, n_alpha + 1, device=device, dtype=torch.float32)[:-1]

    beta_grid, alpha_grid = torch.meshgrid(beta, alpha, indexing='ij')
    sin_beta = torch.sin(beta_grid)
    cos_beta = torch.cos(beta_grid)
    cos_alpha = torch.cos(alpha_grid)
    sin_alpha = torch.sin(alpha_grid)

    # Cartesian points on S^2 (flattened).
    points = torch.stack([
        (sin_beta * cos_alpha).reshape(-1),
        (sin_beta * sin_alpha).reshape(-1),
        cos_beta.reshape(-1),
    ], dim=-1)

    r = points.norm(dim=-1)
    if not torch.allclose(r, torch.ones_like(r), atol=1e-6):
        max_error = (r - 1.0).abs().max().item()
        raise RuntimeError(
            f"Generated GL points are not unit-norm. Max error: {max_error:.3e}. "
            f"This indicates a bug in grid construction."
        )

    # Quadrature weights: w_beta (x) uniform(alpha).
    w_beta_t = torch.tensor(w_beta, device=device, dtype=torch.float32)
    weights = (w_beta_t.unsqueeze(1).expand(n_beta, n_alpha) * (2.0 * math.pi / n_alpha)).reshape(-1)

    invalid = (weights <= 0) | (~torch.isfinite(weights))
    if invalid.any():
        num_invalid = invalid.sum().item()
        invalid_indices = invalid.nonzero(as_tuple=False).squeeze(-1).tolist()[:10]
        invalid_values = weights[invalid][:10].tolist()
        raise RuntimeError(
            f"Generated {num_invalid} invalid GL weights (non-positive or non-finite). "
            f"Indices: {invalid_indices}, values: {invalid_values}. "
            f"This indicates incorrect quadrature construction."
        )

    # Weights sum to 4pi (sphere surface area).
    total = weights.sum().item()
    expected = 4.0 * math.pi
    relative_error = abs(total - expected) / expected
    if relative_error > 1e-6:
        raise RuntimeError(
            f"GL weights do not sum to 4pi. Got {total:.8f}, expected {expected:.8f}. "
            f"Relative error: {relative_error:.3e} > 1e-6. "
            f"This indicates incorrect quadrature normalization."
        )

    return points, weights
