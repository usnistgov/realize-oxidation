import torch
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from realize.networks.grids.fibonacci_grid import generate_fibonacci_grid
from realize.networks.grids.lebedev_grid import LebedevGrid
from realize.networks.grids.gauss_legendre_grid import generate_gauss_legendre_grid

# Grid cache, keyed by a per-grid-type tuple.
_GRID_CACHE: Dict[tuple, 'GridConfig'] = {}

@dataclass
class GridConfig:
    """Immutable cached spherical-grid configuration."""
    grid_type: str
    num_points: int
    points: torch.Tensor  # [num_points, 3]
    weights: torch.Tensor  # [num_points]
    log_weights: torch.Tensor  # [num_points], precomputed

    def __post_init__(self):
        """Validate and freeze grid tensors.

        Raises:
            RuntimeError: If tensors require grad
            TypeError: If tensors are not floating point
        """
        for name, tensor in [('points', self.points),
                            ('weights', self.weights), 
                            ('log_weights', self.log_weights)]:
            if tensor.requires_grad:
                raise RuntimeError(
                    f"Grid tensor '{name}' must not require grad. "
                    f"Grid configurations are immutable."
                )
            
            if not tensor.is_floating_point():
                raise TypeError(
                    f"Grid tensor '{name}' must be floating point, "
                    f"got dtype {tensor.dtype}"
                )
        
        self.points.requires_grad_(False)
        self.weights.requires_grad_(False)
        self.log_weights.requires_grad_(False)

    def to(self, device):
        """Move grid to device, returning a new instance (cache unmodified)."""
        if self.points.device == device:
            return self

        return GridConfig(
            grid_type=self.grid_type,
            num_points=self.num_points,
            points=self.points.to(device),
            weights=self.weights.to(device),
            log_weights=self.log_weights.to(device)
        )

def _validate_grid(points: torch.Tensor, weights: torch.Tensor) -> None:
    """Validate spherical grid properties; raise ValueError on any violation.

    Args:
        points: Grid points [N, 3]
        weights: Quadrature weights [N]
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must be [N, 3], got shape {points.shape}"
        )

    if weights.ndim != 1:
        raise ValueError(
            f"weights must be 1D, got shape {weights.shape}"
        )

    if weights.shape[0] != points.shape[0]:
        raise ValueError(
            f"Shape mismatch: points has {points.shape[0]} points, "
            f"weights has {weights.shape[0]} elements"
        )

    if not torch.isfinite(points).all():
        num_invalid = (~torch.isfinite(points)).any(dim=-1).sum().item()
        raise ValueError(
            f"points contains {num_invalid} rows with NaN/Inf values"
        )

    if not torch.isfinite(weights).all():
        num_invalid = (~torch.isfinite(weights)).sum().item()
        raise ValueError(
            f"weights contains {num_invalid} NaN/Inf values"
        )

    if (weights <= 0).any():
        num_invalid = (weights <= 0).sum().item()
        invalid_indices = (weights <= 0).nonzero(as_tuple=False).squeeze(-1).tolist()[:10]
        raise ValueError(
            f"weights contains {num_invalid} non-positive values. "
            f"First few indices: {invalid_indices}"
        )

    # Points on unit sphere.
    norms = points.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
        max_error = (norms - 1.0).abs().max().item()
        num_bad = ((norms - 1.0).abs() > 1e-6).sum().item()
        raise ValueError(
            f"{num_bad} points not on unit sphere (|r-1| > 1e-6). "
            f"Max error: {max_error:.3e}"
        )

    # Weights sum to 4pi (sphere surface area).
    total = weights.sum().item()
    expected = 4.0 * math.pi
    relative_error = abs(total - expected) / expected

    if relative_error > 1e-6:
        raise ValueError(
            f"Weights do not sum to 4pi. "
            f"Got {total:.8f}, expected {expected:.8f}. "
            f"Relative error: {relative_error:.3e} > 1e-6"
        )

def get_spherical_grid(*,
                       grid_type: str, 
                       lmax: int,
                       device: torch.device,
                       num_points: Optional[int] = None,
                       gl_n_beta: Optional[int] = None,
                       gl_n_alpha: Optional[int] = None,
                       lebedev_degree: Optional[int] = None) -> GridConfig:
    """Get a cached spherical grid; explicit config required (no 'auto' selection).

    Args:
        grid_type: 'lebedev', 'fibonacci', or 'gauss_legendre'
        lmax: Maximum spherical harmonic degree
        device: Target device (required)
        num_points: Number of points (required for 'fibonacci')
        gl_n_beta: Polar resolution (required for 'gauss_legendre')
        gl_n_alpha: Azimuthal resolution (required for 'gauss_legendre')
        lebedev_degree: Lebedev quadrature degree (required for 'lebedev')

    Returns:
        GridConfig with points, weights, and cached metadata

    Raises:
        TypeError: If device is not torch.device
        ValueError: If grid_type is 'auto' or required parameters missing
        RuntimeError: If grid generation fails validation
    """
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be a torch.device, got {type(device)}")

    # 'auto' is rejected: no silent grid selection.
    if grid_type == 'auto':
        raise ValueError(
            "grid_type='auto' is FORBIDDEN. "
            "Explicitly specify 'lebedev', 'gauss_legendre', or 'fibonacci'."
        )
    if grid_type == 'lebedev':
        if lebedev_degree is None:
            raise ValueError(
                "lebedev_degree must be specified explicitly for Lebedev grid. "
                "No default value allowed."
            )
        if not isinstance(lebedev_degree, int) or lebedev_degree <= 0:
            raise ValueError(f"lebedev_degree must be a positive integer, got {lebedev_degree}")
        cache_key = ('lebedev', lebedev_degree, lmax, str(device))    
    elif grid_type == 'gauss_legendre':
        if gl_n_beta is None or gl_n_alpha is None:
            raise ValueError(
                "gl_n_beta and gl_n_alpha must be specified explicitly for Gauss-Legendre grid. "
                f"Got gl_n_beta={gl_n_beta}, gl_n_alpha={gl_n_alpha}. "
                "No default values allowed."
            )
        if not isinstance(gl_n_beta, int) or not isinstance(gl_n_alpha, int):
            raise TypeError(f"gl_n_beta and gl_n_alpha must be integers")
        if gl_n_beta <= 0 or gl_n_alpha <= 0:
            raise ValueError(f"gl_n_beta and gl_n_alpha must be > 0")
        cache_key = ('gauss_legendre', gl_n_beta, gl_n_alpha, lmax, str(device))        
    elif grid_type == 'fibonacci':
        if num_points is None:
            raise ValueError(
                "num_points must be specified explicitly for Fibonacci grid. "
                "No default value allowed."
            )
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError(f"num_points must be a positive integer, got {num_points}")
        cache_key = ('fibonacci', num_points, lmax, str(device))
    else:
        raise ValueError(
            f"Unknown grid_type '{grid_type}'. "
            f"Must be 'lebedev', 'gauss_legendre', or 'fibonacci'."
        )

    if cache_key in _GRID_CACHE:
        return _GRID_CACHE[cache_key]

    if grid_type == 'lebedev':
        points, weights, log_weights = LebedevGrid.get_grid_with_log_weights(lebedev_degree, device)

        if (weights <= 0).any() or not torch.isfinite(weights).all():
            num_invalid = ((weights <= 0) | (~torch.isfinite(weights))).sum().item()
            raise RuntimeError(
                f"Lebedev grid has {num_invalid} invalid weights (non-positive or non-finite)"
            )
        
        total = weights.sum().item()
        expected = 4.0 * math.pi
        if abs(total - expected) > 1e-6 * expected:
            raise RuntimeError(
                f"Lebedev weights do not sum to 4pi. Got {total:.8f}, expected {expected:.8f}"
            )
        
        actual_num_points = points.shape[0]
        
    elif grid_type == 'gauss_legendre':
        points, weights = generate_gauss_legendre_grid(gl_n_beta, gl_n_alpha, device)

        # log without clamping: positive weights required.
        if (weights <= 0).any():
            num_invalid = (weights <= 0).sum().item()
            raise RuntimeError(
                f"GL grid has {num_invalid} non-positive weights. Cannot compute log."
            )

        log_weights = torch.log(weights)
        actual_num_points = points.shape[0]

    else:  # fibonacci
        points = generate_fibonacci_grid(num_points).to(device=device, dtype=torch.float32)

        r = points.norm(dim=-1)
        if not torch.allclose(r, torch.ones_like(r), atol=1e-6):
            max_error = (r - 1.0).abs().max().item()
            raise RuntimeError(
                f"Fibonacci points not on unit sphere. Max error: {max_error:.3e}"
            )

        # Uniform weights: 4pi / num_points.
        weight_value = 4.0 * math.pi / float(num_points)

        if not math.isfinite(weight_value) or weight_value <= 0:
            raise RuntimeError(
                f"Invalid uniform Fibonacci weight: {weight_value}. "
                f"Check num_points={num_points}"
            )

        weights = torch.full((num_points,), weight_value, dtype=torch.float32, device=device)

        if (weights <= 0).any():
            raise RuntimeError("Fibonacci weights are non-positive (should never happen)")

        log_weights = torch.log(weights)
        actual_num_points = num_points

    _validate_grid(points, weights)
    config = GridConfig(
        grid_type=grid_type,
        num_points=actual_num_points,
        points=points,
        weights=weights,
        log_weights=log_weights
    )
    _GRID_CACHE[cache_key] = config
    return config


def clear_grid_cache():
    """Clear the grid cache."""
    global _GRID_CACHE
    _GRID_CACHE.clear()
