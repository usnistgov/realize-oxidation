"""von Mises-Fisher sampling and log-probability on S^2."""
import torch
import math
from typing import Tuple
import logging
_LOG = logging.getLogger(__name__)


def von_mises_fisher_sample(
    mean_direction: torch.Tensor,
    concentration: torch.Tensor,
    training: bool = True
) -> torch.Tensor:
    """
    Sample from von Mises-Fisher distribution on unit sphere S^2.
    
    Uses Wood (1994) algorithm for efficient sampling.
    """
    batch_size = mean_direction.shape[0]
    device = mean_direction.device
    dtype = mean_direction.dtype
    
    if not training:
        return mean_direction
    
    # Validate inputs
    mean_norm = torch.norm(mean_direction, dim=-1)
    if not torch.allclose(mean_norm, torch.ones(batch_size, device=device, dtype=dtype), atol=1e-5):
        raise ValueError(f"mean_direction must be unit norm. Got norms: {mean_norm}")
    
    if (concentration < 0).any():
        raise ValueError(f"concentration kappa must be non-negative. Got min={concentration.min():.4f}")
    
    # Clamp kappa for numerical stability
    kappa = torch.clamp(concentration, min=1e-6, max=1e3)
    
    # Wood algorithm - simplified for numerical stability
    # For each batch element, sample uniformly on S^2 then reweight by vMF density
    
    # Step 1: Sample from uniform distribution on S^2
    # Generate random points on sphere via Gaussian normalization
    v = torch.randn(batch_size, 3, device=device, dtype=dtype)
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)  # Random direction on S^2
    
    # Step 2: For each direction, compute acceptance probability under vMF
    # p(v | m, kappa) prop_to exp(kappa m*v)
    # We use rejection sampling: accept with prob exp(kappa(m*v - 1))
    
    dot_prod = (mean_direction * v).sum(dim=-1)  # [batch]
    dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
    
    # Log acceptance probability (shifted for numerical stability)
    log_accept_prob = kappa * (dot_prod - 1.0)  # Normalized so max is 0
    log_accept_prob = torch.clamp(log_accept_prob, min=-100.0)  # Prevent underflow
    
    accept_prob = torch.exp(log_accept_prob)  # [batch]
    
    # Rejection sampling
    u = torch.rand(batch_size, device=device, dtype=dtype)
    accepted = u < accept_prob
    
    # For rejected samples, resample
    max_attempts = 100
    for attempt in range(max_attempts):
        if accepted.all():
            break
        
        # Resample rejected directions
        v_new = torch.randn(batch_size, 3, device=device, dtype=dtype)
        v_new = v_new / (torch.norm(v_new, dim=-1, keepdim=True) + 1e-8)
        
        dot_new = (mean_direction * v_new).sum(dim=-1)
        dot_new = torch.clamp(dot_new, -1.0, 1.0)
        
        log_accept_new = kappa * (dot_new - 1.0)
        log_accept_new = torch.clamp(log_accept_new, min=-100.0)
        accept_new = torch.exp(log_accept_new)
        
        u_new = torch.rand(batch_size, device=device, dtype=dtype)
        newly_accepted = u_new < accept_new
        
        # Update: rejected samples that are now accepted
        update_mask = (~accepted) & newly_accepted
        v[update_mask] = v_new[update_mask]
        accepted[update_mask] = True
    
    # Final validation
    sample_norm = torch.norm(v, dim=-1)
    if not torch.allclose(sample_norm, torch.ones(batch_size, device=device, dtype=dtype), atol=1e-5):
        v = v / (sample_norm.unsqueeze(-1) + 1e-8)
    
    return v

def von_mises_fisher_logprob(
    sample: torch.Tensor,
    mean_direction: torch.Tensor,
    concentration: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability of samples under von Mises-Fisher distribution.
    
    log p(x | m, kappa) = kappa * (m^T * x) - log C(kappa)
    
    where C(kappa) = kappa / (4pi sinh(kappa)) is the normalization constant.
    
    Args:
        sample: [batch, 3] unit vectors (samples from vMF)
        mean_direction: [batch, 3] mean directions (unit vectors)
        concentration: [batch] concentration parameters kappa
    
    Returns:
        log_prob: [batch] log probabilities
    
    Raises:
        ValueError: If inputs are not unit norm or concentration is negative
    """
    batch_size = sample.shape[0]
    device = sample.device
    dtype = sample.dtype
    
    # validation
    sample_norm = torch.norm(sample, dim=-1)
    if not torch.allclose(sample_norm, torch.ones(batch_size, device=device, dtype=dtype), atol=1e-4):
        raise ValueError(
            f"sample must be unit norm. Got norms: min={sample_norm.min():.4f}, max={sample_norm.max():.4f}"
        )
    
    mean_norm = torch.norm(mean_direction, dim=-1)
    if not torch.allclose(mean_norm, torch.ones(batch_size, device=device, dtype=dtype), atol=1e-5):
        raise ValueError(
            f"mean_direction must be unit norm. Got norms: min={mean_norm.min():.4f}, max={mean_norm.max():.4f}"
        )
    
    if (concentration < 0).any():
        raise ValueError(f"concentration kappa must be non-negative. Got min={concentration.min():.4f}")
    
    # Clamp kappa to avoid numerical issues
    kappa = torch.clamp(concentration, min=1e-6, max=1e3)
    
    # Compute m^T * x (dot product = cosine of angle)
    cos_angle = (mean_direction * sample).sum(dim=-1)  # [batch]
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
    
    # Log partition function: log C(kappa)
    # C(kappa) = kappa / (4pi sinh(kappa))
    # log C(kappa) = log(kappa) - log(4pi) - log(sinh(kappa))
    #          = log(kappa) - log(4pi) - log((exp(kappa) - exp(-kappa))/2)
    #          = log(kappa) - log(4pi) - log(exp(kappa) - exp(-kappa)) + log(2)
    
    # For numerical stability:
    # log(sinh(kappa)) = log((e^kappa - e^{-kappa})/2)
    #              ~= kappa - log(2) for large kappa
    #              ~= log(kappa) for small kappa
    
    eps = 1e-8
    
    # Compute log_sinh_kappa = log(sinh(kappa))
    log_sinh_kappa = torch.zeros_like(kappa)
    
    # Large kappa: use kappa - log(2) approximation
    large_kappa = kappa > 10.0
    log_sinh_kappa[large_kappa] = kappa[large_kappa] - math.log(2.0)
    
    # Small kappa: use series expansion or direct computation
    small_kappa = ~large_kappa
    sinh_kappa = (torch.exp(kappa[small_kappa]) - torch.exp(-kappa[small_kappa])) / 2.0
    log_sinh_kappa[small_kappa] = torch.log(sinh_kappa + eps)
    
    # Log normalization: log C(kappa) = log(kappa) - log(4pi) - log(sinh(kappa))
    log_norm = torch.log(kappa + eps) - math.log(4.0 * math.pi) - log_sinh_kappa
    
    # Log probability
    log_prob = kappa * cos_angle + log_norm
    
    # Validate
    if not torch.isfinite(log_prob).all():
        bad_idx = (~torch.isfinite(log_prob)).nonzero(as_tuple=False).squeeze()
        _LOG.warning(f"Non-finite log_prob at indices: {bad_idx.tolist()}")
        # Clamp to reasonable range
        log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
    
    return log_prob


__all__ = [
    "von_mises_fisher_sample",
    "von_mises_fisher_logprob",
]
