"""Layers and utilities for the spherical-harmonic head."""
from __future__ import annotations

import math
from typing import Optional, List, Iterable, Callable, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.o3 import spherical_harmonics
from e3nn.o3 import wigner_D
import logging
_LOG = logging.getLogger(__name__)


class Unsqueeze(nn.Module):
    """Module wrapper for `x.unsqueeze(dim)` to embed in nn.Sequential."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    """Module wrapper for `x.squeeze(dim)` to embed in nn.Sequential."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)

def l_block_sizes(lmax: int) -> List[int]:
    """Return a list of block sizes [2l+1] for l = 0..lmax."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    return [2 * l + 1 for l in range(lmax + 1)]

def split_by_l(sh: torch.Tensor, lmax: int) -> List[torch.Tensor]:
    """
    Split SH vector with exact dimension validation.
    Split a SH vector [*, sum(2l+1)] into a list of [*, 2l+1] blocks per l.
    """
    lmax = int(lmax)
    
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    
    sizes = l_block_sizes(lmax)
    S_expected = sum(sizes)
    
    if sh.dim() == 1:
        if sh.numel() != S_expected:
            raise RuntimeError(
                f"1D SH vector size mismatch!\n"
                f"  Expected: {S_expected} (lmax={lmax})\n"
                f"  Got: {sh.numel()}"
            )
        parts, i = [], 0
        for s in sizes:
            parts.append(sh[i:i + s])
            i += s
        return parts
    
    elif sh.dim() == 2:
        B, S = sh.shape
        if S != S_expected:
            raise RuntimeError(
                f"2D SH matrix width mismatch!\n"
                f"  Expected: [batch, {S_expected}] (lmax={lmax})\n"
                f"  Got: [{B}, {S}]"
            )
        parts, i = [], 0
        for s in sizes:
            parts.append(sh[:, i:i + s])
            i += s
        return parts
    
    else:
        raise ValueError(f"split_by_l expects 1D or 2D, got {sh.dim()}D shape {tuple(sh.shape)}")

class S2NonlinearityBlock(nn.Module):
    """
    Construct a fixed S^2 grid (Gauss-Legendre in beta x uniform in alpha) and
    precompute (real) spherical harmonics Y_lm on that grid, providing the
    SH->'grid projection matrix and quadrature weights.

    Attributes (buffers):
        grid_xyz:       [P, 3] unit vectors on S^2
        sh_to_grid:     [S, P] with S = (lmax+1)^2, P = res_beta * res_alpha
        w_vec:          [P] positive quadrature weights
        log_w_vec:      [P] log of quadrature weights (for log-space ops)
        w_sum:          ()  scalar sum of weights, integral  dOmega ~= Sigma w_p
    """
    def __init__(
        self,
        lmax: int,
        res_beta: int = 64,
        res_alpha: int = 64,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        normalize_sh: bool = True,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.res_beta = int(res_beta)
        self.res_alpha = int(res_alpha)
        self.activation = activation

        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype or torch.float32

        # ---- Build Gauss-Legendre nodes (beta via cosbeta roots) and alpha uniform
        # Do construction in float64 for stability, then cast buffers to requested dtype.
        from numpy.polynomial.legendre import leggauss
        comp_dtype = torch.float64

        x_np, w_beta_np = leggauss(self.res_beta)
        x_f64 = torch.tensor(x_np, device=device, dtype=comp_dtype)              # cosbeta nodes
        w_beta_f64 = torch.tensor(w_beta_np, device=device, dtype=comp_dtype)    # d(cosbeta)

        beta_f64 = torch.arccos(x_f64)                                           # beta  in  [0, pi]
        alpha_f64 = torch.linspace(
            0, 2 * math.pi * (self.res_alpha - 1) / self.res_alpha, self.res_alpha,
            device=device, dtype=comp_dtype
        )

        beta_mesh, alpha_mesh = torch.meshgrid(beta_f64, alpha_f64, indexing="ij")
        # Convert to xyz (float64)
        sin_beta = torch.sin(beta_mesh)
        xg = sin_beta * torch.cos(alpha_mesh)
        yg = sin_beta * torch.sin(alpha_mesh)
        zg = torch.cos(beta_mesh)
        xyz_f64 = torch.stack([xg, yg, zg], dim=-1).reshape(-1, 3)

        # Store xyz in requested dtype
        self.register_buffer("grid_xyz", xyz_f64.to(dtype), persistent=False)  # [P, 3]

        # Validate grid parameters first
        if self.res_beta <= 0:
            raise ValueError(f"res_beta must be positive, got {self.res_beta}")
        if self.res_alpha <= 0:
            raise ValueError(f"res_alpha must be positive, got {self.res_alpha}")

        # ---- Quadrature weights over the S^2 surface element (compute in float64)
        w_beta_2d_f64 = w_beta_f64[:, None].expand(self.res_beta, self.res_alpha)
        w_alpha_f64 = torch.tensor(2 * math.pi / self.res_alpha, device=device, dtype=comp_dtype)
        w_f64 = (w_beta_2d_f64 * w_alpha_f64).reshape(-1)  # [P], float64
        
        # Validate weights before logging
        if (w_f64 <= 0).any():
            raise RuntimeError(
                f"Quadrature weights contain non-positive values!\n"
                f"  Min weight: {w_f64.min().item():.3e}\n"
                "Grid construction is broken."
            )
        if not torch.isfinite(w_f64).all():
            raise RuntimeError("Quadrature weights contain NaN or Inf values!")
        
        # Register buffers in requested dtype
        self.register_buffer("w_vec", w_f64.to(dtype), persistent=False)
        self.register_buffer("log_w_vec", torch.log(w_f64).to(dtype), persistent=False)  # No epsilon
        self.register_buffer("w_sum", w_f64.sum().to(dtype), persistent=False)
        
        # Validate total weight (should integrate to 4pi)
        four_pi = torch.tensor(4 * math.pi, device=device, dtype=comp_dtype)
        weight_error = (w_f64.sum() - four_pi).abs().item()
        
        if weight_error > 1e-2:
            raise RuntimeError(
                f"Total quadrature weight {self.w_sum.item():.6f} "
                f"not close to 4pi ({four_pi.item():.6f}).\n"
                f"  Error: {weight_error:.6f}\n"
                "Grid integration is incorrect."
            )

        # ---- Precompute SH basis (evaluate in float64 for accuracy)
        try:
            # Newer e3nn accepts an Irreps object
            irreps = Irreps.spherical_harmonics(self.lmax)
            Y_f64 = spherical_harmonics(irreps, xyz_f64, normalize=normalize_sh)  # [P, S]
        except Exception:
            # Fallback: older signature that accepts a list of l
            Y_f64 = spherical_harmonics(list(range(self.lmax + 1)), xyz_f64, normalize=normalize_sh)  # [P, S]        
        self.register_buffer("sh_to_grid", Y_f64.T.to(dtype), persistent=False)  # [S, P]

        # Validate SH dimensions
        S_expected = (self.lmax + 1) ** 2
        if self.sh_to_grid.shape[0] != S_expected:
            raise RuntimeError(
                f"SH basis has wrong shape!\n"
                f"  Expected: [{S_expected}, {xyz_f64.shape[0]}]\n"
                f"  Got: {list(self.sh_to_grid.shape)}\n"
                "e3nn spherical_harmonics returned unexpected dimensions."
            )
        
        if self.sh_to_grid.shape[1] != xyz_f64.shape[0]:
            raise RuntimeError(
                f"SH basis grid points mismatch!\n"
                f"  Expected P={xyz_f64.shape[0]}, got {self.sh_to_grid.shape[1]}"
            )
        
        # Diagnostics (no longer optional - always validate)
        self._validate_orthonormality()

    def _validate_orthonormality(self, tol: float = 1e-2) -> None:
        """SH basis must be orthonormal within tolerance"""
        with torch.no_grad():
            # Check integral  |Y_lm|^2 dOmega ~= 1, numerically
            Yt = self.sh_to_grid.T.to(torch.float64)  # [P, S] in float64
            w = self.w_vec.to(torch.float64).unsqueeze(-1)
            integrated_norms = (w * (Yt ** 2)).sum(dim=0)
            err = (integrated_norms - 1.0).abs().max().item()
    
            if err > tol:
                raise RuntimeError(
                    f"SH basis normalization error {err:.6f} exceeds tolerance {tol}.\n"
                    f"Grid construction or SH evaluation is incorrect."
                )
    
            _LOG.info(f"[INIT] SH basis orthonormality verified (error={err:.6f})")

    def project_sh_to_grid(self, sh: torch.Tensor) -> torch.Tensor:
        """
        Project SH coefficients to grid with dimension validation.
            grid_vals = sh @ sh_to_grid  (if sh is [B,S]) or sh @ [S,P]
        """
        S_expected = self.sh_to_grid.shape[0]
        
        if sh.dim() == 1:
            if sh.numel() != S_expected:
                raise RuntimeError(
                    f"SH vector dimension mismatch!\n"
                    f"  Expected: {S_expected} (lmax={self.lmax})\n"
                    f"  Got: {sh.numel()}"
                )
            return torch.matmul(sh, self.sh_to_grid.to(device=sh.device, dtype=sh.dtype))  # [P]
        
        elif sh.dim() == 2:
            if sh.shape[1] != S_expected:
                raise RuntimeError(
                    f"SH matrix width mismatch!\n"
                    f"  Expected: [batch, {S_expected}] (lmax={self.lmax})\n"
                    f"  Got: {list(sh.shape)}"
                )
            return torch.matmul(sh, self.sh_to_grid.to(device=sh.device, dtype=sh.dtype))  # [B, P]
        
        else:
            raise ValueError(
                f"project_sh_to_grid expects [S] or [B,S], got shape {list(sh.shape)}"
            )

class QuadraticSelfTP(nn.Module):
    """
    Build higher-l content via an equivariant self tensor product h (x) h,
    followed by a learned mixing to a target irrep set up to `max_l_out`.

    The output irreps are densely populated with a chosen multiplicity
    per l (tapered with l).
    """
    def __init__(self, in_irreps: Irreps, max_l_out: int):
        super().__init__()
        self.in_irreps = Irreps(in_irreps)
        self.max_l_out = int(max_l_out)

        # Pick multiplicities per l
        mult = {0: 32, 1: 24, 2: 16, 3: 12, 4: 8, 5: 6, 6: 4, 7: 3, 8: 2}
        out_list = []
        for l in range(self.max_l_out + 1):
            p = "e" if (l % 2 == 0) else "o"
            m = mult.get(l, 2)
            out_list.append(f"{m}x{l}{p}")
        self.out_irreps = Irreps("+".join(out_list))

        self.tp = FullyConnectedTensorProduct(self.in_irreps, self.in_irreps, self.out_irreps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Validate input dimensions"""
        expected_dim = self.in_irreps.dim
        
        if h.dim() != 2:
            raise RuntimeError(
                f"QuadraticSelfTP expects [B, irreps_dim], "
                f"got {h.dim()}D shape {tuple(h.shape)}"
            )
        
        if h.shape[1] != expected_dim:
            raise RuntimeError(
                f"QuadraticSelfTP input dimension mismatch!\n"
                f"  Expected: {expected_dim} ({self.in_irreps})\n"
                f"  Got: {h.shape[1]}"
            )
        
        # h: [B, irreps_dim]
        return self.tp(h, h)


# Spectral convolution in SH space


class SphericalConvolution(nn.Module):
    """
    Lightweight spectral convolution acting independently per-l:

      y(l,m) = Sigma_{l',m'} W_{l} * delta_{l,l'} * x(l',m')

    implemented efficiently by repeating the per-l weight across m
    and performing a channelwise contraction.

    Parameters
    ----------
    lmax : int
        Maximum l supported by the input/output SH grid.
    channels_in : int
        Number of input channels (C_in).
    channels_out : int
        Number of output channels (C_out).

    Attributes
    ----------
    h : nn.Parameter
        Learnable weights with shape [C_out, C_in, lmax+1], one weight per l.
    """
    def __init__(self, lmax: int, channels_in: int, channels_out: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.lmax = int(lmax)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)

        dtype = dtype or torch.float32
        device = device or torch.device("cpu")

        # Per-l weights
        h = torch.randn(self.channels_out, self.channels_in, self.lmax + 1, device=device, dtype=dtype) * 0.1
        self.h = nn.Parameter(h)
        # Precompute a template of per-l scales (float32 CPU). Convert per-forward.
        ls_cpu = torch.arange(self.lmax + 1, dtype=torch.float32)
        w_scales_cpu = 2 * math.pi * torch.sqrt(4 * math.pi / (2 * ls_cpu + 1))
        self.register_buffer("_w_scales_template", w_scales_cpu, persistent=False)  # [lmax+1]

        # Cache repeating sizes for m-expansion
        self.register_buffer("_repeat_counts", torch.tensor([2 * l + 1 for l in range(self.lmax + 1)], dtype=torch.long), persistent=False)

    def forward(self, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Forward with complete dimension validation.
            x_prime: [B, C_in, S], with S = Sigma_{l=0}^{lmax} (2l+1)
            returns: [B, C_out, S]
        """
        if x_prime.dim() != 3:
            raise RuntimeError(
                f"SphericalConvolution expects 3D input [B, C_in, S], "
                f"got {x_prime.dim()}D shape {tuple(x_prime.shape)}"
            )
        
        B, C_in, S = x_prime.shape
        S_expected = sum(2 * l + 1 for l in range(self.lmax + 1))
        
        if S != S_expected:
            raise RuntimeError(
                f"SphericalConvolution SH dimension mismatch!\n"
                f"  Expected S={S_expected} (lmax={self.lmax})\n"
                f"  Got: [B={B}, C_in={C_in}, S={S}]"
            )
        
        if C_in != self.channels_in:
            raise RuntimeError(
                f"SphericalConvolution channel mismatch!\n"
                f"  Expected: {self.channels_in}\n"
                f"  Got: {C_in}"
            )
        
        # Build per-l repeats
        device = x_prime.device
        
        # Adapt cached template to current device/dtype once per forward
        w_scales = self._w_scales_template.to(device=device, dtype=self.h.dtype)            # [lmax+1]
        w_per_l = w_scales.view(1, 1, -1) * self.h                                          # [C_out, C_in, lmax+1]
        w_full = w_per_l.repeat_interleave(self._repeat_counts.to(device), dim=-1)          # [C_out, C_in, S]
        w_full = w_full.to(dtype=x_prime.dtype)                                             # match activation dtype

        # Contract over C_in and S: (B,C_in,S) x (C_out,C_in,S) -> (B,C_out,S)
        y = torch.einsum("bis,ois->bos", x_prime, w_full)        
        return y


# Spectral mixers and balancing


class LearnableSpectralMixer(nn.Module):
    """
    Per-l learnable gains applied blockwise across m:
      y_{l,m} = w_l * x_{l,m},  w_l = exp(theta_l) > 0
    """
    def __init__(self, lmax: int, init_per_l: torch.Tensor):
        super().__init__()
        self.lmax = int(lmax)

        if init_per_l.numel() != self.lmax + 1:
            raise ValueError(
                f"init_per_l must have {self.lmax+1} elements (l=0..{self.lmax}), "
                f"got {init_per_l.numel()}"
            )

        init = init_per_l.float().clamp(min=1e-6)
        self.log_l_weights = nn.Parameter(init.log())  # [lmax+1]

        # cache counts of m per l for repeat_interleave
        self.register_buffer(
            "_repeat_counts",
            torch.tensor([2 * l + 1 for l in range(self.lmax + 1)], dtype=torch.long),
            persistent=False,
        )

    def get_per_l_weights(self, *, device=None, dtype=None) -> torch.Tensor:
        """Current positive weights per-l (detached only if you call .detach() yourself)."""
        w = self.log_l_weights.exp()
        if device is not None or dtype is not None:
            w = w.to(device=device or w.device, dtype=dtype or w.dtype)
        return w

    def forward(self, sh: torch.Tensor) -> torch.Tensor:
        """
        sh: [B, S] with S = sum_{l=0}^{lmax}(2l+1)
        returns: [B, S]
        """
        if sh.dim() != 2:
            raise RuntimeError(
                f"LearnableSpectralMixer expects [B, S], got {sh.dim()}D shape {tuple(sh.shape)}"
            )

        S_expected = int(self._repeat_counts.sum().item())
        if sh.shape[1] != S_expected:
            raise RuntimeError(
                f"LearnableSpectralMixer SH dimension mismatch!\n"
                f"  Expected: [batch, {S_expected}] (lmax={self.lmax})\n"
                f"  Got: {list(sh.shape)}"
            )

        # Expand per-l weights across m and apply in one shot (fast & differentiable)
        w_l = self.get_per_l_weights(device=sh.device, dtype=sh.dtype)                  # [lmax+1]
        w_per_m = w_l.repeat_interleave(self._repeat_counts.to(sh.device))              # [S]
        return sh * w_per_m.unsqueeze(0)                                                # [B, S]

class PreLBalance(nn.Module):
    """
    Fixed per-l multipliers (not learnable) for pre-balancing spectral energy.
    Use this to inject channel-specific hand-crafted shaping of the spectrum.
    """
    def __init__(self, lmax: int, gains: Iterable[float]):
        super().__init__()
        self.lmax = int(lmax)
        
        if self.lmax < 0:
            raise ValueError(f"lmax must be non-negative, got {self.lmax}")
        
        g_list = list(gains)
        if len(g_list) != self.lmax + 1:
            raise ValueError(
                f"PreLBalance gains must have exactly {self.lmax + 1} elements (one per l=0..{self.lmax}).\n"
                f"  Got: {len(g_list)} elements"
            )
        
        g = torch.tensor(g_list, dtype=torch.float32)
        
        # Validate gains are positive and finite
        if (g <= 0).any():
            raise ValueError(
                f"PreLBalance gains must be strictly positive!\n"
                f"  Min gain: {g.min().item():.3e}"
            )
        
        if not torch.isfinite(g).all():
            raise ValueError("PreLBalance gains contain NaN or Inf values!")
        
        self.register_buffer("gains", g, persistent=False)

    def forward(self, sh: torch.Tensor) -> torch.Tensor:
        """Apply fixed per-l gains with dimension validation"""
        S_expected = sum(2 * l + 1 for l in range(self.lmax + 1))
        
        if sh.dim() != 2:
            raise RuntimeError(
                f"PreLBalance expects [B, S], got {sh.dim()}D shape {tuple(sh.shape)}"
            )
        
        if sh.shape[1] != S_expected:
            raise RuntimeError(
                f"PreLBalance SH dimension mismatch!\n"
                f"  Expected: [batch, {S_expected}] (lmax={self.lmax})\n"
                f"  Got: {list(sh.shape)}"
            )
        
        out, idx = [], 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            out.append(sh[:, idx:idx + dim] * self.gains[l])
            idx += dim
        
        # Verify exact indexing
        if idx != S_expected:
            raise RuntimeError(
                f"Internal indexing error in PreLBalance!\n"
                f"  Expected {S_expected} dims, consumed {idx}"
            )
        
        return torch.cat(out, dim=-1)

# Channel-specific spectral initialization helpers


def _init_l_weights_for_channel(channel_idx: int, lmax: int) -> torch.Tensor:
    """
    Initialize per-l gains tailored to each channel's target sharpness.
    
    Make initial gains FLAT or slightly favor high-l
    to counteract the network's natural bias toward low-l at init.

    Returns
    -------
    w : torch.Tensor of shape [lmax+1], positive values
    """
    lmax = int(lmax)
    
    # Start nearly FLAT across all l
    # This prevents any l from dominating at initialization
    ls = torch.arange(lmax + 1, dtype=torch.float32)
    
    if channel_idx == 0:
        # Sharpest: BOOST high-l from the start
        # Linear ramp: 0.5 -> 2.0 from l=0 to l=lmax
        w = 0.5 + 1.5 * (ls / max(lmax, 1))
        w[0] = 0.3  # Keep DC moderate
    elif channel_idx == 1:
        # Medium sharp: moderate boost to high-l
        w = 0.7 + 0.8 * (ls / max(lmax, 1))
        w[0] = 0.4
    elif channel_idx == 2:
        # Broader: nearly flat
        w = torch.ones(lmax + 1) * 0.8
        w[0] = 0.5
    else:
        # Hemispheric: emphasize l=1, suppress high-l
        w = torch.ones(lmax + 1) * 0.3
        if lmax >= 1:
            w[1] = 1.2
        w[0] = 0.4
    
    # Validate computed weights (no clamping to hide issues)
    if (w <= 0).any():
        raise RuntimeError(
            f"_init_l_weights_for_channel produced non-positive weights!\n"
            f"  Channel: {channel_idx}, lmax: {lmax}\n"
            f"  Min weight: {w.min().item():.3e}\n"
            "Weight initialization logic is broken."
        )
    
    if (w > 10.0).any() or (w < 0.01).any():
        raise RuntimeError(
            f"_init_l_weights_for_channel produced unreasonable weights!\n"
            f"  Channel: {channel_idx}, lmax: {lmax}\n"
            f"  Range: [{w.min().item():.3f}, {w.max().item():.3f}]\n"
            "Weight initialization is likely unstable."
        )
    
    return w

def _per_l_gains(channel_idx: int, lmax: int) -> List[float]:
    """
    Fixed pre-l gains used by `PreLBalance` for each channel.

    Mirrors the logic that de-emphasizes zonal dominance in certain
    channels and boosts higher ls for sharper channels.
    """
    lmax = int(lmax)
    if channel_idx == 3:
        return [0.05] + [2.5] + [0.15] * max(0, lmax - 1)
    elif channel_idx == 2:
        # ANTI-RING: suppress l=2 while boosting l>=3
        gains: List[float] = [0.05, 0.3]  # l=0,1
        if lmax >= 2:
            gains.append(0.5)             # l=2
        if lmax >= 3:
            # Boost l=3 strongly, taper for higher l
            for l in range(3, lmax + 1):
                gains.append(2.0 + 0.3 * (l - 3))  # 2.0, 2.3, 2.6, ...
        return gains[: lmax + 1]
    elif channel_idx == 1:
        return [0.0] + [0.6 * (l * (l + 1)) for l in range(1, lmax + 1)]
    else:
        return [0.0] + [0.9 * (l * (l + 1)) for l in range(1, lmax + 1)]


def _per_l_gains_minimal(channel_idx: int, lmax: int) -> List[float]:
    """
    MINIMAL fixed per-l gains to avoid killing high-l.
    
    Philosophy: 
    - DC = 1.0 always (preserve normalization reference)
    - All other l ~= 1.0 (let LearnableSpectralMixer do the shaping)
    - Ch0/Ch1: slight HIGH-l boost to encourage cap formation

    Returns
    -------
    gains : List[float] of length lmax+1
    """
    lmax = int(lmax)
    gains = [1.0]  # DC component always 1.0

    if channel_idx in [0, 1]:
        # Sharp channels: BOOST high-l slightly
        for l in range(1, lmax + 1):
            gains.append(1.0 + 0.2 * (l / max(lmax, 1)))
    elif channel_idx == 2:
        # Medium channel: nearly flat
        gains.extend([1.0] * lmax)
    else:
        # Hemispheric: emphasize l=1
        if lmax >= 1:
            gains.append(1.3)
        if lmax >= 2:
            gains.extend([0.9] * (lmax - 1))

    # Verify exact construction (no padding loops)
    if len(gains) != lmax + 1:
        raise RuntimeError(
            f"Internal error in _per_l_gains_minimal!\n"
            f"  Construction produced {len(gains)} gains for lmax={lmax}, expected {lmax + 1}."
        )
    return gains

def get_wigner_d_matrix(l: int, R: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Get Wigner-D matrix D^l(R) for a given rotation R and degree l.

    Uses e3nn's matrix_to_angles for the Euler-angle convention that yields
    correct Wigner-D matrices for l >= 2.

    Args:
        l: Spherical harmonic degree (0, 1, 2, ...)
        R: [3, 3] rotation matrix
        device: torch device
        dtype: torch dtype
    
    Returns:
        D: [(2l+1), (2l+1)] Wigner-D matrix
    """
    
    R_cpu = R.detach().cpu()
    alpha, beta, gamma = o3.matrix_to_angles(R_cpu)
    
    # Compute Wigner D on CPU (e3nn requirement for some versions)
    D = wigner_D(l=l, alpha=alpha, beta=beta, gamma=gamma)
    
    # Move result to correct device and dtype
    return D.to(device=device, dtype=dtype)

def get_wigner_d_matrix_batched(l: int, R_batch: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Get batched Wigner-D matrices D^l(R) for a batch of rotation matrices.

    Args:
        l: Spherical harmonic degree (0, 1, 2, ...)
        R_batch: [batch, 3, 3] rotation matrices
        device: torch device
        dtype: torch dtype

    Returns:
        D: [batch, (2l+1), (2l+1)] Wigner-D matrices
    """

    if R_batch.dim() != 3:
        raise ValueError(f"Expected R_batch shape [B, 3, 3], got {R_batch.shape}")

    batch_size = R_batch.shape[0]

    # For l=0, D is always identity (1x1)
    if l == 0:
        return torch.ones(batch_size, 1, 1, device=device, dtype=dtype)

    D_list = []
    for b in range(batch_size):
        R_b = R_batch[b]  # [3, 3]

        # Use e3nn's matrix_to_angles
        R_b_cpu = R_b.detach().cpu()
        alpha, beta, gamma = o3.matrix_to_angles(R_b_cpu)

        # Compute Wigner D on CPU
        D_b = wigner_D(l=l, alpha=alpha, beta=beta, gamma=gamma)
        D_list.append(D_b)

    return torch.stack(D_list, dim=0).to(device=device, dtype=dtype)

def rotate_equivariant_embedding_world_to_aligned(
    embedding_world: torch.Tensor,
    R_feat_to_world: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate an equivariant embedding from world frame to v_hat-aligned frame.

    Only 480D embeddings: 128x0e + 64x1e + 32x2e.

    EMBEDDING STRUCTURE (480D):
      - l=0 (indices 0-127): 128 scalars - INVARIANT, no rotation
      - l=1 (indices 128-319): 64 vectors (3 each) - ROTATE with R_world_to_aligned
      - l=2 (indices 320-479): 32 rank-2 tensors (5 each) - ROTATE with D^2(R_world_to_aligned)
    
    Args:
        embedding_world: [batch, 480] equivariant embedding in world frame
        R_feat_to_world: [batch, 3, 3] frame rotation (cols: x_feat, y_feat, z_feat in world)
        device: torch device
    
    Returns:
        embedding_aligned: [batch, 480] rotated to v_hat-aligned frame
    
    Raises:
        ValueError: If embedding dimension is not 480
    """
    batch_size = embedding_world.shape[0]
    embedding_dim = embedding_world.shape[-1]
    dtype = embedding_world.dtype
    
    # Only 480D (128x0e+64x1e+32x2e) is supported
    if embedding_dim != 480:
        raise ValueError(
            f"Expected embedding dim=480 (128x0e+64x1e+32x2e), got {embedding_dim}. "
            f"SimpleO2Combiner preserves raw SevenNet irreps -- no 510-dim TP expansion."
        )
    
    # R_feat_to_world maps aligned -> world, transpose gives world -> aligned
    R_world_to_aligned = R_feat_to_world.transpose(1, 2)  # [batch, 3, 3]
    embedding_aligned = embedding_world.clone()
    
    # Get Wigner-D matrix for l=2
    D2_list = []
    for b in range(batch_size):
        R_b = R_world_to_aligned[b]  # [3, 3]
        D2 = get_wigner_d_matrix(l=2, R=R_b, device=device, dtype=dtype)  # [5, 5]
        D2_list.append(D2)
    D2_batch = torch.stack(D2_list, dim=0)  # [batch, 5, 5]
    
    # l=0 (SCALARS): INVARIANT - indices 0-127
    # Already correct, no rotation needed
    
    # l=1 (VECTORS): ROTATE with R_world_to_aligned - indices 128-319
    vectors_world = embedding_world[:, 128:320]  # [batch, 192]
    vectors_world_reshaped = vectors_world.reshape(batch_size, 64, 3)  # [batch, 64, 3]
    vectors_aligned_reshaped = torch.einsum('bij,bkj->bki', R_world_to_aligned, vectors_world_reshaped)
    vectors_aligned = vectors_aligned_reshaped.reshape(batch_size, 192)
    embedding_aligned[:, 128:320] = vectors_aligned
    
    # l=2 (RANK-2 TENSORS): ROTATE with Wigner-D - indices 320-479
    tensors_world = embedding_world[:, 320:480]  # [batch, 160]
    tensors_world_reshaped = tensors_world.reshape(batch_size, 32, 5)  # [batch, 32, 5]
    tensors_aligned_reshaped = torch.einsum('bij,bkj->bki', D2_batch, tensors_world_reshaped)
    tensors_aligned = tensors_aligned_reshaped.reshape(batch_size, 160)
    embedding_aligned[:, 320:480] = tensors_aligned
    
    return embedding_aligned

def rotate_irreps_tensor(
    tensor: torch.Tensor,
    irreps: Irreps,
    R: torch.Tensor,
    device: torch.device,
    inverse: bool = False
) -> torch.Tensor:
    """
    Rotate an irreps-packed tensor of any structure, using Wigner D^l per block:
    - l=0 (scalars): unchanged (D^0 = 1)
    - l=1 (vectors): 3x3 Wigner D^1 (equivalent to R for Cartesian)
    - l=2 (rank-2):  5x5 Wigner D^2
    - l=3 (rank-3):  7x7 Wigner D^3
    - etc.

    Args:
        tensor: [batch, dim] packed irreps tensor
        irreps: e3nn Irreps object describing the tensor layout
        R: [batch, 3, 3] rotation matrices (e.g., R_feat_to_world)
        device: torch device
        inverse: If True, use R^T (inverse rotation)

    Returns:
        rotated_tensor: [batch, dim] with each block properly rotated

    Example:
        >>> irreps = Irreps("16x0e+8x1e+4x2e+2x3e")  # 74 dims
        >>> tensor = torch.randn(batch, 74)
        >>> R = get_rotation_matrix(...)  # [batch, 3, 3]
        >>> rotated = rotate_irreps_tensor(tensor, irreps, R, device)
    """
    if not isinstance(irreps, Irreps):
        irreps = Irreps(irreps)

    batch_size = tensor.shape[0]
    dtype = tensor.dtype

    # Validate dimension
    expected_dim = irreps.dim
    if tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"Tensor dimension {tensor.shape[-1]} doesn't match irreps {irreps} (dim={expected_dim})"
        )

    # Apply inverse if requested
    if inverse:
        R = R.transpose(1, 2)  # [batch, 3, 3]

    # Pre-compute Wigner D matrices for all unique l values
    unique_ls = sorted(set(ir.l for _, ir in irreps))
    D_cache = {}
    for l in unique_ls:
        if l > 0:  # l=0 doesn't need rotation
            D_cache[l] = get_wigner_d_matrix_batched(l, R, device, dtype)  # [batch, 2l+1, 2l+1]

    # Process each irrep block
    rotated_tensor = tensor.clone()
    idx = 0

    for mul, ir in irreps:
        l = ir.l
        block_size = 2 * l + 1
        total_block_size = mul * block_size

        if l == 0:
            # Scalars are invariant - no rotation needed
            idx += total_block_size
            continue

        # Extract block
        block = tensor[:, idx:idx + total_block_size]  # [batch, mul * (2l+1)]
        block_reshaped = block.reshape(batch_size, mul, block_size)  # [batch, mul, 2l+1]

        # Get Wigner D for this l
        D_l = D_cache[l]  # [batch, 2l+1, 2l+1]

        # Rotate: v_rotated[b,m,i] = sum_j D[b,i,j] * v[b,m,j]
        block_rotated = torch.einsum('bij,bmj->bmi', D_l, block_reshaped)  # [batch, mul, 2l+1]

        # Store back
        rotated_tensor[:, idx:idx + total_block_size] = block_rotated.reshape(batch_size, -1)

        idx += total_block_size

    # Validate we consumed all dimensions
    if idx != expected_dim:
        raise RuntimeError(
            f"Internal error: processed {idx} dims but expected {expected_dim}"
        )

    return rotated_tensor


def rotate_frame_aligned_features(
    frame_features: torch.Tensor,
    R_feat_to_world: torch.Tensor,
    device: torch.device,
    frame_irreps: str = "16x0e+8x1e+4x2e+2x3e"
) -> torch.Tensor:
    """
    Rotate frame-aligned features from feature frame to world frame (or vice versa).

    This is a convenience wrapper for rotate_irreps_tensor specifically for
    the frame encoder output format.

    Args:
        frame_features: [batch, 74] frame-aligned features
        R_feat_to_world: [batch, 3, 3] rotation from feature to world frame
        device: torch device
        frame_irreps: Irreps string for frame features (default: "16x0e+8x1e+4x2e+2x3e")

    Returns:
        rotated_features: [batch, 74] features in world frame
    """
    irreps = Irreps(frame_irreps)
    return rotate_irreps_tensor(
        tensor=frame_features,
        irreps=irreps,
        R=R_feat_to_world,
        device=device,
        inverse=False  # feature -> world
    )


def rotate_o2_asymmetric_to_aligned(
    o2_embedding: torch.Tensor,
    R_feat_to_world: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate O2 asymmetric embedding from world frame to aligned (feature) frame.

    Convenience wrapper for 480D O2 embeddings (128x0e+64x1e+32x2e).

    Args:
        R_feat_to_world: [batch, 3, 3] rotation from feature to world frame
        device: torch device

    Returns:
        rotated_embedding: [batch, embedding_dim] in aligned frame
    """
    embedding_dim = o2_embedding.shape[-1]

    # Determine irreps based on dimension
    if embedding_dim == 480:
        # Raw encoder output: 128x0e + 64x1e + 32x2e
        irreps = Irreps("128x0e+64x1e+32x2e")
    else:
        raise ValueError(
            f"Expected O2 embedding dim 480 (128x0e+64x1e+32x2e), got {embedding_dim}. "
            f"SimpleO2Combiner preserves raw SevenNet irreps -- no 510-dim TP expansion. "
            f"If using a custom irreps structure, use rotate_irreps_tensor directly."
        )

    # World -> aligned means we apply R^T (inverse of feat_to_world)
    return rotate_irreps_tensor(
        tensor=o2_embedding,
        irreps=irreps,
        R=R_feat_to_world,
        device=device,
        inverse=True  # world -> aligned
    )

__all__ = [
    # Shape helpers
    "Unsqueeze", "Squeeze",
    # S^2 / SH utilities
    "S2NonlinearityBlock",
    # Equivariant heads / ops
    "QuadraticSelfTP",
    "SphericalConvolution",
    "LearnableSpectralMixer",
    "PreLBalance",
    # Channel spectral init helpers
    "_init_l_weights_for_channel",
    "_per_l_gains",
    "_per_l_gains_minimal",
    # SH helpers kept
    "split_by_l",
    "l_block_sizes",
    # Equivariant frame rotation
    "rotation_matrix_to_euler_angles",
    "get_wigner_d_matrix",
    "rotate_equivariant_embedding_world_to_aligned",
    # General-purpose irreps rotation
    "get_wigner_d_matrix_batched",
    "rotate_irreps_tensor",
    "rotate_frame_aligned_features",
    "rotate_o2_asymmetric_to_aligned",
]
