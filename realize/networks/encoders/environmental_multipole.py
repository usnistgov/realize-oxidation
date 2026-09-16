"""Environmental Multipole Embedding.

Computes the spherical harmonic expansion of LOCAL substrate atom density
as viewed from the O2 center of mass.

    c_lm = Sigma_j  w_Zj * f(r_j) * Y_lm(r_hat_j)

where:
    r_j = R_j - R_COM   (relative vector from O2 COM to substrate atom j)
    f(r) = 0.5 * (1 + cos(pi * r / r_cut))  for r < r_cut, 0 otherwise
          (cosine cutoff, standard in MLIP literature)
    w_Zj = learnable per-species weight (Si vs O)
    Y_lm = real spherical harmonics (e3nn convention)

Output irreps:  1x0e + 1x1e + 1x2e  =  9 dimensions

Physical meaning:
    l=0:  local weighted atom density ("how much stuff is nearby?")
    l=1:  dipole of local environment ("where are nearest atoms?")
    l=2:  quadrupole ("is the local environment flat, stepped, or pocket-like?")

Properties:
    - E(3)-equivariant: Y_lm(R*r_hat) = Sigma_m' D^l_mm'(R) Y_lm'(r_hat)
    - Local: cosine cutoff at r_cut (default 9.0 Angstrom), smooth and strictly zero beyond
    - Covers SevenNet dead zone (5-8 Angstrom) while remaining physically local
    - Minimal: 9 dims, 2 learnable parameters (per-species weights)
    - r_cut is fixed (architectural choice), not learnable
"""

import torch
import torch.nn as nn
from e3nn import o3
from typing import Optional
import logging
_LOG = logging.getLogger(__name__)


class EnvironmentalMultipole(nn.Module):
    """SH expansion of substrate atom density around the O2 COM.

    Provides long-range equivariant context missing from the local SevenNet embedding.
    """

    def __init__(
        self,
        lmax: int = 2,
        r_cut: float = 9.0,
        n_species: int = 2,     # 0=Si, 1=O (substrate oxygen)
        learnable_weights: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.lmax = lmax
        self.r_cut = r_cut
        self.device = device or torch.device('cpu')

        # Output irreps: 1x0e + 1x1e + ... + 1xlmax_e
        self.output_irreps = o3.Irreps(
            "+".join(f"1x{l}e" for l in range(lmax + 1))
        )
        self.output_dim = self.output_irreps.dim  # (lmax+1)^2 = 9 for lmax=2

        # Per-species weights, init equal (Si=O=1.0); network learns to differentiate.
        if learnable_weights:
            self.species_weights = nn.Parameter(torch.ones(n_species, device=self.device))
        else:
            self.register_buffer('species_weights', torch.ones(n_species, device=self.device))

        # l-indices for spherical_harmonics call
        self._l_list = list(range(lmax + 1))

        _LOG.info(f"[EnvironmentalMultipole] Cosine cutoff r_cut={r_cut:.1f} Angstrom, lmax={lmax}")
        self.to(self.device)

    def forward(
        self,
        com: torch.Tensor,                    # [B, 3] O2 center of mass
        substrate_positions: torch.Tensor,     # [N_sub, 3] all substrate atom positions
        substrate_species_idx: torch.Tensor,   # [N_sub] long tensor: 0=Si, 1=O
        cell: Optional[torch.Tensor] = None,   # [3, 3] cell vectors for MIC
        pbc: Optional[torch.Tensor] = None,    # [3] bool
    ) -> torch.Tensor:
        """Compute multipole coefficients.

        Returns:
            coeffs: [B, (lmax+1)^2] spherical harmonic coefficients.
                    For lmax=2: [B, 9]  (1 + 3 + 5 dims)
        """
        # Relative vectors: [B, N_sub, 3]
        rel_vec = substrate_positions.unsqueeze(0) - com.unsqueeze(1)

        if cell is not None and pbc is not None:
            rel_vec = self._apply_mic(rel_vec, cell, pbc)

        # Distances: [B, N_sub]
        r_sq = (rel_vec ** 2).sum(dim=-1)
        r = r_sq.sqrt().clamp(min=1e-8)

        # Cosine cutoff: f(r) = 0.5*(1 + cos(pi*r/r_cut)) for r < r_cut else 0; smooth, strictly local.
        radial_weight = torch.where(
            r < self.r_cut,
            0.5 * (1.0 + torch.cos(torch.pi * r / self.r_cut)),
            torch.zeros_like(r),
        )  # [B, N_sub]

        sp_weight = self.species_weights[substrate_species_idx]  # [N_sub]
        weight = radial_weight * sp_weight.unsqueeze(0)  # [B, N_sub]

        # Unit direction vectors: [B, N_sub, 3]
        r_norm = r_sq.unsqueeze(-1).sqrt().clamp(min=1e-8)
        r_hat = rel_vec / r_norm

        # Y_lm at each direction: [B, N_sub, (lmax+1)^2]
        ylm = o3.spherical_harmonics(
            self._l_list, r_hat, normalize=True
        )

        # c_lm = Sigma_j w_j * f(r_j) * Y_lm(r_hat_j)
        coeffs = (weight.unsqueeze(-1) * ylm).sum(dim=1)  # [B, (lmax+1)^2]

        return coeffs

    @staticmethod
    def _apply_mic(rel_vec: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
        """Apply minimum image convention for periodic boundary conditions.
        
        ASE convention: cell[i] = i-th lattice vector (row).
        Cartesian -> fractional: frac = cart @ cell_inv  (i.e. cell_inv = inv(cell))
        Fractional -> Cartesian: cart = frac @ cell
        """
        # Cartesian -> fractional: frac_j = Sigma_i cart_i * cell_inv_ij
        cell_inv = torch.linalg.inv(cell)  # [3, 3]
        frac = torch.einsum('...i, ij -> ...j', rel_vec, cell_inv)

        # Wrap to [-0.5, 0.5] on periodic dimensions.
        for dim in range(3):
            if pbc[dim]:
                frac[..., dim] = frac[..., dim] - torch.round(frac[..., dim])

        # Fractional -> Cartesian: cart_j = Sigma_i frac_i * cell_ij
        rel_vec_mic = torch.einsum('...i, ij -> ...j', frac, cell)
        return rel_vec_mic

    def get_species_idx(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Map atomic numbers to species index: Si (Z=14) -> 0, O (Z=8) and all else -> 1."""
        species_idx = torch.ones_like(atomic_numbers, dtype=torch.long)
        species_idx[atomic_numbers == 14] = 0  # Si -> 0
        return species_idx
