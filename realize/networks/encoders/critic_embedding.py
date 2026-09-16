"""CriticEmbedding: rotation-stable, O1/O2-stable embedding for the value network.

All outputs are SO(3)-invariant scalars (distances, norms, dot products); O1/O2 identity
preserved via LAMMPS ordering (no world-axis canonicalization); PBC via minimum-image
convention. No fixed axes, no [0,0,1], no z-coordinates, no surface normals -- all
orientation features derive from the atomic environment.

Gradient flow: lives inside PolicyNetwork (not EmbeddingExtractor). The buffer stores raw
inputs (o1_embedding, o2_embedding, geometry); during PPO update
critic_embedder(o1, o2, geom) -> value_mlp(feat) is recomputed. Pure math here; value_mlp
handles the learning.
"""

import torch
import torch.nn as nn
import numpy as np
from e3nn.o3 import Irreps
from typing import Tuple
import logging
_LOG = logging.getLogger(__name__)


# Strict O1/O2 index picking (no world-axis canonicalization).

def pick_o2_indices_strict(
    atoms,
    assume_last_two: bool = True,
    dmin: float = 0.9,
    dmax: float = 2.5
) -> Tuple[int, int]:
    """Pick O1/O2 indices with strict validation.

    Identity is the LAMMPS ordering (last two atoms); no canonicalization by z or any world
    axis (that would break SO(3)). LAMMPS ordering is deterministic and rotation-preserved,
    so this is SO(3)-safe. Does not mutate the input atoms object.

    Args:
        atoms: ASE Atoms object
        assume_last_two: If True, expect last two atoms to be O2
        dmin: Minimum valid O-O distance (Angstrom)
        dmax: Maximum valid O-O distance (Angstrom), 2.5 allows stretched states

    Returns:
        (o1_idx, o2_idx): Indices as given by LAMMPS ordering

    Raises:
        RuntimeError: If validation fails (fail-fast, no fallback)
    """
    Z = np.asarray(atoms.get_atomic_numbers())

    if assume_last_two:
        o1_idx = len(Z) - 2
        o2_idx = len(Z) - 1
        if not (Z[o1_idx] == 8 and Z[o2_idx] == 8):
            raise RuntimeError(
                f"FAIL: last two atoms are not O atoms. "
                f"Got Z[{o1_idx}]={Z[o1_idx]}, Z[{o2_idx}]={Z[o2_idx]}. "
                f"Expected both to be 8 (oxygen)."
            )
    else:
        o_idx = np.where(Z == 8)[0]
        if len(o_idx) < 2:
            raise RuntimeError(f"FAIL: fewer than 2 oxygen atoms. Found {len(o_idx)}.")
        o1_idx, o2_idx = int(o_idx[-2]), int(o_idx[-1])

    # O-O distance via minimum image convention; copy atoms if PBC not set (no mutation).
    if any(atoms.pbc):
        d = atoms.get_distance(o1_idx, o2_idx, mic=True)
    else:
        atoms_tmp = atoms.copy()
        atoms_tmp.set_pbc((True, True, False))  # x,y periodic, z non-periodic
        d = atoms_tmp.get_distance(o1_idx, o2_idx, mic=True)

    if not (dmin <= d <= dmax):
        raise RuntimeError(
            f"FAIL: O-O distance {d:.3f} Angstrom not in [{dmin}, {dmax}] Angstrom. "
            f"O2 indexing likely wrong or molecule already dissociated."
        )

    return o1_idx, o2_idx


# Irreps-aware invariant extraction.

def irreps_block_norms(x: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Per-copy norms for each (mul, l) block; converts equivariant features to invariant scalars.

    Args:
        x: [B, irreps.dim] equivariant features
        irreps: Irreps specification (e.g., "128x0e+64x1e+32x2e")

    Returns:
        [B, sum(mul)] norms per copy
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [B, D], got shape {list(x.shape)}")

    B = x.shape[0]
    outs = []
    start = 0

    for mul, ir in irreps:
        dim = mul * ir.dim  # ir.dim = 2*l + 1
        block = x[:, start:start + dim].reshape(B, mul, ir.dim)
        norms = torch.sqrt((block ** 2).sum(dim=-1) + 1e-12)  # [B, mul]
        outs.append(norms)
        start += dim

    if start != irreps.dim:
        raise RuntimeError(
            f"FAIL: irreps slicing mismatch. Processed {start}, expected {irreps.dim}"
        )

    return torch.cat(outs, dim=-1)


def irreps_block_dot(x: torch.Tensor, y: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Per-copy dot products between x and y for each (mul, l) block.

    Rotation-invariant: dot(Rx, Ry) = dot(x, y).

    Args:
        x, y: [B, irreps.dim] equivariant features
        irreps: Irreps specification

    Returns:
        [B, sum(mul)] dot products per copy
    """
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(
            f"x, y must have same shape [B, D], got {list(x.shape)} vs {list(y.shape)}"
        )

    B = x.shape[0]
    outs = []
    start = 0

    for mul, ir in irreps:
        dim = mul * ir.dim
        bx = x[:, start:start + dim].reshape(B, mul, ir.dim)
        by = y[:, start:start + dim].reshape(B, mul, ir.dim)
        dots = (bx * by).sum(dim=-1)  # [B, mul]
        outs.append(dots)
        start += dim

    if start != irreps.dim:
        raise RuntimeError(
            f"FAIL: irreps slicing mismatch. Processed {start}, expected {irreps.dim}"
        )

    return torch.cat(outs, dim=-1)


def irreps_select_0e(x: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Extract only the 0e (scalar) components from equivariant features.

    Args:
        x: [B, irreps.dim] equivariant features
        irreps: Irreps specification

    Returns:
        [B, total_0e_mul] scalar features only
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [B, D], got shape {list(x.shape)}")

    B = x.shape[0]
    outs = []
    start = 0

    for mul, ir in irreps:
        dim = mul * ir.dim
        block = x[:, start:start + dim]

        if ir.l == 0:
            outs.append(block.reshape(B, mul))

        start += dim

    if not outs:
        raise RuntimeError("FAIL: no 0e blocks found in irreps")

    return torch.cat(outs, dim=-1)


# PBC utilities (minimum image convention).

def _mic_vector(delta: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
    """Apply minimum image convention to a displacement vector.

    Args:
        delta: [3] displacement vector (r_j - r_i)
        cell: [3, 3] unit cell (row vectors: cell[0] = a, cell[1] = b, cell[2] = c)
        pbc: [3] bool tensor, periodic boundary flags per axis

    Returns:
        [3] mic-corrected displacement vector
    """
    cell_inv = torch.linalg.inv(cell)  # [3, 3]
    frac = delta @ cell_inv  # [3]

    # Wrap periodic dimensions to [-0.5, 0.5).
    wrap_mask = pbc.float()  # 1.0 periodic, 0.0 non-periodic
    frac = frac - torch.round(frac) * wrap_mask

    return frac @ cell


def _mic_distances(
    origin: torch.Tensor,
    targets: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PBC-safe distances and displacement vectors from origin to targets.

    Args:
        origin: [3] reference position
        targets: [M, 3] target positions
        cell: [3, 3] unit cell (row vectors)
        pbc: [3] bool tensor

    Returns:
        distances: [M] minimum-image distances
        deltas: [M, 3] minimum-image displacement vectors (targets - origin, corrected)
    """
    deltas = targets - origin.unsqueeze(0)  # [M, 3]

    cell_inv = torch.linalg.inv(cell)  # [3, 3]
    frac = deltas @ cell_inv  # [M, 3]

    wrap_mask = pbc.float().unsqueeze(0)  # [1, 3]
    frac = frac - torch.round(frac) * wrap_mask

    deltas_mic = frac @ cell  # [M, 3]
    distances = torch.sqrt((deltas_mic ** 2).sum(dim=-1) + 1e-12)  # [M]

    return distances, deltas_mic


# SO(3)-safe, PBC-safe geometry features.

def compute_geometry_features(
    pos: torch.Tensor,
    o1_idx: int,
    o2_idx: int,
    si_mask: torch.Tensor,
    non_o2_mask: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    k_nearest_si: int = 1,
    include_signed_align: bool = True
) -> torch.Tensor:
    """Compute SO(3)-invariant geometry scalars with PBC-safe (minimum-image) distances.

    No fixed axes, no [0,0,1], no z-coordinates, no surface normals; all orientation
    features use environment-derived directions.

    Features (5 or 6 depending on include_signed_align):
        - d_OO: O-O bond length (mic)
        - dmin_O_Si: Min distance from either O to any Si (mic)
        - dmin_COM_Si: Min distance from COM to any Si (mic)
        - dmin_O_any: Min distance from either O to any non-O2 atom (mic)
        - align2: (u * v_hat)^2 where v = COM -> nearest Si (SO(3)-invariant)
        - align_signed (optional): u * v_hat (SO(3)-invariant, direction-sensitive)

    Args:
        pos: [N, 3] atomic positions (all on same device)
        o1_idx, o2_idx: O atom indices (LAMMPS identity)
        si_mask: [N] bool tensor, True for Si atoms
        non_o2_mask: [N] bool tensor, True for all atoms EXCEPT the O2 molecule
        cell: [3, 3] unit cell tensor (row vectors)
        pbc: [3] bool tensor, periodic boundary flags per axis
        k_nearest_si: Number of nearest Si atoms for direction averaging
        include_signed_align: Whether to include signed alignment feature

    Returns:
        [1, F] geometry features (F=5 or 6)
    """
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be [N, 3], got {list(pos.shape)}")
    if pos.device != si_mask.device:
        raise RuntimeError(
            f"FAIL: device mismatch: pos on {pos.device}, si_mask on {si_mask.device}"
        )
    if pos.device != non_o2_mask.device:
        raise RuntimeError(
            f"FAIL: device mismatch: pos on {pos.device}, non_o2_mask on {non_o2_mask.device}"
        )
    if cell.dim() != 2 or cell.shape != (3, 3):
        raise ValueError(f"cell must be [3, 3], got {list(cell.shape)}")
    if pbc.shape != (3,):
        raise ValueError(f"pbc must be [3], got {list(pbc.shape)}")

    r1 = pos[o1_idx]
    r2 = pos[o2_idx]

    # PBC-safe O-O bond vector.
    bond = _mic_vector(r2 - r1, cell, pbc)
    d_OO = torch.sqrt((bond * bond).sum() + 1e-12)
    u = bond / d_OO  # directed O-O axis (rotates with structure)

    com = r1 + 0.5 * bond  # mic-corrected COM

    # Distances to Si (PBC-safe).
    si_pos = pos[si_mask]  # [N_Si, 3]
    if si_pos.shape[0] < 1:
        raise RuntimeError("FAIL: no Si atoms found for geometry features")

    d1_si, _ = _mic_distances(r1, si_pos, cell, pbc)  # [N_Si]
    d2_si, _ = _mic_distances(r2, si_pos, cell, pbc)
    dmin_O_Si = torch.minimum(d1_si.min(), d2_si.min())

    dcom_si, delta_com_si = _mic_distances(com, si_pos, cell, pbc)  # [N_Si], [N_Si, 3]
    dmin_COM_Si = dcom_si.min()

    # Distance to any non-O2 atom (crash detector, PBC-safe).
    other_pos = pos[non_o2_mask]  # [N_other, 3]
    if other_pos.shape[0] < 1:
        raise RuntimeError("FAIL: no non-O2 atoms found")

    d1_any, _ = _mic_distances(r1, other_pos, cell, pbc)
    d2_any, _ = _mic_distances(r2, other_pos, cell, pbc)
    dmin_O_any = torch.minimum(d1_any.min(), d2_any.min())

    # SO(3)-safe alignment: O-O axis vs direction to nearest Si.
    # v = COM -> nearest Si (environment-derived, rotates with structure).
    if k_nearest_si == 1:
        nearest_si_idx = dcom_si.argmin()
        v = delta_com_si[nearest_si_idx]  # mic-corrected COM -> nearest Si
    else:
        _, topk_idx = torch.topk(dcom_si, k=min(k_nearest_si, si_pos.shape[0]), largest=False)
        topk_deltas = delta_com_si[topk_idx]  # [k, 3] mic-corrected vectors
        topk_d = dcom_si[topk_idx].clamp(min=1e-6)  # [k]
        weights = 1.0 / topk_d
        weights = weights / weights.sum()
        v = (weights.unsqueeze(-1) * topk_deltas).sum(dim=0)  # [3]

    v_norm = torch.sqrt((v * v).sum() + 1e-12)
    v_hat = v / v_norm

    # (u * v_hat) is SO(3)-invariant: under rotation R, Ru * Rv_hat = u * v_hat
    dot_uv = (u * v_hat).sum()
    align2 = dot_uv * dot_uv

    feats = [d_OO, dmin_O_Si, dmin_COM_Si, dmin_O_any, align2]

    if include_signed_align:
        feats.append(dot_uv)

    return torch.stack(feats, dim=0).view(1, -1)

def augment_geometry_with_context(
    geometry: torch.Tensor,
    step: int,
    max_steps: int,
    env_multipole: torch.Tensor = None,
) -> torch.Tensor:
    """Append time horizon and multipole invariants to geometry features.

    Fills two critic-input blind spots:
      1) Time: V(s) should be V(s,t) in finite-horizon tasks; same geometry at step 2 vs 22
         has very different expected return.
      2) Long-range signal: when O2 is >5 Angstrom from the surface the embedding is near-zero;
         multipole norms provide smooth invariant context.

    Args:
        geometry: [B, 6] base geometry from compute_geometry_features
        step: current step in episode (0-indexed)
        max_steps: maximum steps per episode
        env_multipole: [B, 9] multipole coefficients (1x0e + 1x1e + 1x2e), or None

    Returns:
        [B, 10] augmented geometry
    """
    B = geometry.shape[0]
    device = geometry.device
    dtype = geometry.dtype

    step_frac = torch.full((B, 1), step / max_steps, device=device, dtype=dtype)

    if env_multipole is not None:
        mp = env_multipole.to(device=device, dtype=dtype)
        if mp.dim() == 1:
            mp = mp.unsqueeze(0)
        mp_l0 = mp[:, 0:1]                                  # [B, 1]
        mp_l1_norm = mp[:, 1:4].norm(dim=-1, keepdim=True)  # [B, 1]
        mp_l2_norm = mp[:, 4:9].norm(dim=-1, keepdim=True)  # [B, 1]
    else:
        mp_l0 = torch.zeros(B, 1, device=device, dtype=dtype)
        mp_l1_norm = torch.zeros(B, 1, device=device, dtype=dtype)
        mp_l2_norm = torch.zeros(B, 1, device=device, dtype=dtype)

    return torch.cat([geometry, step_frac, mp_l0, mp_l1_norm, mp_l2_norm], dim=-1)

class CriticEmbedding(nn.Module):
    """Rotation-stable critic features from equivariant node embeddings + SO(3)-safe geometry.

    Lives inside PolicyNetwork so PPO-update gradients reach value_mlp. The buffer stores
    (o1_embedding, o2_embedding, geometry) and recomputes here during evaluate_actions()
    for consistency. Output = concat([geometry, sym_l0]) where sym_l0 is the l=0 block of
    SYM = 0.5*(e1+e2) (swap-invariant, rotation-invariant).
    """

    def __init__(
        self,
        irreps_atom: Irreps,
        geom_dim: int = 10,
        **unused_kwargs,  # swallow unused kwargs (use_dots, emb_hidden_dim, geo_hidden_dim)
    ):
        """Critic input composer.

        critic input = [geom(10), sym_l0(128)] = 138 scalars, where sym_l0 = l=0 block of
        SYM = 0.5*(e1+e2) (swap-invariant, rotation-invariant because l=0).

        No Linear, no LayerNorm: raw feature composition. Cross-sample per-feature
        normalization is handled externally by CriticRunningNormalizer. LayerNorm is avoided
        because it standardizes *within* each sample, stripping the absolute cross-sample
        scale a value function needs to distinguish states.
        """
        super().__init__()
        self.irreps_atom = Irreps(irreps_atom)
        self.geom_dim = int(geom_dim)
        self._mul_0e = sum(mul for mul, ir in self.irreps_atom if ir.l == 0)
        self.out_dim = self._mul_0e + self.geom_dim  # 128 + 10 = 138

        if unused_kwargs:
            _LOG.info(f"[CriticEmbedding] Ignoring unused kwargs: {list(unused_kwargs.keys())}")
        _LOG.info(f"[CriticEmbedding] Spec-faithful: geom({self.geom_dim}) + "
              f"sym_l0({self._mul_0e}) = {self.out_dim} dims "
              f"(no LayerNorm, no Linear -- external running-stats normalizer instead)")

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        geom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw critic features [B, 138] = concat([geom, sym_l0]).

        No normalization here; CriticRunningNormalizer handles it in PolicyNetwork._compute_value.
        """
        if e1.dim() != 2 or e2.dim() != 2:
            raise ValueError(f"e1, e2 must be [B, D], got {e1.dim()}D and {e2.dim()}D")
        if e1.shape != e2.shape:
            raise ValueError(f"e1, e2 shape mismatch: {list(e1.shape)} vs {list(e2.shape)}")
        if e1.shape[1] != self.irreps_atom.dim:
            raise RuntimeError(
                f"FAIL: embedding dim {e1.shape[1]} != irreps_atom.dim {self.irreps_atom.dim}"
            )
        if geom.dim() != 2 or geom.shape[1] != self.geom_dim:
            raise RuntimeError(
                f"FAIL: geom shape {list(geom.shape)} != expected [B, {self.geom_dim}]"
            )

        # Swap-invariant l=0 block of SYM = 0.5 * (e1 + e2) -- rotation-invariant.
        e1_0e = irreps_select_0e(e1, self.irreps_atom)   # [B, 128]
        e2_0e = irreps_select_0e(e2, self.irreps_atom)   # [B, 128]
        sym_l0 = 0.5 * (e1_0e + e2_0e)                   # [B, 128]

        # Spec order: [geometry, sym_l0]  ->  [B, 138]
        return torch.cat([geom, sym_l0], dim=1)

    def __repr__(self):
        return (
            f"CriticEmbedding(\n"
            f"  irreps_atom={self.irreps_atom},\n"
            f"  geom_dim={self.geom_dim}, use_dots={self.use_dots},\n"
            f"  dual_pathway: emb {self._emb_dim}->emb_branch, geo {self.geom_dim}->geo_branch,\n"
            f"  out_dim={self.out_dim}\n"
            f"  [0e: {self._mul_0e}*2, norms: {self._mul_sum}*2"
            f"{f', dots: {self._mul_sum}' if self.use_dots else ''}"
            f", geom: {self.geom_dim}]\n"
            f")"
        )

# Masks from atoms.

def get_si_mask(atoms, device: torch.device) -> torch.Tensor:
    """Get boolean mask for Si atoms."""
    Z = np.asarray(atoms.get_atomic_numbers())
    return torch.tensor(Z == 14, dtype=torch.bool, device=device)


def get_non_o2_mask(atoms, o1_idx: int, o2_idx: int, device: torch.device) -> torch.Tensor:
    """Get boolean mask for all atoms EXCEPT the O2 molecule."""
    N = len(atoms)
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[o1_idx] = False
    mask[o2_idx] = False
    return mask

# Per-feature running-stats normalizer for critic input.

class CriticRunningNormalizer(nn.Module):
    """Per-feature (across-sample) running mean/variance normalizer for stable value targets.

    Mirrors the actor's StateNormalizer: forward() normalizes with current buffers; stats
    update only via explicit staged updates from the rollout path, held constant across each
    rollout+update window so rollout-time and PPO-update-time V(s) match (H2 consistency).
    freeze() permanently stops updates (post-warmup).

    Unlike LayerNorm (which standardizes within a sample and strips cross-sample scale,
    collapsing the critic toward the batch mean), this standardizes per feature across
    samples, preserving the cross-sample scale that distinguishes state values.
    """

    def __init__(self, feature_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)

        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var',  torch.ones(feature_dim))
        self.register_buffer('count',        torch.tensor(1e-4))

        self._frozen = False
        self._temp_frozen = False

        # Pending batches from a rollout window, applied atomically by flush_pending_updates()
        # after the PPO update. Transient, not part of model state.
        self._pending_batches: list = []

    @torch.no_grad()
    def stage_update(self, x: torch.Tensor):
        """Stage a batch for deferred application at the next flush.

        Called from the rollout path. Does not modify running stats here; they stay constant
        across the whole rollout+update window for H2 consistency.
        """
        if self._frozen or self._temp_frozen:
            return
        if x.dim() != 2 or x.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"CriticRunningNormalizer.stage_update expected [B, {self.feature_dim}], "
                f"got {list(x.shape)}"
            )
        if x.shape[0] == 0:
            return
        self._pending_batches.append(x.detach().clone())

    @torch.no_grad()
    def flush_pending_updates(self):
        """Apply all staged batches via Welford and clear the pending buffer.

        Called by the trainer after each PPO update; the next rollout window sees the
        updated statistics, again held constant for its duration.
        """
        if self._frozen:
            # Permanent freeze: discard pending batches.
            self._pending_batches.clear()
            return
        if not self._pending_batches:
            return

        for batch in self._pending_batches:
            bc = float(batch.shape[0])
            if bc == 0:
                continue
            bm = batch.mean(dim=0)
            bv = batch.var(dim=0, unbiased=False)
            delta = bm - self.running_mean
            new_count = self.count + bc
            self.running_mean = self.running_mean + delta * (bc / new_count)
            m_a = self.running_var * self.count
            m_b = bv * bc
            M2  = m_a + m_b + delta.pow(2) * (self.count * bc / new_count)
            self.running_var = M2 / new_count
            self.count = new_count

        n_applied = len(self._pending_batches)
        self._pending_batches.clear()
        _LOG.debug(f"[CriticRunningNormalizer] Flushed {n_applied} pending batches "
              f"(count now {float(self.count):.0f}, "
              f"mean_abs_max={self.running_mean.abs().max().item():.3f})")

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Alias for stage_update; prefer stage_update() in new code."""
        self.stage_update(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with current buffers; never updates.

        Updates happen only via staged updates from the rollout path, guaranteeing
        rollout-time and PPO-update-time V(s) share statistics (H2 consistency).
        """
        std = torch.sqrt(self.running_var + self.epsilon)
        return (x - self.running_mean) / std

    def freeze(self):
        self._frozen = True
        _LOG.info(f"[CriticRunningNormalizer] Frozen at count={float(self.count):.0f} | "
              f"mean_abs_max={self.running_mean.abs().max().item():.3f} | "
              f"std_min={self.running_var.sqrt().min().item():.3e} | "
              f"std_max={self.running_var.sqrt().max().item():.3e}")

    def set_frozen_temporarily(self, flag: bool):
        """Match StateNormalizer API so PPO-update temp-freeze plumbing works."""
        self._temp_frozen = bool(flag)

    def is_frozen(self) -> bool:
        return self._frozen

    def get_statistics(self):
        return {
            'mean':  self.running_mean.detach().cpu().numpy().copy(),
            'var':   self.running_var.detach().cpu().numpy().copy(),
            'std':   (self.running_var.sqrt()).detach().cpu().numpy().copy(),
            'count': float(self.count.item()),
            'frozen': self._frozen,
        }
