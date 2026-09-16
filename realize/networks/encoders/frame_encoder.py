import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from e3nn import o3
from e3nn.o3 import Irreps, Linear
from realize.networks.irrep_sequential import IrrepSequential, IrrepActivation

from e3nn.o3 import FullyConnectedTensorProduct
import logging
_LOG = logging.getLogger(__name__)


class FrameAlignmentEncoder(nn.Module):
    """Frame alignment and directional feature extraction.

    Input:  480-dim equivariant embedding (128x0e+64x1e+32x2e) from SimpleO2Combiner.
    Output: 60-dim compressed features (16x0e+8x1e+4x2e).

    Parity: SevenNet emits even parity for l=1, and e3nn Linear only connects matching
    parities (1e->1e), so all l=1/l=2 irreps use even parity to match the backbone.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Raw SevenNet output; SimpleO2Combiner preserves the backbone irreps exactly.
        self.embedding_irreps = "128x0e+64x1e+32x2e"
        self.irreps_in = Irreps(self.embedding_irreps)

        _LOG.info(f"[FrameEncoder] Input Irreps: {self.embedding_irreps} (dim={self.irreps_in.dim})")

        self.debug_mode = getattr(config, 'debug_frame_encoder', False)

        # Compresses 480-dim equivariant features to 60-dim; all parities 'e' (SevenNet convention).
        self.frame_encoder = IrrepSequential(
            Linear(self.irreps_in, Irreps("64x0e+32x1e+16x2e")),
            IrrepActivation(Irreps("64x0e+32x1e+16x2e"), acts=[F.silu, None, None]),
            Linear(Irreps("64x0e+32x1e+16x2e"), Irreps("16x0e+8x1e+4x2e")),
        )

        self.output_irreps = Irreps("16x0e+8x1e+4x2e")

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        def safe_init(weight, gain):
            if weight.dim() < 2:
                nn.init.normal_(weight, mean=0.0, std=gain)
            else:
                nn.init.xavier_uniform_(weight, gain=gain)

        with torch.no_grad():
            for module in self.frame_encoder.modules():
                if isinstance(module, Linear) and module.weight is not None:
                    safe_init(module.weight, gain=1.5)

    def forward(self, embeddings: torch.Tensor, direction: torch.Tensor,
                o2_positions: torch.Tensor = None):
        if direction is None or o2_positions is None:
            raise ValueError("Inputs missing")
        R_feat_to_world = self.compute_feature_to_world_rotation(direction, o2_positions)
        features = self.extract_frame_aligned_features_robust(embeddings, R_feat_to_world, direction)
        return {'frame_aligned_features': features, 'R_feat_to_world': R_feat_to_world}

    def extract_unaligned_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        encoded = self.frame_encoder(embeddings)     # [B, 480] -> [B, 60]
        return encoded

    def extract_frame_aligned_features_robust(self, embeddings: torch.Tensor,
                                              R_feat_to_world: torch.Tensor,
                                              translation_direction: torch.Tensor) -> torch.Tensor:
        """Rotate world features into the aligned frame via Wigner D matrices.

        l=0 scalars unchanged; l=1 by Wigner D^1 (3x3); l=2 by Wigner D^2 (5x5).
        Returns output_irreps 16x0e + 8x1e + 4x2e = 60 dims.
        """
        batch_size = embeddings.shape[0]

        encoded_base = self.frame_encoder(embeddings)  # [B, 480] -> [B, 60]

        # Irreps layout: 16x0e dims 0-16; 8x1e dims 16-40 (8x3); 4x2e dims 40-60 (4x5).
        encoded_l0 = encoded_base[:, :16]      # 16x0e scalars (invariant)
        encoded_l1 = encoded_base[:, 16:40]    # 8x1e  vectors
        encoded_l2 = encoded_base[:, 40:60]    # 4x2e  rank-2 tensors

        # World -> aligned uses R^T.
        R_world_to_aligned = R_feat_to_world.transpose(1, 2)  # [B, 3, 3]

        D1 = self._rotation_to_wigner_D(R_world_to_aligned, l=1)  # [B, 3, 3]
        D2 = self._rotation_to_wigner_D(R_world_to_aligned, l=2)  # [B, 5, 5]

        # Rotate l=1 (8 vectors).
        l1_vecs = encoded_l1.reshape(batch_size, 8, 3)  # [B, 8, 3]
        l1_rotated = torch.bmm(l1_vecs, D1.transpose(1, 2))  # [B, 8, 3]
        l1_final = l1_rotated.reshape(batch_size, -1)  # [B, 24]

        # Rotate l=2 (4 tensors).
        l2_tensors = encoded_l2.reshape(batch_size, 4, 5)  # [B, 4, 5]
        l2_rotated = torch.bmm(l2_tensors, D2.transpose(1, 2))  # [B, 4, 5]
        l2_final = l2_rotated.reshape(batch_size, -1)  # [B, 20]

        # 16 + 24 + 20 = 60 dims, matching output_irreps "16x0e+8x1e+4x2e".
        return torch.cat([encoded_l0, l1_final, l2_final], dim=-1)

    def _rotation_to_wigner_D(self, R: torch.Tensor, l: int) -> torch.Tensor:
        """Convert 3x3 rotation matrices to Wigner D matrices for given l.

        Uses e3nn's wigner_D with Euler angles from R. R_i.detach().cpu() cuts the gradient
        through the Wigner D matrices; acceptable because the translation direction is
        categorically sampled (no reparam gradient) and the magnitude/axis heads learn from
        feature values, not the rotation. Would become a bug under continuous direction sampling.

        Args:
            R: [B, 3, 3] rotation matrices
            l: angular momentum quantum number (1, 2, or 3)

        Returns:
            D: [B, 2l+1, 2l+1] Wigner D matrices
        """
        batch_size = R.shape[0]
        device = R.device
        dtype = R.dtype

        D_matrices = []
        for i in range(batch_size):
            R_i = R[i]  # [3, 3]

            # matrix_to_angles / wigner_D run on CPU (device issues otherwise).
            R_i_cpu = R_i.detach().cpu()
            alpha, beta, gamma = o3.matrix_to_angles(R_i_cpu)
            D_i = o3.wigner_D(l, alpha, beta, gamma)  # [2l+1, 2l+1]
            D_matrices.append(D_i)

        D = torch.stack(D_matrices, dim=0).to(device=device, dtype=dtype)

        return D

    def extract_frame_aligned_features(self, embeddings, translation_direction, o2_positions):
        R = self.compute_feature_to_world_rotation(translation_direction, o2_positions)
        return self.extract_frame_aligned_features_robust(embeddings, R, translation_direction)

    def compute_feature_to_world_rotation(self, v_hat_world, o2_positions, embeddings=None):
        # Continuous fixed-reference Gram-Schmidt frame: z-axis is v_hat_world; the in-plane
        # (x,y) comes from Gram-Schmidt on world references e_x, e_y with sigmoid blending:
        #   a = e_x - (e_x . z) z,  ||a||^2 = 1 - (e_x . z)^2
        #   b = e_y - (e_y . z) z,  ||b||^2 = 1 - (e_y . z)^2
        # Since (e_x.z)^2 + (e_y.z)^2 + (e_z.z)^2 = 1, ||a||^2 + ||b||^2 = 1 + (e_z.z)^2 >= 1,
        # so max >= 1/2: one of a,b is always well-defined. Blend
        #   w = sigmoid(k*(||a||^2 - ||b||^2)),  x_unnorm = w*a + (1-w)*b
        # is bounded away from zero for every z in S^2, so normalize is safe everywhere.
        # o2_positions is not used: the x-axis is not bond-aligned; O-O orientation reaches
        # downstream heads via o2_embedding_asymmetric, not the frame.
        import os as _os_frame

        batch_size = v_hat_world.shape[0]
        device = v_hat_world.device
        dtype = v_hat_world.dtype

        z_feat = F.normalize(v_hat_world, dim=-1, eps=1e-8)

        # World references e_x, e_y.
        e_x = torch.zeros_like(z_feat); e_x[:, 0] = 1.0
        e_y = torch.zeros_like(z_feat); e_y[:, 1] = 1.0

        # Gram-Schmidt against z.
        dot_xz = (e_x * z_feat).sum(dim=-1, keepdim=True)  # [B,1]
        dot_yz = (e_y * z_feat).sum(dim=-1, keepdim=True)  # [B,1]
        a = e_x - dot_xz * z_feat  # [B,3]
        b = e_y - dot_yz * z_feat  # [B,3]

        a_norm_sq = (a * a).sum(dim=-1, keepdim=True)  # [B,1]
        b_norm_sq = (b * b).sum(dim=-1, keepdim=True)  # [B,1]

        # Blend toward the larger-norm reference; k=20 is sharp enough to prefer it while
        # staying smooth and differentiable.
        blend_sharpness = 20.0
        w = torch.sigmoid(blend_sharpness * (a_norm_sq - b_norm_sq))  # [B,1]
        x_unnorm = w * a + (1.0 - w) * b  # [B,3]

        # x_unnorm is never zero (see math above).
        x_feat = F.normalize(x_unnorm, dim=-1, eps=1e-8)

        # Complete the orthonormal frame: y = z x x, then re-orthogonalize x = y x z.
        y_feat = torch.cross(z_feat, x_feat, dim=-1)
        y_feat = F.normalize(y_feat, dim=-1, eps=1e-8)
        x_feat = torch.cross(y_feat, z_feat, dim=-1)

        R_feat_to_world = torch.stack([x_feat, y_feat, z_feat], dim=-1)

        # Diagnostic only (no effect on R): logs how often the bond-aligned x-axis would
        # be near-singular, plus blend-weight stats.
        if _os_frame.environ.get('AXIS_FIX_DEBUG', '1') == '1':
            if not hasattr(self, '_frame_fix_call_count'):
                self._frame_fix_call_count = 0
                self._frame_fix_log_every = int(_os_frame.environ.get('AXIS_FIX_DEBUG_EVERY', '200'))
                self._frame_fix_singular_count = 0
                self._frame_fix_total_count = 0
            self._frame_fix_call_count += 1

            with torch.no_grad():
                # Bond-aligned x-axis norm: how often it would be near-singular.
                o2_vec = o2_positions[:, 1] - o2_positions[:, 0]
                dot_bond = (o2_vec * z_feat).sum(dim=-1, keepdim=True)
                x_bond = o2_vec - dot_bond * z_feat
                x_bond_norm = torch.norm(x_bond, dim=-1)  # [B]
                mask_singular_old = (x_bond_norm < 1e-2)
                n_singular = int(mask_singular_old.sum().item())
                n_total = int(x_bond_norm.numel())
                self._frame_fix_singular_count += n_singular
                self._frame_fix_total_count += n_total

                if self._frame_fix_call_count % self._frame_fix_log_every == 0:
                    cum_rate = 100.0 * self._frame_fix_singular_count / max(1, self._frame_fix_total_count)
                    w_mean = w.mean().item(); w_std = w.std().item() if w.numel() > 1 else 0.0
                    a_norm_sq_min = a_norm_sq.min().item()
                    b_norm_sq_min = b_norm_sq.min().item()
                    _LOG.debug(f"[AXIS_FIX FRAME call={self._frame_fix_call_count}] "
                          f"OLD_would_singular: this_batch={n_singular}/{n_total} "
                          f"cumulative_rate={cum_rate:.3f}% "
                          f"| x_bond_norm: min={x_bond_norm.min().item():.4f} "
                          f"max={x_bond_norm.max().item():.4f} mean={x_bond_norm.mean().item():.4f} "
                          f"| blend_w: mean={w_mean:.3f} std={w_std:.3f} "
                          f"| a_norm_sq_min={a_norm_sq_min:.4f} b_norm_sq_min={b_norm_sq_min:.4f}")

        return R_feat_to_world

    def get_output_irreps(self):
        return self.output_irreps
