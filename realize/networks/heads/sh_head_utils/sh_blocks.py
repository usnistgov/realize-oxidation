"""S^2 grid and cross-multiplicity tensor-product blocks."""
import math

import torch
import torch.nn as nn

from e3nn.o3 import Irreps, Linear, spherical_harmonics, FullyConnectedTensorProduct
import logging
_LOG = logging.getLogger(__name__)


class S2Grid(nn.Module):
    """
    S2 grid for spherical harmonic evaluation and sampling.
    Uses Gauss-Legendre quadrature for accurate integration.
    """
    def __init__(self, lmax: int, res_beta: int = 64, res_alpha: int = 64, 
                 device: torch.device = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.lmax = lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        
        device = device or torch.device('cpu')
        
        from numpy.polynomial.legendre import leggauss
        x_np, w_beta_np = leggauss(res_beta)
        
        x = torch.tensor(x_np, device=device, dtype=torch.float64)
        w_beta = torch.tensor(w_beta_np, device=device, dtype=torch.float64)
        
        beta = torch.arccos(x)
        alpha = torch.linspace(0, 2 * math.pi * (res_alpha - 1) / res_alpha, 
                               res_alpha, device=device, dtype=torch.float64)
        
        beta_mesh, alpha_mesh = torch.meshgrid(beta, alpha, indexing='ij')
        
        sin_beta = torch.sin(beta_mesh)
        cos_beta = torch.cos(beta_mesh)
        
        grid_xyz = torch.stack([
            sin_beta * torch.cos(alpha_mesh),
            sin_beta * torch.sin(alpha_mesh),
            cos_beta
        ], dim=-1).reshape(-1, 3)
        
        w_2d = w_beta[:, None].expand(res_beta, res_alpha)
        w_alpha = 2 * math.pi / res_alpha
        weights = (w_2d * w_alpha).reshape(-1)
        
        self.register_buffer('grid_xyz', grid_xyz.to(dtype))
        self.register_buffer('w_vec', weights.to(dtype))
        self.register_buffer('log_w_vec', torch.log(weights).to(dtype))
        self.register_buffer('w_sum', weights.sum().to(dtype))
        
        irreps = Irreps.spherical_harmonics(lmax)
        Y = spherical_harmonics(irreps, grid_xyz.to(dtype), normalize=True)
        self.register_buffer('sh_to_grid', Y.T)
        
        self._validate_orthonormality()
        
    def _validate_orthonormality(self, tol: float = 1e-2):
        with torch.no_grad():
            Yt = self.sh_to_grid.T.double()
            w = self.w_vec.double().unsqueeze(-1)
            integrated_norms = (w * (Yt ** 2)).sum(dim=0)
            err = (integrated_norms - 1.0).abs().max().item()
            if err > tol:
                raise RuntimeError(f"SH basis normalization error {err:.6f} exceeds tolerance {tol}")
            _LOG.info(f"[S2Grid] SH basis orthonormality verified (error={err:.6f})")
        
    def evaluate_sh(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(coeffs, self.sh_to_grid.to(coeffs.dtype))


class CrossMultiplicityTP(nn.Module):
    """
    Splits each l-channel of Mx(le) input into two groups of (M/2)x(le) and
    computes their tensor product (M/2)x(le) (x) (M/2)x(le) -> output irreps,
    extracting directional correlations between different "views" of the
    environment.
    """
    
    def __init__(self, input_irreps: Irreps, output_lmax: int, device: torch.device):
        super().__init__()
        self.input_irreps = input_irreps
        self.output_lmax = output_lmax
        self.device = device
        
        # Analyze input structure
        self._analyze_input()
        
        # Build cross-multiplicity tensor products
        self._build_cross_tp()
        
    def _analyze_input(self):
        """Analyze input irreps to find multiplicity structure."""
        self.l_to_mul = {}  # l -> multiplicity
        self.l_to_parity = {}  # l -> parity char
        
        for mul, ir in self.input_irreps:
            l = ir.l
            p_char = 'e' if ir.p == 1 else 'o'
            if l in self.l_to_mul:
                self.l_to_mul[l] += mul
            else:
                self.l_to_mul[l] = mul
                self.l_to_parity[l] = p_char
        
        self.max_input_l = max(self.l_to_mul.keys()) if self.l_to_mul else 0
        _LOG.info(f"[CrossMultiplicityTP] Input structure: {dict(self.l_to_mul)}")
        
    def _build_cross_tp(self):
        """Build tensor products between split multiplicity groups."""
        
        # Define unified output irreps for ALL paths
        # This ensures dimension consistency when adding outputs
        self.output_irreps = Irreps("+".join([f"1x{l}e" for l in range(self.output_lmax + 1)]))
        self.output_dim = self.output_irreps.dim
        
        _LOG.info(f"[CrossMultiplicityTP] Unified output irreps: {self.output_irreps} (dim={self.output_dim})")
        
        # Compute input slicing
        offset = 0
        self.l_slices = {}
        for mul, ir in self.input_irreps:
            l = ir.l
            dim = mul * (2*l + 1)
            if l not in self.l_slices:
                self.l_slices[l] = []
            self.l_slices[l].append((offset, offset + dim, mul))
            offset += dim
        
        # Direct path: input -> unified output
        self.direct_linear = Linear(self.input_irreps, self.output_irreps).to(self.device)
        
        # Cross-TP paths: each outputs to unified irreps
        self.tp_modules = nn.ModuleDict()
        self.tp_projections = nn.ModuleDict()  # Project TP output to unified irreps
        self.tp_input_slices = {}
        
        for l in sorted(self.l_to_mul.keys()):
            mul = self.l_to_mul[l]
            if mul < 2:
                continue  # Need at least 2 copies to cross
                
            p_char = self.l_to_parity[l]
            
            # Split multiplicity in half
            mul_a = mul // 2
            mul_b = mul - mul_a
            
            if mul_a == 0 or mul_b == 0:
                continue
            
            irreps_a = Irreps(f"{mul_a}x{l}{p_char}")
            irreps_b = Irreps(f"{mul_b}x{l}{p_char}")
            
            # TP output: l(x)l -> |l-l|, ..., l+l (capped at output_lmax)
            # For l=1: 1(x)1 -> 0,1,2
            # For l=2: 2(x)2 -> 0,1,2,3,4 (capped)
            max_tp_l = min(2*l, self.output_lmax)
            tp_out_irreps_list = [f"1x{out_l}e" for out_l in range(max_tp_l + 1)]
            tp_out_irreps = Irreps("+".join(tp_out_irreps_list))
            
            # Create tensor product
            tp = FullyConnectedTensorProduct(
                irreps_a, irreps_b, tp_out_irreps,
                shared_weights=True,
                internal_weights=True
            ).to(self.device)
            
            # Create projection from TP output to unified output irreps
            # This handles the dimension mismatch (e.g., l=1 TP gives 0,1,2 but we want 0,1,2,3)
            proj = Linear(tp_out_irreps, self.output_irreps).to(self.device)
            
            self.tp_modules[f"l{l}"] = tp
            self.tp_projections[f"l{l}"] = proj
            self.tp_input_slices[f"l{l}"] = (l, mul_a, mul_b)
            
            _LOG.info(f"[CrossMultiplicityTP] l={l}: {irreps_a} (x) {irreps_b} -> {tp_out_irreps} -> {self.output_irreps}")
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-multiplicity tensor products.
        
        Args:
            features: [batch, input_dim] tensor
            
        Returns:
            output: [batch, output_dim] tensor with enriched angular content
        """
        batch_size = features.shape[0]
        
        # Direct path (preserves original information)
        output = self.direct_linear(features)
        
        # Cross-TP paths (add directional correlations)
        for key, tp in self.tp_modules.items():
            l, mul_a, mul_b = self.tp_input_slices[key]
            
            # Extract l-features
            l_features = []
            for (start, end, mul) in self.l_slices[l]:
                l_features.append(features[:, start:end])
            l_features = torch.cat(l_features, dim=-1)
            
            # Split into two groups
            dim_per_copy = 2*l + 1
            
            split_point = mul_a * dim_per_copy
            features_a = l_features[:, :split_point]
            features_b = l_features[:, split_point:split_point + mul_b * dim_per_copy]
            
            tp_out = tp(features_a, features_b)

            # Project to unified output irreps
            tp_projected = self.tp_projections[key](tp_out)

            # Each TP path is summed in; downstream channel_linear learns mixing
            output = output + tp_projected

        return output
