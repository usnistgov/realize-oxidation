import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import logging
_LOG = logging.getLogger(__name__)

class BoundedRQSFlow(nn.Module):
    """
    Bounded RQS flow for magnitude prediction on [r_min, r_max].

    base_mu is computed from r_min/r_max; spline constraints are kept soft and
    initialization near-identity to avoid bin collapse.
    """
    
    def __init__(self, 
                 scalar_dim: int,
                 r_min: float = 0.5,
                 r_max: float = 3.0,
                 num_bins: int = 8,
                 num_layers: int = 2,
                 hidden_dim: int = 64,
                 boundary_error: float = 1e-3,
                 min_bin_size: float = 1e-2,
                 device: torch.device = None):
        super().__init__()
        
        self.scalar_dim = scalar_dim
        self.r_min = r_min
        self.r_max = r_max
        self.range_size = r_max - r_min
        self.num_bins = num_bins
        self.num_layers = num_layers
        self.boundary_error = boundary_error
        self.min_bin_size = min_bin_size        
        self.device = device if device else torch.device('cpu')


        # Validate min_bin_size constraint
        if min_bin_size * num_bins >= 1.0:
            raise ValueError(
                f"min_bin_size * num_bins must be < 1.0, got {min_bin_size * num_bins:.4f}"
            )
        
        if min_bin_size * num_bins > 0.8:
            raise ValueError(
                f"min_bin_size * num_bins = {min_bin_size * num_bins:.2f} is too close to 1.0 (> 0.8). "
                f"Reduce min_bin_size or num_bins."
            )

        # Centered base distribution (no bias)
        self.base_kind = 'trunc_normal'
        # Bias toward larger steps initially - typical O2 approach distances
        # mu = 0.8 * range puts peak at 2.5 for [0.5, 3.0]
        self.base_mu = float(r_min + 0.52 * (r_max - r_min))  # = 1.8Angstrom

        # Moderate spread: allows exploration but not too diffuse
        # sigma = range/4 gives ~68% of samples within [1.125, 2.375] for [0.5, 3.0]
        self.base_sigma = float((r_max - r_min) / 4.0)

        _LOG.info(f"[RQS/INIT] Dynamic base: mu={self.base_mu:.3f}, sigma={self.base_sigma:.3f} "
              f"for range [{r_min}, {r_max}]")

        # DIAGNOSTIC: Verify base distribution
        with torch.no_grad():
            dummy_features = torch.zeros(1000, scalar_dim, device=self.device, dtype=torch.float32)
            base_samples = self._sample_base(dummy_features)
            base_mean = base_samples.mean().item()
            base_std = base_samples.std().item()
            
            _LOG.debug(f"[RQS/INIT] Base distribution check:")
            _LOG.debug(f"  Mean: {base_mean:.4f} (target: {self.base_mu:.4f})")
            _LOG.debug(f"  Std:  {base_std:.4f} (target: {self.base_sigma:.4f})")
            
            # Check for boundary clustering
            near_lower = (base_samples < r_min + 0.1 * self.range_size).float().mean().item()
            near_upper = (base_samples > r_max - 0.1 * self.range_size).float().mean().item()
            
            if near_upper > 0.2 or near_lower > 0.2:
                _LOG.warning(f"Base samples clustered: {100*near_lower:.1f}% near lower, "
                      f"{100*near_upper:.1f}% near upper")
            else:
                _LOG.debug(f"[OK] Base distribution well-centered")

        # Parameter networks for each flow layer
        self.param_nets = nn.ModuleList()
        for layer_idx in range(num_layers):
            param_net = nn.Sequential(
                nn.Linear(scalar_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3 * num_bins + 1)
            )
            
            # Better initialization
            with torch.no_grad():
                # Initialize hidden layers normally
                for layer in param_net[:-1]:
                    if hasattr(layer, 'weight'):
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if hasattr(layer, 'bias'):
                        nn.init.zeros_(layer.bias)
                
                # Final layer: ZERO weights to start exactly at identity
                # This ensures output is purely from bias regardless of input features
                final_layer = param_net[-1]
                # Small weights allow state-dependent output while staying near identity
                nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
                
                # Biases for TRUE identity RQS
                # widths and heights: 0.0 -> softmax -> uniform 1/num_bins OK
                final_layer.bias[:self.num_bins] = 0.0
                final_layer.bias[self.num_bins:2*self.num_bins] = 0.0
                
                # Derivative bias chosen so the full pipeline
                # (softplus + 0.1 floor -> 3*tanh(y/3) scaling -> clamp) yields
                # derivative ~= 1.0: x = ln(exp(0.947)-1) ~= 0.45.
                identity_bias = 0.45  # Gives derivative ~= 1.0
                final_layer.bias[2*self.num_bins:] = identity_bias
                
                _LOG.info(f"[RQS/INIT] Identity bias set to {identity_bias} for derivative ~= 1.0")

            self.param_nets.append(param_net)

        self._collapse_warning_printed = False             
        self.to(self.device)

    def _get_eps(self, tensor):
        """Get appropriate epsilon for tensor dtype."""
        if tensor.dtype == torch.float16:
            return 1e-3
        elif tensor.dtype == torch.bfloat16:
            return 1e-2
        else:
            return 1e-6

    def _get_spline_params(self, features: torch.Tensor, layer_idx: int):
        """Get constrained spline parameters with anti-collapse measures."""
        params = self.param_nets[layer_idx](features)
        
        # Split parameters
        widths_logits = params[..., :self.num_bins]
        heights_logits = params[..., self.num_bins:2*self.num_bins]
        derivatives_pre = params[..., 2*self.num_bins:]
        
        # Softer logit constraints
        # Center logits (removes constant offset)
        widths_logits = widths_logits - widths_logits.mean(dim=-1, keepdim=True)
        heights_logits = heights_logits - heights_logits.mean(dim=-1, keepdim=True)
        
        # Limit bin max/min ratio for smoother distributions:
        # max_logit_range=1.0 -> max ratio ~= exp(2) ~= 7x
        max_logit_range = 1.0
        widths_logits = torch.clamp(widths_logits, -max_logit_range, max_logit_range)
        heights_logits = torch.clamp(heights_logits, -max_logit_range, max_logit_range)

        # Softmax with slight temperature for smoother bins
        softmax_temp = 1.2  # Slightly > 1 for smoother distribution
        widths = F.softmax(widths_logits / softmax_temp, dim=-1)
        widths = self.min_bin_size + (1 - self.min_bin_size * self.num_bins) * widths
        
        heights = F.softmax(heights_logits / softmax_temp, dim=-1)
        heights = self.min_bin_size + (1 - self.min_bin_size * self.num_bins) * heights
        
        # Simple derivative pipeline for identity initialization
        # Goal: bias=0.45 -> derivative=1.0 at initialization
        # softplus(0.45) ~= 0.95, +0.05 = 1.0 OK
        derivatives = F.softplus(derivatives_pre) + 0.05
        
        # Soft bounds: allow [0.2, 5.0] range but initialized at 1.0
        derivatives = torch.clamp(derivatives, min=0.2, max=5.0)

        # DEBUG: Print spline params at first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            _LOG.debug(f"[RQS DEBUG] First call spline params:")
            _LOG.debug(f"  widths: min={widths.min():.4f}, max={widths.max():.4f}, mean={widths.mean():.4f}")
            _LOG.debug(f"  heights: min={heights.min():.4f}, max={heights.max():.4f}, mean={heights.mean():.4f}")
            _LOG.debug(f"  derivs: min={derivatives.min():.4f}, max={derivatives.max():.4f}, mean={derivatives.mean():.4f}")
            # Check for uniformity (identity)
            width_std = widths.std().item()
            height_std = heights.std().item()
            _LOG.debug(f"  width_std={width_std:.6f}, height_std={height_std:.6f} (should be ~0 for identity)")

        return widths, heights, derivatives

    def _standard_normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _standard_normal_ppf(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse CDF via erfinv."""
        eps = torch.finfo(p.dtype).eps
        p_safe = torch.clamp(p, eps, 1.0 - eps)
        return math.sqrt(2.0) * torch.erfinv(2.0 * p_safe - 1.0)

    def _base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Log pdf of the base distribution."""
        if self.base_kind == 'uniform':
            return -torch.log(torch.tensor(self.range_size, device=z.device, dtype=z.dtype)).expand_as(z)
        elif self.base_kind == 'trunc_normal':
            mu = torch.tensor(self.base_mu, device=z.device, dtype=z.dtype)        
            sigma = torch.tensor(self.base_sigma, device=z.device, dtype=z.dtype)
            a = (self.r_min - mu) / sigma
            b = (self.r_max - mu) / sigma
    
            z_std = (z - mu) / sigma
            log_pdf = -0.5 * z_std.pow(2) - math.log(math.sqrt(2.0 * math.pi)) - torch.log(sigma)
    
            cdf_a = self._standard_normal_cdf(a)
            cdf_b = self._standard_normal_cdf(b)
            Z = torch.clamp(cdf_b - cdf_a, min=torch.finfo(z.dtype).eps)
            log_Z = torch.log(Z)

            return log_pdf - log_Z
        else:
            raise ValueError(f"Unknown base_kind: '{self.base_kind}'")

    def _base_sample(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Draw base samples in [r_min, r_max]."""
        if self.base_kind == 'uniform':
            u = torch.rand(batch_size, device=device, dtype=dtype)
            return self.r_min + u * self.range_size
        elif self.base_kind == 'trunc_normal':
            mu = torch.tensor(self.base_mu, device=device, dtype=dtype)
            sigma = torch.tensor(self.base_sigma, device=device, dtype=dtype)
            a = (self.r_min - mu) / sigma
            b = (self.r_max - mu) / sigma
            Ua = self._standard_normal_cdf(a)
            Ub = self._standard_normal_cdf(b)
            u = torch.rand(batch_size, device=device, dtype=dtype)
            p = Ua + u * (Ub - Ua)
            z_std = self._standard_normal_ppf(p)
            z = mu + sigma * z_std
            return torch.clamp(z, self.r_min, self.r_max)
        else:
            raise ValueError(f"Unknown base_kind: {self.base_kind}")

    def forward(self, context_features, return_log_prob=True):
        """Sample from flow."""
        batch_size = context_features.shape[0]
        device = context_features.device
        dtype = context_features.dtype

        z = self._base_sample(batch_size, device, dtype)
        base_lp_total = self._base_log_prob(z) if return_log_prob else None

        x = z
        log_det_total = torch.zeros(batch_size, device=device, dtype=dtype)
        for i in reversed(range(self.num_layers)):
            widths, heights, derivatives = self._get_spline_params(context_features, i)
            x, log_det = self._inverse_rqs(x, widths, heights, derivatives)
            log_det_total += log_det

        # Soft clamp instead of hard error
        x = torch.clamp(x, self.r_min + self.boundary_error, self.r_max - self.boundary_error)
        
        # Add exploration noise during training to prevent std collapse
        if self.training:
            min_exploration_std = 0.15  # Minimum noise level
            noise = torch.randn_like(x) * min_exploration_std
            x = x + noise
            x = torch.clamp(x, self.r_min + self.boundary_error, self.r_max - self.boundary_error)
        
        if return_log_prob:
            log_prob = base_lp_total - log_det_total
            return x, log_prob
        else:
            return x

    def log_prob(self, x, context_features):
        """Compute log-probability of x under the flow."""
        if x.dim() > 1:
            x = x.squeeze(-1)

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        context_features = context_features.to(dtype=dtype)
        eps = self._get_eps(x)
        
        x = torch.clamp(x, self.r_min + eps, self.r_max - eps)

        log_det_total = torch.zeros(batch_size, device=device, dtype=dtype)
        z = x
        for i in range(self.num_layers):
            widths, heights, derivatives = self._get_spline_params(context_features, i)
            z, log_det = self._forward_rqs(z, widths, heights, derivatives)
            log_det_total += log_det

        base_lp = self._base_log_prob(z)
        return base_lp + log_det_total

    def _forward_rqs(self, x, widths, heights, derivatives):
        """Forward rational quadratic spline (data -> base)."""
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        eps = self._get_eps(x)

        x_norm = (x - self.r_min) / self.range_size
        x_norm = torch.clamp(x_norm, eps, 1 - eps)

        widths = widths.to(dtype) / widths.to(dtype).sum(-1, keepdim=True)
        heights = heights.to(dtype) / heights.to(dtype).sum(-1, keepdim=True)
        derivatives = derivatives.to(dtype)

        cumw = torch.cumsum(widths, dim=-1)
        cumh = torch.cumsum(heights, dim=-1)
        cumw = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), cumw], dim=1)
        cumh = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), cumh], dim=1)
        cumw[:, -1] = 1.0
        cumh[:, -1] = 1.0

        bin_idx = (x_norm.unsqueeze(1) >= cumw[:, 1:]).sum(dim=1)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1).to(torch.long)
        arange = torch.arange(B, device=device, dtype=torch.long)

        xk = cumw[arange, bin_idx]
        yk = cumh[arange, bin_idx]
        wk = torch.clamp(cumw[arange, bin_idx + 1] - xk, min=eps)
        hk = torch.clamp(cumh[arange, bin_idx + 1] - yk, min=eps)

        sk = derivatives[arange, bin_idx]
        skp1 = derivatives[arange, bin_idx + 1]

        t = torch.clamp((x_norm - xk) / wk, eps, 1 - eps)
        t1 = t * (1 - t)
        delta = hk / wk

        denom = delta + (sk + skp1 - 2.0 * delta) * t1
        y_norm = yk + hk * (delta * t * t + sk * t1) / torch.clamp(denom, min=eps)

        quad = skp1 * t * t + 2.0 * delta * t1 + sk * (1.0 - t) * (1.0 - t)
        dydx = (delta * delta) * quad / torch.clamp(denom * denom, min=eps)

        log_det = torch.log(torch.clamp(dydx, min=eps))

        y = y_norm * self.range_size + self.r_min
        return y, log_det

    def _inverse_rqs(self, y, widths, heights, derivatives):
        """Inverse rational quadratic spline (base -> data)."""
        B = y.shape[0]
        device = y.device
        dtype = y.dtype
        eps = self._get_eps(y)

        y_norm = (y - self.r_min) / self.range_size
        y_norm = torch.clamp(y_norm, eps, 1 - eps)

        widths = widths / widths.sum(-1, keepdim=True)
        heights = heights / heights.sum(-1, keepdim=True)

        cumw = torch.cumsum(widths, dim=-1)
        cumh = torch.cumsum(heights, dim=-1)
        cumw = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), cumw], dim=1)
        cumh = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), cumh], dim=1)
        cumw[:, -1] = 1.0
        cumh[:, -1] = 1.0

        bin_idx = (y_norm.unsqueeze(1) >= cumh[:, 1:]).sum(dim=1)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1).to(torch.long)
        arange = torch.arange(B, device=device, dtype=torch.long)

        xk = cumw[arange, bin_idx]
        yk = cumh[arange, bin_idx]
        wk = torch.clamp(cumw[arange, bin_idx + 1] - xk, min=eps)
        hk = torch.clamp(cumh[arange, bin_idx + 1] - yk, min=eps)

        sk = derivatives[arange, bin_idx]
        skp1 = derivatives[arange, bin_idx + 1]

        a = torch.clamp((y_norm - yk) / torch.clamp(hk, min=eps), eps, 1.0 - eps)
        delta = hk / torch.clamp(wk, min=eps)

        coef = (sk + skp1 - 2.0 * delta)
        Acoef = (delta - sk) + a * coef
        Bcoef = sk - a * coef
        Ccoef = -a * delta

        disc = torch.clamp(Bcoef * Bcoef - 4.0 * Acoef * Ccoef, min=0.0)
        sqrt_disc = torch.sqrt(disc)

        denom_quad = -Bcoef - torch.sign(Bcoef + 1e-12) * sqrt_disc
        t = (2.0 * Ccoef) / torch.clamp(denom_quad, min=eps)

        linear_mask = (Acoef.abs() < 1e-12)
        t_lin = -Ccoef / torch.clamp(Bcoef, min=eps)
        t = torch.where(linear_mask, t_lin, t)
        t = torch.clamp(t, eps, 1.0 - eps)

        # Newton refinement
        for _ in range(6):
            t1 = t * (1.0 - t)
            denom_f = delta + coef * t1
            numer_f = delta * t * t + sk * t1

            y_hat = yk + hk * numer_f / torch.clamp(denom_f, min=eps)
            f = y_hat - y_norm

            numer_deriv = 2.0 * delta * t + sk * (1.0 - 2.0 * t)
            denom_deriv = coef * (1.0 - 2.0 * t)

            dy_dt = hk * (numer_deriv * denom_f - numer_f * denom_deriv)
            dy_dt = dy_dt / torch.clamp(denom_f * denom_f, min=eps)

            step = f / torch.clamp(dy_dt.abs() + eps, min=eps) * torch.sign(dy_dt + eps)
            t = torch.clamp(t - step, eps, 1.0 - eps)

        # Handle Newton failures with bisection
        bad = (~torch.isfinite(t)) | (t < eps) | (t > 1.0 - eps)
        
        if bad.any():
            t_bisect = torch.full_like(t, 0.5)
            lower = torch.zeros_like(t)
            upper = torch.ones_like(t)
            
            for _ in range(50):
                t_test = t_bisect
                t1_test = t_test * (1.0 - t_test)
                denom_f_test = delta + coef * t1_test
                numer_f_test = delta * t_test * t_test + sk * t1_test
                
                y_test = yk + hk * numer_f_test / torch.clamp(denom_f_test, min=eps)
                error = y_test - y_norm
                
                lower = torch.where((error < 0) & bad, t_bisect, lower)
                upper = torch.where((error > 0) & bad, t_bisect, upper)
                t_bisect = 0.5 * (lower + upper)
            
            t = torch.where(bad, t_bisect, t)

        x_norm = xk + wk * t

        t1 = t * (1.0 - t)
        denom_f = delta + coef * t1
        quad = skp1 * t * t + 2.0 * delta * t1 + sk * (1.0 - t) * (1.0 - t)
        dydx = (delta * delta) * quad / torch.clamp(denom_f * denom_f, min=eps)
        log_det = -torch.log(torch.clamp(dydx, min=eps))

        x = x_norm * self.range_size + self.r_min
        return x, log_det

    def sample(self, context_features):
        """Clean sampling for consistent log_prob evaluation."""
        z = self._sample_base(context_features)  # Pass features tensor directly

        x = z
        for i in reversed(range(self.num_layers)):
            widths, heights, derivatives = self._get_spline_params(context_features, i)
            x, _ = self._inverse_rqs(x, widths, heights, derivatives)
        
        # Soft clamp to valid range
        x = torch.clamp(x, self.r_min + self.boundary_error, self.r_max - self.boundary_error)
        return x

    def _sample_base(self, features: torch.Tensor) -> torch.Tensor:
        """Sample from base distribution."""
        batch_size = features.shape[0]
        device = features.device
        dtype = features.dtype

        if self.base_kind == 'uniform':
            return torch.rand(batch_size, device=device, dtype=dtype) * self.range_size + self.r_min

        elif self.base_kind == 'trunc_normal':
            mu = torch.tensor(self.base_mu, device=device, dtype=dtype)
            sigma = torch.tensor(self.base_sigma, device=device, dtype=dtype)
            a = (self.r_min - mu) / sigma
            b = (self.r_max - mu) / sigma

            if isinstance(a, torch.Tensor):
                Phi_a = self._standard_normal_cdf(a.clone().detach().to(device=device, dtype=dtype))
            else:
                Phi_a = self._standard_normal_cdf(torch.tensor(a, device=device, dtype=dtype))
            if isinstance(b, torch.Tensor):
                Phi_b = self._standard_normal_cdf(b.clone().detach().to(device=device, dtype=dtype))
            else:
                Phi_b = self._standard_normal_cdf(torch.tensor(b, device=device, dtype=dtype))

            u = torch.rand(batch_size, device=device, dtype=dtype)
            p = Phi_a + u * (Phi_b - Phi_a)
            z = self._standard_normal_ppf(p)
            samples = mu + sigma * z

            return torch.clamp(samples, self.r_min, self.r_max)

        else:
            raise ValueError(f"Unknown base_kind: {self.base_kind}")

    def verify_consistency(self, context_features, num_samples=10):
        """Verify forward sampling and log_prob are consistent."""
        with torch.no_grad():
            samples = self.sample(context_features[:num_samples])
            log_prob_computed = self.log_prob(samples, context_features[:num_samples])
            
            assert torch.isfinite(log_prob_computed).all(), "Non-finite log probs"
            assert (log_prob_computed > -1e10).all(), "Extremely low log probs"
            
            samples2 = self.sample(context_features[:num_samples])
            log_prob2 = self.log_prob(samples2, context_features[:num_samples])
            
            mean_diff = torch.abs(log_prob_computed.mean() - log_prob2.mean())
            assert mean_diff < 0.5, f"Unstable: mean log_prob varies by {mean_diff:.3f}"
            
            return mean_diff.item()

    def entropy(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Compute differential entropy of the bounded RQS flow distribution.

        For a normalizing flow: H[p_x] = H[p_z] + E[log|det J^{-1}|]
        where p_z is the base distribution and J is the Jacobian.

        Args:
            context_features: [batch, scalar_dim] conditioning features

        Returns:
            entropy: [batch] entropy in nats
        """
        batch_size = context_features.shape[0]
        device = context_features.device
        dtype = context_features.dtype

        # Base distribution entropy
        if self.base_kind == 'uniform':
            # H[Uniform(a,b)] = log(b-a)
            base_entropy = np.log(self.range_size)

        elif self.base_kind == 'trunc_normal':
            # H[TruncNormal] ~= 0.5*log(2pie*sigma^2) - correction_term
            # For simplicity, use Gaussian approximation (accurate when truncation is mild)
            base_entropy = 0.5 * np.log(2 * np.pi * np.e * (self.base_sigma ** 2))

            # Truncation correction (reduces entropy)
            # For symmetric truncation at +/-k*sigma, correction is small if k > 2
            a_std = (self.r_min - self.base_mu) / self.base_sigma
            b_std = (self.r_max - self.base_mu) / self.base_sigma

            # Empirical correction: -log(Phi(b) - Phi(a)) where Phi is standard normal CDF
            with torch.no_grad():
                Phi_a = self._standard_normal_cdf(torch.tensor(a_std, device=device, dtype=dtype))
                Phi_b = self._standard_normal_cdf(torch.tensor(b_std, device=device, dtype=dtype))
                truncation_factor = Phi_b - Phi_a
                base_entropy = base_entropy - float(torch.log(truncation_factor).item())
        else:
            raise ValueError(f"Unknown base_kind: {self.base_kind}")

        # Monte Carlo estimate of E[log|det J^{-1}|]
        # Sample from base and compute average log_det through the flow
        n_samples = 100
        with torch.no_grad():
            # Sample base distribution
            z_samples = self._base_sample(
                batch_size * n_samples,
                device=device,
                dtype=dtype
            )

            # Expand context to match samples
            context_expanded = context_features.repeat_interleave(n_samples, dim=0)

            # Transform through inverse flow (z -> x) and accumulate log_det
            x = z_samples
            log_det_total = torch.zeros(batch_size * n_samples, device=device, dtype=dtype)

            for i in reversed(range(self.num_layers)):
                widths, heights, derivatives = self._get_spline_params(context_expanded, i)
                x, log_det = self._inverse_rqs(x, widths, heights, derivatives)
                log_det_total += log_det

            # Reshape and average over samples
            log_det_total = log_det_total.reshape(batch_size, n_samples)
            mean_log_det = log_det_total.mean(dim=1)  # [batch]

        # Flow entropy = base entropy + E[log|det J^{-1}|]
        # Note: log|det J^{-1}| is typically negative (compressing), so entropy decreases
        entropy = base_entropy + mean_log_det

        return entropy
