"""Sequential containers and activations for e3nn irreps."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, Linear
from typing import Optional, List, Callable, Union
import logging
_LOG = logging.getLogger(__name__)

class IrrepSequential(nn.Module):
    """Sequential container for e3nn irrep-aware layers. Layers are fixed after init."""

    def __init__(self, *layers):
        """Args: *layers: e3nn layers, each exposing irreps_in/irreps_out."""
        super().__init__()
        self.layers = nn.ModuleList(layers)

        # Validate adjacent layer irreps match
        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i + 1]

            if not hasattr(curr_layer, 'irreps_out'):
                raise TypeError(
                    f"Layer {i} ({type(curr_layer).__name__}) missing 'irreps_out' attribute. "
                    f"Cannot validate pipeline without irreps metadata. "
                    f"All layers in IrrepSequential must expose irreps."
                )
            curr_out = curr_layer.irreps_out

            if not hasattr(next_layer, 'irreps_in'):
                raise TypeError(
                    f"Layer {i+1} ({type(next_layer).__name__}) missing 'irreps_in' attribute. "
                    f"Cannot validate pipeline without irreps metadata. "
                    f"All layers in IrrepSequential must expose irreps."
                )
            next_in = next_layer.irreps_in

            if curr_out.simplify() != next_in.simplify():
                raise ValueError(            
                    f"Layer {i} ({type(curr_layer).__name__}) output irreps {curr_out} "
                    f"doesn't match layer {i+1} ({type(next_layer).__name__}) input irreps {next_in}"
                )

        # Expose container irreps from first/last layer
        if len(self.layers) > 0:
            if hasattr(self.layers[0], 'irreps_in'):
                self.irreps_in = self.layers[0].irreps_in
            if hasattr(self.layers[-1], 'irreps_out'):
                self.irreps_out = self.layers[-1].irreps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all layers in sequence."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self) -> str:
        """String representation showing layer structure"""
        lines = ['IrrepSequential(']
        for i, layer in enumerate(self.layers):
            layer_str = f"  ({i}): {layer}"
            if hasattr(layer, 'irreps_in') and hasattr(layer, 'irreps_out'):
                layer_str += f"  # {layer.irreps_in} -> {layer.irreps_out}"
            lines.append(layer_str)
        lines.append(')')
        return '\n'.join(lines)


class IrrepActivation(nn.Module):
    """Activation for irreps: SiLU on scalars, sigmoid-of-norm gating on non-scalars (stable)."""

    def __init__(self,
                 irreps: Irreps,
                 acts: Optional[List[Optional[Callable]]] = None,
                 normalize: bool = True):  # normalize: kept for API compatibility
        super().__init__()
        
        if isinstance(irreps, str):
            irreps = Irreps(irreps)
        
        self.irreps_in = irreps
        self.irreps_out = irreps

        self.total_dim = sum(mul * (2 * ir.l + 1) for mul, ir in irreps)

        # Default: SiLU on scalars, None (norm-based gating) on l>0
        if acts is None:
            acts = []
            for mul, ir in irreps:
                if ir.l == 0:
                    acts.append(F.silu)
                else:
                    acts.append(None)
        else:
            if len(acts) != len(list(irreps)):
                raise ValueError(f"Length of acts ({len(acts)}) must match irrep blocks ({len(list(irreps))})")
        
        self.acts = acts

        # Uses the manual gating implementation for stability.
        self.use_manual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation"""
        if x.shape[-1] != self.total_dim:
            raise ValueError(f"Input dim {x.shape[-1]} != expected {self.total_dim}")
            
        return self._manual_activation(x)

    def _manual_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Scalars (l=0): apply act directly. Non-scalars (l>0): gate by x * sigmoid(norm(x)),
        which is stable and direction-preserving."""
        if not self.acts:
            return x
        
        output_parts = []
        start_idx = 0
        
        for (mul, ir), act_fn in zip(self.irreps_in, self.acts):
            dim = mul * (2 * ir.l + 1)
            end_idx = start_idx + dim
            
            x_part = x[:, start_idx:end_idx]
            
            if ir.l == 0:
                # Scalar: apply act, identity if None
                fn = act_fn if act_fn is not None else nn.Identity()
                x_part = fn(x_part)
            else:
                # Non-scalar: gate by sigmoid(norm); reshape to [batch, mul, 2l+1]
                x_vectors = x_part.reshape(x.shape[0], mul, 2*ir.l+1)

                # eps stabilizes the sqrt gradient at 0 (sqrt(0) has infinite grad)
                norms_sq = (x_vectors**2).sum(dim=2, keepdim=True)
                norms = torch.sqrt(norms_sq + 1e-8)

                # sigmoid(norm) squashes large vectors, smoothly gates small ones
                scales = torch.sigmoid(norms)

                x_part_gated = x_vectors * scales
                x_part = x_part_gated.reshape(x.shape[0], dim)

            output_parts.append(x_part)
            start_idx = end_idx
        
        return torch.cat(output_parts, dim=1)

class GateActivation(nn.Module):
    """Gates non-scalar irreps using scalar gates (manual gating implementation)."""

    def __init__(self,
                 irreps: Irreps,
                 scalar_act: Optional[Callable] = F.silu,
                 gate_act: Optional[Callable] = torch.sigmoid):
        super().__init__()
        
        if isinstance(irreps, str):
            irreps = Irreps(irreps)
        
        self.irreps_in = irreps
        self.irreps_out = irreps
        self.scalar_act = scalar_act
        self.gate_act = gate_act

        self.total_dim = sum(mul * (2 * ir.l + 1) for mul, ir in irreps)

        # Separate scalars and non-scalars, preserving parity
        self.scalars = []
        self.nonscalars = []
        self.irrep_indices = []
        
        idx = 0
        for mul, ir in irreps:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                self.scalars.append((idx, idx + dim, mul))
            else:
                self.nonscalars.append((idx, idx + dim, mul, ir.l, ir.p))
            self.irrep_indices.append((idx, idx + dim, ir.l))
            idx += dim

        # One gate per non-scalar field (sum of multiplicities across l>0 blocks)
        self.num_gates = sum(mul for _, _, mul, _, _ in self.nonscalars)

        total_scalar_channels = sum(mul for _, _, mul in self.scalars)
        if self.num_gates > 0 and total_scalar_channels < self.num_gates:
            raise ValueError(
                f"Insufficient scalar channels ({total_scalar_channels}) "
                f"for gating {self.num_gates} non-scalar fields. "
                f"Need at least {self.num_gates} scalar channels."
            )

        # Prefer contiguous scalars for readability (soft note, no fail)
        if len(self.scalars) > 1 and self.num_gates > 0:
            first_block_size = self.scalars[0][2] if self.scalars else 0
            if first_block_size < self.num_gates:
                _LOG.info(
                    f"[GateActivation] Note: scalar channels are fragmented; "
                    f"collecting across {len(self.scalars)} blocks. "
                    f"For clarity, prefer contiguous scalars (e.g. '16x0e+8x1e').")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation."""
        if x.shape[-1] != self.total_dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} doesn't match "
                f"expected {self.total_dim} from irreps {self.irreps_in}"
            )

        if self.num_gates == 0:
            return self._apply_scalar_activation(x)

        # Collect first num_gates scalar channels across all scalar blocks (may be non-contiguous)
        gate_values = []
        if len(self.scalars) > 0 and self.num_gates > 0:
            scalar_channels = []
            channels_collected = 0

            for start_idx, end_idx, mul in self.scalars:
                if channels_collected >= self.num_gates:
                    break

                channels_needed = self.num_gates - channels_collected
                channels_available = min(mul, channels_needed)

                scalar_part = x[:, start_idx:start_idx + channels_available]
                scalar_channels.append(scalar_part)
                channels_collected += channels_available

            if channels_collected < self.num_gates:
                raise RuntimeError(
                    f"Could not collect {self.num_gates} scalar channels for gating. "
                    f"Only found {channels_collected} scalar channels."
                )
            
            gates_tensor = torch.cat(scalar_channels, dim=1)
            if self.gate_act is not None:
                gates_tensor = self.gate_act(gates_tensor)
            gate_values = gates_tensor.unbind(dim=1)

        output_parts = []
        gate_idx = 0

        for start, end, l in self.irrep_indices:
            x_part = x[:, start:end]

            if l == 0:
                if self.scalar_act is not None:
                    x_part = self.scalar_act(x_part)
            else:
                # Per-field gating: reshape to [B, mul, 2l+1]
                B, D = x_part.shape
                m_dim = 2 * l + 1
                if D % m_dim != 0:
                    raise RuntimeError(
                        f"Gating shape mismatch: block dim {D} not divisible by 2*l+1={m_dim}"
                    )
                mul = D // m_dim
                x_part_reshaped = x_part.view(B, mul, m_dim)

                if gate_idx + mul > len(gate_values):
                    raise RuntimeError(
                        f"Not enough gate scalars: need {gate_idx+mul}, "
                        f"have {len(gate_values)}"
                    )
                gates_block = torch.stack(
                    gate_values[gate_idx:gate_idx + mul], dim=1
                )  # [B, mul]
                gate_idx += mul

                x_part_reshaped = x_part_reshaped * gates_block.unsqueeze(-1)
                x_part = x_part_reshaped.view(B, D)
            
            output_parts.append(x_part)
        
        return torch.cat(output_parts, dim=1)

    def _apply_scalar_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation only to scalar parts."""
        if not self.scalar_act:
            return x
        
        output_parts = []
        for start, end, l in self.irrep_indices:
            x_part = x[:, start:end]
            if l == 0:
                x_part = self.scalar_act(x_part)
            output_parts.append(x_part)
        
        return torch.cat(output_parts, dim=1)

def create_irrep_network(irreps_in: Irreps,
                        irreps_out: Irreps,
                        hidden_irreps: Optional[Irreps] = None,
                        num_layers: int = 2,
                        use_gate: bool = False,
                        normalize: bool = True) -> IrrepSequential:
    """Build an IrrepSequential of Linear+activation layers.

    With use_gate=True, hidden_irreps must have enough scalar channels to gate all
    non-scalar blocks (one gate per l>0 block).
    """
    if isinstance(irreps_in, str):
        irreps_in = Irreps(irreps_in)
    if isinstance(irreps_out, str):
        irreps_out = Irreps(irreps_out)
    
    if hidden_irreps is None:
        raise ValueError(
            "hidden_irreps must be provided explicitly (no default values). "
            "Specify irreps string like '32x0e+16x1e+8x2e' based on your architecture needs."
        )
    elif isinstance(hidden_irreps, str):
        hidden_irreps = Irreps(hidden_irreps)

    layers = []
    
    if num_layers == 1:
        layers.append(Linear(irreps_in, irreps_out))
    elif num_layers == 2:
        layers.append(Linear(irreps_in, hidden_irreps))
        if use_gate:
            layers.append(GateActivation(hidden_irreps))
        else:
            layers.append(IrrepActivation(hidden_irreps, normalize=normalize))
        layers.append(Linear(hidden_irreps, irreps_out))
    else:
        layers.append(Linear(irreps_in, hidden_irreps))
        if use_gate:
            layers.append(GateActivation(hidden_irreps))
        else:
            layers.append(IrrepActivation(hidden_irreps, normalize=normalize))

        for _ in range(num_layers - 2):
            layers.append(Linear(hidden_irreps, hidden_irreps))
            if use_gate:
                layers.append(GateActivation(hidden_irreps))
            else:
                layers.append(IrrepActivation(hidden_irreps, normalize=normalize))

        layers.append(Linear(hidden_irreps, irreps_out))
    
    return IrrepSequential(*layers)

def extract_scalar_features(features: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Extract only 0e (even scalar) components from an irrep tensor [batch, feature_dim]."""
    scalars = []
    idx = 0

    for mul, ir in irreps:
        dim = mul * (2 * ir.l + 1)
        if ir.l == 0 and ir.p == 1:  # 0e
            scalars.append(features[:, idx:idx + dim])
        idx += dim

    if scalars:
        return torch.cat(scalars, dim=1)
    else:
        raise ValueError(
            f"No 0e scalar components present in features for given irreps: {irreps}. "
            f"Cannot extract scalars from irreps with no l=0 terms. "
            f"Check that your network produces scalar outputs."
        )        
