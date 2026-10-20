"""E(3)-equivariant state embeddings from the SevenNet backbone."""
import torch
import gc
import os
import numpy as np
from ase import io
from sevenn.model_build import build_E3_equivariant_model
from sevenn.train.dataload import atoms_to_graph
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.scale import Rescale
import sevenn._keys as KEY
from e3nn.o3 import Irreps, Linear
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List

from realize.networks.encoders.critic_embedding import (
    pick_o2_indices_strict,
    compute_geometry_features,
    get_si_mask,
    get_non_o2_mask
)
import logging
_LOG = logging.getLogger(__name__)

class _CaptureCompleteSentinel(Exception):
    """Raised from the forward hook to halt SevenNet's forward pass after the last
    target-irreps interaction layer has been captured. Downstream modules (Rescale,
    ForceOutput) would otherwise run torch.autograd.grad(E, pos, create_graph=True)
    and build a huge second-order graph that PPO never uses."""
    pass

class SimpleO2Combiner(nn.Module):
    """Parameter-free O2 embedding combiner using sum/diff.

        symmetric  = 0.5 * (O1 + O2)   swap-invariant, for direction/magnitude
        asymmetric = O1 - O2           swap-antisymmetric, for axis prediction

    E(3)-equivariant (sum/diff of equivariant vectors), zero parameters, and preserves
    irreps: output is the input irreps (128x0e+64x1e+32x2e = 480 dims). Sum/diff keeps
    pretrained features intact; a tensor-product combiner with random weights would
    scramble them.
    """

    def __init__(self, irreps_in: str, device: torch.device = None):
        super().__init__()
        if not isinstance(device, torch.device):
            raise TypeError(f"'device' must be torch.device, got {type(device)}")
        self.device = device
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = self.irreps_in  # no expansion
        self.to(device)

    def forward(
        self,
        o1_embedding: torch.Tensor,
        o2_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric (sum) and asymmetric (diff) embeddings.

        Args:
            o1_embedding: [B, D] O1 node embedding from SevenNet
            o2_embedding: [B, D] O2 node embedding from SevenNet

        Returns:
            (asymmetric, symmetric): Both [B, D] with same irreps as input
                asymmetric = O1 - O2  (flips sign under O1<->O2 swap)
                symmetric  = 0.5*(O1 + O2)  (invariant under O1<->O2 swap)
        """
        if o1_embedding.dim() == 1:
            o1_embedding = o1_embedding.unsqueeze(0)
        if o2_embedding.dim() == 1:
            o2_embedding = o2_embedding.unsqueeze(0)

        if o1_embedding.shape != o2_embedding.shape:
            raise ValueError(
                f"Shape mismatch: o1={list(o1_embedding.shape)}, o2={list(o2_embedding.shape)}"
            )
        if o1_embedding.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Embedding dim {o1_embedding.shape[-1]} != irreps dim {self.irreps_in.dim}"
            )

        symmetric = 0.5 * (o1_embedding + o2_embedding)
        asymmetric = o1_embedding - o2_embedding

        return asymmetric, symmetric


class EmbeddingExtractor(nn.Module):
    """Extracts E(3)-equivariant embeddings using the SevenNet GNN.

    Trainable nn.Module: supports per-layer freeze/unfreeze of the actor and critic
    backbones for task-specific fine-tuning.
    """
    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cpu',
                 use_sevennet: bool = True,
                 lammps_path: str = None,
                 expected_irreps: Optional[str] = None,
                 train_backbone: bool = False,
                 unfreeze_last_n_layers: int = 2):

        super().__init__()

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is REQUIRED")
        self.checkpoint_path = checkpoint_path

        self.device_str = str(device)
        self.device = torch.device(device)

        self.use_sevennet = use_sevennet
        self.lammps_path = lammps_path
        self.expected_irreps = expected_irreps

        # Backbone training configuration
        self.train_backbone = train_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        self.model = None
        self.atoms = None
        self.graph_data = None
        self.cutoff = None
        self.checkpoint_config = None
        self.model_state_dict = None
        self.irreps_atom = None
        self.embedding_dim = None

        self.o2_embedder = None  # built in build_model()

        # SO(3)-safe geometry features (no fixed axes).
        self.geom_dim = 6  # [d_OO, dmin_O_Si, dmin_COM_Si, dmin_O_any, align2, align_signed]

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{self.checkpoint_path}' not found.")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if "config" not in checkpoint:
            raise KeyError("Checkpoint missing 'config' key")

        self.checkpoint_config = checkpoint["config"]
        self.model_state_dict = checkpoint.get("model_state_dict", {})

        self.cutoff = self.checkpoint_config.get(KEY.CUTOFF)
        if self.cutoff is None:
            raise ValueError("Checkpoint config missing cutoff value")

        irreps_manual = self.checkpoint_config.get("irreps_manual", None)
        if not irreps_manual or len(irreps_manual) == 0:
            raise ValueError("Checkpoint MUST include 'irreps_manual' list.")

        final_irreps_str = irreps_manual[-1]
        expected_base_irreps = "128x0e+64x1e+32x2e"

        if final_irreps_str != expected_base_irreps:
            raise ValueError(
                f"Checkpoint irreps must be SevenNet base output: {expected_base_irreps}\n"
                f"Got: {final_irreps_str}"
            )

        self.irreps_atom = Irreps(final_irreps_str)
        self.embedding_dim = self.irreps_atom.dim

        shift = float(self.model_state_dict["rescale_atomic_energy.shift"])
        scale = float(self.model_state_dict["rescale_atomic_energy.scale"])
        self.per_atom_energy_mean = shift
        self.force_rms = scale

        del checkpoint
        gc.collect()

    def build_model(self):
        """Build SevenNet model with optional fine-tuning."""
        if self.model is not None:
            return

        if self.checkpoint_config is None:
            self.load_checkpoint()

        model_config = dict(self.checkpoint_config)
        model_config[KEY.SHIFT] = [self.per_atom_energy_mean]
        model_config[KEY.SCALE] = [self.force_rms]

        self.model = build_E3_equivariant_model(model_config)
        self.model.load_state_dict(self.model_state_dict, strict=False)
        self.model.to(self.device)

        if not self.train_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            self._configure_selective_unfreeze()

        # SimpleO2Combiner has zero parameters; nothing to freeze/unfreeze.
        if self.o2_embedder is None:
            self.o2_embedder = SimpleO2Combiner(
                irreps_in=str(self.irreps_atom),
                device=self.device
            )

    def _configure_selective_unfreeze(self):
        """Selectively unfreeze the last N layers of the SevenNet backbone.

        SevenNet uses flat naming: {layer_idx}_{component_name}, e.g.
        "3_convolution.weight", "2_self_interaction_1.weight".
        """
        for param in self.model.parameters():
            param.requires_grad = False

        _LOG.info(f"  Detecting layer structure from parameter names...")

        layer_indices = set()
        for name, _ in self.model.named_parameters():
            parts = name.split('_')
            if parts[0].isdigit():
                layer_indices.add(int(parts[0]))

        if not layer_indices:
            raise RuntimeError(
                "FAIL: Could not detect layer structure from SevenNet param names. "
                "Expected parameter names like '3_convolution.weight'. "
                "No fallback -- check model checkpoint format."
            )

        sorted_layers = sorted(layer_indices)
        total_layers = len(sorted_layers)
        layers_to_unfreeze = sorted_layers[-self.unfreeze_last_n_layers:]

        _LOG.info(f"  Found {total_layers} layers: {sorted_layers}")
        _LOG.info(f"  Unfreezing last {len(layers_to_unfreeze)} layers: {layers_to_unfreeze}")

        unfrozen_params = 0
        unfrozen_by_layer = {idx: 0 for idx in layers_to_unfreeze}
        sample_params = []

        for name, param in self.model.named_parameters():
            parts = name.split('_')
            if parts[0].isdigit():
                layer_idx = int(parts[0])
                if layer_idx in layers_to_unfreeze:
                    param.requires_grad = True
                    unfrozen_params += 1
                    unfrozen_by_layer[layer_idx] += 1

                    if len(sample_params) < 3:
                        sample_params.append(f"{name} {list(param.shape)}")

        _LOG.info(f"  Unfroze {unfrozen_params} parameters across {len(layers_to_unfreeze)} layers:")
        for layer_idx in sorted(unfrozen_by_layer.keys()):
            count = unfrozen_by_layer[layer_idx]
            _LOG.info(f"    Layer {layer_idx}: {count} parameters")

        if sample_params:
            _LOG.info(f"  Sample unfrozen params:")
            for sp in sample_params:
                _LOG.info(f"    {sp}")

        if unfrozen_params == 0:
            raise RuntimeError(
                f"FAIL: Requested to unfreeze last {self.unfreeze_last_n_layers} layers "
                f"but 0 parameters were unfrozen. Layers: {layers_to_unfreeze}. "
                f"Check parameter naming convention."
            )

        # Train mode keeps gradients flowing through the unfrozen layers.
        self.model.train()

    def load_structure(self, atoms=None):
        if atoms is not None:
            self.atoms = atoms
        elif self.lammps_path:
            self.atoms = io.read(self.lammps_path, format="lammps-data", atom_style="charge")
        else:
            return  # No structure to load yet

        n_atoms = len(self.atoms)
        if "y_energy" not in self.atoms.info:
            self.atoms.info["y_energy"] = 0.0
        if "y_force" not in self.atoms.arrays:
            self.atoms.arrays["y_force"] = np.zeros((n_atoms, 3))
        self.chem_species = sorted(set(self.atoms.get_atomic_numbers()))
        self.type_map = {elem: i for i, elem in enumerate(self.chem_species)}

    def convert_to_graph(self):
        if self.atoms is None:
            raise RuntimeError("No atoms loaded")
        gdict = atoms_to_graph(self.atoms, self.cutoff)
        self.graph_data = AtomGraphData.from_numpy_dict(gdict)
        num_nodes = self.graph_data[KEY.POS].shape[0]
        if KEY.BATCH not in self.graph_data:
            self.graph_data[KEY.BATCH] = torch.zeros(num_nodes, dtype=torch.long)

    def get_equivariant_embedding(
        self,
        atoms=None,
        return_symmetric_only: bool = False,
        dmin: float = 0.9,
        dmax: float = 2.5
    ) -> Any:
        """Get embedding for the current structure.

        Returns dict with:
            - 'asymmetric': [1, 480] For axis prediction head (equivariant, swap-antisymmetric)
            - 'symmetric': [1, 480] For direction/magnitude heads (equivariant, swap-invariant)
            - 'o1_embedding': Raw O1 node embedding [1, 480] (for critic)
            - 'o2_embedding': Raw O2 node embedding [1, 480] (for critic)
            - 'geometry': SO(3)-invariant geometry features [1, geom_dim]
            - 'o1_idx', 'o2_idx': LAMMPS-identity indices
        """
        if self.model is None:
            raise RuntimeError("Model not built")

        if atoms is not None:
            self.load_structure(atoms)
            self.convert_to_graph()

        if self.graph_data is None:
            raise RuntimeError("Graph data not prepared")

        data = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.graph_data.items()}
        # Forces require position gradients.
        if KEY.POS in data:
            data[KEY.POS] = data[KEY.POS].requires_grad_(True)

        full_embedding = self._capture_features_with_target_irreps_dim(data)

        o1_idx, o2_idx = pick_o2_indices_strict(
            self.atoms,
            assume_last_two=True,
            dmin=dmin,
            dmax=dmax
        )

        o1_embedding = full_embedding[o1_idx].unsqueeze(0)  # [1, 480]
        o2_embedding = full_embedding[o2_idx].unsqueeze(0)  # [1, 480]

        # First-layer (learned multipole): h_i = Sigma_j W_Zj * R_n(r_ij) * Y_lm(r_hat_ij),
        # a species-weighted multipole with Bessel radial basis. Its symmetric part is a clean
        # substrate directional signal (O1<->O2 cross-terms cancel, substrate terms add).
        first_layer = self._first_layer_embedding
        o1_first = first_layer[o1_idx].unsqueeze(0)  # [1, 480]
        o2_first = first_layer[o2_idx].unsqueeze(0)  # [1, 480]
        _, symmetric_first = self.compute_o2_embeddings(o1_first, o2_first)

        # Actor embeddings via SimpleO2Combiner: asymmetric [1, 480], symmetric [1, 480].
        asymmetric_o2, symmetric_o2 = self.compute_o2_embeddings(o1_embedding, o2_embedding)

        if return_symmetric_only:
            return symmetric_o2.detach() if not self.train_backbone else symmetric_o2

        # SO(3)-safe geometry features for the critic; PBC via minimum image convention.
        pos = data[KEY.POS]  # [N, 3]
        si_mask = get_si_mask(self.atoms, self.device)
        non_o2_mask = get_non_o2_mask(self.atoms, o1_idx, o2_idx, self.device)

        cell_np = np.array(self.atoms.get_cell())
        pbc_np = np.array(self.atoms.get_pbc())
        cell = torch.tensor(cell_np, dtype=torch.float32, device=self.device)
        pbc = torch.tensor(pbc_np, dtype=torch.bool, device=self.device)

        geometry = compute_geometry_features(
            pos=pos,
            o1_idx=o1_idx,
            o2_idx=o2_idx,
            si_mask=si_mask,
            non_o2_mask=non_o2_mask,
            cell=cell,
            pbc=pbc,
            k_nearest_si=1,
            include_signed_align=True
        )  # [1, geom_dim]

        # Detach critic inputs for buffer storage only when the backbone is frozen;
        # if trainable, keep gradients for the RL value loss.
        if self.train_backbone:
            o1_emb_out = o1_embedding
            o2_emb_out = o2_embedding
            geom_out = geometry
        else:
            o1_emb_out = o1_embedding.detach()
            o2_emb_out = o2_embedding.detach()
            geom_out = geometry.detach()

        return {
            'asymmetric': asymmetric_o2,
            'symmetric': symmetric_o2,
            'symmetric_first_layer': symmetric_first.detach(),
            'o1_embedding': o1_emb_out,
            'o2_embedding': o2_emb_out,
            'geometry': geom_out,
            'o1_idx': o1_idx,
            'o2_idx': o2_idx
        }

    def _capture_features_with_target_irreps_dim(self, data: Dict[str, Any]) -> torch.Tensor:
        target_irreps = self.irreps_atom
        target_dim = target_irreps.dim
        captures = []
        handles = []

        # Enumerate matching modules in forward (registration) order so the LAST
        # one can short-circuit the rest of the SevenNet forward.
        matching_modules: List[nn.Module] = []
        for name, m in self.model.named_modules():
            ir = getattr(m, "irreps_out", None)
            if ir is not None and Irreps(ir) == target_irreps:
                matching_modules.append(m)

        if not matching_modules:
            raise RuntimeError(
                f"No modules with irreps_out == {target_irreps} found in SevenNet model. "
                f"No fallback -- check checkpoint irreps_manual."
            )

        last_matching = matching_modules[-1]

        def make_hook(is_last: bool):
            def hook(module, inputs, output):
                ten = None
                if isinstance(output, dict) and KEY.NODE_FEATURE in output:
                    ten = output[KEY.NODE_FEATURE]
                elif isinstance(output, torch.Tensor) and output.dim() == 2 and output.shape[-1] == target_dim:
                    ten = output

                if ten is not None:
                    if self.train_backbone:
                        captures.append(ten)
                    else:
                        captures.append(ten.detach())

                if is_last:
                    # Halt before Rescale + ForceOutput. ForceOutput would call
                    # torch.autograd.grad(E, pos, create_graph=True), wasting memory
                    # on a force graph we never read.
                    raise _CaptureCompleteSentinel()
            return hook

        for m in matching_modules:
            handles.append(m.register_forward_hook(make_hook(is_last=(m is last_matching))))

        try:
            # Grad must remain enabled so the unfrozen last-N layers can backprop
            # during the outer PPO update; the sentinel only stops the forward.
            with torch.set_grad_enabled(True):
                try:
                    _ = self.model(data)
                except _CaptureCompleteSentinel:
                    pass  # expected: forward short-circuited after last target capture
        finally:
            for h in handles:
                h.remove()

        if not captures:
            raise RuntimeError(f"No features captured for {target_irreps}")
        result = captures[-1].to(self.device)
        # First layer (learned multipole, direct geometric encoding) feeds the actor direction
        # signal; last layer is multi-body chemical features after message passing.
        if len(captures) >= 2:
            self._first_layer_embedding = captures[0].to(self.device)
        else:
            self._first_layer_embedding = result
        return result

    def compute_o2_embeddings(
        self,
        o1_embedding: torch.Tensor,
        o2_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric and asymmetric O2 embeddings.

        Returns:
            (asymmetric, symmetric): Both [B, 480] with irreps 128x0e+64x1e+32x2e
        """
        if self.o2_embedder is None:
            raise RuntimeError(
                "FAIL: o2_embedder is None. Call build_model() first. No lazy fallback."
            )
        return self.o2_embedder(o1_embedding, o2_embedding)

    def get_backbone_parameters(self):
        """
        Get ONLY backbone parameters (excluding o2_embedder).
        Used for separate learning rate in optimizer.
        """
        if not self.train_backbone or self.model is None:
            return []

        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def freeze_backbone(self):
        """Temporarily freeze all backbone parameters (for warmup period)."""
        if self.model is None:
            return
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                count += 1
        _LOG.info(f"[EmbeddingExtractor] Froze {count} backbone parameters for warmup")

    def unfreeze_backbone(self):
        """Re-apply selective unfreezing (restores pre-warmup configuration)."""
        if self.model is None or not self.train_backbone:
            return
        self._configure_selective_unfreeze()
        count = sum(1 for p in self.model.parameters() if p.requires_grad)
        _LOG.info(f"[EmbeddingExtractor] Unfroze backbone: {count} trainable parameters")

    def get_o2_embedder_parameters(self):
        """Get o2_embedder parameters (separate optimizer group).

        SimpleO2Combiner has zero params; this stays correct if it gains trainable ones.
        """
        if self.o2_embedder is None:
            return []
        return [p for p in self.o2_embedder.parameters() if p.requires_grad]

    def get_geom_dim(self) -> int:
        """Return dimension of geometry features."""
        return self.geom_dim


def get_embedding_vector(lammps_path: str,
                         checkpoint_path: str,
                         device: torch.device,
                         **kwargs) -> Dict[str, torch.Tensor]:
    # Convenience one-shot helper: builds a fresh extractor on every call.
    extractor = EmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=str(device),
        lammps_path=lammps_path
    )
    extractor.load_checkpoint()
    extractor.load_structure()
    extractor.convert_to_graph()
    extractor.build_model()

    return extractor.get_equivariant_embedding(return_symmetric_only=False)
