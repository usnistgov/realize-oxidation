"""E(3)-equivariant policy and value network."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional

from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Irreps, Linear

from realize.networks.heads.equivariant_sh_head import EquivariantSHHead
from realize.networks.encoders.frame_encoder import FrameAlignmentEncoder
from realize.networks.heads.magnitude_predictor import MagnitudePredictor
from realize.networks.distributions.policy_utils.policy_direction_distribution import DirectionDistributionManager
from realize.networks.distributions.policy_utils.policy_axis_sampling import AxisSamplingManager
from realize.networks.distributions.policy_utils.policy_helpers import extract_o2_positions, validate_temperature
from realize.networks.heads.sh_head_utils.sh_head_layers import rotate_o2_asymmetric_to_aligned
from realize.networks.irrep_sequential import IrrepSequential, IrrepActivation
from realize.networks.encoders.critic_embedding import CriticEmbedding, CriticRunningNormalizer

from realize.networks.policy_network_evaluate import PolicyEvaluateMixin
from realize.networks.policy_network_accessors import PolicyAccessorsMixin
import logging
_LOG = logging.getLogger(__name__)

class PolicyNetwork(PolicyEvaluateMixin, PolicyAccessorsMixin, nn.Module):
    """
    E(3)-equivariant policy network for atomic-scale O2 dissociation.
    """
    def __init__(self, config, device, total_episodes, sh_lr_mult=1.0, 
                 value_lr_mult=1.0, use_shared_model=False, shared_model_dir=None,
                 irreps_atom: str = "128x0e+64x1e+32x2e", geom_dim: int = 10,
                 zero_backbone_l1l2: bool = False,
                 geometry_only_critic: bool = False):                 
        super().__init__()

        self.debug_mode = True
        self.zero_backbone_l1l2 = zero_backbone_l1l2
        self.geometry_only_critic = geometry_only_critic
        self.config = config
        self.device = device
        self.current_episode = 0
        self.bond_length = 1.19725
        
        self.model_config = config.model if hasattr(config, 'model') else config
        self.channel_config = config.channels if hasattr(config, 'channels') else None
        self.sampling_config = config.sampling if hasattr(config, 'sampling') else None
        
        if self.sampling_config is None: raise ValueError("sampling_config is required")
        if self.channel_config is None: raise ValueError("channel_config is required")
        
        self.lmax = self.model_config.lmax
        self.num_channels = self.model_config.num_channels
        self.max_trans_distance = self.model_config.max_trans_distance
        self.mlp_hidden_dim = self.model_config.mlp_hidden_dim
        
        _LOG.info(f"Initializing FrameAlignmentEncoder...")
        self.frame_encoder = FrameAlignmentEncoder(self.model_config, device)

        # Frame encoder output irreps feed the vMF axis head
        enc_out_irreps = (
            self.frame_encoder.get_output_irreps()
            if hasattr(self.frame_encoder, "get_output_irreps")
            else getattr(self.frame_encoder, "output_irreps", None)
        )

        if enc_out_irreps is None:
            raise RuntimeError(
                "frame_encoder must expose get_output_irreps() or output_irreps "
                "to build axis_from_frame."
            )

        if not isinstance(enc_out_irreps, Irreps):
            enc_out_irreps = Irreps(enc_out_irreps)

        _LOG.info(f"[PolicyNetwork] Frame encoder output irreps: {enc_out_irreps} (dim={enc_out_irreps.dim})")

        # O2 asymmetric embedding irreps: 128x0e+64x1e+32x2e = 480 dims
        o2_asym_irreps = Irreps("128x0e+64x1e+32x2e")

        self.axis_from_frame = self._build_equivariant_axis_head(enc_out_irreps, o2_asym_irreps)

        _LOG.info(f"[PolicyNetwork] Axis head built with frame_irreps={enc_out_irreps}, o2_irreps={o2_asym_irreps}")

        # SH head reads encoder_output_irreps (frame_encoder OUTPUT) from model_config
        self.model_config.encoder_output_irreps = str(enc_out_irreps)

        _LOG.info(f"Initializing Equivariant SH Head...")

        self.sh_head = EquivariantSHHead(
            self.model_config,
            self.channel_config,
            device,
            reward_config=getattr(config, "reward", None),
        )        

        self.embedding_extractor = self.frame_encoder
        self.use_radial_conditioning = getattr(config, 'use_radial_conditioning', True)
        self.radial_num_basis = getattr(config, 'radial_num_basis', 32)
        self.radial_max_radius = getattr(config, 'radial_max_radius', 10.0)
        
        self.magnitude_predictor = MagnitudePredictor(
            self.model_config, 
            device, 
            self.sampling_config
        )

        # CriticEmbedding lives inside PolicyNetwork so PPO-update gradients flow through it.
        # Buffer stores raw inputs (o1_embedding, o2_embedding, geometry), NOT critic_feat;
        # _compute_value() recomputes critic_feat during the update.
        self.irreps_atom = Irreps(irreps_atom)
        self.geom_dim = geom_dim

        # Critic input is raw [geom, sym_l0] = 138 dims; no LayerNorm/Linear inside.
        self.geometry_only_critic = False
        self.critic_embedder = CriticEmbedding(
            irreps_atom=self.irreps_atom,
            geom_dim=self.geom_dim,
        )
        critic_out_dim = self.critic_embedder.out_dim  # 138 = geom(10) + sym_l0(128)
        critic_hidden_dim = 128

        # Per-feature running-stats normalizer (replaces LayerNorm). Updated only from the
        # rollout path (update_stats=True) so rollout and PPO-update V(s) share statistics.
        self.critic_input_normalizer = CriticRunningNormalizer(feature_dim=critic_out_dim)

        _LOG.info(f"[PolicyNetwork] CriticEmbedding: irreps={irreps_atom}, "
              f"geom_dim={geom_dim}, out_dim={critic_out_dim}, "
              f"critic_hidden={critic_hidden_dim}")
        # Value MLP; normalization handled by the external running-stats normalizer.
        self.value_mlp = nn.Sequential(
            nn.Linear(critic_out_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

        # Initialize value network with proper scaling
        self._init_value_network()

        self._last_outputs = {}

        self._step_counter = 0
        self._consistency_step = 0

        self.to(device)
        
        self.direction_manager = DirectionDistributionManager(self, device)
        self.axis_manager = AxisSamplingManager(self)

        self._validate_irrep_boundaries()

        # Environmental multipole injection: projects 9-dim multipole (1x0e+1x1e+1x2e) into
        # 480-dim state space, added to state. Small init for a gentle perturbation.
        multipole_irreps = Irreps("1x0e + 1x1e + 1x2e")
        state_irreps = Irreps("128x0e + 64x1e + 32x2e")
        self.multipole_injector = Linear(multipole_irreps, state_irreps).to(self.device)
        with torch.no_grad():
            for p in self.multipole_injector.parameters():
                nn.init.normal_(p, std=0.1)
        # compass_scale: registered but inert. Nothing reads it in forward, so it never gets a
        # gradient and stays at its 0.0 init (output-neutral). It is part of the forward
        # fingerprint (44 tensors / 75978 scalars); carries no RNG.
        self.compass_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        _LOG.info(f"[PolicyNetwork] Multipole injector: {multipole_irreps} -> {state_irreps} (init std=0.1)")

    def _build_equivariant_axis_head(self, frame_irreps: Irreps, o2_asym_irreps: Irreps) -> nn.Module:
        """E(3)-equivariant axis head. Takes frame-aligned features and O2 asymmetric embedding
        (both in the aligned frame), combines via irreps-aware ops, outputs 1x1e (axis direction)
        + 1x0e (log concentration). Returns an nn.Module:
        (frame_features, o2_embedding_asymmetric) -> [B, 4]."""

        class EquivariantAxisHead(nn.Module):
            """Projects frame features and O2 asymmetric embedding to a shared intermediate irreps,
            concatenates, then projects to 1x1e + 1x0e."""
            def __init__(self, frame_irreps, o2_asym_irreps, device):
                super().__init__()
                self.frame_irreps = frame_irreps
                self.o2_asym_irreps = o2_asym_irreps
                self.device = device
                
                self.intermediate_irreps = Irreps("16x0e+8x1e+4x2e")

                self.frame_proj = Linear(frame_irreps, self.intermediate_irreps)
                self.o2_proj = Linear(o2_asym_irreps, self.intermediate_irreps)

                self.combined_irreps = self.intermediate_irreps + self.intermediate_irreps

                # Layer 1: project + gate -> 8x0e + 4x1e
                gate1_scalars = Irreps("8x0e")
                gate1_gates = Irreps("4x0e")  # one gate scalar per 1e channel
                gate1_vectors = Irreps("4x1e")
                self.linear1 = Linear(self.combined_irreps, gate1_scalars + gate1_gates + gate1_vectors)
                self.gate1 = Gate(
                    gate1_scalars, [torch.nn.SiLU()],
                    gate1_gates, [torch.sigmoid],
                    gate1_vectors
                )

                # Layer 2: project + gate -> 4x0e + 2x1e
                gate1_out = Irreps("8x0e+4x1e")
                gate2_scalars = Irreps("4x0e")
                gate2_gates = Irreps("2x0e")
                gate2_vectors = Irreps("2x1e")
                self.linear2 = Linear(gate1_out, gate2_scalars + gate2_gates + gate2_vectors)
                self.gate2 = Gate(
                    gate2_scalars, [torch.nn.SiLU()],
                    gate2_gates, [torch.sigmoid],
                    gate2_vectors
                )

                # Layer 3: final linear -> 1x1e (delta_z correction) + 1x0e (log_kappa_raw).
                # Single directional vMF with geometric prior.
                gate2_out = Irreps("4x0e+2x1e")
                self.final_linear = Linear(gate2_out, Irreps("1x1e+1x0e"))
                # e3nn.o3.Linear has no bias, so the 0e (log_kappa) output is purely W @ input
                # and starts near zero. log_conc_init is a fixed buffer giving a kappa~=7.4
                # starting point; the network's 0e output adds to it. It absorbs no gradient,
                # so state-dependent kappa is learned through the network weights.
                self.register_buffer('log_conc_init', torch.zeros(1))
                self._init_weights()

            def _init_weights(self):
                """Init for geometric residual vMF: zero the l=1 output weights so delta_z=0 at
                init (geometric spear prior controls the initial mean direction); scale down the
                l=0 output weights (non-zero) for early state-dependent kappa variation."""
                with torch.no_grad():
                    if self.final_linear.weight is not None:
                        self.final_linear.weight.mul_(0.1)

                        # Zero the l=1 output weights (delta_z=0 at init); essential or the
                        # geometric prior is defeated
                        for ins, w_view in zip(
                            self.final_linear.instructions,
                            self.final_linear.weight_views()
                        ):
                            i_out = ins.i_out
                            ir_out = self.final_linear.irreps_out[i_out].ir
                            if ir_out.l == 1:
                                w_view.zero_()
            
            def forward(self, frame_features: torch.Tensor, 
                       o2_embedding_asymmetric: torch.Tensor) -> torch.Tensor:
                """Both inputs must be in the SAME (aligned) coordinate frame; the caller rotates
                o2_embedding_asymmetric to the aligned frame first.

                Args:
                    frame_features: [B, frame_dim] aligned frame features
                    o2_embedding_asymmetric: [B, o2_dim] O2 asymmetric embedding (aligned)

                Returns:
                    [B, 4]: [:, :3] delta_z (1e, raw correction), [:, 3] log_kappa_raw (0e)
                """
                batch_size = frame_features.shape[0]

                expected_frame_dim = self.frame_irreps.dim
                expected_o2_dim = self.o2_asym_irreps.dim
                
                if frame_features.shape[-1] != expected_frame_dim:
                    raise RuntimeError(
                        f"frame_features dim mismatch: expected {expected_frame_dim}, "
                        f"got {frame_features.shape[-1]}. "
                        f"frame_irreps = {self.frame_irreps}"
                    )
                
                if o2_embedding_asymmetric.shape[-1] != expected_o2_dim:
                    raise RuntimeError(
                        f"o2_embedding_asymmetric dim mismatch: expected {expected_o2_dim}, "
                        f"got {o2_embedding_asymmetric.shape[-1]}. "
                        f"o2_asym_irreps = {self.o2_asym_irreps}"
                    )
                
                frame_proj = self.frame_proj(frame_features)  # [B, intermediate_dim]
                o2_proj = self.o2_proj(o2_embedding_asymmetric)  # [B, intermediate_dim]

                combined = torch.cat([frame_proj, o2_proj], dim=-1)  # [B, 2*intermediate_dim]

                h = self.linear1(combined)
                h = self.gate1(h)
                h = self.linear2(h)
                h = self.gate2(h)
                output = self.final_linear(h)
                # Output 1x1e (delta_z, raw -- geometric prior applied in AxisSamplingManager) + 1x0e
                delta_z = output[:, :3]
                log_kappa_raw = output[:, 3:4] + self.log_conc_init

                return torch.cat([delta_z, log_kappa_raw], dim=-1)  # [B, 4]
        
        return EquivariantAxisHead(frame_irreps, o2_asym_irreps, self.device).to(self.device)

    def _build_axis_network(self):
        """DEPRECATED: Use _build_equivariant_axis_head instead."""
        raise NotImplementedError(
            "_build_axis_network is deprecated. "
            "Use _build_equivariant_axis_head which properly handles irreps."
        )

    def _init_value_network(self):
        """Initialize value MLP weights to prevent variance collapse."""
        modules = [m for m in self.value_mlp.modules() if isinstance(m, nn.Linear)]
        for i, module in enumerate(modules):
            init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                if i == len(modules) - 1:
                    # Final-layer bias = expected return (~1.0); avoids large initial bias-correction
                    init.constant_(module.bias, 1.0)
                else:
                    init.zeros_(module.bias)

    def _validate_irrep_boundaries(self):
        pass 

    def _compute_value(
        self,
        o1_embedding: torch.Tensor,
        o2_embedding: torch.Tensor,
        geometry: torch.Tensor
    , update_stats: bool = False) -> torch.Tensor:
        """Compute state value from raw critic inputs. Must run inside the forward pass (not from
        precomputed critic_feat) so critic_embedder gets gradients during the PPO update.

        Args:
            o1_embedding: [B, D] raw O1 node embedding (D = irreps_atom.dim)
            o2_embedding: [B, D] raw O2 node embedding
            geometry: [B, geom_dim] SO(3)-invariant geometry features

        Returns:
            [B] value estimates (squeezed from [B, 1])
        """
        if o1_embedding is None or o2_embedding is None or geometry is None:
            raise RuntimeError(
                "FAIL: _compute_value requires o1_embedding, o2_embedding, geometry. "
                "All three must be provided."
            )
        
        if o1_embedding.dim() == 1:
            o1_embedding = o1_embedding.unsqueeze(0)
        if o2_embedding.dim() == 1:
            o2_embedding = o2_embedding.unsqueeze(0)
        if geometry.dim() == 1:
            geometry = geometry.unsqueeze(0)
        
        # Move to device if needed
        o1_embedding = o1_embedding.to(self.device)
        o2_embedding = o2_embedding.to(self.device)
        geometry = geometry.to(self.device)
        
        # critic_embedder produces SO(3)-invariant scalar features
        if self.geometry_only_critic:
            sym_l0 = (0.5 * (o1_embedding + o2_embedding))[:, :128].detach()
            critic_feat = torch.cat([geometry, sym_l0], dim=-1)            
        else:
            critic_feat = self.critic_embedder(o1_embedding, o2_embedding, geometry)
            # Stage running-stats update before normalizing, only from the rollout path.
            # The PPO update path passes update_stats=False so stats stay fixed across epochs.
            if update_stats:
                self.critic_input_normalizer.stage_update(critic_feat.detach())
            critic_feat = self.critic_input_normalizer(critic_feat)

        value = self.value_mlp(critic_feat).squeeze(-1)  # [B]

        # Per-step critic diagnostic
        if not hasattr(self, '_critic_call_count'):
            self._critic_call_count = 0
        self._critic_call_count += 1
        if self._critic_call_count % 1 == 0 and value.numel() == 1:
            g = geometry[0] if geometry.dim() > 1 else geometry
            _LOG.info(f"[CRITIC] call={self._critic_call_count} "
                  f"V={value.item():.4f} "
                  f"geom=[d_OO={g[0]:.2f}, dmin_Si={g[1]:.2f}, "
                  f"dmin_COM={g[2]:.2f}, dmin_any={g[3]:.2f}] "
                  f"feat_norm={critic_feat.norm().item():.2f} "
                  f"feat_std={critic_feat.std().item():.4f}")

        return value

    def forward(
        self,
        state: torch.Tensor,
        episode: int,
        training: bool,
        sampled_direction: Optional[torch.Tensor] = None,
        direction_indices: Optional[torch.Tensor] = None,
        o2_embedding_asymmetric: Optional[torch.Tensor] = None,
        state_uid: Optional[str] = None,
        o2_positions: Optional[torch.Tensor] = None,
        neighbor_positions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        o1_embedding: Optional[torch.Tensor] = None,
        o2_embedding: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        env_multipole: Optional[torch.Tensor] = None,
    ):
        batch_size = state.shape[0]

        if o2_positions is None:
            o2_positions = extract_o2_positions(state)

        # Optionally zero backbone l=1,l=2 blocks. Layout: l=0 [0:128], l=1 [128:320], l=2 [320:480]
        if self.zero_backbone_l1l2:
            state = state.clone()
            state[:, 128:480] = 0.0
        # Inject environmental multipole into state (additive, equivariant)
        if env_multipole is not None:
            multipole_signal = self.multipole_injector(env_multipole.to(state.device))
            state = state + multipole_signal        

        # Sample Direction
        dist, channel_weights, sh_coeffs, head_outputs = \
            self.direction_manager.build_distribution(
                state, episode, for_sampling=training, state_uid=state_uid,
                o2_positions=o2_positions
            )
        # Store all SH head outputs for visualization
        self._last_outputs['channel_log_mass_normalized'] = head_outputs['channel_log_mass_normalized']
        self._last_outputs['mixture_log_mass_normalized'] = head_outputs['mixture_log_mass_normalized']
        self._last_outputs['sh_coeffs'] = sh_coeffs
        self._last_outputs['channel_weights'] = channel_weights
        self._last_outputs['betas'] = head_outputs.get('betas')
        self._last_outputs['grid_points'] = head_outputs['grid_points']
        self._last_outputs['grid_weights'] = head_outputs['grid_weights']
        self._last_outputs['grid_n_beta'] = head_outputs['grid_n_beta']
        self._last_outputs['grid_n_alpha'] = head_outputs['grid_n_alpha']

        sampled_indices_local = None 

        if sampled_direction is None:
            use_langevin = getattr(self.config.model, 'use_langevin_direction', False)
            use_continuous = getattr(self.config.model, 'use_continuous_direction', False)
            if use_langevin and not deterministic:
                # Langevin-on-sphere: gradient-guided MCMC sampling
                n_steps = getattr(self.config.model, 'langevin_steps', 10)
                step_size = getattr(self.config.model, 'langevin_step_size', 0.1)
                noise_scale = getattr(self.config.model, 'langevin_noise_scale', 0.05)
                sampled_direction, direction_log_prob = self.sh_head.langevin_sample_direction(
                    channel_weights, n_steps=n_steps, step_size=step_size,
                    noise_scale=noise_scale, deterministic=False)
                sampled_indices = None
                sampled_indices_local = None
                sampled_direction_feat = sampled_direction
            elif use_continuous and not deterministic:
                # Continuous density: sample from grid, perturb, evaluate continuously
                sampled_direction, _, sampled_indices = \
                    self.direction_manager.sample_direction(dist, channel_weights, deterministic=False)
                perturb_std = getattr(self.config.model, 'direction_perturb_std', 0.1)
                sampled_direction = self.sh_head.perturb_on_tangent_plane(sampled_direction, std=perturb_std)
                direction_log_prob = self.sh_head.continuous_log_prob(
                    sampled_direction, channel_weights)
                sampled_indices = None
                sampled_indices_local = None
                sampled_direction_feat = sampled_direction
            elif direction_indices is not None:
                sampled_direction, direction_log_prob, sampled_indices, sampled_indices_local = \
                    self.direction_manager.sample_direction_deterministic(
                        state, direction_indices, episode, o2_positions
                    )
            else:
                sampled_direction, direction_log_prob, sampled_indices = \
                    self.direction_manager.sample_direction(dist, channel_weights, deterministic=deterministic)
                sampled_indices_local = sampled_indices
                sampled_direction_feat = sampled_direction

            # Verify log_prob consistency between sample and lookup
            if os.environ.get('DEBUG_DIRECTION_CONSISTENCY') == '1' and sampled_indices is not None:
                lp_lookup = dist.log_prob_from_indices(sampled_indices, channel_weights)
                max_diff = (direction_log_prob - lp_lookup).abs().max().item()
                if max_diff > 1e-4:
                    _LOG.debug(f" DIRECTION LOGP MISMATCH: max|lp_sample - lp_lookup|={max_diff:.2e}")
                    _LOG.debug(f" lp_sample: {direction_log_prob[:3].tolist()}")
                    _LOG.debug(f" lp_lookup: {lp_lookup[:3].tolist()}")
                elif self._step_counter % 100 == 0:
                    _LOG.debug(f" Direction logp consistent: max_diff={max_diff:.2e}")

            self._last_outputs['sampled_indices'] = sampled_indices
        else:
            # Forced direction mode (e.g. for testing)
            sampled_direction_feat = sampled_direction
            use_continuous = getattr(self.config.model, 'use_continuous_direction', False)
            use_langevin = getattr(self.config.model, 'use_langevin_direction', False)
            if training:
                if use_continuous or use_langevin:
                    direction_log_prob = self.sh_head.continuous_log_prob(
                        sampled_direction, channel_weights)
                else:
                    direction_log_prob = dist.log_prob_from_indices(sampled_indices, channel_weights)
            else:
                direction_log_prob = torch.zeros(sampled_direction_feat.shape[0], device=self.device)

            # DEBUG: Verify log_prob consistency (catches remaining mismatches)
            if training and os.environ.get('DEBUG_DIRECTION_CONSISTENCY') == '1' and sampled_indices is not None:
                lp_verify = dist.log_prob_from_indices(sampled_indices, channel_weights)
                max_diff = (direction_log_prob - lp_verify).abs().max().item()                
                if max_diff > 1e-5:
                    _LOG.debug(f"\nWARNING  DIRECTION LOGP MISMATCH in act()!")
                    _LOG.debug(f"    max|computed - lookup| = {max_diff:.2e}")
                    if os.environ.get('STRICT_DIRECTION_CONSISTENCY') == '1':
                        raise RuntimeError(f"Direction log_prob mismatch: {max_diff:.2e}")
                elif hasattr(self, '_consistency_step') and self._consistency_step % 200 == 0:
                    _LOG.debug(f"OK Direction logp consistent: diff={max_diff:.2e}")
                
                if not hasattr(self, '_consistency_step'):
                    self._consistency_step = 0
                self._consistency_step += 1

            self._last_outputs['sampled_indices'] = sampled_indices

        direction_entropy = self.direction_manager.compute_entropy(dist, channel_weights)
        self._last_outputs.update({'direction_log_prob': direction_log_prob, 
                                   'log_prob_direction': direction_log_prob,
                                   'direction_entropy': direction_entropy})

        sampled_direction_world = F.normalize(sampled_direction_feat, dim=-1, eps=1e-8)

        # DIRECTION DIAGNOSTIC: Does embedding encode substrate direction?
        if not hasattr(self, '_dir_diag_count'):
            self._dir_diag_count = 0
        self._dir_diag_count += 1

        if self._dir_diag_count % 1 == 0:
            with torch.no_grad():
                # 1. SYM l=1 vectors: do they encode direction?
                # state is the SYM embedding [B, 480]
                # irreps: 128x0e (dims 0:128) + 64x1e (dims 128:320) + 32x2e (dims 320:480)
                _sym = state[0]  # first batch element
                _l1_block = _sym[128:320].reshape(64, 3)  # [64, 3] -- 64 l=1 vectors
                _l1_mean = _l1_block.mean(dim=0)  # [3] -- mean l=1 vector
                _l1_norm = _l1_mean.norm().item()
                _l1_dir = _l1_mean / (_l1_mean.norm() + 1e-8)

                # Per-vector norms (are l=1 vectors large or tiny?)
                _l1_per_vec_norms = _l1_block.norm(dim=-1)  # [64]
                _l0_block = _sym[0:128]  # 128 scalars
                _l0_norm = _l0_block.norm().item()

                # 1b. Check RAW (pre-normalization) embedding if available
                # The state passed here is NORMALIZED. We need to check
                # if the normalizer is crushing the l=1 signal.
                if hasattr(self, '_last_raw_sym') and self._last_raw_sym is not None:
                    _raw = self._last_raw_sym[0]
                    _raw_l1 = _raw[128:320].reshape(64, 3)
                    _raw_l1_mean = _raw_l1.mean(dim=0)
                    _raw_l1_norm = _raw_l1_mean.norm().item()
                    _raw_l1_dir = _raw_l1_mean / (_raw_l1_mean.norm() + 1e-8)
                    _raw_l1_per_vec = _raw_l1.norm(dim=-1)
                    _raw_l0_norm = _raw[0:128].norm().item()
                    _LOG.debug(f"  RAW (pre-norm) l=0 norm={_raw_l0_norm:.4f} | "
                          f"l=1 per-vec: mean={_raw_l1_per_vec.mean().item():.4f} "
                          f"max={_raw_l1_per_vec.max().item():.4f}")
                    _LOG.debug(f"  RAW l=1 MEAN vec=[{_raw_l1_dir[0].item():.3f}, {_raw_l1_dir[1].item():.3f}, {_raw_l1_dir[2].item():.3f}] "
                          f"norm={_raw_l1_norm:.4f}")
                else:
                    _LOG.debug(f"  RAW embedding not captured -- add _last_raw_sym storage")

                # 2. SH coefficients: l=0 vs l=1 vs higher
                _sh = sh_coeffs[0] if sh_coeffs.dim() > 1 else sh_coeffs  # [36] or [C, 36]
                if _sh.dim() > 1:
                    _sh = _sh[0]  # first channel
                _sh_l0 = _sh[0].item()                    # 1 coeff
                _sh_l1 = _sh[1:4]                          # 3 coeffs
                _sh_l1_norm = _sh_l1.norm().item()
                _sh_l2_norm = _sh[4:9].norm().item()       # 5 coeffs
                _sh_l3_norm = _sh[9:16].norm().item()      # 7 coeffs
                _sh_higher = _sh[16:].norm().item()        # l=4,5

                # 3. Sampled direction components
                _dir = sampled_direction_world[0]
                _dir_z = _dir[2].item()

                # 4. Beta
                _betas = self._last_outputs.get('betas', None)
                _beta_val = _betas[0, 0].item() if _betas is not None and _betas.numel() > 0 else -1.0

                _LOG.debug(f"\n[DIR_DIAG] step={self._dir_diag_count}")
                _LOG.debug(f"  SYM l=0 norm={_l0_norm:.4f} | "
                      f"l=1 per-vec norm: mean={_l1_per_vec_norms.mean().item():.4f} "
                      f"max={_l1_per_vec_norms.max().item():.4f}")
                _LOG.debug(f"  SYM l=1 MEAN vec=[{_l1_dir[0].item():.3f}, {_l1_dir[1].item():.3f}, {_l1_dir[2].item():.3f}] "
                      f"norm={_l1_norm:.4f}")
                _LOG.info(f"  SH coeffs: l0={_sh_l0:.4f} | l1={_sh_l1_norm:.4f} | "
                      f"l2={_sh_l2_norm:.4f} | l3={_sh_l3_norm:.4f} | l4+5={_sh_higher:.4f}")
                # Track SH energy distribution (% in each l)
                _sh_total_sq = _sh_l0**2 + _sh_l1_norm**2 + _sh_l2_norm**2 + _sh_l3_norm**2 + _sh_higher**2
                _pct_l01 = 100 * (_sh_l0**2 + _sh_l1_norm**2) / (_sh_total_sq + 1e-8)
                _pct_l45 = 100 * _sh_higher**2 / (_sh_total_sq + 1e-8)

                _LOG.debug(f"  Sampled dir=[{_dir[0].item():.3f}, {_dir[1].item():.3f}, {_dir[2].item():.3f}] "
                      f"z={_dir_z:.3f} | beta={_beta_val:.2f}")

                # Multipole signal check
                if hasattr(self, 'multipole_injector') and env_multipole is not None:
                    _mp = env_multipole[0]
                    _mp_l0 = _mp[0].item()
                    _mp_l1 = _mp[1:4]
                    _mp_l1_norm = _mp_l1.norm().item()
                    _mp_l1_dir = _mp_l1 / (_mp_l1.norm() + 1e-8)
                    _mp_l2_norm = _mp[4:9].norm().item()
                    _inj = multipole_signal[0] if 'multipole_signal' in dir() else None
                    _inj_l1_norm = 0.0
                    if _inj is not None:
                        _inj_l1 = _inj[128:320].reshape(64, 3)
                        _inj_l1_norm = _inj_l1.norm(dim=-1).mean().item()
                    _LOG.debug(f"  MULTIPOLE: l0={_mp_l0:.4f} | l1 dir=[{_mp_l1_dir[0].item():.3f}, "
                          f"{_mp_l1_dir[1].item():.3f}, {_mp_l1_dir[2].item():.3f}] norm={_mp_l1_norm:.4f} | "
                          f"l2={_mp_l2_norm:.4f} | injected l1 norm={_inj_l1_norm:.4f}")
                else:
                    _LOG.debug(f"  MULTIPOLE: not available")

                # First-layer directional signal check
                if hasattr(self, '_first_layer_sym') and self._first_layer_sym is not None:
                    _fl = self._first_layer_sym[0] if self._first_layer_sym.dim() > 1 else self._first_layer_sym
                    _fl_l1 = _fl[128:320].reshape(64, 3)
                    _fl_l1_mean = _fl_l1.mean(dim=0)
                    _fl_l1_norm = _fl_l1_mean.norm().item()
                    _fl_l1_pvn = _fl_l1.norm(dim=-1)
                    _fl_l1_dir = _fl_l1_mean / (_fl_l1_mean.norm() + 1e-8)
                    _LOG.debug(f"  FIRST-LAYER l=1: per-vec norm mean={_fl_l1_pvn.mean().item():.4f} max={_fl_l1_pvn.max().item():.4f} | "
                          f"MEAN vec=[{_fl_l1_dir[0].item():.3f}, {_fl_l1_dir[1].item():.3f}, {_fl_l1_dir[2].item():.3f}] "
                          f"norm={_fl_l1_norm:.4f}")
                else:
                    _LOG.debug(f"  FIRST-LAYER: not captured yet")
                _LOG.debug(f"  SH energy: l0+l1={_pct_l01:.1f}% | l4+l5={_pct_l45:.1f}% | "
                      f"{'DIRECTIONAL' if _pct_l01 > 40 else 'HIGH-FREQ LOCKED'}")

                # Track per-step z statistics for episode outcome correlation
                if not hasattr(self, '_episode_z_dirs'):
                    self._episode_z_dirs = []
                self._episode_z_dirs.append(_dir_z)


        # Frame Alignment
        R_feat_to_world_equivariant = self.frame_encoder.compute_feature_to_world_rotation(
            v_hat_world=sampled_direction_world,
            o2_positions=o2_positions
        )

        # Feature Extraction
        frame_encoder_output = self.frame_encoder(
            state,
            direction=sampled_direction_world,
            o2_positions=o2_positions
        )
        frame_features = frame_encoder_output['frame_aligned_features']
        R_feat_to_world = frame_encoder_output.get('R_feat_to_world', R_feat_to_world_equivariant)

        if R_feat_to_world is not None:
            self.sh_head.set_feature_to_world_transform(R_feat_to_world)

        # Neighbors & Magnitude
        neighbor_positions_feat = None
        o2_com_feat = None
        
        if neighbor_positions is None:
            if state.shape[-1] > 480:
                num_neighbor_features = state.shape[-1] - 6
                num_neighbors = num_neighbor_features // 3
                neighbor_positions = state[:, 6:].reshape(batch_size, num_neighbors, 3)
            else:
                neighbor_positions = None
        
        if neighbor_positions is not None:
            o2_com = o2_positions.mean(dim=1)
            R_feat_to_world_cast = R_feat_to_world.to(dtype=neighbor_positions.dtype)
            neighbor_positions_feat = torch.bmm(neighbor_positions, R_feat_to_world_cast)
            o2_com_feat = torch.bmm(o2_com.unsqueeze(1), R_feat_to_world_cast).squeeze(1)
        
        elif o2_positions is not None:
            o2_com = o2_positions.mean(dim=1)
            R_feat_to_world_cast = R_feat_to_world.to(dtype=o2_positions.dtype)
            o2_com_feat = torch.bmm(o2_com.unsqueeze(1), R_feat_to_world_cast).squeeze(1)

        dir_aligned = torch.zeros(batch_size, 3, device=self.device, dtype=frame_features.dtype)
        dir_aligned[:, 2] = 1.0

        # Call Magnitude Predictor
        trans_magnitude, magnitude_log_prob = self.magnitude_predictor(
            frame_features,             
            direction=dir_aligned,
            neighbor_positions=neighbor_positions_feat, 
            o2_com=o2_com_feat,                           
            temperature=self.sh_head.current_temperature,
            return_log_prob=True,
            deterministic=deterministic
        )

        self._last_outputs['magnitude_log_prob'] = magnitude_log_prob
        self._last_outputs['log_prob_magnitude'] = magnitude_log_prob

        # Store magnitude features for entropy computation in act()
        directional_scalars = self.magnitude_predictor._extract_directional_scalars(frame_features)
        if self.magnitude_predictor.use_bessel and neighbor_positions_feat is not None:
            bessel_coeffs = self.magnitude_predictor._compute_bessel_embedding(
                dir_aligned, neighbor_positions_feat, o2_com_feat
            )
            magnitude_features_stored = torch.cat([directional_scalars, bessel_coeffs], dim=-1)
        else:
            magnitude_features_stored = directional_scalars

        # Dimension validation/padding (same as magnitude_predictor.forward())
        if hasattr(self.magnitude_predictor, 'rqs_flow') and self.magnitude_predictor.use_rqs_flow:
            if magnitude_features_stored.shape[-1] != self.magnitude_predictor.rqs_flow.scalar_dim:
                current_dim = magnitude_features_stored.shape[-1]
                target_dim = self.magnitude_predictor.rqs_flow.scalar_dim

                if current_dim > target_dim:
                    magnitude_features_stored = magnitude_features_stored[:, :target_dim]
                elif current_dim < target_dim:
                    padding = torch.zeros(batch_size, target_dim - current_dim, device=self.device)
                    magnitude_features_stored = torch.cat([magnitude_features_stored, padding], dim=-1)

        self._last_outputs['magnitude_features'] = magnitude_features_stored

        # Save magnitude visualization data
        with torch.no_grad():
            mag_viz = self.magnitude_predictor.get_visualization_data(
                frame_features,
                direction=dir_aligned,
                sampled_magnitude=trans_magnitude
            )
            if mag_viz is not None:
                self._last_outputs['magnitude'] = mag_viz

        # Axis Prediction
        o2_embedding_asymmetric_for_axis = o2_embedding_asymmetric
        if o2_embedding_asymmetric_for_axis is not None:
             o2_embedding_aligned = rotate_o2_asymmetric_to_aligned(
                o2_embedding_asymmetric_for_axis, R_feat_to_world, self.device
             )
        else:
             o2_embedding_aligned = None
        # Compute current O-O direction in aligned frame for geometric spear prior
        # Convention: axis_aligned = axis_world @ R_feat_to_world
        o2_dir_world = F.normalize(o2_positions[:, 0] - o2_positions[:, 1], dim=-1)  # [B, 3]
        o2_dir_aligned = torch.bmm(
            o2_dir_world.unsqueeze(1), R_feat_to_world
        ).squeeze(1)  # [B, 3] world -> aligned frame

        axis_params = self.axis_manager.predict_axis_distribution_params(
            frame_features,
            o2_embedding_asymmetric=o2_embedding_aligned,
            o2_dir_aligned=o2_dir_aligned,
        )

        axis_frame, axis_log_prob = self.axis_manager.sample_axis_stochastic(
            axis_params=axis_params, sampled_direction=sampled_direction_world,
            training=training, R_feat_to_world=R_feat_to_world
        )
        
        axis_direction = self.axis_manager.rotate_axis_to_global(axis_frame, R_feat_to_world)
        
        self._last_outputs['axis_log_prob'] = axis_log_prob
        self._last_outputs['log_prob_axis'] = axis_log_prob
        self._last_outputs['log_concentration'] = axis_params['log_concentration']

        # Visualization Data
        with torch.no_grad():
            mean_direction = axis_params['mean_direction']
            log_concentration = axis_params['log_concentration']
            concentration = torch.exp(log_concentration)
            
            # Transform mean_direction from aligned frame to world frame
            if R_feat_to_world is not None:
                mean_direction_world = torch.bmm(
                    mean_direction.unsqueeze(1), 
                    R_feat_to_world.transpose(1, 2)
                ).squeeze(1)
            else:
                mean_direction_world = mean_direction
            
            # Regular vMF: theta = actual angle between sampled axis and mean direction
            # Samples should cluster near mean_direction (theta ~= 0) when kappa is large
            cos_theta = (axis_direction * mean_direction_world).sum(dim=-1).clamp(-1.0, 1.0)
            theta_vis = torch.arccos(cos_theta)  #  in  [0, pi]
            kappa_vis = min(concentration[0].item(), 100.0)
            w_vis = 0.5  # placeholder mixture weight

            # Angle between mu and v_hat (both world frame)
            mu_w = mean_direction_world[0].detach().cpu().numpy()
            vhat_w = sampled_direction_world[0].detach().cpu().numpy()
            mu_w = mu_w / (np.linalg.norm(mu_w) + 1e-12)
            vhat_w = vhat_w / (np.linalg.norm(vhat_w) + 1e-12)
            alpha = np.arccos(np.clip(np.dot(mu_w, vhat_w), -1, 1))

            # Bipolar vMF marginal for angle theta from v_hat:
            # p(theta) ~ sin(theta) * I0(kappa*sin(alpha)*sin(theta))
            #            * [w*exp(kappa*cos(alpha)*cos(theta)) + (1-w)*exp(-kappa*cos(alpha)*cos(theta))]
            from scipy.special import i0e
            theta_grid = np.linspace(0, np.pi, 200)
            sin_theta = np.sin(theta_grid)
            cos_theta = np.cos(theta_grid)

            if kappa_vis > 1e-3:
                cos_a = np.cos(alpha)
                sin_a = np.sin(alpha)
                bessel_arg = kappa_vis * sin_a * sin_theta
                log_I0 = np.log(i0e(bessel_arg) + 1e-30) + np.abs(bessel_arg)
                log_sin = np.log(np.clip(sin_theta, 1e-12, None))
                exp_plus = kappa_vis * cos_a * cos_theta
                exp_minus = -kappa_vis * cos_a * cos_theta
                log_weighted = np.logaddexp(
                    np.log(max(w_vis, 1e-6)) + exp_plus,
                    np.log(max(1 - w_vis, 1e-6)) + exp_minus
                )
                log_p_unnorm = log_sin + log_I0 + log_weighted
                log_p_unnorm -= log_p_unnorm.max()
                p_unnorm = np.exp(log_p_unnorm)
                p_norm = p_unnorm / (np.trapz(p_unnorm, theta_grid) + 1e-12)
                log_p_theta = np.log(p_norm + 1e-30)
            else:
                log_p_theta = np.log(np.clip(sin_theta, 1e-12, None)) - np.log(2)

            # theta_sample: angle between sampled axis and v_hat (world frame)
            axis_w = axis_direction[0].detach().cpu().numpy()
            axis_w = axis_w / (np.linalg.norm(axis_w) + 1e-12)
            theta_sample_vhat = np.arccos(np.clip(np.dot(axis_w, vhat_w), -1, 1))

            axis_angle_pdf_dict = {
                'theta_sample': float(theta_sample_vhat),
                'alpha_mu_vhat': float(alpha),
                'theta_mean': float(alpha),
                'theta_std': 1.0 / np.sqrt(kappa_vis + 1e-6),
                'mix_weight': float(w_vis),
                'mean_direction': mu_w,
                'log_concentration': float(log_concentration[0].item()),
                'sampled_axis': axis_frame[0].detach().cpu().numpy(),
                'theta_grid': theta_grid,
                'log_p_theta': log_p_theta,
            }
            self._last_outputs['axis_angle_pdf'] = axis_angle_pdf_dict

        # Value via CriticEmbedding -> value_mlp (gradient flow for PPO)
        state_value = self._compute_value(o1_embedding, o2_embedding, geometry, update_stats=True)
        trans_vector_world = sampled_direction_world * trans_magnitude.unsqueeze(-1)

        self._last_outputs.update({
            'translation_direction': sampled_direction_world,
            'translation_magnitude': trans_magnitude,
            'translation_vector': trans_vector_world,
            'axis_direction': axis_direction,
            'axis_frame': axis_frame,  # ADD: aligned frame axis for exact log_prob replay
            'axis_mean_direction': mean_direction_world,
            'axis_log_concentration': axis_params['log_concentration'],
            'value': state_value,
            'direction_indices': sampled_indices_local
        })
        action_dict = {
            'translation_direction': sampled_direction_world,
            'translation_magnitude': trans_magnitude,
            'translation_vector': trans_vector_world,
            'axis_direction': axis_direction,
            'axis_frame': axis_frame,  # aligned frame axis for exact log_prob replay
            'axis_mean_direction_aligned': axis_params['mean_direction'],  # ALIGNED frame for exact replay
            'axis_mean_direction': mean_direction_world,  # WORLD frame for visualization
            'axis_log_concentration': axis_params['log_concentration'],        
            'value': state_value,
            'direction_indices': sampled_indices_local,
            # Critic raw inputs; buffer stores these and evaluate_actions() recomputes critic_feat
            'o1_embedding': o1_embedding,
            'o2_embedding': o2_embedding,
            'geometry': geometry,
        }

        return action_dict

    def act(
        self,
        states: torch.Tensor,
        episode: int = 0,
        step: int = 0,
        training: bool = True,
        exploration_phase: bool = False,
        o2_embedding_asymmetric: Optional[torch.Tensor] = None,
        neighbor_positions: Optional[torch.Tensor] = None,
        o1_embedding: Optional[torch.Tensor] = None,
        o2_embedding: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        o2_positions: Optional[torch.Tensor] = None,        
        **kwargs,
    ):            
        sampled_direction = kwargs.get('sampled_direction', None)
        direction_indices = kwargs.get('direction_indices', None)
        state_uid = kwargs.get('state_uid', None)

        env_multipole = kwargs.get('env_multipole', None)
        outputs = self.forward(
            state=states,
            episode=episode,
            training=training,
            sampled_direction=sampled_direction,
            direction_indices=direction_indices,
            o2_embedding_asymmetric=o2_embedding_asymmetric,
            state_uid=state_uid,
            neighbor_positions=neighbor_positions,
            o1_embedding=o1_embedding,
            o2_embedding=o2_embedding,
            geometry=geometry,
            o2_positions=o2_positions,
            env_multipole=env_multipole,
        )

        direction_log_prob = self._last_outputs.get('direction_log_prob')
        magnitude_log_prob = self._last_outputs.get('magnitude_log_prob')
        axis_log_prob = self._last_outputs.get('axis_log_prob')
        
        if direction_log_prob is None:
            raise RuntimeError("direction_log_prob not cached")
        
        log_prob_aggregate = direction_log_prob + magnitude_log_prob + axis_log_prob
        self._last_outputs['log_prob_aggregate'] = log_prob_aggregate
        
        action = {
            'translation_vector': outputs['translation_vector'],
            'translation_direction': outputs['translation_direction'],
            'translation_magnitude': outputs['translation_magnitude'],
            'axis_direction': outputs['axis_direction'],
            'axis_frame': outputs['axis_frame'],
            'axis_mean_direction': outputs['axis_mean_direction'],
            'axis_log_concentration': outputs['axis_log_concentration'],
            'direction_indices': outputs['direction_indices'],
            # Critic raw inputs for buffer storage (detached by extractor)
            'o1_embedding': outputs['o1_embedding'],
            'o2_embedding': outputs['o2_embedding'],
            'geometry': outputs['geometry'],
        }

        # Compute total policy entropy (sum of all three heads)
        direction_entropy = self._last_outputs.get('direction_entropy')
        if direction_entropy is None:
            direction_entropy = torch.tensor(0.0, device=self.device)
        
        # Magnitude entropy
        magnitude_features = self._last_outputs.get('magnitude_features')
        if magnitude_features is not None and hasattr(self.magnitude_predictor, 'compute_entropy'):
            temperature = self.sh_head.current_temperature if hasattr(self.sh_head, 'current_temperature') else 1.0
            magnitude_entropy = self.magnitude_predictor.compute_entropy(
                magnitude_features, temperature
            ).mean()  # Average over batch
        else:
            magnitude_entropy = torch.tensor(0.0, device=self.device)
        
        # Axis entropy
        axis_params = {
            'log_concentration': self._last_outputs.get('log_concentration')
        }
        if axis_params['log_concentration'] is not None and hasattr(self.axis_manager, 'compute_entropy'):
            axis_entropy = self.axis_manager.compute_entropy(axis_params).mean()  # Average over batch
        else:
            axis_entropy = torch.tensor(0.0, device=self.device)
        
        # Total entropy = sum of all heads
        total_entropy = direction_entropy + magnitude_entropy + axis_entropy
        
        # Store components for debugging
        self._last_outputs['magnitude_entropy'] = magnitude_entropy
        self._last_outputs['axis_entropy'] = axis_entropy
        self._last_outputs['total_entropy'] = total_entropy
        
        value = outputs['value']
        return action, log_prob_aggregate, value, total_entropy

    def get_value(
        self,
        o1_embedding: torch.Tensor,
        o2_embedding: torch.Tensor,
        geometry: torch.Tensor
    ) -> torch.Tensor:
        """Public V(s) interface for value prediction without a full forward pass (e.g. truncation
        bootstrapping). Caller provides raw critic inputs from
        EmbeddingExtractor.get_equivariant_embedding().

        Args:
            o1_embedding: [B, D] O1 node embedding
            o2_embedding: [B, D] O2 node embedding
            geometry: [B, geom_dim] SO(3)-invariant geometry features

        Returns:
            Value estimates [B]
        """
        with torch.no_grad():
            return self._compute_value(o1_embedding, o2_embedding, geometry, update_stats=True)

