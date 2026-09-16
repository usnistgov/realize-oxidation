"""
evaluation/deterministic_action.py

Action selection for evaluation runs.

This module implements two action-selection paths through PolicyNetwork's
trained components, sharing 99% of the code:

    deterministic=True   ->  pick the MODE of every head:
        - Direction: argmax over the 4096-point GL grid
        - Magnitude: push z = base_mu (truncated-Gaussian mode) through the RQS
        - Axis:      mu_b_hat from the geometric residual vMF, rotated to world

    deterministic=False  ->  SAMPLE from each head, exactly as training does:
        - Direction: categorical draw on the GL grid (per build_distribution)
        - Magnitude: RQS forward (base sample -> spline pipeline)
        - Axis:      vMF Householder sample around mu_b_hat

Why both paths in one function?
    Eval runs deterministic=True on the FIRST attempt of every step. If the
    environment rejects the proposed move (e.g., NEB fails), the env's
    current_try_num counter increments. On retry attempts, we switch to
    deterministic=False so that fresh sampling can find a different proposed
    move -- exactly mirroring how training escapes failed steps. Without this,
    a deterministic run can get stuck repeatedly proposing the same losing
    move at the same state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from realize.networks.distributions.policy_utils.policy_helpers import extract_o2_positions
from realize.networks.heads.sh_head_utils.sh_head_layers import rotate_o2_asymmetric_to_aligned


@torch.no_grad()
def _rqs_forward_at_z(rqs_flow, z: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
    """
    Push a base-space point z through the RQS layer stack in inverse order
    (matches BoundedRQSFlow.forward()'s training-time path), with NO base
    sampling and NO exploration noise.

    z must be in [r_min, r_max]. For the truncated-Gaussian base, the mode
    is base_mu (interior to the truncation interval given the trained config).
    """
    x = z
    for i in reversed(range(rqs_flow.num_layers)):
        widths, heights, derivatives = rqs_flow._get_spline_params(context_features, i)
        x, _log_det = rqs_flow._inverse_rqs(x, widths, heights, derivatives)
    x = torch.clamp(
        x,
        rqs_flow.r_min + rqs_flow.boundary_error,
        rqs_flow.r_max - rqs_flow.boundary_error,
    )
    return x


@torch.no_grad()
def select_action_for_eval(
    policy,
    state: torch.Tensor,
    *,
    o2_positions: torch.Tensor,
    o2_embedding_asymmetric: torch.Tensor,
    neighbor_positions: Optional[torch.Tensor],
    env_multipole: Optional[torch.Tensor],
    o1_embedding: torch.Tensor,
    o2_embedding: torch.Tensor,
    geometry: torch.Tensor,
    episode: int = 0,
    deterministic: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Single forward pass through the trained policy, selecting an action either
    by mode (deterministic=True) or by sampling (deterministic=False).

    All inputs must be on policy.device with batch dim = 1.

    Returns:
        (action_dict, diagnostics)
        action_dict: keys consumed by step_environment_with_rollback
        diagnostics: per-step scalars for logging
    """
    if state.shape[0] != 1:
        raise RuntimeError(f"Eval is single-batch only; got state batch={state.shape[0]}")
    device = policy.device
    batch_size = 1

    # ---- 1. Multipole injection (mirrors PolicyNetwork.forward()) ----
    state_in = state
    if policy.zero_backbone_l1l2:
        state_in = state_in.clone()
        state_in[:, 128:480] = 0.0
    if env_multipole is not None:
        multipole_signal = policy.multipole_injector(env_multipole.to(device))
        state_in = state_in + multipole_signal

    # ---- 2. Build direction distribution ----
    dist, channel_weights, sh_coeffs, head_outputs = (
        policy.direction_manager.build_distribution(
            state_in,
            episode=episode,
            for_sampling=True,
            state_uid=None,
            o2_positions=o2_positions,
        )
    )

    # ---- 2b. Sample direction (argmax if deterministic, else categorical draw) ----
    # The direction_manager.sample_direction() handles both branches via
    # its `deterministic` flag -- argmax over GL grid vs categorical sampling
    # via the head's sample_direction() with full categorical RNG.
    sampled_direction, direction_log_prob, sampled_indices = (
        policy.direction_manager.sample_direction(
            dist, channel_weights, deterministic=deterministic
        )
    )
    sampled_direction_world = F.normalize(sampled_direction, dim=-1, eps=1e-8)

    # ---- 3. Frame alignment (identical to forward()) ----
    R_feat_to_world_eq = policy.frame_encoder.compute_feature_to_world_rotation(
        v_hat_world=sampled_direction_world,
        o2_positions=o2_positions,
    )
    frame_encoder_output = policy.frame_encoder(
        state_in,
        direction=sampled_direction_world,
        o2_positions=o2_positions,
    )
    frame_features = frame_encoder_output["frame_aligned_features"]
    R_feat_to_world = frame_encoder_output.get("R_feat_to_world", R_feat_to_world_eq)
    if R_feat_to_world is not None:
        policy.sh_head.set_feature_to_world_transform(R_feat_to_world)

    # ---- 4. Neighbor / COM in feature frame for Bessel embedding ----
    neighbor_positions_feat = None
    o2_com_feat = None
    if neighbor_positions is None and state_in.shape[-1] > 480:
        num_neighbor_features = state_in.shape[-1] - 6
        num_neighbors = num_neighbor_features // 3
        neighbor_positions = state_in[:, 6:].reshape(batch_size, num_neighbors, 3)

    if neighbor_positions is not None:
        o2_com = o2_positions.mean(dim=1)
        R_cast = R_feat_to_world.to(dtype=neighbor_positions.dtype)
        neighbor_positions_feat = torch.bmm(neighbor_positions, R_cast)
        o2_com_feat = torch.bmm(o2_com.unsqueeze(1), R_cast).squeeze(1)
    elif o2_positions is not None:
        o2_com = o2_positions.mean(dim=1)
        R_cast = R_feat_to_world.to(dtype=o2_positions.dtype)
        o2_com_feat = torch.bmm(o2_com.unsqueeze(1), R_cast).squeeze(1)

    # In the aligned frame, the direction is +z by construction.
    dir_aligned = torch.zeros(batch_size, 3, device=device, dtype=frame_features.dtype)
    dir_aligned[:, 2] = 1.0

    # ---- 5. Magnitude ----
    mp = policy.magnitude_predictor
    if not mp.use_rqs_flow:
        raise RuntimeError(
            "Eval expects RQS flow magnitude predictor (model.rqs_flow_enabled=True). "
            "Clamped-MLP path is not supported in eval."
        )
    directional_scalars = mp._extract_directional_scalars(frame_features)
    if mp.use_bessel and neighbor_positions_feat is not None:
        bessel_coeffs = mp._compute_bessel_embedding(
            dir_aligned, neighbor_positions_feat, o2_com_feat
        )
        rqs_features = torch.cat([directional_scalars, bessel_coeffs], dim=-1)
    else:
        rqs_features = directional_scalars

    target_dim = mp.rqs_flow.scalar_dim
    cur_dim = rqs_features.shape[-1]
    if cur_dim > target_dim:
        rqs_features = rqs_features[:, :target_dim]
    elif cur_dim < target_dim:
        pad = torch.zeros(batch_size, target_dim - cur_dim, device=device, dtype=rqs_features.dtype)
        rqs_features = torch.cat([rqs_features, pad], dim=-1)

    if deterministic:
        # Mode: push z = base_mu through the inverse-RQS chain.
        z_mode = torch.full(
            (batch_size,), float(mp.rqs_flow.base_mu),
            device=device, dtype=rqs_features.dtype,
        )
        trans_magnitude = _rqs_forward_at_z(mp.rqs_flow, z_mode, rqs_features).view(batch_size)
    else:
        # Sample: use the RQS flow's own forward() -- base draw -> inverse pipeline.
        # This matches training's stochastic path. policy.eval() disables the
        # in-training exploration noise inside BoundedRQSFlow.forward(), so we
        # get clean flow samples (no extra Gaussian on top).
        sampled_mag = mp.rqs_flow(rqs_features, return_log_prob=False)
        trans_magnitude = sampled_mag.view(batch_size)

    # ---- 6. Axis ----
    if o2_embedding_asymmetric is None:
        raise RuntimeError("Eval requires o2_embedding_asymmetric for axis prediction.")
    o2_embedding_aligned = rotate_o2_asymmetric_to_aligned(
        o2_embedding_asymmetric, R_feat_to_world, device
    )
    o2_dir_world = F.normalize(o2_positions[:, 0] - o2_positions[:, 1], dim=-1)
    o2_dir_aligned = torch.bmm(o2_dir_world.unsqueeze(1), R_feat_to_world).squeeze(1)

    axis_params = policy.axis_manager.predict_axis_distribution_params(
        frame_features,
        o2_embedding_asymmetric=o2_embedding_aligned,
        o2_dir_aligned=o2_dir_aligned,
    )
    mean_direction_aligned = axis_params["mean_direction"]  # [B, 3] aligned frame
    log_concentration = axis_params["log_concentration"]    # [B] log-kappa

    if deterministic:
        # Mode: vMF mean direction (mu_b_hat in aligned frame).
        axis_aligned = mean_direction_aligned
    else:
        # Sample from vMF via the policy's existing Householder sampler.
        axis_aligned, _axis_log_prob = policy.axis_manager.sample_axis_stochastic(
            axis_params=axis_params,
            sampled_direction=sampled_direction_world,
            training=True,                       # forces sampling
            R_feat_to_world=R_feat_to_world,
        )

    # Rotate axis from aligned to world frame.
    axis_direction_world = torch.bmm(
        axis_aligned.unsqueeze(1),
        R_feat_to_world.transpose(1, 2),
    ).squeeze(1)
    axis_direction_world = F.normalize(axis_direction_world, dim=-1, eps=1e-8)

    # ---- 7. Translation vector + value (value logged only) ----
    trans_vector_world = sampled_direction_world * trans_magnitude.unsqueeze(-1)
    value = policy._compute_value(
        o1_embedding, o2_embedding, geometry, update_stats=False
    )

    # ---- 8. Action dict -- keys consumed by step_environment_with_rollback ----
    action_dict = {
        "translation_direction": sampled_direction_world,
        "translation_magnitude": trans_magnitude,
        "translation_vector": trans_vector_world,
        "axis_direction": axis_direction_world,
        "axis_frame": axis_aligned,
        "axis_mean_direction_aligned": mean_direction_aligned,
        "axis_mean_direction": axis_direction_world,
        "axis_log_concentration": log_concentration,
        "value": value,
        "direction_indices": sampled_indices,
        "o1_embedding": o1_embedding,
        "o2_embedding": o2_embedding,
        "geometry": geometry,
    }

    diagnostics = {
        "deterministic": deterministic,
        "direction_log_prob": float(direction_log_prob.detach().cpu().item()),
        "magnitude": float(trans_magnitude.detach().cpu().item()),
        "log_kappa": float(log_concentration.detach().cpu().item()),
        "value": float(value.detach().cpu().item()),
        "grid_idx": int(sampled_indices.detach().cpu().item()) if sampled_indices is not None else -1,
    }

    return action_dict, diagnostics


# Backward-compatible alias.
deterministic_act = select_action_for_eval
