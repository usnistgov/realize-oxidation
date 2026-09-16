import math
import torch
import torch.nn.functional as F
import numpy as np

from realize.networks.distributions.policy_utils.policy_helpers import extract_o2_positions
from realize.networks.heads.sh_head_utils.sh_head_layers import rotate_o2_asymmetric_to_aligned
import logging
_LOG = logging.getLogger(__name__)


class PolicyEvaluateMixin:
    """PPO-update evaluation path for PolicyNetwork.

    ``evaluate`` re-runs the policy on stored states to recompute log-probs /
    entropy / values during the PPO update. It creates no nn.Parameters and runs
    only on the update path, so it does not affect PolicyNetwork's forward
    fingerprint (44 tensors / 75978 scalars), which is fixed by __init__ and the
    construction helpers. Submodules / config / device resolve through the MRO.
    """
    def evaluate(self, states, episode_idx, old_actions=None, state_uid=None,
                 o2_embeddings_asymmetric=None,
                 o1_embeddings=None, o2_embeddings=None, geometries=None,
                 stored_temperatures=None,
                 o2_positions=None,
                 env_multipoles=None):            
        """Evaluate actions under the current policy for the PPO update. Uses the actual steered
        direction from old_actions for frame alignment."""
        batch_size = states.shape[0]

        if not hasattr(self, '_axis_frame_avail_debug'):
            self._axis_frame_avail_debug = 0
        if not hasattr(self, '_axis_params_debug'):
            self._axis_params_debug = 0
        if not hasattr(self, '_axis_frame_debug_count'):
            self._axis_frame_debug_count = 0

        if o1_embeddings is None or o2_embeddings is None or geometries is None:
            raise RuntimeError(
                "FAIL: evaluate() requires o1_embeddings, o2_embeddings, geometries "
                "for critic value computation."
            )

        # Normalize critic raw inputs to correct shape and device
        if isinstance(o1_embeddings, list):
            o1_embeddings = torch.stack(o1_embeddings, dim=0)
        o1_embeddings = o1_embeddings.to(self.device)
        if o1_embeddings.dim() == 3 and o1_embeddings.shape[1] == 1:
            o1_embeddings = o1_embeddings.squeeze(1)
        
        if isinstance(o2_embeddings, list):
            o2_embeddings = torch.stack(o2_embeddings, dim=0)
        o2_embeddings = o2_embeddings.to(self.device)
        if o2_embeddings.dim() == 3 and o2_embeddings.shape[1] == 1:
            o2_embeddings = o2_embeddings.squeeze(1)
        
        if isinstance(geometries, list):
            geometries = torch.stack(geometries, dim=0)
        geometries = geometries.to(self.device)
        if geometries.dim() == 3 and geometries.shape[1] == 1:
            geometries = geometries.squeeze(1)

        # 1. Sample Direction
        if o2_positions is None:
            o2_positions = extract_o2_positions(states)

        # Stored temperature is required: evaluate() must use the same temp as the rollout
        if stored_temperatures is None:
            raise RuntimeError(
                "stored_temperatures is None in evaluate(). "
                "PPO requires stored_temperatures for correct log-prob computation. "
                "Ensure buffer stores per-transition temperature and passes it to evaluate()."
            )
        
        # Require essentially identical temperature across the minibatch; even small
        # variations cause log-prob drift over many updates
        temp_std = stored_temperatures.std().item()
        temp_mean = stored_temperatures.mean().item()
        temp_range = stored_temperatures.max().item() - stored_temperatures.min().item()

        relative_variation = temp_std / (temp_mean + 1e-8)

        if temp_range > 1e-6:
            raise RuntimeError(
                f"minibatch has mixed temperatures: "
                f"range=[{stored_temperatures.min().item():.6f}, {stored_temperatures.max().item():.6f}], "
                f"std={temp_std:.2e}. "
                f"PPO requires temperature-homogeneous batches for correct log-prob computation. "
                f"Ensure batching groups transitions by episode/temperature."
            )

        eval_temperature = float(stored_temperatures[0].item())

        # Zero backbone l=1,l=2 (must match rollout path)
        if self.zero_backbone_l1l2:
            states = states.clone()
            states[:, 128:480] = 0.0
        if env_multipoles is not None:
            env_mp = env_multipoles.to(states.device) if isinstance(env_multipoles, torch.Tensor) else env_multipoles
            states = states + self.multipole_injector(env_mp)            

        dist, channel_weights, sh_coeffs, head_outputs = \
            self.direction_manager.build_distribution(
                states, episode_idx, for_sampling=True, state_uid=state_uid,
                o2_positions=o2_positions,
                override_temperature=eval_temperature
            )

        # Store SH head outputs for visualization
        self._last_outputs['channel_log_mass_normalized'] = head_outputs['channel_log_mass_normalized']
        self._last_outputs['mixture_log_mass_normalized'] = head_outputs['mixture_log_mass_normalized']
        self._last_outputs['sh_coeffs'] = sh_coeffs
        self._last_outputs['channel_weights'] = channel_weights
        self._last_outputs['betas'] = head_outputs.get('betas')
        self._last_outputs['grid_points'] = head_outputs['grid_points']
        self._last_outputs['grid_weights'] = head_outputs['grid_weights']
        self._last_outputs['grid_n_beta'] = head_outputs['grid_n_beta']
        self._last_outputs['grid_n_alpha'] = head_outputs['grid_n_alpha']

        # 2. Extract the steered direction actually taken
        old_dirs_feat = []
        for act in old_actions:
            d = act.get('translation_direction')
            if d is None: raise RuntimeError("Old action missing translation_direction")
            if isinstance(d, np.ndarray): d = torch.from_numpy(d)
            old_dirs_feat.append(d.to(self.device).float())
        old_dirs_feat = torch.stack(old_dirs_feat).view(batch_size, 3)

        # 3. Direction log_prob
        use_continuous = getattr(self.config.model, 'use_continuous_direction', False)
        use_langevin = getattr(self.config.model, 'use_langevin_direction', False)
        if use_continuous or use_langevin:
            # Evaluate SH at stored direction vectors (no indices)
            log_probs_direction = self.sh_head.continuous_log_prob(
                old_dirs_feat, channel_weights)
            sampled_indices = None
        else:
            sampled_indices_list = []
            for act in old_actions:
                idx = act.get('direction_indices')
                if idx is None: raise RuntimeError("Old action missing direction_indices")
                if hasattr(idx, 'item'): idx = idx.item()
                sampled_indices_list.append(idx)
            sampled_indices = torch.tensor(sampled_indices_list, device=self.device, dtype=torch.long)

        # 4. Frame alignment and features using the steered direction
        if o2_positions is None:
            o2_positions = extract_o2_positions(states)

        enc_output = self.frame_encoder(
            states, 
            direction=old_dirs_feat, 
            o2_positions=o2_positions
        )
        frame_features = enc_output['frame_aligned_features']
        R_feat_to_world = enc_output['R_feat_to_world']
        
        # 5. Recompute direction log-probs
        if not use_continuous and not use_langevin:
            log_probs_direction = dist.log_prob_from_indices(sampled_indices, channel_weights)

        # 6. Recompute magnitude log-probs
        log_probs_magnitude = None
        
        if hasattr(self.magnitude_predictor, 'rqs_flow') and self.magnitude_predictor.use_rqs_flow:
            directional_scalars = self.magnitude_predictor._extract_directional_scalars(frame_features) 
            if states.shape[-1] > 480:
                num_neighbor_features = states.shape[-1] - 6
                num_neighbors = num_neighbor_features // 3
                neighbor_positions = states[:, 6:].reshape(batch_size, num_neighbors, 3)
            else:
                neighbor_positions = None
                
            if self.magnitude_predictor.use_bessel:
                if neighbor_positions is None:
                    bessel_coeffs = torch.zeros(
                        (batch_size, self.magnitude_predictor.num_bessel), 
                        device=self.device, 
                        dtype=frame_features.dtype
                    )                     
                else:
                    o2_com = o2_positions.mean(dim=1)
                    R_cast = R_feat_to_world.to(dtype=neighbor_positions.dtype)
                    neighbor_positions_feat = torch.bmm(neighbor_positions, R_cast)
                    o2_com_feat = torch.bmm(o2_com.unsqueeze(1), R_cast).squeeze(1)
                    
                    dir_aligned = torch.zeros(batch_size, 3, device=self.device, dtype=frame_features.dtype)
                    dir_aligned[:, 2] = 1.0
                    
                    bessel_coeffs = self.magnitude_predictor._compute_bessel_embedding(
                        dir_aligned,  
                        neighbor_positions_feat,
                        o2_com_feat
                    )                    
                
                magnitude_features = torch.cat([directional_scalars, bessel_coeffs], dim=-1)
            else:
                magnitude_features = directional_scalars

            # Dimension validation/padding (matches forward())
            if magnitude_features.shape[-1] != self.magnitude_predictor.rqs_flow.scalar_dim:
                current_dim = magnitude_features.shape[-1]
                target_dim = self.magnitude_predictor.rqs_flow.scalar_dim
                
                if current_dim > target_dim:
                    magnitude_features = magnitude_features[:, :target_dim]
                elif current_dim < target_dim:
                    padding = torch.zeros(batch_size, target_dim - current_dim, device=self.device)
                    magnitude_features = torch.cat([magnitude_features, padding], dim=-1)

            old_mags = []
            for act in old_actions:
                m = act.get('translation_magnitude')
                if hasattr(m, 'item'): m = m.item()
                old_mags.append(float(m))
            
            old_mags_tensor = torch.tensor(old_mags, device=self.device, dtype=torch.float32)
            
            # Use the mixture log-prob (flow + uniform) to match the sampling distribution in act()
            if stored_temperatures is not None:
                magnitude_temperature = float(stored_temperatures.mean().item())
            else:
                magnitude_temperature = self.sh_head.current_temperature if hasattr(self.sh_head, 'current_temperature') else 1.0
            
            alpha = self.magnitude_predictor._get_exploration_alpha(magnitude_temperature)

            logp_flow = self.magnitude_predictor.rqs_flow.log_prob(
                old_mags_tensor.unsqueeze(-1), magnitude_features
            ).view(batch_size)

            r_min = self.magnitude_predictor.r_min
            r_max = self.magnitude_predictor.r_max
            logp_uniform = -math.log(r_max - r_min)

            # Mixture log-prob (matches act())
            log_probs_magnitude = torch.logaddexp(
                math.log(1.0 - alpha + 1e-10) + logp_flow,
                math.log(alpha + 1e-10) + torch.full_like(logp_flow, logp_uniform)
            )
        else:
            log_probs_magnitude = torch.zeros_like(log_probs_direction)

        # 7. Recompute axis log-probs
        if o2_embeddings_asymmetric is not None:
            if isinstance(o2_embeddings_asymmetric, list):
                 o2_embeddings_asymmetric = torch.as_tensor(np.array(o2_embeddings_asymmetric), device=self.device)
            elif isinstance(o2_embeddings_asymmetric, torch.Tensor):
                 o2_embeddings_asymmetric = o2_embeddings_asymmetric.to(self.device)

            if o2_embeddings_asymmetric.dim() == 3 and o2_embeddings_asymmetric.shape[1] == 1:
                o2_embeddings_asymmetric = o2_embeddings_asymmetric.squeeze(1)

            o2_emb_aligned = rotate_o2_asymmetric_to_aligned(
                o2_embeddings_asymmetric, R_feat_to_world, self.device
            )
        else:
            o2_emb_aligned = None            

        # Always recompute axis_params from the CURRENT policy. PPO's new_log_prob must come from
        # the current policy; using stored axis_params would force ratio=1.0 and prevent learning.
        # Compute current O-O direction in the aligned frame for the geometric spear prior.
        o2_dir_world_eval = F.normalize(o2_positions[:, 0] - o2_positions[:, 1], dim=-1)
        o2_dir_aligned_eval = torch.bmm(
            o2_dir_world_eval.unsqueeze(1), R_feat_to_world
        ).squeeze(1)

        axis_params = self.axis_manager.predict_axis_distribution_params(
            frame_features,
            o2_embedding_asymmetric=o2_emb_aligned,
            o2_dir_aligned=o2_dir_aligned_eval,
        )
        
        # Diagnostic: compare recomputed vs stored params (first few calls only)
        if not hasattr(self, '_axis_recompute_debug'):
            self._axis_recompute_debug = 0
        self._axis_recompute_debug += 1
        
        if self._axis_recompute_debug <= 5:
            first_act = old_actions[0] if old_actions else {}
            stored_mean_aligned = first_act.get('axis_mean_direction_aligned')
            stored_log_conc = first_act.get('axis_log_concentration')
            
            _LOG.debug(f"\n[AXIS RECOMPUTE CHECK] call #{self._axis_recompute_debug}")
            _LOG.debug(f"  Recomputed mean_dir[0]: {axis_params['mean_direction'][0].detach().cpu().numpy()}")
            _LOG.debug(f"  Recomputed log_conc[0]: {axis_params['log_concentration'][0].item():.4f}")
            
            if stored_mean_aligned is not None:
                if isinstance(stored_mean_aligned, np.ndarray):
                    stored_mean_aligned = torch.from_numpy(stored_mean_aligned)
                stored_mean_aligned = stored_mean_aligned.to(self.device).float().flatten()
                
                if isinstance(stored_log_conc, np.ndarray):
                    stored_log_conc = torch.from_numpy(stored_log_conc)
                elif isinstance(stored_log_conc, (int, float)):
                    stored_log_conc = torch.tensor(stored_log_conc)
                stored_log_conc = stored_log_conc.to(self.device).float()
                
                mean_diff = (axis_params['mean_direction'][0] - stored_mean_aligned).abs().max().item()
                conc_diff = (axis_params['log_concentration'][0] - stored_log_conc).abs().item()
                
                _LOG.debug(f"  Stored mean_dir[0]:     {stored_mean_aligned.cpu().numpy()}")
                _LOG.debug(f"  Stored log_conc[0]:     {stored_log_conc.item():.4f}")
                _LOG.debug(f"  Diff: mean={mean_diff:.6f}, conc={conc_diff:.6f}")
                
                if mean_diff < 1e-4 and conc_diff < 1e-4:
                    _LOG.debug(f"  WARNING Params identical (policy hasn't updated much yet)")
                else:
                    _LOG.debug(f"  OK Params differ (policy is updating)")
            
            # Verify gradient flow
            _LOG.debug(f"  axis_params['mean_direction'].requires_grad: {axis_params['mean_direction'].requires_grad}")
            _LOG.debug(f"  axis_params['mean_direction'].grad_fn: {axis_params['mean_direction'].grad_fn}")

        if not hasattr(self, '_axis_comprehensive_debug'):
            self._axis_comprehensive_debug = 0
        self._axis_comprehensive_debug += 1
        
        if self._axis_comprehensive_debug <= 3:
            _LOG.debug(f"\n[AXIS COMPREHENSIVE DEBUG] call #{self._axis_comprehensive_debug}")
            
            # 1. Check stored axis_params from action
            first_act = old_actions[0]
            stored_mean_world = first_act.get('axis_mean_direction')
            stored_log_conc = first_act.get('axis_log_concentration')
            stored_axis_frame = first_act.get('axis_frame')
            
            if stored_mean_world is not None:
                if isinstance(stored_mean_world, np.ndarray):
                    stored_mean_world = torch.from_numpy(stored_mean_world)
                stored_mean_world = stored_mean_world.to(self.device).float()
                _LOG.debug(f"  Stored mean_dir (WORLD): {stored_mean_world.flatten().cpu().numpy()}")
            
            if stored_axis_frame is not None:
                if isinstance(stored_axis_frame, np.ndarray):
                    stored_axis_frame = torch.from_numpy(stored_axis_frame)
                stored_axis_frame = stored_axis_frame.to(self.device).float()
                _LOG.debug(f"  Stored axis_frame (ALIGNED): {stored_axis_frame.flatten().cpu().numpy()}")
            
            # 2. Recomputed axis_params
            recomputed_mean_aligned = axis_params['mean_direction'][0]
            recomputed_log_conc = axis_params['log_concentration'][0]
            _LOG.debug(f"  Recomputed mean_dir (ALIGNED): {recomputed_mean_aligned.detach().cpu().numpy()}")
            _LOG.debug(f"  Recomputed log_conc: {recomputed_log_conc.item():.4f}")
            
            # 3. Transform recomputed to world for comparison
            if R_feat_to_world is not None:
                recomputed_mean_world = torch.matmul(recomputed_mean_aligned, R_feat_to_world[0].T)
                _LOG.debug(f"  Recomputed mean_dir (WORLD): {recomputed_mean_world.detach().cpu().numpy()}")
                
                # Compare
                if stored_mean_world is not None:
                    diff_world = (stored_mean_world.flatten() - recomputed_mean_world.flatten()).abs().max().item()
                    _LOG.debug(f"  WORLD frame diff: {diff_world:.6f}")
            
            # 4. Check inputs to predict_axis_distribution_params
            _LOG.debug(f"\n  --- INPUTS TO AXIS HEAD ---")
            _LOG.debug(f"  frame_features[0] norm: {frame_features[0].norm().item():.6f}")
            _LOG.debug(f"  frame_features[0] first 5: {frame_features[0, :5].detach().cpu().numpy()}")
            if o2_emb_aligned is not None:
                _LOG.debug(f"  o2_emb_aligned[0] norm: {o2_emb_aligned[0].norm().item():.6f}")
                _LOG.debug(f"  o2_emb_aligned[0] first 5: {o2_emb_aligned[0, :5].detach().cpu().numpy()}")
            
            # 5. Check R_feat_to_world
            if R_feat_to_world is not None:
                _LOG.debug(f"  R_feat_to_world[0]:\n{R_feat_to_world[0].detach().cpu().numpy()}")
            
            # 6. Check old_dirs_feat (the direction used for frame alignment)
            _LOG.debug(f"  old_dirs_feat[0]: {old_dirs_feat[0].detach().cpu().numpy()}")
            
            # 7. Compute log_prob under stored params for comparison
            if stored_mean_world is not None and stored_axis_frame is not None:
                from realize.networks.distributions.policy_utils.vMF_utils import von_mises_fisher_logprob

                # Transform stored_mean_world to aligned frame
                if R_feat_to_world is not None:
                    stored_mean_aligned = torch.matmul(stored_mean_world.flatten(), R_feat_to_world[0])
                else:
                    stored_mean_aligned = stored_mean_world.flatten()
                
                stored_conc = torch.exp(torch.tensor(stored_log_conc.item() if hasattr(stored_log_conc, 'item') else float(stored_log_conc), device=self.device))
                recomputed_conc = torch.exp(recomputed_log_conc)
                
                # Log prob with STORED params
                lp_stored_plus = von_mises_fisher_logprob(stored_axis_frame.unsqueeze(0), stored_mean_aligned.unsqueeze(0), stored_conc.unsqueeze(0))
                lp_stored_minus = von_mises_fisher_logprob(stored_axis_frame.unsqueeze(0), -stored_mean_aligned.unsqueeze(0), stored_conc.unsqueeze(0))
                lp_stored = torch.logaddexp(lp_stored_plus, lp_stored_minus) - math.log(2.0)
                
                # Log prob with RECOMPUTED params
                lp_recomp_plus = von_mises_fisher_logprob(stored_axis_frame.unsqueeze(0), recomputed_mean_aligned.unsqueeze(0), recomputed_conc.unsqueeze(0))
                lp_recomp_minus = von_mises_fisher_logprob(stored_axis_frame.unsqueeze(0), -recomputed_mean_aligned.unsqueeze(0), recomputed_conc.unsqueeze(0))
                lp_recomp = torch.logaddexp(lp_recomp_plus, lp_recomp_minus) - math.log(2.0)
                
                _LOG.debug(f"\n  --- LOG PROB COMPARISON ---")
                _LOG.debug(f"  Log prob with STORED params: {lp_stored.item():.4f}")
                _LOG.debug(f"  Log prob with RECOMPUTED params: {lp_recomp.item():.4f}")
                _LOG.debug(f"  Diff: {abs(lp_stored.item() - lp_recomp.item()):.4f}")

        if not hasattr(self, '_axis_frame_debug_count'):
            self._axis_frame_debug_count = 0
        self._axis_frame_debug_count += 1
        
        if self._axis_frame_debug_count <= 3:
            _LOG.debug(f"\n[AXIS FRAME DEBUG] call #{self._axis_frame_debug_count}")
            
            # Get stored log_prob from buffer for comparison
            stored_axis_lp = []
            stored_axes = []
            for act in old_actions:
                lp = self._last_outputs.get('log_prob_axis', None)
                ax = act.get('axis_direction')
                if isinstance(ax, np.ndarray): ax = torch.from_numpy(ax)
                stored_axes.append(ax)
            stored_axes = torch.stack(stored_axes).to(self.device)
            
            mean_dir = axis_params['mean_direction']
            conc = torch.exp(axis_params['log_concentration'])
            
            _LOG.debug(f"  mean_direction: {mean_dir[0].detach().cpu().numpy()}")
            _LOG.debug(f"  concentration: {conc[0].item():.4f}")
            _LOG.debug(f"  old_axes (world): {stored_axes[0].detach().cpu().numpy()}")
            _LOG.debug(f"  old_dirs_feat (direction used for frame): {old_dirs_feat[0].detach().cpu().numpy()}")
            
            if R_feat_to_world is not None:
                _LOG.debug(f"  R_feat_to_world[0]:\n{R_feat_to_world[0].detach().cpu().numpy()}")
                
                from realize.networks.distributions.policy_utils.vMF_utils import von_mises_fisher_logprob
                def axial_lp(ax, mu, kappa):
                    lp_p = von_mises_fisher_logprob(ax, mu, kappa)
                    lp_m = von_mises_fisher_logprob(ax, -mu, kappa)
                    return torch.logaddexp(lp_p, lp_m) - math.log(2.0)

                # mean_direction -> world, old_axes stay in world
                mean_dir_world = torch.bmm(mean_dir.unsqueeze(1), R_feat_to_world.transpose(1, 2)).squeeze(1)
                lp_world = axial_lp(stored_axes, mean_dir_world, conc)

                # old_axes -> aligned, mean_direction stays in aligned
                old_axes_aligned_RT = torch.bmm(stored_axes.unsqueeze(1), R_feat_to_world.transpose(1, 2)).squeeze(1)
                old_axes_aligned_R = torch.bmm(stored_axes.unsqueeze(1), R_feat_to_world).squeeze(1)
                lp_aligned_RT = axial_lp(old_axes_aligned_RT, mean_dir, conc)
                lp_aligned_R = axial_lp(old_axes_aligned_R, mean_dir, conc)
                
                _LOG.debug(f"  log_prob (mean->world, axes stay): {lp_world[0].item():.4f}")
                _LOG.debug(f"  log_prob (axes@R.T, mean stay):     {lp_aligned_RT[0].item():.4f}")
                _LOG.debug(f"  log_prob (axes@R, mean stay):       {lp_aligned_R[0].item():.4f}")
        # axis_params['mean_direction'] stays in the aligned frame, matching how log_prob was
        # computed during sampling.
        old_axes_aligned_list = []

        # Diagnostic: compare stored vs recomputed axis params
        if not hasattr(self, '_axis_params_debug'):
            self._axis_params_debug = 0
        self._axis_params_debug += 1
        if self._axis_params_debug <= 3:
            _LOG.debug(f"\n[AXIS PARAMS COMPARISON] call #{self._axis_params_debug}")
            
            # Get stored params from first action
            first_act = old_actions[0]
            stored_mean_dir_world = first_act.get('axis_mean_direction')
            stored_log_conc = first_act.get('axis_log_concentration')
            
            if stored_mean_dir_world is not None:
                if isinstance(stored_mean_dir_world, np.ndarray):
                    stored_mean_dir_world = torch.from_numpy(stored_mean_dir_world)
                stored_mean_dir_world = stored_mean_dir_world.to(self.device).float()
                
                if isinstance(stored_log_conc, np.ndarray):
                    stored_log_conc = torch.from_numpy(stored_log_conc)
                stored_log_conc = stored_log_conc.to(self.device).float()
                
                # Recomputed params (mean_direction is in aligned frame)
                recomputed_mean_dir_aligned = axis_params['mean_direction'][0]
                recomputed_log_conc = axis_params['log_concentration'][0]
                
                # Transform recomputed to world frame for comparison
                if R_feat_to_world is not None:
                    recomputed_mean_dir_world = torch.matmul(
                        recomputed_mean_dir_aligned,
                        R_feat_to_world[0].T
                    )
                else:
                    recomputed_mean_dir_world = recomputed_mean_dir_aligned
                
                # Compare
                mean_dir_diff = (stored_mean_dir_world.flatten() - recomputed_mean_dir_world.flatten()).abs().max().item()
                log_conc_diff = abs(stored_log_conc.item() - recomputed_log_conc.item())
                
                _LOG.debug(f"  Stored mean_dir (world):     {stored_mean_dir_world.flatten().detach().cpu().numpy()}")
                _LOG.debug(f"  Recomputed mean_dir (world): {recomputed_mean_dir_world.detach().cpu().numpy()}")
                _LOG.debug(f"  Mean direction diff: {mean_dir_diff:.6f}")
                _LOG.debug(f"  Stored log_conc:     {stored_log_conc.item():.4f}")
                _LOG.debug(f"  Recomputed log_conc: {recomputed_log_conc.item():.4f}")
                _LOG.debug(f"  Log concentration diff: {log_conc_diff:.6f}")
                
                if mean_dir_diff > 0.01 or log_conc_diff > 0.01:
                    _LOG.debug(f"  WARNING: Axis params changed significantly!")
                else:
                    _LOG.debug(f"  OK: Axis params match (diff < 0.01)")        
            first_act = old_actions[0] if old_actions else {}
            has_axis_frame = 'axis_frame' in first_act
            _LOG.debug(f"\n[AXIS_FRAME AVAILABILITY] call #{self._axis_frame_avail_debug}")
            _LOG.debug(f"  'axis_frame' in action: {has_axis_frame}")
            _LOG.debug(f"  action keys: {list(first_act.keys())}")
            if has_axis_frame:
                af = first_act['axis_frame']
                if isinstance(af, np.ndarray):
                    _LOG.debug(f"  axis_frame value: {af}")
                else:
                    _LOG.debug(f"  axis_frame value: {af.detach().cpu().numpy() if hasattr(af, 'detach') else af}")

        for act in old_actions:
            # Require axis_direction (world frame): axis_params['mean_direction'] is in the current
            # aligned frame, so we transform world -> aligned for correct log_prob evaluation.
            ax_world = act.get('axis_direction')
            if ax_world is None:
                raise RuntimeError(
                    "axis_direction (world frame) missing in stored action. "
                    "The axis_frame fallback is disabled to prevent frame mismatch bugs. "
                    "Ensure all actions store axis_direction during sampling."
                )
            if isinstance(ax_world, np.ndarray):
                ax_world = torch.from_numpy(ax_world)
            old_axes_aligned_list.append(ax_world.to(self.device))

        old_axes_aligned = torch.stack(old_axes_aligned_list)

        # Transform world-frame axes to the current aligned frame, matching
        # axis_params['mean_direction']. rotate_axis_to_global gives axis_world = axis_aligned @ R^T;
        # inverting (R orthogonal): axis_aligned = axis_world @ R.
        if R_feat_to_world is not None:
            old_axes_aligned = torch.bmm(old_axes_aligned.unsqueeze(1), R_feat_to_world).squeeze(1)        
            old_axes_aligned = old_axes_aligned / (torch.norm(old_axes_aligned, dim=-1, keepdim=True) + 1e-8)

        # Batch integrity check (first 3 calls only)
        if not hasattr(self, '_axis_batch_check_count'):
            self._axis_batch_check_count = 0
        self._axis_batch_check_count += 1
        if self._axis_batch_check_count <= 3:
            norms = old_axes_aligned.norm(dim=-1)
            _LOG.debug(f"[AXIS BATCH CHECK] call #{self._axis_batch_check_count}: "
                  f"batch_size={old_axes_aligned.shape[0]}, "
                  f"norm_range=[{norms.min().item():.4f}, {norms.max().item():.4f}], "
                  f"all_world_frame=True (strict mode)")

        if not hasattr(self, '_actual_axis_debug'):
            self._actual_axis_debug = 0
        self._actual_axis_debug += 1
        
        if self._actual_axis_debug <= 3:
            _LOG.debug(f"\n[ACTUAL AXIS EVAL DEBUG] call #{self._actual_axis_debug}")
            _LOG.debug(f"  frame_mode: world_to_current_aligned (strict)")
            _LOG.debug(f"  old_axes_aligned[0]: {old_axes_aligned[0].detach().cpu().numpy()}")
            _LOG.debug(f"  axis_params mean[0]: {axis_params['mean_direction'][0].detach().cpu().numpy()}")            
            _LOG.debug(f"  axis_params conc[0]: {torch.exp(axis_params['log_concentration'][0]).item():.4f}")

        # Diagnostic: check axis_params is recomputed from current policy (connected to graph)
        if not hasattr(self, '_axis_frozen_debug_count'):
            self._axis_frozen_debug_count = 0
        self._axis_frozen_debug_count += 1

        if self._axis_frozen_debug_count <= 5:
            # grad_fn present means axis_params is connected to the current policy
            mean_has_grad = axis_params['mean_direction'].grad_fn is not None
            conc_has_grad = axis_params['log_concentration'].grad_fn is not None

            _LOG.debug(f"\n[AXIS FROZEN DIAGNOSTIC] call #{self._axis_frozen_debug_count}")
            _LOG.debug(f"  axis_params['mean_direction'].requires_grad: {axis_params['mean_direction'].requires_grad}")
            _LOG.debug(f"  axis_params['mean_direction'].grad_fn: {axis_params['mean_direction'].grad_fn}")
            _LOG.debug(f"  axis_params['log_concentration'].requires_grad: {axis_params['log_concentration'].requires_grad}")
            _LOG.debug(f"  axis_params['log_concentration'].grad_fn: {axis_params['log_concentration'].grad_fn}")

            if not mean_has_grad or not conc_has_grad:
                _LOG.debug(f"  WARNING WARNING: axis_params is DETACHED from computation graph!")
                _LOG.debug(f"     This means axis log_prob won't have gradients -> ratio will be 1.000")
            else:
                _LOG.debug(f"  OK axis_params is connected to computation graph")

            # Check axis_params varies across the batch
            mean_std = axis_params['mean_direction'].std().item()
            conc_std = axis_params['log_concentration'].std().item()
            _LOG.debug(f"  axis_params batch variation: mean_std={mean_std:.6f}, conc_std={conc_std:.6f}")
            if mean_std < 1e-8 and conc_std < 1e-8:
                _LOG.debug(f"  WARNING WARNING: axis_params is constant across batch (possible bug)")

            # Get stored axis_frame for comparison
            first_act = old_actions[0]
            stored_af = first_act.get('axis_frame')
            if stored_af is not None:
                if isinstance(stored_af, np.ndarray):
                    stored_af = torch.from_numpy(stored_af)
                stored_af = stored_af.to(self.device).float()
                diff = (old_axes_aligned[0] - stored_af.flatten()).abs().max().item()
                _LOG.debug(f"  Stored axis_frame[0]: {stored_af.flatten().cpu().numpy()}")
                _LOG.debug(f"  Diff from stored: {diff:.6f}")

                status = 'OK' if diff <= 1e-4 else ('MISMATCH' if diff <= 1e-2 else 'SEVERE_MISMATCH')

                _LOG.debug(f"  [AXIS_FIX FRAME_DIFF] stored_vs_eval_diff={diff:.2e} status={status}")

                if diff > 1e-4:

                    _LOG.debug(f"    stored_frame={stored_af.flatten().cpu().numpy()}")

                    _LOG.debug(f"    eval_frame  ={old_axes_aligned[0].detach().cpu().numpy()}")
        
        log_probs_axis = self.axis_manager.evaluate_axis_log_prob(
            axis_params, old_axes_aligned, old_dirs_feat, R_feat_to_world
        )
        
        # Verify axis log_probs have gradient and variation
        if not hasattr(self, '_axis_logprob_check_count'):
            self._axis_logprob_check_count = 0
        self._axis_logprob_check_count += 1

        lp_std = log_probs_axis.std().item()
        lp_has_grad = log_probs_axis.requires_grad

        if self._axis_logprob_check_count <= 10:
            _LOG.debug(f"\n[AXIS LOG_PROB CHECK] call #{self._axis_logprob_check_count}")
            _LOG.debug(f"  log_probs_axis.requires_grad: {lp_has_grad}")
            _LOG.debug(f"  log_probs_axis.grad_fn: {log_probs_axis.grad_fn}")
            _LOG.debug(f"  log_probs_axis std: {lp_std:.6f}")
            _LOG.debug(f"  log_probs_axis range: [{log_probs_axis.min().item():.4f}, {log_probs_axis.max().item():.4f}]")
        
        # Detached axis head won't learn: raise unless this is a frozen copy.
        is_frozen_copy = not any(p.requires_grad for p in self.parameters())
        if not lp_has_grad and not is_frozen_copy:
            raise RuntimeError(
                "log_probs_axis.requires_grad is False. "
                "Axis head is detached from computation graph and won't learn. "
                "Ensure axis_params is recomputed from current policy, not loaded from buffer."
            )
        
        # Post-warmup: zero variation means the axis head is likely frozen.
        if self._axis_logprob_check_count >= 5 and lp_std < 1e-8:
            raise RuntimeError(
                f"log_probs_axis has ~zero variation across batch (std={lp_std:.2e}). "
                f"Axis head is likely outputting constant values. Check axis_manager inputs."
            )

        if hasattr(self, '_axis_debug_count'):
            self._axis_debug_count += 1
        else:
            self._axis_debug_count = 0
        
        if self._axis_debug_count < 10:
            _LOG.debug(f"\n[AXIS GRADIENT TRACE] call #{self._axis_debug_count}")
            _LOG.debug(f"  states.requires_grad: {states.requires_grad}")
            _LOG.debug(f"  frame_features.requires_grad: {frame_features.requires_grad}")
            _LOG.debug(f"  o2_emb_aligned.requires_grad: {o2_emb_aligned.requires_grad if o2_emb_aligned is not None else 'N/A'}")
            _LOG.debug(f"  axis_params['mean_direction'].requires_grad: {axis_params['mean_direction'].requires_grad}")
            _LOG.debug(f"  axis_params['mean_direction'].grad_fn: {axis_params['mean_direction'].grad_fn}")
            _LOG.debug(f"  log_probs_axis.requires_grad: {log_probs_axis.requires_grad}")
            _LOG.debug(f"  log_probs_axis.grad_fn: {log_probs_axis.grad_fn}")
            _LOG.debug(f"  log_probs_axis values: mean={log_probs_axis.mean():.4f}, std={log_probs_axis.std():.6f}")

        total_log_probs = log_probs_direction + log_probs_magnitude + log_probs_axis

        # Value via critic_embedder -> value_mlp; gradient flows through critic_embedder in PPO update.
        state_values = self._compute_value(o1_embeddings, o2_embeddings, geometries)

        direction_entropy = self.direction_manager.compute_entropy(dist, channel_weights)

        # Magnitude entropy: recompute with same context as log_prob.
        if log_probs_magnitude is not None and not torch.all(log_probs_magnitude == 0):
            if hasattr(self.magnitude_predictor, 'compute_entropy'):
                temperature = self.sh_head.current_temperature if hasattr(self.sh_head, 'current_temperature') else 1.0
                magnitude_entropy = self.magnitude_predictor.compute_entropy(
                    magnitude_features, temperature
                ).mean()
            else:
                magnitude_entropy = torch.tensor(0.0, device=self.device)
        else:
            magnitude_entropy = torch.tensor(0.0, device=self.device)

        if hasattr(self.axis_manager, 'compute_entropy'):
            axis_entropy = self.axis_manager.compute_entropy(axis_params).mean()
        else:
            axis_entropy = torch.tensor(0.0, device=self.device)

        total_entropy = direction_entropy + magnitude_entropy + axis_entropy

        per_head_log_probs = {
            'direction': log_probs_direction,
            'magnitude': log_probs_magnitude,
            'axis': log_probs_axis
        }

        return total_log_probs, state_values, total_entropy, per_head_log_probs

