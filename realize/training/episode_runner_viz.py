"""Spherical-harmonic visualization data for episodes."""
import torch
import numpy as np
import os
import pickle

from realize.config.paths import get_results_dir
from realize.training.episode_serialization import _to_serializable
import logging
_LOG = logging.getLogger(__name__)


class EpisodeVisualizationMixin:
    """Spherical-harmonics visualization data dumping for EpisodeRunner.

    ``_save_sh_visualization_data`` runs every step via run_episode. ``self`` and
    all instance attributes resolve via the MRO back to EpisodeRunner, a plain
    orchestrator with no torch Parameters, so this mixin cannot perturb the policy
    fingerprint.
    """

    def _save_sh_visualization_data(self, episode: int, step: int, action_dict: dict):
        """Save spherical harmonics data for visualization"""

        # Get SH data from policy
        if not hasattr(self.agent, 'policy'):
            raise RuntimeError(
                "Agent has no policy attribute (required for SH visualization). "
                "Check agent initialization."
            )

        if not hasattr(self.agent.policy, '_last_outputs'):
            raise RuntimeError(
                "Policy has no _last_outputs attribute. "
                "SH data not captured - ensure policy.set_capture_distributions(True) was called."
            )

        outputs = self.agent.policy._last_outputs

        # Extract BOTH per-channel and aggregated coefficients
        per_channel_coeffs = outputs.get('sh_coeffs', None)  # Should be tensor [batch, num_channels, sh_dim]
        aggregated_coeffs = outputs.get('aggregated_coeffs', None)  # Should be tensor [batch, sh_dim]

        # Extract betas from outputs
        betas_raw = outputs.get('betas', None)
        if betas_raw is not None:
            if hasattr(betas_raw, 'detach'):
                betas_raw = betas_raw.detach().cpu().numpy()
        else:
            _LOG.debug(f"[DEBUG SH SAVE] No betas in _last_outputs")

        # Process per-channel coefficients
        per_channel_coeffs = self._to_numpy_safe(per_channel_coeffs, "per_channel_coeffs")

        if per_channel_coeffs.ndim == 3 and per_channel_coeffs.shape[0] == 1:
            per_channel_coeffs = per_channel_coeffs.squeeze(0)  # Remove batch -> [num_channels, sh_dim]

        # Process aggregated coefficients
        aggregated_coeffs = self._to_numpy_safe(aggregated_coeffs, "aggregated_coeffs")
        if aggregated_coeffs is not None and aggregated_coeffs.ndim == 2 and aggregated_coeffs.shape[0] == 1:
            aggregated_coeffs = aggregated_coeffs.squeeze(0)

        # Validate per-channel coefficients exist (NO SYNTHETIC DATA)
        if per_channel_coeffs is None:
            raise RuntimeError(
                "Missing per-channel SH coefficients ('sh_coeffs') in _last_outputs. "
                "Policy is not storing per-channel data correctly. "
                "Check networks/heads/equivariant_sh_head.py forward() to ensure it stores 'sh_coeffs'."
            )

        # Determine num_channels
        if per_channel_coeffs is not None and per_channel_coeffs.ndim == 2:
            num_channels = per_channel_coeffs.shape[0]
        else:
            num_channels = 1

        # Extract translation direction
        trans_dir = action_dict.get('translation_direction', None)
        if trans_dir is not None and hasattr(trans_dir, 'detach'):
            trans_dir = trans_dir.detach().cpu().numpy()
            if trans_dir.ndim > 1:
                trans_dir = trans_dir.reshape(-1)[:3]  # Ensure it's [3]

        # Validate translation direction
        if trans_dir is None:
            raise RuntimeError(
                "translation_direction is None in action_dict. "
                "Policy must provide valid translation direction."
            )

        trans_dir_norm = np.linalg.norm(trans_dir)
        if trans_dir_norm < 1e-6:
            raise RuntimeError(
                f"translation_direction has near-zero norm: {trans_dir_norm:.3e}. "
                f"Direction: {trans_dir}. Policy must provide valid unit direction."
            )

        # Extract translation magnitude
        trans_mag = action_dict.get('translation_magnitude', 0)
        if hasattr(trans_mag, 'item'):
            trans_mag = trans_mag.item()
        elif hasattr(trans_mag, 'detach'):
            trans_mag = trans_mag.detach().cpu().item()

        # Extract channel weights
        ch_weights = outputs.get('channel_weights', None)

        if ch_weights is not None:
            # Convert to numpy if it's a tensor
            if hasattr(ch_weights, 'detach'):
                ch_weights_np = ch_weights.detach().cpu().numpy()
            elif isinstance(ch_weights, np.ndarray):
                ch_weights_np = ch_weights
            else:
                ch_weights_np = np.array(ch_weights)

            # Squeeze to remove ALL singleton dimensions
            original_shape = ch_weights_np.shape
            ch_weights_squeezed = np.squeeze(ch_weights_np)

            # If still not 1D, take appropriate slice
            if ch_weights_squeezed.ndim > 1:
                _LOG.debug(f"Channel weights still {ch_weights_squeezed.ndim}D after squeeze, taking first row/column")
                if ch_weights_squeezed.shape[0] == num_channels:
                    ch_weights_final = ch_weights_squeezed[:, 0] if ch_weights_squeezed.shape[1] == 1 else ch_weights_squeezed[0, :]
                else:
                    ch_weights_final = ch_weights_squeezed.flatten()[:num_channels]
            else:
                ch_weights_final = ch_weights_squeezed

            # Handle scalar case
            if ch_weights_final.ndim == 0 or ch_weights_final.size == 1:
                scalar_val = ch_weights_final.item() if hasattr(ch_weights_final, 'item') else float(ch_weights_final)
                ch_weights_final = np.full(num_channels, scalar_val)

            ch_weights = ch_weights_final
        else:
            _LOG.debug(f"[DEBUG SH SAVE] Channel weights is None in outputs")

        # Get lmax from policy
        lmax = self.agent.policy.get_lmax() if hasattr(self.agent.policy, 'get_lmax') else 2

        # Extract channel_lmax_values (needed for visualization)
        channel_lmax_values = outputs.get('channel_lmax_values', None)
        if channel_lmax_values is None and hasattr(self.agent.policy, 'sh_head'):
            if hasattr(self.agent.policy.sh_head, 'channel_lmax_values'):
                channel_lmax_values = self.agent.policy.sh_head.channel_lmax_values
        if channel_lmax_values is not None:
            # Convert to list for JSON serialization
            if hasattr(channel_lmax_values, 'tolist'):
                channel_lmax_values = channel_lmax_values.tolist()
            elif hasattr(channel_lmax_values, 'detach'):
                channel_lmax_values = channel_lmax_values.detach().cpu().numpy().tolist()
            else:
                channel_lmax_values = list(channel_lmax_values)

        # Get magnitude distribution data - check multiple locations
        magnitude_data = None

        # First check outputs directly (try both key names)
        magnitude_data = outputs.get('magnitude') or outputs.get('magnitude_data')

        # Then check magnitude predictor
        if magnitude_data is None and hasattr(self.agent.policy, 'magnitude_predictor'):
            mag_predictor = self.agent.policy.magnitude_predictor

            if hasattr(mag_predictor, '_last_magnitude_data') and mag_predictor._last_magnitude_data is not None:
                magnitude_data = mag_predictor._last_magnitude_data.copy()
            elif hasattr(mag_predictor, 'get_magnitude_log_data'):
                magnitude_data = mag_predictor.get_magnitude_log_data()

        if magnitude_data is None:
            _LOG.debug(f"No magnitude distribution data available for visualization")

        # Extract world transform if available
        R_feat_to_world = None
        if hasattr(self.agent.policy.sh_head, 'has_world_frame') and self.agent.policy.sh_head.has_world_frame():
            R_raw = self.agent.policy.sh_head._R_feat_to_world
            if R_raw is not None:
                if hasattr(R_raw, 'detach'):
                    R_feat_to_world = R_raw.detach().cpu().numpy()
                else:
                    R_feat_to_world = np.array(R_raw)


        # Coordinate frame: with R_feat_to_world, data is in feature frame and
        # needs the transform; otherwise it is already in world frame.
        frame = outputs.get('frame', None)
        if frame is None:
            frame = 'feature' if R_feat_to_world is not None else 'world'

        # Pull required items directly from policy
        last = self.agent.policy._last_outputs

        axis_angle_pdf_full = last.get('axis_angle_pdf', None)
        if axis_angle_pdf_full is None:
            raise RuntimeError(
                "policy._last_outputs['axis_angle_pdf'] missing. "
                "PolicyNetwork.forward() must construct and cache the axis-angle PDF dict."
            )

        # Validate exact structure
        required_pdf_keys = ['theta_grid', 'log_p_theta', 'theta_sample']
        missing_pdf = [k for k in required_pdf_keys if k not in axis_angle_pdf_full or axis_angle_pdf_full[k] is None]
        if missing_pdf:
            raise RuntimeError(f"axis_angle_pdf missing keys: {missing_pdf}")

        # Normalize types (no fallback conversions)
        theta_grid_np  = np.asarray(axis_angle_pdf_full['theta_grid'], dtype=float).reshape(-1)
        log_p_theta_np = np.asarray(axis_angle_pdf_full['log_p_theta'], dtype=float).reshape(-1)
        theta_sample_f = float(axis_angle_pdf_full['theta_sample'])
        theta_mean_val = None if axis_angle_pdf_full.get('theta_mean') is None else float(axis_angle_pdf_full['theta_mean'])
        theta_std_val  = None if axis_angle_pdf_full.get('theta_std')  is None else float(axis_angle_pdf_full['theta_std'])

        # axis_direction must exist
        axis_direction_world = last.get('axis_direction', None)
        if axis_direction_world is None:
            raise RuntimeError("policy._last_outputs['axis_direction'] missing.")

        # Take first batch, ensure [3]
        if torch.is_tensor(axis_direction_world):
            axis_direction_world = axis_direction_world.detach().cpu().numpy()
        axis_direction_world = np.asarray(axis_direction_world, dtype=float)

        # Handle batch dimension explicitly
        if axis_direction_world.ndim == 2:
            if axis_direction_world.shape[0] != 1:
                raise RuntimeError(
                    f"axis_direction has batch size {axis_direction_world.shape[0]}, "
                    f"expected batch size 1. Shape: {axis_direction_world.shape}"
                )
            axis_direction_world = axis_direction_world[0]
        elif axis_direction_world.ndim != 1:
            raise RuntimeError(
                f"axis_direction must be 1D or 2D (batch, 3), "
                f"got ndim={axis_direction_world.ndim}, shape={axis_direction_world.shape}"
            )

        if axis_direction_world.shape != (3,):
            raise RuntimeError(
                f"axis_direction shape must be (3,), got {axis_direction_world.shape}"
            )

        # Build the entry EXACTLY as the plotter expects
        sh_entry = {
            'sh_coeffs': per_channel_coeffs,
            'aggregated_coeffs': aggregated_coeffs,
            'translation_direction': trans_dir,
            'translation_magnitude': float(trans_mag),
            'channel_log_mass_normalized': _to_serializable(outputs.get('channel_log_mass_normalized')),
            'mixture_log_mass_normalized': _to_serializable(outputs.get('mixture_log_mass_normalized')),
            'grid_weights': _to_serializable(outputs.get('grid_weights')),
            'grid_n_beta': outputs.get('grid_n_beta'),
            'grid_n_alpha': outputs.get('grid_n_alpha'),
            'temperature': self.agent.policy.get_current_temperature() if hasattr(self.agent.policy, 'get_current_temperature') else 1.0,
            'channel_weights': ch_weights,
            'channel_temps': outputs.get('channel_temps', None),
            'channel_scales': outputs.get('channel_scales', None),
            'betas': betas_raw,
            'frame': frame,
            'R_feat_to_world': R_feat_to_world,
            'magnitude': magnitude_data,
            'lmax': lmax,
            'num_channels': num_channels,
            'channel_lmax_values': channel_lmax_values,

            # REQUIRED top-level keys for the figure code (no alternates)
            'axis_direction': _to_serializable(axis_direction_world),
            'env_multipole': _to_serializable(getattr(self, 'current_env_multipole', None)),
            'axis_angle_pdf': {
                'theta_grid': _to_serializable(theta_grid_np),
                'log_p_theta': _to_serializable(log_p_theta_np),
                'theta_sample': theta_sample_f,
                'theta_mean': theta_mean_val,
                'theta_std': theta_std_val,
            },
        }

        # Validate the new axis-angle PDF bundle
        pdf_bundle = sh_entry.get('axis_angle_pdf')
        if pdf_bundle is None:
            raise RuntimeError("missing 'axis_angle_pdf' in saved entry")
        theta = np.asarray(pdf_bundle['theta_grid'], dtype=float).reshape(-1)
        logp  = np.asarray(pdf_bundle['log_p_theta'], dtype=float).reshape(-1)
        if theta.ndim != 1 or logp.ndim != 1 or theta.shape != logp.shape:
            raise RuntimeError(f"theta/logp shape mismatch: {theta.shape} vs {logp.shape}")
        pdf = np.exp(logp)
        # Verify PDF is properly normalized from policy_network
        area = np.trapz(pdf, theta)
        if abs(area - 1.0) > 0.05:  # Allow tiny numerical error
            raise RuntimeError(
                f"axis_angle_pdf not normalized from policy_network.\n"
                f"  Expected: integral  p dtheta ~= 1.0\n"
                f"  Got: {area:.6f}\n"
                f"This indicates a bug in PolicyNetwork._build_axis_network() or sample_axis_stochastic().\n"
                f"Policy must produce properly normalized PDFs - no patching allowed."
            )

        # Load existing data or create new
        results_dir = get_results_dir()
        pkl_file = os.path.join(results_dir, 'sphere_params.pkl')

        if os.path.exists(pkl_file):
            # Truncated/corrupt file (kill mid-pickle.dump) would otherwise
            # send every step into an EOFError loop. Quarantine and start
            # fresh -- this is resilient persistence, not policy bypass.
            try:
                with open(pkl_file, 'rb') as f:
                    all_data = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as _load_err:
                import time as _t
                _bad = pkl_file + f'.corrupt.{int(_t.time())}'
                os.rename(pkl_file, _bad)
                print(f"[SH viz] corrupt {pkl_file} quarantined to {_bad}: "
                      f"{_load_err}", flush=True)
                all_data = {}
        else:
            # File doesn't exist yet, create new
            all_data = {}

        # Add to episode data
        if episode not in all_data:
            all_data[episode] = {}
        all_data[episode][step] = sh_entry

        # Atomic write: SIGKILL between open() and replace() leaves the
        # original pkl intact rather than truncated.
        _tmp = pkl_file + '.tmp'
        with open(_tmp, 'wb') as f:
            pickle.dump(all_data, f)
        os.replace(_tmp, pkl_file)

