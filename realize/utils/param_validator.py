"""Parameter validation utilities to ensure data-flow integrity."""
import torch
import numpy as np
from typing import Dict, List, Any
import logging
_LOG = logging.getLogger(__name__)

def validate_viz_params(params: Dict[str, Any], source: str = "unknown") -> bool:
    """Validate visualization params for completeness and consistency.

    Returns True if all checks pass; raises ValueError listing every error and
    warning otherwise. `source` is used only in messages.
    """
    _LOG.info(f"\n[PARAM VALIDATION] Source: {source}")

    errors = []
    warnings = []

    # Required parameters.
    critical_params = {
        'sh_coeffs': 'SH coefficients for all channels',
        'betas': 'Beta concentration parameters per channel',
        'channel_weights': 'Channel mixing weights (gate output)',
        'lmax': 'Global maximum l value',
        'num_channels': 'Number of channels',
    }
    
    for param_name, description in critical_params.items():
        if param_name not in params or params[param_name] is None:
            errors.append(f"MISSING '{param_name}' ({description})")
    
    # Recommended parameters.
    recommended_params = {
        'channel_lmax_values': 'Per-channel lmax values (needed for correct reconstruction)',
        'channel_temps': 'Per-channel temperature values',
        'area_at_mass_targets': 'Target area@mass fraction per channel',
        'target_mass_fraction': 'Mass fraction threshold (e.g., 0.70)',
    }
    
    for param_name, description in recommended_params.items():
        if param_name not in params or params[param_name] is None:
            warnings.append(f"MISSING RECOMMENDED: '{param_name}' ({description})")
    
    # Cross-validate dimensions against num_channels and lmax.
    if 'sh_coeffs' in params and params['sh_coeffs'] is not None:
        sh_coeffs = params['sh_coeffs']
        if isinstance(sh_coeffs, torch.Tensor):
            sh_coeffs = sh_coeffs.cpu().numpy()

        if sh_coeffs.ndim == 2:
            actual_num_channels = sh_coeffs.shape[0]
            actual_sh_dim = sh_coeffs.shape[1]
            _LOG.info(f"  sh_coeffs.shape: ({actual_num_channels}, {actual_sh_dim})")

            if 'num_channels' in params:
                expected_channels = params['num_channels']
                if actual_num_channels != expected_channels:
                    errors.append(
                        f"DIMENSION MISMATCH: sh_coeffs has {actual_num_channels} channels "
                        f"but num_channels={expected_channels}"
                    )

            # sh_dim should be (lmax+1)^2.
            if 'lmax' in params:
                global_lmax = params['lmax']
                expected_sh_dim = (global_lmax + 1) ** 2
                _LOG.info(f"  global lmax={global_lmax} -> expected sh_dim={expected_sh_dim}")
                
                if actual_sh_dim > expected_sh_dim:
                    warnings.append(
                        f"SH DIM MISMATCH: sh_coeffs has {actual_sh_dim} dims "
                        f"but global lmax={global_lmax} only needs {expected_sh_dim}"
                    )
        else:
            warnings.append(f"sh_coeffs has unexpected ndim={sh_coeffs.ndim} (expected 2)")
    
    if 'betas' in params and params['betas'] is not None:
        betas = params['betas']
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        _LOG.info(f"  betas: {betas}")
        
        if 'num_channels' in params:
            if len(betas) != params['num_channels']:
                errors.append(
                    f"DIMENSION MISMATCH: betas has {len(betas)} values "
                    f"but num_channels={params['num_channels']}"
                )
    
    # channel_weights must sum to 1 within 1e-3.
    if 'channel_weights' in params and params['channel_weights'] is not None:
        weights = params['channel_weights']
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()
        _LOG.info(f"  channel_weights: {weights}")

        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > 1e-3:
            errors.append(
                f"NORMALIZATION ERROR: channel_weights sum to {weight_sum:.6f} "
                f"(must be 1.0 within 1e-3). "
                f"Check gate network softmax normalization."
            )

    if 'channel_lmax_values' in params and params['channel_lmax_values'] is not None:
        ch_lmax = params['channel_lmax_values']
        _LOG.info(f"  channel_lmax_values: {ch_lmax}")

        if 'num_channels' in params:
            if len(ch_lmax) != params['num_channels']:
                errors.append(
                    f"DIMENSION MISMATCH: channel_lmax_values has {len(ch_lmax)} values "
                    f"but num_channels={params['num_channels']}"
                )

        # Per-channel non-zero coefficient count must not exceed (lmax_c+1)^2.
        if 'sh_coeffs' in params and params['sh_coeffs'] is not None:
            sh_coeffs = params['sh_coeffs']
            if isinstance(sh_coeffs, torch.Tensor):
                sh_coeffs = sh_coeffs.cpu().numpy()

            for c, lmax_c in enumerate(ch_lmax):
                expected_dim = (lmax_c + 1) ** 2
                actual_coeffs = sh_coeffs[c] if sh_coeffs.ndim == 2 else sh_coeffs

                if isinstance(actual_coeffs, np.ndarray):
                    nonzero_count = np.count_nonzero(actual_coeffs)
                else:
                    nonzero_count = len(actual_coeffs)
                
                if nonzero_count > expected_dim:
                    warnings.append(
                        f"Ch{c}: has {nonzero_count} coefficients but lmax={lmax_c} "
                        f"only needs {expected_dim}"
                    )
    
    if errors:
        _LOG.info(f"\n  VALIDATION ERRORS ({len(errors)}):")
        for error in errors:
            _LOG.info(f"    - {error}")

    if warnings:
        _LOG.info(f"\n   WARNINGS ({len(warnings)}):")
        for warning in warnings:
            _LOG.info(f"    - {warning}")

    # Any error or warning is fatal here.
    if errors or warnings:
        all_issues = errors + warnings
        issues_text = "\n".join(f"  - {issue}" for issue in all_issues)
        raise ValueError(
            f"[{source}] Validation failed with {len(errors)} errors and {len(warnings)} warnings:\n"
            f"{issues_text}\n"
            f"All parameters must be valid - no warnings allowed."
        )
    
    _LOG.info(f"  All validations passed!")
    return True


def validate_model_output(model_output: Dict[str, Any], model_name: str = "unknown") -> bool:
    """Validate a model.forward() output dict before training/visualization.

    Returns True if all checks pass; raises ValueError otherwise.
    """
    _LOG.info(f"\n[MODEL OUTPUT VALIDATION] Model: {model_name}")

    errors = []
    warnings = []

    required_keys = {
        'channel_log_mass_normalized': 'Normalized log probability mass per channel',
        'channel_weights': 'Gate mixing weights',
        'grid_points': 'Sphere grid points (xyz)',
        'grid_weights': 'Quadrature weights',
        'betas': 'Beta concentration parameters',
    }
    
    for key, description in required_keys.items():
        if key not in model_output:
            errors.append(f"MISSING OUTPUT: '{key}' ({description})")
    
    if 'channel_log_mass_normalized' in model_output:
        log_mass = model_output['channel_log_mass_normalized']
        _LOG.info(f"  channel_log_mass_normalized.shape: {log_mass.shape}")

        if log_mass.dim() != 3:
            errors.append(
                f"SHAPE ERROR: channel_log_mass_normalized has {log_mass.dim()} dims "
                f"(expected 3: [batch, channels, grid_points])"
            )

    # Each [batch, channel] distribution must integrate to 1 over the grid.
    if 'channel_log_mass_normalized' in model_output and 'grid_weights' in model_output:
        log_mass = model_output['channel_log_mass_normalized']
        w = model_output['grid_weights']

        if w.dim() != 1:
            errors.append(
                f"SHAPE ERROR: grid_weights must be 1D, got {w.dim()}D"
            )
        elif w.numel() != log_mass.shape[-1]:
            errors.append(
                f"SHAPE ERROR: grid_weights length {w.numel()} != "
                f"channel_log_mass_normalized last dim {log_mass.shape[-1]}"
            )
        else:
            B, C, P = log_mass.shape
            mass = torch.exp(log_mass)  # [B, C, P]
            integral = (mass * w.view(1, 1, -1)).sum(-1)  # [B, C]

            error_threshold = 1e-3
            bad = (integral - 1.0).abs() > error_threshold

            if bad.any():
                bad_indices = bad.nonzero(as_tuple=False).tolist()
                # Report at most the first 10 violations.
                for b, c in bad_indices[:10]:
                    integral_val = integral[b, c].item()
                    error_val = abs(integral_val - 1.0)
                    errors.append(
                        f"NORMALIZATION ERROR: Batch {b}, Channel {c} integrates to "
                        f"{integral_val:.6f} (error={error_val:.6f} > {error_threshold})"
                    )
                
                if len(bad_indices) > 10:
                    errors.append(
                        f"... and {len(bad_indices) - 10} more normalization errors"
                    )

    if errors:
        _LOG.info(f"\n  VALIDATION ERRORS ({len(errors)}):")
        for error in errors:
            _LOG.info(f"    - {error}")

    if warnings:
        _LOG.info(f"\n WARNINGS ({len(warnings)}):")
        for warning in warnings:
            _LOG.info(f" {warning}")

    if errors or warnings:
        all_issues = errors + warnings
        issues_text = "\n".join(f"  {issue}" for issue in all_issues)
        raise ValueError(
            f"[{model_name}] Model output validation failed with "
            f"{len(errors)} errors and {len(warnings)} warnings:\n"
            f"{issues_text}\n"
            f"Model must produce valid outputs - no warnings allowed."
        )
    
    _LOG.info(f"  All validations passed!")
    return True    
