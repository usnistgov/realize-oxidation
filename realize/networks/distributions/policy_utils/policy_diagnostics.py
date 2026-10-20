"""Gradient, equivariance, and policy diagnostics."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from e3nn.o3 import Irreps, Linear, rand_matrix
import torch.nn.functional as F
import logging
_LOG = logging.getLogger(__name__)


class PolicyDiagnosticsManager:
    """Manager for gradient monitoring, equivariance testing, and diagnostics."""
    
    def __init__(self, policy_network):
        """
        Initialize diagnostics manager.
        
        Args:
            policy_network: Reference to parent PolicyNetwork instance
        """
        self.policy = policy_network
        self.gradient_monitor = {}
        self.gradient_hooks = []
        self.gradient_log_frequency = 1
        self.gradient_step_counter = 0
        self.param_snapshots = {}
    
    def register_gradient_hooks(self):
        """Register hooks to monitor gradient flow through the network."""
        
        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    grad_max = grad.abs().max().item()
                    
                    # Store gradient statistics
                    if name not in self.gradient_monitor:
                        self.gradient_monitor[name] = []
                    
                    self.gradient_monitor[name].append({
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'max': grad_max,
                        'shape': list(grad.shape),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item()
                    })
                    
                    # Check for gradient issues
                    if grad_norm == 0:
                        _LOG.debug(f"  [GRAD WARNING] {name}: ZERO gradient!")
                    elif grad_norm < 1e-8:
                        _LOG.debug(f"  [GRAD WARNING] {name}: Very small gradient norm = {grad_norm:.2e}")
                    elif grad_norm > 100:
                        _LOG.debug(f"  [GRAD WARNING] {name}: Large gradient norm = {grad_norm:.2f}")
                    
                    if torch.isnan(grad).any():
                        _LOG.debug(f"  [GRAD ERROR] {name}: NaN in gradients!")
                    if torch.isinf(grad).any():
                        _LOG.debug(f"  [GRAD ERROR] {name}: Inf in gradients!")
                
                return grad
            return hook
        
        # Register hooks for critical parameters
        critical_params = {}
        
        # --- Channel gating network (guard structure/attrs) ---
        gate = getattr(self.policy.sh_head, 'channel_gate', None)
        if isinstance(gate, (nn.Sequential, list, tuple)) and len(gate) > 0:
            last = gate[-1]
            first = gate[0]
            if hasattr(last, 'weight'):
                critical_params['channel_gate.final'] = last.weight
            if hasattr(first, 'weight'):
                critical_params['channel_gate.first'] = first.weight
        
        # --- Channel network parameters (wrapped/unwrapped safe access) ---
        if hasattr(self.policy.sh_head, 'channel_nets') and len(self.policy.sh_head.channel_nets) > 0:
            net0 = self.policy.sh_head.channel_nets[0]
            base_seq = getattr(net0, 'base_net', net0)
            
            # first layer
            if isinstance(base_seq, (nn.Sequential, list, tuple)) and len(base_seq) > 0:
                first_layer = base_seq[0]
                if hasattr(first_layer, 'weight'):
                    critical_params['channel_net_0.first'] = first_layer.weight
                
                # last layer
                last_layer = base_seq[-1]
                if hasattr(last_layer, 'weight'):
                    critical_params['channel_net_0.last'] = last_layer.weight
                elif hasattr(last_layer, 'gains'):
                    critical_params['channel_net_0.per_l_gains'] = last_layer.gains
        
        # RQS flow parameters
        if hasattr(self.policy.magnitude_predictor, 'rqs_flow'):
            rqs_flow = self.policy.magnitude_predictor.rqs_flow
            if hasattr(rqs_flow, 'param_nets') and len(rqs_flow.param_nets) > 0:
                first_param_net = rqs_flow.param_nets[0]
                critical_params['magnitude.rqs.layer0.first'] = first_param_net[0].weight
                critical_params['magnitude.rqs.layer0.last'] = first_param_net[-1].weight
                
                if len(rqs_flow.param_nets) > 1:
                    last_param_net = rqs_flow.param_nets[-1]
                    critical_params[f'magnitude.rqs.layer{len(rqs_flow.param_nets)-1}.last'] = last_param_net[-1].weight
        
        # Frame encoder
        if hasattr(self.policy.frame_encoder, 'frame_encoder'):
            for i, module in enumerate(self.policy.frame_encoder.frame_encoder.modules()):
                if isinstance(module, Linear):
                    critical_params['frame_encoder.first'] = module.weight
                    break
        
        # Scalar extractor
        if hasattr(self.policy.frame_encoder, 'scalar_extractor'):
            scalar_extractor = self.policy.frame_encoder.scalar_extractor
            if isinstance(scalar_extractor, Linear):
                critical_params['frame_encoder.scalar_extractor'] = scalar_extractor.weight
        
        # Value network
        if hasattr(self.policy, 'fc1'):
            critical_params['value.fc1'] = self.policy.fc1.weight
        if hasattr(self.policy, 'fc_value'):
            critical_params['value.fc_final'] = self.policy.fc_value[-1].weight
        
        # Actually register the hooks
        for name, param in critical_params.items():
            if param is not None and param.requires_grad:
                handle = param.register_hook(make_hook(name))
                self.gradient_hooks.append(handle)
                _LOG.debug(f"  Registered gradient hook for {name} (shape: {param.shape})")
            elif param is not None and not param.requires_grad:
                _LOG.debug(f"  Skipped {name} (frozen parameter)")
    
    def log_gradient_flow(self, step: int = None, episode: int = None):
        """Log gradient flow statistics."""
        
        self.gradient_step_counter += 1
        
        # Only log at specified frequency
        if self.gradient_step_counter % self.gradient_log_frequency != 0:
            return
        
        _LOG.debug(f"\n[GRADIENT FLOW] Episode {episode}, Step {step}:")
        _LOG.debug("=" * 60)
        
        # Summarize gradients for each monitored parameter
        for name, grad_history in self.gradient_monitor.items():
            if len(grad_history) > 0:
                recent = grad_history[-10:] if len(grad_history) > 10 else grad_history
                
                avg_norm = np.mean([g['norm'] for g in recent])
                avg_std = np.mean([g['std'] for g in recent])
                max_norm = max([g['norm'] for g in recent])
                min_norm = min([g['norm'] for g in recent])
                
                # Check for issues
                num_zeros = sum(1 for g in recent if g['norm'] == 0)
                num_small = sum(1 for g in recent if g['norm'] < 1e-6)
                has_nan = any(g['has_nan'] for g in recent)
                has_inf = any(g['has_inf'] for g in recent)
                
                status = "OK"
                if num_zeros > len(recent) * 0.5:
                    status = "DEAD (mostly zeros)"
                elif num_small > len(recent) * 0.5:
                    status = "VANISHING"
                elif max_norm > 100:
                    status = "EXPLODING"
                elif has_nan:
                    status = "NaN DETECTED"
                elif has_inf:
                    status = "Inf DETECTED"
                
                _LOG.debug(f"  {name:30s}: norm=[{min_norm:.2e}, {max_norm:.2e}] avg={avg_norm:.2e} | {status}")
                
                # Detailed warning for problematic layers
                if status != "OK":
                    _LOG.debug(f"    Shape: {recent[0]['shape']}, Zeros: {num_zeros}/{len(recent)}, Small: {num_small}/{len(recent)}")
        
        # Check parameter updates
        self.check_parameter_updates(episode, step)
        
        _LOG.debug("=" * 60)
    
    def check_parameter_updates(self, episode: int = None, step: int = None):
        """Check if parameters are actually changing between updates."""
        
        params_to_check = {
            'channel_gate_final': self.policy.sh_head.channel_gate[-1].weight,
            'channel_weights_bias': self.policy.sh_head.channel_gate[-1].bias if hasattr(self.policy.sh_head.channel_gate[-1], 'bias') else None,
            'channel_temps': self.policy.sh_head.channel_log_temps if hasattr(self.policy.sh_head, 'channel_log_temps') else None,
        }
        
        _LOG.debug(f"\n  [PARAMETER CHANGES]:")
        for name, param in params_to_check.items():
            if param is None:
                continue
            
            param_data = param.detach().cpu().numpy() if hasattr(param, 'detach') else param
            
            # Skip empty arrays
            if param_data.size == 0:
                _LOG.debug(f"    {name}: EMPTY parameter (skipped)")
                continue
            
            if name in self.param_snapshots:
                old_data = self.param_snapshots[name]
                
                # Check both arrays have same shape and are non-empty
                if old_data.size == 0 or param_data.shape != old_data.shape:
                    _LOG.debug(f"    {name}: Shape changed or was empty")
                    self.param_snapshots[name] = param_data.copy()
                    continue
                
                change = np.abs(param_data - old_data).max()
                
                if change == 0:
                    _LOG.debug(f"    {name}: NO CHANGE (frozen?)")
                elif change < 1e-8:
                    _LOG.debug(f"    {name}: Tiny change = {change:.2e}")
                else:
                    _LOG.debug(f"    {name}: Max change = {change:.6f}")
            
            # Update snapshot
            self.param_snapshots[name] = param_data.copy()
    
    def analyze_channel_gradients(self):
        """Specific analysis of channel-related gradients."""
        
        _LOG.debug(f"\n[CHANNEL GRADIENT ANALYSIS]:")
        
        # Check if channel gate is receiving gradients
        gate_params = dict(self.policy.sh_head.channel_gate.named_parameters())
        
        for name, param in gate_params.items():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                _LOG.debug(f"  Gate {name}: grad_norm={grad_norm:.2e}, grad_max={grad_max:.2e}")
            else:
                _LOG.debug(f"  Gate {name}: NO GRADIENT")
        
        # Check channel network gradients
        if hasattr(self.policy.sh_head, 'channel_nets'):
            for i in range(min(2, len(self.policy.sh_head.channel_nets))):
                channel_net = self.policy.sh_head.channel_nets[i]
                
                if hasattr(channel_net, 'base_net'):
                    first_layer = channel_net.base_net[0]
                else:
                    first_layer = channel_net[0]
                
                if hasattr(first_layer, 'weight') and first_layer.weight.grad is not None:
                    grad_norm = first_layer.weight.grad.norm().item()
                    _LOG.debug(f"  Channel {i} first layer: grad_norm={grad_norm:.2e}")
                else:
                    _LOG.debug(f"  Channel {i} first layer: NO GRADIENT")
    
    def check_gradient_health(self) -> Dict[str, Any]:
        """Get a summary of gradient health across the network."""
        
        health = {
            'dead_params': [],
            'vanishing_params': [],
            'exploding_params': [],
            'healthy_params': [],
            'total_params': 0
        }
        
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                health['total_params'] += 1
                
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    
                    if grad_norm == 0:
                        health['dead_params'].append(name)
                    elif grad_norm < 1e-7:
                        health['vanishing_params'].append(name)
                    elif grad_norm > 100:
                        health['exploding_params'].append(name)
                    else:
                        health['healthy_params'].append(name)
                else:
                    health['dead_params'].append(name)
        
        # Summary
        health['percent_dead'] = len(health['dead_params']) / max(1, health['total_params']) * 100
        health['percent_vanishing'] = len(health['vanishing_params']) / max(1, health['total_params']) * 100
        health['percent_healthy'] = len(health['healthy_params']) / max(1, health['total_params']) * 100
        
        return health
    
    def run_equivariance_tests(self, state: torch.Tensor, episode: int = 0) -> Dict[str, bool]:
        """
        Run periodic equivariance checks in debug mode.
        
        Args:
            state: Input state [batch, state_dim]
            episode: Current episode number for periodic checking
        
        Returns:
            Dictionary of test results
        """
        debug_enabled = getattr(self.policy.model_config, 'debug_equivariance', False)
        check_frequency = getattr(self.policy.model_config, 'equivariance_check_frequency', 100)
        
        if not debug_enabled or episode % check_frequency != 0:
            return {}
        
        results = {}
        
        if not hasattr(self.policy, '_rotate_state'):
            return {}
        
        # Generate test direction
        batch_size = state.shape[0]
        test_direction = F.normalize(torch.randn(batch_size, 3, device=self.policy.device), dim=-1)
        
        # Run frame encoder tests
        if hasattr(self.policy, 'frame_encoder'):
            frame_results = self.policy.frame_encoder.run_equivariance_tests(
                state, test_direction
            )
            results['frame_encoder'] = frame_results
        
        # Policy-level invariance test
        results['policy_invariance'] = self._test_policy_invariance(state)
        
        # Log summary
        self._log_equivariance_summary(results, episode)
        
        return results
    
    def _test_policy_invariance(self, state: torch.Tensor, tolerance: float = 1e-5) -> bool:
        """
        Test actual E(3) equivariance of the policy.
        
        Uses unaligned features for scalar invariance check.
        """
        batch_size = state.shape[0]
        
        # Generate random rotation matrix
        R = rand_matrix(batch_size).to(self.policy.device)
        
        # Apply rotation to input state
        from realize.networks.distributions.policy_utils.policy_helpers import rotate_state
        state_rotated = rotate_state(state, R)
        
        # Extract unaligned features
        embeddings_orig = self.policy.frame_encoder.extract_unaligned_features(state)
        embeddings_rot = self.policy.frame_encoder.extract_unaligned_features(state_rotated)
        
        # Get encoder irreps
        if hasattr(self.policy.frame_encoder, 'frame_encoder') and hasattr(self.policy.frame_encoder.frame_encoder, 'irreps_out'):
            encoder_irreps_raw = self.policy.frame_encoder.frame_encoder.irreps_out
        elif hasattr(self.policy.frame_encoder, 'output_irreps'):
            encoder_irreps_raw = self.policy.frame_encoder.output_irreps
        else:
            _LOG.debug(f"  [WARNING] Could not find encoder irreps directly, using sh_head.adapter_out_irreps")
            encoder_irreps_raw = self.policy.sh_head.adapter_out_irreps
        
        encoder_irreps = Irreps(encoder_irreps_raw) if not isinstance(encoder_irreps_raw, Irreps) else encoder_irreps_raw
        
        # Extract scalar and vector slices
        scalar_slice = None
        vector_slice = None
        offset = 0
        
        for mul, ir in encoder_irreps:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:  # 0e
                if scalar_slice is None:
                    scalar_slice = (offset, offset + dim)
            elif ir.l == 1 and ir.p == -1:  # 1o
                if vector_slice is None:
                    vector_slice = (offset, offset + dim, mul)
            offset += dim
        
        # Check invariance of scalars
        if scalar_slice is not None:
            scalar_orig = embeddings_orig[:, scalar_slice[0]:scalar_slice[1]]
            scalar_rot = embeddings_rot[:, scalar_slice[0]:scalar_slice[1]]
            scalar_error = (scalar_orig - scalar_rot).abs().max().item()
        else:
            scalar_error = 0.0
        
        # Check equivariance of vectors
        if vector_slice is not None:
            v_start, v_end, v_mul = vector_slice
            vectors_orig = embeddings_orig[:, v_start:v_end].reshape(batch_size, v_mul, 3)
            vectors_rot = embeddings_rot[:, v_start:v_end].reshape(batch_size, v_mul, 3)
            
            vectors_orig_rotated = torch.bmm(R, vectors_orig.transpose(-2, -1)).transpose(-2, -1)
            vector_error = (vectors_orig_rotated - vectors_rot).abs().max().item()
        else:
            vector_error = 0.0
        
        is_invariant = scalar_error < tolerance and vector_error < tolerance
        
        if not is_invariant:
            _LOG.debug(f"[POLICY INVARIANCE TEST FAILED]")
            _LOG.debug(f"  Scalar error: {scalar_error:.3e} (tolerance: {tolerance})")
            _LOG.debug(f"  Vector error: {vector_error:.3e} (tolerance: {tolerance})")
        
        return is_invariant
    
    def _log_equivariance_summary(self, results: Dict, episode: int):
        """Log summary of equivariance tests."""
        
        # Count passes/failures
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for test_name, passed in category_results.items():
                    total_tests += 1
                    if passed:
                        passed_tests += 1
            elif isinstance(category_results, bool):
                total_tests += 1
                if category_results:
                    passed_tests += 1
        
        # Log summary
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            if pass_rate == 100:
                _LOG.debug(f"Episode {episode}: All {total_tests} equivariance tests passed")
            else:
                _LOG.debug(f"Episode {episode}: {passed_tests}/{total_tests} equivariance tests passed ({pass_rate:.1f}%)")
