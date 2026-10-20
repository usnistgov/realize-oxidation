"""Gradient-flow diagnostics for the PPO optimizer."""
import numpy as np


class PPOGradientFlowMixin:
    """Gradient-flow diagnostics for PPOOptimizer.

    ``_check_all_gradient_flow``, ``_check_backbone_gradient_flow`` and
    ``_check_axis_gradients`` inspect parameter .grad norms (tensor methods, no
    ``torch.`` module access) and aggregate with numpy; policy/extractors resolve
    through the MRO back to PPOOptimizer (not an nn.Module), so this mixin cannot
    perturb the policy fingerprint.
    """
    def _check_all_gradient_flow(self, policy, episode_idx):
        """Audit that every trainable component receives gradients."""
        self._dlog("GRADIENT", f"Gradient flow audit (episode {episode_idx})")
        
        components_to_check = {
            'SH Head (direction)': policy.sh_head if hasattr(policy, 'sh_head') else None,
            'Frame Encoder': policy.frame_encoder if hasattr(policy, 'frame_encoder') else None,
            'Magnitude Predictor': policy.magnitude_predictor if hasattr(policy, 'magnitude_predictor') else None,
            'Axis Network': policy.axis_from_frame if hasattr(policy, 'axis_from_frame') else None,
            'Critic Embedder': policy.critic_embedder if hasattr(policy, 'critic_embedder') else None,
            'Value MLP': policy.value_mlp if hasattr(policy, 'value_mlp') else None,            
        }
        
        all_healthy = True
        
        for comp_name, component in components_to_check.items():
            if component is None:
                self._dlog("GRADIENT", f"{comp_name}: MISSING")
                continue
            
            total_grad_norm = 0.0
            total_params = 0
            zero_grad_params = 0
            none_grad_params = 0
            
            dead_param_names = []
            for name, param in component.named_parameters():
                total_params += 1
                if param.grad is None:
                    none_grad_params += 1
                    all_healthy = False
                    dead_param_names.append(f"{name} (grad=None)")
                elif param.grad.norm().item() == 0.0:
                    zero_grad_params += 1
                    all_healthy = False
                    dead_param_names.append(f"{name} (grad=0)")
                else:
                    total_grad_norm += param.grad.norm().item() ** 2

            if total_params > 0:
                total_grad_norm = np.sqrt(total_grad_norm)
                
                status = "OK" if (none_grad_params == 0 and zero_grad_params == 0) else "DEAD"
                self._dlog("GRADIENT", f"{comp_name}",
                          grad_norm=f"{total_grad_norm:.3e}",
                          total_params=total_params,
                          none_grad=none_grad_params,
                          zero_grad=zero_grad_params,
                          status=status)
                
                if dead_param_names:
                    for dead_name in dead_param_names:
                        self._dlog("GRADIENT", f"  DEAD: {dead_name}", status="DEAD")
        
        # Per-channel SH head diagnostics
        if hasattr(policy, 'sh_head') and hasattr(policy.sh_head, 'channel_nets'):
            for ch_idx, ch_net in enumerate(policy.sh_head.channel_nets):
                ch_grad_norm = 0.0
                ch_params = 0
                ch_dead = 0
                
                for param in ch_net.parameters():
                    ch_params += 1
                    if param.grad is None or param.grad.norm().item() == 0.0:
                        ch_dead += 1
                    else:
                        ch_grad_norm += param.grad.norm().item() ** 2
                
                if ch_params > 0:
                    ch_grad_norm = np.sqrt(ch_grad_norm)
                    status = "OK" if ch_dead == 0 else "DEAD"
                    self._dlog("CHANNEL", f"Channel[{ch_idx}] gradient flow",
                              grad_norm=f"{ch_grad_norm:.3e}",
                              dead=f"{ch_dead}/{ch_params}",
                              status=status)
        
        # Gate network diagnostics
        gate_attr = None
        if hasattr(policy, 'sh_head'):
            if hasattr(policy.sh_head, 'gate_network'):
                gate_attr = policy.sh_head.gate_network
            elif hasattr(policy.sh_head, 'channel_gate'):
                gate_attr = policy.sh_head.channel_gate
        
        if gate_attr is not None:
            gate_grad_norm = 0.0
            gate_params = 0
            gate_dead = 0
            
            for param in gate_attr.parameters():            
                gate_params += 1
                if param.grad is None or param.grad.norm().item() == 0.0:
                    gate_dead += 1
                else:
                    gate_grad_norm += param.grad.norm().item() ** 2
            
            if gate_params > 0:
                gate_grad_norm = np.sqrt(gate_grad_norm)
                status = "OK" if gate_dead == 0 else "DEAD"
                self._dlog("GRADIENT", "Gate network",
                          grad_norm=f"{gate_grad_norm:.3e}",
                          dead=f"{gate_dead}/{gate_params}",
                          status=status)
        
        if all_healthy:
            self._dlog("GRADIENT", "All components healthy", status="OK")
        else:
            self._dlog("GRADIENT", "Some components have dead gradients", status="WARN")

    def _check_backbone_gradient_flow(self, actor_extractor, critic_extractor, episode_idx):
        """Verify dual-backbone gradient flow: actor backbone has gradients
        (trainable), critic backbone has none (frozen), magnitudes are reasonable.
        """
        self._dlog("BACKBONE", f"Backbone gradient check (episode {episode_idx})")

        actor_grad_norm = 0.0
        actor_params = 0
        actor_dead = 0

        if actor_extractor is not None and hasattr(actor_extractor, 'model'):
            for name, param in actor_extractor.model.named_parameters():
                if param.requires_grad:
                    actor_params += 1
                    if param.grad is None:
                        actor_dead += 1
                    elif param.grad.norm().item() == 0.0:
                        actor_dead += 1
                    else:
                        actor_grad_norm += param.grad.norm().item() ** 2

            actor_grad_norm = np.sqrt(actor_grad_norm) if actor_grad_norm > 0 else 0.0

            if actor_params == 0:
                status = "NO_TRAINABLE"
            elif actor_grad_norm < 1e-8:
                status = "ZERO_GRAD"
            elif actor_dead > actor_params * 0.5:
                status = "MOSTLY_DEAD"
            else:
                status = "OK"

            self._dlog("BACKBONE", "Actor backbone",
                      trainable_params=actor_params,
                      dead_grads=f"{actor_dead}/{actor_params}",
                      grad_norm=f"{actor_grad_norm:.6f}",
                      status=status)

        # Critic backbone should be frozen (no trainable params, no grads)
        critic_trainable = 0
        critic_has_grad = 0

        if critic_extractor is not None and hasattr(critic_extractor, 'model'):
            for param in critic_extractor.model.parameters():
                if param.requires_grad:
                    critic_trainable += 1
                if param.grad is not None:
                    critic_has_grad += 1

            if critic_trainable > 0:
                status = "NOT_FROZEN"
            elif critic_has_grad > 0:
                status = "HAS_GRADS"
            else:
                status = "OK"

            self._dlog("BACKBONE", "Critic backbone",
                      trainable_params=critic_trainable,
                      params_with_grad=critic_has_grad,
                      status=status)

    def _check_axis_gradients(self, policy, episode_idx):
        """Backward-compatible wrapper - calls the comprehensive check instead."""
        self._check_all_gradient_flow(policy, episode_idx)

