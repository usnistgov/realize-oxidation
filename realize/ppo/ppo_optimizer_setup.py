"""Optimizer construction and backbone parameter injection."""
import torch
from typing import Dict, Optional, Any


class PPOSetupMixin:
    """Optimizer construction + backbone parameter injection + replay-batch
    setters for PPOOptimizer.

    ``_setup_optimizer`` (param-group build) runs at construction; the
    inject_*/set_* methods configure backbone fine-tuning and replay buffers. All
    resolve policy/config/device/optimizer state through the MRO back to
    PPOOptimizer (which is NOT an nn.Module and owns no nn.Parameters of its own --
    it only builds torch.optim groups over the policy's parameters), so this mixin
    cannot perturb the policy fingerprint.
    """
    def _setup_optimizer(self):
        """Build optimizer parameter groups."""
        optimizer_params = []
        # Accept both full PPOConfig and standalone AlgorithmConfig
        if hasattr(self.config, 'algorithm'):
            base_lr = self.config.algorithm.lr
        else:
            base_lr = self.config.lr

        seen_params = set()

        def add_param_group(name, params, lr, wd):
            """Add a parameter group; raises on duplicate parameters or empty group."""
            param_list = []
            for p in params:
                param_id = id(p)
                if param_id in seen_params:
                    raise RuntimeError(
                        f"Duplicate parameter detected in group '{name}'. "
                        f"Parameter already assigned to another group. "
                        f"Check for overlapping parameter selections in _setup_optimizer()."
                    )
                param_list.append(p)
                seen_params.add(param_id)
            
            if not param_list:
                raise RuntimeError(
                    f"Parameter group '{name}' is empty. "
                    f"This indicates pattern mismatch in pick() or missing module. "
                    f"Check that the policy has all expected components."
                )
            
            optimizer_params.append({'params': param_list, 'lr': lr, 'weight_decay': wd, 'name': name})            
            self._log("OPTIMIZER", f"Group '{name}'", num_params=len(param_list), lr=f"{lr:.2e}", wd=wd)

        # Collect SH head parameters by category
        sh_named_params = list(self.policy.sh_head.named_parameters())

        def pick(substrs, exclude=()):
            """Parameters whose name matches a substring and none of exclude."""
            keep = []
            for n, p in sh_named_params:
                if any(s in n for s in substrs) and not any(x in n for x in exclude):
                    keep.append(p)
            return keep
    
        # 1. Channel linears (e3nn Linear layers - core equivariant projection)
        channel_linear_params = pick(['channel_linears'])
        if channel_linear_params:
            add_param_group('sh_channel_linears', channel_linear_params, base_lr * 2.0, 1e-5)
        else:
            self._log("OPTIMIZER", "channel_linears not found - skipping")
    
        # 2. Gate network (channel weighting)
        gate_params = pick(['gate_network', 'channel_gate', 'gate_input_proj'])        
        if gate_params:
            add_param_group('sh_gate', gate_params, base_lr * 2.0, 0.0)
        else:
            self._log("OPTIMIZER", "channel_gate/gate_input_proj not found - skipping")

        # 2.5. Cross TP layers (generates actual SH coefficients)
        cross_tp_params = pick(['cross_tp'])
        if cross_tp_params:
            add_param_group('sh_cross_tp', cross_tp_params, base_lr * 2.0, 1e-5)
        else:
            self._log("OPTIMIZER", "cross_tp not found - skipping")

        # 3. Everything else in SH head (catch-all)
        exclude_list = ['channel_linears', 'channel_gate', 'gate_input_proj', 'gate_network', 'cross_tp']
        exclude_list_with_beta = exclude_list  # log_beta stays in sh_other
        sh_other = [p for n, p in sh_named_params if all(k not in n for k in exclude_list)]
        if sh_other:
            add_param_group('sh_head_other', sh_other, base_lr * 0.5, 1e-5)
        else:
            self._log("OPTIMIZER", "sh_head_other empty - skipping")

        # 5. Frame encoder (conditional - might be empty in thin wrappers)
        frame_params = list(self.policy.frame_encoder.parameters())
        if frame_params:
            add_param_group('frame_encoder', frame_params, base_lr*0.33, 1e-5)
        else:
            self._log("OPTIMIZER", "frame_encoder has no parameters - skipping")
    
        # 6. Magnitude predictor (conditional - check both components)
        mag_params = list(self.policy.magnitude_predictor.parameters())
        if hasattr(self.policy, 'magnitude_from_frame_with_radius'):
            mag_params.extend(self.policy.magnitude_from_frame_with_radius.parameters())
        
        if mag_params:
            add_param_group('magnitude_nets', mag_params, base_lr*2.0, 1e-5)
        else:
            self._log("OPTIMIZER", "magnitude_nets have no parameters - skipping")
    
        # 7. Axis network (conditional)
        if hasattr(self.policy, 'axis_from_frame'):
            axis_params = list(self.policy.axis_from_frame.parameters())
            if axis_params:
                add_param_group('axis_net', axis_params, base_lr * 2.0, 1e-5)
            else:
                self._log("OPTIMIZER", "axis_from_frame has no parameters - skipping")
        # 7a. Compass + multipole injector params (log_beta already lives in
        #     sh_head_other). compass_scale is inert but registered; append it first,
        #     then injector params, to fix compass_multipole group ordering.
        compass_params = []
        if hasattr(self.policy, 'compass_scale'):
            compass_params.append(self.policy.compass_scale)
        if hasattr(self.policy, 'multipole_injector'):
            compass_params.extend(list(self.policy.multipole_injector.parameters()))
        if compass_params:
            add_param_group('compass_multipole', compass_params, base_lr * 5.0, 0.0)
        else:
            self._log("OPTIMIZER", "No compass/multipole parameters found")

        # 7b. O2 embedder params, via the embedding extractor when available.
        #     The default combiner has 0 params; this includes a trainable one if used.
        if hasattr(self, '_embedding_extractor') and self._embedding_extractor is not None:
            o2_emb_params = self._embedding_extractor.get_o2_embedder_parameters()
            if o2_emb_params:
                add_param_group('o2_embedder', o2_emb_params, base_lr * 0.5, 1e-5)
                self._log("OPTIMIZER", f"Added {len(o2_emb_params)} o2_embedder parameters")
            else:
                self._log("OPTIMIZER", "o2_embedder has no trainable parameters (expected for SimpleO2Combiner)")

        # 8. Distance estimator (conditional)
        if hasattr(self.policy, 'distance_estimator'):
            dist_params = list(self.policy.distance_estimator.parameters())
            if dist_params:
                add_param_group('distance_estimator', dist_params, base_lr, 1e-5)
            else:
                self._log("OPTIMIZER", "distance_estimator has no parameters - skipping")

        # 9. Value head (components must exist)
        value_params = []

        value_head_components = ['critic_embedder', 'value_mlp']
        missing_components = []
        
        for component_name in value_head_components:
            if hasattr(self.policy, component_name):
                component = getattr(self.policy, component_name)
                value_params.extend(component.parameters())
            else:
                missing_components.append(component_name)
        
        if missing_components:
            raise RuntimeError(
                f"Value head components missing: {missing_components}. "
                f"Policy must have all value head components: {value_head_components}. "
                f"Check policy initialization."
            )
        
        add_param_group('value_head', value_params, base_lr * 2.0, 1e-5)  # value_head uses base_lr * 2.0

        # Backbone params (if training enabled). The backbone lives in
        # episode_runner.env.embedding_extractor, not in the policy.
        backbone_params = []
        if hasattr(self.policy, 'frame_encoder'):
            pass

        if hasattr(self.config, 'backbone') and self.config.backbone.train_backbone:
            self._log("BACKBONE", "Backbone training ENABLED - looking for EmbeddingExtractor")
            self._log("BACKBONE", "Backbone params will be added via inject_backbone_parameters()")
        
        if not optimizer_params:
            raise RuntimeError(
                "No parameters collected for optimizer. "
                "This indicates all parameter groups were empty or policy has no trainable parameters. "
                "Check policy initialization and parameter group patterns."
            )

        if len(seen_params) == 0:
            raise RuntimeError(
                "No unique parameters found in policy. "
                "Policy must have trainable parameters."
            )

        # COVERAGE INVARIANT (fail-fast): every trainable policy parameter must
        # land in exactly one optimizer group. A reorg that renames/moves a
        # submodule must never silently drop params from training -- that is the
        # exact regression that left the SH/direction path un-optimized in
        # test62 (clip_frac=0%, log_beta frozen, 0% dissociation). If any
        # requires_grad param is unassigned, crash here instead of training a
        # crippled policy for hundreds of episodes.
        _assigned_ids = set(seen_params)
        _orphans = [
            name for name, p in self.policy.named_parameters()
            if p.requires_grad and id(p) not in _assigned_ids
        ]
        if _orphans:
            raise RuntimeError(
                f"{len(_orphans)} trainable policy parameter(s) are in NO "
                f"optimizer group and would never train: {_orphans}. "
                f"Add a param group covering them in _setup_optimizer()."
            )

        self.optimizer = torch.optim.Adam(optimizer_params)

        self._log("OPTIMIZER", "Optimizer created",
                  total_groups=len(optimizer_params),
                  total_unique_params=len(seen_params))

    def set_embedding_extractor(self, embedding_extractor):
        """Store the actor's embedding extractor (for o2_embedder param access),
        before optimizer creation.
        """
        self._embedding_extractor = embedding_extractor

    def inject_backbone_parameters(self, embedding_extractor):
        """Add backbone params to the optimizer after init. The extractor lives in
        episode_runner (not the policy), so this runs once episode_runner exists.
        """
        if not hasattr(self.config, 'backbone') or not self.config.backbone.train_backbone:
            self._log("BACKBONE", "Backbone training disabled - skipping injection")
            return

        backbone_params = embedding_extractor.get_backbone_parameters()

        if not backbone_params:
            self._log("BACKBONE", "No trainable backbone parameters found")
            return

        # Same base_lr resolution as _setup_optimizer
        if hasattr(self.config, 'algorithm'):
            base_lr = self.config.algorithm.lr
        else:
            base_lr = self.config.lr
        backbone_lr = base_lr * self.config.backbone.backbone_lr_multiplier

        self._log("BACKBONE", "Injecting backbone parameters",
                  num_params=len(backbone_params),
                  backbone_lr=f"{backbone_lr:.2e}",
                  multiplier=self.config.backbone.backbone_lr_multiplier)

        self.optimizer.add_param_group({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': 1e-5
        })

        self._log("BACKBONE", "Backbone parameters injected successfully",
                  total_groups=len(self.optimizer.param_groups))

    def inject_critic_backbone_parameters(self, critic_extractor):
        """Add critic backbone params to the optimizer (only when
        config.backbone.critic_train_backbone is True). Uses a separate, typically
        smaller LR so large backbone updates don't destabilize the value function.
        """
        if not hasattr(self.config, 'backbone') or not self.config.backbone.critic_train_backbone:
            self._log("CRITIC_BACKBONE", "Critic backbone training disabled - skipping")
            return

        critic_params = critic_extractor.get_backbone_parameters()
        if not critic_params:
            self._log("CRITIC_BACKBONE", "No trainable critic backbone parameters found")
            return

        # Critic-specific LR multiplier from config
        if hasattr(self.config, 'algorithm'):
            base_lr = self.config.algorithm.lr
        else:
            base_lr = self.config.lr
    
        critic_lr = base_lr * self.config.backbone.critic_backbone_lr_multiplier
    
        self._log("CRITIC_BACKBONE", "Injecting critic backbone parameters",
                  num_params=len(critic_params),
                  critic_lr=f"{critic_lr:.2e}",
                  multiplier=self.config.backbone.critic_backbone_lr_multiplier)
    
        self.optimizer.add_param_group({
            'params': critic_params,
            'lr': critic_lr,
            'weight_decay': 1e-5
        })
    
        self._log("CRITIC_BACKBONE", "Critic backbone parameters injected",
                  total_groups=len(self.optimizer.param_groups))

    def set_replay_batch(self, replay_batch: Optional[Dict[str, torch.Tensor]]):
        """Set the critic replay batch for the next optimize() call."""
        self._current_replay_batch = replay_batch

    def set_policy_replay_batch(self, replay_batch: Optional[Dict[str, Any]]):
        """Set the Stage 2 policy replay batch for the next optimize() call."""
        self._current_policy_replay_batch = replay_batch

