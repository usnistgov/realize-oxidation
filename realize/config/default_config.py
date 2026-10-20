"""Configuration dataclasses and their defaults."""
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import warnings
import numpy as np


@dataclass
class ModelConfig:
    """Neural network architecture parameters"""
    # E3nn embedding
    embedding_irreps: str = "128x0e+64x1e+32x2e"  
    encoder_output_irreps: str = "16x0e+8x1e+4x2e"  
    use_symmetric_embedding: bool = True
    scalar_dim: int = 32
    vector_dim: int = 8   # number of vector channels (8x1o)

    # Hidden dimensions
    mlp_hidden_dim: int = 128 
    
    # Spherical harmonics
    lmax: int = 4
    num_channels: int = 1

    use_continuous_direction: bool = True
    direction_perturb_std: float = 0.1  # tangent-plane noise std (radians)

    use_langevin_direction: bool = True
    langevin_steps: int = 10
    langevin_step_size: float = 0.03
    langevin_noise_scale: float = 0.05

    # S^2 nonlinearity block
    enable_s2_block: bool = True
    s2_grid_resolution: int = 32  # S^2 grid is 32x32
    s2_activation: Literal['relu', 'gelu', 'tanh'] = 'gelu'
    s2_convolution_kernel_size: int = 3
    legendre_max_degree: int = 10  # for Legendre-parameterized kernel
    
    # Frame alignment
    use_frame_alignment: bool = True
    frame_encoder_hidden: str = "64x0e+32x1e+16x2e+8x3e" 
    frame_encoder_output: str = "16x0e+8x1e+4x2e" 

    
    # Radial conditioning
    use_radial_conditioning: bool = True
    radial_num_basis: int = 32
    radial_max_radius: float = 10.0
    
    # Magnitude prediction
    max_trans_distance: float = 3.0
    rqs_flow_enabled: bool = True  # Rational Quadratic Spline flow
    r_min: float = 0.5  # min translation distance
    r_max: float = 3.0  # max translation distance (== max_trans_distance)
    rqs_num_bins: int = 8
    rqs_num_layers: int = 2
    use_clamped_magnitude: bool = False  # fallback to clamped scalar head
        
    
    def validate(self):
        assert self.lmax >= 0, "lmax must be non-negative"
        assert self.num_channels >= 1, "Need at least 1 channel"
        assert self.scalar_dim > 0, "scalar_dim must be positive"
        assert self.mlp_hidden_dim > 0, "mlp_hidden_dim must be positive"
        assert self.max_trans_distance > 0, "max_trans_distance must be positive"

@dataclass
class ChannelConfig:
    """Channel gating and temperature parameters"""
    num_channels: int = 1
    lmax_per_channel: List[int] = field(default_factory=lambda: [3,3,2,2])
    gate_hidden: List[int] = field(default_factory=lambda: [64,64])
    gate_dropout: float = 0.05
    gate_activation: str = "gelu"
    gate_norm: Optional[str] = "layernorm"  # or "batchnorm" or None
    gate_bias: bool = True
    gate_temp_init: float = 1.0
    gate_temp_min: float = 0.35
    gate_temp_anneal: float = 0.997
    gate_hard: bool = False

    # Channel gates
    gate_activation: Literal['sigmoid', 'softmax'] = 'softmax'
    gate_bias_init: float = 0.0
    gate_temperature: float = 1.0
    gate_threshold: float = 0.1
    min_active_channels: int = 1
    
    # Channel temperatures (unused; active architecture uses per-channel beta concentration)
    channel_temp_scales: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    channel_temp_min: float = 0.01
    channel_temp_max: float = 5.0
    channel_temp_learning_rate: float = 0.01
    learnable_channel_temps: bool = True
    channel_temp_init_strategy: Literal['uniform', 'diverse', 'fixed'] = 'fixed'

    # Channel diversity
    channel_diversity_weight: float = 1.0  # currently unused
    channel_diversity_annealing: bool = True
    channel_similarity_threshold: float = 0.6
    channel_entropy_weight: float = 0.01  # entropy regularization on channel weights
    channel_dirichlet_alpha: float = 1.0  # Dirichlet prior strength for channel weights
    channel_reinit_threshold: int = 50  # episodes before reinitializing a collapsed channel
    monitor_channel_collapse: bool = True
    
    def validate(self):
        assert self.gate_temperature > 0, "gate_temperature must be positive"
        assert 0 <= self.gate_threshold <= 1, "gate_threshold must be in [0,1]"
        assert self.channel_temp_min > 0, "channel_temp_min must be positive"
        assert self.channel_temp_max > self.channel_temp_min, "temp_max must exceed temp_min"
        assert all(t > 0 for t in self.channel_temp_scales), "All temp scales must be positive"
        assert self.min_active_channels >= 1, "Need at least 1 active channel"
        assert self.channel_diversity_weight >= 0, "channel_diversity_weight must be non-negative"
        assert 0 < self.channel_similarity_threshold <= 1, "channel_similarity_threshold must be in (0,1]"

    def get_channel_temperature(self, channel_idx: int) -> float:
        """Temperature for a channel. Raises IndexError if channel_idx is out of range."""
        if not (0 <= channel_idx < len(self.channel_temp_scales)):
            raise IndexError(
                f"channel_idx {channel_idx} out of range for "
                f"{len(self.channel_temp_scales)} channel_temp_scales. "
                f"Config must specify temperature for every channel. "
                f"Check that len(channel_temp_scales) >= num_channels."
            )
        
        return self.channel_temp_scales[channel_idx]

@dataclass
class SamplingConfig:
    """Direction sampling parameters"""
    # Grid type must be explicit ('auto' not allowed).
    grid_type: Literal['lebedev', 'fibonacci'] = 'fibonacci'
    fibonacci_points: int = 4096
    lebedev_degree: int = 5  # 5->50, 9->110, 11->170 points
    num_grid_points: int = 1000  # base grid points for distribution creation

    # Dense Gauss-Legendre sampling (instead of sparse grid)
    use_dense_gl: bool = True
    gl_n_beta: int = 64   # polar angle resolution
    gl_n_alpha: int = 64  # azimuthal angle resolution

    # Sampling temperature
    temperature_init: float = 1.0
    temperature_min: float = 0.01
    temperature_decay: float = 0.99
    temperature_anneal_episodes: int = 200   

    # Langevin sampling
    use_langevin_sampling: bool = False
    langevin_warmup_episodes: int = 50
    langevin_num_steps: int = 10
    langevin_init_step_size: float = 0.1
    langevin_use_mh_correction: bool = True
    langevin_decay_schedule: Literal['linear', 'exponential', 'cosine'] = 'linear'

    # Eval-time Langevin refinement
    eval_langevin_enabled: bool = False
    eval_langevin_steps: int = 5
    eval_langevin_step_size: float = 0.05

    use_cache: bool = True  # cache SH evaluations

    # Equivariance debugging (DOES change RNG/numerics; keep off for training)
    debug_equivariance: bool = True
    equivariance_check_frequency: int = 20  # check every N episodes
    debug_frame_encoder: bool = True
    equivariance_tolerance: float = 1e-5

    # Gumbel-Softmax
    gumbel_temperature_init: float = 1.0
    gumbel_temperature_min: float = 0.1
    gumbel_temperature_anneal_episodes: int = 300
    use_straight_through: bool = True  # straight-through estimator for gradients

    def validate(self):
        if self.grid_type == 'auto':
            raise ValueError(
                "grid_type='auto' is not allowed. "
                "Must explicitly choose 'lebedev' or 'fibonacci'. "
                "Set sampling.grid_type to one of: 'lebedev', 'fibonacci'"
            )        
        assert self.fibonacci_points > 0, "fibonacci_points must be positive"
        assert self.lebedev_degree in [5, 9, 11], "lebedev_degree must be 5, 9, or 11"
        assert self.temperature_init > 0, "temperature_init must be positive"
        assert self.temperature_min > 0, "temperature_min must be positive"
        assert 0 < self.temperature_decay <= 1, "temperature_decay must be in (0,1]"
        assert self.langevin_num_steps > 0, "langevin_num_steps must be positive"
        assert self.langevin_init_step_size > 0, "langevin_init_step_size must be positive"
        assert self.gumbel_temperature_init > 0, "gumbel_temperature_init must be positive"
        assert self.gumbel_temperature_min > 0, "gumbel_temperature_min must be positive"
        assert self.gumbel_temperature_anneal_episodes > 0, "gumbel_temperature_anneal_episodes must be positive"        
        assert self.gl_n_beta > 0, "gl_n_beta must be positive"
        assert self.gl_n_alpha > 0, "gl_n_alpha must be positive"
        assert self.num_grid_points > 0, "num_grid_points must be positive"

    def get_grid_points(self) -> int:
        """Actual number of grid points for the configured grid_type. Raises ValueError if invalid."""
        if self.grid_type == 'lebedev':
            mapping = {5: 50, 9: 110, 11: 170}
            if self.lebedev_degree not in mapping:
                raise ValueError(
                    f"Invalid lebedev_degree: {self.lebedev_degree}. "
                    f"Must be one of: {list(mapping.keys())}"
                )
            return mapping[self.lebedev_degree]
        elif self.grid_type == 'fibonacci':
            return self.fibonacci_points
        else:
            raise ValueError(
                f"Invalid grid_type: {self.grid_type}. "
                f"Must be 'lebedev' or 'fibonacci'."
            )

@dataclass
class PPOAlgorithmConfig:
    """PPO hyperparameters, tuned for sparse rewards."""
    lr: float = 3e-4
    gamma: float = 0.99
    eps_clip: float = 0.3
    k_epochs: int = 6
    critic_extra_epochs: int = 25
    mc_warmup_episodes: int = 0

    # GAE lambda: bias-variance tradeoff (0=one-step TD, 1=Monte Carlo).
    gae_lambda: float = 0.95  # 0.97-0.98 for very sparse rewards

    # Normalize rewards by running std to stabilize training.
    normalize_rewards: bool = False

    # Oversample successful (dissociation) episodes; 3.0 = 3x more likely, 1.0 = uniform.
    success_oversample_ratio: float = 3.0
    
    # Entropy
    entropy_coeff: float = 0.005
    entropy_anneal: bool = True
    entropy_anneal_episodes: int = 1000
    final_entropy_coeff: float = 0.0005
    
    # Loss coefficients (balance policy vs value learning)
    policy_loss_coeff: float = 1.0
    value_loss_coeff: float = 0.5
    clip_value_loss: bool = False
    value_clip_range: float = 1.0  
    normalize_value: bool = True

    # Gradient clipping
    max_grad_norm: float = 5.0
    grad_clip_enabled: bool = True
    
    # KL regularization (optional)
    use_kl_regularization: bool = False
    kl_target: float = 0.01
    kl_coeff: float = 0.01
    
    def validate(self):
        assert self.lr > 0, "lr must be positive"
        assert 0 <= self.gamma <= 1, "gamma must be in [0,1]"
        assert 0 < self.eps_clip < 1, "eps_clip must be in (0,1)"
        assert self.k_epochs > 0, "k_epochs must be positive"
        assert self.entropy_coeff >= 0, "entropy_coeff must be non-negative"
        assert self.final_entropy_coeff >= 0, "final_entropy_coeff must be non-negative"
        assert self.value_loss_coeff > 0, "value_loss_coeff must be positive"
        
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0,1]"
        assert self.value_clip_range > 0, "value_clip_range must be positive"
        assert self.success_oversample_ratio >= 1.0, "success_oversample_ratio must be >= 1.0"
    
    def get_entropy_coeff(self, episode: int) -> float:
        """Get annealed entropy coefficient"""
        if not self.entropy_anneal:
            return self.entropy_coeff
        
        if episode >= self.entropy_anneal_episodes:
            return self.final_entropy_coeff
        
        progress = episode / self.entropy_anneal_episodes
        return self.entropy_coeff * (1 - progress) + self.final_entropy_coeff * progress

@dataclass
class RewardConfig:
    """Reward function parameters"""
    # Main reward components
    use_neb_reward: bool = True
    ea_weight: float = 4.0
    dissociation_weight: float = 2.0
    pe_reward_weight: float = 0.01
    
    # Beta-solver area@mass targets, per channel: sharpest -> broadest.
    area_at_mass_targets: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.35, 0.50])
    use_learnable_beta: bool = True        # Policy learns distribution sharpness (single-channel mode)
    learnable_beta_init: float = 2.0
    learnable_beta_min: float = 0.5        # Prevent uniform collapse
    learnable_beta_max: float = 25.0       # Prevent delta collapse
    target_mass_fraction: float = 0.70  # For A@70% metric
    
    # Success criteria
    require_dissociation: bool = True
    success_reward_weight: float = 1.0
    
    # Penalties
    timeout_penalty: float = 0.5
    invalid_step_penalty: float = 0.005
    high_barrier_threshold: float = 2.5
    high_barrier_penalty: float = 0.001
    oscillation_penalty: float = 0.005
    
    # Potential-based PE shaping: F(s,s') = PE(s) - gamma*PE(s')
    # Ensures cumulative PE reward depends only on start/end states, not path length
    pe_shaping_gamma: float = 0.99
    
    # Max cumulative invalid penalty per episode (keeps vacuum from being an escape route).
    invalid_penalty_cap: float = -0.3
    
    # Sigmoid shaping for Ea
    ea_sigmoid_k: float = 0.086
    ea_sigmoid_midpoint: float = 1.6
    
    # Reward processing
    cap_unsuccessful_reward: bool = True
    Ea_gamma: float = 0.99
    use_translation_in_pathway: bool = True
    
    def validate(self):
        assert self.ea_weight >= 0, "ea_weight must be non-negative"
        assert self.dissociation_weight >= 0, "dissociation_weight must be non-negative"
        assert all(p >= 0 for p in [self.timeout_penalty, self.invalid_step_penalty, 
                                     self.high_barrier_penalty, self.oscillation_penalty]), \
               "Penalties must be non-negative"
        assert self.ea_sigmoid_k > 0, "ea_sigmoid_k must be positive"
        assert self.ea_sigmoid_midpoint > 0, "ea_sigmoid_midpoint must be positive"
        assert len(self.area_at_mass_targets) > 0, "area_at_mass_targets must not be empty"
        assert all(0 < a <= 1 for a in self.area_at_mass_targets), "area_at_mass_targets must be in (0,1]"
        assert 0 < self.target_mass_fraction <= 1, "target_mass_fraction must be in (0,1]"
        assert 0 < self.pe_shaping_gamma <= 1, "pe_shaping_gamma must be in (0,1]"
        assert self.invalid_penalty_cap < 0, "invalid_penalty_cap must be negative"

@dataclass
class BackboneConfig:
    """SevenNet backbone fine-tuning parameters"""
    # Actor backbone
    train_backbone: bool = True
    unfreeze_mode: Literal['last_layer', 'last_2_layers', 'last_3_layers', 'all'] = 'last_2_layers'
    backbone_lr_multiplier: float = 0.003  # backbone_lr = policy_lr * this
    backbone_gradient_clip: float = 0.1    # tighter than policy
    warm_start_episodes: int = 150         # keep actor backbone frozen for N episodes

    # Critic backbone (frozen by default for stable value targets)
    critic_train_backbone: bool = False
    critic_unfreeze_mode: Literal['last_layer', 'last_2_layers', 'last_3_layers', 'all'] = 'last_layer'
    critic_backbone_lr_multiplier: float = 0.005  # critic_backbone_lr = policy_lr * this
    critic_backbone_gradient_clip: float = 0.05
    
    def validate(self):
        assert self.unfreeze_mode in ['last_layer', 'last_2_layers', 'last_3_layers', 'all'], \
            f"unfreeze_mode must be one of: last_layer, last_2_layers, last_3_layers, all"
        assert 0 < self.backbone_lr_multiplier <= 1.0, \
            "backbone_lr_multiplier must be in (0, 1.0]"
        assert self.backbone_gradient_clip > 0, \
            "backbone_gradient_clip must be positive"
        assert self.warm_start_episodes >= 0, \
            "warm_start_episodes must be non-negative"

        assert self.critic_unfreeze_mode in ['last_layer', 'last_2_layers', 'last_3_layers', 'all'], \
            f"critic_unfreeze_mode must be one of: last_layer, last_2_layers, last_3_layers, all"
        assert 0 < self.critic_backbone_lr_multiplier <= 1.0, \
            "critic_backbone_lr_multiplier must be in (0, 1.0]"
        assert self.critic_backbone_gradient_clip > 0, \
            "critic_backbone_gradient_clip must be positive"

        if self.train_backbone and self.critic_train_backbone:
            warnings.warn(
                "Both actor AND critic backbones are trainable. "
                "This can destabilize value learning. "
                "Consider using a smaller critic_backbone_lr_multiplier.",
                UserWarning
            )
    
    def get_unfreeze_n_layers(self) -> int:
        """Convert actor unfreeze_mode to number of layers."""
        return self._mode_to_n_layers(self.unfreeze_mode)
    
    def get_critic_unfreeze_n_layers(self) -> int:
        """Convert critic unfreeze_mode to number of layers."""
        return self._mode_to_n_layers(self.critic_unfreeze_mode)
    
    @staticmethod
    def _mode_to_n_layers(mode: str) -> int:
        """Convert unfreeze_mode string to number of layers."""
        if mode == 'last_layer':
            return 1
        elif mode == 'last_2_layers':
            return 2
        elif mode == 'last_3_layers':
            return 3
        elif mode == 'all':
            return 999  # unfreeze all layers
        else:
            raise ValueError(f"Unknown unfreeze_mode: {mode}")

@dataclass
class RuntimeConfig:
    """Training runtime parameters"""
    # Random seed
    seed: int = 42

    # Training duration
    total_episodes: int = 1000
    max_steps: int = 25

    # Episodes accumulated per PPO update (single-episode batches give ~zero
    # advantage variance, so no learning).
    episodes_per_update: int = 24

    # Checkpointing
    save_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    resume_path: Optional[str] = None  # Path to checkpoint to resume from
    
    # Distributed training
    use_shared_model: bool = False
    shared_model_dir: str = "../../shared_net"
    
    # Surface reset (dissociation gets harder as O atoms accumulate). Either condition
    # triggers a reset; set to 0 to disable.
    reset_surface_every_n_episodes: int = 10000  # periodic reset every N episodes
    max_cumulative_o_before_reset: int = 80      # reset when O count reaches threshold

    num_parallel_workers: int = 1
    cores_per_worker: int = 7

    # Logging
    log_interval: int = 1
    save_sh_interval: int = 1
    sh_resolution: int = 64

    # Logging verbosity (stdout only, no effect on numerics):
    #   verbose=False -> INFO: milestones plus the per-episode analysis lines the parsers need.
    #   verbose=True  -> DEBUG: full diagnostic firehose (per-step, gradient detail, etc.).
    # Compute-side debug flags (enable_equivariance_tests, sampling.debug_equivariance,
    # sampling.debug_frame_encoder) DO affect RNG/numerics and are kept separate below.
    verbose: bool = False

    # Debugging
    log_gradient_norms: bool = True
    visualize_sh_coeffs: bool = True
    enable_equivariance_tests: bool = True

    
    # Exploration
    adaptive_exploration: bool = True
    novelty_threshold: float = 0.85
    novelty_memory_size: int = 1000
    
    # Rollback
    max_rollback_attempts: int = 10
    rollback_penalty_multiplier: float = 1.5
    
    def validate(self):
        assert self.total_episodes > 0, "total_episodes must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.save_interval > 0, "save_interval must be positive"
        assert self.novelty_memory_size > 0, "novelty_memory_size must be positive"
        
        assert self.reset_surface_every_n_episodes >= 0, \
            "reset_surface_every_n_episodes must be non-negative (0 to disable)"
        assert self.max_cumulative_o_before_reset >= 0, \
            "max_cumulative_o_before_reset must be non-negative (0 to disable)"

        if self.reset_surface_every_n_episodes > 0 and self.max_cumulative_o_before_reset > 0:
            warnings.warn(
                f"Both reset_surface_every_n_episodes ({self.reset_surface_every_n_episodes}) "
                f"and max_cumulative_o_before_reset ({self.max_cumulative_o_before_reset}) are set. "
                f"Environment will reset when EITHER condition is met.",
                UserWarning
            )    


@dataclass
class PPOConfig:
    """Complete PPO configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    algorithm: PPOAlgorithmConfig = field(default_factory=PPOAlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    
    def validate(self):
        """Validate all sub-configs and check cross-config consistency"""
        self.model.validate()
        self.channels.validate()
        self.sampling.validate()
        self.algorithm.validate()
        self.reward.validate()
        self.runtime.validate()
        self.backbone.validate()

        # Cross-config validation (no auto-extension)
        if len(self.channels.channel_temp_scales) < self.model.num_channels:
            raise ValueError(
                f"Insufficient channel_temp_scales: "
                f"have {len(self.channels.channel_temp_scales)}, "
                f"need {self.model.num_channels} (one per channel). "
                f"Specify channel_temp_scales explicitly for all channels. "
                f"Example: channel_temp_scales=[0.25, 0.5, 1.0, 2.0] for 4 channels."
            )
        
        # area_at_mass_targets length must match num_channels (only when not using learnable beta)
        if not self.reward.use_learnable_beta and len(self.reward.area_at_mass_targets) < self.model.num_channels:
            raise ValueError(
                f"Insufficient area_at_mass_targets: "
                f"have {len(self.reward.area_at_mass_targets)}, "
                f"need {self.model.num_channels} (one per channel). "
                f"Specify area_at_mass_targets explicitly for all channels. "
                f"Example: area_at_mass_targets=[0.18, 0.30, 0.40, 0.50] for 4 channels."
            )

        if self.algorithm.entropy_anneal:
            assert self.algorithm.final_entropy_coeff <= self.algorithm.entropy_coeff, \
                   "final_entropy_coeff should not exceed initial entropy_coeff"
        
        for temp in self.channels.channel_temp_scales:
            assert self.channels.channel_temp_min <= temp <= self.channels.channel_temp_max, \
                   f"Channel temperature {temp} outside bounds [{self.channels.channel_temp_min}, {self.channels.channel_temp_max}]"

    # Convenience accessors for backward compatibility
    @property
    def obs_dim(self) -> int:
        """DEPRECATED: use model.scalar_dim."""
        warnings.warn("obs_dim is deprecated, use model.scalar_dim", DeprecationWarning)
        return self.model.scalar_dim
    
    @property
    def hidden_dim(self) -> int:
        return self.model.mlp_hidden_dim
    
    @property
    def lmax(self) -> int:
        return self.model.lmax
    
    @property
    def num_channels(self) -> int:
        return self.model.num_channels
    
    @property
    def embedding_irreps(self) -> str:
        return self.model.embedding_irreps
    
    @property
    def lr(self) -> float:
        return self.algorithm.lr
    
    @property
    def total_episodes(self) -> int:
        return self.runtime.total_episodes
    
    @property
    def max_steps(self) -> int:
        return self.runtime.max_steps
    
    @property
    def num_grid_points(self) -> int:
        """Get actual number of grid points based on sampling config"""
        return self.sampling.get_grid_points()
    
    # Convenience accessors for sparse reward improvements
    @property
    def gae_lambda(self) -> float:
        """GAE lambda parameter for advantage estimation"""
        return self.algorithm.gae_lambda
    
    @property
    def normalize_rewards(self) -> bool:
        """Whether to normalize rewards"""
        return self.algorithm.normalize_rewards
    
    @property
    def clip_value_loss(self) -> bool:
        """Whether to clip value function updates"""
        return self.algorithm.clip_value_loss
    
    @property
    def value_clip_range(self) -> float:
        """Clip range for value function updates"""
        return self.algorithm.value_clip_range
    
    @property
    def success_oversample_ratio(self) -> float:
        """How much to oversample successful episodes"""
        return self.algorithm.success_oversample_ratio


def create_debug_config() -> PPOConfig:
    """Create configuration with debug/equivariance testing enabled"""
    config = PPOConfig()

    config.runtime.verbose = True
    config.runtime.enable_equivariance_tests = True
    config.runtime.log_gradient_norms = True

    config.sampling.debug_equivariance = True
    config.sampling.equivariance_check_frequency = 20
    config.sampling.debug_frame_encoder = True
    config.sampling.equivariance_tolerance = 1e-5

    config.channels.monitor_channel_collapse = True

    config.validate()
    return config

def create_production_config() -> PPOConfig:
    """Create optimized configuration for production training"""
    config = PPOConfig()

    config.runtime.verbose = False
    config.runtime.enable_equivariance_tests = False
    config.runtime.log_gradient_norms = False

    config.sampling.debug_equivariance = False
    config.sampling.debug_frame_encoder = False

    config.sampling.use_cache = True

    config.validate()
    return config

def create_default_config() -> PPOConfig:
    """Create default configuration for standard training"""
    config = PPOConfig()

    config.runtime.verbose = False
    config.runtime.enable_equivariance_tests = False
    config.runtime.log_gradient_norms = False

    config.sampling.debug_equivariance = False
    config.sampling.debug_frame_encoder = False
    config.channels.monitor_channel_collapse = True
    config.sampling.use_cache = True

    config.validate()
    return config


def create_sparse_reward_config() -> PPOConfig:
    """
    Create configuration optimized for very sparse rewards (success rate < 5%).
    
    Key adjustments:
    - Higher GAE lambda (0.98) for long-horizon credit assignment
    - Aggressive success oversampling (4.0x)
    - Reward normalization enabled
    - Value clipping enabled with tighter range
    """
    config = PPOConfig()

    config.algorithm.gae_lambda = 0.98  # long-horizon credit
    config.algorithm.normalize_rewards = False
    config.algorithm.clip_value_loss = True
    config.algorithm.value_clip_range = 0.15
    config.algorithm.success_oversample_ratio = 4.0

    # Higher entropy to encourage exploration.
    config.algorithm.entropy_coeff = 0.03
    config.algorithm.final_entropy_coeff = 0.01

    config.runtime.verbose = False
    config.sampling.use_cache = True
    config.channels.monitor_channel_collapse = True

    config.validate()
    return config


# Lazy DEFAULT_CONFIG accessor: builds/validates on first access, so importing this
# module has no side effect (no create_default_config()/validate() at import time).
def __getattr__(name):
    if name == "DEFAULT_CONFIG":
        return create_default_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
