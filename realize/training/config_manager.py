import copy
from typing import Any, Dict, Union
from dataclasses import is_dataclass, fields
import logging
_LOG = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration updates from command line arguments."""
    
    def __init__(self, default_config):
        self.default_config = default_config
        self.is_ppo_config = self._is_ppo_config(default_config)
    
    def _is_ppo_config(self, config) -> bool:
        """Check if config is a structured PPOConfig (has nested sections)."""
        return (hasattr(config, 'model') and
                hasattr(config, 'runtime') and
                hasattr(config, 'algorithm'))
    
    def _deep_copy_config(self, config):
        """Create a deep copy preserving the original type."""
        if is_dataclass(config):
            return copy.deepcopy(config)
        elif isinstance(config, dict):
            return copy.deepcopy(config)
        else:
            return copy.deepcopy(config)
    
    def update_from_args(self, args):
        """Update config based on command line arguments."""
        config = self._deep_copy_config(self.default_config)

        if self.is_ppo_config:
            self._update_ppo_config(config, args)
        else:
            config = self._update_flat_config(config, args)
        
        return config
    
    def _update_ppo_config(self, config, args):
        """Update structured PPOConfig from args."""
        # Runtime parameters
        if args.episodes is not None:
            config.runtime.total_episodes = args.episodes
        if args.steps is not None:
            config.runtime.max_steps = args.steps
        if args.random_seed is not None:
            if hasattr(config.runtime, 'seed'):
                config.runtime.seed = args.random_seed
            else:
                setattr(config.runtime, 'seed', args.random_seed)
        if args.save_interval is not None:
            config.runtime.save_interval = args.save_interval
        
        # Model parameters (spherical harmonics)
        if args.lmax is not None:
            config.model.lmax = args.lmax
        if args.num_channels is not None:
            config.model.num_channels = args.num_channels
        
        # Sampling parameters (temperatures)
        if args.temperature_init is not None:
            config.sampling.temperature_init = args.temperature_init
        if args.temperature_min is not None:
            config.sampling.temperature_min = args.temperature_min
        
        # Algorithm parameters (PER)
        if hasattr(args, 'use_per') and args.use_per:
            if not hasattr(config.algorithm, 'use_per'):
                setattr(config.algorithm, 'use_per', True)
            else:
                config.algorithm.use_per = True
            
            if args.per_alpha is not None:
                if not hasattr(config.algorithm, 'per_alpha'):
                    setattr(config.algorithm, 'per_alpha', args.per_alpha)
                else:
                    config.algorithm.per_alpha = args.per_alpha
            
            if args.per_beta is not None:
                if not hasattr(config.algorithm, 'per_beta'):
                    setattr(config.algorithm, 'per_beta', args.per_beta)
                else:
                    config.algorithm.per_beta = args.per_beta
        
        # Reward parameters (NEB)
        if hasattr(args, 'use_neb') and args.use_neb is not None:
            use_neb_value = bool(args.use_neb)
            if hasattr(config, 'reward'):
                if not hasattr(config.reward, 'use_neb_reward'):
                    setattr(config.reward, 'use_neb_reward', use_neb_value)
                else:
                    config.reward.use_neb_reward = use_neb_value

                # Alias.
                if not hasattr(config.reward, 'use_neb'):
                    setattr(config.reward, 'use_neb', use_neb_value)
    
    def _update_flat_config(self, config, args):
        """Update flat dict config from args."""
        # Ensure config is a dict for updating.
        if not isinstance(config, dict):
            config_dict = {}
            for key in dir(config):
                if not key.startswith('_'):
                    config_dict[key] = getattr(config, key)
            config = config_dict
        
        # Update flat config values
        if args.episodes is not None:
            config['total_episodes'] = args.episodes
        if args.steps is not None:
            config['max_steps'] = args.steps
        if args.random_seed is not None:
            config['random_seed'] = args.random_seed
            config['seed'] = args.random_seed  # Alias
        if args.save_interval is not None:
            config['save_interval'] = args.save_interval
        
        # Spherical harmonics
        if args.lmax is not None:
            config['lmax'] = args.lmax
        if args.num_channels is not None:
            config['num_channels'] = args.num_channels
        if args.temperature_init is not None:
            config['temperature_init'] = args.temperature_init
        if args.temperature_min is not None:
            config['temperature_min'] = args.temperature_min
        
        # Training options
        if hasattr(args, 'use_per') and args.use_per:
            config['use_per'] = True
            if args.per_alpha is not None:
                config['per_alpha'] = args.per_alpha
            if args.per_beta is not None:
                config['per_beta'] = args.per_beta
        
        # NEB reward - only update if provided
        if hasattr(args, 'use_neb') and args.use_neb is not None:
            config['use_neb_reward'] = bool(args.use_neb)
            config['use_neb'] = bool(args.use_neb)  # Alias
        
        # Convert back to object if original was object-like
        if not isinstance(self.default_config, dict):
            class Config:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            return Config(config)
        
        return config
    
    def print_summary(self, config):
        """Print configuration summary."""
        _LOG.info("\nConfiguration Summary:")
        _LOG.info("-" * 40)
        
        if self.is_ppo_config:
            # Print structured config
            sections = ['model', 'runtime', 'algorithm', 'sampling', 'reward', 'channels']
            for section in sections:
                if hasattr(config, section):
                    _LOG.info(f"\n{section.upper()}:")
                    section_obj = getattr(config, section)
                    
                    if is_dataclass(section_obj):
                        # For dataclasses, iterate through fields
                        for field in fields(section_obj):
                            value = getattr(section_obj, field.name)
                            _LOG.info(f"  {field.name}: {value}")
                    else:
                        # For regular objects, iterate through attributes
                        for key in dir(section_obj):
                            if not key.startswith('_'):
                                value = getattr(section_obj, key)
                                if not callable(value):
                                    _LOG.info(f"  {key}: {value}")
        else:
            # Print flat config
            if isinstance(config, dict):
                for key in sorted(config.keys()):
                    _LOG.info(f"  {key}: {config[key]}")
            else:
                for key in sorted(dir(config)):
                    if not key.startswith('_'):
                        value = getattr(config, key)
                        if not callable(value):
                            _LOG.info(f"  {key}: {value}")
        
        _LOG.info("-" * 40 + "\n")
