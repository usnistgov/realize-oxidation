import torch
import numpy as np
import random
from typing import Dict, Any
from realize.config.default_config import PPOConfig

def set_global_seed(seed: int):
    """Set seed for all random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Reproducible cudnn.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_config(config: PPOConfig):
    """Apply configuration globally"""
    set_global_seed(config.runtime.seed)

    if config.runtime.verbose:
        torch.autograd.set_detect_anomaly(False)

    import logging
    logging.basicConfig(level=logging.DEBUG if config.runtime.verbose else logging.INFO)

def config_to_dict(config: PPOConfig) -> Dict[str, Any]:
    """Convert config to dictionary for serialization"""
    return {
        'model': config.model.__dict__,
        'channels': config.channels.__dict__,
        'sampling': config.sampling.__dict__,
        'algorithm': config.algorithm.__dict__,
        'reward': config.reward.__dict__,
        'runtime': config.runtime.__dict__
    }

def dict_to_config(d: Dict[str, Any]) -> PPOConfig:
    """Reconstruct config from dictionary"""
    from realize.config.default_config import (
        ModelConfig, ChannelConfig, SamplingConfig,
        PPOAlgorithmConfig, RewardConfig, RuntimeConfig
    )
    
    config = PPOConfig(
        model=ModelConfig(**d.get('model', {})),
        channels=ChannelConfig(**d.get('channels', {})),
        sampling=SamplingConfig(**d.get('sampling', {})),
        algorithm=PPOAlgorithmConfig(**d.get('algorithm', {})),
        reward=RewardConfig(**d.get('reward', {})),
        runtime=RuntimeConfig(**d.get('runtime', {}))
    )
    config.validate()
    return config
