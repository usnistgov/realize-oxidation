"""Collects visualization data."""
import numpy as np
from typing import Dict, Any, Optional
from realize.utils.visualization.shape_validator import ShapeValidator


class VisualizationDataLogger:
    """Central logger for visualization data with clean shapes."""
    
    def __init__(self):
        self.data: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.validator = ShapeValidator()
    
    def log_single_channel(self, episode: int, step: int, 
                          sh_coeffs: np.ndarray, o2_positions: np.ndarray,
                          temperature: Optional[float] = None) -> None:
        """Log single-channel SH distribution. sh_coeffs (sh_dim,), o2_positions (2, 3)."""
        if episode not in self.data:
            self.data[episode] = {}

        o2_center = np.mean(o2_positions, axis=0)  # (3,)
        o2_distance = np.linalg.norm(o2_positions[1] - o2_positions[0])

        self.data[episode][step] = {
            'sh_coeffs': sh_coeffs.copy(),
            'o2_positions': o2_positions.copy(),
            'o2_center': o2_center,
            'o2_distance': float(o2_distance),
            'temperature': float(temperature) if temperature is not None else 1.0,
            'num_channels': 1,
            'channel_weights': np.array([1.0], dtype=np.float32)
        }
    
    def log_multi_channel(self, episode: int, step: int,
                         sh_coeffs: np.ndarray, o2_positions: np.ndarray,
                         num_channels: int, channel_weights: Optional[np.ndarray],
                         temperature: Optional[float] = None,
                         translation_direction: Optional[np.ndarray] = None,
                         translation_magnitude: Optional[float] = None,
                         axis_direction: Optional[np.ndarray] = None) -> None:
        """Log multi-channel SH distribution.

        sh_coeffs: (num_channels, sh_dim), or (sh_dim,) for a single channel.
        o2_positions: (2, 3). channel_weights: (num_channels,) or None (uniform).
        """
        if episode not in self.data:
            self.data[episode] = {}

        if sh_coeffs.ndim == 1:
            sh_coeffs = sh_coeffs.reshape(1, -1)

        if channel_weights is None:
            channel_weights = np.ones(num_channels, dtype=np.float32) / num_channels

        o2_center = np.mean(o2_positions, axis=0)  # (3,)
        o2_distance = np.linalg.norm(o2_positions[1] - o2_positions[0])

        self.data[episode][step] = {
            'sh_coeffs': sh_coeffs.copy(),
            'o2_positions': o2_positions.copy(),
            'o2_center': o2_center,
            'o2_distance': float(o2_distance),
            'temperature': float(temperature) if temperature is not None else 1.0,
            'num_channels': int(num_channels),
            'channel_weights': channel_weights.copy()
        }
        
        if translation_direction is not None:
            self.data[episode][step]['translation_direction'] = self.validator.clean_direction(translation_direction)
        if translation_magnitude is not None:
            self.data[episode][step]['translation_magnitude'] = float(translation_magnitude)
        if axis_direction is not None:
            self.data[episode][step]['axis_direction'] = self.validator.clean_direction(axis_direction)
    
    def clear(self):
        """Clear all logged data."""
        self.data.clear()
    
    def get_episode_data(self, episode: int) -> Dict[int, Dict[str, Any]]:
        """Get all data for a specific episode."""
        return self.data.get(episode, {})
    
    def get_latest_episodes(self, n: int = 5) -> list:
        """Get the n most recent episode numbers."""
        return sorted(self.data.keys())[-n:]
