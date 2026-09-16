from typing import Dict, Any, Optional
from realize.utils.visualization.data_logger import VisualizationDataLogger
from realize.utils.visualization.shape_validator import ShapeValidator

_global_logger = VisualizationDataLogger()


def log_multi_channel_sh_distribution(sh_data: Dict, episode: int, step: int,
                                     sh_coeffs: Any, o2_positions: Any,
                                     temperature: Optional[float] = None,
                                     num_channels: Optional[int] = None,
                                     channel_weights: Optional[Any] = None,
                                     **kwargs) -> None:
    """Clean inputs and log a multi-channel SH distribution into both the global
    logger and the shared `sh_data` dict.

    Stored shapes: sh_coeffs (num_channels, sh_dim), o2_positions (2, 3),
    channel_weights (num_channels,).
    """
    validator = ShapeValidator()
    clean_coeffs = validator.clean_sh_coeffs(sh_coeffs)
    clean_positions = validator.clean_o2_positions(o2_positions)

    if num_channels is None:
        num_channels = clean_coeffs.shape[0] if clean_coeffs.ndim == 2 else 1

    clean_weights = validator.clean_channel_weights(channel_weights, num_channels) if channel_weights is not None else None

    _global_logger.log_multi_channel(
        episode, step, clean_coeffs, clean_positions,
        num_channels, clean_weights, temperature,
        kwargs.get('translation_direction'),
        kwargs.get('translation_magnitude'),
        kwargs.get('axis_direction')
    )

    if episode not in sh_data:
        sh_data[episode] = {}
    sh_data[episode][step] = _global_logger.data[episode][step]
