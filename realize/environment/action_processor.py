import numpy as np
from typing import Dict, Any, List
import logging
_LOG = logging.getLogger(__name__)


class ActionProcessor:
    """Processes and validates O2 move actions; detects oscillation."""

    def __init__(self, structure_manager, logger, max_trans_distance=3.0):
        self.structure_manager = structure_manager
        self.logger = logger
        self.max_trans_distance = max_trans_distance

        # History buffers
        self.position_history = []
        self.center_history = []    # O2 center positions, for oscillation detection
        self.movement_vectors = []

        # Oscillation detection: reject moves whose center shifts < threshold (Angstrom)
        self.oscillation_threshold = 0.3
        self.max_history = 5

        if self.oscillation_threshold <= 0:
            raise ValueError(
                f"oscillation_threshold must be > 0, got {self.oscillation_threshold}"
            )
        if self.max_history <= 0:
            raise ValueError(
                f"max_history must be > 0, got {self.max_history}"
            )

    def is_oscillating(self, o1_pos, o2_pos, threshold=None):
        """True if the O2 center moved less than threshold from a recent center."""
        if threshold is None:
            threshold = self.oscillation_threshold

        current_center = (np.array(o1_pos) + np.array(o2_pos)) / 2.0

        # Need at least 2 prior centers to judge oscillation
        if len(self.center_history) < 2:
            self.center_history.append(current_center)
            if len(self.center_history) > self.max_history:
                self.center_history.pop(0)
            return False

        for i, past_center in enumerate(self.center_history[-2:]):
            center_dist = np.linalg.norm(current_center - past_center)

            if center_dist < threshold:
                if self.logger:
                    self.logger.log_debug(
                        f"Oscillation detected: center moved only {center_dist:.4f} Angstrom "
                        f"(< {threshold} Angstrom threshold)"
                    )
                return True

        self.center_history.append(current_center)
        if len(self.center_history) > self.max_history:
            self.center_history.pop(0)

        return False

    def reset_history(self):
        """Clear all history buffers; call at episode start."""
        self.position_history = []
        self.center_history = []
        self.movement_vectors = []
        if self.logger:
            self.logger.log_debug("ActionProcessor history reset")

    def process_action(self, trans_action: np.ndarray, o2_config: np.ndarray,
                       state_revision: int) -> Dict[str, Any]:
        """Validate a (translation, axis) action and return new O2 positions."""

        # Current O2 positions before applying the action
        o1_current, o2_current = self.structure_manager.get_o2_positions()
        current_distance = np.linalg.norm(o1_current - o2_current)
        current_center = (o1_current + o2_current) / 2.0

        # Expected action: numpy array shape (2,3) -> [translation, axis]
        if isinstance(o2_config, np.ndarray) and o2_config.shape == (2, 3):
            trans_vector = o2_config[0].copy()  # copy so input is not mutated
            axis_direction = o2_config[1].copy()

            axis_norm = np.linalg.norm(axis_direction)

            # Translation magnitude must be within limit (no auto-scaling)
            trans_magnitude = np.linalg.norm(trans_vector)
            if trans_magnitude > self.max_trans_distance:
                if self.logger:
                    self.logger.log_error(
                        f"Translation magnitude {trans_magnitude:.3f} Angstrom exceeds limit "
                        f"{self.max_trans_distance:.3f} Angstrom"
                    )
                raise ValueError(
                    f"Translation magnitude {trans_magnitude:.3f} Angstrom exceeds max_trans_distance "
                    f"{self.max_trans_distance:.3f} Angstrom. Policy must generate valid magnitudes. "
                    f"Check magnitude predictor output and bounds."
                )

            # Axis must be a finite, non-zero unit vector (no auto-normalization)
            axis_norm = np.linalg.norm(axis_direction)

            if not np.isfinite(axis_norm):
                if self.logger:
                    self.logger.log_error(
                        f"Axis norm is non-finite: {axis_norm}"
                    )
                raise ValueError(
                    f"Axis direction norm is non-finite: {axis_norm}. "
                    f"Check axis predictor for NaN/Inf output."
                )
            
            if axis_norm < 1e-6:
                if self.logger:
                    self.logger.log_error(
                        f"Axis direction too small (norm={axis_norm:.6e})"
                    )
                raise ValueError(
                    f"Axis direction norm too small: {axis_norm:.6e} < 1e-6. "
                    f"Policy must generate non-zero axis. "
                    f"Check axis predictor output."
                )
            
            if abs(axis_norm - 1.0) > 1e-5:
                if self.logger:
                    self.logger.log_error(
                        f"Axis direction not unit vector (norm={axis_norm:.6f})"
                    )
                raise ValueError(
                    f"Axis direction must be unit vector. Got norm={axis_norm:.6f}. "
                    f"Policy must normalize axis before returning. "
                    f"Check axis predictor normalization."
                )

            if not np.all(np.isfinite(trans_vector)):
                if self.logger:
                    self.logger.log_error(f"Invalid translation vector (contains NaN/Inf): {trans_vector}")
                return {'valid': False, 'reason': 'invalid_translation'}

            if not np.all(np.isfinite(axis_direction)):
                if self.logger:
                    self.logger.log_error(f"Invalid axis direction (contains NaN/Inf): {axis_direction}")
                return {'valid': False, 'reason': 'invalid_axis'}

            # New O2 positions: translate center, place atoms +/- half bond along axis
            new_center = current_center + trans_vector
            bond_length = 1.19725  # O2 bond length in Angstroms
            half_bond = bond_length / 2.0

            o1_new = new_center + half_bond * axis_direction
            o2_new = new_center - half_bond * axis_direction

            movement = trans_vector

            self.movement_vectors.append(movement.copy())
            if len(self.movement_vectors) > self.max_history:
                self.movement_vectors.pop(0)

            movement_norm = np.linalg.norm(movement)
            if movement_norm < 1e-8:
                if self.logger:
                    self.logger.log_error(
                        f"Zero translation not allowed (norm={movement_norm:.3e})"
                    )
                raise ValueError(
                    f"Translation magnitude too small: {movement_norm:.3e} < 1e-8. "
                    f"Policy must generate non-zero translation. "
                    f"This may indicate magnitude predictor collapsed to zero."
                )            
            
        else:
            o2_shape = getattr(o2_config, 'shape', None)
            o2_type = type(o2_config)

            if self.logger:
                self.logger.log_error(
                    f"Unsupported action format: type={o2_type}, shape={o2_shape}"
                )
            
            raise TypeError(
                f"Unsupported action format. Expected numpy array with shape (2, 3), "
                f"got type={o2_type}, shape={o2_shape}. "
                f"Check that policy returns [translation_vector(3), axis_direction(3)]."
            )

        _LOG.debug(f"O1 new: {o1_new}")
        _LOG.debug(f"O2 new: {o2_new}")
        
        if self.is_oscillating(o1_new, o2_new):
            _LOG.debug(f"Action REJECTED due to oscillation")
            _LOG.debug(f"Current O1: {o1_new}")
            _LOG.debug(f"Current O2: {o2_new}")
            _LOG.debug(f"Position history length: {len(self.position_history)}")
            if len(self.position_history) > 0:
                _LOG.debug(f"  Last position: {self.position_history[-1]}")
            if len(self.position_history) > 1:
                _LOG.debug(f"  2nd last position: {self.position_history[-2]}")
            
            if self.logger:
                self.logger.log_debug("Oscillation detected - rejecting action")
                        
            return {
                'valid': False,
                'reason': 'oscillation',
                'distance': current_distance,
                'o1_pos': o1_current,
                'o2_pos': o2_current
            }
        
        self.position_history.append((o1_new.tolist(), o2_new.tolist()))
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        new_distance = np.linalg.norm(o1_new - o2_new)
        
        return {
            'valid': True,
            'o1_pos': o1_new,
            'o2_pos': o2_new,
            'distance': new_distance
        }

