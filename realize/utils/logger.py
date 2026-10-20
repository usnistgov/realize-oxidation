"""Logging for environment, step, and episode events."""
import os
import json
import numpy as np
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows doesn't have fcntl
import platform
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
_LOG = logging.getLogger(__name__)

# Unit symbols.
ANGSTROM = "\u212B"
DEGREE = "\u00B0"

# Decimal places used when rounding for deduplication.
DISTANCE_PRECISION = 4
ENERGY_PRECISION = 6

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and torch tensors."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, 'detach'):  # torch tensors
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

def to_jsonable(obj: Any) -> Any:
    """Convert object to a JSON-serializable form, or str() if it cannot be."""
    try:
        return json.loads(json.dumps(obj, cls=NumpyEncoder))
    except (TypeError, ValueError) as e:
        return str(obj)

class EnvironmentLogger:
    """Handles all environment-related logging with deduplication."""
    
    def __init__(self, log_file: str, verbosity: str = 'INFO'):
        self.log_file = log_file
        self.verbosity = verbosity
        self._last_logged = {}  # dedup cache for log_dissociation_check
        self._o2_status_cache = {}  # dedup cache for log_o2_status

    def log_o2_status(self, stage: str, o1_pos: np.ndarray, o2_pos: np.ndarray,
                      distance: float, cell: Any = None) -> None:
        """Log O2 status once per (stage, distance)."""
        cache_key = f"{stage}_{distance:.4f}"

        if cache_key in self._o2_status_cache:
            return
            
        self._o2_status_cache[cache_key] = True
        
        if self.verbosity == 'DEBUG':
            _LOG.info(f"\n--- O2 Status: {stage} ---")
            _LOG.info(f"O1: [{o1_pos[0]:.3f}, {o1_pos[1]:.3f}, {o1_pos[2]:.3f}]")
            _LOG.info(f"O2: [{o2_pos[0]:.3f}, {o2_pos[1]:.3f}, {o2_pos[2]:.3f}]")
            _LOG.info(f"O-O distance: {distance:.3f} {ANGSTROM}")
            if cell is not None:
                _LOG.info(f"Cell: {cell}")
            _LOG.info("--- End O2 Status ---\n")
    
    def log_dissociation_check(self, distance: float, threshold: float, 
                               si_count: int, is_dissociated: bool) -> None:
        """Log dissociation check once per unique state."""
        cache_key = f"dissoc_{distance:.3f}_{si_count}_{is_dissociated}"

        if cache_key in self._last_logged:
            return
            
        self._last_logged[cache_key] = True
        
        status = "DISSOCIATED" if is_dissociated else "NOT dissociated"
        _LOG.debug(f"O2 {status}: distance={distance:.3f}Angstrom (threshold={threshold:.2f}Angstrom), "
              f"Si atoms within 2.0Angstrom: {si_count}")
    
    def clear_cache(self):
        """Clear caches at the start of a new episode."""
        self._last_logged.clear()
        self._o2_status_cache.clear()


class StepLogger:
    """Handles step-level logging with consolidation and policy-aware action logging."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.current_step_data = {}
        self._logged_neb = False
        self._logged_reward = False
        self._policy_type = None  # drives action-logging format
        
    def begin_step(self, episode: int, step: int, try_num: int):
        """Initialize logging for a new step."""
        self.current_step_data = {
            'episode': int(episode),
            'step': int(step),
            'try_num': int(try_num),
            'actions': {},
            'results': {},
            'rewards': {}
        }
        self._logged_neb = False
        self._logged_reward = False
        _LOG.debug(f"\n--- Episode {episode}, Step {step} (try={try_num}) ---")


    def log_action(self, trans_action: Union[np.ndarray, List],
                   rot_action: Union[np.ndarray, List],
                   policy_type: Optional[str] = None):
        """Log the action with policy-aware formatting."""
        if policy_type:
            self._policy_type = policy_type
            if policy_type == "spherical_harmonics":
                self._log_spherical_action(trans_action, rot_action)
            elif policy_type == "mog":
                self._log_mog_action(trans_action, rot_action)
            else:
                self._log_generic_action(trans_action, rot_action)
        else:
            # No explicit type: infer from rot_action shape ((2,3)=SH, (3,3)=MoG).
            if isinstance(rot_action, np.ndarray) and rot_action.shape == (2, 3):
                self._policy_type = "spherical_harmonics"
                self._log_spherical_action(trans_action, rot_action)
            elif isinstance(rot_action, np.ndarray) and rot_action.shape == (3, 3):
                self._policy_type = "mog"
                self._log_mog_action(trans_action, rot_action)
            else:
                self._policy_type = "unknown"
                self._log_generic_action(trans_action, rot_action)
    
    def _log_spherical_action(self, trans_action: Union[np.ndarray, List], 
                             o2_positions: np.ndarray):
        """Log spherical harmonics policy action."""
        o1_pos = o2_positions[0]
        o2_pos = o2_positions[1]

        o_o_distance = np.linalg.norm(o2_pos - o1_pos)
        com = (o1_pos + o2_pos) / 2.0

        if isinstance(trans_action, np.ndarray) and len(trans_action) == 3:
            trans_norm = np.linalg.norm(trans_action)
        else:
            trans_norm = 0.0
        
        self.current_step_data['actions'] = {
            'policy_type': 'spherical_harmonics',
            'o1_position': o1_pos.tolist(),
            'o2_position': o2_pos.tolist(),
            'o_o_distance': float(o_o_distance),
            'center_of_mass': com.tolist(),
            'auxiliary_translation': trans_action.tolist() if isinstance(trans_action, np.ndarray) else trans_action,
            'auxiliary_trans_norm': float(trans_norm)
        }
        
        _LOG.debug(f"SH Action: O1=[{o1_pos[0]:.3f}, {o1_pos[1]:.3f}, {o1_pos[2]:.3f}]")
        _LOG.debug(f"          O2=[{o2_pos[0]:.3f}, {o2_pos[1]:.3f}, {o2_pos[2]:.3f}]")
        _LOG.debug(f"          O-O distance={o_o_distance:.3f}Angstrom, COM=[{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
    
    def _log_mog_action(self, translation: np.ndarray, rotation: np.ndarray):
        """Log traditional MoG policy action."""
        trans_norm = np.linalg.norm(translation)

        # Rotation angle from the matrix trace.
        trace = np.trace(rotation)
        cos_angle = np.clip((trace - 1) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        self.current_step_data['actions'] = {
            'policy_type': 'mog',
            'translation': translation.tolist(),
            'translation_norm': float(trans_norm),
            'rotation_matrix': rotation.tolist(),
            'rotation_angle_rad': float(angle),
            'rotation_angle_deg': float(np.degrees(angle))
        }
        
        _LOG.debug(f"MoG Action: trans_norm={trans_norm:.3f}Angstrom, rot_angle={np.degrees(angle):.1f}deg")
    
    def _log_generic_action(self, trans_action: Union[np.ndarray, List], 
                           rot_action: Union[np.ndarray, List]):
        """Log action with unknown/generic format."""
        if isinstance(trans_action, np.ndarray):
            trans_norm = np.linalg.norm(trans_action)
            trans_list = trans_action.tolist()
        else:
            trans_norm = 0.0
            trans_list = trans_action
        
        self.current_step_data['actions'] = {
            'policy_type': 'generic',
            'translation': trans_list,
            'translation_norm': float(trans_norm),
            'rotation': rot_action.tolist() if isinstance(rot_action, np.ndarray) else str(rot_action)
        }
        
        _LOG.debug(f"Generic Action: trans_norm={trans_norm:.3f}Angstrom")
    
    def log_spherical_harmonics_data(self, sh_coeffs: np.ndarray, lmax: int, 
                                    temperature: float, num_channels: int = 1,
                                    channel_weights: Optional[np.ndarray] = None):
        """Log spherical harmonics specific data with multi-channel support."""
        if self._policy_type == "spherical_harmonics":
            if 'spherical_harmonics' not in self.current_step_data:
                self.current_step_data['spherical_harmonics'] = {}
            
            if isinstance(sh_coeffs, np.ndarray) and sh_coeffs.ndim > 1 and num_channels > 1:
                # Multi-channel: reshape to [num_channels, coeffs_per_channel].
                coeffs_per_channel = sh_coeffs.shape[-1] // num_channels
                coeffs_reshaped = sh_coeffs.reshape(-1, num_channels, coeffs_per_channel)

                channel_magnitudes = [np.linalg.norm(coeffs_reshaped[0, ch, :]) for ch in range(num_channels)]
                dominant_channel = int(np.argmax(channel_magnitudes))
                
                self.current_step_data['spherical_harmonics'].update({
                    'coefficients': sh_coeffs.tolist(),
                    'lmax': int(lmax),
                    'temperature': float(temperature),
                    'num_channels': int(num_channels),
                    'coeffs_per_channel': int(coeffs_per_channel),
                    'channel_magnitudes': [float(mag) for mag in channel_magnitudes],
                    'dominant_channel': dominant_channel,
                    'channel_weights': channel_weights.tolist() if channel_weights is not None else None,
                    'total_coeffs': len(sh_coeffs) if hasattr(sh_coeffs, '__len__') else 0
                })
                
                mag_str = ', '.join([f'{mag:.3f}' for mag in channel_magnitudes])
                _LOG.debug(f"SH Data: lmax={lmax}, temp={temperature:.3f}, channels={num_channels}")
                _LOG.debug(f"         Channel magnitudes=[{mag_str}], dominant=ch{dominant_channel}")
            else:
                # Single channel.
                self.current_step_data['spherical_harmonics'].update({
                    'coefficients': sh_coeffs.tolist() if isinstance(sh_coeffs, np.ndarray) else sh_coeffs,
                    'lmax': int(lmax),
                    'temperature': float(temperature),
                    'num_channels': 1,
                    'coeffs_per_channel': len(sh_coeffs) if hasattr(sh_coeffs, '__len__') else 0,
                    'total_coeffs': len(sh_coeffs) if hasattr(sh_coeffs, '__len__') else 0
                })
                
                _LOG.debug(f"SH Data: lmax={lmax}, temp={temperature:.3f}, coeffs={len(sh_coeffs)}")

    def log_mog_data(self, means: np.ndarray, stds: np.ndarray, weights: np.ndarray,
                     chosen_component: int):
        """Log mixture of Gaussians specific data."""
        if self._policy_type == "mog":
            if 'mixture_of_gaussians' not in self.current_step_data:
                self.current_step_data['mixture_of_gaussians'] = {}
            
            self.current_step_data['mixture_of_gaussians'].update({
                'means': means.tolist() if isinstance(means, np.ndarray) else means,
                'stds': stds.tolist() if isinstance(stds, np.ndarray) else stds,
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'chosen_component': int(chosen_component),
                'num_components': len(weights) if hasattr(weights, '__len__') else 0
            })
            
            weights_str = ', '.join([f'{w:.3f}' for w in weights])
            _LOG.debug(f"MoG Data: components={len(weights)}, chosen={chosen_component}, weights=[{weights_str}]")
    
    def log_neb_result(self, activation_energy: float, valid: bool, energy_profile=None):
        """Record NEB result into the current step's data."""
        if not hasattr(self, 'current_step_data'):
            self.current_step_data = {}

        if 'results' not in self.current_step_data:
            self.current_step_data['results'] = {}

        self.current_step_data['results']['neb'] = {
            'activation_energy': activation_energy,
            'valid': valid,
            'energy_profile': energy_profile.tolist() if energy_profile is not None else None
        }

    def log_reward(self, total: float, components: Dict[str, float]):
        """Log reward once per step."""
        if self._logged_reward:
            return

        self._logged_reward = True
        self.current_step_data['rewards'] = {
            'total': float(total),
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in components.items()}
        }

        comp_str = ", ".join([f"{k}={float(v):.3f}" for k, v in components.items() if isinstance(v, (int, float, np.number))])
        _LOG.debug(f"Step reward: {float(total):.4f} ({comp_str})")
    
    def log_step_result(self, is_valid: bool, is_done: bool, 
                        dissociated: bool, distance: float):
        """Log final step results."""
        self.current_step_data['results'].update({
            'valid': bool(is_valid),
            'done': bool(is_done),
            'dissociated': bool(dissociated),
            'o2_distance': float(distance) if not np.isinf(distance) else -1.0
        })
        
        if not is_valid:
            _LOG.debug("Step: INVALID (rolled back)")
        elif dissociated:
            _LOG.debug(f"Step: DISSOCIATED! (distance={distance:.3f}Angstrom)")
        elif is_done:
            _LOG.debug(f"Step: TERMINATED (distance={distance:.3f}Angstrom)")
        else:
            _LOG.debug(f"Step: VALID (O-O distance={distance:.3f}Angstrom)")
    
    def finalize_step(self):
        """No-op; UnifiedLogger drives step finalization."""
        pass

class EpisodeLogger:
    """Handles episode-level logging and summaries with enhanced policy awareness."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.episode_data = {}
        self._barrier_count = 0
        self._policy_stats = {}  # per-policy-type running stats
        
    def begin_episode(self, episode: int):
        """Initialize logging for a new episode."""
        self.episode_data = {
            'episode': int(episode),
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'pathway_barriers': [],
            'rewards': {},
            'stats': {},
            'policy_analytics': {}
        }
        self._barrier_count = 0
        self._policy_stats = {}
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"EPISODE {episode} STARTED")
        _LOG.info(f"{'='*60}")
    
    def add_step(self, step_data: Dict):
        """Append (JSON-cleaned) step data and update policy analytics."""
        clean_step_data = json.loads(json.dumps(step_data, cls=NumpyEncoder))
        self.episode_data['steps'].append(clean_step_data)

        self._update_policy_analytics(clean_step_data)
    
    def _update_policy_analytics(self, step_data: Dict):
        """Update policy-specific analytics from step data."""
        actions = step_data.get('actions', {})
        policy_type = actions.get('policy_type', 'unknown')
        
        if policy_type not in self._policy_stats:
            self._policy_stats[policy_type] = {
                'step_count': 0,
                'total_translation_norm': 0.0,
                'max_translation_norm': 0.0,
                'min_translation_norm': float('inf')
            }
        
        stats = self._policy_stats[policy_type]
        stats['step_count'] += 1

        if policy_type == 'spherical_harmonics':
            # SH policy uses O-O distance as the movement metric.
            o_o_distance = actions.get('o_o_distance', 0.0)
            stats['total_translation_norm'] += o_o_distance
            stats['max_translation_norm'] = max(stats['max_translation_norm'], o_o_distance)
            stats['min_translation_norm'] = min(stats['min_translation_norm'], o_o_distance)

            if 'o2_geometry' not in stats:
                stats['o2_geometry'] = {
                    'avg_bond_length': 0.0,
                    'min_bond_length': float('inf'),
                    'max_bond_length': 0.0,
                    'bond_length_samples': 0
                }
            
            bond_stats = stats['o2_geometry']
            bond_stats['avg_bond_length'] = ((bond_stats['avg_bond_length'] * bond_stats['bond_length_samples'] + o_o_distance) / 
                                           (bond_stats['bond_length_samples'] + 1))
            bond_stats['min_bond_length'] = min(bond_stats['min_bond_length'], o_o_distance)
            bond_stats['max_bond_length'] = max(bond_stats['max_bond_length'], o_o_distance)
            bond_stats['bond_length_samples'] += 1

            sh_data = step_data.get('spherical_harmonics', {})
            if sh_data and sh_data.get('num_channels', 1) > 1:
                if 'multi_channel_stats' not in stats:
                    stats['multi_channel_stats'] = {
                        'channel_usage': {},
                        'avg_temperature': 0.0,
                        'temperature_samples': 0,
                        'channel_diversity': []
                    }
                
                mc_stats = stats['multi_channel_stats']

                dominant_ch = sh_data.get('dominant_channel', 0)
                if dominant_ch not in mc_stats['channel_usage']:
                    mc_stats['channel_usage'][dominant_ch] = 0
                mc_stats['channel_usage'][dominant_ch] += 1

                temp = sh_data.get('temperature', 0.0)
                mc_stats['avg_temperature'] = ((mc_stats['avg_temperature'] * mc_stats['temperature_samples'] + temp) /
                                             (mc_stats['temperature_samples'] + 1))
                mc_stats['temperature_samples'] += 1

                # Channel diversity = spread of coefficient magnitudes.
                channel_mags = sh_data.get('channel_magnitudes', [])
                if len(channel_mags) > 1:
                    diversity = max(channel_mags) - min(channel_mags)
                    mc_stats['channel_diversity'].append(diversity)
        elif policy_type == 'mog':
            # MoG policy uses translation norm as the movement metric.
            trans_norm = actions.get('translation_norm', 0.0)
            stats['total_translation_norm'] += trans_norm
            stats['max_translation_norm'] = max(stats['max_translation_norm'], trans_norm)
            stats['min_translation_norm'] = min(stats['min_translation_norm'], trans_norm)

            rotation_angle = actions.get('rotation_angle_deg', 0.0)
            if 'rotation_stats' not in stats:
                stats['rotation_stats'] = {
                    'avg_rotation_angle': 0.0,
                    'max_rotation_angle': 0.0,
                    'rotation_samples': 0
                }
            
            rot_stats = stats['rotation_stats']
            rot_stats['avg_rotation_angle'] = ((rot_stats['avg_rotation_angle'] * rot_stats['rotation_samples'] + rotation_angle) / 
                                             (rot_stats['rotation_samples'] + 1))
            rot_stats['max_rotation_angle'] = max(rot_stats['max_rotation_angle'], rotation_angle)
            rot_stats['rotation_samples'] += 1
    
    def add_barrier(self, barrier: float):
        """Append an activation barrier, skipping a repeat of the last value."""
        barrier_float = float(barrier)

        if (self.episode_data['pathway_barriers'] and
            abs(self.episode_data['pathway_barriers'][-1] - barrier_float) < 1e-6):
            return

        self.episode_data['pathway_barriers'].append(barrier_float)
        self._barrier_count += 1
        _LOG.debug(f"  Pathway barrier #{self._barrier_count}: {barrier_float:.4f} eV")
    
    def finalize_episode(self, dissociated: bool, total_reward: float, 
                        reward_components: Dict[str, float], 
                        stats: Optional[Dict] = None):
        """Finalize and log the episode summary with policy analytics."""
        for policy_type, policy_stats in self._policy_stats.items():
            if policy_stats['step_count'] > 0:
                policy_stats['avg_translation_norm'] = policy_stats['total_translation_norm'] / policy_stats['step_count']
                if policy_stats['min_translation_norm'] == float('inf'):
                    policy_stats['min_translation_norm'] = 0.0

        self.episode_data.update({
            'end_time': datetime.now().isoformat(),
            'dissociated': bool(dissociated),
            'total_reward': float(total_reward) if not np.isnan(total_reward) else 0.0,
            'rewards': {k: float(v) if isinstance(v, (np.floating, np.integer, int, float)) else 0.0 
                       for k, v in reward_components.items()},
            'stats': {},
            'policy_analytics': self._policy_stats
        })

        self._update_learning_curve(
            episode=self.episode_data['episode'],
            total_reward=total_reward,
            reward_components=reward_components,
            dissociated=dissociated
        )

        # Coerce stats values to JSON-friendly types.
        if stats:
            for k, v in stats.items():
                if isinstance(v, (np.floating, np.integer, int, float)):
                    self.episode_data['stats'][k] = float(v)
                elif isinstance(v, (list, tuple)):
                    self.episode_data['stats'][k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    self.episode_data['stats'][k] = v
        
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"EPISODE {self.episode_data['episode']} SUMMARY")
        _LOG.info(f"{'='*60}")
        # Prefer the step count reported in stats; fall back to logged steps.
        actual_steps = stats.get('steps', len(self.episode_data['steps'])) if stats else len(self.episode_data['steps'])
        valid_steps = stats.get('valid_steps', 0) if stats else 0

        actual_steps = stats.get('steps', len(self.episode_data['steps'])) if stats else len(self.episode_data['steps'])
        valid_steps = stats.get('valid_steps', 0) if stats else 0

        _LOG.info(f"Steps: {actual_steps} (valid: {valid_steps})")
        _LOG.info(f"Dissociated: {'YES' if dissociated else 'NO'}")

        if stats and 'magnitude_stats' in stats and stats['magnitude_stats']:
            mag_stats = stats['magnitude_stats']
            _LOG.info(f"Translation magnitudes: mean={mag_stats['mean']:.3f}, "
                  f"std={mag_stats['std']:.3f}, max={mag_stats['max']:.3f}")

        for policy_type, policy_stats in self._policy_stats.items():
            if policy_stats['step_count'] > 0:
                _LOG.info(f"Policy ({policy_type}): {policy_stats['step_count']} steps")
                _LOG.info(f"  Avg movement: {policy_stats['avg_translation_norm']:.3f}")
                _LOG.info(f"  Max movement: {policy_stats['max_translation_norm']:.3f}")

                if policy_type == 'spherical_harmonics' and 'o2_geometry' in policy_stats:
                    geom = policy_stats['o2_geometry']
                    _LOG.info(f"  O-O bond: avg={geom['avg_bond_length']:.3f}Angstrom, "
                          f"range=[{geom['min_bond_length']:.3f}, {geom['max_bond_length']:.3f}]Angstrom")

                    if 'multi_channel_stats' in policy_stats:
                        mc_stats = policy_stats['multi_channel_stats']
                        _LOG.info(f"  Multi-channel: avg_temp={mc_stats['avg_temperature']:.3f}")

                        if mc_stats['channel_usage']:
                            total_usage = sum(mc_stats['channel_usage'].values())
                            usage_pct = {ch: (count/total_usage)*100 for ch, count in mc_stats['channel_usage'].items()}
                            usage_str = ', '.join([f"ch{ch}:{pct:.1f}%" for ch, pct in usage_pct.items()])
                            _LOG.info(f"  Channel usage: {usage_str}")

                        if mc_stats['channel_diversity']:
                            avg_diversity = sum(mc_stats['channel_diversity']) / len(mc_stats['channel_diversity'])
                            max_diversity = max(mc_stats['channel_diversity'])
                            _LOG.info(f"  Channel diversity: avg={avg_diversity:.3f}, max={max_diversity:.3f}")
                
                elif policy_type == 'mog' and 'rotation_stats' in policy_stats:
                    rot = policy_stats['rotation_stats']
                    _LOG.info(f"  Rotation: avg={rot['avg_rotation_angle']:.1f}deg, max={rot['max_rotation_angle']:.1f}deg")

        if self.episode_data['pathway_barriers']:
            barriers = self.episode_data['pathway_barriers']
            _LOG.info(f"Pathway barriers: {[f'{b:.4f}' for b in barriers]}")
            _LOG.info(f"Max barrier: {max(barriers):.4f} eV")
        
        # total_reward holds only the episode-end rewards (EA + dissoc);
        # per-step PE rewards are accumulated separately during env.step.
        pe_reward = reward_components.get('pe_step_rewards', 0.0)
        ea_reward = reward_components.get('ea', 0.0)
        dissoc_reward = reward_components.get('dissociation', 0.0)

        _LOG.info(f"Episode-end rewards (EA+dissoc): {ea_reward + dissoc_reward:.4f}")
        _LOG.info(f"Per-step PE rewards total: {pe_reward:.4f}")
        _LOG.info(f"Total rewards seen by optimizer: {ea_reward + dissoc_reward + pe_reward:.4f}")
        _LOG.info(f"Reward breakdown:")

        components_to_print = {
            'ea': ea_reward,
            'dissociation': dissoc_reward,
            'pe_step_rewards': pe_reward
        }
        
        for component, value in components_to_print.items():
            if isinstance(value, (int, float, np.number)):
                label = "  - " + component
                if component == 'pe_step_rewards':
                    label += " (accumulated per-step)"
                _LOG.info(f"{label}: {float(value):.4f}")

        if stats:
            _LOG.info(f"Additional stats:")
            for key, value in stats.items():
                if isinstance(value, (int, float, np.number)):
                    _LOG.info(f"  - {key}: {float(value):.4f}")
                else:
                    _LOG.info(f"  - {key}: {value}")
        
        _LOG.info(f"{'='*60}\n")

        # Append the episode record (logs/ created lazily on first write).
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(self.episode_data, cls=NumpyEncoder) + '\n')
        except Exception as e:
            _LOG.warning(f"Failed to write episode log: {e}")

    def _update_learning_curve(self, episode: int, total_reward: float,
                              reward_components: Dict[str, float],
                              dissociated: bool):
        """Append one row to logs/learning_curve.dat."""
        import os

        base_dir = os.path.dirname(os.path.dirname(self.log_file))
        log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        curve_file = os.path.join(log_dir, 'learning_curve.dat')

        # Running average over the last 10 episodes.
        if not hasattr(self, '_reward_history'):
            self._reward_history = []

        self._reward_history.append(total_reward)

        if len(self._reward_history) > 10:
            self._reward_history = self._reward_history[-10:]

        avg_reward = sum(self._reward_history) / len(self._reward_history)

        ea_reward = reward_components.get('ea', 0.0)
        dissoc_reward = reward_components.get('dissociation', 0.0)
        dissoc_flag = 1 if dissociated else 0

        if episode == 0 or not os.path.exists(curve_file):
            with open(curve_file, 'w') as f:
                f.write("# Episode  Reward     AvgReward(10)  EA_Reward  Dissoc_Reward  Dissociated\n")

        with open(curve_file, 'a') as f:
            f.write(f"{episode:8d} {total_reward:12.6f} {avg_reward:12.6f} "
                    f"{ea_reward:12.6f} {dissoc_reward:14.6f} {dissoc_flag:12d}\n")


class UnifiedLogger:
    """Main logger that coordinates all logging subsystems with policy awareness."""
    
    def __init__(self, base_dir: str, verbosity: str = 'INFO'):
        self.base_dir = base_dir
        self.verbosity = verbosity
        
        # logs/ is created lazily on the first actual write, so a worker that
        # hands its data back to the main process leaves no empty logs/ behind.
        self.log_dir = os.path.join(base_dir, 'logs')

        self.env_logger = EnvironmentLogger(
            os.path.join(self.log_dir, 'environment.log'),
            verbosity
        )
        self.step_logger = StepLogger(
            os.path.join(self.log_dir, 'steps.jsonl')
        )
        self.episode_logger = EpisodeLogger(
            os.path.join(self.log_dir, 'episodes.jsonl')
        )

        self._current_episode = None
        self._current_step = None
        
    def begin_episode(self, episode: int):
        """Start a new episode."""
        self._current_episode = episode
        self._current_step = None
        self.env_logger.clear_cache()
        self.episode_logger.begin_episode(episode)
    
    def begin_step(self, step: int, try_num: int):
        """Start a new step."""
        self._current_step = step
        self.step_logger.begin_step(self._current_episode, step, try_num)
    
    def log_action(self, translation: Union[np.ndarray, List], 
                   rotation: Union[np.ndarray, List]):
        """Log action taken with automatic policy detection."""
        self.step_logger.log_action(translation, rotation)
    
    def log_spherical_harmonics_data(self, sh_coeffs: np.ndarray, lmax: int, 
                                    temperature: float, num_channels: int = 1,
                                    channel_weights: Optional[np.ndarray] = None):
        """Log spherical harmonics specific data with multi-channel support."""
        self.step_logger.log_spherical_harmonics_data(sh_coeffs, lmax, temperature, 
                                                     num_channels, channel_weights)

    def log_mog_data(self, means: np.ndarray, stds: np.ndarray, weights: np.ndarray,
                     chosen_component: int):
        """Log mixture of Gaussians specific data."""
        self.step_logger.log_mog_data(means, stds, weights, chosen_component)
    
    def log_training_stability(self, kl_divergence: float, gradient_norm: float, 
                              policy_entropy: float, temperature: Optional[float] = None,
                              lr_adjustments: Optional[Dict[str, float]] = None):
        """Log training stability metrics for enhanced spherical harmonics training."""
        stability_data = {
            'kl_divergence': float(kl_divergence),
            'gradient_norm': float(gradient_norm),
            'policy_entropy': float(policy_entropy),
            'timestamp': datetime.now().isoformat()
        }
        
        if temperature is not None:
            stability_data['temperature'] = float(temperature)
        
        if lr_adjustments:
            stability_data['lr_adjustments'] = {k: float(v) for k, v in lr_adjustments.items()}

        if hasattr(self.step_logger, 'current_step_data') and self.step_logger.current_step_data:
            if 'training_stability' not in self.step_logger.current_step_data:
                self.step_logger.current_step_data['training_stability'] = []
            self.step_logger.current_step_data['training_stability'].append(stability_data)

        # Warn on out-of-band stability metrics.
        if kl_divergence > 0.1:
            self.log_warning(f"High KL divergence: {kl_divergence:.4f} (>0.1)")
        if gradient_norm > 5.0:
            self.log_warning(f"High gradient norm: {gradient_norm:.4f} (>5.0)")
        if policy_entropy < 0.1:
            self.log_warning(f"Low policy entropy: {policy_entropy:.4f} (<0.1)")

        # Emit a detailed line every 10th call.
        if hasattr(self, '_stability_log_counter'):
            self._stability_log_counter += 1
        else:
            self._stability_log_counter = 1
        
        if self._stability_log_counter % 10 == 0:
            self.log_info(f"Training stability: KL={kl_divergence:.4f}, "
                         f"grad_norm={gradient_norm:.4f}, entropy={policy_entropy:.4f}")
            if temperature is not None:
                _LOG.info(f"                   temperature={temperature:.4f}")

    def log_o2_status(self, stage: str, o1_pos: np.ndarray, o2_pos: np.ndarray, 
                      distance: float, cell: Any = None):
        """Log O2 molecule status."""
        self.env_logger.log_o2_status(stage, o1_pos, o2_pos, distance, cell)
    
    def log_dissociation_check(self, distance: float, threshold: float, 
                               si_count: int, is_dissociated: bool):
        """Log dissociation check."""
        self.env_logger.log_dissociation_check(distance, threshold, si_count, is_dissociated)
    
    def log_neb_result(self, activation_energy: float, valid: bool,
                       energy_profile: Optional[List[float]] = None):
        """Record a NEB result (step logger handles deduplication)."""
        self.step_logger.log_neb_result(activation_energy, valid, energy_profile)

    def log_neb_convergence(self, steps: int, converged: bool, max_force: float,
                           energy_barrier: float, max_atom_force: Optional[float] = None,
                           force_tolerance: float = 0.433634):
        """Log NEB convergence. Forces in eV/Angstrom, barrier in eV."""
        convergence_status = "CONVERGED" if converged else "NOT CONVERGED"

        message = (
            f"NEB {convergence_status} after {steps} steps | "
            f"Max force: {max_force:.3f} eV/Angstrom (tol: {force_tolerance:.3f} eV/Angstrom) | "
            f"Barrier: {energy_barrier:.4f} eV"
        )

        if max_atom_force is not None:
            message += f" | Max atom force: {max_atom_force:.3f} eV/Angstrom"

        if converged:
            self.log_info(message)
        else:
            self.log_warning(message)

        if hasattr(self.step_logger, 'current_step_data') and self.step_logger.current_step_data:
            if 'results' not in self.step_logger.current_step_data:
                self.step_logger.current_step_data['results'] = {}
                
            self.step_logger.current_step_data['results']['neb_convergence'] = {
                'steps': int(steps),
                'converged': bool(converged),
                'max_force': float(max_force),
                'max_atom_force': float(max_atom_force) if max_atom_force is not None else None,
                'energy_barrier': float(energy_barrier),
                'force_tolerance': float(force_tolerance)
            }

    def log_update(self, episode, losses, metrics):
        """Log per-update losses and metrics (and mirror to tensorboard if set)."""
        if losses:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            self.log_info(f"[Episode {episode}] Losses - {loss_str}")

        if metrics:
            metric_items = []
            for k, v in metrics.items():
                if v is not None:
                    if isinstance(v, float):
                        metric_items.append(f"{k}: {v:.4f}")
                    elif isinstance(v, int):
                        metric_items.append(f"{k}: {v}")
                    else:
                        metric_items.append(f"{k}: {v}")

            if metric_items:
                metric_str = ", ".join(metric_items)
                self.log_info(f"[Episode {episode}] Metrics - {metric_str}")

        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer:
            for key, value in losses.items():
                if value is not None:
                    self.tensorboard_writer.add_scalar(f'Loss/{key}', value, episode)
            
            for key, value in metrics.items():
                if value is not None and isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f'Metrics/{key}', value, episode)

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on verbosity."""
        levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        current_level = levels.get(self.verbosity, 1)
        msg_level = levels.get(level, 1)
        return msg_level >= current_level

    def log_reward(self, total: float, **components):
        """Log reward calculation with deduplication."""
        self.step_logger.log_reward(total, components)
    
    def log_step_result(self, is_valid: bool, is_done: bool, 
                        dissociated: bool, distance: float):
        """Log step results."""
        self.step_logger.log_step_result(is_valid, is_done, dissociated, distance)
    
    def finalize_step(self):
        """Add the current step to the episode and write it to the step log."""
        if hasattr(self.step_logger, 'current_step_data') and self.step_logger.current_step_data:
            self.episode_logger.add_step(self.step_logger.current_step_data)
            self._write_jsonl_safe(self.step_logger.log_file, self.step_logger.current_step_data)

    def finalize_episode(self, dissociated: bool, total_reward: float, 
                        reward_components: Dict[str, float], 
                        stats: Optional[Dict] = None):
        """Finalize current episode."""
        self.episode_logger.finalize_episode(
            dissociated, total_reward, reward_components, stats
        )
        
    def log_debug(self, message: str):
        """Log debug message (shown only when verbose logging is enabled)."""
        if self._should_log('DEBUG'):
            _LOG.debug(message)

    def log_info(self, message: str):
        """Log info message."""
        if self._should_log('INFO'):
            _LOG.info(message)

    def log_warning(self, message: str):
        """Log warning message."""
        _LOG.warning(message)

    def log_error(self, message: str):
        """Log error message."""
        _LOG.error(message)

    def _write_jsonl_safe(self, filepath: str, data: dict):
        """Append a JSON line, with an fcntl lock where available."""
        json_line = json.dumps(data, cls=NumpyEncoder) + '\n'

        # Per-worker filename suffix isolates concurrent writers.
        if hasattr(self, 'worker_id'):
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}_{self.worker_id}{ext}"

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a') as f:
                if HAS_FCNTL and platform.system() != 'Windows':
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json_line)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    # No fcntl (e.g. Windows): rely on per-worker filenames.
                    f.write(json_line)
        except Exception as e:
            _LOG.warning(f"Failed to write to {filepath}: {e}")

    def log_equivariance_test(self, test_name: str, passed: bool, details: Optional[Dict] = None):
        """Log a single equivariance test result (DEBUG only)."""
        if not self._should_log('DEBUG'):
            return

        status = "PASSED" if passed else "FAILED"
        message = f"Equivariance test '{test_name}': {status}"

        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" ({detail_str})"

        self.log_debug(message)

    def log_equivariance_summary(self, results: Dict[str, Union[bool, Dict]], episode: int):
        """Log a pass/fail summary over all equivariance tests (DEBUG only)."""
        if not self._should_log('DEBUG'):
            return

        all_tests = []
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for test_name, passed in category_results.items():
                    all_tests.append((f"{category}/{test_name}", passed))
            else:
                all_tests.append((category, category_results))

        total = len(all_tests)
        passed = sum(1 for _, p in all_tests if p)

        if total == 0:
            return

        pass_rate = (passed / total) * 100

        self.log_debug(f"Episode {episode} Equivariance Tests: {passed}/{total} passed ({pass_rate:.1f}%)")

        if passed < total:
            for test_name, test_passed in all_tests:
                if not test_passed:
                    self.log_warning(f"  Equivariance test failed: {test_name}")

    def log_wigner_d_validation(self, l: int, shape: Tuple, is_unitary: bool,
                                det_value: float, max_error: float):
        """Log Wigner-D matrix validation (shape, unitarity, |det|) at DEBUG."""
        if not self._should_log('DEBUG'):
            return

        expected_dim = 2 * l + 1
        is_correct_shape = shape == (expected_dim, expected_dim) or shape[1:] == (expected_dim, expected_dim)
        is_valid_det = abs(abs(det_value) - 1.0) < 1e-5

        status = "OK" if (is_correct_shape and is_unitary and is_valid_det) else "FAIL"

        self.log_debug(
            f"{status} Wigner-D(l={l}): shape={shape}, unitary={is_unitary}, "
            f"det={det_value:.6f}, max_error={max_error:.6e}"
        )

def print_banner():
    """Print the REALIZE startup banner to stdout."""
    bar = "=" * 80
    lines = [
        bar,
        "",
        r"   ____  _____    _    _     ___ _____ _____",
        r"  |  _ \| ____|  / \  | |   |_ _|__  /| ____|",
        r"  | |_) |  _|   / _ \ | |    | |  / / |  _|",
        r"  |  _ <| |___ / ___ \| |___ | | / /_ | |___",
        r"  |_| \_\_____/_/   \_\_____|___/____||_____|",
        "",
        "   Reinforcement-learning framework for Atomistic Long-timescale",
        "   Simulation in Experimentally relevant environments",
        "",
        "   by Wonseok Jeong and Brian DeCost",
        "",
        bar,
    ]
    _LOG.info("\n".join(lines))


def create_logger(results_dir: str, verbosity: str = 'INFO') -> UnifiedLogger:
    """Create and return a unified logger instance."""
    return UnifiedLogger(results_dir, verbosity)


def integrate_with_lammps_env(env, logger: UnifiedLogger):
    """Attach the logger to env and add a get_o2_status() helper if missing."""
    env.logger = logger

    def get_o2_status():
        """Return (o1_pos, o2_pos, O-O distance, cell), or Nones on failure."""
        try:
            from ase import io
            atoms = io.read(env.coo_file, format="lammps-data", atom_style="charge")
            if len(atoms) < 2:
                return None, None, 0.0, None

            o1_idx = len(atoms) - 2
            o2_idx = len(atoms) - 1
            positions = atoms.get_positions()
            o1_pos = positions[o1_idx]
            o2_pos = positions[o2_idx]
            distance = atoms.get_distance(o1_idx, o2_idx, mic=True)

            return o1_pos, o2_pos, distance, atoms.get_cell()
        except Exception as e:
            logger.log_error(f"Error getting O2 status: {e}")
            return None, None, 0.0, None

    if not hasattr(env, 'get_o2_status'):
        env.get_o2_status = get_o2_status

def integrate_with_train_script(logger: UnifiedLogger):
    """Placeholder showing the logger call sequence for a training loop."""
    pass

def log_multi_channel_training_integration(logger: UnifiedLogger, agent, episode: int):
    """Log training-stability and multi-channel SH data after a policy update."""
    try:
        if hasattr(agent, 'policy') and hasattr(agent.policy, 'current_temperature'):
            temperature = agent.policy.current_temperature
        else:
            temperature = None

        kl_div = getattr(agent, '_last_kl_divergence', 0.0)
        grad_norm = getattr(agent, '_last_gradient_norm', 0.0)
        policy_entropy = getattr(agent, '_last_policy_entropy', 1.0)
        lr_adjustments = getattr(agent, '_lr_adjustments', None)

        logger.log_training_stability(
            kl_divergence=kl_div,
            gradient_norm=grad_norm,
            policy_entropy=policy_entropy,
            temperature=temperature,
            lr_adjustments=lr_adjustments
        )

        if hasattr(agent.policy, '_last_sh_coeffs') and hasattr(agent.policy, 'num_channels'):
            logger.log_spherical_harmonics_data(
                sh_coeffs=agent.policy._last_sh_coeffs,
                lmax=agent.policy.lmax,
                temperature=temperature or 1.0,
                num_channels=agent.policy.num_channels
            )

            # Mirror the distribution to the visualization system.
            try:
                # Local import avoids a circular dependency.
                from realize.utils.visualization import log_multi_channel_sh_distribution

                current_step = getattr(agent, '_current_step', 0)

                o2_positions = getattr(agent.policy, '_last_o2_positions', None)
                if o2_positions is None:
                    # Placeholder positions; do not affect the SH distribution plot.
                    o2_positions = np.array([[0, 0, 0], [0, 0, 1.2]], dtype=np.float32)

                channel_weights = None
                if hasattr(agent.policy, 'channel_weights'):
                    channel_weights = agent.policy.channel_weights
                elif hasattr(agent.policy, 'get_channel_weights'):
                    channel_weights = agent.policy.get_channel_weights()

                sh_data = {}
                log_multi_channel_sh_distribution(
                    sh_data=sh_data,
                    episode=episode,
                    step=current_step,
                    sh_coeffs=agent.policy._last_sh_coeffs,
                    o2_positions=o2_positions,
                    temperature=temperature,
                    num_channels=agent.policy.num_channels,
                    channel_weights=channel_weights
                )
            except ImportError:
                pass
            except Exception as viz_e:
                logger.log_debug(f"Could not log visualization data: {viz_e}")

    except Exception as e:
        logger.log_warning(f"Error in multi-channel training integration: {e}")
