"""Episode reward: Fermi-Dirac barrier term on the effective activation energy, plus dissociation reward."""
import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any
import logging
_LOG = logging.getLogger(__name__)


def calculate_pathway_reward(
    activation_energies: Union[List[float], np.ndarray, None],
    temperature: float = 1000.0,
    dissociated: bool = False,
    ea_weight: float = 3.0,
    dissociation_weight: float = 0.3,
    require_dissociation: bool = True,
    episode_ended: bool = False,
    timeout_penalty: float = 0.5,
    Ea_gamma: float = 0.99,
    E_threshold: float = 0.05,
    cap_unsuccessful_reward: bool = True,
    ea_sigmoid_k: float = 0.2,
    ea_sigmoid_midpoint: float = 1.2,
    debug: bool = False
) -> Union[Tuple[float, float, float], Tuple[Tuple[float, float, float], Dict[str, Any]]]:
    """Pathway reward from the effective activation barrier.

    Effective barrier via numerically stable log-sum-exp:
        Ea_eff = k_B * T * log(Sigma(exp(E_i/(k_B*T))))

    Args:
        activation_energies: List/array of activation energies in eV
        temperature: System temperature in Kelvin
        dissociated: Whether O2 has dissociated
        ea_weight: Weight for activation energy component
        dissociation_weight: Weight for dissociation binary reward
        require_dissociation: Whether dissociation is required for positive reward
        episode_ended: Whether episode has ended
        timeout_penalty: Penalty for ending without dissociation
        Ea_gamma: Discount factor for path length (0 < gamma < 1)
        E_threshold: Energy threshold below which barriers are negligible (eV)
        cap_unsuccessful_reward: Whether to cap reward at -1.0 for unsuccessful episodes
        ea_sigmoid_k: Sigmoid steepness parameter (eV)
        ea_sigmoid_midpoint: Sigmoid midpoint (eV)
        debug: If True, return diagnostics dictionary
    
    Returns:
        If debug=False: (total_reward, ea_reward, dissociation_reward) as Python floats
        If debug=True: ((total_reward, ea_reward, dissociation_reward), diagnostics_dict)
    
    Units:
        - All energies in eV
        - Temperature in Kelvin
        - k_B in eV/K
    """
    k_B = 8.617333262e-5  # Boltzmann constant, eV/K

    diagnostics = {
        'Ea_eff': None,
        'num_barriers': 0,
        'max_barrier': None,
        'path_length': 0,
        'filtered_barriers': []
    }

    clean_barriers = _validate_energies(activation_energies)
    diagnostics['path_length'] = len(clean_barriers) if clean_barriers is not None else 0

    ea_reward = 0.0
    dissociation_reward = 0.0

    if dissociated:
        # Handle dissociation even without recorded barriers
        if clean_barriers is not None and len(clean_barriers) > 0:
            significant_barriers = [E for E in clean_barriers if E > E_threshold]
            diagnostics['filtered_barriers'] = significant_barriers
            diagnostics['num_barriers'] = len(significant_barriers)
        else:
            significant_barriers = []
            diagnostics['filtered_barriers'] = []
            diagnostics['num_barriers'] = 0

        if len(significant_barriers) == 0:
            # Spontaneous reaction or missed logging
            Ea_eff = 0.0
            diagnostics['Ea_eff'] = 0.0
            diagnostics['max_barrier'] = 0.0
        else:
            Ea_eff = _calculate_effective_barrier(
                significant_barriers, k_B, temperature
            )
            diagnostics['Ea_eff'] = Ea_eff
            diagnostics['max_barrier'] = max(significant_barriers)

            if not np.isfinite(Ea_eff):
                Ea_eff = 10.0  # 10 eV is essentially impassable
                diagnostics['Ea_eff'] = Ea_eff
                diagnostics['numerical_issue'] = True

        Ea_reward_raw = _sigmoid_reward(Ea_eff, ea_sigmoid_midpoint, ea_sigmoid_k)

        # Path-length penalty over filtered barrier count
        N = len(significant_barriers)
        path_penalty = Ea_gamma ** N if N > 0 else 1.0

        ea_reward = ea_weight * Ea_reward_raw * path_penalty
        dissociation_reward = dissociation_weight

    else:
        if require_dissociation:
            dissociation_reward = -timeout_penalty
        else:
            dissociation_reward = 0.0

    total_reward = ea_reward + dissociation_reward

    if cap_unsuccessful_reward and not dissociated and episode_ended:
        total_reward = max(total_reward, -1.0)

    total_reward = float(total_reward)
    ea_reward = float(ea_reward)
    dissociation_reward = float(dissociation_reward)
    
    if debug:
        return (total_reward, ea_reward, dissociation_reward), diagnostics
    else:
        return total_reward, ea_reward, dissociation_reward


def _validate_energies(
    activation_energies: Union[List[float], np.ndarray, None]
) -> Optional[List[float]]:
    """Validate activation energies (no silent filtering).

    Raises ValueError on non-finite/negative energies or None input, TypeError if
    the input cannot be converted to an array.
    """
    if activation_energies is None:
        raise ValueError(
            "activation_energies cannot be None.\n"
            "Rewards must be computed from actual activation energies.\n"
            "Pass non-None activation_energies list or handle upstream."
        )
    
    if isinstance(activation_energies, list):
        energies = np.array(activation_energies)
    elif isinstance(activation_energies, np.ndarray):
        energies = activation_energies.copy()
    else:
        try:
            energies = np.array(activation_energies)
        except Exception as e:
            raise TypeError(
                f"Cannot convert activation_energies to array: {type(activation_energies)}"
            ) from e

    energies = energies.flatten()

    # Reject invalid values rather than filtering them
    invalid = []
    for i, E in enumerate(energies):
        if not np.isfinite(E):
            invalid.append(f"energies[{i}]={E} (non-finite)")
        elif E < 0:
            invalid.append(f"energies[{i}]={E:.3f} (negative)")
    
    if invalid:
        raise ValueError(
            f"Invalid activation energies found: {invalid[:10]}"
            f"{'...' if len(invalid) > 10 else ''}. "
            f"All energies must be finite and non-negative."
        )
    
    return [float(E) for E in energies.tolist()]

def _calculate_effective_barrier(
    barriers: List[float], 
    k_B: float, 
    temperature: float
) -> float:
    """Effective activation energy in eV, no fallbacks (fails fast):
        Ea_eff = k_B * T * log(Sigma(exp(E_i/(k_B*T))))

    barriers must be non-empty, temperature > 0. Raises on overflow risk or a
    non-finite result.
    """
    if not barriers:
        raise ValueError("barriers list is empty")

    kT = k_B * temperature

    if kT <= 0:
        raise ValueError(
            f"Non-positive temperature: T={temperature}K (kT={kT:.3e} eV). "
            f"Temperature must be positive."
        )
    
    # Log-sum-exp for numerical stability
    scaled_barriers = np.array(barriers) / kT

    max_scaled = np.max(scaled_barriers)
    if max_scaled > 700:  # exp(700) near overflow
        raise OverflowError(
            f"log-sum-exp overflow risk: max(E/kT)={max_scaled:.1f} > 700. "
            f"Barriers: {barriers[:5]}{'...' if len(barriers) > 5 else ''}, "
            f"T={temperature}K. This indicates unrealistically high barriers."
        )

    exp_sum = np.sum(np.exp(scaled_barriers - max_scaled))
    log_sum = max_scaled + np.log(exp_sum)

    Ea_eff = float(kT * log_sum)

    if not np.isfinite(Ea_eff):
        raise FloatingPointError(
            f"Non-finite Ea_eff: {Ea_eff}. "
            f"Barriers: {barriers}, T={temperature}K, kT={kT:.3e} eV"
        )
    
    return Ea_eff

def _sigmoid_reward(
    Ea_eff: float,
    midpoint: float,
    steepness: float
) -> float:
    """Sigmoid-shaped reward in [0, 1] (steepness must be > 0):
        reward = 1 / (1 + exp((Ea_eff - midpoint) / steepness))
    Lower effective barrier -> higher reward.
    """
    if steepness <= 0:
        raise ValueError(
            f"steepness must be > 0, got {steepness}. "
            f"Check reward config ea_sigmoid_k parameter."
        )

    arg = (Ea_eff - midpoint) / steepness
    arg = np.clip(arg, -50, 50)  # prevent overflow
    reward = 1.0 / (1.0 + np.exp(arg))

    return float(reward)


def calculate_potential_energy_reward(
    pe_before: float,
    pe_after: float,
    weight: float = 0.1,
    max_delta: float = 10.0,
    gamma: float = 0.99
) -> float:
    """Potential-based shaped reward for a PE change, with Phi(s) = -PE(s):
        F(s, s') = Phi(s) - gamma*Phi(s') = PE_before - gamma*PE_after

    Over an episode the shaped rewards telescope to Phi(s_0) - gamma^T*Phi(s_T),
    depending only on start/end states, so the agent can't farm reward by taking
    longer paths. Raises ValueError on non-finite energies or gamma outside (0, 1].
    """
    if not np.isfinite(pe_before):
        raise ValueError(f"Non-finite pe_before: {pe_before}")
    if not np.isfinite(pe_after):
        raise ValueError(f"Non-finite pe_after: {pe_after}")
    if not (0 < gamma <= 1):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    shaped_delta = pe_before - gamma * pe_after
    shaped_delta = np.clip(shaped_delta, -max_delta, max_delta)  # guard numerics
    pe_reward = weight * shaped_delta

    return float(pe_reward)


class AdaptiveRewardScaler:
    """
    When return variance drops (agent saturating), increase ea_weight
    so small Ea differences produce meaningful advantage signal.
    Decays back toward base when variance recovers.
    """
    def __init__(self,
                 base_ea_weight=4.0,
                 max_ea_weight=12.0,
                 variance_threshold=1.0,
                 scale_step=0.5,
                 decay_step=0.1,
                 window_size=20):
        self.ea_weight = base_ea_weight
        self.base_ea_weight = base_ea_weight
        self.max_ea_weight = max_ea_weight
        self.variance_threshold = variance_threshold
        self.scale_step = scale_step
        self.decay_step = decay_step
        self.window_size = window_size
        self.return_history = []

    def update(self, episode_return):
        """Call after each episode with total return."""
        self.return_history.append(float(episode_return))
        if len(self.return_history) > self.window_size:
            self.return_history = self.return_history[-self.window_size:]

        if len(self.return_history) >= self.window_size:
            variance = float(np.var(self.return_history))
            mean_return = float(np.mean(self.return_history))

            if variance < self.variance_threshold and mean_return > 4.0:
                old = self.ea_weight
                self.ea_weight = min(self.max_ea_weight, self.ea_weight + self.scale_step)
                if self.ea_weight > old:
                    _LOG.info(f"[AdaptiveReward] var={variance:.3f} < {self.variance_threshold}, "
                          f"mean={mean_return:.2f} -> ea_weight: {old:.1f} -> {self.ea_weight:.1f}")
            else:
                if self.ea_weight > self.base_ea_weight:
                    old = self.ea_weight
                    self.ea_weight = max(self.base_ea_weight, self.ea_weight - self.decay_step)
                    if self.ea_weight < old:
                        _LOG.info(f"[AdaptiveReward] var={variance:.3f} healthy "
                              f"-> ea_weight: {old:.1f} -> {self.ea_weight:.1f}")

    def get_ea_weight(self):
        return self.ea_weight

    def state_dict(self):
        return {
            'ea_weight': self.ea_weight,
            'return_history': self.return_history,
        }

    def load_state_dict(self, d):
        self.ea_weight = d['ea_weight']
        self.return_history = d.get('return_history', [])    
