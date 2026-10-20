"""Substrate reset between oxidation cycles."""
import os
import shutil
from ase import io as ase_io

from realize.config.paths import get_structure_file_path
import logging
_LOG = logging.getLogger(__name__)


class PPOTrainerEnvResetMixin:
    """Environment substrate-reset logic for PPOTrainer.

    ``_should_reset_environment`` (per-episode check) and
    ``_reset_to_fresh_substrate`` (restores the pristine slab + advances the
    Ea-midpoint cycle schedule) resolve config/env/logger and the cycle
    bookkeeping attributes through the MRO back to PPOTrainer (a plain orchestrator
    with no torch Parameters), so this mixin cannot perturb the policy fingerprint.
    """
    def _should_reset_environment(self, episode: int) -> bool:
        """Check if environment should be reset based on config thresholds."""
        reset_interval = 0
        max_o = 0

        if hasattr(self.config, 'runtime'):
            reset_interval = getattr(self.config.runtime, 'reset_surface_every_n_episodes', 0)
            max_o = getattr(self.config.runtime, 'max_cumulative_o_before_reset', 0)
        else:
            reset_interval = getattr(self.config, 'reset_surface_every_n_episodes', 0)
            max_o = getattr(self.config, 'max_cumulative_o_before_reset', 0)

        if reset_interval > 0 and episode > 0 and episode % reset_interval == 0:
            _LOG.debug(f"  [RESET CHECK] Periodic reset triggered (episode {episode} % {reset_interval} == 0)")
            return True

        if max_o > 0:
            cumulative_o = getattr(self.env.structure_manager, 'cumulative_dissociated_o_count', 0)
            if cumulative_o >= max_o:
                _LOG.debug(f"  [RESET CHECK] O threshold triggered ({cumulative_o} >= {max_o})")
                return True

        return False

    def _reset_to_fresh_substrate(self, episode: int):
        """Reset environment to original clean substrate."""

        # Adaptive per-cycle midpoint schedule (manuscript "ADAPTIVE MIDPOINT SCHEDULE",
        # property #25). mu tracks a low quantile of the achieved (reward-consistent,
        # TS-relative) Ea_eff distribution, tightens at most one rate-decade per cycle,
        # is floored at a physical timescale, and may loosen only under a sustained rise
        # in achieved median. mu is stored as ea_sigmoid_midpoint.
        from realize.ppo.schedulers import adaptive_mu_schedule
        import math as _math
        import numpy as _np
        _k_B = 8.617333262e-5
        _T = 1000.0
        if hasattr(self.config, 'reward'):
            _rc = self.config.reward
        elif hasattr(self.config, 'reward_config'):
            _rc = self.config.reward_config
        else:
            _rc = self.config
        _nu0 = _rc.nu_0
        _mu_floor = _k_B * _T * _math.log(_nu0 * _rc.t_target_floor)
        _mu_init = _k_B * _T * _math.log(_nu0 * _rc.t_target_init)
        _old_mid = getattr(_rc, 'ea_sigmoid_midpoint', _mu_init)
        if self._current_cycle == 0 and not self._cycle_avg_ea:
            _old_mid = _mu_init
        _n_diss = len(self._cycle_ea_effs)
        _cycle_avg_ea_eff = (sum(self._cycle_ea_effs) / len(self._cycle_ea_effs)) if self._cycle_ea_effs else None
        _new_mid, _sched_info = adaptive_mu_schedule(
            _old_mid, self._cycle_ea_effs, self._cycle_avg_ea,
            T=_T, nu_0=_nu0,
            mu_quantile=_rc.mu_quantile, mu_margin=_rc.mu_margin, mu_floor=_mu_floor,
            max_decades_per_cycle=_rc.mu_max_decades_per_cycle,
            ratchet_release_cycles=_rc.mu_ratchet_release_cycles,
        )
        if hasattr(self.config, 'reward'):
            self.config.reward.ea_sigmoid_midpoint = _new_mid
        elif hasattr(self.config, 'reward_config'):
            self.config.reward_config.ea_sigmoid_midpoint = _new_mid
        else:
            self.config.ea_sigmoid_midpoint = _new_mid
        if self._cycle_ea_effs:
            self._cycle_avg_ea.append(float(_np.median(self._cycle_ea_effs)))
        _LOG.info(f"  [ADAPTIVE MIDPOINT] Cycle {self._current_cycle} complete: "
              f"{_n_diss} successful; mu {_old_mid:.4f} -> {_new_mid:.4f} eV "
              f"[{_sched_info.get('reason','?')}, floor={_mu_floor:.4f}]")
        self._cycle_ea_effs = []
        self._current_cycle += 1
        _LOG.info(f"  [CYCLE] Entering cycle {self._current_cycle} "
              f"(history: {[f'{e:.4f}' if e is not None else 'N/A' for e in self._cycle_avg_ea]})")

        old_count = getattr(self.env.structure_manager, 'cumulative_dissociated_o_count', 0)

        reset_interval = getattr(self.config.runtime, 'reset_surface_every_n_episodes', 0)
        max_o = getattr(self.config.runtime, 'max_cumulative_o_before_reset', 0)

        trigger = "UNKNOWN"
        if reset_interval > 0 and episode > 0 and episode % reset_interval == 0:
            trigger = f"PERIODIC (every {reset_interval} episodes)"
        elif max_o > 0 and old_count >= max_o:
            trigger = f"O_COUNT_THRESHOLD ({old_count} >= {max_o})"

        # Three dirname() climbs from this module reach the repo root; the
        # pristine slab lives in data/ there (== get_structure_file_path()).
        original_structure = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'str.dat')

        if not os.path.exists(original_structure):
            original_structure = get_structure_file_path()
            _LOG.info(f"[RESET] WARNING: data/str.dat not found, using {original_structure}")

        shutil.copy2(original_structure, self.env.coo_file)
        self.env.structure_manager.invalidate_cache()
        self.env.structure_manager.cumulative_dissociated_o_count = 0
        self.env.cumulative_dissociated_o_count = 0

        try:
            fresh_atoms = ase_io.read(original_structure, format="lammps-data", atom_style="charge")
            self.env.structure_manager.original_atom_count = len(fresh_atoms)
            symbols = fresh_atoms.get_chemical_symbols()
            self.env.structure_manager.original_si_count = sum(1 for s in symbols if s == 'Si')
            self.env.structure_manager.original_o_count = sum(1 for s in symbols if s == 'O')
            _LOG.debug(f"  Reset atom counts: {self.env.structure_manager.original_atom_count} total, "
                  f"{self.env.structure_manager.original_si_count} Si, "
                  f"{self.env.structure_manager.original_o_count} O")
        except Exception as e:
            _LOG.info(f"  WARNING: Could not update atom counts: {e}")

        if hasattr(self.env, 'episode_outcomes'):
            self.env.episode_outcomes.clear()
        self.env.previous_episode_dissociated = False
        self.env.o2_removed = False
        self.env.state_revision += 1

        self.logger.log_info(f"Environment reset at episode {episode} (was at {old_count} O atoms)")
