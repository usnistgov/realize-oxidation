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

        # Fixed per-cycle Ea-sigmoid midpoint schedule. A fixed schedule gives
        # consistent per-cycle improvement pressure; an adaptive mu = mean(cycle
        # Ea_eff) would track the policy and invert the gradient for the same
        # physical process. Ea_eff values are still collected and logged but do
        # not drive mu.
        _EA_MIDPOINT_SCHEDULE = [1.6000, 1.5000, 1.4000, 1.3000, 1.2000]
        _next_cycle_idx = self._current_cycle + 1
        _new_mid = _EA_MIDPOINT_SCHEDULE[min(_next_cycle_idx, len(_EA_MIDPOINT_SCHEDULE) - 1)]

        if self._cycle_ea_effs:
            _cycle_avg_ea_eff = sum(self._cycle_ea_effs) / len(self._cycle_ea_effs)
            _n_diss = len(self._cycle_ea_effs)
        else:
            _cycle_avg_ea_eff = None
            _n_diss = 0

        if hasattr(self.config, 'reward'):
            _old_mid = self.config.reward.ea_sigmoid_midpoint
            self.config.reward.ea_sigmoid_midpoint = _new_mid
        elif hasattr(self.config, 'reward_config'):
            _old_mid = self.config.reward_config.ea_sigmoid_midpoint
            self.config.reward_config.ea_sigmoid_midpoint = _new_mid
        else:
            _old_mid = getattr(self.config, 'ea_sigmoid_midpoint', 1.2)
            self.config.ea_sigmoid_midpoint = _new_mid

        if _cycle_avg_ea_eff is not None:
            _LOG.info(f"  [FIXED MIDPOINT SCHEDULE] Cycle {self._current_cycle} complete: "
                  f"avg Ea_eff = {_cycle_avg_ea_eff:.4f} eV ({_n_diss} successful episodes). "
                  f"Midpoint: {_old_mid:.4f} -> {_new_mid:.4f} eV "
                  f"(scheduled for cycle {_next_cycle_idx})")
        else:
            _LOG.info(f"  [FIXED MIDPOINT SCHEDULE] Cycle {self._current_cycle} complete: "
                  f"no successful dissociations. Midpoint: {_old_mid:.4f} -> {_new_mid:.4f} eV "
                  f"(scheduled for cycle {_next_cycle_idx})")

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
