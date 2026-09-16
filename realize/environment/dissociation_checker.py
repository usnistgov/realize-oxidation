import numpy as np
from typing import Dict, Optional

DEFAULT_DISSOCIATION_THRESHOLD = 1.8  # Angstroms

class DissociationChecker:
    """Dissociation detection, cached by state revision."""

    def __init__(self, structure_manager, logger):
        self.structure_manager = structure_manager
        self.logger = logger

        # Cache keyed by state revision
        self._cache: Dict[int, bool] = {}
        self.distance_threshold = DEFAULT_DISSOCIATION_THRESHOLD

    def check_dissociation(self, state_revision: int,
                          force_check: bool = False) -> bool:
        """
        Dissociated if O-O distance > threshold OR the Si oxidation criterion is met.

        Args:
            state_revision: cache key
            force_check: force recalculation
        """
        if not force_check and state_revision in self._cache:
            return self._cache[state_revision]

        try:
            atoms = self.structure_manager.get_atoms()
            o1_idx, o2_idx = self.structure_manager.get_o2_indices()

            o_o_dist = atoms.get_distance(o1_idx, o2_idx, mic=True)

            # Non-finite distance indicates an error state
            if not np.isfinite(o_o_dist):
                if self.logger:
                    self.logger.log_warning(f"Non-finite O-O distance: {o_o_dist}")
                return False

            is_dissociated_by_distance = o_o_dist > self.distance_threshold

            # Fall back to oxidation criterion only if distance says not dissociated
            is_dissociated_by_oxidation = False
            if not is_dissociated_by_distance:
                is_dissociated_by_oxidation = self._check_oxidation_criterion()

            is_dissociated = is_dissociated_by_distance or is_dissociated_by_oxidation

            self._cache[state_revision] = is_dissociated

            # Bound cache size: keep 50 most recent revisions
            if len(self._cache) > 100:
                recent_revisions = sorted(self._cache.keys())[-50:]
                self._cache = {k: self._cache[k] for k in recent_revisions}

            if force_check or is_dissociated:
                if self.logger:
                    if is_dissociated_by_distance:
                        status = f"DISSOCIATED by distance: {o_o_dist:.3f} Angstrom > {self.distance_threshold:.2f} Angstrom"
                    elif is_dissociated_by_oxidation:
                        status = "DISSOCIATED by oxidation criterion"
                    else:
                        status = f"NOT dissociated: distance={o_o_dist:.3f} Angstrom"
                    self.logger.log_debug(status)

            return is_dissociated

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error checking dissociation: {e}")
            return False

    def _check_oxidation_criterion(self) -> bool:
        """Dissociated if >= 2 nearby Si atoms reach oxidation state >= 3."""
        try:
            atoms = self.structure_manager.get_atoms()
            if not atoms:
                return False

            o1_idx, o2_idx = self.structure_manager.get_o2_indices()

            charges = self.structure_manager.get_charges()
            if not charges:
                return False

            atom_types = self.structure_manager.get_atom_types()

            # Si oxidation state binned from ReaxFF charge
            def get_oxidation_state(charge):
                if charge < 0.175:
                    return 0
                elif charge < 0.525:
                    return 1
                elif charge < 0.875:
                    return 2
                elif charge < 1.225:
                    return 3
                else:
                    return 4

            # Si atoms (type 1) within cutoff of either O atom
            cutoff = 2.0
            si_near_o = set()

            for i in range(len(atoms) - 2):  # exclude the O2 pair (last 2 atoms)
                if atom_types[i] == 1:
                    dist_o1 = atoms.get_distance(i, o1_idx, mic=True)
                    dist_o2 = atoms.get_distance(i, o2_idx, mic=True)

                    if dist_o1 <= cutoff or dist_o2 <= cutoff:
                        si_near_o.add(i)

            if len(si_near_o) == 0:
                return False

            high_ox_count = 0
            for si_idx in si_near_o:
                if si_idx < len(charges):
                    charge = charges[si_idx]
                    ox_state = get_oxidation_state(charge)
                    if ox_state >= 3:
                        high_ox_count += 1

            return high_ox_count >= 2

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error in oxidation criterion: {e}")
            return False
