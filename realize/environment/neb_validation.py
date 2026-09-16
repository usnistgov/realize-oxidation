"""NEB result-validation diagnostics mixed into NEBCalculator.

Read-only validators (no __init__, no method overrides), so
`class NEBCalculator(NEBValidationMixin)` leaves the MRO unchanged:
  - has_single_local_maximum: energy-profile shape check (<=1 significant maximum)
  - has_nan_energies: scans log.lammps / per-replica logs / final.dump for NaN/Inf
"""
import numpy as np
import os
import re
import logging
_LOG = logging.getLogger(__name__)


class NEBValidationMixin:
    @staticmethod
    def has_single_local_maximum(arr, min_barrier_threshold=0.05):
        """
        Check if energy profile has at most one significant local maximum.

        Pathways with very small barriers (< min_barrier_threshold) are considered valid
        regardless of how many local maxima they have.

        Args:
            arr (numpy.ndarray): Energy profile array
            min_barrier_threshold (float, optional): Minimum energy barrier to consider (eV)

        Returns:
            bool: True if valid path, False otherwise
        """
        if len(arr) < 3:
            return True

        # Very small total barrier: valid regardless of profile shape
        max_barrier = np.max(arr) - arr[0]
        if max_barrier < min_barrier_threshold:
            _LOG.debug(f"Path has small barrier ({max_barrier:.4f} eV < {min_barrier_threshold:.4f} eV) - treating as valid despite shape")
            return True

        left = arr[:-2]
        center = arr[1:-1]
        right = arr[2:]
        local_maxima = (center > left) & (center > right)
        count = np.count_nonzero(local_maxima)
        return count <= 1


    def has_nan_energies(self) -> bool:
        """
        Check if NEB calculation produced NaN energies.

        Returns:
            bool: True if any NaN values detected
        """
        try:
            # log.lammps lives in the NEB working directory
            if os.path.exists('log.lammps'):
                with open('log.lammps', 'r', encoding='latin-1', errors='ignore') as f:
                    content = f.read()
                    # Word boundaries avoid matching "info", "infinite", etc.
                    nan_pattern = re.compile(r'\b(nan|-nan|inf|-inf)\b', re.IGNORECASE)
                    if nan_pattern.search(content):
                        if self.logger:
                            self.logger.log_warning("NaN or Inf detected in NEB log.lammps")
                        else:
                            _LOG.warning("NaN or Inf detected in NEB log.lammps")
                        return True

            # Check individual replica logs
            nan_pattern = re.compile(r'\b(nan|-nan|inf|-inf)\b', re.IGNORECASE)
            for i in range(self.num_images):
                replica_log = f"log.lammps.{i}"
                if os.path.exists(replica_log):
                    with open(replica_log, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read()
                        if nan_pattern.search(content):
                            if self.logger:
                                self.logger.log_warning(f"NaN or Inf detected in {replica_log}")
                            else:
                                _LOG.warning(f"NaN or Inf detected in {replica_log}")
                            return True

            final_dump = "final.dump"
            if os.path.exists(final_dump):
                try:
                    from ase import io
                    images = io.read(final_dump, index=':', format='lammps-dump-text')
                    for img in images:
                        if hasattr(img, 'get_potential_energy'):
                            try:
                                pe = img.get_potential_energy()
                                if np.isnan(pe) or np.isinf(pe):
                                    if self.logger:
                                        self.logger.log_warning("NaN or Inf energy in final.dump")
                                    return True
                            except:
                                pass
                except Exception as e:
                    if self.logger:
                        self.logger.log_debug(f"Could not check final.dump energies: {e}")

            return False

        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error checking for NaN energies: {e}")
            else:
                _LOG.warning(f"Error checking for NaN energies: {e}")
            return False
