import numpy as np
import random
from typing import Tuple, Optional
from ase import io
from realize.config.paths import get_o2_file_path


class O2Manager:
    """O2 molecule creation, transformation, and removal."""

    def __init__(self, structure_manager, logger):
        self.structure_manager = structure_manager
        self.logger = logger
        self.bond_length = 1.19725  # O2 bond length in Angstroms
        
    def create_initial_o2(self, top_z: float, lammps_interface=None):
        """Create initial O2 molecule above substrate."""

        self.structure_manager.invalidate_cache()
        atoms = self.structure_manager.get_atoms()
        current_count = len(atoms)
        expected_substrate = (self.structure_manager.original_atom_count +
                             self.structure_manager.cumulative_dissociated_o_count)

        # O2 already present (2 extra atoms beyond expected substrate)
        if current_count >= expected_substrate + 2:
            if self.logger:
                self.logger.log_debug(f"O2 already present, skipping creation")
            return False

        # Exactly the substrate count: add O2
        if current_count == expected_substrate:
            try:
                from ase import Atoms

                # Logged spawn position within central 20%-80% of cell
                cell_x = atoms.cell[0, 0]
                cell_y = atoms.cell[1, 1]

                x_pos = np.random.uniform(0.2 * cell_x, 0.8 * cell_x)
                y_pos = np.random.uniform(0.2 * cell_y, 0.8 * cell_y)

                # Actual O2 placement: anywhere in cell (PBC in x,y), z above surface
                cell_x = atoms.cell[0, 0]
                cell_y = atoms.cell[1, 1]

                o2_center = np.array([
                    np.random.uniform(0, cell_x),
                    np.random.uniform(0, cell_y),
                    top_z + 2.5
                ])

                if self.logger:
                    self.logger.log_debug(f"O2 spawn position: x={x_pos:.2f}, y={y_pos:.2f}")

                # Random orientation (unit vector for O2 axis)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)

                axis = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])

                half_bond = self.bond_length / 2.0
                o1_pos = o2_center + half_bond * axis
                o2_pos = o2_center - half_bond * axis

                o2_molecule = Atoms('OO', positions=[o1_pos, o2_pos])
                combined = atoms + o2_molecule

                if len(combined) != expected_substrate + 2:
                    if self.logger:
                        self.logger.log_error(f"O2 addition failed: expected {expected_substrate + 2} atoms, got {len(combined)}")
                    return False
                
                self.structure_manager.update_atoms(combined)

                if self.logger:
                    self.logger.log_debug(f"Added O2 at z={top_z + 2.5:.2f} with bond length {self.bond_length:.3f}")
                    self.logger.log_debug(f"Structure now has {len(combined)} atoms (substrate: {expected_substrate}, O2: 2)")

                if lammps_interface is not None:
                    lammps_interface.relax_structure(self.structure_manager.coo_file)

                return True

            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Failed to create O2: {e}")
                return False

        if self.logger:
            self.logger.log_error(f"Unexpected atom count: {current_count} (expected substrate: {expected_substrate})")
        return False

    def apply_o2_transformation(self, o1_pos: np.ndarray, o2_pos: np.ndarray) -> bool:
        """Apply new O2 positions to structure."""
        try:
            atoms = self.structure_manager.get_atoms()
            o1_idx, o2_idx = self.structure_manager.get_o2_indices()

            positions = atoms.get_positions()
            positions[o1_idx] = o1_pos
            positions[o2_idx] = o2_pos
            atoms.set_positions(positions)

            atoms.wrap()  # PBC

            initial_revision = getattr(self.structure_manager, '_last_revision', -1)
            self.structure_manager.update_atoms(atoms)
            new_revision = getattr(self.structure_manager, '_last_revision', -1)

            # State revision must increment after an update
            if initial_revision >= 0 and new_revision <= initial_revision:
                if self.logger:
                    self.logger.log_warning(
                        f"State revision not incremented after O2 transformation "
                        f"({initial_revision} -> {new_revision})"
                    )
                    
            return True            
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to apply O2 transformation: {e}")
            return False
    
    def check_o2_distance(self) -> Tuple[float, np.ndarray]:
        """Return O-O distance and vector (minimum-image convention)."""
        try:
            atoms = self.structure_manager.get_atoms()
            o1_idx, o2_idx = self.structure_manager.get_o2_indices()

            dist = atoms.get_distance(o1_idx, o2_idx, mic=True)
            vec = atoms.get_distance(o1_idx, o2_idx, mic=True, vector=True)

            return dist, vec

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error checking O2 distance: {e}")
            # inf signals error/unknown state
            return np.inf, np.zeros(3)

    def remove_o2(self) -> bool:
        """Remove O2 molecule, keeping only substrate atoms."""
        try:
            atoms = self.structure_manager.get_atoms()
            expected_substrate = (self.structure_manager.original_atom_count +
                                self.structure_manager.cumulative_dissociated_o_count)

            if len(atoms) <= expected_substrate:
                return False

            substrate = atoms[:expected_substrate].copy()
            self.structure_manager.update_atoms(substrate)

            return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to remove O2: {e}")
            return False

    def remove_o2_molecule(self) -> bool:
        """Alias for remove_o2."""
        return self.remove_o2()

    def has_o2(self) -> bool:
        """True if an O2 molecule is present in the structure."""
        atoms = self.structure_manager.get_atoms()
        expected_substrate = (self.structure_manager.original_atom_count +
                             self.structure_manager.cumulative_dissociated_o_count)
        return len(atoms) >= expected_substrate + 2
