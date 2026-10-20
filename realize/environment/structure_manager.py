"""Reads and writes the working atomic structure."""
import os
import shutil
from typing import Optional, Tuple
from ase import Atoms, io
import numpy as np
from realize.utils.file_utils import standardize_file_format, write_structure_file


class StructureManager:
    """Atomic structure I/O against coo_file, with a cached Atoms object."""

    def __init__(self, coo_file: str, logger):
        self.coo_file = coo_file
        self.logger = logger

        self._cached_atoms: Optional[Atoms] = None
        self._cache_valid = False

        # Original substrate counts (set in initialize_from_file)
        self.original_atom_count = 0
        self.original_si_count = 0
        self.original_o_count = 0
        self.cumulative_dissociated_o_count = 0

        # Set by O2LammpsEnv
        self.parent_env = None

        self._last_revision = 0

    def initialize_from_file(self, data_file: str):
        """Load original structure, copy to coo_file, record substrate counts."""

        atoms = io.read(data_file, format="lammps-data", atom_style="charge")

        # Record original count before any O2 is added
        self.original_atom_count = len(atoms)

        shutil.copy2(data_file, self.coo_file)
        standardize_file_format(self.coo_file, target_style='charge')

        self._cached_atoms = atoms
        self._cache_valid = True

        symbols = atoms.get_chemical_symbols()
        self.original_si_count = sum(1 for s in symbols if s == 'Si')
        self.original_o_count = sum(1 for s in symbols if s == 'O')

        if self.logger:
            self.logger.log_info(
                f"Initialized structure: {self.original_atom_count} atoms "
                f"(Si: {self.original_si_count}, O: {self.original_o_count})"
            )

    def get_atoms(self, force_reload: bool = False) -> Atoms:
        """Return cached structure, reloading from coo_file when invalid (PBC x,y only)."""
        if not self._cache_valid or force_reload:
            self._cached_atoms = io.read(
                self.coo_file,
                format="lammps-data",
                atom_style="charge"
            )
            self._cached_atoms.set_pbc((True, True, False))
            self._cache_valid = True
        return self._cached_atoms

    def update_atoms(self, atoms):
        """Write atoms to coo_file (standardized), invalidate cache, bump revision."""
        from realize.utils.file_utils import write_structure_file, standardize_file_format

        write_structure_file(atoms, self.coo_file, atom_style="charge")
        standardize_file_format(self.coo_file, target_style="charge")

        self.invalidate_cache()

        if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'state_revision'):
            self.parent_env.state_revision += 1
            self._last_revision = self.parent_env.state_revision

    def invalidate_cache(self):
        """Mark cached structure stale."""
        self._cache_valid = False

    def verify_substrate_integrity(self) -> bool:
        """False if Si atom count differs from the original substrate."""
        atoms = self.get_atoms()
        symbols = atoms.get_chemical_symbols()
        si_count = sum(1 for s in symbols if s == 'Si')
        
        if si_count != self.original_si_count:
            self.logger.log_error(
                f"Si count mismatch: {si_count} != {self.original_si_count}"
            )
            return False
        return True
    
    def save_state(self, filename: str) -> bool:
        """Write current structure to filename."""
        try:
            atoms = self.get_atoms()
            write_structure_file(atoms, filename, atom_style="charge")
            return True
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to save state: {e}")
            return False

    def load_state(self, filename: str) -> bool:
        """Load state into coo_file; rejects files with a changed Si count."""
        try:
            test_atoms = io.read(filename, format="lammps-data", atom_style="charge")
            symbols = test_atoms.get_chemical_symbols()
            si_count = sum(1 for s in symbols if s == 'Si')

            if si_count != self.original_si_count:
                if self.logger:
                    self.logger.log_error(f"Invalid Si count in state file")
                return False

            shutil.copy2(filename, self.coo_file)
            standardize_file_format(self.coo_file, target_style='charge')

            self._cache_valid = False

            return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to load state: {e}")
            return False


    def get_o2_indices(self) -> tuple:
        """Indices of the O2 pair (last two atoms); validates count and O type."""
        atoms = self.get_atoms()
        if atoms is None:
            raise ValueError("No atoms loaded")

        total_atoms = len(atoms)
        expected_substrate = self.original_atom_count + self.cumulative_dissociated_o_count

        if total_atoms < expected_substrate + 2:
            raise ValueError(f"O2 not present: only {total_atoms} atoms, expected at least {expected_substrate + 2}")

        o1_idx = total_atoms - 2
        o2_idx = total_atoms - 1

        symbols = atoms.get_chemical_symbols()
        if symbols[o1_idx] != 'O' or symbols[o2_idx] != 'O':
            raise ValueError(f"Last two atoms are not O: {symbols[o1_idx]}, {symbols[o2_idx]}")
    
        return o1_idx, o2_idx


    def get_o2_positions(self):
        """
        Get current positions of O2 atoms.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (o1_pos, o2_pos) or (None, None) if not found
        """
        try:
            o1_idx, o2_idx = self.get_o2_indices()
        except (ValueError, Exception) as e:
            # O2 absent or read error -> (None, None)
            if self.logger:
                self.logger.log_warning(f"O2 not found: {e}")
            return None, None
        
        if o1_idx is None or o2_idx is None:
            if self.logger:
                self.logger.log_warning("O2 indices not found")
            return None, None

        atoms = self.get_atoms()
        if atoms is None:
            return None, None
        
        positions = atoms.get_positions()
        return positions[o1_idx].copy(), positions[o2_idx].copy()
    
    def verify_integrity(self):
        """Alias for verify_substrate_integrity."""
        return self.verify_substrate_integrity()

    def get_charges(self) -> Optional[list]:
        """Parse per-atom charges (column 3) from the coo_file Atoms section."""
        try:
            with open(self.coo_file, 'r') as f:
                lines = f.readlines()

            atoms_section_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("Atoms"):
                    atoms_section_idx = i
                    break

            if atoms_section_idx is None:
                return None

            charges = []
            data_start = atoms_section_idx + 2  # skip "Atoms" header + blank line

            atoms = self.get_atoms()
            if not atoms:
                return None
                
            for i in range(len(atoms)):
                line_idx = data_start + i
                if line_idx < len(lines):
                    parts = lines[line_idx].split()
                    if len(parts) >= 3:
                        charges.append(float(parts[2]))
            
            return charges
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error getting charges: {e}")
            return None
    
    def get_atom_types(self) -> Optional[list]:
        """Parse per-atom type ids (column 2) from the coo_file Atoms section."""
        try:
            with open(self.coo_file, 'r') as f:
                lines = f.readlines()

            atoms_section_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("Atoms"):
                    atoms_section_idx = i
                    break

            if atoms_section_idx is None:
                return None

            atom_types = []
            data_start = atoms_section_idx + 2  # skip "Atoms" header + blank line

            atoms = self.get_atoms()
            if not atoms:
                return None
                
            for i in range(len(atoms)):
                line_idx = data_start + i
                if line_idx < len(lines):
                    parts = lines[line_idx].split()
                    if len(parts) >= 2:
                        atom_types.append(int(parts[1]))
            
            return atom_types
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error getting atom types: {e}")
            return None    
