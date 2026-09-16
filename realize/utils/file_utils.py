import os
from ase import io
import numpy as np
from realize.config.paths import get_structure_file_path
import logging
_LOG = logging.getLogger(__name__)

def standardize_file_format(filename, target_style='charge', reference_file=None):
    """Force Si atoms to type 1 and O atoms to type 2 in a LAMMPS data file.

    The Si count is taken from the reference file (defaults to str.dat).
    """
    if reference_file is None:
        reference_file = get_structure_file_path()
    try:
        # Count Si atoms (type 1) in the reference file.
        si_count = 0
        with open(reference_file, 'r') as f:
            ref_lines = f.readlines()
        total_atoms = 0
        for line in ref_lines:
            if 'atoms' in line:
                total_atoms = int(line.strip().split()[0])
                break

        atoms_section = None
        for i, line in enumerate(ref_lines):
            if line.strip().startswith("Atoms"):
                atoms_section = i
                break
                
        if atoms_section is None:
            _LOG.info(f"Cannot find Atoms section in reference file {reference_file}")
            return False
            
        # Skip blank lines after the Atoms header to reach the data.
        data_start = atoms_section + 1
        while data_start < len(ref_lines) and not ref_lines[data_start].strip():
            data_start += 1

        # Read exactly total_atoms valid atom lines, skipping blanks/comments.
        atoms_read = 0
        i = data_start
        while i < len(ref_lines) and atoms_read < total_atoms:
            line = ref_lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom_type = int(parts[1])
                        if atom_type == 1:
                            si_count += 1
                        atoms_read += 1
                    except ValueError:
                        pass
            i += 1
        
        if atoms_read != total_atoms:
            _LOG.info(f"Warning: Expected {total_atoms} atoms in reference, found {atoms_read}")

        # Process the target file.
        with open(filename, 'r') as f:
            lines = f.readlines()

        atom_count = None
        atom_section_idx = None
        masses_section_idx = None
        
        for i, line in enumerate(lines):
            if 'atoms' in line and atom_count is None:
                atom_count = int(line.strip().split()[0])
            if line.strip().startswith("Atoms"):
                atom_section_idx = i
            if line.strip().startswith("Masses"):
                masses_section_idx = i
                
        if atom_count is None or atom_section_idx is None:
            _LOG.info(f"Cannot find atom count or Atoms section in {filename}")
            return False

        data_start = atom_section_idx + 1
        while data_start < len(lines) and not lines[data_start].strip():
            data_start += 1

        atom_data = []
        for i in range(data_start, data_start + atom_count):
            if i < len(lines):
                parts = lines[i].split()
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    current_type = int(parts[1])
                    charge = float(parts[2])
                    x, y, z = map(float, parts[3:6])

                    atom_data.append({
                        'id': atom_id,
                        'type': current_type,
                        'charge': charge,
                        'pos': (x, y, z),
                        'line': lines[i],
                        'parts': parts
                    })
        
        atom_data.sort(key=lambda x: x['id'])

        # Assign types by ID: the first si_count atoms are Si (type 1), the
        # trailing O2 pair and everything else are O (type 2).
        si_indices = []
        o_indices = []

        if len(atom_data) >= 2:
            o2_ids = [atom_data[-2]['id'], atom_data[-1]['id']]
        else:
            o2_ids = []

        for i, atom in enumerate(atom_data):
            if atom['id'] in o2_ids:
                atom['new_type'] = 2
                o_indices.append(i)
            elif i < si_count:
                atom['new_type'] = 1
                si_indices.append(i)
            else:
                atom['new_type'] = 2
                o_indices.append(i)

        new_atom_lines = []
        for atom in atom_data:
            parts = atom['parts'].copy()
            parts[1] = str(atom['new_type'])
            new_line = " ".join(parts) + "\n"
            new_atom_lines.append(new_line)

        masses_lines = [
            "Masses\n",
            "\n",
            "1 28.0855\n",  # Si
            "2 15.999\n"    # O
        ]
        
        # Rebuild the file: header up to Masses (or Atoms), the corrected
        # Masses block, the original Atoms header, then the retyped atoms.
        new_lines = []
        if masses_section_idx is not None:
            new_lines.extend(lines[:masses_section_idx])
        else:
            new_lines.extend(lines[:atom_section_idx])

        new_lines.extend(masses_lines)

        new_lines.append(lines[atom_section_idx])
        if data_start > atom_section_idx + 1:
            new_lines.append("\n")

        new_lines.extend(new_atom_lines)

        with open(filename, 'w') as f:
            f.writelines(new_lines)

        return True
        
    except Exception as e:
        _LOG.info(f"Error fixing element types in {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_lammps_data_file(filename):
    """Validate charge-style LAMMPS data format. Returns (is_valid, message)."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        atom_count = None
        for line in lines:
            if 'atoms' in line:
                atom_count = int(line.strip().split()[0])
                break
        if atom_count is None:
            return False, "Atom count not found"

        atom_section_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Atoms"):
                # First non-empty line after the Atoms header.
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                atom_section_idx = j
                break
        if atom_section_idx is None:
            return False, "Atoms section not found"

        for i in range(atom_count):
            parts = lines[atom_section_idx + i].split()
            if len(parts) < 6:
                return False, f"Invalid atom line: {lines[atom_section_idx + i]}"
            int(parts[0])  # ID
            int(parts[1])  # type
            float(parts[2])  # charge
            float(parts[3])
            float(parts[4])
            float(parts[5])
        atom_ids = []
        for i in range(atom_count):
            line_idx = atom_section_idx + i
            if line_idx >= len(lines):
                return False, f"File too short, expected {atom_count} atoms but found {i}"
        
            parts = lines[line_idx].split()
            if not parts:
                continue

            try:
                atom_id = int(parts[0])
                atom_ids.append(atom_id)
            except:
                return False, f"Invalid atom ID at line {line_idx+1}: {parts[0]}"

        if len(set(atom_ids)) != len(atom_ids):
            return False, "Duplicate atom IDs found"
        return True, "Valid charge format"
    except Exception as e:
        return False, f"Exception: {e}"


def convert_file_to_charge_style(filename):
    """Convert a full/atomic-style LAMMPS data file to charge style
    (ID type charge x y z 0 0 0), preserving existing types and coordinates.
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        atom_count = None
        for line in lines:
            if 'atoms' in line:
                atom_count = int(line.strip().split()[0])
                break

        atoms_section = None
        for i, line in enumerate(lines):
            if "Atoms" in line:
                atoms_section = i
                break

        if atoms_section is None or atom_count is None:
            raise ValueError("Missing Atoms section or atom count")

        atom_start = atoms_section + 2

        lines[atoms_section] = "Atoms # charge\n"

        for i in range(atom_count):
            idx = atom_start + i
            parts = lines[idx].split()
            if len(parts) >= 7:  # full: ID mol-ID type q x y z
                id_, _, type_, q, x, y, z = parts[:7]
            elif len(parts) == 5:  # atomic: ID type x y z
                id_, type_, x, y, z = parts[:5]
                q = "0.0"
            elif len(parts) == 6:  # charge without velocities: ID type q x y z
                id_, type_, q, x, y, z = parts[:6]
            else:
                raise ValueError(f"Cannot parse atom line: {lines[idx]}")

            new_line = f"{id_} {type_} {q} {x} {y} {z} 0 0 0\n"
            lines[idx] = new_line

        with open(filename, 'w') as f:
            f.writelines(lines)

    except Exception as e:
        _LOG.error(f"Failed to convert {filename} to charge format: {e}")
        import traceback
        traceback.print_exc()

def write_lammps_data(atoms, filename, atom_style='charge'):
    """Write atoms to a LAMMPS data file, then standardize and validate types."""
    try:
        atomic_numbers = atoms.get_atomic_numbers()

        original_indices = np.arange(len(atoms))
        original_positions = atoms.get_positions().copy()
        original_atomic_numbers = atomic_numbers.copy()

        atoms.write(filename, format='lammps-data', atom_style=atom_style)

        standardize_file_format(filename, target_style=atom_style)

        is_valid, msg = validate_lammps_data_file(filename)
        if not is_valid:
            _LOG.info(f"Warning: Invalid file after write: {msg}")

        return True
    except Exception as e:
        _LOG.info(f"Error in write_lammps_data: {e}")
        import traceback
        traceback.print_exc()
        return False

def write_structure_file(atoms, filename, atom_style='charge'):
    """Write atoms with Si->tag 1, O->tag 2, then standardize and validate."""
    try:
        atoms_copy = atoms.copy()

        # Tag by element before writing so ASE assigns the right LAMMPS types.
        symbols = atoms_copy.get_chemical_symbols()
        for i, symbol in enumerate(symbols):
            if symbol == 'Si':
                atoms_copy[i].tag = 1
            elif symbol == 'O':
                atoms_copy[i].tag = 2
            else:
                raise ValueError(f"Unexpected element: {symbol}")

        atoms_copy.write(filename, format='lammps-data', atom_style=atom_style)

        standardize_file_format(filename, target_style=atom_style)

        is_valid, msg = validate_lammps_data_file(filename)
        if not is_valid:
            _LOG.info(f"Warning: File validation failed after write: {msg}")

        return True
    except Exception as e:
        _LOG.info(f"Error in write_structure_file: {e}")
        import traceback
        traceback.print_exc()
        return False
