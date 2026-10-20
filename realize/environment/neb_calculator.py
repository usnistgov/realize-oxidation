"""NEB barrier calculations with partitioned LAMMPS."""
import numpy as np
import re
import os
import gc
import subprocess
import time
import json
from typing import Optional
from lammps import lammps
from ase.calculators.lammpsrun import LAMMPS
from ase import io
from ase.mep import NEB
from ase.io.lammpsrun import read_lammps_dump_text
from realize.utils.file_utils import standardize_file_format
from realize.config.paths import (
    get_neb_executable_path, get_neb_results_dir, get_results_dir, get_temp_dir, get_neb_input_file_path, get_reaxff_file_path
)
from realize.config.paths import INPUT_DIR
import shutil
from realize.environment.neb_validation import NEBValidationMixin
import logging
_LOG = logging.getLogger(__name__)


# Energy/force unit conversions
KCAL_MOL_TO_EV = 0.0433634  # kcal/mol -> eV
EV_TO_KCAL_MOL = 23.06035   # eV -> kcal/mol (1/KCAL_MOL_TO_EV)
KCAL_MOL_PER_ANG_TO_EV_PER_ANG = KCAL_MOL_TO_EV  # (kcal/mol)/Angstrom -> eV/Angstrom

class NEBCalculator(NEBValidationMixin):
    """
    Nudged Elastic Band barrier calculation via LAMMPS for O2 dissociation on SiO2.
    Mixes in NEBValidationMixin (read-only validators; MRO unchanged).
    """
    def __init__(self, initial_file, final_file, neb_in_file="neb_lammps.in", logger=None, env=None, num_procs_override=None):
        """
        Args:
            initial_file (str): initial state file
            final_file (str): final state file
            neb_in_file (str, optional): NEB input file
            logger: optional logger
            env: optional O2LammpsEnv used for relaxation
        """
        self.logger = logger
        self.env = env  # used for structure relaxation

        # Cache paths
        self.temp_dir = get_temp_dir()
        self.results_dir = get_neb_results_dir()
        self.input_dir = INPUT_DIR

        self.reaxff_file = get_reaxff_file_path()

        self.init_file = os.path.abspath(initial_file)
        self.final_file = os.path.abspath(final_file)

        # Default neb_in_file resolves to the one shipped in INPUT_DIR
        if neb_in_file == "neb_lammps.in":
            self.neb_in_file = os.path.join(self.input_dir, "neb_lammps.in")
        else:
            self.neb_in_file = neb_in_file

        self.lammps_exe = get_neb_executable_path()

        os.makedirs(self.results_dir, exist_ok=True)

        self.images = []
        self.num_images = 7
        self.energy_array = None
        self._num_procs_override = num_procs_override

    def relax_structure_using_env(self, structure_file, max_steps=1000):
        """
        Relax structure_file via the environment's shared LAMMPS interface.

        Returns the relaxed ASE Atoms, or None on failure.
        """
        try:
            if self.env is None or not hasattr(self.env, 'lammps_interface'):
                if self.logger:
                    self.logger.log_error("No environment with LAMMPS interface available")
                return None

            # Use the shared interface so LAMMPS settings stay consistent
            relaxed_file = os.path.join(get_temp_dir(), "neb_relaxed_temp.data")

            shutil.copy2(structure_file, relaxed_file)

            self.env.lammps_interface.relax_structure(relaxed_file)

            if os.path.exists(relaxed_file):
                relaxed = io.read(relaxed_file, format='lammps-data', atom_style='charge')
                relaxed.set_pbc((True, True, True))

                if relaxed_file != structure_file:
                    shutil.copy2(relaxed_file, structure_file)

                os.remove(relaxed_file)

                return relaxed
            else:
                if self.logger:
                    self.logger.log_warning("Relaxed structure file not found")
                return None

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error during structure relaxation: {e}")
            return None

    def create_images(self, relax_final=True, relax_steps=1000):
        """
        Build NEB images by interpolating initial->final structures.

        Args:
            relax_final (bool): relax the final structure and use it as the last image
            relax_steps (int): relaxation step count
        """
        temp_final_file = None

        initial = io.read(self.init_file, format='lammps-data', atom_style='charge')
        final = io.read(self.final_file, format='lammps-data', atom_style='charge')

        # Image count comes from line 4 of the NEB input file
        with open(self.neb_in_file, 'r') as f:
            lines = f.readlines()
        base_num_images = int(lines[3].split()[-1])

        initial.set_pbc((True, True, True))
        final.set_pbc((True, True, True))

        atom_num = len(initial)
        if len(final) != atom_num:
            raise ValueError(f"Atom count mismatch: {atom_num} vs {len(final)}")

        # O2 pair is the last two atoms
        o1_idx = atom_num - 2
        o2_idx = atom_num - 1
        initial_o_o_dist = initial.get_distance(o1_idx, o2_idx, mic=True)
        final_o_o_dist = final.get_distance(o1_idx, o2_idx, mic=True)

        if self.logger:
            self.logger.log_debug(f"Initial O-O distance: {initial_o_o_dist:.3f} Angstrom")
            self.logger.log_debug(f"Final (unrelaxed) O-O distance: {final_o_o_dist:.3f} Angstrom")

        # Match final atom ordering to initial, unwrapping with MIC so the
        # interpolation path doesn't jump across periodic boundaries
        reordered_final = initial.copy()

        initial_positions = initial.get_positions()
        final_positions = final.get_positions()
        cell = initial.get_cell()

        reordered_positions = initial_positions.copy()
        for i in range(atom_num):
            diff = final_positions[i] - initial_positions[i]
            # Minimum image convention per axis
            for dim in range(3):
                if cell[dim, dim] > 0:
                    while diff[dim] > cell[dim, dim] / 2:
                        diff[dim] -= cell[dim, dim]
                    while diff[dim] < -cell[dim, dim] / 2:
                        diff[dim] += cell[dim, dim]
            reordered_positions[i] = initial_positions[i] + diff

        reordered_final.set_positions(reordered_positions)

        relaxed_final = None
        if relax_final:
            if self.env is not None:
                if self.logger:
                    self.logger.log_debug("Relaxing final structure...")

                temp_final_file = os.path.join(get_temp_dir(), "temp_final_for_relax.data")
                io.write(temp_final_file, reordered_final, format='lammps-data', atom_style='charge')
                standardize_file_format(temp_final_file, target_style='charge')

                relaxed = self.relax_structure_using_env(temp_final_file, relax_steps)

                if relaxed is not None:
                    relaxed_final = relaxed
                    relaxed_o_o_dist = relaxed_final.get_distance(o1_idx, o2_idx, mic=True)
                    if self.logger:
                        self.logger.log_debug(f"Relaxed final O-O distance: {relaxed_o_o_dist:.3f} Angstrom")

                    self.num_images = base_num_images
                else:
                    if self.logger:
                        self.logger.log_warning("Relaxation failed, using unrelaxed structure only")
                    relax_final = False
                    self.num_images = base_num_images
            else:
                if self.logger:
                    self.logger.log_warning("No environment instance provided, cannot relax final structure")
                relax_final = False
                self.num_images = base_num_images
        else:
            self.num_images = base_num_images

        # IDPP-interpolate between initial and unrelaxed final
        num_intermediate = base_num_images - 2 if base_num_images > 2 else 4
        temp_images = [initial] + [initial.copy() for _ in range(num_intermediate)] + [reordered_final]
        neb = NEB(temp_images)
        neb.interpolate('idpp')

        if relax_final and relaxed_final is not None:
            # Use relaxed final as the last image (total stays base_num_images)
            self.images = temp_images[:-1] + [relaxed_final]
            if self.logger:
                self.logger.log_debug(f"Created {len(self.images)} images: initial + {num_intermediate} IDPP + relaxed final")
        else:
            self.images = temp_images
            if self.logger:
                self.logger.log_debug(f"Created {len(self.images)} images: initial + {num_intermediate} IDPP + final")

        # Write each image as a LAMMPS NEB image file (id x y z per atom)
        tmp_dir = get_temp_dir()
        for i, image in enumerate(self.images):
            image.wrap()

            tmp_filename = os.path.join(tmp_dir, f'idpp_image_{i}.dat')
            io.write(tmp_filename, image, format='lammps-data', atom_style='charge')
            standardize_file_format(tmp_filename, target_style='charge')

            coords = []
            with open(tmp_filename, 'r') as f:
                lines = f.readlines()

            atom_start_idx = None
            for idx, line in enumerate(lines):
                if "Atoms" in line:
                    atom_start_idx = idx + 2
                    break

            if atom_start_idx is None:
                raise ValueError(f"Could not find 'Atoms' section")

            for line in lines[atom_start_idx:atom_start_idx + atom_num]:
                parts = line.split()
                coords.append(f"  {parts[0]} {parts[3]} {parts[4]} {parts[5]}\n")

            with open(f'image.{i+1}', 'w') as out:
                out.write(f"{atom_num}\n")
                for coord_line in coords:
                    out.write(coord_line)

        for i in range(len(self.images)):
            tmp_file = os.path.join(tmp_dir, f'idpp_image_{i}.dat')
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        if temp_final_file is not None and os.path.exists(temp_final_file):
            try:
                os.remove(temp_final_file)
            except Exception as e:
                if self.logger:
                    self.logger.log_debug(f"Could not remove temp file: {e}")

    def extract_neb_convergence_info(self, log_file_path='log.lammps'):
        """
        Parse NEB convergence info from a log file; forces returned in eV/Angstrom.
        """
        FORCE_CONVERSION = KCAL_MOL_PER_ANG_TO_EV_PER_ANG
        ENERGY_CONVERSION = KCAL_MOL_TO_EV

        convergence_info = {
            'total_steps': 0,
            'final_energy': None,
            'final_max_force': None,
            'converged': False,
            'final_ebf': None,
            'final_ebr': None,
            'max_replica_force': None,
            'max_atom_force': None
        }

        try:
            with open(log_file_path, 'r', encoding='latin-1', errors='ignore') as f:
                lines = f.readlines()

            # Collect numeric rows under the "Step ... MaxReplicaForce MaxAtomForce" header
            neb_data_lines = []
            in_neb_section = False

            for line in lines:
                if 'Step' in line and 'MaxReplicaForce' in line and 'MaxAtomForce' in line:
                    in_neb_section = True
                    continue

                if in_neb_section:
                    parts = line.strip().split()
                    if len(parts) > 10 and parts[0].isdigit():
                        neb_data_lines.append(line.strip())
                    elif len(parts) == 0 or (parts and not parts[0].isdigit()):
                        in_neb_section = False

            if neb_data_lines:
                # Last row holds the final converged values
                last_line = neb_data_lines[-1]
                parts = last_line.split()

                convergence_info['total_steps'] = int(parts[0])

                convergence_info['max_replica_force'] = float(parts[1]) * FORCE_CONVERSION
                convergence_info['max_atom_force'] = float(parts[2]) * FORCE_CONVERSION

                # ebf/ebr from the log are unreliable; barrier is computed from the
                # energy profile in get_activation_energy instead
                convergence_info['final_ebf'] = None
                convergence_info['final_ebr'] = None

                # Converged if it finished in under 2000 steps
                convergence_info['converged'] = convergence_info['total_steps'] < 2000

                convergence_info['final_max_force'] = convergence_info['max_replica_force']

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error extracting NEB convergence info: {e}")
            else:
                _LOG.info(f"Error extracting NEB convergence info: {e}")

        return convergence_info

    def parse_neb_logs_for_all_replicas(self, num_replicas=None):
        """Aggregate convergence info across all per-replica logs."""
        if num_replicas is None:
            num_replicas = self.num_images

        all_convergence_info = {
            'replica_steps': [],
            'replica_energies': [],
            'replica_forces': [],
            'overall_converged': True,
            'max_steps_taken': 0,
            'average_steps': 0
        }

        for i in range(num_replicas):
            log_file = f'log.lammps.{i}'
            if os.path.exists(log_file):
                info = self.extract_neb_convergence_info(log_file)

                all_convergence_info['replica_steps'].append(info['total_steps'])
                if info['final_energy'] is not None:
                    all_convergence_info['replica_energies'].append(info['final_energy'])
                if info['final_max_force'] is not None:
                    all_convergence_info['replica_forces'].append(info['final_max_force'])

                all_convergence_info['overall_converged'] &= info['converged']

                if info['total_steps'] > all_convergence_info['max_steps_taken']:
                    all_convergence_info['max_steps_taken'] = info['total_steps']

        if all_convergence_info['replica_steps']:
            all_convergence_info['average_steps'] = np.mean(all_convergence_info['replica_steps'])

        return all_convergence_info

    def get_neb_convergence_summary(self):
        """Summarize NEB convergence; barrier comes from the energy profile."""
        FORCE_CONVERSION = 0.0433634

        main_convergence = self.extract_neb_convergence_info('log.lammps')
        replica_info = self.parse_neb_logs_for_all_replicas(self.num_images)

        actual_activation_energy = self.get_activation_energy(silent=True)

        # Barrier uses the computed energy profile, not raw log ebf/ebr
        summary = {
            'total_steps': main_convergence['total_steps'] or replica_info['max_steps_taken'],
            'converged': main_convergence['converged'],
            'final_max_replica_force': main_convergence['max_replica_force'],
            'final_max_atom_force': main_convergence['max_atom_force'],
            'energy_barrier_forward': actual_activation_energy,
            'energy_barrier_reverse': main_convergence['final_ebr'],
            'raw_ebf': main_convergence['final_ebf'],
            'raw_ebr': main_convergence['final_ebr'],
            'force_tolerance': 10.0 * FORCE_CONVERSION,
            'force_tolerance_original': 10.0,
            'energy_tolerance': 1e-09,
            'max_steps_allowed': 10000,
            'replica_info': replica_info
        }

        return summary


    def run_neb(self, episode=0, step=0, try_num=0):
        """
        Run the NEB calculation via mpirun + LAMMPS; returns success status.

        episode/step/try_num are used for logging and output file naming.
        """
        output_file = None
        try:
            from realize.config.paths import get_reaxff_file_path

            if self.logger:
                self.logger.log_debug(f"run_neb() called with episode={episode}, step={step}, try={try_num}")
            else:
                _LOG.debug(f"run_neb() called with episode={episode}, step={step}, try={try_num}")


            # Copy endpoints into the working directory in standardized format
            shutil.copy2(self.init_file, "init")
            standardize_file_format("init", target_style='charge')
            shutil.copy2(self.final_file, "final")
            standardize_file_format("final", target_style='charge')

            # Rewrite NEB input with absolute ReaxFF path and actual image count
            temp_neb_in = "temp_neb.in"
            with open(self.neb_in_file, 'r') as src:
                with open(temp_neb_in, 'w') as dst:
                    for line in src:
                        if "pair_coeff * *" in line and "41524_2023_1112_MOESM1_ESM.txt" in line:
                            dst.write(f"pair_coeff * * {get_reaxff_file_path()} Si O\n")
                        elif "variable numreplicas equal" in line or "numreplicas equal" in line:
                            dst.write(f"variable numreplicas equal {self.num_images}\n")
                        else:
                            dst.write(line)

            # Available MPI ranks: worker override, else SLURM_NTASKS, else num_images
            if self._num_procs_override is not None:
                available_ranks = self._num_procs_override
            else:
                slurm_ntasks = os.environ.get("SLURM_NTASKS")
                available_ranks = int(slurm_ntasks) if slurm_ntasks is not None else self.num_images

            # Partition requires total_ranks == num_images * procs_per_image
            procs_per_image = max(1, available_ranks // self.num_images)
            total_procs = self.num_images * procs_per_image

            mpirun_path = os.environ.get("NEB_MPIRUN", "/usr/bin/mpirun")

            cmd = [mpirun_path,
                   "-np", str(total_procs),
                   self.lammps_exe,
                   "-in", temp_neb_in,
                   "-partition", f"{self.num_images}x{procs_per_image}"]

            # Keep SLURM/PMI/PMIX so MPI can bootstrap; drop vars that force a
            # conflicting MPI stack/launcher
            clean_env = os.environ.copy()
            for k in list(clean_env.keys()):
                if k.startswith(("I_MPI", "HYDRA", "MPICH")):
                    clean_env.pop(k, None)

            for k in list(clean_env.keys()):
                if k.startswith(("I_MPI", "HYDRA")):
                    clean_env.pop(k, None)


            if self.logger:
                self.logger.log_debug(f"NEB: {self.num_images} images x {procs_per_image} procs = {total_procs} total")

            if self.logger:
                self.logger.log_debug(f"Running NEB command: {' '.join(cmd)}")
            else:
                _LOG.debug(f"Running NEB command: {' '.join(cmd)}")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env)
            stdout, stderr = proc.communicate(timeout=120)

            # Partitioned LAMMPS often returns nonzero on MPI cleanup despite valid
            # output; treat a substantial log.lammps as success
            if proc.returncode != 0:
                if os.path.exists('log.lammps') and os.path.getsize('log.lammps') > 1000:
                    proc.returncode = 0
                else:
                    stderr_text = stderr.decode('utf-8', errors='ignore')
                    _LOG.debug(f"[NEB DEBUG] Return code: {proc.returncode}")
                    _LOG.debug(f"[NEB DEBUG] STDERR: {stderr_text}")
                    _LOG.debug(f"[NEB DEBUG] STDOUT: {stdout.decode('utf-8', errors='ignore')[-2000:]}")

            if proc.returncode == 0:
                try:
                    convergence_summary = self.get_neb_convergence_summary()

                    energy_profile = self.read_neb_energy(verbose=False)

                    _LOG.info(f"NEB energy profile: {energy_profile}")

                    calc_barrier = np.max(energy_profile) if len(energy_profile) > 0 else -1
                    _LOG.debug(f"NEB completed: {convergence_summary['total_steps']} steps, "
                          f"converged={convergence_summary['converged']}, "
                          f"max_force={convergence_summary['final_max_replica_force']:.3f}, "
                          f"barrier={calc_barrier:.4f} eV (raw_ebf={convergence_summary.get('raw_ebf', 'N/A')})")

                    # Cached for get_activation_energy's convergence check
                    self._last_max_force = convergence_summary.get('final_max_replica_force', 0.0)
                    self._last_converged = convergence_summary.get('converged', True)

                    self._last_max_force = convergence_summary.get('final_max_replica_force', 0.0)
                    self._last_converged = convergence_summary.get('converged', True)

                    convergence_file = os.path.join(
                        self.results_dir,
                        f"neb_convergence_ep{episode}_step{step}_try{try_num}.json"
                    )
                    with open(convergence_file, 'w') as f:
                        json.dump(convergence_summary, f, indent=2)

                    # Append to master log for offline analysis
                    master_log = os.path.join(self.results_dir, "neb_convergence_master.jsonl")
                    convergence_summary['episode'] = episode
                    convergence_summary['step'] = step
                    convergence_summary['try_num'] = try_num
                    convergence_summary['timestamp'] = time.time()

                    with open(master_log, 'a') as f:
                        f.write(json.dumps(convergence_summary) + '\n')

                except Exception as conv_e:
                    if self.logger:
                        self.logger.log_warning(f"Could not extract convergence info: {conv_e}")
                    else:
                        _LOG.info(f"Warning: Could not extract convergence info: {conv_e}")

            if proc.returncode != 0:
                if self.logger:
                    self.logger.log_warning(f"NEB calculation failed with return code {proc.returncode}")
                    # Log stderr for debugging
                    if stderr:
                        self.logger.log_debug(f"NEB stderr: {stderr.decode('utf-8', errors='ignore')[:500]}")
                else:
                    _LOG.info(f"NEB calculation failed with return code {proc.returncode}")

                # Surface common failure modes from log.lammps (CWD)
                if os.path.exists('log.lammps'):
                    with open('log.lammps', 'r') as f:
                        log_content = f.read()
                    if 'ERROR' in log_content:
                        error_lines = [line for line in log_content.split('\n') if 'ERROR' in line]
                        if self.logger:
                            self.logger.log_error(f"LAMMPS errors: {error_lines[:3]}")
                    # Lost atoms: atoms moved outside the box (common NEB failure)
                    if 'Lost atoms' in log_content:
                        if self.logger:
                            self.logger.log_error("NEB failed due to lost atoms - atoms moved outside box")
                        else:
                            _LOG.info("NEB failed due to lost atoms - atoms moved outside box")
                
                return False

            # Concatenate each replica's last frame into one trajectory file
            if episode is not None:
                try_num = 0 if try_num is None else try_num
                step = 0 if step is None else step

                output_file = os.path.join(self.results_dir, f"dump_neb_ep{episode}_step{step}_try{try_num}.lammpstrj")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                _LOG.info(f"Creating NEB trajectory file: {output_file}")

                with open(output_file, 'w') as outfile:
                    for rep in range(1, self.num_images + 1):
                        dump_file = f"replica-{rep}.dump"
                        if os.path.exists(dump_file):
                            try:
                                with open(dump_file, 'r') as infile:
                                    lines = infile.readlines()

                                    # Last frame starts at the final "ITEM: TIMESTEP"
                                    last_timestep_idx = -1
                                    for i in range(len(lines) - 1, -1, -1):
                                        if lines[i].startswith("ITEM: TIMESTEP"):
                                            last_timestep_idx = i
                                            break

                                    if last_timestep_idx >= 0:
                                        last_frame = lines[last_timestep_idx:]
                                        outfile.writelines(last_frame)
                                    else:
                                        _LOG.debug(f"Warning: No ITEM: TIMESTEP found in {dump_file}")

                            except Exception as e:
                                _LOG.debug(f"Error processing {dump_file}: {e}")

            if proc.returncode != 0:
                _LOG.info(f"NEB calculation failed with return code {proc.returncode}")
                return False

            return True

        except Exception as e:
            _LOG.info(f"Error running NEB: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            for temp_file in ["init", "final", "temp_neb.in"]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass

            # Always leave an output file behind, even on failure
            if episode is not None and step is not None and try_num is not None:
                if output_file is None:
                    output_file = os.path.join(self.results_dir, f"dump_neb_ep{episode}_step{step}_try{try_num}.lammpstrj")

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                if not os.path.exists(output_file):
                    with open(output_file, 'w') as f:
                        f.write(f"# NEB failed or skipped for episode {episode}, step {step}, try {try_num}\n")

    def get_activation_energy(self, min_barrier_threshold=0.05, high_barrier_threshold=2.5, silent=False):
        """
        Activation energy (eV) from the NEB energy profile, or -1.0 if the path is invalid.

        Args:
            min_barrier_threshold (float): below this, any-shape path is accepted (eV)
            high_barrier_threshold (float): above this, path is rejected (eV)
            silent (bool): suppress debug logging
        """
        energy_array = self.read_neb_energy(verbose=not silent)

        if len(energy_array) == 0:
            return -1.0

        # Reject unconverged NEB with extreme residual force
        # (converged max_force <= 0.5; garbage runs reach 100-800)
        _max_force = getattr(self, '_last_max_force', 0.0)
        _converged = getattr(self, '_last_converged', True)
        if not _converged and _max_force > 10.0:
            if not silent:
                _LOG.debug(f"NEB RESULT: Rejected unconverged NEB (max_force={_max_force:.1f} > 10.0)")
            return -1.0

        # energy_array is relative to the initial state, so barrier == max value
        max_barrier = np.max(energy_array)

        # Barrier measured from the lowest point before the peak (initial may not be the minimum)
        max_idx = np.argmax(energy_array)
        if max_idx > 0:
            min_before_max = np.min(energy_array[:max_idx+1])
            max_barrier_from_min = energy_array[max_idx] - min_before_max
        else:
            max_barrier_from_min = max_barrier

        if max_barrier < min_barrier_threshold:
            if not silent:
                _LOG.debug(f"NEB RESULT: Valid path (small barrier) found! Activation energy: {max_barrier:.4f} eV")
            return max_barrier

        if max_barrier > high_barrier_threshold:
            if not silent:
                _LOG.debug(f"NEB RESULT: Invalid path detected (barrier exceeds threshold). Activation energy: {max_barrier:.4f} eV")
            return -1.0

        is_valid_path = self.has_single_local_maximum(energy_array, min_barrier_threshold)

        if is_valid_path:
            if not silent:
                _LOG.debug(f"NEB RESULT: Valid path found! Activation energy: {max_barrier:.4f} eV")
            return max_barrier
        else:
            if not silent:
                _LOG.debug(f"NEB RESULT: Invalid path detected (multiple significant barriers). Activation energy: {max_barrier:.4f} eV")
            return -1.0

    def read_neb_energy(self, verbose=True):
        """
        Read per-image energies (eV, relative to the initial image) from per-replica logs.

        Returns an empty array on failure.
        """
        try:
            num_image = self.num_images

            missing_logs = []
            for i in range(num_image):
                if not os.path.exists(f'log.lammps.{i}'):
                    missing_logs.append(i)

            if missing_logs:
                if self.logger:
                    self.logger.log_warning(f"Missing log files for images: {missing_logs}")
                if len(missing_logs) == num_image:
                    return np.array([])

            energy_array = np.zeros(num_image)
            init_energy = None
            for i in range(num_image):
                try:
                    log_filename = f'log.lammps.{i}'
                    # latin-1 tolerates any binary bytes in the log
                    with open(log_filename, 'r', encoding='latin-1', errors='ignore') as f:
                        found = False
                        for line in f:
                            if "Energy initial, next-to-last, final" in line:
                                next_line = next(f)
                                found = True
                                break
                    if not found:
                        if self.logger:
                            self.logger.log_debug(f"Could not find energy line in {log_filename}")
                        else:
                            _LOG.debug(f"Could not find energy line in {log_filename}")
                        # Energy line absent: leave this image at 0.0
                        if i == 0:
                            init_energy = 0.0
                            energy_array[i] = 0.0
                        else:
                            energy_array[i] = 0.0
                    else:
                        try:
                            energy = float(next_line.split()[-1])
                            if i == 0:
                                init_energy = energy
                            # Shift to initial reference, convert kcal/mol -> eV
                            energy_array[i] = (energy - init_energy) * KCAL_MOL_TO_EV

                        except (ValueError, IndexError) as e:
                            if self.logger:
                                self.logger.log_error(f"Error parsing energy from {log_filename}: {e}")
                            # Empty array signals failure (fail-fast)
                            return np.array([])
                except Exception as e:
                    if self.logger:
                        self.logger.log_debug(f"Error reading log file {log_filename}: {e}")
                    else:
                        _LOG.debug(f"Error reading log file {log_filename}: {e}")
                    # Carry forward previous image's value (0.0 for the first)
                    energy_array[i] = 0.0 if i == 0 else energy_array[i-1]

            self.energy_array = energy_array
            return energy_array
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error in read_neb_energy: {e}")
            else:
                _LOG.info(f"Error in read_neb_energy: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def get_endpoint_energies(self):
        """
        Initial and final state energies (eV), or (None, None) if unavailable.
        """
        try:
            # Prefer the cached energy array (relative to initial = 0.0)
            if self.energy_array is not None and len(self.energy_array) > 0:
                initial_energy = 0.0
                final_energy = self.energy_array[-1]
                return initial_energy, final_energy

            # Fall back to reading the per-replica logs directly
            initial_energy = None
            final_energy = None

            try:
                with open('log.lammps.0', 'r', encoding='latin-1', errors='ignore') as f:
                    for line in f:
                        if "Energy initial, next-to-last, final" in line:
                            next_line = next(f)
                            initial_energy = float(next_line.split()[-1])
                            break
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Error reading initial energy: {e}")

            last_log = f'log.lammps.{self.num_images - 1}'
            try:
                with open(last_log, 'r', encoding='latin-1', errors='ignore') as f:
                    for line in f:
                        if "Energy initial, next-to-last, final" in line:
                            next_line = next(f)
                            final_energy = float(next_line.split()[-1])
                            break
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Error reading final energy: {e}")

            if initial_energy is not None and final_energy is not None:
                # LAMMPS energies are kcal/mol -> eV
                initial_energy_ev = initial_energy / 23.06
                final_energy_ev = final_energy / 23.06
                return initial_energy_ev, final_energy_ev

            return None, None

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error in get_endpoint_energies: {e}")
            return None, None
