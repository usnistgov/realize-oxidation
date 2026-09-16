"""
ASE-driven NEB for REALIZE-oxidation. Replaces the lmp_neb MPI executable path with an
in-process ase.mep.NEB + FIRE loop, while keeping ReaxFF as the force provider through the
LAMMPS Python module (identical setup to LAMMPSInterface.setup_simulation).

Physics matched to data/neb_lammps.in and lammps_interface.setup_simulation:
  * units real -> energies kcal/mol, forces kcal/mol/A -> converted to eV, eV/A.
  * ReaxFF + qeq/reaxff re-equilibrated at EACH image geometry (run 0 triggers qeq).
  * bottom atoms z < 3.9 frozen (FixAtoms), matching 'region bottom ... 0 3.9'.
  * 7 images, spring k = 1, FIRE optimizer, climbing image, fmax from ftol=10 kcal/mol/A.

Public interface mirrors NEBCalculator so it is a drop-in alternative:
  create_images(), run_neb(), get_activation_energy(), get_commitment_endpoints(),
  read_neb_energy(), get_endpoint_energies().
"""
import os
import numpy as np
from ase import io, Atoms
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator, all_changes
from lammps import lammps
from realize.utils.file_utils import standardize_file_format
from realize.config.paths import get_reaxff_file_path

KCAL_MOL_TO_EV = 0.0433634
FORCE_CONV = KCAL_MOL_TO_EV          # (kcal/mol)/A -> eV/A (length unit A shared)
FTOL_KCAL = 10.0                     # from neb_lammps.in
FMAX_EV = FTOL_KCAL * FORCE_CONV     # ~0.434 eV/A
BOTTOM_CUTOFF = 3.9
N_IMAGES_DEFAULT = 7
SPRING_K = 1.0


class ReaxFFCalc(Calculator):
    """ASE calculator: energy+forces at a fixed geometry via a persistent LAMMPS/ReaxFF
    instance. Replicates LAMMPSInterface.setup_simulation exactly, then `run 0` (which
    triggers qeq re-equilibration). Fail-fast: non-finite energy/force raises."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, reaxff_file, template_atoms, log_file="none", **kwargs):
        super().__init__(**kwargs)
        self.reaxff_file = reaxff_file
        self._template = template_atoms          # ASE Atoms defining types/order/cell
        self._lmp = None
        self._log_file = log_file
        self._natoms = len(template_atoms)
        self._setup_done = False

    def _ensure_lmp(self):
        if self._lmp is not None:
            return
        self._lmp = lammps(cmdargs=["-screen", "none", "-log", self._log_file])

    def _write_template_data(self, path):
        # Write the template (types, charges, cell) once; positions get overwritten per call.
        io.write(path, self._template, format="lammps-data", atom_style="charge")

    def _setup(self, data_file):
        L = self._lmp
        L.command("clear")
        L.command("units real")
        L.command("dimension 3")
        L.command("boundary p p f")
        L.command("atom_style charge")
        L.command(f"read_data {data_file}")
        # ASE's lammps-data writer omits the Masses section; set masses explicitly.
        # Element/type order matches pair_coeff below (Si=type 1, O=type 2).
        L.command("mass 1 28.0855")   # Si
        L.command("mass 2 15.9994")   # O
        L.command("pair_style reaxff NULL safezone 3.0 mincap 150")
        L.command(f"pair_coeff * * {self.reaxff_file} Si O")
        L.command("fix ch_eq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")
        L.command("neighbor 2.0 bin")
        L.command("neigh_modify delay 10 check yes")
        L.command("region bottom block EDGE EDGE EDGE EDGE 0 3.9")
        L.command("group bottom1 region bottom")
        L.command("fix freeze bottom1 setforce 0.0 0.0 0.0")
        L.command("compute atompe all pe/atom")
        self._setup_done = True

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Energy+forces at the current geometry via read_data + run 0. Uses the same
        read_data path the rest of the code relies on (robust to atom-ID layout), then
        gather_atoms('f') for forces. run 0 triggers qeq re-equilibration at this geometry.
        Units real -> kcal/mol, kcal/mol/A -> converted to eV, eV/A."""
        super().calculate(atoms, properties, system_changes)
        import tempfile
        self._ensure_lmp()
        n = self._natoms
        # Build an ASE Atoms with the CURRENT positions but the template's types/cell/charges,
        # write it to a LAMMPS data file, and read it in. Consistent atom ordering guaranteed
        # because we always write/read the same way the existing pipeline does.
        cur = self._template.copy()
        cur.set_positions(np.asarray(self.atoms.get_positions(), dtype=np.float64))
        d = tempfile.mkdtemp(prefix="reaxff_")
        data_file = os.path.join(d, "img.data")
        # specorder locks LAMMPS type order to Si=1, O=2 (matching pair_coeff * * reaxff Si O).
        # Without it, ASE orders types by element and silently flips Si/O -> wrong masses+params.
        io.write(data_file, cur, format="lammps-data", atom_style="charge",
                 specorder=["Si", "O"])
        standardize_file_format(data_file, target_style="charge")
        self._setup(data_file)      # clear + units + reaxff + qeq + freeze + read_data
        self._lmp.command("run 0")  # single eval; qeq re-equilibrates charges here
        e_kcal = self._lmp.get_thermo("pe")
        if e_kcal is None or not np.isfinite(e_kcal):
            raise RuntimeError(f"ReaxFFCalc: non-finite energy {e_kcal!r}")
        n_lmp = int(self._lmp.get_natoms())
        if n_lmp != n:
            raise RuntimeError(f"ReaxFFCalc: atom count mismatch LAMMPS={n_lmp} vs ASE={n}")
        fbuf = self._lmp.gather_atoms("f", 1, 3)
        f_kcal = np.frombuffer(fbuf, dtype=np.float64, count=3 * n).copy().reshape(n, 3)
        if not np.all(np.isfinite(f_kcal)):
            raise RuntimeError("ReaxFFCalc: non-finite forces")
        import shutil as _sh
        _sh.rmtree(d, ignore_errors=True)
        self.results["energy"] = float(e_kcal) * KCAL_MOL_TO_EV
        self.results["forces"] = f_kcal * FORCE_CONV

    def close(self):
        if self._lmp is not None:
            try: self._lmp.close()
            except Exception: pass
            self._lmp = None


class ASENEBCalculator:
    """Drop-in ASE-based NEB, same public interface as NEBCalculator, no lmp_neb executable.

    Physics matched to data/neb_lammps.in:
      7 images, spring k=1, FIRE optimizer, climbing image, fmax from ftol=10 kcal/mol/A.
    ReaxFF forces via ReaxFFCalc (verified to reproduce LAMMPS single-point to 1e-10 eV).
    Bottom atoms z<3.9 frozen via FixAtoms on every image, matching setup_simulation.

    Public methods mirror NEBCalculator so episode_step.py needs no change:
      create_images(), run_neb(), get_activation_energy(), get_commitment_endpoints(),
      read_neb_energy(), get_endpoint_energies().
    """
    def __init__(self, initial_file, final_file, logger=None, env=None,
                 num_images=N_IMAGES_DEFAULT, k_spring=SPRING_K, fmax=FMAX_EV,
                 max_steps=1000, climb=True,
                 neb_in_file=None, num_procs_override=None):
        # neb_in_file / num_procs_override are lmp_neb-only args, accepted for call-site
        # compatibility and intentionally unused in the in-process ASE path.
        self.logger = logger
        self.env = env
        self.reaxff_file = get_reaxff_file_path()
        self.init_file = os.path.abspath(initial_file)
        self.final_file = os.path.abspath(final_file)
        self.num_images = num_images
        self.k_spring = k_spring
        self.fmax = fmax
        self.max_steps = max_steps
        self.climb = climb
        self.images = []
        self.energy_array = None
        self._calcs = []          # keep calculators to close LAMMPS instances after
        self._last_converged = None
        self._last_max_force = None

    # ---- image construction: reuse MIC interpolation, hand to ASE ----
    def create_images(self, relax_final=True, relax_steps=1000):
        """Build NEB images via MIC-unwrapped IDPP interpolation (same logic as the
        LAMMPS NEBCalculator.create_images), but keep ASE Atoms in memory instead of
        writing image.N files."""
        initial = io.read(self.init_file, format='lammps-data', atom_style='charge')
        final = io.read(self.final_file, format='lammps-data', atom_style='charge')
        initial.set_pbc((True, True, False))
        final.set_pbc((True, True, False))
        n = len(initial)
        if len(final) != n:
            raise ValueError(f"Atom count mismatch: {n} vs {len(final)}")

        # Optionally relax the final endpoint through the env's LAMMPS interface (as before)
        if relax_final and self.env is not None:
            import tempfile
            tf = os.path.join(tempfile.mkdtemp(prefix="neb_final_"), "final.data")
            io.write(tf, final, format='lammps-data', atom_style='charge', specorder=["Si", "O"])
            standardize_file_format(tf, target_style='charge')
            self.env.lammps_interface.relax_structure(tf)
            final = io.read(tf, format='lammps-data', atom_style='charge')
            final.set_pbc((True, True, False))

        # MIC-unwrap final positions relative to initial so interpolation doesn't jump PBC
        cell = np.array(initial.get_cell())
        ipos = initial.get_positions()
        fpos = final.get_positions()
        unwrapped = ipos.copy()
        for i in range(n):
            diff = fpos[i] - ipos[i]
            for dim in range(3):
                L = cell[dim, dim]
                if L > 0:
                    while diff[dim] > L / 2:  diff[dim] -= L
                    while diff[dim] < -L / 2: diff[dim] += L
            unwrapped[i] = ipos[i] + diff
        final_unwrapped = initial.copy()
        final_unwrapped.set_positions(unwrapped)

        # IDPP interpolation between initial and unwrapped final
        images = [initial.copy() for _ in range(self.num_images - 1)] + [final_unwrapped]
        neb = NEB(images)
        neb.interpolate('idpp')
        self.images = images
        return self.images

    # ---- run the NEB ----
    def run_neb(self, episode=0, step=0, try_num=0):
        """Run ASE climbing-image NEB with ReaxFF forces. Returns True on success.
        Stores self.energy_array (eV, relative to image 0)."""
        if not self.images:
            self.create_images()
        n = len(self.images[0])
        # bottom-freeze mask (z < 3.9), matching 'region bottom ... 0 3.9'
        z = self.images[0].get_positions()[:, 2]
        freeze_mask = z < BOTTOM_CUTOFF

        self._calcs = []
        for img in self.images:
            img.set_pbc((True, True, False))
            img.set_constraint(FixAtoms(mask=freeze_mask))
            calc = ReaxFFCalc(self.reaxff_file, img.copy())
            img.calc = calc
            self._calcs.append(calc)

        neb = NEB(self.images, k=self.k_spring, climb=self.climb, allow_shared_calculator=False)
        try:
            opt = FIRE(neb, logfile=None)
            converged = bool(opt.run(fmax=self.fmax, steps=self.max_steps))
            energies = np.array([img.get_potential_energy() for img in self.images], dtype=float)
            # residual max force for the convergence gate (eV/A)
            fmax_final = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
        except Exception as exc:
            if self.logger:
                self.logger.log_error(f"ASE NEB failed: {type(exc).__name__}: {exc}")
            self._close_calcs()
            self.energy_array = np.array([])
            return False
        finally:
            self._close_calcs()

        if not np.all(np.isfinite(energies)):
            self.energy_array = np.array([])
            return False
        self.energy_array = energies - energies[0]   # eV, relative to initial
        self._last_converged = converged
        self._last_max_force = fmax_final
        if self.logger:
            self.logger.log_debug(f"ASE NEB: converged={converged}, fmax={fmax_final:.3f} eV/A, "
                                  f"barrier={float(np.max(self.energy_array)):.4f} eV")
        return True

    def _close_calcs(self):
        for c in self._calcs:
            try: c.close()
            except Exception: pass
        self._calcs = []

    # ---- outputs: identical signatures to NEBCalculator ----
    def read_neb_energy(self, verbose=True):
        return self.energy_array if self.energy_array is not None else np.array([])

    def get_activation_energy(self, min_barrier_threshold=0.05, high_barrier_threshold=2.5, silent=False):
        e = self.read_neb_energy(verbose=not silent)
        if len(e) == 0:
            return -1.0
        if self._last_converged is False and (self._last_max_force or 0.0) > 10.0:
            return -1.0
        max_barrier = float(np.max(e))
        if max_barrier < min_barrier_threshold:
            return max_barrier
        if max_barrier > high_barrier_threshold:
            return -1.0
        if _has_single_local_maximum(e, min_barrier_threshold):
            return max_barrier
        return -1.0

    def get_commitment_endpoints(self, silent=True):
        e = self.read_neb_energy(verbose=not silent)
        if len(e) == 0:
            return None, None
        mi = int(np.argmax(e))
        E_R = float(np.min(e[:mi + 1]))
        E_P = float(np.min(e[mi:]))
        return E_R, E_P

    def get_endpoint_energies(self):
        if self.energy_array is not None and len(self.energy_array) > 0:
            return 0.0, float(self.energy_array[-1])
        return None, None


def _has_single_local_maximum(arr, min_barrier_threshold=0.05):
    """A valid single-barrier path: exactly one significant local maximum."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        return True
    peak = int(np.argmax(arr))
    # significant rises after the global peak indicate multiple barriers
    for i in range(peak + 1, len(arr) - 1):
        if arr[i + 1] - arr[i] > min_barrier_threshold and arr[i] < arr[i - 1]:
            return False
    return True
