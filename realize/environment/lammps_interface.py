"""Wrapper around a reused LAMMPS (ReaxFF) instance."""
import gc
from typing import Optional
from lammps import lammps
from realize.config.paths import get_reaxff_file_path
from realize.utils.file_utils import standardize_file_format


class LAMMPSInterface:
    """LAMMPS operations on a reused instance (units real, ReaxFF/Si-O)."""

    def __init__(self, logger):
        self.logger = logger
        self._lmp_instance: Optional[lammps] = None
        self.reaxff_file = get_reaxff_file_path()

    @staticmethod
    def create_lammps_instance(log_file: str = "lammps_log.txt") -> lammps:
        """Create a LAMMPS instance with screen output redirected to log_file."""
        return lammps(cmdargs=["-screen", log_file, "-log", "none"])

    def get_instance(self) -> lammps:
        """
        Get or create the single reused LAMMPS instance. setup_simulation()
        issues 'clear' for a clean state. Each parallel worker needs its own
        LAMMPSInterface.
        """
        if self._lmp_instance is None:
            self._lmp_instance = self.create_lammps_instance("lammps_main.log")
        return self._lmp_instance

    def setup_simulation(self, coo_file: str):
        """Load structure and set up ReaxFF (units real, boundary p p f, charge qeq)."""
        lmp = self.get_instance()

        lmp.command("clear")
        lmp.command("units real")
        lmp.command("dimension 3")
        lmp.command("boundary p p f")
        lmp.command("atom_style charge")
        lmp.command(f"read_data {coo_file}")
        lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
        lmp.command(f"pair_coeff * * {self.reaxff_file} Si O")
        lmp.command("fix ch_eq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")
        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify delay 10 check yes")

        # Freeze bottom atoms (z < 3.9) to anchor the substrate
        lmp.command("variable zlo equal bound(all,zmin)")
        lmp.command("variable bottom_cutoff equal 3.9")
        lmp.command("region bottom block EDGE EDGE EDGE EDGE 0 3.9")
        lmp.command("group bottom1 region bottom")
        lmp.command("fix freeze bottom1 setforce 0.0 0.0 0.0")
        lmp.command("velocity bottom1 set 0.0 0.0 0.0")
    
    def relax_structure(self, coo_file: str):
        """CG energy minimization; writes res.lammpstrj and the relaxed coo_file."""
        lmp = self.get_instance()

        self.setup_simulation(coo_file)

        lmp.command("compute atompe all pe/atom")
        lmp.command("dump dump1 all custom 1 res.lammpstrj id type x y z fx fy fz q c_atompe")
        lmp.command("dump_modify dump1 sort id append yes")

        lmp.command("min_style cg")
        lmp.command("min_modify line quadratic dmax 0.05")
        lmp.command("minimize 0 0.01 50 50")

        lmp.command("undump dump1")

        lmp.command(f"write_data {coo_file}")

        standardize_file_format(coo_file, target_style='charge')

    def get_potential_energy(self, coo_file: str) -> float:
        """Potential energy of the current structure (run 0)."""
        lmp = self.get_instance()

        self.setup_simulation(coo_file)
        lmp.command("variable pe equal pe")
        lmp.command("run 0")

        pe = lmp.extract_variable("pe", None, 0)
        return pe

    def cleanup(self):
        """Close and release the LAMMPS instance."""
        if self._lmp_instance is not None:
            try:
                self._lmp_instance.close()
            except:
                pass
            self._lmp_instance = None
            gc.collect()

    def __del__(self):
        self.cleanup()
