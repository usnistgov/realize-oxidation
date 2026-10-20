"""RL environment: apply an O2 move, relax, then check dissociation and barriers."""
import os
import numpy as np
import gc
import torch
from lammps import lammps
from typing import Optional, Tuple, Dict, Any
from ase import io
import time 
import signal

from ase import Atom


class MDTimeoutError(Exception):
    """Raised when MD verification exceeds time limit."""
    pass
from realize.environment.structure_manager import StructureManager
from realize.environment.lammps_interface import LAMMPSInterface
from realize.environment.o2_manager import O2Manager
from realize.environment.dissociation_checker import DissociationChecker
from realize.environment.action_processor import ActionProcessor
from realize.utils.logger import UnifiedLogger
from realize.utils.file_utils import standardize_file_format 
from realize.utils.file_utils import write_structure_file
from realize.config.paths import get_temp_dir, get_structure_file_path
from realize.config.paths import get_movie_file_path 

from realize.networks.encoders.embedding_extractor import get_embedding_vector
import logging
_LOG = logging.getLogger(__name__)


def get_symmetric_embedding(lammps_path, checkpoint_path=None,
                           device=None, config=None, use_tensor_product=True):
    """
    Extract the symmetric embedding (used for state construction) for a structure.

    Returns a [1, embedding_dim] tensor.
    """
    embeddings_dict = get_embedding_vector(
        lammps_path=lammps_path,
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
        use_tensor_product=use_tensor_product
    )
    return embeddings_dict['symmetric']

def get_dual_embeddings(lammps_path, checkpoint_path=None,
                       device=None, config=None, use_tensor_product=True):
    """
    Return (asymmetric, symmetric) embeddings; asymmetric is used for axis prediction.
    """
    embeddings_dict = get_embedding_vector(
        lammps_path=lammps_path,
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
        use_tensor_product=use_tensor_product
    )
    return embeddings_dict['asymmetric'], embeddings_dict['symmetric']

class O2LammpsEnv:
    """O2-substrate RL environment: action -> relax -> dissociation/NEB checks."""

    def __init__(self, data_file: Optional[str] = None,
                 coo_file: Optional[str] = None,
                 logger: Optional[UnifiedLogger] = None,
                 embedding_func=None):
        """Wire up the component managers and load the initial structure."""

        self.data_file = data_file or get_structure_file_path()
        if coo_file is None or os.path.dirname(coo_file) == '':
            self.coo_file = os.path.join(get_temp_dir(), "coo.data")
        else:
            self.coo_file = coo_file

        self.logger = logger

        # O2 lifecycle flag. True iff an env-owned O2 is currently in the
        # structure (set by episode_reset after create_initial_O2; cleared by
        # verified dissociation in episode_step or by safely_remove_o2).
        # Replaces fragile geometric has_o2 re-detection in validate_atom_counts.
        self.active_o2 = False

        self.get_embedding = embedding_func or get_symmetric_embedding
        self.get_dual_embeddings = get_dual_embeddings

        self.structure_manager = StructureManager(self.coo_file, logger)
        self.structure_manager.parent_env = self
        self.lammps_interface = LAMMPSInterface(logger)
        self.o2_manager = O2Manager(self.structure_manager, logger)
        self.dissociation_checker = DissociationChecker(self.structure_manager, logger)

        max_trans_distance = 3.0
        if hasattr(self, 'config'):
            max_trans_distance = getattr(self.config, 'max_trans_distance', 3.0)

        self.action_processor = ActionProcessor(
            self.structure_manager,
            self.logger,
            max_trans_distance=max_trans_distance
        )

        self.structure_manager.initialize_from_file(self.data_file)

        self.episode_outcomes = {}
        self.current_episode = 0

        self.state_revision = 0  # monotonic counter for cache invalidation
        self.o2_removed = False
        self.previous_episode_dissociated = False

        self.global_catastrophic_failures = 0
        self.force_next_step = False
        self.neb_failure_tracker = {}  # consecutive NEB failures per step

    def start_lammps(self, lmp=None):
        """Ensure the internal LAMMPS instance exists; ignores any passed lmp."""
        self.lammps_interface.get_instance()
        return True

    def relax_str(self, lmp=None):
        """Relax the structure: internal interface when lmp is None, else use lmp."""
        if lmp is None:
            self.lammps_interface.relax_structure(self.coo_file)
        else:
            lmp.command("compute atompe all pe/atom")
            lmp.command("dump dump1 all custom 1 res.lammpstrj id type x y z fx fy fz q c_atompe")
            lmp.command("dump_modify dump1 sort id append yes")
            lmp.command("min_style cg")
            lmp.command("min_modify line quadratic dmax 0.1")
            lmp.command("minimize 0 0.02 30 30")
            lmp.command("undump dump1")
            lmp.command(f"write_data {self.coo_file}")
            gc.collect()
            standardize_file_format(self.coo_file, target_style='charge')
        
        self.state_revision += 1

    def run_dissociation_verification_md(self):
        """
        Heat/hold/cool MD to confirm dissociation is stable.
        Returns True if O2 stays dissociated; restores pre-MD state otherwise.
        """
        try:
            md_initial_state = os.path.join(get_temp_dir(), "md_initial_state.data")
            self.save_current_state(md_initial_state)
            standardize_file_format(md_initial_state, target_style='charge')

            # 600s alarm guards against LAMMPS deadlock hanging the worker
            try:
                def _md_timeout_handler(signum, frame):
                    raise MDTimeoutError("MD verification timed out after 600s")
                signal.signal(signal.SIGALRM, _md_timeout_handler)
                signal.alarm(600)
            except (ValueError, OSError):
                pass  # not main thread, or signals unsupported

            if self.logger:
                self.logger.log_debug("Starting MD verification for dissociation")

            episode_num = getattr(self, 'current_episode', 0)

            # Pre-MD snapshot for the main movie
            main_movie_file = get_movie_file_path(episode_num + 1)
            pre_md_snapshot = os.path.join(get_temp_dir(), "pre_md_snapshot.lammpstrj")
            self.write_lammps_trajectory(pre_md_snapshot)

            md_log_file = os.path.join(get_temp_dir(), f"lammps_md_ep{episode_num}.log")
            lmp = None
            lmp = LAMMPSInterface.create_lammps_instance(md_log_file)

            # Read structure first to establish the box
            lmp.command("units real")
            lmp.command("dimension 3")
            lmp.command("boundary p p f")
            lmp.command("atom_style charge")
            lmp.command(f"read_data {self.coo_file}")

            # ReaxFF force field and qeq setup
            from realize.config.paths import get_reaxff_file_path
            reaxff_file = get_reaxff_file_path()
            lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
            lmp.command(f"pair_coeff * * {reaxff_file} Si O")
            lmp.command("fix ch_eq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")
            lmp.command("neighbor 2.0 bin")
            lmp.command("neigh_modify delay 10 check yes")

            # Freeze bottom atoms (z < 3.9)
            lmp.command("region bottom block EDGE EDGE EDGE EDGE 0.0 3.9")
            lmp.command("group bottom1 region bottom")
            lmp.command("fix freeze bottom1 setforce 0.0 0.0 0.0")
            lmp.command("velocity bottom1 set 0.0 0.0 0.0")

            md_movie_file = main_movie_file.replace('.lammpstrj', '_md.lammpstrj')

            lmp.command("region rest block EDGE EDGE EDGE EDGE 3.9 EDGE")
            lmp.command("group rest1 region rest")

            lmp.command("compute atompe all pe/atom")
            lmp.command(f"dump md_dump all custom 10 {md_movie_file} id type x y z fx fy fz q c_atompe")
            lmp.command("dump_modify md_dump sort id")

            lmp.command("timestep 2")
            # Quick minimization before heating: prevents freshly-dissociated O
            # from being ejected by a strained local environment
            lmp.command("min_style cg")
            lmp.command("minimize 0 0.5 100 100")
            lmp.command("velocity rest1 create 300.0 2234")

            # Heat 300K -> 1000K
            lmp.command("fix 1 rest1 nvt temp 300 1000 100")
            lmp.command("run 500")
            lmp.command("unfix 1")

            # Hold at 1000K
            lmp.command("fix 1 rest1 nvt temp 1000 1000 100")
            lmp.command("run 1000")
            lmp.command("unfix 1")

            # Cool 1000K -> 300K
            lmp.command("fix 1 rest1 nvt temp 1000 300 100")
            lmp.command("run 500")
            lmp.command("unfix 1")

            lmp.command("min_style cg")
            lmp.command("min_modify line quadratic dmax 0.1")
            lmp.command("minimize 0 0.1 50 50")
            lmp.command("min_style fire")
            lmp.command("min_modify dmax 0.05")
            lmp.command("minimize 0 0.02 50 50")

            lmp.command("undump md_dump")

            lmp.command(f"write_data {self.coo_file}")

            standardize_file_format(self.coo_file, target_style='charge')

            # Check for atom loss during MD (O flying past the f boundary at 1000K)
            post_md_atoms = io.read(self.coo_file, format="lammps-data", atom_style="charge")
            pre_md_atoms = io.read(md_initial_state, format="lammps-data", atom_style="charge")
            atoms_lost = len(pre_md_atoms) - len(post_md_atoms)
            if atoms_lost > 0:
                pre_symbols = pre_md_atoms.get_chemical_symbols()
                post_symbols = post_md_atoms.get_chemical_symbols()
                pre_o = sum(1 for s in pre_symbols if s == 'O')
                post_o = sum(1 for s in post_symbols if s == 'O')
                pre_si = sum(1 for s in pre_symbols if s == 'Si')
                post_si = sum(1 for s in post_symbols if s == 'Si')
                o_lost = pre_o - post_o
                si_lost = pre_si - post_si
                _LOG.info(f"[MD VERIFY] WARNING: {atoms_lost} atom(s) lost during MD "
                      f"(O_lost={o_lost}, Si_lost={si_lost}, "
                      f"pre={len(pre_md_atoms)}, post={len(post_md_atoms)}).")
                if si_lost > 0:
                    # Substrate corrupted -> restore, not dissociated
                    _LOG.info(f"[MD VERIFY] Si lost -> substrate corrupted, restoring.")
                    self.load_state(md_initial_state)
                    return False
                elif o_lost == 2:
                    # Both O ejected past boundary -> not dissociated
                    _LOG.info(f"[MD VERIFY] Both O ejected -> not dissociated, restoring.")
                    self.load_state(md_initial_state)
                    return False
                elif o_lost == 1:
                    # One O incorporated, one ejected -> valid dissociation
                    _LOG.info(f"[MD VERIFY] One O incorporated, one ejected -> valid dissociation.")
                    return True
                else:
                    # Unexpected loss pattern -> restore conservatively
                    _LOG.info(f"[MD VERIFY] Unexpected atom loss pattern -> restoring.")
                    self.load_state(md_initial_state)
                    return False

            # Distance-only dissociation check (no oxidation-state criterion)
            is_still_dissociated = self.check_o2_distance_only(distance_threshold=1.8)

            if self.logger:
                self.logger.log_debug(f"MD verification complete. Still dissociated: {is_still_dissociated}")

            # Detect free O atoms in vacuum (not bonded to any nearby Si)
            if is_still_dissociated:
                post_check = io.read(self.coo_file, format='lammps-data', atom_style='charge')
                post_check.set_pbc((True, True, True))
                p_syms = post_check.get_chemical_symbols()
                p_chgs = post_check.get_initial_charges()
                free_o_idx = []
                for i, (sym, chg) in enumerate(zip(p_syms, p_chgs)):
                    if sym != 'O':
                        continue
                    if chg > -0.2:
                        has_si = any(
                            p_syms[j] == 'Si' and post_check.get_distance(i, j, mic=True) < 2.5
                            for j in range(len(p_syms))
                        )
                        if not has_si:
                            free_o_idx.append(i)
                if len(free_o_idx) >= 2:
                    d = post_check.get_distance(free_o_idx[0], free_o_idx[1], mic=True)
                    if d < 1.8:
                        # Two free O reformed O2 in vacuum -> rollback to pre-episode
                        _LOG.info(f'[MD VERIFY] Scenario 3: free O2 in vacuum (d={d:.3f}Angstrom) -> rollback.')
                        pre_ep = getattr(self, 'pre_episode_substrate_state', None)
                        if pre_ep and os.path.exists(pre_ep):
                            self.load_state(pre_ep)
                        else:
                            self.load_state(md_initial_state)
                        self._scenario3_rollback = True
                        is_still_dissociated = False
                elif len(free_o_idx) == 1:
                    # Single free O in vacuum: delete it, keep dissociation.
                    # Count was +2 for this dissociation but only 1 O incorporated,
                    # so the removal keeps it consistent with actual substrate atoms.
                    _LOG.info(f'[MD VERIFY] Scenario 2: mono-O in vacuum -> removing, adjusting count.')
                    clean = post_check.copy()
                    del clean[free_o_idx[0]]
                    from realize.environment.structure_manager import write_structure_file
                    write_structure_file(clean, self.coo_file, atom_style='charge')
                    standardize_file_format(self.coo_file, target_style='charge')
                    self._scenario2_o_removed = True
                    is_still_dissociated = True

            if not is_still_dissociated:
                # O2 recombined: restore pre-MD state, keep its snapshot in the movie
                if self.logger:
                    self.logger.log_debug("O2 recombined during MD. Restoring pre-MD state.")
                self.load_state(md_initial_state)

                with open(pre_md_snapshot, 'rb') as src:
                    with open(main_movie_file, 'ab') as dst:
                        dst.write(src.read())
            else:
                # Stayed dissociated: append pre- and post-MD snapshots to the movie
                post_md_snapshot = os.path.join(get_temp_dir(), "post_md_snapshot.lammpstrj")
                self.write_lammps_trajectory(post_md_snapshot)

                with open(pre_md_snapshot, 'rb') as src:
                    with open(main_movie_file, 'ab') as dst:
                        dst.write(src.read())
                with open(post_md_snapshot, 'rb') as src:
                    with open(main_movie_file, 'ab') as dst:
                        dst.write(src.read())

                try:
                    os.remove(post_md_snapshot)
                except:
                    pass

            for temp_file in [md_initial_state, pre_md_snapshot]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            return is_still_dissociated

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error during MD verification: {e}")
            else:
                _LOG.info(f"Error during MD verification: {e}")
            # Restore pre-MD state on any failure (including timeout)
            try:
                if os.path.exists(md_initial_state):
                    self.load_state(md_initial_state)
            except Exception:
                pass
            return False
        finally:
            # Cancel MD timeout
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
            except Exception:
                pass
            if lmp is not None:
                try:
                    lmp.close()
                except Exception:
                    pass
            gc.collect()

    def check_o2_distance_only(self, distance_threshold=1.8):
        """Dissociated iff O-O distance > threshold (Angstrom); ignores oxidation state."""
        try:
            if hasattr(self, 'o2_removed') and self.o2_removed:
                return False

            distance, _ = self.o2_manager.check_o2_distance()
            
            if self.logger:
                self.logger.log_debug(f"O-O distance (MD check): {distance:.3f} Angstrom (threshold: {distance_threshold} Angstrom)")
            
            return distance > distance_threshold
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error in check_o2_distance_only: {e}")
            return False

    def create_initial_O2(self, lmp, top_z):
        """Create the initial O2; relaxation runs via the internal LAMMPS interface."""
        success = self.o2_manager.create_initial_o2(
            top_z,
            lammps_interface=self.lammps_interface
        )
        if success:
            self.state_revision += 1
        return success

    def check_top(self):
        """Max z coordinate of substrate atoms (O2 excluded)."""
        atoms = self.structure_manager.get_atoms()
        if atoms is not None:
            positions = atoms.get_positions()
            total_atoms = len(atoms)

            cumulative_dissociated = getattr(self.structure_manager, 'cumulative_dissociated_o_count', 0)
            expected_substrate = self.structure_manager.original_atom_count + cumulative_dissociated

            if total_atoms == expected_substrate + 2:
                # O2 present: exclude the last 2 atoms
                substrate_positions = positions[:-2]
                if self.logger:
                    self.logger.log_debug(f"check_top: Excluding O2, using {len(substrate_positions)} substrate atoms")
            elif total_atoms == expected_substrate:
                # No O2: use all atoms
                substrate_positions = positions
                if self.logger:
                    self.logger.log_debug(f"check_top: No O2, using all {len(substrate_positions)} atoms")
            elif total_atoms > expected_substrate + 2:
                # More than expected: keep only the expected substrate atoms
                substrate_positions = positions[:expected_substrate]
                if self.logger:
                    self.logger.log_warning(
                        f"check_top: Have {total_atoms} atoms, expected {expected_substrate} substrate + 2 O2. "
                        f"Using first {expected_substrate} as substrate."
                    )
            else:
                # Fewer than expected: use what we have
                substrate_positions = positions
                if self.logger:
                    self.logger.log_warning(
                        f"check_top: Only {total_atoms} atoms, expected >= {expected_substrate}. Using all."
                    )

            if len(substrate_positions) > 0:
                return np.max(substrate_positions[:, 2])
            else:
                if self.logger:
                    self.logger.log_error("No substrate atoms found!")
                return 0.0
        return 0.0

    def check_surrounding_o2(self, radius=8.0):
        """
        True if O2 is within radius (Angstrom) of any substrate atom.
        Otherwise remove only the O2 pair and set o2_removed; dissociated O are never removed.
        """
        try:
            atoms = io.read(self.coo_file, format="lammps-data", atom_style="charge")
            atoms.set_pbc((True, True, True))
            atoms.wrap()

            total_atoms = len(atoms)
            symbols = atoms.get_chemical_symbols()

            # Dissociated O atoms must be preserved
            cumulative_dissociated = getattr(self.structure_manager, 'cumulative_dissociated_o_count', 0)
            expected_substrate_atoms = self.structure_manager.original_atom_count + cumulative_dissociated

            if total_atoms <= expected_substrate_atoms:
                _LOG.debug(f"  No O2 molecule present")
                self.o2_removed = True
                return False

            # Expect exactly 2 extra atoms (the O2 pair)
            extra_atoms = total_atoms - expected_substrate_atoms
            if extra_atoms != 2:
                _LOG.info(f"  WARNING: Expected 2 extra atoms for O2, but found {extra_atoms}")
                if extra_atoms < 2:
                    self.o2_removed = True
                    return False

            if len(symbols) >= 2 and symbols[-2:] != ['O', 'O']:
                _LOG.info(f"  WARNING: Last two atoms are not O2: {symbols[-2:]}")
                return True  # not O atoms: don't remove

            # O2 is the last two atoms; substrate is everything before
            o2_indices = [total_atoms - 2, total_atoms - 1]
            substrate_indices = list(range(expected_substrate_atoms))

            positions = atoms.get_positions()
            o2_positions = positions[o2_indices]
            o2_com = np.mean(o2_positions, axis=0)

            # Min distance from O2 center of mass to any substrate atom (append dummy X at COM)
            min_distance = float('inf')
            temp_atoms = atoms.copy()
            temp_atoms.append(Atom('X', position=o2_com))
            com_idx = len(temp_atoms) - 1

            for idx in substrate_indices:
                distance = temp_atoms.get_distance(idx, com_idx, mic=True)
                min_distance = min(min_distance, distance)

            if min_distance < radius:
                return True
            else:
                _LOG.debug(f"  O2 too far from surface (min_dist={min_distance:.2f}Angstrom). Removing O2.")

                # Keep substrate atoms only
                substrate_atoms = atoms[:expected_substrate_atoms].copy()

                if len(substrate_atoms) != expected_substrate_atoms:
                    _LOG.info(f"  ERROR: Substrate count mismatch! Keeping {len(substrate_atoms)}, expected {expected_substrate_atoms}")
                    return True  # wrong count: don't remove

                # Refuse the removal if it would drop Si atoms
                substrate_symbols = substrate_atoms.get_chemical_symbols()
                substrate_si_count = sum(1 for sym in substrate_symbols if sym == 'Si')

                if substrate_si_count != self.structure_manager.original_si_count:
                    _LOG.info(f"  ERROR: Would lose Si atoms! Expected {self.structure_manager.original_si_count}, got {substrate_si_count}")
                    return True

                write_structure_file(substrate_atoms, self.coo_file, atom_style='charge')
                standardize_file_format(self.coo_file, target_style='charge')

                self.o2_removed = True
                _LOG.debug(f"  O2 molecule removed. Remaining atoms: {len(substrate_atoms)}")

                return False

        except Exception as e:
            _LOG.info(f"  ERROR in check_surrounding_o2: {e}")
            import traceback
            traceback.print_exc()
            # Flag for safety, leave structure untouched
            self.o2_removed = True
            return False

    def get_last_snapshot(self):
        """Write the current structure as a single-frame last.lammpstrj in the temp dir."""
        output_file = os.path.join(get_temp_dir(), "last.lammpstrj")

        # Always regenerate fresh so the format is correct after atom-count changes
        try:
            success = self.write_lammps_trajectory(output_file)

            # Header must start with "ITEM: TIMESTEP"; regenerate once if missing
            if success and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line.startswith("ITEM: TIMESTEP"):
                        if self.logger:
                            self.logger.log_warning("Snapshot missing proper header, regenerating")
                        success = self.write_lammps_trajectory(output_file)

            return success
        except Exception as e:
            if self.logger:
                self.logger.log_debug(f"Could not create snapshot: {e}")
            else:
                _LOG.debug(f"Could not create snapshot: {e}")
            return False

    def safely_remove_o2(self):
        """Remove the O2 molecule from the structure."""
        result = self.o2_manager.remove_o2_molecule()
        if result:
            # Mirror the structure-side removal in the lifecycle flag.
            self.active_o2 = False
        return result

    def verify_substrate_integrity(self):
        """True if substrate atom counts match the original."""
        return self.structure_manager.verify_integrity()

    def step(self, trans_action: np.ndarray, o2_config: np.ndarray, 
             checkpoint_path: str, device: torch.device = None,
             oscillation_penalty: float = 0.1) -> Tuple[float, Any, float, bool, bool]:
        """
        Apply an O2 move, relax, and check surroundings/dissociation.

        Returns (distance, embedding, reward, done, valid):
          - distance: O-O bond distance (np.inf if O2 removed)
          - embedding: state embedding (None when invalid or on dissociation)
          - reward: negative for invalid/oscillation
          - done: True if dissociated or O2 removed
          - valid: False if oscillation detected

        Raises ValueError if device is not provided.
        """
        if device is None:
            raise ValueError(
                "device parameter is REQUIRED for env.step(). "
                "Must be torch.device('cuda') or torch.device('cpu'). "
                "No auto-selection allowed."
            )

        self.state_revision += 1

        action_result = self.action_processor.process_action(
            trans_action, o2_config, self.state_revision
        )

        if not action_result['valid']:
            _LOG.debug(f"    Reason: {action_result.get('reason', 'Unknown')}")

        if not action_result['valid']:
            _LOG.debug(f"  Returning invalid step (oscillation penalty={oscillation_penalty})")
            return (
                action_result.get('distance', np.inf),
                None,
                -oscillation_penalty,
                False,
                False
            )

        success = self.o2_manager.apply_o2_transformation(
            action_result['o1_pos'],
            action_result['o2_pos']
        )

        if not success:
            _LOG.debug(f"  Transformation failed - returning penalty")
            return (np.inf, None, -0.1, True, False)

        self.lammps_interface.relax_structure(self.coo_file)
        self.state_revision += 1

        # Is O2 still near the substrate surface?
        o2_near_surface = self.check_surrounding_o2(radius=8.0)

        if not o2_near_surface:
            _LOG.info(f"  O2 too far from substrate (>8Angstrom from nearest atom) - episode ending")
            return (np.inf, None, -1.0, True, True)

        distance, distance_vector = self.o2_manager.check_o2_distance()

        if distance > 40.0:
            _LOG.debug(f"  O2 flew away (distance > 40 Angstrom) - marking as removed")
            self.o2_removed = True
            return (np.inf, None, -1.0, True, True)

        is_dissociated = self.dissociation_checker.check_dissociation(
            self.state_revision
        )

        # On dissociation the episode terminates; no embedding needed
        if is_dissociated:
            return (distance, None, 0.0, True, True)

        # Compute embedding now so downstream novelty never sees None
        try:
            import torch
            embedding = self.get_embedding(self.coo_file, checkpoint_path, device=device)
            # Normalize to a [1, 480] tensor on the target device
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).float().to(device)
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.to(device)
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
            else:
                raise TypeError(f"Embedding must be numpy array or torch tensor, got {type(embedding)}")
            if embedding is None:
                raise RuntimeError("Embedding extractor returned None")
        except Exception as e:
            raise RuntimeError(
                f"Embedding extraction failed: {e}. "
                f"Cannot continue training with invalid embedding. "
                f"Check embedding model and checkpoint."
            )
        return (distance, embedding, 0.0, is_dissociated, True)

    def check_o2_distance(self) -> Tuple[float, np.ndarray]:
        """Return (O-O distance, vector)."""
        return self.o2_manager.check_o2_distance()

    def is_dissociated(self, force_check: bool = False) -> bool:
        """True if O2 is dissociated."""
        return self.dissociation_checker.check_dissociation(
            self.state_revision,
            force_check
        )

    def save_current_state(self, filename: str) -> bool:
        """Write the current structure to filename."""
        return self.structure_manager.save_state(filename)

    def write_lammps_trajectory(self, filepath: str) -> bool:
        """Dump the current structure as a single trajectory frame to filepath."""
        lmp = None
        try:
            traj_log_file = os.path.join(get_temp_dir(), "lammps_trajectory.log")
            lmp = LAMMPSInterface.create_lammps_instance(traj_log_file)
            lmp.command("units real")
            lmp.command("dimension 3")
            lmp.command("boundary p p f")
            lmp.command("atom_style charge")
            lmp.command(f"read_data {self.coo_file}")

            from realize.config.paths import get_reaxff_file_path
            reaxff_file = get_reaxff_file_path()
            lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
            lmp.command(f"pair_coeff * * {reaxff_file} Si O")
            lmp.command("fix ch_eq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")
            lmp.command("neighbor 2.0 bin")
            lmp.command("neigh_modify delay 10 check yes")

            lmp.command("compute atompe all pe/atom")
            lmp.command(f"dump dump1 all custom 1 {filepath} id type x y z fx fy fz q c_atompe")
            lmp.command("dump_modify dump1 sort id")
            lmp.command("run 0")
            lmp.command("undump dump1")
    
            return os.path.exists(filepath)

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to write trajectory: {e}")
            return False
        finally:
            if lmp is not None:
                try:
                    lmp.close()
                except Exception:
                    pass
            gc.collect()

    def load_state(self, filename: str) -> bool:
        """Load a saved state from file."""
        success = self.structure_manager.load_state(filename)
        if success:
            self.state_revision += 1
        return success

    def record_episode_outcome(self, episode: int, dissociated: bool,
                              o2_removed: bool, activation_energies: list = None,
                              path_len: float = None):
        """
        Record an episode outcome for history tracking.

        path_len is the sum of accepted-step translation magnitudes, stored on the
        episode_outcomes line so the analysis parser need not re-sum per-step logs.
        """
        self.episode_outcomes[episode] = {
            'dissociated': dissociated,
            'o2_removed': o2_removed,
            'activation_energies': activation_energies or [],
            'path_len': path_len,
            'timestamp': time.time()
        }

        self.previous_episode_dissociated = dissociated

    def check_dissociation_status(self, force_check: bool = False) -> dict:
        """Return dict with is_dissociated, distance, by_distance, by_oxidation."""
        distance, _ = self.check_o2_distance()

        is_dissociated = self.dissociation_checker.check_dissociation(
            self.state_revision, force_check
        )

        by_distance = distance > self.dissociation_checker.distance_threshold
        by_oxidation = is_dissociated and not by_distance

        return {
            'is_dissociated': is_dissociated,
            'distance': distance,
            'by_distance': by_distance,
            'by_oxidation': by_oxidation
        }

    def close(self):
        """Release LAMMPS resources and clear cached episode data."""
        if hasattr(self, 'lammps_interface'):
            self.lammps_interface.cleanup()
        if hasattr(self, 'action_processor'):
            self.action_processor.reset_history()

        self.episode_outcomes.clear()

        if self.logger:
            self.logger.log_debug("Environment closed and resources cleaned up")
