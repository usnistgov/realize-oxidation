import os

# Base directories. Inputs live in data/ at the repo root; outputs (results/, tmp/)
# are written at the repo root too, keeping the realize/ package free of run artifacts.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../realize
BASE_DIR = os.path.dirname(_PACKAGE_DIR)                                    # repo root
INPUT_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
NEB_RESULTS_DIR = os.path.join(RESULTS_DIR, "neb_results")
TEMP_DIR = os.path.join(BASE_DIR, "tmp")

# Spherical harmonics specific directories
SH_DATA_DIR = os.path.join(RESULTS_DIR, "spherical_harmonics")
SH_VISUALIZATION_DIR = os.path.join(SH_DATA_DIR, "visualizations")
SH_COEFFICIENTS_DIR = os.path.join(SH_DATA_DIR, "coefficients")

SH_SPHERE_VISUALIZATION_PATH = os.path.join(SH_VISUALIZATION_DIR, "sphere_plots")

# Directories are NOT created at import time. Creation is lazy: a directory getter
# creates its own dir on first call; ensure_all_directories() / PathManager create the
# always-written top-level dirs at run start. Importing this module writes nothing.

# Input file paths
STRUCTURE_FILE = os.path.join(INPUT_DIR, "str.dat")
O2_FILE = os.path.join(INPUT_DIR, "O2.dat")
EMBEDDING_CHECKPOINT_PATH = os.path.join(INPUT_DIR, "checkpoint.pth")
NEB_EXECUTABLE = os.path.join(INPUT_DIR, "lmp_neb")
NEB_INPUT_FILE = os.path.join(INPUT_DIR, "neb_lammps.in")
REAXFF_FILE = os.path.join(INPUT_DIR, "41524_2023_1112_MOESM1_ESM.txt")

# Output file paths
PPO_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pth")

# Traditional MoG policy data paths
MOG_DATA_PATH = os.path.join(RESULTS_DIR, "mog_distributions.pkl")

# Spherical harmonics data paths
SH_COEFFICIENTS_PATH = os.path.join(SH_COEFFICIENTS_DIR, "sh_coefficients.pkl")

# Logging paths
LEARNING_CURVE_PATH = os.path.join(LOG_DIR, "learning_curve.dat")
ENTROPY_LOG_PATH = os.path.join(LOG_DIR, "entropy_log.npy")
TRANS_MEANS_LOG_PATH = os.path.join(LOG_DIR, "trans_means_log.npy")
TRANS_WEIGHTS_LOG_PATH = os.path.join(LOG_DIR, "trans_weights_log.npy")

# Novelty and pathway analysis paths
NOVELTY_LOG_PATH = os.path.join(RESULTS_DIR, "novelty_log.json")
PATHWAY_STATS_PATH = os.path.join(RESULTS_DIR, "pathway_stats.json")

# Embedding and equivariant data paths
EMBEDDING_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "embedding_analysis")
EQUIVARIANT_DATA_PATH = os.path.join(EMBEDDING_ANALYSIS_DIR, "equivariant_features.pkl")

# Directory getters lazily create their own directory on first call, so a path is
# guaranteed to exist by the time it is handed out.
def get_input_dir():
    """Get input directory path."""
    return INPUT_DIR

def get_results_dir():
    """Get results directory path (created on demand)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR

def get_checkpoint_dir():
    """Get checkpoint directory path (created on demand)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return CHECKPOINT_DIR

def get_temp_dir():
    """Get temporary directory path (created on demand)."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR

def get_log_dir():
    """Get logs directory path (created on demand)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR

def get_neb_results_dir():
    """Get NEB results directory path (created on demand)."""
    os.makedirs(NEB_RESULTS_DIR, exist_ok=True)
    return NEB_RESULTS_DIR

# Getter functions - Spherical harmonics
def get_sh_data_dir():
    """Get spherical harmonics data directory path (created on demand)."""
    os.makedirs(SH_DATA_DIR, exist_ok=True)
    return SH_DATA_DIR

def get_sh_visualization_dir():
    """Get spherical harmonics visualization directory path (created on demand)."""
    os.makedirs(SH_VISUALIZATION_DIR, exist_ok=True)
    return SH_VISUALIZATION_DIR

def get_sh_coefficients_dir():
    """Get spherical harmonics coefficients directory path (created on demand)."""
    os.makedirs(SH_COEFFICIENTS_DIR, exist_ok=True)
    return SH_COEFFICIENTS_DIR

def get_sh_coefficients_path():
    """Get spherical harmonics coefficients file path."""
    return SH_COEFFICIENTS_PATH

def get_sh_sphere_visualization_path():
    """Get spherical harmonics sphere visualization base path (created on demand)."""
    os.makedirs(SH_SPHERE_VISUALIZATION_PATH, exist_ok=True)
    return SH_SPHERE_VISUALIZATION_PATH

# Getter functions - Input files
def get_ppo_checkpoint_path():
    """Get PPO checkpoint file path."""
    return PPO_CHECKPOINT_PATH

def get_embedding_checkpoint_path():
    """Get embedding model checkpoint file path."""
    return EMBEDDING_CHECKPOINT_PATH

def get_structure_file_path():
    """Get structure file path."""
    return STRUCTURE_FILE

def get_o2_file_path():
    """Get O2 molecule file path."""
    return O2_FILE

def get_neb_executable_path():
    """Get NEB executable path."""
    return NEB_EXECUTABLE

def get_neb_input_file_path():
    """Get NEB input file path."""
    return NEB_INPUT_FILE

def get_reaxff_file_path():
    """Get ReaxFF parameter file path."""
    return REAXFF_FILE

# Getter functions - Output files
def get_mog_data_path():
    """Get MoG distributions data file path."""
    return MOG_DATA_PATH

def get_learning_curve_path():
    """Get learning curve data file path."""
    return LEARNING_CURVE_PATH

def get_entropy_log_path():
    """Get entropy log file path."""
    return ENTROPY_LOG_PATH

def get_trans_means_log_path():
    """Get translation means log file path."""
    return TRANS_MEANS_LOG_PATH

def get_trans_weights_log_path():
    """Get translation weights log file path."""
    return TRANS_WEIGHTS_LOG_PATH

def get_novelty_log_path():
    """Get novelty analysis log file path."""
    return NOVELTY_LOG_PATH

def get_pathway_stats_path():
    """Get pathway statistics file path."""
    return PATHWAY_STATS_PATH

# Getter functions - Embedding analysis
def get_embedding_analysis_dir():
    """Get embedding analysis directory path (created on demand)."""
    os.makedirs(EMBEDDING_ANALYSIS_DIR, exist_ok=True)
    return EMBEDDING_ANALYSIS_DIR

def get_equivariant_data_path():
    """Get equivariant features data file path."""
    return EQUIVARIANT_DATA_PATH

# Dynamic path generators
def get_movie_file_path(episode):
    """Returns the path for a movie file for a specific episode."""
    return os.path.join(RESULTS_DIR, f"movie{episode}.lammpstrj")

def get_episode_sh_data_path(episode):
    """Returns the path for spherical harmonics data for a specific episode."""
    return os.path.join(SH_DATA_DIR, f"episode_{episode}_sh_data.json")

def get_sphere_plot_path(episode, step=None):
    """Returns the path for a sphere plot file for a specific episode and optional step."""
    if step is not None:
        filename = f"sphere_ep{episode}_step{step}.png"
    else:
        filename = f"sphere_ep{episode}.png"
    return os.path.join(SH_SPHERE_VISUALIZATION_PATH, filename)

def get_episode_embedding_path(episode):
    """Returns the path for embedding analysis data for a specific episode."""
    return os.path.join(EMBEDDING_ANALYSIS_DIR, f"episode_{episode}_embeddings.pkl")

def get_checkpoint_backup_path(timestamp):
    """Returns the path for a backup checkpoint file."""
    return os.path.join(CHECKPOINT_DIR, f"ppo_checkpoint_backup_{timestamp}.pth")

def get_neb_state_path(episode, step, try_num=None):
    """Returns the path for a NEB state file."""
    if try_num is not None:
        filename = f"state_ep{episode}_step{step}_try{try_num}.data"
    else:
        filename = f"state_ep{episode}_step{step}.data"
    return os.path.join(NEB_RESULTS_DIR, filename)

def get_episode_trajectory_path(episode):
    """Returns the path for episode trajectory data."""
    return os.path.join(RESULTS_DIR, f"trajectory_ep{episode}.json")

# Utility functions for path management
def ensure_directory_exists(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

def ensure_all_directories():
    """Ensure the always-written top-level directories exist.

    Only the directories the main process reliably writes into are created up
    front: results/, logs/, checkpoints/ and the temp scratch dir. The optional
    output trees (neb_results/, spherical_harmonics/, embedding_analysis/) are
    created on demand by their getters at the point something is actually
    written, so a run never leaves empty placeholder directories behind.
    """
    directories = [RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR, TEMP_DIR]
    for directory in directories:
        ensure_directory_exists(directory)

def get_policy_specific_data_path(policy_type, filename):
    """Get policy-specific data file path."""
    if policy_type == "spherical":
        return os.path.join(SH_DATA_DIR, filename)
    elif policy_type == "mog":
        return os.path.join(RESULTS_DIR, filename)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def cleanup_old_files(directory, pattern, keep_count=10):
    """Clean up old files in a directory, keeping only the most recent ones."""
    import glob
    import time
    
    if not os.path.exists(directory):
        return
    
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) <= keep_count:
        return

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for file_path in files[keep_count:]:
        try:
            os.remove(file_path)
        except OSError:
            pass

def get_latest_checkpoint():
    """Get the path to the most recent checkpoint file."""
    import glob
    
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_checkpoint*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def create_experiment_subdirectory(experiment_name):
    """Create a subdirectory for a specific experiment."""
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    
    # Create subdirectories for this experiment
    subdirs = [
        os.path.join(experiment_dir, "checkpoints"),
        os.path.join(experiment_dir, "logs"),
        os.path.join(experiment_dir, "neb_results"),
        os.path.join(experiment_dir, "spherical_harmonics"),
        os.path.join(experiment_dir, "embedding_analysis")
    ]
    
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)
    
    return experiment_dir

# Path validation
def validate_input_files():
    """Validate that all required input files exist."""
    required_files = [
        (STRUCTURE_FILE, "Structure file"),
        (O2_FILE, "O2 molecule file"),
        (EMBEDDING_CHECKPOINT_PATH, "Embedding checkpoint")
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{description}: {file_path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required input files:\n" + "\n".join(missing_files)
        )
    
    return True

def get_disk_usage_summary():
    """Get disk usage summary for all project directories."""
    import shutil
    
    directories = [
        ("Results", RESULTS_DIR),
        ("Temp", TEMP_DIR),
        ("Spherical Harmonics", SH_DATA_DIR),
        ("NEB Results", NEB_RESULTS_DIR),
        ("Logs", LOG_DIR)
    ]
    
    summary = {}
    for name, path in directories:
        if os.path.exists(path):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(path)
                for filename in filenames
            )
            summary[name] = total_size / (1024 * 1024)  # MB
        else:
            summary[name] = 0

    return summary


class PathManager:
    """Structured access to project directories with lazy creation.

    - Nothing is created at import; setup() (or any directory getter) creates on demand.
    - work_dir gives each SLURM array task its own scratch directory under tmp/, so
      parallel tasks do not collide on transient LAMMPS/NEB scratch files (scratch has
      no effect on numerics). task_id <= 0 (single run / unset SLURM_ARRAY_TASK_ID)
      maps to tmp/ itself.
    """

    def __init__(self, task_id: int = 0):
        self.task_id = int(task_id)

    @classmethod
    def from_env(cls) -> "PathManager":
        """Build from SLURM_ARRAY_TASK_ID (defaults to 0 when unset)."""
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        task_id = int(raw) if raw not in (None, "") else 0
        return cls(task_id=task_id)

    # Output directories (lazily created via the getters).
    @property
    def input_dir(self) -> str:
        return get_input_dir()

    @property
    def results_dir(self) -> str:
        return get_results_dir()

    @property
    def checkpoint_dir(self) -> str:
        return get_checkpoint_dir()

    @property
    def log_dir(self) -> str:
        return get_log_dir()

    @property
    def neb_results_dir(self) -> str:
        return get_neb_results_dir()

    @property
    def temp_dir(self) -> str:
        return get_temp_dir()

    @property
    def work_dir(self) -> str:
        """Per-task scratch directory (chdir target). tmp/ for task 0, else tmp/task_<id>."""
        if self.task_id <= 0:
            base = get_temp_dir()
        else:
            base = os.path.join(TEMP_DIR, f"task_{self.task_id}")
            os.makedirs(base, exist_ok=True)
        return base

    def setup(self) -> "PathManager":
        """Create the always-written top-level dirs and the per-task scratch dir up front."""
        ensure_all_directories()
        _ = self.work_dir  # materialize the per-task scratch dir
        return self
