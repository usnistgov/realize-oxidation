import os
import time
import torch
import json
import shutil
import pickle
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
_LOG = logging.getLogger(__name__)

# torch.load(weights_only=...) is available from torch 2.0.
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
SUPPORTS_WEIGHTS_ONLY = TORCH_VERSION >= (2, 0)


class SharedModelManager:
    """Shared model access for Distributed Collective PPO training.

    All workers start from identical weights (initialize_or_sync), collect
    episodes in parallel into a shared file-locked buffer, and when the buffer
    reaches eps_per_update the triggering worker runs the update; others then
    sync to the new weights by version check. Keeps updates on-policy while
    parallelizing collection across workers.
    """

    def __init__(self, shared_dir: str = "../../shared_net", worker_id: Optional[str] = None,
                 policy_type: str = "spherical", lock_timeout: Optional[float] = None,
                 eps_per_update: int = 32):
        """Initialize the shared model manager.

        worker_id defaults to SLURM_JOB_ID (else local_<pid>). lock_timeout
        defaults to env SHARED_MODEL_LOCK_TIMEOUT or 60s. eps_per_update is the
        buffer size that triggers an update. policy_type is "spherical" or "mog".
        """
        valid_policies = {"spherical", "mog"}
        if policy_type not in valid_policies:
            raise ValueError(f"Invalid policy_type '{policy_type}'. Must be one of {valid_policies}")

        if worker_id is None:
            self.worker_id = os.environ.get('SLURM_JOB_ID', f'local_{os.getpid()}')
        else:
            self.worker_id = worker_id

        self.shared_dir = os.path.abspath(shared_dir)
        self.policy_type = policy_type
        self.lock_timeout = lock_timeout or float(os.environ.get('SHARED_MODEL_LOCK_TIMEOUT', '60.0'))
        self.eps_per_update = eps_per_update

        # Local version, compared against the shared version to detect syncs.
        self.local_model_version = 0

        # Checkpoint holds weights + optimizer state; paths differ per policy.
        if policy_type == "spherical":
            self.checkpoint_path = os.path.join(self.shared_dir, "ppo_spherical_checkpoint.pth")
            self.sh_data_path = os.path.join(self.shared_dir, "shared_sh_data.json")
            self.embedding_stats_path = os.path.join(self.shared_dir, "embedding_stats.json")
        else:
            self.checkpoint_path = os.path.join(self.shared_dir, "ppo_checkpoint.pth")
            self.mog_data_path = os.path.join(self.shared_dir, "shared_mog_data.pkl")

        # Pickle list of episode dicts collected since the last update.
        self.trajectory_buffer_path = os.path.join(self.shared_dir, "trajectory_buffer.pkl")

        # Lightweight JSON for version checks without loading the full checkpoint.
        self.model_metadata_path = os.path.join(self.shared_dir, "model_metadata.json")

        self.lock_path = os.path.join(self.shared_dir, "model.lock")
        self.stats_path = os.path.join(self.shared_dir, "training_stats.json")
        self.log_path = os.path.join(self.shared_dir, "shared_model_log.txt")
        self.novelty_stats_path = os.path.join(self.shared_dir, "novelty_stats.json")

        os.makedirs(self.shared_dir, exist_ok=True)

        if shared_dir == "../../shared_net":
            self._log("Using default relative shared directory. Consider using an absolute path.", "WARNING")

        self._log(f"SharedModelManager initialized for DISTRIBUTED COLLECTIVE PPO")
        self._log(f"  Policy type: {policy_type.upper()}")
        self._log(f"  Episodes per update: {eps_per_update}")
        self._log(f"  Shared directory: {self.shared_dir}")
        self._log(f"  Checkpoint path: {self.checkpoint_path}")

    def _log(self, message: str, level: str = "INFO"):
        """Append a timestamped line to the shared log, rotating at 10 MB."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] [Worker {self.worker_id}] {message}\n"

        try:
            max_size = 10 * 1024 * 1024  # 10 MB
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > max_size:
                backup_path = f"{self.log_path}.1"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(self.log_path, backup_path)

            with open(self.log_path, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            # Logging must never break training.
            _LOG.warning(f"Failed to write to log: {e}")

    @contextmanager
    def file_lock(self, timeout: Optional[float] = None, stale_lock_timeout: float = 300.0):
        """Acquire an exclusive file lock, breaking locks older than
        stale_lock_timeout seconds. Waits up to timeout (default self.lock_timeout).
        """
        timeout = timeout or self.lock_timeout
        lock_acquired = False
        lock_file = None
        start_time = time.time()

        self._log(f"Attempting to acquire lock (timeout: {timeout}s)")

        try:
            while not lock_acquired and (time.time() - start_time) < timeout:
                try:
                    lock_file = open(self.lock_path, 'x')
                    lock_info = {
                        'worker_id': self.worker_id,
                        'pid': os.getpid(),
                        'timestamp': time.time(),
                        'operation': 'general'
                    }
                    json.dump(lock_info, lock_file)
                    lock_file.flush()
                    lock_acquired = True
                    self._log(f"Lock acquired after {time.time() - start_time:.2f}s")

                except FileExistsError:
                    try:
                        lock_age = time.time() - os.path.getmtime(self.lock_path)
                        if lock_age > stale_lock_timeout:
                            try:
                                with open(self.lock_path, 'r') as f:
                                    lock_info = json.load(f)
                                old_worker = lock_info.get('worker_id', 'unknown')
                                self._log(f"Breaking stale lock from worker {old_worker} "
                                        f"(age: {lock_age:.1f}s)", "WARNING")
                            except:
                                self._log(f"Breaking stale lock (age: {lock_age:.1f}s)", "WARNING")

                            os.remove(self.lock_path)
                            continue
                    except:
                        pass

                    time.sleep(0.5)

            if not lock_acquired:
                self._log(f"Failed to acquire lock after {timeout}s", "ERROR")
                raise TimeoutError(f"Could not acquire lock after {timeout}s")

            yield

        finally:
            if lock_file and lock_acquired:
                lock_file.close()
                try:
                    os.remove(self.lock_path)
                    self._log("Lock released")
                except:
                    pass

    # Core distributed-collective-PPO methods.

    def initialize_or_sync(self, policy_state_dict: Optional[Dict] = None,
                          optimizer_state_dict: Optional[Dict] = None,
                          device: str = 'cpu') -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """Create the shared model or sync to an existing one. Call once at startup.

        Returns (version, policy_state_dict, optimizer_state_dict): the first
        worker gets (0, None, None) and keeps its own weights; a joining worker
        gets the shared version and loaded state dicts. policy/optimizer args are
        used only when this worker creates the shared model.
        """
        self._log("=" * 60)
        self._log("INITIALIZE_OR_SYNC: Starting distributed initialization")
        self._log("=" * 60)

        with self.file_lock(timeout=120.0):  # initialization may load a large checkpoint
            if os.path.exists(self.checkpoint_path) and os.path.exists(self.model_metadata_path):
                # Join existing training.
                self._log("Found existing shared model - joining as worker")

                with open(self.model_metadata_path, 'r') as f:
                    metadata = json.load(f)

                current_version = metadata.get('model_version', 0)
                buffer_size = metadata.get('buffer_size', 0)

                self._log(f"  Current model version: {current_version}")
                self._log(f"  Current buffer size: {buffer_size}/{self.eps_per_update}")

                if SUPPORTS_WEIGHTS_ONLY:
                    checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                else:
                    checkpoint = torch.load(self.checkpoint_path, map_location=device)
                
                self.local_model_version = current_version
                
                self._log(f"Successfully synchronized to shared model v{current_version}")
                
                return (
                    current_version,
                    checkpoint.get('policy_state_dict'),
                    checkpoint.get('optimizer_state_dict')
                )
            
            else:
                # First worker: create the shared model.
                self._log("No existing shared model - initializing as MASTER")

                if policy_state_dict is None:
                    raise ValueError(
                        "First worker must provide policy_state_dict for initialization. "
                        "This worker appears to be first but no initial weights were provided."
                    )

                initial_checkpoint = {
                    'schema_version': 3,
                    'policy_state_dict': policy_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                    'model_version': 0,
                    'global_episodes': 0,
                    'global_updates': 0,
                    'all_rewards': [],
                    'creator_worker': self.worker_id,
                    'timestamp': time.time(),
                    'policy_type': self.policy_type
                }
                
                torch.save(initial_checkpoint, self.checkpoint_path)
                self._log(f"Saved initial checkpoint to {self.checkpoint_path}")

                initial_metadata = {
                    'model_version': 0,
                    'buffer_size': 0,
                    'global_episodes': 0,
                    'global_updates': 0,
                    'last_update_worker': self.worker_id,
                    'last_update_time': time.time(),
                    'eps_per_update': self.eps_per_update,
                    'policy_type': self.policy_type
                }
                
                with open(self.model_metadata_path, 'w') as f:
                    json.dump(initial_metadata, f, indent=2)
                self._log(f"Saved initial metadata to {self.model_metadata_path}")

                with open(self.trajectory_buffer_path, 'wb') as f:
                    pickle.dump([], f)
                self._log(f"Created empty trajectory buffer at {self.trajectory_buffer_path}")

                self.local_model_version = 0

                self._log("MASTER initialization complete - other workers can now join")

                # None state dicts: the first worker keeps its own weights.
                return (0, None, None)

    def submit_episode(self, trajectory_data: Dict[str, Any]) -> Tuple[bool, Optional[List[Dict]], int]:
        """Append a completed episode to the shared buffer under lock.

        Returns (should_update, accumulated_data, buffer_size). When the buffer
        reaches eps_per_update, should_update is True and accumulated_data holds
        every buffered trajectory; otherwise accumulated_data is None.

        IMPORTANT: when should_update is True the lock is STILL HELD; the caller
        must call finish_update_and_clear_buffer() after running the update.
        """
        self._log(f"Submitting episode (reward: {trajectory_data.get('episode_reward', 'N/A'):.4f})")

        if trajectory_data.get('empty', False):
            self._log(f"Skipping empty trajectory (no valid steps)")
            try:
                metadata = self._load_metadata()
                return False, None, metadata.get('buffer_size', 0)
            except:
                return False, None, 0

        required_fields = ['states', 'actions', 'log_probs', 'rewards', 'state_values', 'is_terminals']
        missing_fields = [f for f in required_fields if f not in trajectory_data]
        if missing_fields:
            raise ValueError(f"trajectory_data missing required fields: {missing_fields}")

        trajectory_data['worker_id'] = self.worker_id
        trajectory_data['submit_time'] = time.time()
        trajectory_data['collected_with_version'] = trajectory_data.get('model_version', self.local_model_version)

        with self.file_lock():
            metadata = self._load_metadata()
            current_version = metadata.get('model_version', 0)

            if metadata.get('update_in_progress', False):
                update_worker = metadata.get('update_worker', 'unknown')
                update_start = metadata.get('update_start_time', 0)
                update_age = time.time() - update_start if update_start else 0

                STALE_UPDATE_TIMEOUT = 300.0  # 5 min; assume the worker crashed

                if update_age > STALE_UPDATE_TIMEOUT:
                    self._log(f"Clearing stale update from worker {update_worker} "
                             f"(age: {update_age:.1f}s)", "WARNING")
                    metadata['update_in_progress'] = False
                    self._save_metadata(metadata)
                else:
                    # An update is running: hold this episode, do not trigger another.
                    self._log(f"Update in progress by worker {update_worker} "
                             f"(age: {update_age:.1f}s) - holding episode")
                    buffer_size = metadata.get('buffer_size', 0)
                    return (False, None, buffer_size)

            # Drop episodes more than one version behind; warn at exactly one.
            collected_version = trajectory_data.get('collected_with_version', current_version)
            version_drift = current_version - collected_version

            if version_drift > 1:
                self._log(f"DISCARDING stale episode (collected with v{collected_version}, "
                         f"current is v{current_version})", "WARNING")
                return (False, None, metadata.get('buffer_size', 0))

            if version_drift == 1:
                self._log(f"Accepting slightly stale episode (v{collected_version} vs v{current_version})",
                         "WARNING")

            buffer = self._load_trajectory_buffer()
            buffer.append(trajectory_data)
            new_size = len(buffer)

            self._log(f"Buffer: {new_size - 1} -> {new_size}/{self.eps_per_update}")

            if new_size >= self.eps_per_update:
                self._log("=" * 50)
                self._log(f"UPDATE THRESHOLD REACHED! This worker will perform PPO update.")
                self._log("=" * 50)

                accumulated_data = buffer.copy()

                # Mark update in progress and clear the buffer atomically, so no
                # other worker triggers a second update on the same data.
                metadata['update_in_progress'] = True
                metadata['update_worker'] = self.worker_id
                metadata['update_start_time'] = time.time()
                metadata['buffer_size'] = 0
                self._save_metadata(metadata)

                self._save_trajectory_buffer([])

                return (True, accumulated_data, new_size)

            else:
                self._save_trajectory_buffer(buffer)
                metadata['buffer_size'] = new_size
                self._save_metadata(metadata)

                return (False, None, new_size)

    def finish_update_and_clear_buffer(self, new_policy_state_dict: Dict,
                                       new_optimizer_state_dict: Dict,
                                       training_info: Optional[Dict] = None) -> int:
        """Commit the PPO update: save new weights, bump version, clear buffer.

        Must be called after submit_episode returns should_update=True (while
        that lock context's flags are still set). Returns the new model version.
        """
        self._log("Finishing update and committing new model...")

        with self.file_lock():
            metadata = self._load_metadata()

            expected_worker = metadata.get('update_worker')
            if expected_worker != self.worker_id:
                self._log(f"WARNING: Expected update worker {expected_worker}, "
                         f"but we are {self.worker_id}", "WARNING")

            old_version = metadata.get('model_version', 0)
            new_version = old_version + 1

            # Carry cumulative stats forward from the existing checkpoint.
            global_episodes = 0
            global_updates = 0
            all_rewards = []

            if os.path.exists(self.checkpoint_path):
                try:
                    if SUPPORTS_WEIGHTS_ONLY:
                        old_checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
                    else:
                        old_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                    global_episodes = old_checkpoint.get('global_episodes', 0)
                    global_updates = old_checkpoint.get('global_updates', 0)
                    all_rewards = old_checkpoint.get('all_rewards', [])
                except Exception as e:
                    self._log(f"Could not load old checkpoint for stats: {e}", "WARNING")

            # Rewards come from training_info; the buffer was cleared in submit_episode.
            new_rewards = []
            episodes_in_batch = self.eps_per_update
            if training_info:
                new_rewards = training_info.get('episode_rewards', [])
                episodes_in_batch = training_info.get('episodes_in_batch', self.eps_per_update)

            global_episodes += episodes_in_batch
            global_updates += 1
            all_rewards.extend(new_rewards)

            if len(all_rewards) > 1000:
                all_rewards = all_rewards[-1000:]

            new_checkpoint = {
                'schema_version': 3,
                'policy_state_dict': new_policy_state_dict,
                'optimizer_state_dict': new_optimizer_state_dict,
                'model_version': new_version,
                'global_episodes': global_episodes,
                'global_updates': global_updates,
                'all_rewards': all_rewards,
                'last_update_worker': self.worker_id,
                'timestamp': time.time(),
                'policy_type': self.policy_type
            }
            
            if training_info:
                new_checkpoint['last_training_info'] = training_info

            # Write to a temp file then rename for an atomic checkpoint swap.
            temp_path = self.checkpoint_path + f'.{self.worker_id}.tmp'
            torch.save(new_checkpoint, temp_path)
            shutil.move(temp_path, self.checkpoint_path)

            self._log(f"Saved new checkpoint (v{new_version})")

            metadata['model_version'] = new_version
            metadata['buffer_size'] = 0
            metadata['global_episodes'] = global_episodes
            metadata['global_updates'] = global_updates
            metadata['last_update_worker'] = self.worker_id
            metadata['last_update_time'] = time.time()
            metadata['update_in_progress'] = False
            metadata['update_worker'] = None
            metadata['update_start_time'] = None
            
            if new_rewards:
                metadata['last_batch_avg_reward'] = sum(new_rewards) / len(new_rewards)
                metadata['last_batch_size'] = len(new_rewards)
            
            self._save_metadata(metadata)
            
            self.local_model_version = new_version
            
            self._log("=" * 50)
            self._log(f"UPDATE COMPLETE: v{old_version} -> v{new_version}")
            self._log(f"  Global episodes: {global_episodes}")
            self._log(f"  Global updates: {global_updates}")
            if new_rewards:
                self._log(f"  Batch avg reward: {sum(new_rewards)/len(new_rewards):.4f}")
            self._log("=" * 50)
            
            self._update_collective_stats(episodes_in_batch, new_rewards, training_info)
            
            return new_version

    def check_and_reload_if_needed(self, device: str = 'cpu') -> Tuple[bool, Optional[Dict], Optional[Dict], int]:
        """Reload weights if a newer shared version exists. Call before each episode.

        Returns (was_reloaded, policy_state_dict, optimizer_state_dict, version);
        the state dicts are None when nothing was reloaded.
        """
        # Cheap lock-free version check before paying for the lock.
        if not os.path.exists(self.model_metadata_path):
            return (False, None, None, self.local_model_version)

        try:
            with open(self.model_metadata_path, 'r') as f:
                metadata = json.load(f)

            shared_version = metadata.get('model_version', 0)

            if shared_version <= self.local_model_version:
                return (False, None, None, self.local_model_version)

            self._log(f"Newer model available: v{shared_version} > local v{self.local_model_version}")

            with self.file_lock():
                # Re-read under lock; the version may have moved while waiting.
                with open(self.model_metadata_path, 'r') as f:
                    metadata = json.load(f)
                shared_version = metadata.get('model_version', 0)

                if shared_version <= self.local_model_version:
                    return (False, None, None, self.local_model_version)

                if SUPPORTS_WEIGHTS_ONLY:
                    checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                else:
                    checkpoint = torch.load(self.checkpoint_path, map_location=device)
                
                self.local_model_version = shared_version
                
                self._log(f"Reloaded model: v{self.local_model_version}")
                
                return (
                    True,
                    checkpoint.get('policy_state_dict'),
                    checkpoint.get('optimizer_state_dict'),
                    shared_version
                )
                
        except Exception as e:
            self._log(f"Error checking for updates: {e}", "WARNING")
            return (False, None, None, self.local_model_version)

    def get_current_version(self) -> int:
        """Get the current shared model version without loading weights."""
        if not os.path.exists(self.model_metadata_path):
            return 0
        
        try:
            with open(self.model_metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('model_version', 0)
        except:
            return 0

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current status of the trajectory buffer."""
        try:
            with open(self.model_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'buffer_size': metadata.get('buffer_size', 0),
                'eps_per_update': self.eps_per_update,
                'model_version': metadata.get('model_version', 0),
                'global_episodes': metadata.get('global_episodes', 0),
                'global_updates': metadata.get('global_updates', 0),
                'update_in_progress': metadata.get('update_in_progress', False)
            }
        except:
            return {
                'buffer_size': 0,
                'eps_per_update': self.eps_per_update,
                'model_version': 0,
                'global_episodes': 0,
                'global_updates': 0,
                'update_in_progress': False
            }

    # Internal helpers (call within a file_lock context).

    def _load_trajectory_buffer(self) -> List[Dict]:
        """Load trajectory buffer from disk. Must be called within file_lock context."""
        if not os.path.exists(self.trajectory_buffer_path):
            return []
        
        try:
            with open(self.trajectory_buffer_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self._log(f"Error loading trajectory buffer: {e}", "WARNING")
            return []

    def _save_trajectory_buffer(self, buffer: List[Dict]):
        """Save trajectory buffer to disk. Must be called within file_lock context."""
        temp_path = self.trajectory_buffer_path + '.tmp'
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(buffer, f)
            shutil.move(temp_path, self.trajectory_buffer_path)
        except Exception as e:
            self._log(f"Error saving trajectory buffer: {e}", "ERROR")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _load_metadata(self) -> Dict:
        """Load model metadata. Must be called within file_lock context."""
        if not os.path.exists(self.model_metadata_path):
            return {
                'model_version': 0,
                'buffer_size': 0,
                'global_episodes': 0,
                'global_updates': 0
            }
        
        try:
            with open(self.model_metadata_path, 'r') as f:
                return json.load(f)
        except:
            return {
                'model_version': 0,
                'buffer_size': 0,
                'global_episodes': 0,
                'global_updates': 0
            }

    def _save_metadata(self, metadata: Dict):
        """Save model metadata. Must be called within file_lock context."""
        temp_path = self.model_metadata_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            shutil.move(temp_path, self.model_metadata_path)
        except Exception as e:
            self._log(f"Error saving metadata: {e}", "ERROR")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _update_collective_stats(self, episodes_in_batch: int, rewards: List[float],
                                 training_info: Optional[Dict] = None):
        """Update collective training statistics."""
        try:
            stats = {}
            if os.path.exists(self.stats_path):
                try:
                    with open(self.stats_path, 'r') as f:
                        stats = json.load(f)
                except:
                    stats = {}

            if 'start_time' not in stats:
                stats['start_time'] = time.time()
            
            stats['policy_type'] = self.policy_type
            stats['eps_per_update'] = self.eps_per_update
            stats['last_update_time'] = time.time()
            stats['time_elapsed_hours'] = (time.time() - stats['start_time']) / 3600

            if 'workers' not in stats:
                stats['workers'] = {}

            if self.worker_id not in stats['workers']:
                stats['workers'][self.worker_id] = {
                    'episodes_contributed': 0,
                    'updates_performed': 0,
                    'first_seen': time.time()
                }
            
            worker_stats = stats['workers'][self.worker_id]
            worker_stats['updates_performed'] += 1
            worker_stats['last_update'] = time.time()

            metadata = self._load_metadata()
            stats['total_episodes'] = metadata.get('global_episodes', 0)
            stats['total_updates'] = metadata.get('global_updates', 0)
            stats['current_model_version'] = metadata.get('model_version', 0)
            
            if rewards:
                stats['last_batch_avg_reward'] = sum(rewards) / len(rewards)
                stats['last_batch_success_rate'] = sum(1 for r in rewards if r > 0) / len(rewards)

            if training_info:
                stats['last_training_info'] = training_info

            with open(self.stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            self._log(f"Error updating stats: {e}", "WARNING")

    # Deprecated methods (retained as shims; emit a deprecation warning).

    def load_shared_checkpoint(self, device='cpu') -> Dict:
        """Deprecated: use initialize_or_sync()/check_and_reload_if_needed()."""
        self._log("WARNING: load_shared_checkpoint is deprecated. Use initialize_or_sync() "
                 "and check_and_reload_if_needed() for proper distributed training.", "WARNING")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"No shared checkpoint found at {self.checkpoint_path}")

        with self.file_lock():
            if SUPPORTS_WEIGHTS_ONLY:
                checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
            else:
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
            
            return checkpoint

    def save_shared_checkpoint(self, policy_state_dict: Dict, optimizer_state_dict: Dict,
                             episodes_completed: int, training_rewards: list,
                             additional_info: Optional[Dict] = None,
                             policy_specific_data: Optional[Dict] = None) -> bool:
        """Deprecated: use submit_episode()/finish_update_and_clear_buffer()."""
        self._log("WARNING: save_shared_checkpoint is deprecated. Use submit_episode() "
                 "and finish_update_and_clear_buffer() for proper distributed training.", "WARNING")

        try:
            self.finish_update_and_clear_buffer(
                policy_state_dict,
                optimizer_state_dict,
                training_info=additional_info
            )
            return True
        except Exception as e:
            self._log(f"Error in deprecated save_shared_checkpoint: {e}", "ERROR")
            return False

    def get_training_summary(self) -> Dict:
        """Get training progress summary (falls back to metadata if no stats file)."""
        if not os.path.exists(self.stats_path):
            status = self.get_buffer_status()
            return {
                'policy_type': self.policy_type,
                'total_episodes': status['global_episodes'],
                'total_updates': status['global_updates'],
                'current_model_version': status['model_version'],
                'buffer_size': status['buffer_size'],
                'eps_per_update': self.eps_per_update
            }
        
        try:
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)
            
            return {
                'policy_type': stats.get('policy_type', 'unknown'),
                'total_episodes': stats.get('total_episodes', 0),
                'total_updates': stats.get('total_updates', 0),
                'current_model_version': stats.get('current_model_version', 0),
                'time_elapsed_hours': stats.get('time_elapsed_hours', 0),
                'last_batch_avg_reward': stats.get('last_batch_avg_reward'),
                'last_batch_success_rate': stats.get('last_batch_success_rate'),
                'active_workers': len(stats.get('workers', {})),
                'workers': stats.get('workers', {})
            }
        except Exception as e:
            self._log(f"Error getting training summary: {e}", "WARNING")
            return {}

    def get_log_contents(self, last_n_lines: int = 100) -> str:
        """Get recent log entries."""
        if not os.path.exists(self.log_path):
            return "No log file found."
        
        try:
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-last_n_lines:])
        except Exception as e:
            return f"Error reading log: {e}"

