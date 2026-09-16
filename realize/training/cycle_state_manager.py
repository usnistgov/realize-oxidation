"""Coordinate the Ea-midpoint cycle schedule across parallel workers via quorum.

Workers submit Ea_eff data when they complete a cycle (reach the O threshold).
Once quorum (default 2) workers have submitted, the cycle advances and the midpoint
steps to the next value of the fixed schedule [1.6, 1.5, 1.4, 1.3, 1.2] eV (matching
the serial path and legacy). Pooled Ea_eff values are logged but do not drive the
midpoint. Stragglers detect the advance at the next episode start and reset to adopt
the new midpoint. A file-locked JSON state file makes concurrent access safe.
"""

import json
import os
import time
import fcntl
from typing import List, Optional, Tuple
import logging
_LOG = logging.getLogger(__name__)


class CycleStateManager:
    """Coordinate the fixed-schedule midpoint across parallel workers via shared file.

    State file format (cycle_state.json):
    {
        "cycle": 0,
        "current_midpoint": 1.2,
        "quorum": 2,
        "completed_workers": {
            "worker_0": [0.91, 1.03, 0.87],
            "worker_3": [0.95, 0.88]
        },
        "midpoint_history": [1.2],
        "updated_at": 1234567890.0
    }
    """

    def __init__(self, state_file: str, quorum: int = 2,
                 initial_midpoint: float = 1.2,
                 worker_id: Optional[str] = None):
        self.state_file = state_file
        self.quorum = max(1, quorum)
        self.initial_midpoint = initial_midpoint
        self.worker_id = worker_id or f"pid_{os.getpid()}"

        # Local tracking
        self._my_cycle = 0
        self._submitted_this_cycle = False

        # Initialize state file if it doesn't exist (first worker to start)
        if not os.path.exists(self.state_file):
            self._write_state(self._default_state())
            _LOG.info(f"  [CYCLE STATE] Initialized {self.state_file} "
                  f"(quorum={self.quorum}, midpoint={self.initial_midpoint:.4f}, "
                  f"worker={self.worker_id})")
        else:
            state = self._read_state()
            self._my_cycle = state.get("cycle", 0)
            _LOG.info(f"  [CYCLE STATE] Joined existing state at cycle {self._my_cycle} "
                  f"(midpoint={state.get('current_midpoint', self.initial_midpoint):.4f}, "
                  f"worker={self.worker_id})")

    def _default_state(self):
        return {
            "cycle": 0,
            "current_midpoint": self.initial_midpoint,
            "quorum": self.quorum,
            "completed_workers": {},
            "midpoint_history": [self.initial_midpoint],
            "updated_at": time.time()
        }

    def _read_state(self) -> dict:
        """Read state with shared (read) lock."""
        try:
            with open(self.state_file, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                return state
        except (json.JSONDecodeError, FileNotFoundError):
            return self._default_state()

    def _write_state(self, state: dict):
        """Write state with exclusive lock."""
        state["updated_at"] = time.time()
        tmp = self.state_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_file)

    def _locked_read_modify_write(self, modify_fn) -> dict:
        """Atomic read-modify-write with exclusive lock."""
        with open(self.state_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                try:
                    state = json.load(f)
                except json.JSONDecodeError:
                    state = self._default_state()

                state = modify_fn(state)
                state["updated_at"] = time.time()

                f.seek(0)
                f.truncate()
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return state

    def check_cycle_advanced(self) -> Tuple[bool, float]:
        """Check if global cycle has advanced past our local cycle.

        Called at episode start. If another worker triggered quorum,
        this worker needs to reset and adopt new midpoint.

        Returns:
            (advanced, current_midpoint)
        """
        state = self._read_state()
        if state["cycle"] > self._my_cycle:
            old_cycle = self._my_cycle
            self._my_cycle = state["cycle"]
            self._submitted_this_cycle = False
            _LOG.info(f"  [CYCLE STATE] Worker {self.worker_id}: cycle advanced "
                  f"{old_cycle} -> {self._my_cycle} "
                  f"(new midpoint={state['current_midpoint']:.4f} eV)")
            return True, state["current_midpoint"]
        return False, state["current_midpoint"]

    def submit_cycle_data(self, ea_effs: List[float]) -> Tuple[bool, float]:
        """Submit this worker's Ea_eff data when it finishes a cycle.

        Called when worker reaches O threshold (e.g., 80 O atoms).

        Args:
            ea_effs: Ea_eff values from successful dissociations in this cycle.

        Returns:
            (quorum_met, current_midpoint)
            quorum_met=True means this worker triggered the cycle advance.
        """
        if self._submitted_this_cycle:
            # Already submitted -- just check if cycle advanced
            state = self._read_state()
            advanced = state["cycle"] > self._my_cycle
            if advanced:
                self._my_cycle = state["cycle"]
                self._submitted_this_cycle = False
            return advanced, state["current_midpoint"]

        # A submission carrying no NEB-verified barriers is not a completed
        # cycle's worth of measurement: counting it toward quorum lets a cycle
        # advance with an empty pooled Ea_eff set, leaving the midpoint schedule
        # uncalibrated. Decline it (visibly) and let the worker submit later.
        if not ea_effs:
            state = self._read_state()
            _LOG.warning(
                f"  [CYCLE STATE] Worker {self.worker_id}: submission REJECTED -- "
                f"0 Ea values (no NEB-verified dissociation this cycle). "
                f"Not counted toward quorum; will retry after further episodes."
            )
            return False, state["current_midpoint"]

        result = {"quorum_met": False, "new_midpoint": self.initial_midpoint}

        def modify(state):
            # Guard: don't submit to a cycle that already advanced
            if state["cycle"] > self._my_cycle:
                result["quorum_met"] = True
                result["new_midpoint"] = state["current_midpoint"]
                return state

            # Append our data
            state["completed_workers"][self.worker_id] = ea_effs
            n_submitted = len(state["completed_workers"])
            quorum = state.get("quorum", self.quorum)

            if n_submitted >= quorum:
                # QUORUM MET -- advance cycle
                all_ea = []
                for w_ea in state["completed_workers"].values():
                    all_ea.extend(w_ea)

                # Fixed per-cycle Ea-sigmoid midpoint schedule (matches the serial
                # path and legacy). A fixed schedule gives consistent per-cycle
                # improvement pressure; an adaptive mu = mean(cycle Ea_eff) would
                # track the policy and invert the gradient for the same physical
                # process. Ea_eff values are still pooled and logged but do not
                # drive mu.
                _EA_MIDPOINT_SCHEDULE = [1.6000, 1.5000, 1.4000, 1.3000, 1.2000]
                completed_cycle = state["cycle"]
                next_cycle_idx = completed_cycle + 1
                old_mid = state["current_midpoint"]
                new_mid = _EA_MIDPOINT_SCHEDULE[min(next_cycle_idx, len(_EA_MIDPOINT_SCHEDULE) - 1)]

                state["current_midpoint"] = new_mid
                state["cycle"] += 1
                state["completed_workers"] = {}
                state["midpoint_history"].append(new_mid)

                result["quorum_met"] = True
                result["new_midpoint"] = new_mid

                # Contract log line consumed by cycle_ea_eff.py / barrier_distributions.py.
                if all_ea:
                    _cycle_avg_ea_eff = sum(all_ea) / len(all_ea)
                    _LOG.info(f"  [FIXED MIDPOINT SCHEDULE] Cycle {completed_cycle} complete: "
                          f"avg Ea_eff = {_cycle_avg_ea_eff:.4f} eV ({len(all_ea)} successful episodes). "
                          f"Midpoint: {old_mid:.4f} -> {new_mid:.4f} eV "
                          f"(scheduled for cycle {next_cycle_idx})")
                else:
                    _LOG.info(f"  [FIXED MIDPOINT SCHEDULE] Cycle {completed_cycle} complete: "
                          f"no successful dissociations. Midpoint: {old_mid:.4f} -> {new_mid:.4f} eV "
                          f"(scheduled for cycle {next_cycle_idx})")

                _LOG.info(f"  [CYCLE STATE] QUORUM MET -- {n_submitted}/{quorum} workers | "
                      f"Cycle {completed_cycle} -> {state['cycle']} | "
                      f"{len(all_ea)} Ea values across {n_submitted} workers")
                _LOG.info(f"  Midpoint history: {[f'{m:.4f}' for m in state['midpoint_history']]}")
            else:
                result["new_midpoint"] = state["current_midpoint"]
                _LOG.info(f"  [CYCLE STATE] Worker {self.worker_id} submitted "
                      f"({n_submitted}/{quorum} workers). "
                      f"{len(ea_effs)} Ea values. "
                      f"Waiting for quorum...")

            return state

        state = self._locked_read_modify_write(modify)
        self._submitted_this_cycle = True

        if result["quorum_met"]:
            self._my_cycle = state["cycle"]
            self._submitted_this_cycle = False

        return result["quorum_met"], result["new_midpoint"]

    @property
    def submitted_this_cycle(self) -> bool:
        """Whether this worker has already submitted data for the current cycle."""
        return self._submitted_this_cycle

    def get_current_midpoint(self) -> float:
        """Read current midpoint from shared state."""
        state = self._read_state()
        return state.get("current_midpoint", self.initial_midpoint)
