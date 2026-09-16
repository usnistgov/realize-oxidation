"""
evaluation/eval_runner.py

Single-worker pure-execution run.

Loads a trained checkpoint and runs episodes with deterministic action
selection (mode of each policy head). NEB + MD verification + rollback
are kept identical to training. The run stops once `--n-o-target` oxygen
atoms have been incorporated into the substrate (matches the per-cycle
definition used in training; default 80).

CLI:
    python -m evaluation.eval_runner \
        --checkpoint <path/to/checkpoint_ep549.pth> \
        --output-dir <dir>                          \
        --n-o-target 80                             \
        --max-steps 25                              \
        --device cuda                               \
        --random-seed 0
"""
from __future__ import annotations

import argparse
import faulthandler
faulthandler.enable()

import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ase import io as ase_io


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic execution run (single worker, no PPO update)."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained checkpoint .pth file (e.g. checkpoint_ep549.pth).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for eval outputs (logs, per-step records, summary).")
    p.add_argument("--n-o-target", type=int, default=80,
                   help="Number of dissociated O atoms to accumulate before stopping.")
    p.add_argument("--max-steps", type=int, default=25,
                   help="Max steps per episode (matches training default).")
    p.add_argument("--max-episodes", type=int, default=2000,
                   help="Hard cap on episodes, in case the O target is never reached.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for policy inference (cuda/cpu). NEB/MD always run via LAMMPS.")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Seed for any residual stochasticity (LAMMPS MD verification, etc.).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build trainer + load checkpoint
# ---------------------------------------------------------------------------

def _build_trainer_then_load(checkpoint_path: str,
                             max_steps: int,
                             device: torch.device,
                             logger) -> Any:
    """
    Build a PPOTrainer with discovery disabled, then manually load the
    user-specified checkpoint into the trainer's components.

    The trainer's __init__ wires up policy, dual backbones, dual normalizers,
    episode_runner, env, etc. We reuse all of it. We only suppress its
    automatic checkpoint discovery by pointing checkpoint_dir at an empty
    temp dir and disabling shared-model mode.
    """
    from realize.training.trainer import PPOTrainer
    from realize.training.config_manager import ConfigManager
    from realize.config.default_config import DEFAULT_CONFIG

    cfg_mgr = ConfigManager(DEFAULT_CONFIG)

    # Mock argparse-style namespace with only the fields update_from_args reads.
    # All others stay at their default-config values.
    class _Args:
        episodes = None
        steps = max_steps
        random_seed = None
        save_interval = None
        lmax = None
        num_channels = None
        temperature_init = None
        temperature_min = None
        use_neb = None
    config = cfg_mgr.update_from_args(_Args())

    # ------------------------------------------------------------------
    # Disable trainer's automatic checkpoint discovery.
    #
    # The trainer's __init__ (checkpoint_discovery_v1 patch) scans TWO dirs:
    #   (a) config.runtime.checkpoint_dir
    #   (b) os.path.join(get_results_dir(), 'checkpoints')
    # We must defeat BOTH so it doesn't auto-resume from the project's own
    # latest checkpoint. We:
    #   1. Point (a) at an empty temp dir.
    #   2. Monkey-patch config.paths.get_results_dir() -- the only API the
    #      trainer uses to compute (b) -- to also return our empty temp dir.
    # The monkey-patch is reverted in a finally block after trainer init.
    # ------------------------------------------------------------------
    empty_ckpt_dir = tempfile.mkdtemp(prefix="eval_empty_ckpt_")
    # The trainer scans <results_dir>/checkpoints -- make that subdir exist & empty too.
    os.makedirs(os.path.join(empty_ckpt_dir, "checkpoints"), exist_ok=True)

    config.runtime.checkpoint_dir = empty_ckpt_dir
    config.runtime.use_shared_model = False
    config.runtime.resume_path = None
    config.runtime.shared_model_dir = None
    config.runtime.episodes_per_update = 999_999_999  # never trigger

    import realize.config.paths as _paths_module
    import realize.training.trainer as _trainer_module
    _orig_get_results_dir = _paths_module.get_results_dir

    def _eval_get_results_dir():
        return empty_ckpt_dir

    _paths_module.get_results_dir = _eval_get_results_dir
    # The trainer module already imported get_results_dir into its own namespace
    # (`from realize.config.paths import get_results_dir`) -- patch that binding too.
    if hasattr(_trainer_module, "get_results_dir"):
        _trainer_module.get_results_dir = _eval_get_results_dir

    try:
        # Build trainer -- this initializes env, agent (policy + policy_old + optimizer),
        # episode_runner (dual backbones, dual normalizers).
        # With both discovery dirs empty, trainer falls through to "Level 3: from scratch",
        # so start_episode=0 and the substrate-state check is bypassed.
        trainer = PPOTrainer(config=config, device=device, logger=logger,
                             worker_id=0, world_size=1)
    finally:
        # Restore the real get_results_dir BEFORE running episodes -- episode_runner
        # and downstream callers need it to point at the project's actual results dir
        # for movie files, NEB outputs, etc.
        _paths_module.get_results_dir = _orig_get_results_dir
        if hasattr(_trainer_module, "get_results_dir"):
            _trainer_module.get_results_dir = _orig_get_results_dir

    # Now load the user-specified checkpoint manually into the wired components.
    print(f"\n[EVAL] Loading checkpoint: {checkpoint_path}", flush=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Required top-level keys.
    for k in ("policy_state_dict",):
        if k not in ckpt:
            raise RuntimeError(f"Checkpoint missing required key '{k}'.")

    # 1. Policy weights (includes sh_head, frame_encoder, axis_from_frame,
    #    magnitude_predictor, value_mlp, multipole_injector, critic_input_normalizer
    #    running stats, axis log_conc_init buffer, etc.)
    missing, unexpected = trainer.agent.policy.load_state_dict(
        ckpt["policy_state_dict"], strict=False
    )
    if missing:
        print(f"[EVAL] WARNING: missing keys when loading policy: {len(missing)} keys "
              f"(first few: {list(missing)[:5]})", flush=True)
    if unexpected:
        print(f"[EVAL] WARNING: unexpected keys when loading policy: {len(unexpected)} keys "
              f"(first few: {list(unexpected)[:5]})", flush=True)
    print(f"[EVAL] Loaded policy_state_dict ({len(ckpt['policy_state_dict'])} keys)", flush=True)

    # Mirror to policy_old (we won't use it, but keeps invariant intact).
    trainer.agent.policy_old.load_state_dict(trainer.agent.policy.state_dict())

    # 2. Actor backbone (fine-tuned SevenNet).
    if "actor_backbone_state_dict" in ckpt:
        trainer.episode_runner.actor_embedding_extractor.load_state_dict(
            ckpt["actor_backbone_state_dict"], strict=False
        )
        print(f"[EVAL] Loaded actor_backbone_state_dict", flush=True)
    else:
        print(f"[EVAL] WARNING: no actor_backbone_state_dict in checkpoint; "
              f"actor backbone stays at the freshly loaded MLIP weights.", flush=True)

    # 3. Critic backbone (frozen during training; load anyway for exact reproducibility).
    if "critic_backbone_state_dict" in ckpt:
        trainer.episode_runner.critic_embedding_extractor.load_state_dict(
            ckpt["critic_backbone_state_dict"], strict=False
        )
        print(f"[EVAL] Loaded critic_backbone_state_dict", flush=True)

    # 4. Actor StateNormalizer (480-dim running stats).
    if "normalizer_state" in ckpt:
        ns = ckpt["normalizer_state"]
        # The normalizer initializes running_stats lazily on first normalize() call.
        # We need stats present before that, so trigger initialization explicitly.
        if trainer.episode_runner.actor_state_normalizer.running_stats is None:
            from realize.training.state_normalizer import StateRunningMeanStd
            mean = np.asarray(ns["mean"], dtype=np.float64)
            trainer.episode_runner.actor_state_normalizer.running_stats = StateRunningMeanStd(
                shape=(mean.shape[0],)
            )
        rs = trainer.episode_runner.actor_state_normalizer.running_stats
        rs.mean = np.asarray(ns["mean"], dtype=np.float64)
        rs.var = np.asarray(ns["var"], dtype=np.float64)
        rs.count = float(ns["count"])
        trainer.episode_runner.actor_state_normalizer._initialized = True
        print(f"[EVAL] Loaded actor normalizer_state "
              f"(count={rs.count:.0f}, dim={rs.mean.shape[0]})", flush=True)
    else:
        print(f"[EVAL] WARNING: no normalizer_state in checkpoint.", flush=True)

    # 5. Critic StateNormalizer if present (legacy; current code keeps critic
    #    normalization inside policy via critic_input_normalizer buffers, which
    #    are already loaded with policy_state_dict above).
    if "critic_normalizer_state" in ckpt:
        # Locate the critic state normalizer if the runner exposes it.
        cs = ckpt["critic_normalizer_state"]
        crit_norm = getattr(trainer.episode_runner, "critic_state_normalizer", None)
        if crit_norm is not None:
            from realize.training.state_normalizer import StateRunningMeanStd
            if crit_norm.running_stats is None:
                mean = np.asarray(cs["mean"], dtype=np.float64)
                crit_norm.running_stats = StateRunningMeanStd(shape=(mean.shape[0],))
            crit_norm.running_stats.mean = np.asarray(cs["mean"], dtype=np.float64)
            crit_norm.running_stats.var = np.asarray(cs["var"], dtype=np.float64)
            crit_norm.running_stats.count = float(cs["count"])
            print(f"[EVAL] Loaded critic_normalizer_state", flush=True)

    # 6. Cycle state (for adaptive midpoint mu -- used only for logging R_Ea).
    if "cycle_state_full" in ckpt:
        cs = ckpt["cycle_state_full"]
        print(f"[EVAL] Cycle state at checkpoint: "
              f"cycle={cs.get('current_cycle', '?')}, "
              f"midpoint={cs.get('current_midpoint', '?')} eV", flush=True)

    saved_episode = ckpt.get("episode", -1)
    print(f"[EVAL] Checkpoint episode index: {saved_episode}", flush=True)

    # ---- FREEZE EVERYTHING ----
    trainer.agent.policy.eval()
    trainer.agent.policy_old.eval()
    for p in trainer.agent.policy.parameters():
        p.requires_grad = False
    for p in trainer.agent.policy_old.parameters():
        p.requires_grad = False
    trainer.episode_runner.actor_embedding_extractor.model.eval()
    trainer.episode_runner.critic_embedding_extractor.model.eval()
    for p in trainer.episode_runner.actor_embedding_extractor.model.parameters():
        p.requires_grad = False
    for p in trainer.episode_runner.critic_embedding_extractor.model.parameters():
        p.requires_grad = False

    # Freeze actor StateNormalizer (training-time it adapts; in eval it must not).
    trainer.episode_runner.actor_state_normalizer.freeze()
    print(f"[EVAL] Policy + dual backbones in eval(), all params frozen, "
          f"actor normalizer frozen.", flush=True)

    return trainer, saved_episode


# ---------------------------------------------------------------------------
# Episode loop (single worker, mirrors EpisodeRunner.run_episode logic, but
# uses deterministic_act() instead of agent.select_action()).
# ---------------------------------------------------------------------------

def _build_geometry_for_critic(env, o2_positions_world: np.ndarray,
                               env_multipole: torch.Tensor) -> torch.Tensor:
    """
    Build the 10-dim invariant geometry vector that CriticEmbedding expects.
    Layout follows the training-time augment_geometry_with_context contract:
    we compute the same hand-crafted invariants used in episode_runner.
    """
    from realize.networks.encoders.critic_embedding import augment_geometry_with_context
    # augment_geometry_with_context(env, o2_positions, env_multipole) is what
    # episode_runner uses; we delegate to it for an exact match.
    return augment_geometry_with_context(env, o2_positions_world, env_multipole)


def _read_o2_indices_and_positions(env) -> tuple:
    """Get O2 atom indices and world positions from structure manager."""
    o1_idx, o2_idx = env.structure_manager.get_o2_indices()
    atoms = env.structure_manager.get_atoms()
    pos = atoms.get_positions()
    o2_positions_world = np.stack([pos[o1_idx], pos[o2_idx]], axis=0)  # [2, 3]
    return o1_idx, o2_idx, o2_positions_world


def _get_neighbor_positions_world(env, o1_idx: int, o2_idx: int,
                                   device: torch.device) -> Optional[torch.Tensor]:
    """All non-O2 atom positions, [1, N_substrate, 3]."""
    atoms = env.structure_manager.get_atoms()
    pos = atoms.get_positions()
    mask = np.ones(len(atoms), dtype=bool)
    mask[o1_idx] = False
    mask[o2_idx] = False
    sub = pos[mask]
    if sub.shape[0] == 0:
        return None
    return torch.from_numpy(sub).float().to(device).unsqueeze(0)


def run_episodes(trainer, args, output_dir: str, logger) -> Dict[str, Any]:
    """
    Main eval loop: keep running episodes until n_o_target dissociated O atoms
    have been incorporated, or max_episodes is reached.
    """
    from realize.environment.episode_manager import (
        reset_episode,
        step_environment_with_rollback,
    )
    from realize.ppo.rewards import calculate_pathway_reward
    from evaluation.deterministic_action import deterministic_act
    from realize.config.paths import get_embedding_checkpoint_path

    env = trainer.env
    policy = trainer.agent.policy
    runner = trainer.episode_runner
    device = trainer.device
    cfg = trainer.config
    embedding_extractor = runner.actor_embedding_extractor

    # Reward params (for LOGGING R_Ea -- not used to choose actions).
    reward_cfg = cfg.reward
    runtime_cfg = cfg.runtime

    # Per-episode records.
    records: List[Dict[str, Any]] = []
    cumulative_o_at_start = 0
    n_o_target = args.n_o_target

    # Initialize the env's dissociated-O counter if reset_episode hasn't run yet.
    # In training this attribute is created lazily inside reset_episode(); we need
    # it to exist before the loop's pre-check on the first iteration.
    if not hasattr(env, "cumulative_dissociated_o_count"):
        env.cumulative_dissociated_o_count = 0
    if hasattr(env, "structure_manager") and not hasattr(
        env.structure_manager, "cumulative_dissociated_o_count"
    ):
        env.structure_manager.cumulative_dissociated_o_count = 0

    summary_path = os.path.join(output_dir, "eval_summary.json")
    per_episode_path = os.path.join(output_dir, "per_episode.jsonl")
    per_step_path = os.path.join(output_dir, "per_step.jsonl")
    open(per_episode_path, "w").close()
    open(per_step_path, "w").close()

    t_run_start = time.time()

    for episode in range(args.max_episodes):
        if env.cumulative_dissociated_o_count >= n_o_target:
            print(f"\n[EVAL] Target reached ({env.cumulative_dissociated_o_count} >= "
                  f"{n_o_target} O atoms). Stopping.", flush=True)
            break

        cumulative_o_at_start = env.cumulative_dissociated_o_count
        print(f"\n{'=' * 70}", flush=True)
        print(f"[EVAL] Episode {episode}  "
              f"(O incorporated so far: {cumulative_o_at_start}/{n_o_target})", flush=True)
        print(f"{'=' * 70}", flush=True)

        try:
            # 1. Reset environment (places fresh O2, relaxes, returns initial embeddings).
            dist0, embeddings_dict = reset_episode(
                env=env,
                episode_id=episode,
                checkpoint_path=get_embedding_checkpoint_path(),
                device=device,
                embedding_extractor=embedding_extractor,
            )
            # NOTE: embeddings_dict is unused here -- the per-step loop below
            # re-extracts embeddings fresh from env.structure_manager.get_atoms()
            # on every iteration, exactly mirroring training/episode_runner.py.
            del embeddings_dict, dist0
        except (TypeError, AttributeError, KeyError, NameError, ImportError):
            print(f"[EVAL] FATAL: reset_episode contract mismatch at episode {episode}:", flush=True)
            traceback.print_exc()
            raise
        except (RuntimeError, OSError) as e:
            print(f"[EVAL] reset_episode runtime error at episode {episode}: {e}", flush=True)
            traceback.print_exc()
            records.append({
                "episode": episode,
                "outcome": "reset_failed",
                "error": str(e),
            })
            continue

        # Save initial state file for the rollback of step 0.
        from realize.config.paths import get_temp_dir
        prev_state_file = os.path.join(get_temp_dir(), f"eval_prev_ep{episode}.data")
        env.save_current_state(prev_state_file)

        episode_steps: List[Dict[str, Any]] = []
        ea_list: List[float] = []
        episode_dissociated = False
        episode_done = False
        outcome = "incomplete"
        neb_failures = 0

        for step in range(args.max_steps):
            # NOTE: the inner try/except below catches ONLY runtime errors from
            # LAMMPS / NEB (RuntimeError, OSError). Programming-error types
            # (TypeError, AttributeError, KeyError) are deliberately re-raised
            # so the runner crashes loudly on any signature/contract mismatch
            # instead of silently looping through broken episodes.
            try:
                # ---- 2. Build inputs the SAME way training/episode_runner does. ----
                # (mirrors training/episode_runner.py around lines 575-608)

                atoms_current = env.structure_manager.get_atoms()
                positions = atoms_current.get_positions()  # [N, 3]
                o2_positions_np = positions[-2:]           # last two atoms = O2 (training convention)
                o2_positions = (
                    torch.from_numpy(o2_positions_np).float().to(device).unsqueeze(0)
                )  # [1, 2, 3]

                # Actor backbone -- symmetric (raw) + asymmetric (for axis head).
                # episode_runner gets symmetric and asymmetric in two separate calls;
                # we do the same to stay faithful.
                with torch.no_grad():
                    actor_emb_dict = embedding_extractor.get_equivariant_embedding(
                        atoms_current, return_symmetric_only=False
                    )
                actor_sym_raw = actor_emb_dict["symmetric"].detach()
                actor_asym = actor_emb_dict["asymmetric"].detach()

                # Critic backbone -- full dict for CriticEmbedding.
                with torch.no_grad():
                    critic_emb_dict = runner.critic_embedding_extractor.get_equivariant_embedding(
                        atoms_current, return_symmetric_only=False
                    )
                o1_embedding = critic_emb_dict["o1_embedding"].detach()
                o2_embedding = critic_emb_dict["o2_embedding"].detach()
                geometry_base = critic_emb_dict["geometry"].detach()

                # Normalize actor embedding (frozen normalizer in eval).
                actor_state = runner.actor_state_normalizer.normalize(actor_sym_raw)

                # Ensure all tensors are batched [1, ...] and on the right device.
                def _ensure_batched(t):
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    return t.to(device)
                actor_state = _ensure_batched(actor_state)
                actor_asym = _ensure_batched(actor_asym)
                o1_embedding = _ensure_batched(o1_embedding)
                o2_embedding = _ensure_batched(o2_embedding)
                geometry_base = _ensure_batched(geometry_base)

                # ---- 3. Environmental multipole -- match episode_runner's 5-arg call. ----
                with torch.no_grad():
                    all_positions = torch.tensor(
                        positions, dtype=torch.float32, device=device
                    )
                    substrate_positions = all_positions[:-2]                 # [N_sub, 3]
                    o2_com = all_positions[-2:].mean(dim=0, keepdim=True)    # [1, 3]
                    atomic_numbers = torch.tensor(
                        atoms_current.get_atomic_numbers(), device=device
                    )
                    substrate_species = runner.env_multipole.get_species_idx(
                        atomic_numbers[:-2]
                    )
                    cell_t = torch.tensor(
                        np.array(atoms_current.get_cell()),
                        dtype=torch.float32, device=device,
                    )
                    pbc_t = torch.tensor(
                        np.array(atoms_current.get_pbc()),
                        dtype=torch.bool, device=device,
                    )
                    env_multipole = runner.env_multipole(
                        o2_com, substrate_positions, substrate_species, cell_t, pbc_t
                    )  # [1, 9]

                # ---- 4. Geometry -- augment with step-fraction + multipole invariants. ----
                from realize.networks.encoders.critic_embedding import augment_geometry_with_context
                geometry = augment_geometry_with_context(
                    geometry_base, step, args.max_steps, env_multipole
                )
                geometry = _ensure_batched(geometry)

                # ---- 5. Neighbor positions for the magnitude head's Bessel embedding. ----
                neighbor_positions = (
                    torch.tensor(positions[:-2], dtype=torch.float32, device=device)
                    .unsqueeze(0)
                )  # [1, N_sub, 3]

                # ---- 6. Action selection: mode on first attempt, sample on retry. ----
                #
                # env.current_try_num is the count of consecutive failed attempts
                # at this step. step_environment_with_rollback maintains it:
                # incrementing on NEB failure, resetting to 0 on success or
                # catastrophic failure. The training loop in
                # training/episode_runner.py escapes failed steps because
                # agent.select_action() resamples on every call. A pure
                # deterministic policy (mode of every head) is a function of
                # state alone and would return the same losing move on retry,
                # producing a death spiral. So: try_num=0 -> mode (best guess),
                # try_num>=1 -> sample stochastically (matches training retry).
                current_try_num = int(getattr(env, "current_try_num", 0))
                use_deterministic = (current_try_num == 0)
                action_dict, diag = deterministic_act(
                    policy=policy,
                    state=actor_state,
                    o2_positions=o2_positions,
                    o2_embedding_asymmetric=actor_asym,
                    neighbor_positions=neighbor_positions,
                    env_multipole=env_multipole,
                    o1_embedding=o1_embedding,
                    o2_embedding=o2_embedding,
                    geometry=geometry,
                    episode=episode,
                    deterministic=use_deterministic,
                )
                if not use_deterministic:
                    print(f"  [EVAL] try_num={current_try_num} -> stochastic re-sample "
                          f"(direction_logp={diag['direction_log_prob']:.3f}, "
                          f"|s|={diag['magnitude']:.3f})", flush=True)

                # ---- 7. Build the (trans_action, rot_action) pair the env step expects. ----
                trans_vec_np = action_dict["translation_vector"][0].cpu().numpy()
                axis_dir_np = action_dict["axis_direction"][0].cpu().numpy()
                rot_action_np = np.stack([trans_vec_np, axis_dir_np], axis=0)  # [2, 3]
                trans_action_np = trans_vec_np  # SH path uses dummy here

                # 5. Step the environment with rollback (NEB + invalid-step rejection).
                step_t0 = time.time()
                step_out = step_environment_with_rollback(
                    env=env,
                    trans_action=trans_action_np,
                    rot_action=rot_action_np,
                    episode=episode,
                    prev_state=prev_state_file,
                    use_neb=reward_cfg.use_neb_reward,
                    ea_weight=reward_cfg.ea_weight,
                    dissociation_weight=reward_cfg.dissociation_weight,
                    require_dissociation=reward_cfg.require_dissociation,
                    is_final_step=(step == args.max_steps - 1),
                    checkpoint_path=get_embedding_checkpoint_path(),
                    timeout_penalty=reward_cfg.timeout_penalty,
                    invalid_step_penalty=reward_cfg.invalid_step_penalty,
                    high_barrier_threshold=reward_cfg.high_barrier_threshold,
                    high_barrier_penalty=reward_cfg.high_barrier_penalty,
                    oscillation_penalty=reward_cfg.oscillation_penalty,
                    step_count=step,
                    max_steps=args.max_steps,
                    reward_decay_factor=reward_cfg.Ea_gamma,
                    agent=trainer.agent,
                    current_try_num=0,
                    max_trans_distance=cfg.model.max_trans_distance,
                    alpha=0.0,                     # no exploration mixing in eval
                    calculate_neb_output=None,
                    logger=logger,
                    pe_reward_weight=reward_cfg.pe_reward_weight,
                    action_dict=action_dict,
                    device=device,
                    pe_shaping_gamma=reward_cfg.pe_shaping_gamma,
                    invalid_penalty_cap=reward_cfg.invalid_penalty_cap,
                    embedding_extractor=embedding_extractor,
                )
                step_dt = time.time() - step_t0

                # step_environment_with_rollback returns an 8-tuple:
                #   (dist_new, embedding_new, reward, is_done, is_valid,
                #    activation_energy, try_num, md_verified_dissociation)
                # Confirmed by every return statement in environment/episode_manager.py
                # within the function body. Reject anything else strictly.
                if not isinstance(step_out, tuple) or len(step_out) != 8:
                    raise RuntimeError(
                        f"step_environment_with_rollback returned {type(step_out).__name__} "
                        f"of length {len(step_out) if hasattr(step_out, '__len__') else '?'}; "
                        f"expected an 8-tuple."
                    )
                (dist_new, embedding_new, raw_reward, is_done, is_valid,
                 activation_energy, try_num_returned, md_verified_in_step) = step_out

                # 6. Per-step activation energy (NEB barrier).
                #    The function uses -1.0 to signal "no NEB result for this step"
                #    (e.g. invalid step, rollback, NEB failure).
                step_ea = None
                if activation_energy is not None and float(activation_energy) >= 0.0:
                    step_ea = float(activation_energy)
                    if is_valid:
                        ea_list.append(step_ea)

                # 7. Update prev_state_file for next step's rollback context.
                if is_valid:
                    env.save_current_state(prev_state_file)
                else:
                    neb_failures += 1

                # NOTE: no embedding refresh needed -- the next iteration re-extracts
                # embeddings fresh from env.structure_manager.get_atoms() at the top
                # of the loop, matching training/episode_runner.py.

                # 8. Record per-step.
                step_rec = {
                    "episode": episode,
                    "step": step,
                    "valid": bool(is_valid),
                    "done": bool(is_done),
                    "try_num_in": current_try_num,
                    "try_num_out": int(try_num_returned),
                    "deterministic": bool(use_deterministic),
                    "md_verified_in_step": bool(md_verified_in_step),
                    "reward_raw": float(raw_reward),
                    "ea": step_ea,
                    "d_oo": float(dist_new) if dist_new is not None else None,
                    "magnitude": diag["magnitude"],
                    "log_kappa": diag["log_kappa"],
                    "value": diag["value"],
                    "grid_idx": diag["grid_idx"],
                    "direction_log_prob": diag["direction_log_prob"],
                    "step_time_s": step_dt,
                }
                episode_steps.append(step_rec)
                with open(per_step_path, "a") as f:
                    f.write(json.dumps(step_rec) + "\n")

                _det_marker = "MODE" if use_deterministic else f"SAMPLE(try={current_try_num})"
                print(f"  step {step:02d} [{_det_marker}] valid={is_valid}  d_OO={dist_new:.3f}  "
                      f"|s|={diag['magnitude']:.3f}  Ea={step_ea}  "
                      f"raw_R={raw_reward:.4f}  done={is_done}  ({step_dt:.1f}s)", flush=True)

                if is_done:
                    episode_done = True
                    # step_environment_with_rollback already ran MD verification
                    # internally when a dissociation candidate was flagged.
                    # The 8th return value tells us the verified outcome.
                    #
                    # IMPORTANT: when md_verified_in_step is True, the env has ALREADY
                    # incremented env.cumulative_dissociated_o_count internally. We
                    # must NOT add another +2 here, or the counter drifts ahead of
                    # reality and reset_episode crashes with "Missing atoms".
                    if md_verified_in_step:
                        episode_dissociated = True
                        outcome = "dissociated"
                        print(f"  [EVAL] MD-VERIFIED dissociation. "
                              f"Cumulative O = {env.cumulative_dissociated_o_count}", flush=True)
                    else:
                        # is_done with md_verified_in_step=False means termination
                        # for a non-dissociation reason (escape, catastrophic NEB,
                        # etc.) -- see the various return statements in
                        # environment/episode_manager.py around lines 972, 997, 1218.
                        outcome = "terminated_no_dissociation"
                        print(f"  [EVAL] Episode terminated without dissociation "
                              f"(d_OO={dist_new:.3f}).", flush=True)
                    break

            except (TypeError, AttributeError, KeyError, NameError, ImportError) as e:
                # Programming-error class -- signature/contract mismatch, missing
                # attribute, wrong dict key, etc. These indicate the eval runner
                # is out of sync with the training code path. Re-raise immediately
                # so the bug is visible at the FIRST occurrence rather than
                # buried under hundreds of silently-failed episodes.
                print(f"[EVAL] FATAL programming error at episode {episode} step {step}:", flush=True)
                traceback.print_exc()
                raise
            except (RuntimeError, OSError) as e:
                # Genuine runtime issue (LAMMPS crash, NEB convergence failure,
                # filesystem error). Log and end this episode, continue to the next.
                print(f"[EVAL] Episode {episode} step {step} runtime error: {e}", flush=True)
                traceback.print_exc()
                outcome = f"step_error:{type(e).__name__}"
                break

        else:
            # Fell through max_steps with no termination.
            outcome = "timeout"

        # 10. Episode-level reward (for logging the same R_Ea, Ea_eff as in training).
        try:
            if len(ea_list) > 0:
                (R_tot, R_Ea, R_diss), diag = calculate_pathway_reward(
                    activation_energies=ea_list,
                    temperature=1000.0,
                    dissociated=episode_dissociated,
                    ea_weight=reward_cfg.ea_weight,
                    dissociation_weight=reward_cfg.dissociation_weight,
                    require_dissociation=reward_cfg.require_dissociation,
                    episode_ended=True,
                    timeout_penalty=reward_cfg.timeout_penalty,
                    Ea_gamma=reward_cfg.Ea_gamma,
                    cap_unsuccessful_reward=reward_cfg.cap_unsuccessful_reward,
                    ea_sigmoid_k=reward_cfg.ea_sigmoid_k,
                    ea_sigmoid_midpoint=reward_cfg.ea_sigmoid_midpoint,
                    debug=True,
                )
                ea_eff = diag.get("Ea_eff")
            else:
                R_tot = R_Ea = R_diss = 0.0
                ea_eff = None
        except (TypeError, AttributeError, KeyError) as e:
            # Same principle -- programming errors must surface.
            print(f"[EVAL] FATAL: reward computation contract mismatch at episode {episode}:", flush=True)
            traceback.print_exc()
            raise
        except RuntimeError as e:
            print(f"[EVAL] reward computation runtime error at episode {episode}: {e}", flush=True)
            R_tot = R_Ea = R_diss = float("nan")
            ea_eff = None

        ep_rec = {
            "episode": episode,
            "outcome": outcome,
            "dissociated": episode_dissociated,
            "n_steps": len(episode_steps),
            "n_valid_neb": len(ea_list),
            "n_neb_failures": neb_failures,
            "ea_eff": ea_eff,
            "R_tot": float(R_tot),
            "R_Ea": float(R_Ea),
            "R_diss": float(R_diss),
            "cumulative_o_after": env.cumulative_dissociated_o_count,
        }
        records.append(ep_rec)
        with open(per_episode_path, "a") as f:
            f.write(json.dumps(ep_rec) + "\n")

        print(f"[EVAL] Episode {episode} summary: outcome={outcome}, "
              f"steps={len(episode_steps)}, Ea_eff={ea_eff}, "
              f"R_tot={R_tot:.3f}, R_Ea={R_Ea:.3f}", flush=True)

        # Mark MD verification flags so the next reset_episode picks the right branch.
        if episode_dissociated:
            env.previous_episode_dissociated = True
            env.last_md_verified_dissociation = True
        else:
            env.previous_episode_dissociated = False
            env.last_md_verified_dissociation = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary.
    t_run_total = time.time() - t_run_start
    n_episodes = len(records)
    n_success = sum(1 for r in records if r.get("dissociated"))
    summary = {
        "checkpoint": args.checkpoint,
        "n_o_target": n_o_target,
        "cumulative_o_dissociated": env.cumulative_dissociated_o_count,
        "n_episodes": n_episodes,
        "n_success": n_success,
        "success_rate": n_success / max(1, n_episodes),
        "wall_time_s": t_run_total,
        "max_steps": args.max_steps,
        "device": str(trainer.device),
        "seed": args.random_seed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[EVAL] Wrote summary to {summary_path}", flush=True)
    print(f"[EVAL] {json.dumps(summary, indent=2)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve ALL user-supplied paths to absolute BEFORE any chdir,
    # since args may be relative (e.g. "results/checkpoints/...") and
    # we are about to chdir into the project's tmp/ directory below
    # to mirror main.py's behaviour (LAMMPS NEB writes scratch files
    # like image.*, replica-*.dump, log.lammps.* with relative paths).
    # ------------------------------------------------------------------
    abs_checkpoint = os.path.abspath(args.checkpoint)
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_dir = abs_output_dir  # downstream code uses this name

    # Seeds.
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device.
    if "cuda" in args.device.lower() and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"[EVAL] device = {device}", flush=True)

    # Logger (uses absolute path -- survives chdir below).
    from realize.utils.logger import UnifiedLogger
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = UnifiedLogger(base_dir=log_dir)

    # ------------------------------------------------------------------
    # chdir into project's tmp/ directory, exactly like main.py does
    # at line 224. LAMMPS NEB and replica processes write a flock of
    # scratch files (image.1-7, replica-*.dump, log.lammps.0-N,
    # screen.0-N, res.lammpstrj, tmp.lammps.variable) using RELATIVE
    # paths, so they end up in whatever cwd Python is in. Mirror the
    # training-time setup so they land in tmp/ instead of the project
    # root.
    #
    # All path getters in config/paths.py return ABSOLUTE paths
    # (computed from BASE_DIR), so chdir doesn't break any of them.
    # ------------------------------------------------------------------
    from realize.config.paths import get_temp_dir
    original_cwd = os.getcwd()
    temp_dir = get_temp_dir()
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)
    print(f"[EVAL] chdir: {original_cwd} -> {temp_dir}", flush=True)

    try:
        # Build trainer + load checkpoint.
        trainer, saved_episode = _build_trainer_then_load(
            checkpoint_path=abs_checkpoint,
            max_steps=args.max_steps,
            device=device,
            logger=logger,
        )

        # Run.
        # Patch args.output_dir to the absolute version so run_episodes
        # writes JSONL/summary to the right place after the chdir.
        args.output_dir = abs_output_dir
        args.checkpoint = abs_checkpoint
        summary = run_episodes(trainer, args, output_dir, logger)
        return 0 if summary["n_success"] > 0 else 1
    finally:
        # Always restore cwd, even if eval crashed mid-run.
        os.chdir(original_cwd)
        print(f"[EVAL] chdir restored: {os.getcwd()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
