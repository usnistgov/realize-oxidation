"""
Evaluation module.

Pure-execution mode: load a trained checkpoint and run episodes with
deterministic action selection (highest-probability mode for direction,
magnitude, and axis). NEB + MD verification + rollback are kept identical
to training. No PPO update, no rollout buffer, no parallel workers.
"""
