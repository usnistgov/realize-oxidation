"""Episode-end reward calculation and distribution over steps."""
import numpy as np
from typing import Optional

from realize.ppo.rewards import calculate_pathway_reward


class EpisodeRewardMixin:
    """Episode-end reward calculation/distribution + reward-buffer bookkeeping
    for EpisodeRunner.

    All four methods are invoked from run_episode. ``self`` (config/agent/
    reward_scaler plus the buffer helpers) resolves via the MRO back to
    EpisodeRunner, a plain orchestrator with no torch Parameters, so this mixin
    cannot perturb the policy fingerprint.
    """

    def _get_buffer_index(self) -> int:
        """Get current buffer index."""
        return self.agent.get_buffer_length()

    def _update_buffer_reward(self, index: int, reward: float, done: bool):
        """Set a per-step reward (PE-based reward, oscillation/invalid penalty) in
        the buffer immediately. additive=False replaces any placeholder rather than
        stacking; episode-end rewards (Ea, dissociation) are applied separately.
        """
        if reward != 0.0:
            if not np.isfinite(reward):
                raise ValueError(f"Step reward {reward} is not finite (NaN/Inf)")

            if not hasattr(self.agent, 'update_rewards'):
                raise RuntimeError("Agent has no update_rewards method")

            self.agent.update_rewards(
                index=index,
                reward_total=reward,
                done=done,         # False for per-step rewards
                additive=False     # replace placeholder, not cumulative
            )

    def _calculate_final_rewards(self, activations, dissociated, episode):
        """Calculate final rewards for the episode."""

        # Get reward weights from config (no defaults).
        if hasattr(self.config, 'reward'):
            ea_weight = self.config.reward.ea_weight
            dissociation_weight = self.config.reward.dissociation_weight
            timeout_penalty = self.config.reward.timeout_penalty
        elif hasattr(self.config, 'reward_config'):
            ea_weight = self.config.reward_config.ea_weight
            dissociation_weight = self.config.reward_config.dissociation_weight
            timeout_penalty = self.config.reward_config.timeout_penalty
        else:
            raise ValueError(
                "Reward config not found. Config must have either 'reward' or 'reward_config' attribute. "
                "Required fields: ea_weight, dissociation_weight, timeout_penalty"
            )

        # Use adaptive ea_weight if scaler is active
        adaptive_ea_weight = self.reward_scaler.get_ea_weight()
        total, ea, dissoc = calculate_pathway_reward(
            activation_energies=activations,
            temperature=1000.0,
            dissociated=dissociated,
            ea_weight=adaptive_ea_weight,
            dissociation_weight=dissociation_weight,
            timeout_penalty=timeout_penalty,
            **self._get_reward_kwargs()
        )

        # Update adaptive reward scaler
        self.reward_scaler.update(total)

        return {
            'total': total,
            'components': {'ea': ea, 'dissociation': dissoc}
        }

    def _distribute_episode_reward(self, episode_reward: float, valid_step_indices: list,
                                   ea_component: float, dissoc_component: float,
                                   lambda_decay: float = 0.95,
                                   is_truncated: bool = False,
                                   is_true_terminal: bool = False,
                                   truncation_value: Optional[float] = None) -> None:
        """Distribute episode-end reward across valid steps with exponential decay,
        giving most credit to the last step but partial credit to earlier ones:
            weight[t] = lambda^(T-1-t) / Sigma(lambda^(T-1-i))
            reward[t] = weight[t] * episode_reward

        Distinguishes truncation (timeout, bootstrap from V) from true terminal
        (dissociation, zero bootstrap) so GAE flags are set correctly.

        Args:
            episode_reward: Total episode-end reward (Ea + dissociation)
            valid_step_indices: Buffer indices of all valid steps
            ea_component: EA component for routing
            dissoc_component: Dissociation component for routing
            lambda_decay: Credit decay factor (0.95 = 5% decay per step back)
            is_truncated: Episode ended on timeout (bootstrap from V)
            is_true_terminal: Episode ended on terminal state (dissociation)
        """
        if not valid_step_indices or episode_reward == 0:
            return

        T = len(valid_step_indices)

        # Exponential decay from the end (last step weight 1.0, then lambda, ...),
        # normalized to sum to 1.0 so total reward is preserved.
        weights = np.array([lambda_decay ** (T - 1 - t) for t in range(T)])
        weights = weights / weights.sum()

        # GAE flags on the last step: truncated -> (terminal=False, truncated=True,
        # bootstrap from V); true terminal -> (terminal=True, truncated=False, zero
        # bootstrap).

        self._log('CREDIT', "Reward distribution setup",
                  is_truncated=is_truncated,
                  is_true_terminal=is_true_terminal,
                  num_valid_steps=T)

        # Distribute rewards
        for t, buffer_idx in enumerate(valid_step_indices):
            step_fraction = weights[t] * episode_reward
            step_ea = weights[t] * ea_component
            step_dissoc = weights[t] * dissoc_component

            # Only the last step carries terminal flags.
            is_last_step = (t == T - 1)

            if is_last_step:
                if is_truncated:
                    step_is_terminal = False
                    step_is_truncated = True
                    last_step_type = "TRUNCATED (bootstrap from V)"
                elif is_true_terminal:
                    step_is_terminal = True
                    step_is_truncated = False
                    last_step_type = "TRUE TERMINAL (bootstrap=0)"
                else:
                    step_is_terminal = False
                    step_is_truncated = False
                    last_step_type = "Unknown (treating as non-terminal)"

                self._log('CREDIT', f"Last step classification",
                          buffer_idx=buffer_idx,
                          type=last_step_type)
            else:
                step_is_terminal = False
                step_is_truncated = False

            update_kwargs = {
                'index': buffer_idx,
                'reward_total': step_fraction,
                'done': step_is_terminal,
                'is_truncated': step_is_truncated,
                'ea_reward': step_ea,
                'dissociation_reward': step_dissoc,
                'additive': True
            }

            # Pass truncation bootstrap value only for the last truncated step.
            if is_last_step and step_is_truncated and truncation_value is not None:
                update_kwargs['truncation_value'] = truncation_value

            self.agent.update_rewards(**update_kwargs)

        self._log('CREDIT', f"Reward distribution complete",
                  total_reward=f"{episode_reward:.4f}",
                  num_steps=T,
                  last_5_weights=str(weights[-5:].round(3)),
                  last_5_rewards=str((weights[-5:] * episode_reward).round(4)))

