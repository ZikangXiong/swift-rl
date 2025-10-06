"""Swift-SARSA: On-Policy Control with Adaptive Step-Size."""

from typing import Optional

import torch
import torch.nn as nn


class SwiftSARSA(nn.Module):
    """Swift-SARSA module for on-policy control with adaptive step-size.

    Implements the algorithm from Javed, Sutton (arXiv 2025).
    Extends Swift-TD to control by maintaining separate value functions per action.

    Args:
        feature_dim: Dimension of input feature vectors (must be > 0)
        num_actions: Number of discrete actions (must be > 1)
        alpha: Base learning rate (must be in (0, 1])
        gamma: Discount factor (must be in [0, 1])
        lambda_: Eligibility trace decay (must be in [0, 1])

    Mathematical formulation:
        Action-values: Q(s, a) = w_a · φ(s)
        TD error: δ_t = r_t + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        Eligibility trace (per action): e_t^a ← γλe_{t-1}^a + φ_t
    """

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        alpha: float,
        gamma: float,
        lambda_: float,
    ) -> None:
        super().__init__()
        self._validate_params(feature_dim, num_actions, alpha, gamma, lambda_)

        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        # Learnable weights: one vector per action
        self.weights = nn.Parameter(torch.randn(num_actions, feature_dim) * 0.01)

        # State buffers (batch=1 initially, will expand)
        self.register_buffer(
            "eligibility_traces", torch.zeros(1, num_actions, feature_dim)
        )
        self.register_buffer(
            "step_sizes", torch.ones(1, num_actions, feature_dim) * alpha
        )
        self.register_buffer("prev_action_values", torch.zeros(1, num_actions))

    def _validate_params(
        self,
        feature_dim: int,
        num_actions: int,
        alpha: float,
        gamma: float,
        lambda_: float,
    ) -> None:
        """Validate hyperparameters."""
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}")
        if num_actions <= 1:
            raise ValueError(f"num_actions must be > 1, got {num_actions}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0 <= lambda_ <= 1):
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    def forward(
        self,
        features: torch.Tensor,
        reward: torch.Tensor,
        action: torch.Tensor,
        terminal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute action-values and update weights via Swift-SARSA.

        Args:
            features: Input features, shape (batch, feature_dim)
            reward: Rewards, shape (batch,)
            action: Actions taken, shape (batch,), dtype long
            terminal: Optional terminal flags, shape (batch,), dtype bool

        Returns:
            Action-values, shape (batch, num_actions)
        """
        self._expand_buffers_if_needed(features.shape[0])
        action_values = self._compute_action_values(features)
        td_error = self._compute_td_error(reward, action_values, action)
        self._update_traces_and_weights(features, action, td_error)
        self._handle_terminal_reset(terminal)
        self.prev_action_values.copy_(action_values.detach())
        return action_values

    def _expand_buffers_if_needed(self, batch_size: int) -> None:
        """Expand buffers to match batch size."""
        if self.eligibility_traces.shape[0] < batch_size:
            self._expand_traces(batch_size)
            self._expand_step_sizes(batch_size)
            self._expand_prev_values(batch_size)

    def _expand_traces(self, batch_size: int) -> None:
        """Expand eligibility traces buffer."""
        diff = batch_size - self.eligibility_traces.shape[0]
        new_traces = torch.zeros(
            diff,
            self.num_actions,
            self.feature_dim,
            device=self.eligibility_traces.device,
        )
        self.eligibility_traces = torch.cat([self.eligibility_traces, new_traces])

    def _expand_step_sizes(self, batch_size: int) -> None:
        """Expand step-sizes buffer."""
        diff = batch_size - self.step_sizes.shape[0]
        new_sizes = (
            torch.ones(
                diff, self.num_actions, self.feature_dim, device=self.step_sizes.device
            )
            * self.alpha
        )
        self.step_sizes = torch.cat([self.step_sizes, new_sizes])

    def _expand_prev_values(self, batch_size: int) -> None:
        """Expand prev_action_values buffer."""
        diff = batch_size - self.prev_action_values.shape[0]
        new_val = torch.zeros(
            diff, self.num_actions, device=self.prev_action_values.device
        )
        self.prev_action_values = torch.cat([self.prev_action_values, new_val])

    def _compute_action_values(self, features: torch.Tensor) -> torch.Tensor:
        """Compute action-values: Q(s, a) = w_a · φ, shape (batch, num_actions)."""
        return torch.matmul(features, self.weights.t())

    def _compute_td_error(
        self, reward: torch.Tensor, action_values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute TD error: δ = r + γQ(s',a') - Q(s,a), shape (batch,)."""
        batch_size = reward.shape[0]
        current_q = action_values.gather(1, action.unsqueeze(1)).squeeze(1)
        prev_q = (
            self.prev_action_values[:batch_size]
            .gather(1, action.unsqueeze(1))
            .squeeze(1)
        )
        return reward + self.gamma * current_q - prev_q

    def _update_traces_and_weights(
        self, features: torch.Tensor, action: torch.Tensor, td_error: torch.Tensor
    ) -> None:
        """Update eligibility traces and weights for selected actions."""
        batch_size = features.shape[0]
        self._update_eligibility_traces(batch_size, features, action)
        self._update_adaptive_step_sizes(batch_size, action)
        self._apply_weight_update(batch_size, action, td_error)

    def _update_eligibility_traces(
        self, batch_size: int, features: torch.Tensor, action: torch.Tensor
    ) -> None:
        """Update eligibility traces: e^a ← γλe^a + φ for selected actions."""
        # Decay all traces
        self.eligibility_traces[:batch_size] *= self.gamma * self.lambda_

        # Add features to selected action's trace
        for i in range(batch_size):
            self.eligibility_traces[i, action[i]] += features[i]

    def _update_adaptive_step_sizes(
        self, batch_size: int, action: torch.Tensor
    ) -> None:
        """Update adaptive step-sizes with decay for selected actions."""
        for i in range(batch_size):
            self.step_sizes[i, action[i]] = torch.clamp(
                self.step_sizes[i, action[i]] * 0.99, min=self.alpha * 0.1
            )

    def _apply_weight_update(
        self, batch_size: int, action: torch.Tensor, td_error: torch.Tensor
    ) -> None:
        """Apply weight update: w^a ← w^a + α δ e^a for selected actions."""
        for a in range(self.num_actions):
            mask = action == a
            if mask.any():
                indices = torch.where(mask)[0]
                weight_delta = (
                    self.step_sizes[indices, a]
                    * td_error[indices].unsqueeze(1)
                    * self.eligibility_traces[indices, a]
                ).mean(dim=0)
                self.weights.data[a].add_(weight_delta)

    def _handle_terminal_reset(self, terminal: Optional[torch.Tensor]) -> None:
        """Reset traces/step-sizes for terminal states."""
        if terminal is not None:
            terminal_indices = torch.where(terminal)[0]
            if len(terminal_indices) > 0:
                self.eligibility_traces[terminal_indices] = 0.0
                self.step_sizes[terminal_indices] = self.alpha

    def reset(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """Reset internal state for specified batch indices.

        Args:
            batch_indices: Indices to reset. If None, reset all.
        """
        if batch_indices is None:
            self.eligibility_traces.zero_()
            self.step_sizes.fill_(self.alpha)
            self.prev_action_values.zero_()
        else:
            self.eligibility_traces[batch_indices] = 0.0
            self.step_sizes[batch_indices] = self.alpha
            self.prev_action_values[batch_indices] = 0.0
