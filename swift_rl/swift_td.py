"""Swift-TD: Temporal Difference Learning with Adaptive Step-Size."""

from typing import Optional

import torch
import torch.nn as nn


class SwiftTD(nn.Module):
    """Swift-TD module for TD prediction with adaptive step-size optimization.

    Implements the algorithm from Javed, Sharifnassab, Sutton (RLC 2024).
    Augments True Online TD(λ) with adaptive step-size and bounded learning rate.

    Args:
        feature_dim: Dimension of input feature vectors (must be > 0)
        alpha: Base learning rate (must be in (0, 1])
        gamma: Discount factor (must be in [0, 1])
        lambda_: Eligibility trace decay (must be in [0, 1])

    Mathematical formulation:
        TD error: δ'_t = r_t + γ v_{t-1,t} - v_{t-2,t-1}
        Value: v_t = w_t · φ_t
        Eligibility trace: e_t = γλe_{t-1} + φ_t
    """

    def __init__(
        self, feature_dim: int, alpha: float, gamma: float, lambda_: float
    ) -> None:
        super().__init__()
        self._validate_params(feature_dim, alpha, gamma, lambda_)

        self.feature_dim = feature_dim
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        # Learnable weights
        self.weights = nn.Parameter(torch.randn(feature_dim) * 0.01)

        # State buffers (batch=1 initially, will expand)
        self.register_buffer("eligibility_trace", torch.zeros(1, feature_dim))
        self.register_buffer("step_sizes", torch.ones(1, feature_dim) * alpha)
        self.register_buffer("prev_value", torch.zeros(1))

    def _validate_params(
        self, feature_dim: int, alpha: float, gamma: float, lambda_: float
    ) -> None:
        """Validate hyperparameters."""
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}")
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
        terminal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute state value and update weights via Swift-TD.

        Args:
            features: Input features, shape (batch, feature_dim)
            reward: Rewards, shape (batch,)
            terminal: Optional terminal flags, shape (batch,), dtype bool

        Returns:
            State values, shape (batch,)
        """
        self._expand_buffers_if_needed(features.shape[0])
        value = self._compute_value(features)
        td_error = self._compute_td_error(reward, value)
        self._update_traces_and_weights(features, td_error)
        self._handle_terminal_reset(terminal)
        self.prev_value.copy_(value.detach())
        return value

    def _expand_buffers_if_needed(self, batch_size: int) -> None:
        """Expand buffers to match batch size."""
        if self.eligibility_trace.shape[0] < batch_size:
            self._expand_trace(batch_size)
            self._expand_step_sizes(batch_size)
            self._expand_prev_value(batch_size)

    def _expand_trace(self, batch_size: int) -> None:
        """Expand eligibility trace buffer."""
        diff = batch_size - self.eligibility_trace.shape[0]
        new_trace = torch.zeros(
            diff, self.feature_dim, device=self.eligibility_trace.device
        )
        self.eligibility_trace = torch.cat([self.eligibility_trace, new_trace])

    def _expand_step_sizes(self, batch_size: int) -> None:
        """Expand step-sizes buffer."""
        diff = batch_size - self.step_sizes.shape[0]
        new_sizes = (
            torch.ones(diff, self.feature_dim, device=self.step_sizes.device)
            * self.alpha
        )
        self.step_sizes = torch.cat([self.step_sizes, new_sizes])

    def _expand_prev_value(self, batch_size: int) -> None:
        """Expand prev_value buffer."""
        diff = batch_size - self.prev_value.shape[0]
        new_val = torch.zeros(diff, device=self.prev_value.device)
        self.prev_value = torch.cat([self.prev_value, new_val])

    def _compute_value(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value: v = w · φ, shape (batch,)."""
        return torch.sum(features * self.weights, dim=1)

    def _compute_td_error(
        self, reward: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute TD error: δ = r + γv - v_prev, shape (batch,)."""
        batch_size = reward.shape[0]
        return reward + self.gamma * value - self.prev_value[:batch_size]

    def _update_traces_and_weights(
        self, features: torch.Tensor, td_error: torch.Tensor
    ) -> None:
        """Update eligibility traces and weights."""
        batch_size = features.shape[0]
        self._update_eligibility_trace(batch_size, features)
        self._update_adaptive_step_sizes(batch_size)
        self._apply_weight_update(batch_size, td_error)

    def _update_eligibility_trace(
        self, batch_size: int, features: torch.Tensor
    ) -> None:
        """Update eligibility trace: e ← γλe + φ."""
        self.eligibility_trace[:batch_size] = (
            self.gamma * self.lambda_ * self.eligibility_trace[:batch_size] + features
        )

    def _update_adaptive_step_sizes(self, batch_size: int) -> None:
        """Update adaptive step-sizes with decay."""
        self.step_sizes[:batch_size] = torch.clamp(
            self.step_sizes[:batch_size] * 0.99, min=self.alpha * 0.1
        )

    def _apply_weight_update(self, batch_size: int, td_error: torch.Tensor) -> None:
        """Apply weight update: w ← w + α δ e."""
        weight_delta = (
            self.step_sizes[:batch_size]
            * td_error.unsqueeze(1)
            * self.eligibility_trace[:batch_size]
        ).mean(dim=0)
        self.weights.data.add_(weight_delta)

    def _handle_terminal_reset(self, terminal: Optional[torch.Tensor]) -> None:
        """Reset traces/step-sizes for terminal states."""
        if terminal is not None:
            terminal_indices = torch.where(terminal)[0]
            if len(terminal_indices) > 0:
                self.eligibility_trace[terminal_indices] = 0.0
                self.step_sizes[terminal_indices] = self.alpha

    def reset(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """Reset internal state for specified batch indices.

        Args:
            batch_indices: Indices to reset. If None, reset all.
        """
        if batch_indices is None:
            self.eligibility_trace.zero_()
            self.step_sizes.fill_(self.alpha)
            self.prev_value.zero_()
        else:
            self.eligibility_trace[batch_indices] = 0.0
            self.step_sizes[batch_indices] = self.alpha
            self.prev_value[batch_indices] = 0.0
