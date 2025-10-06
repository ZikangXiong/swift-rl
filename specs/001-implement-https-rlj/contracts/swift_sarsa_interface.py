"""
SwiftSARSA Module Interface Contract

This contract defines the expected interface for the SwiftSARSA module.
All implementations must satisfy this contract.
"""

from typing import Optional
import torch
import torch.nn as nn


class SwiftSARSAInterface(nn.Module):
    """Interface contract for Swift-SARSA on-policy control module."""

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        alpha: float,
        gamma: float,
        lambda_: float
    ) -> None:
        """
        Initialize Swift-SARSA module.

        Args:
            feature_dim: Dimension of input feature vectors (must be > 0)
            num_actions: Number of discrete actions (must be > 1)
            alpha: Learning rate (must be in (0, 1])
            gamma: Discount factor (must be in [0, 1])
            lambda_: Eligibility trace decay (must be in [0, 1])

        Raises:
            ValueError: If hyperparameters out of valid ranges
        """
        ...

    def forward(
        self,
        features: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        terminal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute action-values and perform Swift-SARSA update for selected action.

        Args:
            features: Input feature tensor, shape (batch, feature_dim)
            reward: Optional reward tensor, shape (batch,)
            action: Optional action tensor, shape (batch,), dtype long
            terminal: Optional terminal flag tensor, shape (batch,), dtype bool

        Returns:
            Action-value tensor, shape (batch, num_actions)

        Raises:
            ValueError: If input shapes are incompatible or action out of range
            RuntimeError: If device mismatch between module and inputs

        Note:
            If reward and action are provided, Swift update is performed for
            the selected action. Otherwise, only forward pass (no update).
        """
        ...

    def reset(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Reset internal state (eligibility traces, step-sizes, prev_action_values).

        Args:
            batch_indices: Optional indices to reset. If None, reset all.
                          Shape (num_indices,), dtype long

        Note:
            This is called manually or automatically when terminal flag is True.
        """
        ...

    @property
    def weights(self) -> nn.Parameter:
        """
        Learnable weight matrix for action-value functions.

        Returns:
            Weight parameter, shape (num_actions, feature_dim)
        """
        ...

    @property
    def eligibility_traces(self) -> torch.Tensor:
        """
        Current eligibility trace state for all actions.

        Returns:
            Eligibility trace buffer, shape (batch, num_actions, feature_dim)
        """
        ...

    @property
    def step_sizes(self) -> torch.Tensor:
        """
        Current adaptive step-size state for all actions.

        Returns:
            Step-size buffer, shape (batch, num_actions, feature_dim)
        """
        ...


# Contract validation assertions (to be used in tests)

def validate_swift_sarsa_contract(module: SwiftSARSAInterface) -> None:
    """
    Validate that a module satisfies the SwiftSARSA interface contract.

    Args:
        module: Module instance to validate

    Raises:
        AssertionError: If contract violated
    """
    # Check attributes exist
    assert hasattr(module, 'feature_dim')
    assert hasattr(module, 'num_actions')
    assert hasattr(module, 'alpha')
    assert hasattr(module, 'gamma')
    assert hasattr(module, 'lambda_')
    assert hasattr(module, 'weights')
    assert hasattr(module, 'eligibility_traces')
    assert hasattr(module, 'step_sizes')

    # Check methods exist
    assert callable(getattr(module, 'forward', None))
    assert callable(getattr(module, 'reset', None))

    # Check types
    assert isinstance(module.weights, nn.Parameter)
    assert isinstance(module.eligibility_traces, torch.Tensor)
    assert isinstance(module.step_sizes, torch.Tensor)

    # Check shapes
    feature_dim = module.feature_dim
    num_actions = module.num_actions
    assert module.weights.shape == (num_actions, feature_dim)
    assert module.eligibility_traces.ndim == 3
    assert module.step_sizes.ndim == 3
    assert module.eligibility_traces.shape[1:] == (num_actions, feature_dim)
    assert module.step_sizes.shape[1:] == (num_actions, feature_dim)
