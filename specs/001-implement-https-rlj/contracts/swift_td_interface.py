"""
SwiftTD Module Interface Contract

This contract defines the expected interface for the SwiftTD module.
All implementations must satisfy this contract.
"""

from typing import Optional
import torch
import torch.nn as nn


class SwiftTDInterface(nn.Module):
    """Interface contract for Swift-TD temporal difference learning module."""

    def __init__(
        self,
        feature_dim: int,
        alpha: float,
        gamma: float,
        lambda_: float
    ) -> None:
        """
        Initialize Swift-TD module.

        Args:
            feature_dim: Dimension of input feature vectors (must be > 0)
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
        reward: torch.Tensor,
        terminal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute state value and perform Swift-TD update.

        Args:
            features: Input feature tensor, shape (batch, feature_dim)
            reward: Reward tensor, shape (batch,)
            terminal: Optional terminal flag tensor, shape (batch,), dtype bool

        Returns:
            State value tensor, shape (batch,)

        Raises:
            ValueError: If input shapes are incompatible
            RuntimeError: If device mismatch between module and inputs
        """
        ...

    def reset(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Reset internal state (eligibility traces, step-sizes, prev_value).

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
        Learnable weight vector for value function.

        Returns:
            Weight parameter, shape (feature_dim,)
        """
        ...

    @property
    def eligibility_trace(self) -> torch.Tensor:
        """
        Current eligibility trace state.

        Returns:
            Eligibility trace buffer, shape (batch, feature_dim)
        """
        ...

    @property
    def step_sizes(self) -> torch.Tensor:
        """
        Current adaptive step-size state.

        Returns:
            Step-size buffer, shape (batch, feature_dim)
        """
        ...


# Contract validation assertions (to be used in tests)

def validate_swift_td_contract(module: SwiftTDInterface) -> None:
    """
    Validate that a module satisfies the SwiftTD interface contract.

    Args:
        module: Module instance to validate

    Raises:
        AssertionError: If contract violated
    """
    # Check attributes exist
    assert hasattr(module, 'feature_dim')
    assert hasattr(module, 'alpha')
    assert hasattr(module, 'gamma')
    assert hasattr(module, 'lambda_')
    assert hasattr(module, 'weights')
    assert hasattr(module, 'eligibility_trace')
    assert hasattr(module, 'step_sizes')

    # Check methods exist
    assert callable(getattr(module, 'forward', None))
    assert callable(getattr(module, 'reset', None))

    # Check types
    assert isinstance(module.weights, nn.Parameter)
    assert isinstance(module.eligibility_trace, torch.Tensor)
    assert isinstance(module.step_sizes, torch.Tensor)

    # Check shapes
    feature_dim = module.feature_dim
    assert module.weights.shape == (feature_dim,)
    assert module.eligibility_trace.ndim == 2
    assert module.step_sizes.ndim == 2
    assert module.eligibility_trace.shape[1] == feature_dim
    assert module.step_sizes.shape[1] == feature_dim
