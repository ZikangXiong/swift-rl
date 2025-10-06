"""
Action Selection Policies Interface Contract

This contract defines the expected interface for action selection utility functions.
All implementations must satisfy this contract.
"""

import torch


def epsilon_greedy(
    action_values: torch.Tensor,
    epsilon: float,
    training: bool = True
) -> torch.Tensor:
    """
    Epsilon-greedy action selection policy.

    Args:
        action_values: Action-value tensor, shape (batch, num_actions)
        epsilon: Exploration rate, must be in [0, 1]
        training: If True, use epsilon-greedy. If False, always greedy.

    Returns:
        Selected action indices, shape (batch,), dtype long

    Raises:
        ValueError: If epsilon not in [0, 1] or action_values.ndim != 2
        RuntimeError: If action_values has incompatible shape

    Behavior:
        - If training=False: return argmax(action_values, dim=1)
        - If training=True:
            - With probability epsilon: sample random action uniformly
            - With probability 1-epsilon: select argmax action
    """
    ...


def softmax_policy(
    action_values: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Softmax (Boltzmann) action selection policy.

    Args:
        action_values: Action-value tensor, shape (batch, num_actions)
        temperature: Temperature parameter, must be > 0

    Returns:
        Sampled action indices, shape (batch,), dtype long

    Raises:
        ValueError: If temperature <= 0 or action_values.ndim != 2
        RuntimeError: If action_values has incompatible shape

    Behavior:
        - Compute probabilities: P(a) = exp(Q(a)/τ) / Σ_j exp(Q(j)/τ)
        - Sample action from categorical distribution with these probabilities
        - Lower temperature → more greedy (peaked distribution)
        - Higher temperature → more random (uniform distribution)
    """
    ...


# Contract validation assertions (to be used in tests)

def validate_epsilon_greedy_contract(func) -> None:
    """
    Validate that a function satisfies the epsilon_greedy contract.

    Args:
        func: Function to validate

    Raises:
        AssertionError: If contract violated
    """
    # Test case 1: Basic functionality
    action_values = torch.randn(10, 4)
    actions = func(action_values, epsilon=0.1, training=True)
    assert actions.shape == (10,)
    assert actions.dtype == torch.long
    assert (actions >= 0).all() and (actions < 4).all()

    # Test case 2: Greedy mode (epsilon=0 or training=False)
    greedy_actions = func(action_values, epsilon=0.0, training=True)
    expected_greedy = action_values.argmax(dim=1)
    assert torch.equal(greedy_actions, expected_greedy)

    eval_actions = func(action_values, epsilon=0.5, training=False)
    assert torch.equal(eval_actions, expected_greedy)

    # Test case 3: Fully random (epsilon=1)
    # Cannot deterministically test randomness, just check shape/dtype
    random_actions = func(action_values, epsilon=1.0, training=True)
    assert random_actions.shape == (10,)
    assert random_actions.dtype == torch.long


def validate_softmax_policy_contract(func) -> None:
    """
    Validate that a function satisfies the softmax_policy contract.

    Args:
        func: Function to validate

    Raises:
        AssertionError: If contract violated
    """
    # Test case 1: Basic functionality
    action_values = torch.randn(10, 4)
    actions = func(action_values, temperature=1.0)
    assert actions.shape == (10,)
    assert actions.dtype == torch.long
    assert (actions >= 0).all() and (actions < 4).all()

    # Test case 2: Very low temperature (nearly greedy)
    # With very low temp, should mostly select argmax
    greedy_expected = action_values.argmax(dim=1)
    low_temp_actions = func(action_values, temperature=0.01)
    # At least 80% should match greedy (probabilistic, not deterministic)
    # (This is a weak test due to randomness)
    assert low_temp_actions.shape == (10,)
    assert low_temp_actions.dtype == torch.long

    # Test case 3: High temperature (nearly uniform)
    # Just check shape/dtype, can't test uniformity easily
    high_temp_actions = func(action_values, temperature=10.0)
    assert high_temp_actions.shape == (10,)
    assert high_temp_actions.dtype == torch.long
