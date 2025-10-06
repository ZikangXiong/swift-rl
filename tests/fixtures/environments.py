"""Test fixtures: simple environments for algorithm testing."""

from typing import Tuple

import torch


def linear_prediction_task(
    batch_size: int, feature_dim: int, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple linear prediction task.

    Args:
        batch_size: Number of samples
        feature_dim: Dimension of feature vectors
        device: Device to create tensors on

    Returns:
        Tuple of (features, target_values) where target = sum of features
    """
    features = torch.randn(batch_size, feature_dim, device=device)
    target_values = features.sum(dim=1)
    return features, target_values


def tabular_mdp_step(
    state: int, action: int, num_states: int = 5
) -> Tuple[int, float, bool]:
    """Simple deterministic tabular MDP for testing.

    Args:
        state: Current state index [0, num_states)
        action: Action taken (0=left, 1=right)
        num_states: Total number of states

    Returns:
        Tuple of (next_state, reward, terminal)

    Environment:
        - Linear chain of states: 0 -> 1 -> 2 -> 3 -> 4
        - Action 0 (left): move towards 0
        - Action 1 (right): move towards terminal state
        - Reward: +1 at terminal state (num_states - 1), 0 elsewhere
    """
    if action == 0:  # Move left
        next_state = max(0, state - 1)
    else:  # Move right (action == 1)
        next_state = min(num_states - 1, state + 1)

    terminal = next_state == (num_states - 1)
    reward = 1.0 if terminal else 0.0

    return next_state, reward, terminal


def create_state_features(
    state: int, num_states: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Create one-hot features for tabular state.

    Args:
        state: State index
        num_states: Total number of states
        device: Device to create tensor on

    Returns:
        One-hot tensor, shape (num_states,)
    """
    features = torch.zeros(num_states, device=device)
    features[state] = 1.0
    return features
