"""Action selection policies for reinforcement learning."""

import torch


def epsilon_greedy(
    action_values: torch.Tensor, epsilon: float, training: bool = True
) -> torch.Tensor:
    """Epsilon-greedy action selection policy.

    Args:
        action_values: Action-value tensor, shape (batch, num_actions)
        epsilon: Exploration rate in [0, 1]
        training: If True, use epsilon-greedy. If False, always greedy.

    Returns:
        Selected action indices, shape (batch,), dtype long
    """
    _validate_epsilon_greedy_inputs(action_values, epsilon)

    if not training or epsilon == 0:
        return action_values.argmax(dim=1)

    return _epsilon_greedy_select(action_values, epsilon)


def _validate_epsilon_greedy_inputs(
    action_values: torch.Tensor, epsilon: float
) -> None:
    """Validate epsilon-greedy inputs."""
    if action_values.ndim != 2:
        raise ValueError(f"action_values must be 2D, got shape {action_values.shape}")
    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")


def _epsilon_greedy_select(action_values: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Select actions with epsilon-greedy strategy."""
    batch_size, num_actions = action_values.shape
    explore_mask = torch.rand(batch_size, device=action_values.device) < epsilon
    greedy = action_values.argmax(dim=1)
    random = torch.randint(0, num_actions, (batch_size,), device=action_values.device)
    return torch.where(explore_mask, random, greedy)


def softmax_policy(action_values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax (Boltzmann) action selection policy.

    Args:
        action_values: Action-value tensor, shape (batch, num_actions)
        temperature: Temperature parameter, must be > 0

    Returns:
        Sampled action indices, shape (batch,), dtype long

    Behavior:
        - Compute probabilities: P(a) = exp(Q(a)/τ) / Σ_j exp(Q(j)/τ)
        - Sample from categorical distribution
        - Lower temp → more greedy, Higher temp → more random
    """
    if action_values.ndim != 2:
        raise ValueError(f"action_values must be 2D, got shape {action_values.shape}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Compute softmax probabilities
    probs = torch.softmax(action_values / temperature, dim=1)

    # Sample from categorical distribution
    return torch.multinomial(probs, num_samples=1).squeeze(1)
