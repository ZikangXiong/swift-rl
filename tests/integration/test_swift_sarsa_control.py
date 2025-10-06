"""Integration tests for SwiftSARSA control."""

import pytest
import torch

from swift_rl import SwiftSARSA
from swift_rl.policies import epsilon_greedy
from tests.fixtures import create_state_features, tabular_mdp_step


def test_swiftsarsa_tabular_control():
    """Test SwiftSARSA learns control policy in tabular MDP."""
    # Simple chain MDP: 0 -> 1 -> 2 -> 3 -> 4 (terminal, reward=1)
    num_states = 5
    num_actions = 2  # 0=left, 1=right
    swift_sarsa = SwiftSARSA(
        feature_dim=num_states,
        num_actions=num_actions,
        alpha=0.3,
        gamma=0.9,
        lambda_=0.5,
    )

    # Store initial weights
    initial_weights = swift_sarsa.weights.data.clone()

    # Run episodes
    num_episodes = 200
    epsilon = 0.2

    for episode in range(num_episodes):
        state = 0  # Start at state 0
        features = create_state_features(state, num_states).unsqueeze(0)

        # Select initial action
        with torch.no_grad():
            action_values = swift_sarsa(
                features, torch.zeros(1), torch.zeros(1, dtype=torch.long)
            )
            action = epsilon_greedy(action_values, epsilon=epsilon, training=True)

        # Episode trajectory
        for step in range(10):
            # Take action in environment
            next_state, reward, terminal = tabular_mdp_step(
                state, action.item(), num_states=num_states
            )

            # Get next features
            next_features = create_state_features(next_state, num_states).unsqueeze(0)
            reward_tensor = torch.tensor([reward])
            terminal_tensor = torch.tensor([terminal])

            # Update with current action
            next_action_values = swift_sarsa(
                next_features, reward_tensor, action, terminal_tensor
            )

            if terminal:
                swift_sarsa.reset()
                break

            # Select next action
            with torch.no_grad():
                action = epsilon_greedy(
                    next_action_values, epsilon=epsilon, training=True
                )

            state = next_state
            features = next_features

    # Check that weights have changed (learning occurred)
    weight_change = (swift_sarsa.weights.data - initial_weights).abs().sum().item()
    assert weight_change > 0.1, "Weights should have changed during learning"

    # Check that action values have been learned (non-zero)
    test_features = create_state_features(0, num_states).unsqueeze(0)
    with torch.no_grad():
        action_values = swift_sarsa(
            test_features, torch.zeros(1), torch.zeros(1, dtype=torch.long)
        )

    # Action values should be non-trivial (learning occurred)
    assert (
        action_values.abs().sum() > 0.01
    ), "Should have learned non-zero action values"


def test_swiftsarsa_action_value_learning():
    """Test SwiftSARSA distinguishes between action values."""
    num_states = 3
    num_actions = 3
    swift_sarsa = SwiftSARSA(
        feature_dim=num_states,
        num_actions=num_actions,
        alpha=0.2,
        gamma=0.9,
        lambda_=0.7,
    )

    # Train with different rewards for different actions
    for episode in range(100):
        state = torch.randint(0, num_states, (1,))
        features = create_state_features(state.item(), num_states).unsqueeze(0)

        # Action 2 always gives best reward, action 0 worst
        action = torch.randint(0, num_actions, (1,))
        reward = torch.tensor(
            [2.0 if action.item() == 2 else 0.5 if action.item() == 1 else 0.0]
        )

        swift_sarsa(features, reward, action)

    # Check that learned action values reflect reward structure
    test_features = create_state_features(0, num_states).unsqueeze(0)
    with torch.no_grad():
        action_values = swift_sarsa(
            test_features, torch.zeros(1), torch.zeros(1, dtype=torch.long)
        )

    # Action 2 should have highest value
    assert (
        action_values[0, 2] == action_values[0].max()
    ), "Action 2 should have highest value"


def test_swiftsarsa_batch_learning():
    """Test SwiftSARSA handles batch of environments correctly."""
    batch_size = 8
    num_states = 4
    num_actions = 2
    swift_sarsa = SwiftSARSA(
        feature_dim=num_states,
        num_actions=num_actions,
        alpha=0.1,
        gamma=0.9,
        lambda_=0.8,
    )

    # Create batch of different states
    states = torch.randint(0, num_states, (batch_size,))
    features = torch.stack(
        [create_state_features(s.item(), num_states) for s in states]
    )

    # All take same action initially
    actions = torch.zeros(batch_size, dtype=torch.long)
    rewards = torch.randn(batch_size)

    # Should handle batch without error
    action_values = swift_sarsa(features, rewards, actions)

    assert action_values.shape == (batch_size, num_actions)
    assert swift_sarsa.eligibility_traces.shape[0] >= batch_size


def test_swiftsarsa_terminal_state_handling():
    """Test SwiftSARSA correctly handles terminal states in batch."""
    batch_size = 6
    num_states = 3
    num_actions = 2
    swift_sarsa = SwiftSARSA(
        feature_dim=num_states,
        num_actions=num_actions,
        alpha=0.1,
        gamma=0.9,
        lambda_=0.8,
    )

    # Run some updates to build up traces
    for _ in range(5):
        features = torch.randn(batch_size, num_states)
        rewards = torch.randn(batch_size)
        actions = torch.randint(0, num_actions, (batch_size,))
        swift_sarsa(features, rewards, actions)

    # Now mark some as terminal
    features = torch.randn(batch_size, num_states)
    rewards = torch.randn(batch_size)
    actions = torch.randint(0, num_actions, (batch_size,))
    terminal = torch.tensor([False, True, False, True, False, True])

    swift_sarsa(features, rewards, actions, terminal)

    # Terminal states should have reset traces
    assert swift_sarsa.eligibility_traces[1].abs().sum() == 0.0
    assert swift_sarsa.eligibility_traces[3].abs().sum() == 0.0
    assert swift_sarsa.eligibility_traces[5].abs().sum() == 0.0

    # Non-terminal states should have non-zero traces
    assert swift_sarsa.eligibility_traces[0].abs().sum() > 0.0
    assert swift_sarsa.eligibility_traces[2].abs().sum() > 0.0
    assert swift_sarsa.eligibility_traces[4].abs().sum() > 0.0
