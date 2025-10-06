"""Standalone SwiftSARSA example for on-policy control."""

import torch

from swift_rl import SwiftSARSA
from swift_rl.policies import epsilon_greedy


def main():
    """Demonstrate SwiftSARSA on a simple control task."""
    # Setup
    batch_size = 8
    feature_dim = 16
    num_actions = 4
    num_episodes = 100

    # Create SwiftSARSA module
    swift_sarsa = SwiftSARSA(
        feature_dim=feature_dim,
        num_actions=num_actions,
        alpha=0.05,  # Learning rate
        gamma=0.95,  # Discount factor
        lambda_=0.8,  # Eligibility trace decay
    )

    print(f"SwiftSARSA initialized:")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Num actions: {num_actions}")
    print(f"  Alpha: {swift_sarsa.alpha}")
    print(f"  Gamma: {swift_sarsa.gamma}")
    print(f"  Lambda: {swift_sarsa.lambda_}")
    print()

    # Simple control task: learn to maximize reward based on features
    print("Training on control task...")
    epsilon = 0.2  # Exploration rate

    for episode in range(num_episodes):
        # Reset for new episode
        swift_sarsa.reset()

        # Generate random initial features
        features = torch.randn(batch_size, feature_dim)

        # Select initial action using epsilon-greedy
        with torch.no_grad():
            action_values = swift_sarsa(
                features,
                torch.zeros(batch_size),
                torch.zeros(batch_size, dtype=torch.long),
            )
            action = epsilon_greedy(action_values, epsilon=epsilon, training=True)

        episode_reward = 0.0

        # Episode steps
        for step in range(20):
            # Generate next features
            next_features = torch.randn(batch_size, feature_dim)

            # Simulate reward (higher for certain feature patterns)
            reward = (next_features.sum(dim=1) > 0).float()

            # Check if terminal (simplified: 10% chance per step)
            terminal = torch.rand(batch_size) < 0.1

            # Update with current action
            next_action_values = swift_sarsa(next_features, reward, action, terminal)

            # Select next action
            with torch.no_grad():
                action = epsilon_greedy(
                    next_action_values, epsilon=epsilon, training=True
                )

            episode_reward += reward.mean().item()

            features = next_features

            # Reset terminated environments
            if terminal.any():
                swift_sarsa.reset(torch.where(terminal)[0])

        # Print progress
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}: Avg reward = {episode_reward / 20:.4f}")

    print("\nTraining complete!")
    print(f"Learned {num_actions} action-value functions")
    print(f"Final weights shape: {swift_sarsa.weights.shape}")


if __name__ == "__main__":
    main()
