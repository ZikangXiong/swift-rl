"""Neural network integration example with SwiftSARSA as final layer."""

import torch
import torch.nn as nn

from swift_rl import SwiftSARSA
from swift_rl.policies import softmax_policy


def main():
    """Demonstrate SwiftSARSA as final layer of neural network."""
    # Setup
    state_dim = 64
    feature_dim = 32
    num_actions = 4
    batch_size = 32
    num_episodes = 50

    print("Creating neural network with SwiftSARSA final layer...")

    # Feature extraction network
    feature_network = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, feature_dim),
    )

    # SwiftSARSA as final layer
    swift_sarsa_layer = SwiftSARSA(
        feature_dim=feature_dim,
        num_actions=num_actions,
        alpha=0.02,
        gamma=0.95,
        lambda_=0.85,
    )

    # Optimizer for feature network only
    optimizer = torch.optim.Adam(feature_network.parameters(), lr=0.001)

    print(f"Feature network: {state_dim} -> {feature_dim}")
    print(f"SwiftSARSA layer: {feature_dim} features, {num_actions} actions")
    print(f"Optimizer: Adam (lr=0.001) for feature network")
    print()

    print("Training with gradient flow and softmax policy...")
    temperature = 1.0

    for episode in range(num_episodes):
        swift_sarsa_layer.reset()

        # Generate random initial state
        state = torch.randn(batch_size, state_dim)

        # Extract features and select initial action
        with torch.no_grad():
            features = feature_network(state)
            action_values = swift_sarsa_layer(
                features,
                torch.zeros(batch_size),
                torch.zeros(batch_size, dtype=torch.long),
            )
            action = softmax_policy(action_values, temperature=temperature)

        episode_reward = 0.0

        for step in range(30):
            # Generate next state and reward
            next_state = torch.randn(batch_size, state_dim)
            reward = torch.randn(batch_size)

            # Forward through feature network (with gradient tracking)
            next_features = feature_network(next_state)

            # SwiftSARSA computes action-values and performs Swift updates
            next_action_values = swift_sarsa_layer(next_features, reward, action)

            # Optional: Add supervision loss for gradient flow
            # Target could be from Q-learning target or Monte Carlo returns
            target_q = reward.unsqueeze(1) + 0.95 * torch.randn(batch_size, num_actions)
            loss = (next_action_values - target_q).pow(2).mean()

            # Backward: gradients flow to feature network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Select next action
            with torch.no_grad():
                action = softmax_policy(next_action_values, temperature=temperature)

            episode_reward += reward.mean().item()
            state = next_state

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = episode_reward / 30
            print(f"  Episode {episode + 1}: Avg reward = {avg_reward:.4f}")

        # Decay temperature for less exploration
        temperature *= 0.99

    print("\nTraining complete!")
    print("Feature network learned state representations")
    print("SwiftSARSA layer learned action-value functions")
    print(f"Final Swift-SARSA weights shape: {swift_sarsa_layer.weights.shape}")


if __name__ == "__main__":
    main()
