"""Neural network integration example with SwiftTD as final layer."""

import torch
import torch.nn as nn

from swift_rl import SwiftTD


def main():
    """Demonstrate SwiftTD as final layer of neural network."""
    # Setup
    state_dim = 64
    feature_dim = 32
    batch_size = 32
    num_episodes = 50

    print("Creating neural network with SwiftTD final layer...")

    # Feature extraction network
    feature_network = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, feature_dim),
    )

    # SwiftTD as final layer
    swift_td_layer = SwiftTD(
        feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9
    )

    # Optimizer for feature network only (SwiftTD updates its own weights)
    optimizer = torch.optim.Adam(feature_network.parameters(), lr=0.001)

    print(f"Feature network: {state_dim} -> {feature_dim}")
    print(f"SwiftTD layer: {feature_dim} features")
    print(f"Optimizer: Adam (lr=0.001) for feature network")
    print()

    print("Training with gradient flow to feature network...")

    for episode in range(num_episodes):
        swift_td_layer.reset()

        # Generate random states
        state = torch.randn(batch_size, state_dim)

        episode_loss = 0.0

        for step in range(30):
            # Forward through feature network
            features = feature_network(state)  # Gradients tracked

            # Generate next state and reward
            next_state = torch.randn(batch_size, state_dim)
            reward = torch.randn(batch_size)

            # SwiftTD layer computes value and performs Swift updates
            value = swift_td_layer(features, reward)

            # Optional: Add supervision loss for gradient flow
            # In real use, target_value would come from environment
            target_value = reward + 0.99 * torch.randn(batch_size)
            loss = (value - target_value).pow(2).mean()

            # Backward pass: gradients flow to feature_network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            state = next_state

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_loss = episode_loss / 30
            print(f"  Episode {episode + 1}: Avg loss = {avg_loss:.4f}")

    print("\nTraining complete!")
    print("Feature network learned representations")
    print("SwiftTD layer learned value function")
    print(f"Final Swift-TD weights shape: {swift_td_layer.weights.shape}")


if __name__ == "__main__":
    main()
