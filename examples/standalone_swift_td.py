"""Standalone SwiftTD example for temporal difference prediction."""

import torch

from swift_rl import SwiftTD


def main():
    """Demonstrate SwiftTD on a simple prediction task."""
    # Setup
    batch_size = 8
    feature_dim = 16
    num_steps = 100

    # Create SwiftTD module
    swift_td = SwiftTD(
        feature_dim=feature_dim,
        alpha=0.01,  # Learning rate
        gamma=0.99,  # Discount factor
        lambda_=0.9,  # Eligibility trace decay
    )

    print(f"SwiftTD initialized:")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Alpha: {swift_td.alpha}")
    print(f"  Gamma: {swift_td.gamma}")
    print(f"  Lambda: {swift_td.lambda_}")
    print()

    # Simple prediction task: learn to predict sum of features
    print("Training on prediction task...")
    for step in range(num_steps):
        # Generate random features
        features = torch.randn(batch_size, feature_dim)

        # Target: sum of features (simple linear relationship)
        target_value = features.sum(dim=1)

        # Simulate reward as difference from target
        if step == 0:
            reward = torch.zeros(batch_size)
        else:
            reward = target_value - prev_target
            prev_target = target_value.clone()

        # Forward pass (Swift-TD update happens automatically)
        value = swift_td(features, reward)

        if step == 0:
            prev_target = target_value.clone()

        # Print progress
        if (step + 1) % 20 == 0:
            mse = ((value - target_value) ** 2).mean().item()
            print(f"  Step {step + 1}: MSE = {mse:.4f}")

    print("\nTraining complete!")
    print(f"Final weights norm: {swift_td.weights.norm().item():.4f}")


if __name__ == "__main__":
    main()
