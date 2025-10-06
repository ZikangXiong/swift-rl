"""Pendulum control using SwiftSARSA for inverted pendulum swing-up task."""

import math

import torch

from swift_rl import SwiftSARSA
from swift_rl.policies import epsilon_greedy


def create_pendulum_features(angle: float, angular_vel: float) -> torch.Tensor:
    """Create feature vector from pendulum state (angle, angular velocity)."""
    return torch.tensor(
        [
            math.cos(angle),
            math.sin(angle),
            angular_vel / 8.0,
            (math.cos(angle) * angular_vel) / 8.0,
            (math.sin(angle) * angular_vel) / 8.0,
        ],
        dtype=torch.float32,
    )


def pendulum_step(angle: float, angular_vel: float, action: int) -> tuple[float, float, float]:
    """Simple pendulum dynamics: angle in [-π, π], angular_vel in [-8, 8]."""
    dt = 0.05
    g = 10.0
    m = 1.0
    length = 1.0

    # Actions: 0=left torque, 1=no torque, 2=right torque
    torque = (action - 1) * 2.0

    # Physics update
    angular_acc = -3 * g / (2 * length) * math.sin(angle) + 3.0 / (m * length**2) * torque
    new_angular_vel = angular_vel + angular_acc * dt
    new_angular_vel = max(-8.0, min(8.0, new_angular_vel))
    new_angle = angle + new_angular_vel * dt

    # Normalize angle to [-π, π]
    new_angle = ((new_angle + math.pi) % (2 * math.pi)) - math.pi

    # Reward: negative distance from upright position and velocity penalty
    reward = -(new_angle**2 + 0.1 * new_angular_vel**2 + 0.001 * torque**2)

    return new_angle, new_angular_vel, reward


def main():
    """Train SwiftSARSA to solve inverted pendulum swing-up."""
    print("Pendulum Control with SwiftSARSA\n")

    # Setup
    feature_dim = 5
    num_actions = 3  # left, none, right
    num_episodes = 500
    max_steps = 200

    # Initialize SwiftSARSA
    swift_sarsa = SwiftSARSA(
        feature_dim=feature_dim,
        num_actions=num_actions,
        alpha=0.05,
        gamma=0.99,
        lambda_=0.95,
    )

    print(f"SwiftSARSA initialized: {feature_dim} features, {num_actions} actions")
    print(f"Hyperparameters: α=0.05, γ=0.99, λ=0.95\n")

    # Training
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 0.995

    epsilon = epsilon_start
    best_reward = float("-inf")

    for episode in range(num_episodes):
        # Reset environment
        angle = torch.rand(1).item() * 2 * math.pi - math.pi
        angular_vel = (torch.rand(1).item() - 0.5) * 4.0

        # Get initial features and action
        features = create_pendulum_features(angle, angular_vel).unsqueeze(0)

        with torch.no_grad():
            action_values = swift_sarsa(
                features, torch.zeros(1), torch.zeros(1, dtype=torch.long)
            )
            action = epsilon_greedy(action_values, epsilon=epsilon, training=True)

        episode_reward = 0.0
        swift_sarsa.reset()

        # Episode rollout
        for step in range(max_steps):
            # Execute action
            next_angle, next_angular_vel, reward = pendulum_step(
                angle, angular_vel, int(action.item())
            )

            # Get next features
            next_features = create_pendulum_features(
                next_angle, next_angular_vel
            ).unsqueeze(0)
            reward_tensor = torch.tensor([reward])

            # SwiftSARSA update
            next_action_values = swift_sarsa(next_features, reward_tensor, action)

            # Select next action
            with torch.no_grad():
                action = epsilon_greedy(next_action_values, epsilon=epsilon, training=True)

            episode_reward += reward
            angle = next_angle
            angular_vel = next_angular_vel

        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Decay exploration
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1:3d}: Reward = {episode_reward:8.2f}, "
                  f"Best = {best_reward:8.2f}, ε = {epsilon:.3f}")

    print(f"\nTraining complete!")
    print(f"Best episode reward: {best_reward:.2f}")

    # Evaluation run
    print("\nEvaluation (greedy policy):")
    angle = 0.0  # Start hanging down
    angular_vel = 0.0
    eval_reward = 0.0

    for step in range(max_steps):
        features = create_pendulum_features(angle, angular_vel).unsqueeze(0)

        with torch.no_grad():
            action_values = swift_sarsa(
                features, torch.zeros(1), torch.zeros(1, dtype=torch.long)
            )
            action = epsilon_greedy(action_values, epsilon=0.0, training=False)

        angle, angular_vel, reward = pendulum_step(angle, angular_vel, int(action.item()))
        eval_reward += reward

        if step % 50 == 0:
            print(f"  Step {step:3d}: angle={angle:6.3f}, "
                  f"vel={angular_vel:6.3f}, action={action.item()}")

    print(f"\nEvaluation reward: {eval_reward:.2f}")
    print(f"Final angle: {angle:.3f} rad ({math.degrees(angle):.1f}°)")
    print(f"Final angular velocity: {angular_vel:.3f} rad/s")


if __name__ == "__main__":
    main()
