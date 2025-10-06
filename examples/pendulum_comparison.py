"""Compare SwiftSARSA vs standard linear SARSA on pendulum control."""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from swift_rl import SwiftSARSA
from swift_rl.policies import epsilon_greedy


def create_pendulum_features(angle: float, angular_vel: float) -> torch.Tensor:
    """Create feature vector from pendulum state."""
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
    """Pendulum dynamics with discrete actions."""
    dt = 0.05
    g = 10.0
    m = 1.0
    length = 1.0
    torque = (action - 1) * 2.0
    angular_acc = -3 * g / (2 * length) * math.sin(angle) + 3.0 / (m * length**2) * torque
    new_angular_vel = max(-8.0, min(8.0, angular_vel + angular_acc * dt))
    new_angle = ((angle + new_angular_vel * dt + math.pi) % (2 * math.pi)) - math.pi
    reward = -(new_angle**2 + 0.1 * new_angular_vel**2 + 0.001 * torque**2)
    return new_angle, new_angular_vel, reward


class BaselineSARSA(nn.Module):
    """Standard linear SARSA with fixed step-size (no adaptive step)."""

    def __init__(self, feature_dim: int, num_actions: int, alpha: float, gamma: float, lambda_: float):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.weights = nn.Parameter(torch.randn(num_actions, feature_dim) * 0.01)
        self.register_buffer("eligibility_traces", torch.zeros(1, num_actions, feature_dim))
        self.register_buffer("prev_action_values", torch.zeros(1, num_actions))

    def _expand_buffers(self, batch_size: int) -> None:
        """Expand buffers for batch processing."""
        if self.eligibility_traces.shape[0] < batch_size:
            new_traces = torch.zeros(batch_size, self.num_actions, self.feature_dim, device=self.weights.device)
            new_traces[: self.eligibility_traces.shape[0]] = self.eligibility_traces
            self.eligibility_traces = new_traces
            new_prev = torch.zeros(batch_size, self.num_actions, device=self.weights.device)
            new_prev[: self.prev_action_values.shape[0]] = self.prev_action_values
            self.prev_action_values = new_prev

    def forward(self, features: torch.Tensor, reward: torch.Tensor, action: torch.Tensor, terminal: torch.Tensor | None = None) -> torch.Tensor:
        """SARSA update with fixed step-size."""
        self._expand_buffers(features.shape[0])
        action_values = features @ self.weights.T
        td_error = reward + self.gamma * self.prev_action_values.gather(1, action.unsqueeze(1)).squeeze(1) - self.prev_action_values.max(dim=1).values
        batch_indices = torch.arange(features.shape[0], device=features.device)
        self.eligibility_traces[batch_indices, action] = self.gamma * self.lambda_ * self.eligibility_traces[batch_indices, action] + features
        self.weights.data += self.alpha * (td_error.unsqueeze(1).unsqueeze(2) * self.eligibility_traces).sum(dim=0)
        if terminal is not None:
            terminal_indices = torch.where(terminal)[0]
            if len(terminal_indices) > 0:
                self.eligibility_traces[terminal_indices] = 0.0
        self.prev_action_values.copy_(action_values.detach())
        return action_values

    def reset(self, indices: torch.Tensor | None = None) -> None:
        """Reset traces."""
        if indices is None:
            self.eligibility_traces.zero_()
        else:
            self.eligibility_traces[indices] = 0.0


def train_agent(agent: nn.Module, num_episodes: int, max_steps: int, epsilon_params: tuple, seed: int) -> Tuple[List[float], List[np.ndarray]]:
    """Train agent and return episode rewards and step-size history."""
    torch.manual_seed(seed)
    epsilon_start, epsilon_end, epsilon_decay = epsilon_params
    epsilon = epsilon_start
    episode_rewards = []
    step_size_history = []

    for episode in range(num_episodes):
        angle = torch.rand(1).item() * 2 * math.pi - math.pi
        angular_vel = (torch.rand(1).item() - 0.5) * 4.0
        features = create_pendulum_features(angle, angular_vel).unsqueeze(0)

        with torch.no_grad():
            action_values = agent(features, torch.zeros(1), torch.zeros(1, dtype=torch.long))
            action = epsilon_greedy(action_values, epsilon=epsilon, training=True)

        episode_reward = 0.0
        agent.reset()

        for step in range(max_steps):
            next_angle, next_angular_vel, reward = pendulum_step(angle, angular_vel, int(action.item()))
            next_features = create_pendulum_features(next_angle, next_angular_vel).unsqueeze(0)
            next_action_values = agent(next_features, torch.tensor([reward]), action)

            with torch.no_grad():
                action = epsilon_greedy(next_action_values, epsilon=epsilon, training=True)

            episode_reward += reward
            angle, angular_vel = next_angle, next_angular_vel

        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Record step-sizes (attention) for SwiftSARSA
        if hasattr(agent, "step_sizes") and (episode + 1) % 10 == 0:
            # Average across actions only: (num_actions, feature_dim) -> (feature_dim,)
            step_sizes = agent.step_sizes[0].mean(dim=0).cpu().numpy()
            step_size_history.append(step_sizes)

    return episode_rewards, step_size_history


def plot_results(swift_all: List[List[float]], baseline_all: List[List[float]], swift_step_sizes: List[List[np.ndarray]]) -> None:
    """Plot learning curves with multi-seed statistics and feature attention."""
    feature_names = ["cos(θ)", "sin(θ)", "ω/8", "cos·ω", "sin·ω"]

    plt.figure(figsize=(15, 5))

    # Learning curves with error bars
    plt.subplot(1, 3, 1)
    swift_arr = np.array(swift_all)
    baseline_arr = np.array(baseline_all)

    swift_mean = swift_arr.mean(axis=0)
    swift_std = swift_arr.std(axis=0)
    baseline_mean = baseline_arr.mean(axis=0)
    baseline_std = baseline_arr.std(axis=0)

    episodes = np.arange(len(swift_mean))

    plt.plot(episodes, swift_mean, label="SwiftSARSA", color="blue", linewidth=2)
    plt.fill_between(episodes, swift_mean - swift_std, swift_mean + swift_std, alpha=0.2, color="blue")

    plt.plot(episodes, baseline_mean, label="Baseline SARSA", color="orange", linewidth=2)
    plt.fill_between(episodes, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.2, color="orange")

    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Episode Reward", fontsize=11)
    plt.title("Learning Curves (Mean ± Std)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Moving average across seeds
    plt.subplot(1, 3, 2)
    window = 20

    def moving_average(data: np.ndarray, w: int) -> np.ndarray:
        return np.convolve(data, np.ones(w), 'valid') / w

    swift_ma = np.array([moving_average(run, window) for run in swift_all])
    baseline_ma = np.array([moving_average(run, window) for run in baseline_all])

    swift_ma_mean = swift_ma.mean(axis=0)
    swift_ma_std = swift_ma.std(axis=0)
    baseline_ma_mean = baseline_ma.mean(axis=0)
    baseline_ma_std = baseline_ma.std(axis=0)

    episodes_ma = np.arange(len(swift_ma_mean))

    plt.plot(episodes_ma, swift_ma_mean, label="SwiftSARSA", color="blue", linewidth=2)
    plt.fill_between(episodes_ma, swift_ma_mean - swift_ma_std, swift_ma_mean + swift_ma_std, alpha=0.2, color="blue")

    plt.plot(episodes_ma, baseline_ma_mean, label="Baseline", color="orange", linewidth=2)
    plt.fill_between(episodes_ma, baseline_ma_mean - baseline_ma_std, baseline_ma_mean + baseline_ma_std, alpha=0.2, color="orange")

    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Reward (MA-20)", fontsize=11)
    plt.title("Smoothed Learning Curves", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Feature attention (step-sizes) evolution
    plt.subplot(1, 3, 3)

    step_size_arr = np.array(swift_step_sizes[0])  # Use first seed
    episodes_step = np.arange(10, len(swift_all[0]) + 1, 10)[:len(step_size_arr)]

    for i, name in enumerate(feature_names):
        plt.plot(episodes_step, step_size_arr[:, i], label=name, linewidth=2, marker='o', markersize=3)

    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Adaptive Step-Size (α)", fontsize=11)
    plt.title("Feature Attention Evolution", fontsize=12)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pendulum_comparison.png", dpi=150)
    print("\nPlot saved to: pendulum_comparison.png")


def main():
    """Compare SwiftSARSA with baseline linear SARSA across multiple seeds."""
    print("Pendulum Control: SwiftSARSA vs Baseline SARSA\n")

    feature_dim = 5
    num_actions = 3
    num_episodes = 500
    max_steps = 200
    epsilon_params = (0.9, 0.05, 0.995)
    num_seeds = 5

    swift_all_rewards = []
    baseline_all_rewards = []
    swift_all_step_sizes = []

    print(f"Running experiments with {num_seeds} seeds...\n")

    for seed in range(num_seeds):
        print(f"Seed {seed + 1}/{num_seeds}:")

        print("  Training SwiftSARSA (adaptive step-size)...")
        swift_sarsa = SwiftSARSA(feature_dim=feature_dim, num_actions=num_actions, alpha=0.05, gamma=0.99, lambda_=0.95)
        swift_rewards, step_sizes = train_agent(swift_sarsa, num_episodes, max_steps, epsilon_params, seed)
        swift_all_rewards.append(swift_rewards)
        swift_all_step_sizes.append(step_sizes)
        print(f"    Final 50-ep avg: {sum(swift_rewards[-50:]) / 50:7.2f}")

        print("  Training Baseline SARSA (fixed step-size)...")
        baseline = BaselineSARSA(feature_dim=feature_dim, num_actions=num_actions, alpha=0.05, gamma=0.99, lambda_=0.95)
        baseline_rewards, _ = train_agent(baseline, num_episodes, max_steps, epsilon_params, seed)
        baseline_all_rewards.append(baseline_rewards)
        print(f"    Final 50-ep avg: {sum(baseline_rewards[-50:]) / 50:7.2f}\n")

    # Compute statistics
    swift_arr = np.array(swift_all_rewards)
    baseline_arr = np.array(baseline_all_rewards)

    swift_final_mean = swift_arr[:, -50:].mean()
    swift_final_std = swift_arr[:, -50:].std()
    baseline_final_mean = baseline_arr[:, -50:].mean()
    baseline_final_std = baseline_arr[:, -50:].std()

    print("=" * 60)
    print("Performance Summary (across all seeds):")
    print(f"  SwiftSARSA  - Final 50-ep: {swift_final_mean:7.2f} ± {swift_final_std:6.2f}")
    print(f"  Baseline    - Final 50-ep: {baseline_final_mean:7.2f} ± {baseline_final_std:6.2f}")
    print(f"  Improvement: {((swift_final_mean - baseline_final_mean) / abs(baseline_final_mean)) * 100:.1f}%")
    print("=" * 60)

    plot_results(swift_all_rewards, baseline_all_rewards, swift_all_step_sizes)


if __name__ == "__main__":
    main()
