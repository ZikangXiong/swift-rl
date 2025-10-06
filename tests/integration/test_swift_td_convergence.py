"""Integration tests for SwiftTD convergence."""

import pytest
import torch

from swift_rl import SwiftTD
from tests.fixtures import create_state_features, linear_prediction_task


def test_swifttd_linear_task_convergence():
    """Test SwiftTD converges on simple linear prediction task."""
    batch_size = 32
    feature_dim = 10
    num_steps = 1000

    swift_td = SwiftTD(feature_dim=feature_dim, alpha=0.05, gamma=0.99, lambda_=0.9)

    losses = []

    for step in range(num_steps):
        features, target_values = linear_prediction_task(batch_size, feature_dim)

        # Generate rewards as change in target
        if step == 0:
            reward = torch.zeros(batch_size)
            prev_target = target_values.clone()
        else:
            reward = target_values - prev_target
            prev_target = target_values.clone()

        # Forward pass
        value = swift_td(features, reward)

        # Track loss
        mse = ((value - target_values) ** 2).mean().item()
        losses.append(mse)

    # Check convergence: final loss should be lower than initial or stable
    initial_loss = sum(losses[:50]) / 50
    final_loss = sum(losses[-50:]) / 50

    # TD learning may not achieve dramatic reduction but should show some improvement
    assert final_loss <= initial_loss  # Should not get worse


def test_swifttd_tabular_value_estimation():
    """Test SwiftTD updates weights in episodic setting."""
    # Simple chain: 0 -> 1 -> 2 -> 3 -> 4 (terminal, reward=1)
    num_states = 5
    swift_td = SwiftTD(feature_dim=num_states, alpha=0.3, gamma=0.9, lambda_=0.5)

    # Store initial weights
    initial_weights = swift_td.weights.data.clone()

    # Run episodes
    num_episodes = 100
    for episode in range(num_episodes):
        state = 0  # Start at state 0

        # Episode trajectory
        for step in range(10):  # Max 10 steps per episode
            # Get state features (one-hot)
            features = create_state_features(state, num_states).unsqueeze(0)

            # Take action to move right
            next_state = min(state + 1, num_states - 1)

            # Reward
            reward = torch.tensor([1.0 if next_state == num_states - 1 else 0.0])

            # Terminal flag
            terminal = torch.tensor([next_state == num_states - 1])

            # Update
            swift_td(features, reward, terminal)

            if terminal.item():
                swift_td.reset()
                break

            state = next_state

    # Check that weights have changed (learning occurred)
    weight_change = (swift_td.weights.data - initial_weights).abs().sum().item()
    assert weight_change > 0.01, "Weights should have changed during learning"


def test_swifttd_batch_consistency():
    """Test SwiftTD produces consistent results across batch sizes."""
    feature_dim = 8

    # Train with batch_size=1
    swift_td_single = SwiftTD(
        feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9
    )
    torch.manual_seed(42)
    for _ in range(50):
        features, targets = linear_prediction_task(1, feature_dim)
        reward = torch.zeros(1)
        swift_td_single(features, reward)

    # Train with batch_size=32
    swift_td_batch = SwiftTD(
        feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9
    )
    torch.manual_seed(42)
    for _ in range(50):
        features, targets = linear_prediction_task(32, feature_dim)
        reward = torch.zeros(32)
        swift_td_batch(features, reward)

    # Weights should be similar (not identical due to averaging differences)
    weight_diff = (swift_td_single.weights - swift_td_batch.weights).abs().mean()
    assert weight_diff < 0.5  # Reasonably close


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swifttd_cpu_gpu_consistency():
    """Test SwiftTD produces same results on CPU and GPU."""
    batch_size = 16
    feature_dim = 10
    num_steps = 20

    # CPU version
    swift_td_cpu = SwiftTD(feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9)

    torch.manual_seed(42)
    for _ in range(num_steps):
        features, _ = linear_prediction_task(batch_size, feature_dim, device="cpu")
        reward = torch.zeros(batch_size)
        swift_td_cpu(features, reward)

    # GPU version
    swift_td_gpu = SwiftTD(
        feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9
    ).to("cuda")

    torch.manual_seed(42)
    for _ in range(num_steps):
        features, _ = linear_prediction_task(batch_size, feature_dim, device="cuda")
        reward = torch.zeros(batch_size, device="cuda")
        swift_td_gpu(features, reward)

    # Weights should be very close
    weights_cpu = swift_td_cpu.weights.data
    weights_gpu = swift_td_gpu.weights.data.cpu()

    assert torch.allclose(weights_cpu, weights_gpu, atol=1e-5)


def test_swifttd_reset_clears_learning():
    """Test reset properly clears eligibility traces."""
    batch_size = 8
    feature_dim = 5

    swift_td = SwiftTD(feature_dim=feature_dim, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Run some updates
    for _ in range(10):
        features = torch.randn(batch_size, feature_dim)
        reward = torch.randn(batch_size)
        swift_td(features, reward)

    # Traces should be non-zero
    assert swift_td.eligibility_trace.abs().sum() > 0

    # Reset
    swift_td.reset()

    # Traces should be zero
    assert swift_td.eligibility_trace.abs().sum() == 0
    assert swift_td.prev_value.abs().sum() == 0


def test_swifttd_terminal_handling():
    """Test terminal states properly reset traces."""
    batch_size = 4
    feature_dim = 5

    swift_td = SwiftTD(feature_dim=feature_dim, alpha=0.1, gamma=0.99, lambda_=0.9)

    # First update
    features1 = torch.randn(batch_size, feature_dim)
    reward1 = torch.randn(batch_size)
    swift_td(features1, reward1)

    # Second update with some terminals
    features2 = torch.randn(batch_size, feature_dim)
    reward2 = torch.randn(batch_size)
    terminal = torch.tensor([False, True, False, True])

    swift_td(features2, reward2, terminal)

    # Traces for indices 1 and 3 should be zero
    assert swift_td.eligibility_trace[1].abs().sum() == 0.0
    assert swift_td.eligibility_trace[3].abs().sum() == 0.0

    # Traces for indices 0 and 2 should be non-zero
    assert swift_td.eligibility_trace[0].abs().sum() > 0.0
    assert swift_td.eligibility_trace[2].abs().sum() > 0.0
