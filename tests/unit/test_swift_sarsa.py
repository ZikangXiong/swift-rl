"""Unit tests for SwiftSARSA module."""

import pytest
import torch

from swift_rl import SwiftSARSA
from tests.fixtures import linear_prediction_task

# ============================================================================
# T020: Initialization Tests
# ============================================================================


def test_swiftsarsa_initialization_valid():
    """Test SwiftSARSA initializes with valid parameters."""
    swift_sarsa = SwiftSARSA(
        feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    assert swift_sarsa.feature_dim == 10
    assert swift_sarsa.num_actions == 4
    assert swift_sarsa.alpha == 0.1
    assert swift_sarsa.gamma == 0.99
    assert swift_sarsa.lambda_ == 0.9
    assert swift_sarsa.weights.shape == (4, 10)


def test_swiftsarsa_initialization_buffers():
    """Test SwiftSARSA initializes buffers correctly."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    assert swift_sarsa.eligibility_traces.shape == (1, 3, 5)
    assert swift_sarsa.step_sizes.shape == (1, 3, 5)
    assert swift_sarsa.prev_action_values.shape == (1, 3)


def test_swiftsarsa_invalid_feature_dim():
    """Test SwiftSARSA rejects invalid feature_dim."""
    with pytest.raises(ValueError, match="feature_dim must be > 0"):
        SwiftSARSA(feature_dim=0, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9)

    with pytest.raises(ValueError, match="feature_dim must be > 0"):
        SwiftSARSA(feature_dim=-5, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9)


def test_swiftsarsa_invalid_num_actions():
    """Test SwiftSARSA rejects invalid num_actions."""
    with pytest.raises(ValueError, match="num_actions must be > 1"):
        SwiftSARSA(feature_dim=10, num_actions=1, alpha=0.1, gamma=0.99, lambda_=0.9)

    with pytest.raises(ValueError, match="num_actions must be > 1"):
        SwiftSARSA(feature_dim=10, num_actions=0, alpha=0.1, gamma=0.99, lambda_=0.9)


def test_swiftsarsa_invalid_alpha():
    """Test SwiftSARSA rejects invalid alpha."""
    with pytest.raises(ValueError, match="alpha must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=0.0, gamma=0.99, lambda_=0.9)

    with pytest.raises(ValueError, match="alpha must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=1.5, gamma=0.99, lambda_=0.9)


def test_swiftsarsa_invalid_gamma():
    """Test SwiftSARSA rejects invalid gamma."""
    with pytest.raises(ValueError, match="gamma must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=0.1, gamma=-0.1, lambda_=0.9)

    with pytest.raises(ValueError, match="gamma must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=0.1, gamma=1.5, lambda_=0.9)


def test_swiftsarsa_invalid_lambda():
    """Test SwiftSARSA rejects invalid lambda."""
    with pytest.raises(ValueError, match="lambda_ must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=-0.1)

    with pytest.raises(ValueError, match="lambda_ must be in"):
        SwiftSARSA(feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=1.5)


# ============================================================================
# T021: Forward Pass Tests
# ============================================================================


def test_swiftsarsa_forward_output_shape():
    """Test forward pass returns correct shape."""
    swift_sarsa = SwiftSARSA(
        feature_dim=8, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9
    )
    batch_size = 16

    features = torch.randn(batch_size, 8)
    reward = torch.randn(batch_size)
    action = torch.randint(0, 4, (batch_size,))

    action_values = swift_sarsa(features, reward, action)

    assert action_values.shape == (batch_size, 4)
    assert action_values.dtype == torch.float32


def test_swiftsarsa_forward_buffer_expansion():
    """Test buffers expand to match batch size."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Initial buffers are size 1
    assert swift_sarsa.eligibility_traces.shape[0] == 1

    # Forward with batch_size=10
    features = torch.randn(10, 5)
    reward = torch.randn(10)
    action = torch.randint(0, 3, (10,))
    swift_sarsa(features, reward, action)

    # Buffers should expand
    assert swift_sarsa.eligibility_traces.shape[0] == 10
    assert swift_sarsa.step_sizes.shape[0] == 10
    assert swift_sarsa.prev_action_values.shape[0] == 10


def test_swiftsarsa_forward_with_terminal():
    """Test forward pass with terminal flags."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    features = torch.randn(8, 5)
    reward = torch.randn(8)
    action = torch.randint(0, 3, (8,))
    terminal = torch.tensor([False, False, True, False, False, True, False, False])

    action_values = swift_sarsa(features, reward, action, terminal)

    assert action_values.shape == (8, 3)
    # Terminal states should have reset traces
    assert swift_sarsa.eligibility_traces[2].abs().sum() == 0.0
    assert swift_sarsa.eligibility_traces[5].abs().sum() == 0.0


def test_swiftsarsa_forward_action_value_computation():
    """Test action-value computation: Q = features @ weights.T."""
    swift_sarsa = SwiftSARSA(
        feature_dim=3, num_actions=2, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Set weights to known values
    swift_sarsa.weights.data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reward = torch.zeros(2)
    action = torch.tensor([0, 1])

    action_values = swift_sarsa(features, reward, action)

    # Expected: [[1*1+0*2+0*3, 1*4+0*5+0*6], [0*1+1*2+0*3, 0*4+1*5+0*6]]
    #         = [[1.0, 4.0], [2.0, 5.0]]
    assert torch.allclose(action_values[0], torch.tensor([1.0, 4.0]), atol=1e-3)
    assert torch.allclose(action_values[1], torch.tensor([2.0, 5.0]), atol=1e-3)


def test_swiftsarsa_forward_updates_weights():
    """Test that forward pass updates weights."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    initial_weights = swift_sarsa.weights.data.clone()

    features = torch.randn(4, 5)
    reward = torch.randn(4)
    action = torch.randint(0, 3, (4,))

    # Multiple steps to ensure updates
    for _ in range(10):
        swift_sarsa(features, reward, action)

    # Weights should change
    assert not torch.allclose(swift_sarsa.weights.data, initial_weights)


def test_swiftsarsa_forward_eligibility_trace_update():
    """Test eligibility trace update for selected actions."""
    swift_sarsa = SwiftSARSA(
        feature_dim=3, num_actions=2, alpha=0.1, gamma=0.9, lambda_=0.8
    )

    # First step - select action 0
    features1 = torch.tensor([[1.0, 0.0, 0.0]])
    reward1 = torch.zeros(1)
    action1 = torch.tensor([0])
    swift_sarsa(features1, reward1, action1)

    # Trace for action 0 should be approximately features1
    expected_trace = features1
    assert torch.allclose(
        swift_sarsa.eligibility_traces[0, 0], expected_trace[0], atol=1e-2
    )
    # Trace for action 1 should still be near zero (decayed)
    assert swift_sarsa.eligibility_traces[0, 1].abs().sum() < 0.1


# ============================================================================
# T022: Reset Method Tests
# ============================================================================


def test_swiftsarsa_reset_all():
    """Test reset without indices resets all buffers."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Run some updates
    features = torch.randn(8, 5)
    reward = torch.randn(8)
    action = torch.randint(0, 3, (8,))
    swift_sarsa(features, reward, action)

    # Buffers should be non-zero
    assert swift_sarsa.eligibility_traces.abs().sum() > 0

    # Reset all
    swift_sarsa.reset()

    # All buffers should be zero or reset to alpha
    assert swift_sarsa.eligibility_traces.abs().sum() == 0.0
    assert torch.allclose(
        swift_sarsa.step_sizes, torch.full_like(swift_sarsa.step_sizes, 0.1)
    )
    assert swift_sarsa.prev_action_values.abs().sum() == 0.0


def test_swiftsarsa_reset_specific_indices():
    """Test reset with specific batch indices."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Run some updates
    features = torch.randn(8, 5)
    reward = torch.randn(8)
    action = torch.randint(0, 3, (8,))
    swift_sarsa(features, reward, action)

    # Store current values
    trace_before = swift_sarsa.eligibility_traces.clone()

    # Reset indices [1, 3, 5]
    reset_indices = torch.tensor([1, 3, 5])
    swift_sarsa.reset(reset_indices)

    # Only specified indices should be reset
    assert swift_sarsa.eligibility_traces[1].abs().sum() == 0.0
    assert swift_sarsa.eligibility_traces[3].abs().sum() == 0.0
    assert swift_sarsa.eligibility_traces[5].abs().sum() == 0.0

    # Other indices should remain unchanged
    assert torch.allclose(swift_sarsa.eligibility_traces[0], trace_before[0])
    assert torch.allclose(swift_sarsa.eligibility_traces[2], trace_before[2])


def test_swiftsarsa_reset_empty_indices():
    """Test reset with empty indices tensor."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Run some updates
    features = torch.randn(4, 5)
    reward = torch.randn(4)
    action = torch.randint(0, 3, (4,))
    swift_sarsa(features, reward, action)

    trace_before = swift_sarsa.eligibility_traces.clone()

    # Reset with empty indices (should do nothing)
    swift_sarsa.reset(torch.tensor([], dtype=torch.long))

    assert torch.allclose(swift_sarsa.eligibility_traces, trace_before)


# ============================================================================
# T023: Device Handling Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swiftsarsa_cuda_device():
    """Test SwiftSARSA on CUDA device."""
    device = torch.device("cuda")
    swift_sarsa = SwiftSARSA(
        feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9
    ).to(device)

    features = torch.randn(16, 10, device=device)
    reward = torch.randn(16, device=device)
    action = torch.randint(0, 4, (16,), device=device)

    action_values = swift_sarsa(features, reward, action)

    assert action_values.device == device
    assert swift_sarsa.weights.device == device
    assert swift_sarsa.eligibility_traces.device == device


def test_swiftsarsa_cpu_device():
    """Test SwiftSARSA on CPU device."""
    device = torch.device("cpu")
    swift_sarsa = SwiftSARSA(
        feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9
    ).to(device)

    features = torch.randn(16, 10, device=device)
    reward = torch.randn(16, device=device)
    action = torch.randint(0, 4, (16,), device=device)

    action_values = swift_sarsa(features, reward, action)

    assert action_values.device == device
    assert swift_sarsa.weights.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swiftsarsa_device_transfer():
    """Test transferring SwiftSARSA between devices."""
    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    # Start on CPU
    assert swift_sarsa.weights.device == torch.device("cpu")

    # Move to CUDA
    swift_sarsa = swift_sarsa.to("cuda")
    assert swift_sarsa.weights.device.type == "cuda"
    assert swift_sarsa.eligibility_traces.device.type == "cuda"

    # Move back to CPU
    swift_sarsa = swift_sarsa.to("cpu")
    assert swift_sarsa.weights.device == torch.device("cpu")


def test_swiftsarsa_mixed_device_inputs_raises():
    """Test that mixed device inputs raise error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    swift_sarsa = SwiftSARSA(
        feature_dim=5, num_actions=3, alpha=0.1, gamma=0.99, lambda_=0.9
    ).to("cuda")

    # Features on CUDA, reward on CPU (mismatched)
    features = torch.randn(4, 5, device="cuda")
    reward = torch.randn(4, device="cpu")
    action = torch.randint(0, 3, (4,), device="cuda")

    with pytest.raises(RuntimeError):
        swift_sarsa(features, reward, action)
