"""Unit tests for SwiftTD module."""

import pytest
import torch

from swift_rl import SwiftTD
from tests.fixtures import linear_prediction_task

# ============================================================================
# T016: Initialization Tests
# ============================================================================


def test_swifttd_initialization_valid():
    """Test SwiftTD initializes with valid parameters."""
    swift_td = SwiftTD(feature_dim=10, alpha=0.1, gamma=0.99, lambda_=0.9)

    assert swift_td.feature_dim == 10
    assert swift_td.alpha == 0.1
    assert swift_td.gamma == 0.99
    assert swift_td.lambda_ == 0.9
    assert swift_td.weights.shape == (10,)


def test_swifttd_initialization_buffers():
    """Test SwiftTD initializes buffers correctly."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    assert swift_td.eligibility_trace.shape == (1, 5)
    assert swift_td.step_sizes.shape == (1, 5)
    assert swift_td.prev_value.shape == (1,)


def test_swifttd_invalid_feature_dim():
    """Test SwiftTD rejects invalid feature_dim."""
    with pytest.raises(ValueError, match="feature_dim must be > 0"):
        SwiftTD(feature_dim=0, alpha=0.1, gamma=0.99, lambda_=0.9)

    with pytest.raises(ValueError, match="feature_dim must be > 0"):
        SwiftTD(feature_dim=-5, alpha=0.1, gamma=0.99, lambda_=0.9)


def test_swifttd_invalid_alpha():
    """Test SwiftTD rejects invalid alpha."""
    with pytest.raises(ValueError, match="alpha must be in"):
        SwiftTD(feature_dim=10, alpha=0.0, gamma=0.99, lambda_=0.9)

    with pytest.raises(ValueError, match="alpha must be in"):
        SwiftTD(feature_dim=10, alpha=1.5, gamma=0.99, lambda_=0.9)


def test_swifttd_invalid_gamma():
    """Test SwiftTD rejects invalid gamma."""
    with pytest.raises(ValueError, match="gamma must be in"):
        SwiftTD(feature_dim=10, alpha=0.1, gamma=-0.1, lambda_=0.9)

    with pytest.raises(ValueError, match="gamma must be in"):
        SwiftTD(feature_dim=10, alpha=0.1, gamma=1.5, lambda_=0.9)


def test_swifttd_invalid_lambda():
    """Test SwiftTD rejects invalid lambda."""
    with pytest.raises(ValueError, match="lambda_ must be in"):
        SwiftTD(feature_dim=10, alpha=0.1, gamma=0.99, lambda_=-0.1)

    with pytest.raises(ValueError, match="lambda_ must be in"):
        SwiftTD(feature_dim=10, alpha=0.1, gamma=0.99, lambda_=1.5)


# ============================================================================
# T017: Forward Pass Tests
# ============================================================================


def test_swifttd_forward_output_shape():
    """Test forward pass returns correct shape."""
    swift_td = SwiftTD(feature_dim=8, alpha=0.1, gamma=0.99, lambda_=0.9)
    batch_size = 16

    features = torch.randn(batch_size, 8)
    reward = torch.randn(batch_size)

    value = swift_td(features, reward)

    assert value.shape == (batch_size,)
    assert value.dtype == torch.float32


def test_swifttd_forward_buffer_expansion():
    """Test buffers expand to match batch size."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Initial buffers are size 1
    assert swift_td.eligibility_trace.shape[0] == 1

    # Forward with batch_size=10
    features = torch.randn(10, 5)
    reward = torch.randn(10)
    swift_td(features, reward)

    # Buffers should expand
    assert swift_td.eligibility_trace.shape[0] == 10
    assert swift_td.step_sizes.shape[0] == 10
    assert swift_td.prev_value.shape[0] == 10


def test_swifttd_forward_with_terminal():
    """Test forward pass with terminal flags."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    features = torch.randn(8, 5)
    reward = torch.randn(8)
    terminal = torch.tensor([False, False, True, False, False, True, False, False])

    value = swift_td(features, reward, terminal)

    assert value.shape == (8,)
    # Terminal states should have reset traces
    assert swift_td.eligibility_trace[2].abs().sum() == 0.0
    assert swift_td.eligibility_trace[5].abs().sum() == 0.0


def test_swifttd_forward_value_computation():
    """Test value computation: v = w · φ."""
    swift_td = SwiftTD(feature_dim=3, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Set weights to known values
    swift_td.weights.data = torch.tensor([1.0, 2.0, 3.0])

    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    reward = torch.zeros(2)

    value = swift_td(features, reward)

    # Expected: [1*1 + 0*2 + 0*3, 0*1 + 1*2 + 1*3] = [1.0, 5.0]
    assert torch.allclose(value, torch.tensor([1.0, 5.0]), atol=1e-3)


def test_swifttd_forward_updates_weights():
    """Test that forward pass updates weights."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    initial_weights = swift_td.weights.data.clone()

    features = torch.randn(4, 5)
    reward = torch.randn(4)

    # Multiple steps to ensure updates
    for _ in range(10):
        swift_td(features, reward)

    # Weights should change
    assert not torch.allclose(swift_td.weights.data, initial_weights)


def test_swifttd_forward_eligibility_trace_update():
    """Test eligibility trace update: e ← γλe + φ."""
    swift_td = SwiftTD(feature_dim=3, alpha=0.1, gamma=0.9, lambda_=0.8)

    # First step
    features1 = torch.tensor([[1.0, 0.0, 0.0]])
    reward1 = torch.zeros(1)
    swift_td(features1, reward1)

    # Trace should be approximately features1 (since initial trace was 0)
    expected_trace1 = features1
    assert torch.allclose(swift_td.eligibility_trace[:1], expected_trace1, atol=1e-2)

    # Second step
    features2 = torch.tensor([[0.0, 1.0, 0.0]])
    reward2 = torch.zeros(1)
    swift_td(features2, reward2)

    # Trace should be γλ * trace1 + features2
    expected_trace2 = 0.9 * 0.8 * expected_trace1 + features2
    assert torch.allclose(swift_td.eligibility_trace[:1], expected_trace2, atol=1e-2)


# ============================================================================
# T018: Reset Method Tests
# ============================================================================


def test_swifttd_reset_all():
    """Test reset without indices resets all buffers."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Run some updates
    features = torch.randn(8, 5)
    reward = torch.randn(8)
    swift_td(features, reward)

    # Buffers should be non-zero
    assert swift_td.eligibility_trace.abs().sum() > 0

    # Reset all
    swift_td.reset()

    # All buffers should be zero or reset to alpha
    assert swift_td.eligibility_trace.abs().sum() == 0.0
    assert torch.allclose(
        swift_td.step_sizes, torch.full_like(swift_td.step_sizes, 0.1)
    )
    assert swift_td.prev_value.abs().sum() == 0.0


def test_swifttd_reset_specific_indices():
    """Test reset with specific batch indices."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Run some updates
    features = torch.randn(8, 5)
    reward = torch.randn(8)
    swift_td(features, reward)

    # Store current values
    trace_before = swift_td.eligibility_trace.clone()

    # Reset indices [1, 3, 5]
    reset_indices = torch.tensor([1, 3, 5])
    swift_td.reset(reset_indices)

    # Only specified indices should be reset
    assert swift_td.eligibility_trace[1].abs().sum() == 0.0
    assert swift_td.eligibility_trace[3].abs().sum() == 0.0
    assert swift_td.eligibility_trace[5].abs().sum() == 0.0

    # Other indices should remain unchanged
    assert torch.allclose(swift_td.eligibility_trace[0], trace_before[0])
    assert torch.allclose(swift_td.eligibility_trace[2], trace_before[2])


def test_swifttd_reset_empty_indices():
    """Test reset with empty indices tensor."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Run some updates
    features = torch.randn(4, 5)
    reward = torch.randn(4)
    swift_td(features, reward)

    trace_before = swift_td.eligibility_trace.clone()

    # Reset with empty indices (should do nothing)
    swift_td.reset(torch.tensor([], dtype=torch.long))

    assert torch.allclose(swift_td.eligibility_trace, trace_before)


# ============================================================================
# T019: Device Handling Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swifttd_cuda_device():
    """Test SwiftTD on CUDA device."""
    device = torch.device("cuda")
    swift_td = SwiftTD(feature_dim=10, alpha=0.1, gamma=0.99, lambda_=0.9).to(device)

    features = torch.randn(16, 10, device=device)
    reward = torch.randn(16, device=device)

    value = swift_td(features, reward)

    assert value.device == device
    assert swift_td.weights.device == device
    assert swift_td.eligibility_trace.device == device


def test_swifttd_cpu_device():
    """Test SwiftTD on CPU device."""
    device = torch.device("cpu")
    swift_td = SwiftTD(feature_dim=10, alpha=0.1, gamma=0.99, lambda_=0.9).to(device)

    features = torch.randn(16, 10, device=device)
    reward = torch.randn(16, device=device)

    value = swift_td(features, reward)

    assert value.device == device
    assert swift_td.weights.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swifttd_device_transfer():
    """Test transferring SwiftTD between devices."""
    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9)

    # Start on CPU
    assert swift_td.weights.device == torch.device("cpu")

    # Move to CUDA
    swift_td = swift_td.to("cuda")
    assert swift_td.weights.device.type == "cuda"
    assert swift_td.eligibility_trace.device.type == "cuda"

    # Move back to CPU
    swift_td = swift_td.to("cpu")
    assert swift_td.weights.device == torch.device("cpu")


def test_swifttd_mixed_device_inputs_raises():
    """Test that mixed device inputs raise error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    swift_td = SwiftTD(feature_dim=5, alpha=0.1, gamma=0.99, lambda_=0.9).to("cuda")

    # Features on CUDA, reward on CPU (mismatched)
    features = torch.randn(4, 5, device="cuda")
    reward = torch.randn(4, device="cpu")

    with pytest.raises(RuntimeError):
        swift_td(features, reward)
