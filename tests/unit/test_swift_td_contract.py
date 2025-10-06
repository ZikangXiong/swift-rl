"""Contract tests for SwiftTD module."""

import pytest
import torch

# This will fail until SwiftTD is implemented
try:
    from swift_rl import SwiftTD

    SWIFT_TD_AVAILABLE = True
except ImportError:
    SWIFT_TD_AVAILABLE = False


@pytest.mark.skipif(not SWIFT_TD_AVAILABLE, reason="SwiftTD not implemented yet")
def test_swift_td_contract():
    """Test that SwiftTD satisfies the interface contract."""
    # Initialize module
    module = SwiftTD(feature_dim=64, alpha=0.01, gamma=0.99, lambda_=0.9)

    # Check attributes exist
    assert hasattr(module, "feature_dim")
    assert hasattr(module, "alpha")
    assert hasattr(module, "gamma")
    assert hasattr(module, "lambda_")

    # Check methods exist
    assert callable(getattr(module, "forward", None))
    assert callable(getattr(module, "reset", None))

    # Check properties
    assert hasattr(module, "weights")
    assert hasattr(module, "eligibility_trace")
    assert hasattr(module, "step_sizes")


@pytest.mark.skipif(not SWIFT_TD_AVAILABLE, reason="SwiftTD not implemented yet")
def test_swift_td_initialization():
    """Test SwiftTD initialization."""
    feature_dim = 32
    module = SwiftTD(feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9)

    assert module.feature_dim == feature_dim
    assert module.alpha == 0.01
    assert module.gamma == 0.99
    assert module.lambda_ == 0.9


@pytest.mark.skipif(not SWIFT_TD_AVAILABLE, reason="SwiftTD not implemented yet")
def test_swift_td_forward_shape():
    """Test SwiftTD forward pass output shape."""
    batch_size = 16
    feature_dim = 64
    module = SwiftTD(feature_dim=feature_dim, alpha=0.01, gamma=0.99, lambda_=0.9)

    features = torch.randn(batch_size, feature_dim)
    reward = torch.randn(batch_size)

    value = module(features, reward)
    assert value.shape == (batch_size,)


def test_swift_td_not_implemented():
    """Verify SwiftTD is not yet implemented (this test should pass initially)."""
    assert (
        not SWIFT_TD_AVAILABLE
    ), "SwiftTD should not be implemented yet during TDD phase"
