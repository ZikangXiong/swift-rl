"""Contract tests for SwiftSARSA module interface."""

import pytest
import torch

# Test that SwiftSARSA module can be imported
try:
    from swift_rl import SwiftSARSA

    SWIFT_SARSA_AVAILABLE = True
except ImportError:
    SWIFT_SARSA_AVAILABLE = False
    SwiftSARSA = None  # type: ignore


def test_swift_sarsa_contract():
    """Contract: SwiftSARSA module should exist and be callable."""
    assert callable(SwiftSARSA), "SwiftSARSA must be a callable class"


def test_swift_sarsa_initialization():
    """Contract: SwiftSARSA should initialize with required hyperparameters."""
    if not SWIFT_SARSA_AVAILABLE:
        pytest.skip("SwiftSARSA not yet implemented")

    swift_sarsa = SwiftSARSA(
        feature_dim=10, num_actions=4, alpha=0.1, gamma=0.99, lambda_=0.9
    )

    assert hasattr(swift_sarsa, "feature_dim")
    assert hasattr(swift_sarsa, "num_actions")
    assert hasattr(swift_sarsa, "alpha")
    assert hasattr(swift_sarsa, "gamma")
    assert hasattr(swift_sarsa, "lambda_")
    assert hasattr(swift_sarsa, "weights")


def test_swift_sarsa_forward_shape():
    """Contract: SwiftSARSA.forward should return action-values with correct shape."""
    if not SWIFT_SARSA_AVAILABLE:
        pytest.skip("SwiftSARSA not yet implemented")

    batch_size = 8
    feature_dim = 10
    num_actions = 4

    swift_sarsa = SwiftSARSA(
        feature_dim=feature_dim,
        num_actions=num_actions,
        alpha=0.1,
        gamma=0.99,
        lambda_=0.9,
    )

    features = torch.randn(batch_size, feature_dim)
    reward = torch.randn(batch_size)
    action = torch.randint(0, num_actions, (batch_size,))

    action_values = swift_sarsa(features, reward, action)

    assert action_values.shape == (batch_size, num_actions)
    assert action_values.dtype == torch.float32


def test_swift_sarsa_not_implemented():
    """Verify SwiftSARSA is not yet implemented (this test should pass initially)."""
    assert (
        not SWIFT_SARSA_AVAILABLE
    ), "SwiftSARSA should not be implemented yet during TDD phase"
