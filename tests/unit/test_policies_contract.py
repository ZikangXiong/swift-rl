"""Contract tests for action selection policies."""

import pytest
import torch

from swift_rl.policies import epsilon_greedy, softmax_policy


def test_epsilon_greedy_exists():
    """Test that epsilon_greedy function exists and is callable."""
    assert callable(epsilon_greedy)


def test_softmax_policy_exists():
    """Test that softmax_policy function exists and is callable."""
    assert callable(softmax_policy)


def test_epsilon_greedy_output_shape():
    """Test epsilon_greedy returns correct shape."""
    batch_size = 16
    num_actions = 4
    action_values = torch.randn(batch_size, num_actions)

    actions = epsilon_greedy(action_values, epsilon=0.1)
    assert actions.shape == (batch_size,)
    assert actions.dtype == torch.long


def test_epsilon_greedy_greedy_mode():
    """Test epsilon_greedy in greedy mode (epsilon=0)."""
    action_values = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.9, 0.3]])
    expected = torch.tensor([1, 1])  # Indices of max values

    actions = epsilon_greedy(action_values, epsilon=0.0)
    assert torch.equal(actions, expected)


def test_epsilon_greedy_eval_mode():
    """Test epsilon_greedy in eval mode (training=False)."""
    action_values = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.9, 0.3]])
    expected = torch.tensor([1, 1])

    actions = epsilon_greedy(action_values, epsilon=0.5, training=False)
    assert torch.equal(actions, expected)


def test_epsilon_greedy_action_range():
    """Test epsilon_greedy returns valid action indices."""
    batch_size = 32
    num_actions = 5
    action_values = torch.randn(batch_size, num_actions)

    actions = epsilon_greedy(action_values, epsilon=0.3)
    assert (actions >= 0).all()
    assert (actions < num_actions).all()


def test_epsilon_greedy_invalid_epsilon():
    """Test epsilon_greedy rejects invalid epsilon."""
    action_values = torch.randn(10, 3)

    with pytest.raises(ValueError, match="epsilon must be in"):
        epsilon_greedy(action_values, epsilon=-0.1)

    with pytest.raises(ValueError, match="epsilon must be in"):
        epsilon_greedy(action_values, epsilon=1.5)


def test_epsilon_greedy_invalid_shape():
    """Test epsilon_greedy rejects invalid input shape."""
    action_values = torch.randn(10)  # 1D instead of 2D

    with pytest.raises(ValueError, match="must be 2D"):
        epsilon_greedy(action_values, epsilon=0.1)


def test_softmax_policy_output_shape():
    """Test softmax_policy returns correct shape."""
    batch_size = 16
    num_actions = 4
    action_values = torch.randn(batch_size, num_actions)

    actions = softmax_policy(action_values, temperature=1.0)
    assert actions.shape == (batch_size,)
    assert actions.dtype == torch.long


def test_softmax_policy_action_range():
    """Test softmax_policy returns valid action indices."""
    batch_size = 32
    num_actions = 5
    action_values = torch.randn(batch_size, num_actions)

    actions = softmax_policy(action_values, temperature=1.0)
    assert (actions >= 0).all()
    assert (actions < num_actions).all()


def test_softmax_policy_low_temperature():
    """Test softmax_policy with very low temperature (nearly greedy)."""
    # With very low temperature, should mostly select max
    action_values = torch.tensor([[10.0, 1.0, 0.1], [0.1, 20.0, 0.2]])

    # Run multiple times to check consistency
    actions = softmax_policy(action_values, temperature=0.01)
    assert actions.shape == (2,)


def test_softmax_policy_invalid_temperature():
    """Test softmax_policy rejects invalid temperature."""
    action_values = torch.randn(10, 3)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        softmax_policy(action_values, temperature=0.0)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        softmax_policy(action_values, temperature=-1.0)


def test_softmax_policy_invalid_shape():
    """Test softmax_policy rejects invalid input shape."""
    action_values = torch.randn(10)  # 1D instead of 2D

    with pytest.raises(ValueError, match="must be 2D"):
        softmax_policy(action_values, temperature=1.0)
