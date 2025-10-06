"""Unit tests for action selection policies."""

import pytest
import torch

from swift_rl.policies import epsilon_greedy, softmax_policy

# ============================================================================
# T024: Epsilon-Greedy Policy Tests
# ============================================================================


def test_epsilon_greedy_deterministic_behavior():
    """Test epsilon_greedy is deterministic with seed."""
    action_values = torch.randn(10, 4)
    epsilon = 0.3

    # Set seed for reproducibility
    torch.manual_seed(42)
    actions1 = epsilon_greedy(action_values, epsilon)

    torch.manual_seed(42)
    actions2 = epsilon_greedy(action_values, epsilon)

    assert torch.equal(actions1, actions2)


def test_epsilon_greedy_exploration_rate():
    """Test epsilon_greedy respects exploration rate approximately."""
    batch_size = 10000
    num_actions = 4
    epsilon = 0.2

    # Create action values where action 0 is always best
    action_values = torch.zeros(batch_size, num_actions)
    action_values[:, 0] = 10.0  # Action 0 has highest value

    actions = epsilon_greedy(action_values, epsilon, training=True)

    # Count non-greedy actions
    # With epsilon=0.2: 20% explore, but 1/4 of random actions will still be action 0
    # Expected non-greedy = epsilon * (num_actions - 1) / num_actions
    # = 0.2 * 3/4 = 0.15
    non_greedy = (actions != 0).float().mean().item()
    expected_non_greedy = epsilon * (num_actions - 1) / num_actions

    # Should be approximately expected (within 2% due to randomness)
    assert abs(non_greedy - expected_non_greedy) < 0.02


def test_epsilon_greedy_greedy_selection():
    """Test epsilon_greedy selects argmax in greedy mode."""
    action_values = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0], [2.0, 2.0, 8.0]])

    # epsilon=0 should be fully greedy
    actions = epsilon_greedy(action_values, epsilon=0.0, training=True)
    expected = torch.tensor([1, 2, 2])
    assert torch.equal(actions, expected)

    # training=False should also be fully greedy
    actions = epsilon_greedy(action_values, epsilon=0.5, training=False)
    assert torch.equal(actions, expected)


def test_epsilon_greedy_all_random():
    """Test epsilon_greedy with epsilon=1 is fully random."""
    batch_size = 1000
    num_actions = 4

    # All same values, so greedy would pick action 0
    action_values = torch.ones(batch_size, num_actions)

    actions = epsilon_greedy(action_values, epsilon=1.0, training=True)

    # Should have roughly uniform distribution over all actions
    action_counts = torch.bincount(actions, minlength=num_actions).float()
    action_probs = action_counts / batch_size

    # Each action should appear ~25% of time (within 5% tolerance)
    for prob in action_probs:
        assert abs(prob - 0.25) < 0.05


def test_epsilon_greedy_batch_independence():
    """Test each batch element is processed independently."""
    action_values = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])

    # With epsilon=0, should get different actions per batch
    actions = epsilon_greedy(action_values, epsilon=0.0)
    assert torch.equal(actions, torch.tensor([0, 1, 2]))


def test_epsilon_greedy_edge_case_single_action():
    """Test epsilon_greedy with single action."""
    action_values = torch.randn(10, 1)
    actions = epsilon_greedy(action_values, epsilon=0.5)

    # With only 1 action, all should be 0
    assert torch.all(actions == 0)


# ============================================================================
# T025: Softmax Policy Tests
# ============================================================================


def test_softmax_policy_deterministic_with_seed():
    """Test softmax_policy is deterministic with seed."""
    action_values = torch.randn(10, 4)
    temperature = 1.0

    torch.manual_seed(42)
    actions1 = softmax_policy(action_values, temperature)

    torch.manual_seed(42)
    actions2 = softmax_policy(action_values, temperature)

    assert torch.equal(actions1, actions2)


def test_softmax_policy_temperature_effect():
    """Test temperature affects action selection."""
    batch_size = 5000
    action_values = torch.tensor([[10.0, 1.0, 1.0]] * batch_size)

    # Very low temperature: nearly greedy (mostly action 0)
    torch.manual_seed(42)
    actions_low = softmax_policy(action_values, temperature=0.01)
    greedy_ratio_low = (actions_low == 0).float().mean().item()

    # High temperature: more random (less action 0)
    torch.manual_seed(42)
    actions_high = softmax_policy(action_values, temperature=10.0)
    greedy_ratio_high = (actions_high == 0).float().mean().item()

    # Low temp should have much higher greedy ratio
    assert greedy_ratio_low > greedy_ratio_high
    assert greedy_ratio_low > 0.95  # Should be nearly always greedy
    assert greedy_ratio_high < 0.6  # Should be much more random


def test_softmax_policy_uniform_values():
    """Test softmax with uniform values gives uniform distribution."""
    batch_size = 10000
    num_actions = 4

    # All actions have same value
    action_values = torch.ones(batch_size, num_actions)

    actions = softmax_policy(action_values, temperature=1.0)

    # Should have roughly uniform distribution
    action_counts = torch.bincount(actions, minlength=num_actions).float()
    action_probs = action_counts / batch_size

    # Each action should appear ~25% of time (within 3% tolerance)
    for prob in action_probs:
        assert abs(prob - 0.25) < 0.03


def test_softmax_policy_respects_preferences():
    """Test softmax respects action value preferences."""
    batch_size = 5000

    # Action 2 has much higher value
    action_values = torch.tensor([[1.0, 1.0, 10.0, 1.0]] * batch_size)

    actions = softmax_policy(action_values, temperature=1.0)

    # Action 2 should be selected most often
    action_counts = torch.bincount(actions, minlength=4).float()

    assert action_counts[2] > action_counts[0]
    assert action_counts[2] > action_counts[1]
    assert action_counts[2] > action_counts[3]


def test_softmax_policy_batch_independence():
    """Test each batch element is sampled independently."""
    # Different best actions per batch element
    action_values = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])

    # With very low temperature, should mostly get best action per batch
    torch.manual_seed(42)
    actions = softmax_policy(action_values, temperature=0.01)

    # Run multiple times to check tendency
    action_counts = torch.zeros(3, 3)
    for _ in range(100):
        acts = softmax_policy(action_values, temperature=0.01)
        for i, a in enumerate(acts):
            action_counts[i, a] += 1

    # Each batch element should mostly select its best action
    assert action_counts[0, 0] > 90  # Batch 0 mostly selects action 0
    assert action_counts[1, 1] > 90  # Batch 1 mostly selects action 1
    assert action_counts[2, 2] > 90  # Batch 2 mostly selects action 2


def test_softmax_policy_numerical_stability():
    """Test softmax handles extreme values without overflow."""
    batch_size = 10
    num_actions = 4

    # Very large values
    action_values = torch.randn(batch_size, num_actions) * 1000

    # Should not raise overflow warnings or errors
    result = softmax_policy(action_values, temperature=1.0)

    assert result.shape == (batch_size,)
    assert (result >= 0).all()
    assert (result < num_actions).all()


def test_softmax_policy_edge_case_single_action():
    """Test softmax with single action."""
    action_values = torch.randn(10, 1)
    actions = softmax_policy(action_values, temperature=1.0)

    # With only 1 action, all should be 0
    assert torch.all(actions == 0)


def test_softmax_policy_temperature_extremes():
    """Test softmax at temperature extremes."""
    action_values = torch.tensor([[5.0, 1.0, 1.0]] * 100)

    # Very low temperature (near-greedy)
    actions_low = softmax_policy(action_values, temperature=0.001)
    assert (actions_low == 0).float().mean() > 0.99

    # Very high temperature (near-uniform)
    torch.manual_seed(42)
    action_values_uniform = torch.tensor([[5.0, 1.0, 1.0]] * 3000)
    actions_high = softmax_policy(action_values_uniform, temperature=100.0)
    counts = torch.bincount(actions_high, minlength=3).float() / 3000

    # Should be closer to uniform than greedy
    assert counts.min() > 0.2  # All actions get selected sometimes
