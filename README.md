# Swift-RL: Fast and Robust Reinforcement Learning

Implementation of **Swift-TD** and **Swift-SARSA** algorithms in PyTorch.

## Overview

Swift-RL provides PyTorch implementations of:
- **Swift-TD**: Temporal difference learning with adaptive step-size optimization (Javed, Sharifnassab, Sutton - RLC 2024 Best Paper)
- **Swift-SARSA**: On-policy control with discrete actions (Javed, Sutton - arXiv 2025)

Both algorithms support:
- Standalone usage as linear reinforcement learning modules
- Integration as the final layer of neural networks
- GPU acceleration and batch processing (up to 128 parallel environments)
- Automatic state reset on episode boundaries

## Installation

```bash
# Clone repository
git clone <repo-url>
cd swift-rl

# Install package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Quick Start

### SwiftTD for Prediction

```python
import torch
from swift_rl import SwiftTD

# Create Swift-TD module
swift_td = SwiftTD(
    feature_dim=64,
    alpha=0.01,      # Learning rate
    gamma=0.99,      # Discount factor
    lambda_=0.9      # Eligibility trace decay
)

# Training loop
for episode in range(100):
    features = env.reset()  # Shape: (batch, feature_dim)
    done = torch.zeros(batch_size, dtype=torch.bool)

    while not done.all():
        next_features, reward, step_done = env.step()

        # Compute value (Swift updates happen automatically)
        value = swift_td(next_features, reward, terminal=step_done)

        done = done | step_done
```

### Action Selection Policies

```python
from swift_rl import epsilon_greedy, softmax_policy

# Epsilon-greedy
action = epsilon_greedy(action_values, epsilon=0.1, training=True)

# Softmax with temperature
action = softmax_policy(action_values, temperature=1.0)
```

## Features

### Swift-TD
- ✅ Adaptive step-size optimization per feature
- ✅ Bounded learning rates for stability
- ✅ Eligibility trace decay (True Online TD(λ))
- ✅ Automatic terminal state handling
- ✅ GPU-optimized batch processing

### Swift-SARSA (Coming Soon)
- 🚧 Multi-action value functions
- 🚧 Per-action Swift-TD updates
- 🚧 On-policy control

### Action Selection
- ✅ Epsilon-greedy policy
- ✅ Softmax (Boltzmann) policy

## Architecture

The implementation follows constitutional principles:
- **Modular Design**: All functions ≤ 20 lines
- **GPU-First**: Vectorized tensor operations
- **Type Safety**: Complete type annotations
- **Code Formatting**: Black + isort enforced
- **Algorithmic Clarity**: Implementation mirrors mathematical formulations

## Examples

See `examples/` directory:
- `standalone_swift_td.py` - Basic Swift-TD usage
- `neural_network_swift_td.py` - Swift-TD as final layer of neural network (coming soon)
- `standalone_swift_sarsa.py` - Basic Swift-SARSA usage (coming soon)
- `neural_network_swift_sarsa.py` - Swift-SARSA with neural network (coming soon)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swift_rl --cov-report=html

# Run specific tests
pytest tests/unit/test_swift_td_contract.py
```

## Development

```bash
# Format code
black .
isort .

# Type check
mypy swift_rl/

# Install pre-commit hooks
pre-commit install
```

## References

- **Swift-TD**: Javed, K., Sharifnassab, A., & Sutton, R. S. (2024). "Swift-TD: Fast and Robust Temporal Difference Learning." RLC 2024 Best Paper.
- **Swift-SARSA**: Javed, K., & Sutton, R. S. (2025). "Swift-Sarsa: Fast and Robust Linear Control." arXiv preprint.

## License

MIT

## Status

🚧 **Early Development**
- ✅ SwiftTD implemented and tested
- 🚧 SwiftSARSA in progress
- 🚧 Neural network integration examples in progress
- 🚧 Full test coverage in progress
