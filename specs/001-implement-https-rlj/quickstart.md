# Quickstart: Swift-TD and Swift-SARSA

## Installation

```bash
# Clone repository
git clone <repo-url>
cd swift-rl

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## Quick Examples

### Example 1: Standalone Swift-TD for Prediction

```python
import torch
from swift_rl import SwiftTD

# Setup
batch_size = 32
feature_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Swift-TD module
swift_td = SwiftTD(
    feature_dim=feature_dim,
    alpha=0.01,        # Learning rate
    gamma=0.99,        # Discount factor
    lambda_=0.9        # Eligibility trace decay
).to(device)

# Training loop
for episode in range(100):
    # Get initial features from environment
    features = env.reset()  # Shape: (batch, feature_dim)
    features = torch.from_numpy(features).float().to(device)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while not done.all():
        # Get next state and reward
        next_features, reward, step_done = env.step()

        next_features = torch.from_numpy(next_features).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        step_done = torch.from_numpy(step_done).bool().to(device)

        # Compute value (Swift updates happen in backward)
        value = swift_td(next_features, reward, terminal=step_done)

        # Optional: compute loss for gradient flow (if needed)
        # loss = some_criterion(value, target)
        # loss.backward()

        done = done | step_done

    # Traces reset automatically when terminal=True

print(f"Final weights shape: {swift_td.weights.shape}")
```

### Example 2: Standalone Swift-SARSA for Control

```python
import torch
from swift_rl import SwiftSARSA
from swift_rl.policies import epsilon_greedy

# Setup
batch_size = 16
feature_dim = 32
num_actions = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Swift-SARSA module
swift_sarsa = SwiftSARSA(
    feature_dim=feature_dim,
    num_actions=num_actions,
    alpha=0.01,
    gamma=0.95,
    lambda_=0.8
).to(device)

# Training loop
epsilon = 0.1

for episode in range(200):
    features = env.reset()
    features = torch.from_numpy(features).float().to(device)

    # Select initial action
    with torch.no_grad():
        action_values = swift_sarsa(features, reward=None, action=None)
        action = epsilon_greedy(action_values, epsilon=epsilon, training=True)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while not done.all():
        # Execute action, get next state and reward
        next_features, reward, step_done = env.step(action.cpu().numpy())

        next_features = torch.from_numpy(next_features).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        step_done = torch.from_numpy(step_done).bool().to(device)

        # Compute next action-values (with Swift update for previous action)
        next_action_values = swift_sarsa(
            next_features,
            reward=reward,
            action=action,
            terminal=step_done
        )

        # Select next action
        with torch.no_grad():
            action = epsilon_greedy(next_action_values, epsilon=epsilon, training=True)

        done = done | step_done

print(f"Learned {num_actions} action-value functions")
```

### Example 3: Swift-TD as Final Layer of Neural Network

```python
import torch
import torch.nn as nn
from swift_rl import SwiftTD

# Setup
state_dim = 128
feature_dim = 64
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature extraction network
feature_network = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, feature_dim)
).to(device)

# Swift-TD as final layer
swift_td_layer = SwiftTD(
    feature_dim=feature_dim,
    alpha=0.01,
    gamma=0.99,
    lambda_=0.9
).to(device)

# Optimizer for feature network only (Swift-TD updates its own weights)
optimizer = torch.optim.Adam(feature_network.parameters(), lr=0.001)

# Training loop
for episode in range(100):
    state = env.reset()
    state = torch.from_numpy(state).float().to(device)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while not done.all():
        next_state, reward, step_done = env.step()

        next_state = torch.from_numpy(next_state).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        step_done = torch.from_numpy(step_done).bool().to(device)

        # Forward pass through feature network
        features = feature_network(next_state)  # Gradients tracked

        # Swift-TD layer (performs Swift updates in backward)
        value = swift_td_layer(features, reward, terminal=step_done)

        # Optional: add supervision loss
        # loss = (value - target_value).pow(2).mean()
        # optimizer.zero_grad()
        # loss.backward()  # Gradients flow back to feature_network
        # optimizer.step()

        done = done | step_done

print("Feature network and Swift-TD trained together")
```

### Example 4: Swift-SARSA with Softmax Policy

```python
import torch
from swift_rl import SwiftSARSA
from swift_rl.policies import softmax_policy

# Setup
batch_size = 32
feature_dim = 16
num_actions = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create module
swift_sarsa = SwiftSARSA(
    feature_dim=feature_dim,
    num_actions=num_actions,
    alpha=0.02,
    gamma=0.99,
    lambda_=0.85
).to(device)

# Training with temperature-based exploration
temperature = 1.0

for episode in range(150):
    features = env.reset()
    features = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        action_values = swift_sarsa(features, reward=None, action=None)
        action = softmax_policy(action_values, temperature=temperature)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while not done.all():
        next_features, reward, step_done = env.step(action.cpu().numpy())

        next_features = torch.from_numpy(next_features).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        step_done = torch.from_numpy(step_done).bool().to(device)

        # Update with current action
        next_action_values = swift_sarsa(
            next_features,
            reward=reward,
            action=action,
            terminal=step_done
        )

        # Sample next action
        with torch.no_grad():
            action = softmax_policy(next_action_values, temperature=temperature)

        done = done | step_done

    # Decay temperature for less exploration over time
    temperature *= 0.99

print("Training complete with softmax policy")
```

## Manual State Reset

```python
from swift_rl import SwiftTD

swift_td = SwiftTD(feature_dim=64, alpha=0.01, gamma=0.99, lambda_=0.9)

# Reset all batch indices
swift_td.reset()

# Reset specific environments in batch
batch_indices = torch.tensor([0, 5, 10], dtype=torch.long)
swift_td.reset(batch_indices=batch_indices)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swift_rl --cov-report=html

# Run specific test file
pytest tests/unit/test_swift_td.py

# Run integration tests only
pytest tests/integration/
```

## Code Formatting

```bash
# Format all code
black .
isort .

# Type check
mypy swift_rl/
```

## Validation Checklist

After running these examples, verify:
- [ ] Swift-TD computes state values from features
- [ ] Swift-SARSA computes action-values for discrete actions
- [ ] Epsilon-greedy policy selects actions from action-values
- [ ] Softmax policy samples actions with temperature control
- [ ] Manual reset clears eligibility traces and step-sizes
- [ ] Automatic reset triggered by terminal flag
- [ ] Neural network integration maintains gradient flow
- [ ] Both modules work on GPU when available
- [ ] Batch processing handles multiple environments correctly
- [ ] Functions respect 20-line limit (check source)
- [ ] Type annotations present on all functions (check with mypy)
- [ ] Code formatted with black and isort

## Next Steps

1. Read `data-model.md` for detailed entity descriptions
2. Review `research.md` for algorithm foundations
3. Explore `tests/` for comprehensive test examples
4. Check `examples/` for more usage patterns
5. Refer to docstrings for API documentation

## Common Issues

**Issue**: Module not updating weights
- **Solution**: Ensure reward and action (for SARSA) are provided in forward()

**Issue**: Device mismatch errors
- **Solution**: Move all inputs to same device as module with `.to(device)`

**Issue**: Out of memory on GPU
- **Solution**: Reduce batch size or feature dimensions

**Issue**: No gradient flow to feature network
- **Solution**: Ensure feature computation is not in `torch.no_grad()` context

## Performance Tips

1. Use GPU for batch sizes > 16
2. Keep feature_dim as power of 2 for GPU efficiency (64, 128, 256)
3. Adjust batch size based on available GPU memory
4. Use in-place operations are already optimized internally
5. Avoid unnecessary `.cpu()` calls in training loop
