# Data Model: Swift-TD and Swift-SARSA

## Overview

This document describes the entities, their attributes, state management, and
relationships for the Swift-TD and Swift-SARSA implementation.

## Entities

### SwiftTD (PyTorch Module)

**Purpose**: Temporal difference prediction with adaptive step-size optimization

**Attributes**:
- `feature_dim: int` - Dimension of input feature vectors
- `alpha: float` - Base learning rate (α)
- `gamma: float` - Discount factor (γ)
- `lambda_: float` - Eligibility trace decay (λ)
- `weights: nn.Parameter` - Learnable weight vector, shape (feature_dim,)
- `eligibility_trace: torch.Tensor` - Eligibility trace buffer, shape (batch, feature_dim)
- `step_sizes: torch.Tensor` - Adaptive step-size state, shape (batch, feature_dim)
- `prev_value: torch.Tensor` - Previous value estimate, shape (batch,)

**Methods**:
- `__init__(feature_dim, alpha, gamma, lambda_)` - Initialize module
- `forward(features, reward, terminal=None)` - Compute value and perform update
- `reset(batch_indices=None)` - Reset internal state for specified batch indices
- `_compute_td_error(reward, value, prev_value)` - Helper for TD error
- `_update_eligibility_trace(features, gamma, lambda_)` - Helper for trace update
- `_update_step_sizes(features)` - Helper for adaptive step-size
- `_update_weights(td_error, trace, step_sizes)` - Helper for weight update

**Invariants**:
- feature_dim > 0
- 0 < alpha ≤ 1
- 0 ≤ gamma ≤ 1
- 0 ≤ lambda_ ≤ 1
- eligibility_trace.shape == (batch, feature_dim)
- step_sizes.shape == (batch, feature_dim)
- weights.shape == (feature_dim,)

**State Transitions**:
1. Initialization → Ready (weights random, traces/step-sizes zero)
2. Ready → Computing (forward pass)
3. Computing → Updating (backward hook)
4. Updating → Ready (state updated)
5. Any state → Ready (on reset)

### SwiftSARSA (PyTorch Module)

**Purpose**: On-policy control with discrete actions using Swift-TD per action

**Attributes**:
- `feature_dim: int` - Dimension of input feature vectors
- `num_actions: int` - Number of discrete actions
- `alpha: float` - Base learning rate (α)
- `gamma: float` - Discount factor (γ)
- `lambda_: float` - Eligibility trace decay (λ)
- `weights: nn.Parameter` - Weight matrix, shape (num_actions, feature_dim)
- `eligibility_traces: torch.Tensor` - Traces per action, shape (batch, num_actions, feature_dim)
- `step_sizes: torch.Tensor` - Step-sizes per action, shape (batch, num_actions, feature_dim)
- `prev_action_values: torch.Tensor` - Previous action-values, shape (batch, num_actions)

**Methods**:
- `__init__(feature_dim, num_actions, alpha, gamma, lambda_)` - Initialize module
- `forward(features, reward, action, terminal=None)` - Compute action-values and update selected action
- `reset(batch_indices=None)` - Reset internal state for specified batch indices
- `_compute_action_values(features)` - Helper for Q-value computation
- `_compute_td_error(reward, action_values, prev_action_values, action)` - Helper for TD error
- `_update_eligibility_trace_for_action(features, action, gamma, lambda_)` - Helper for trace
- `_update_step_sizes_for_action(features, action)` - Helper for step-size
- `_update_weights_for_action(td_error, trace, step_sizes, action)` - Helper for weights

**Invariants**:
- feature_dim > 0
- num_actions > 1
- 0 < alpha ≤ 1
- 0 ≤ gamma ≤ 1
- 0 ≤ lambda_ ≤ 1
- eligibility_traces.shape == (batch, num_actions, feature_dim)
- step_sizes.shape == (batch, num_actions, feature_dim)
- weights.shape == (num_actions, feature_dim)

**State Transitions**:
Similar to SwiftTD but maintains separate state per action

### ActionSelectionUtilities (Stateless Functions)

**Purpose**: Provide policy functions for action selection

**Functions**:

#### epsilon_greedy
- **Signature**: `(action_values: torch.Tensor, epsilon: float, training: bool = True) -> torch.Tensor`
- **Input**: action_values shape (batch, num_actions), epsilon ∈ [0, 1]
- **Output**: selected_actions shape (batch,), dtype long
- **Behavior**:
  - If training: With probability ε sample random, else argmax
  - If not training: Always argmax (greedy)

#### softmax_policy
- **Signature**: `(action_values: torch.Tensor, temperature: float) -> torch.Tensor`
- **Input**: action_values shape (batch, num_actions), temperature > 0
- **Output**: sampled_actions shape (batch,), dtype long
- **Behavior**:
  - Compute probabilities: exp(Q/τ) / Σ exp(Q/τ)
  - Sample from categorical distribution

### UtilityFunctions (Stateless Helpers)

**Purpose**: Common operations shared across modules

**Functions**:

#### validate_tensor_shape
- **Signature**: `(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str) -> None`
- **Behavior**: Raise ValueError if shape mismatch

#### ensure_device_compatibility
- **Signature**: `(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor`
- **Behavior**: Move tensor to module's device if needed

#### masked_reset
- **Signature**: `(buffer: torch.Tensor, indices: Optional[torch.Tensor]) -> None`
- **Behavior**: Zero out buffer at specified batch indices (in-place)

## Relationships

### SwiftTD ↔ Features
- **Cardinality**: SwiftTD processes many feature vectors (batch)
- **Interaction**: Features input to forward(), used in value computation and trace updates

### SwiftSARSA ↔ Features
- **Cardinality**: SwiftSARSA processes many feature vectors (batch)
- **Interaction**: Features input to forward(), used in action-value computation

### SwiftSARSA ↔ Actions
- **Cardinality**: SwiftSARSA maintains num_actions value functions
- **Interaction**: Action index selects which value function to update

### SwiftSARSA ↔ ActionSelectionUtilities
- **Cardinality**: SwiftSARSA outputs used as input to policy functions
- **Interaction**: SwiftSARSA.forward() returns action-values → policy function → selected action

### Modules ↔ PyTorch Autograd
- **Interaction**: Backward hooks registered on weights
- **Flow**: forward() → loss.backward() → hook triggered → Swift updates applied

## Data Flow

### SwiftTD Training Step
1. **Input**: features (batch, feature_dim), reward (batch,), optional terminal (batch,)
2. **Compute**: value = features @ weights  # (batch,)
3. **TD Error**: δ = reward + γ * value - prev_value  # (batch,)
4. **Backward Hook**:
   - Update eligibility traces: e ← γλe + ∇value
   - Update step-sizes: α_adapted per feature
   - Update weights: w ← w + α_adapted * δ * e
5. **Store**: prev_value ← value
6. **Reset**: If terminal[i], zero traces and step-sizes for environment i
7. **Output**: value (batch,)

### SwiftSARSA Training Step
1. **Input**: features (batch, feature_dim), reward (batch,), action (batch,), optional terminal (batch,)
2. **Compute**: action_values = features @ weights.T  # (batch, num_actions)
3. **TD Error**: δ = reward + γ * action_values[action] - prev_action_values[action]  # (batch,)
4. **Backward Hook**:
   - For selected action only:
     - Update eligibility trace
     - Update step-size
     - Update weights
5. **Store**: prev_action_values ← action_values
6. **Reset**: If terminal[i], zero traces and step-sizes for environment i
7. **Output**: action_values (batch, num_actions)

### Neural Network Integration
1. **Input**: raw_state (batch, state_dim)
2. **Neural Network**: features = network(raw_state)  # (batch, feature_dim)
3. **Swift Layer**: value = swift_module(features, reward, terminal)  # (batch,)
4. **Loss**: loss = criterion(value, target)
5. **Backward**: loss.backward()
   - Gradients flow to network: ∂loss/∂network_weights
   - Swift updates applied to swift_module.weights
6. **Optimizer**: optimizer.step() updates network_weights (not swift_module.weights)

## Validation Rules

### Input Validation (SwiftTD)
- features.ndim == 2 and features.shape[1] == feature_dim
- reward.ndim == 1 and reward.shape[0] == features.shape[0]
- If terminal provided: terminal.ndim == 1, terminal.dtype == bool

### Input Validation (SwiftSARSA)
- features.ndim == 2 and features.shape[1] == feature_dim
- reward.ndim == 1 and reward.shape[0] == features.shape[0]
- action.ndim == 1 and action.dtype == long
- 0 ≤ action < num_actions
- If terminal provided: terminal.ndim == 1, terminal.dtype == bool

### Hyperparameter Validation (Both)
- alpha > 0 and alpha ≤ 1
- gamma >= 0 and gamma ≤ 1
- lambda_ >= 0 and lambda_ ≤ 1
- feature_dim > 0
- For SwiftSARSA: num_actions > 1

## Memory Layout

### SwiftTD Memory per Batch
- weights: feature_dim floats (shared across batch)
- eligibility_trace: batch × feature_dim floats
- step_sizes: batch × feature_dim floats
- prev_value: batch floats
- **Total per environment**: ~2 × feature_dim floats

### SwiftSARSA Memory per Batch
- weights: num_actions × feature_dim floats (shared across batch)
- eligibility_traces: batch × num_actions × feature_dim floats
- step_sizes: batch × num_actions × feature_dim floats
- prev_action_values: batch × num_actions floats
- **Total per environment**: ~2 × num_actions × feature_dim floats

### Scaling Example
- batch=128, feature_dim=256, num_actions=4 (SwiftSARSA)
- Memory: 128 × 2 × 4 × 256 × 4 bytes = ~1 MB
- Acceptable for GPU

## Device Handling

All state tensors (weights, traces, step_sizes) managed via PyTorch's device system:
- Weights are nn.Parameter → move with module.to(device)
- Traces and step_sizes are buffers → move with module.to(device)
- Input tensors validated for device compatibility
- No explicit CPU↔GPU transfers in forward/backward

## Summary

The data model supports:
- Batch processing (up to 128 environments)
- GPU-compatible state management
- Dual-mode usage (standalone and neural network integration)
- Clean separation between modules and utilities
- Type-safe interfaces with clear validation rules
