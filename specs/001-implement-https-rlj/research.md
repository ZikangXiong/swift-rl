# Research: Swift-TD and Swift-SARSA Implementation

## Algorithm Foundations

### Swift-TD (Javed, Sharifnassab, Sutton - RLC 2024 Best Paper)

**Decision**: Implement Swift-TD as extension of True Online TD(λ) with adaptive step-size

**Core Algorithm Components**:
1. **TD Error Computation**: δ'_t = r_t + γ v_{t-1,t} - v_{t-2,t-1}
2. **Eligibility Trace**: Following True Online TD(λ) update rules
3. **Adaptive Step-Size**: Dynamic learning rate adjustment per feature
4. **Bounded Learning Rate**: Prevent divergence with effective rate bounds
5. **Step-Size Decay**: Gradual reduction over time

**Rationale**: Swift-TD improves upon standard TD(λ) by adapting step-sizes per feature,
making it robust to high-dimensional noisy features. The adaptive mechanism allows the
algorithm to learn relevant signals without prior knowledge of feature structure.

**Alternatives Considered**:
- Standard TD(λ): Less robust, no adaptive step-size
- True Online TD(λ): Better than standard but lacks adaptive optimization
- Neural network-based TD: More complex, violates modular design principle

### Swift-SARSA (Javed, Sutton - arXiv 2025)

**Decision**: Implement Swift-SARSA by applying Swift-TD principles to each action's value function

**Core Algorithm Components**:
1. **Multi-Action Value Functions**: Separate weight vector per action
2. **Action-Value Computation**: v_{t-1,t}^j = Σ_i w_{t-1}^j[i] φ_t[i]
3. **Per-Action Swift Updates**: Apply Swift-TD mechanism to each action independently
4. **On-Policy Control**: SARSA-style updates with current policy

**Rationale**: Extends Swift-TD's robustness to control problems. Each action maintains
its own Swift-TD learner, enabling efficient on-policy control with discrete actions.

**Alternatives Considered**:
- Single shared adaptive mechanism: Less flexible, doesn't capture per-action dynamics
- Q-learning variant: Off-policy, different use case
- Actor-critic: More complex, requires separate policy network

## PyTorch Integration Strategy

### Custom Backward Hook Approach

**Decision**: Use PyTorch's backward hooks to integrate Swift updates

**Implementation Strategy**:
1. Modules inherit from nn.Module
2. Weights registered as nn.Parameter for PyTorch compatibility
3. Register backward hooks to intercept gradient computation
4. Within hook: compute TD error, update eligibility traces, adjust step-sizes
5. Modify gradients to reflect Swift-specific updates
6. Allow gradient flow to earlier layers when used in neural networks

**Rationale**: Backward hooks provide clean integration with PyTorch's autograd while
allowing custom update logic. This enables both standalone usage and neural network
integration without modifying PyTorch internals.

**Alternatives Considered**:
- Custom autograd Function: More low-level, harder to maintain gradient flow
- Manual optimizer: Doesn't integrate with backward(), breaks PyTorch conventions
- Separate update step: Awkward API, requires manual synchronization

### State Management

**Decision**: Store eligibility traces and step-size state as buffers (not parameters)

**Implementation**:
- Use register_buffer() for traces and step-size state
- Buffers move with module to GPU/CPU automatically
- Not treated as learnable parameters by optimizers
- Reset via dedicated method or automatic terminal detection

**Rationale**: Buffers are perfect for non-learnable state that needs device
synchronization. Prevents optimizers from modifying algorithm state while maintaining
PyTorch device management benefits.

## Batch Processing Design

### Vectorized Operations

**Decision**: Full vectorization with batch dimension throughout

**Design**:
- Input shape: (batch, features) for Swift-TD
- Input shape: (batch, features) for Swift-SARSA
- Output shape: (batch,) for Swift-TD values
- Output shape: (batch, num_actions) for Swift-SARSA action-values
- Internal state shapes: (batch, features) for traces, (batch, features) for step-sizes
- All operations vectorized across batch dimension

**Rationale**: Vectorization essential for GPU efficiency and batch processing up to 128
environments. PyTorch's broadcasting and reduction operations eliminate need for Python
loops.

**Alternatives Considered**:
- Loop over batch: Too slow, defeats GPU purpose
- Single-sample only: Doesn't meet batch requirement
- Dynamic batching: Adds complexity without benefit

### Terminal State Handling

**Decision**: Optional terminal flag with automatic reset

**Implementation**:
- forward() accepts optional terminal tensor (batch,)
- When terminal[i] == True, reset traces and step-sizes for environment i
- Manual reset() method for explicit control
- Masked operations preserve non-terminal environment states

**Rationale**: Supports both automatic episode boundary handling and manual control.
Masked updates ensure one environment's terminal doesn't affect others in batch.

## Testing Strategy

### Unit Tests

**Coverage**:
- Module initialization
- Forward pass (value computation)
- Backward pass (Swift updates)
- State reset (manual and automatic)
- Device placement (CPU/GPU)
- Batch processing
- Edge cases (λ=0, λ=1, invalid inputs)

### Integration Tests

**Coverage**:
- Convergence on simple prediction task (Swift-TD)
- Policy improvement on simple control task (Swift-SARSA)
- Neural network integration (gradient flow verification)
- Comparison with baseline TD(λ) and SARSA(λ)
- GPU vs CPU consistency
- Batch size scaling (1 to 128)

### Test Environments

**Decision**: Implement minimal test environments in fixtures

- Linear prediction task (for Swift-TD)
- Tabular control task (for Swift-SARSA)
- Random MDP for stress testing

**Rationale**: Lightweight environments sufficient for algorithm verification without
external dependencies (e.g., OpenAI Gym). Keeps tests fast and deterministic.

## Action Selection Utilities

### Epsilon-Greedy

**Decision**: Separate function taking action-values and epsilon

**Signature**: `epsilon_greedy(action_values: torch.Tensor, epsilon: float) -> torch.Tensor`

**Implementation**:
- With probability ε: sample random action
- With probability 1-ε: select argmax action
- Supports batched action-values: (batch, num_actions)
- Returns action indices: (batch,)

**Rationale**: Decouples action selection from value computation, allowing users to
implement custom policies. Pure function with no state.

### Softmax Policy

**Decision**: Separate function with temperature parameter

**Signature**: `softmax_policy(action_values: torch.Tensor, temperature: float) -> torch.Tensor`

**Implementation**:
- Compute probabilities: P(a) = exp(v[a]/τ) / Σ_j exp(v[j]/τ)
- Sample from categorical distribution
- Supports batched action-values: (batch, num_actions)
- Returns sampled action indices: (batch,)

**Rationale**: Temperature-controlled exploration. Separate from module for flexibility.

## Development Dependencies

### Core Dependencies

**Decision**: Minimal production dependencies

- PyTorch (≥2.0, with CUDA support)
- Python standard library (typing, dataclasses if needed)

**Rationale**: Keep package lightweight. PyTorch provides all necessary tensor operations.

### Development Dependencies

**Decision**: Standard Python ML development stack

- pytest: Testing framework
- pytest-cov: Coverage tracking
- black: Code formatting
- isort: Import sorting
- mypy: Static type checking
- numpy: Test data generation (optional, can use torch)

**Rationale**: Aligns with constitutional requirements and Python ecosystem standards.

## Performance Considerations

### GPU Optimization

**Strategies**:
1. Keep all state on device (no CPU-GPU transfers in hot path)
2. Fused operations where possible (e.g., eligibility trace + weight update)
3. In-place operations for state updates (traces.mul_(), not traces * scalar)
4. Avoid python loops, use masked operations for terminal handling

### Memory Efficiency

**Strategies**:
1. Reuse buffers for intermediate computations
2. In-place operations reduce allocations
3. Avoid unnecessary copies (use views where safe)
4. State size scales with (batch * features), acceptable for batch=128

**Rationale**: Memory is bottleneck for large batch sizes. In-place ops and buffer reuse
critical for scaling to 128 environments.

## Documentation Plan

### Module Docstrings

- Algorithm overview
- Mathematical formulation
- Usage examples (standalone and neural network modes)
- Parameter descriptions

### Function Docstrings

- Args section with types
- Returns section with type and meaning
- Examples demonstrating typical usage
- References to equations in algorithm papers

### Shape Comments

- Every tensor operation annotated with shapes
- Example: `td_error = reward + gamma * next_value  # (batch,)`
- Helps readers understand tensor dimensions

**Rationale**: Comprehensive documentation supports algorithmic clarity principle and
makes code accessible to RL researchers.

## Summary

All technical decisions made with constitutional principles in mind:
- Modular design: Clear separation (Swift-TD, Swift-SARSA, policies, utils)
- GPU-first: Vectorized operations, device-aware buffers, batch processing
- Type safety: Complete annotations planned
- Formatting: black + isort tooling selected
- Algorithmic clarity: Implementation mirrors mathematical formulations

No NEEDS CLARIFICATION items remain. Ready to proceed to Phase 1.
