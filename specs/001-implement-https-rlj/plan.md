
# Implementation Plan: Swift-TD and Swift-SARSA PyTorch Modules

**Branch**: `001-implement-https-rlj` | **Date**: 2025-10-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/zikangxiong/Documents/dev/swfit-rl/specs/001-implement-https-rlj/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code, or `AGENTS.md` for all other agents).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Implement Swift-TD and Swift-SARSA as two separate PyTorch modules that support both
standalone usage and integration as the final layer of neural networks. Swift-TD
provides temporal difference prediction with adaptive step-size optimization, while
Swift-SARSA extends this to on-policy control with discrete actions. Both modules
integrate algorithm-specific updates (TD errors, eligibility traces, adaptive step-size)
into PyTorch's backward pass while maintaining gradient flow for earlier layers.

## Technical Context
**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch (latest stable), typing, numpy (for testing)
**Storage**: N/A (in-memory state management only)
**Testing**: pytest, pytest-cov (coverage tracking)
**Target Platform**: Linux/macOS/Windows with CUDA support (GPU-first design, CPU fallback)
**Project Type**: single (Python package)
**Performance Goals**: Efficient batch processing up to 128 parallel environments, GPU-optimized tensor operations
**Constraints**: Functions ≤20 lines (constitutional requirement), complete type annotations, black+isort formatting
**Scale/Scope**: Two main modules (Swift-TD, Swift-SARSA), action selection utilities, comprehensive test suite

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Modular Design (20-line function limit)
**Status**: PASS
- All functions will be decomposed to ≤20 lines (excluding docstrings)
- PyTorch module methods (forward, reset, etc.) naturally fit this constraint
- Complex operations (TD error computation, eligibility updates) will be broken into helper functions

### II. GPU-First Architecture
**Status**: PASS
- All tensor operations designed for GPU execution
- Batch processing supports up to 128 parallel environments
- Minimize CPU-GPU transfers by keeping state on device
- Vectorized operations eliminate Python loops over dimensions

### III. Type Safety
**Status**: PASS
- All function signatures will include complete type annotations
- Use torch.Tensor, Optional, Union, Literal for clarity
- Type hints for hyperparameters (float), dimensions (int), flags (bool)

### IV. Code Formatting (NON-NEGOTIABLE)
**Status**: PASS
- black and isort will be applied to all code before commit
- Pre-commit hooks will enforce formatting
- requirements-dev.txt will include black, isort, mypy

### V. Algorithmic Clarity
**Status**: PASS
- Implementation directly reflects Swift-TD and Swift-SARSA mathematical formulations
- Docstrings will include algorithm equations (TD error, eligibility trace updates)
- Variable names match paper notation where possible (α, γ, λ, δ, traces)
- Shape comments for tensor operations (e.g., # (batch, features))

### Testing Requirements
**Status**: PASS
- Unit tests for all public methods
- Integration tests verifying algorithm correctness against baselines
- GPU execution path tests
- Target: >80% test coverage

### Documentation Requirements
**Status**: PASS
- Module docstrings with purpose and usage
- Function docstrings with Args, Returns, Examples
- Mathematical formulations in algorithm docstrings
- Shape comments for complex tensor operations

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
swift_rl/
├── __init__.py
├── swift_td.py          # Swift-TD module implementation
├── swift_sarsa.py       # Swift-SARSA module implementation
├── policies.py          # Action selection utilities (epsilon-greedy, softmax)
└── utils.py             # Shared utilities (tensor validation, device handling)

tests/
├── unit/
│   ├── test_swift_td.py
│   ├── test_swift_sarsa.py
│   ├── test_policies.py
│   └── test_utils.py
├── integration/
│   ├── test_swift_td_convergence.py
│   ├── test_swift_sarsa_control.py
│   └── test_neural_network_integration.py
└── fixtures/
    └── environments.py   # Test RL environments

examples/
├── standalone_swift_td.py
├── standalone_swift_sarsa.py
├── neural_network_swift_td.py
└── neural_network_swift_sarsa.py

requirements.txt          # PyTorch and minimal dependencies
requirements-dev.txt      # black, isort, mypy, pytest, pytest-cov
setup.py or pyproject.toml
README.md
```

**Structure Decision**: Single Python package structure. The `swift_rl/` directory
contains core modules (swift_td.py, swift_sarsa.py, policies.py, utils.py). Tests
are organized by type (unit, integration) with fixtures for reusable test
environments. Examples demonstrate both standalone and neural network usage modes.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data-model.md, quickstart.md)
- Contract interface files → Contract validation test tasks [P]
- Each entity (SwiftTD, SwiftSARSA, policies, utils) → Implementation task
- Each quickstart example → Integration test task
- Constitutional requirements → Formatting and type-checking tasks

**Specific Task Categories**:

1. **Setup Tasks** (Phase 3.1):
   - Create project structure (swift_rl/, tests/, examples/)
   - Initialize requirements.txt and requirements-dev.txt
   - Configure pyproject.toml or setup.py
   - Set up pre-commit hooks for black/isort

2. **Contract Validation Tests** (Phase 3.2 - TDD):
   - Test SwiftTD interface contract [P]
   - Test SwiftSARSA interface contract [P]
   - Test policies interface contract [P]
   - These tests MUST fail initially (no implementation yet)

3. **Core Implementation** (Phase 3.3):
   - Implement swift_rl/utils.py (validation, device handling) [P]
   - Implement swift_rl/swift_td.py (SwiftTD module)
   - Implement swift_rl/swift_sarsa.py (SwiftSARSA module)
   - Implement swift_rl/policies.py (epsilon_greedy, softmax_policy) [P]
   - Implement swift_rl/__init__.py (package exports) [P]

4. **Unit Tests** (Phase 3.4):
   - Unit tests for SwiftTD (initialization, forward, reset, device) [P]
   - Unit tests for SwiftSARSA (initialization, forward, reset, device) [P]
   - Unit tests for policies (epsilon_greedy, softmax_policy) [P]
   - Unit tests for utils [P]

5. **Integration Tests** (Phase 3.5):
   - Integration test: SwiftTD convergence on prediction task
   - Integration test: SwiftSARSA control on simple MDP
   - Integration test: Neural network integration (gradient flow)
   - Integration test: Batch processing scaling (1 to 128)
   - Integration test: GPU vs CPU consistency

6. **Examples & Documentation** (Phase 3.6):
   - Create examples/standalone_swift_td.py [P]
   - Create examples/standalone_swift_sarsa.py [P]
   - Create examples/neural_network_swift_td.py [P]
   - Create examples/neural_network_swift_sarsa.py [P]
   - Create README.md with installation and usage
   - Verify docstrings complete with mathematical formulations

7. **Polish & Validation** (Phase 3.7):
   - Run black and isort on all code
   - Run mypy for type checking
   - Run pytest with coverage (target >80%)
   - Verify no function exceeds 20 lines (constitutional check)
   - Run all quickstart examples to validate
   - Performance profiling (batch processing efficiency)

**Ordering Strategy**:
- TDD order: Contract tests before implementation (Phase 3.2 before 3.3)
- Dependency order: utils before modules, modules before examples
- Mark [P] for parallel execution (different files, no dependencies)
- Unit tests can run in parallel with each other
- Integration tests depend on core implementation completing

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md

**Task Breakdown Rationale**:
- Each module (SwiftTD, SwiftSARSA, policies, utils) is a separate file → [P]
- Tests for different modules can run in parallel → [P]
- Examples are independent → [P]
- Setup and polish tasks are sequential (dependencies on previous phases)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (None - all checks PASS)

**Artifacts Generated**:
- [x] plan.md (this file)
- [x] research.md
- [x] data-model.md
- [x] quickstart.md
- [x] contracts/swift_td_interface.py
- [x] contracts/swift_sarsa_interface.py
- [x] contracts/policies_interface.py
- [x] tasks.md (47 implementation tasks)
- [x] CLAUDE.md (agent context file)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
