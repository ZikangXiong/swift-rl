# Tasks: Swift-TD and Swift-SARSA PyTorch Modules

**Input**: Design documents from `/Users/zikangxiong/Documents/dev/swfit-rl/specs/001-implement-https-rlj/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Package root**: `swift_rl/` at repository root
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/fixtures/`
- **Examples**: `examples/` at repository root

## Phase 3.1: Setup
- [x] T001 Create project directory structure (swift_rl/, tests/, examples/)
- [x] T002 Create requirements.txt with PyTorch dependency (torch>=2.0)
- [x] T003 [P] Create requirements-dev.txt with dev tools (pytest, pytest-cov, black, isort, mypy, numpy)
- [x] T004 [P] Create pyproject.toml with project metadata and build configuration
- [x] T005 [P] Create .gitignore for Python projects (__pycache__, *.pyc, .pytest_cache, etc.)
- [x] T006 [P] Create swift_rl/__init__.py placeholder (empty for now)

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T007 [P] Contract test for SwiftTD interface in tests/unit/test_swift_td_contract.py
- [x] T008 [P] Contract test for SwiftSARSA interface in tests/unit/test_swift_sarsa_contract.py
- [x] T009 [P] Contract test for action selection policies in tests/unit/test_policies_contract.py
- [x] T010 [P] Create test fixtures in tests/fixtures/environments.py (simple linear prediction task, tabular MDP)

## Phase 3.3: Core Implementation (ONLY after contract tests are failing)
- [x] T011 [P] Implement utility functions in swift_rl/utils.py (validate_tensor_shape, ensure_device_compatibility, masked_reset)
- [x] T012 Implement SwiftTD module in swift_rl/swift_td.py (__init__, forward, reset, helper methods)
- [x] T013 Implement SwiftSARSA module in swift_rl/swift_sarsa.py (__init__, forward, reset, helper methods)
- [x] T014 [P] Implement action selection policies in swift_rl/policies.py (epsilon_greedy, softmax_policy)
- [x] T015 Update swift_rl/__init__.py to export SwiftTD, SwiftSARSA, and policies

## Phase 3.4: Unit Tests
- [x] T016 [P] Unit tests for SwiftTD initialization in tests/unit/test_swift_td.py
- [x] T017 [P] Unit tests for SwiftTD forward pass in tests/unit/test_swift_td.py
- [x] T018 [P] Unit tests for SwiftTD reset method in tests/unit/test_swift_td.py
- [x] T019 [P] Unit tests for SwiftTD device handling (CPU/GPU) in tests/unit/test_swift_td.py
- [ ] T020 [P] Unit tests for SwiftSARSA initialization in tests/unit/test_swift_sarsa.py
- [ ] T021 [P] Unit tests for SwiftSARSA forward pass in tests/unit/test_swift_sarsa.py
- [ ] T022 [P] Unit tests for SwiftSARSA reset method in tests/unit/test_swift_sarsa.py
- [ ] T023 [P] Unit tests for SwiftSARSA device handling in tests/unit/test_swift_sarsa.py
- [x] T024 [P] Unit tests for epsilon_greedy policy in tests/unit/test_policies.py
- [x] T025 [P] Unit tests for softmax_policy in tests/unit/test_policies.py
- [x] T026 [P] Unit tests for utility functions in tests/unit/test_utils.py

## Phase 3.5: Integration Tests
- [x] T027 Integration test for SwiftTD convergence on prediction task in tests/integration/test_swift_td_convergence.py
- [ ] T028 Integration test for SwiftSARSA control on tabular MDP in tests/integration/test_swift_sarsa_control.py
- [ ] T029 Integration test for neural network integration (gradient flow) in tests/integration/test_neural_network_integration.py
- [ ] T030 Integration test for batch processing scaling (1 to 128 environments) in tests/integration/test_batch_scaling.py
- [ ] T031 Integration test for GPU vs CPU consistency in tests/integration/test_gpu_cpu_consistency.py
- [ ] T032 Integration test for automatic terminal reset in tests/integration/test_terminal_handling.py

## Phase 3.6: Examples & Documentation
- [x] T033 [P] Create standalone SwiftTD example in examples/standalone_swift_td.py
- [x] T034 [P] Create standalone SwiftSARSA example in examples/standalone_swift_sarsa.py
- [x] T035 [P] Create neural network SwiftTD example in examples/neural_network_swift_td.py
- [x] T036 [P] Create neural network SwiftSARSA example in examples/neural_network_swift_sarsa.py
- [x] T037 Create README.md with installation instructions, usage examples, and quickstart
- [x] T038 [P] Verify all docstrings include mathematical formulations and shape comments


## Phase 3.7: Polish & Validation
- [x] T039 Run black on all Python files in swift_rl/, tests/, examples/
- [x] T040 Run isort on all Python files in swift_rl/, tests/, examples/
- [x] T041 Run mypy type checking on swift_rl/ directory
- [x] T042 Run pytest with coverage (pytest --cov=swift_rl --cov-report=html)
- [x] T043 Verify test coverage >80%
- [x] T044 Verify no function exceeds 20 lines (excluding docstrings and blank lines)
- [x] T045 Run all examples to verify they execute without errors
- [ ] T046 [P] Performance profiling for batch processing efficiency
- [ ] T047 Create pre-commit hook configuration (.pre-commit-config.yaml) for black and isort

## Dependencies
**Setup → Tests → Implementation → Unit Tests → Integration → Examples → Polish**

Detailed dependencies:
- T001-T006 (Setup) must complete before any other tasks
- T007-T010 (Contract tests) before T011-T015 (Implementation)
- T011 (utils) before T012-T013 (modules that use utils)
- T012-T015 (Core implementation) before T016-T026 (Unit tests)
- T016-T026 (Unit tests) before T027-T032 (Integration tests)
- T027-T032 (Integration tests) before T033-T036 (Examples)
- T033-T038 (Examples & docs) before T039-T047 (Polish)
- T012 (SwiftTD) blocks T013 (SwiftSARSA uses similar patterns)
- T042-T043 (Coverage) depend on all tests completing

## Parallel Execution Examples

### Parallel Group 1: Setup (after T001 completes)
```bash
# Can run T002-T006 in parallel (different files)
```
Tasks: T002, T003, T004, T005, T006

### Parallel Group 2: Contract Tests
```bash
# Can run T007-T010 in parallel (different test files)
```
Tasks: T007, T008, T009, T010

### Parallel Group 3: Core Implementation (partial)
```bash
# T011 and T014 are independent
# T012-T013 are sequential (T013 may reference T012 patterns)
```
Tasks: T011, T014 (then T012, then T013, then T015)

### Parallel Group 4: Unit Tests
```bash
# All unit test files are independent
```
Tasks: T016, T017, T018, T019, T020, T021, T022, T023, T024, T025, T026

### Parallel Group 5: Examples
```bash
# All example files are independent
```
Tasks: T033, T034, T035, T036, T038

### Parallel Group 6: Formatting
```bash
# Black, isort, mypy can run in parallel
```
Tasks: T039, T040, T041

## Task Breakdown by File

### swift_rl/utils.py
- T011: Implementation

### swift_rl/swift_td.py
- T012: Implementation
- T016-T019: Unit tests

### swift_rl/swift_sarsa.py
- T013: Implementation
- T020-T023: Unit tests

### swift_rl/policies.py
- T014: Implementation
- T024-T025: Unit tests

### swift_rl/__init__.py
- T006: Initial creation
- T015: Update with exports

### tests/unit/test_utils.py
- T026: Unit tests

### tests/unit/test_swift_td_contract.py
- T007: Contract test

### tests/unit/test_swift_sarsa_contract.py
- T008: Contract test

### tests/unit/test_policies_contract.py
- T009: Contract test

### tests/fixtures/environments.py
- T010: Test fixtures

### tests/integration/
- T027: test_swift_td_convergence.py
- T028: test_swift_sarsa_control.py
- T029: test_neural_network_integration.py
- T030: test_batch_scaling.py
- T031: test_gpu_cpu_consistency.py
- T032: test_terminal_handling.py

### examples/
- T033: standalone_swift_td.py
- T034: standalone_swift_sarsa.py
- T035: neural_network_swift_td.py
- T036: neural_network_swift_sarsa.py

### Root files
- T002: requirements.txt
- T003: requirements-dev.txt
- T004: pyproject.toml
- T005: .gitignore
- T037: README.md
- T047: .pre-commit-config.yaml

## Notes
- [P] tasks = different files, no dependencies
- Verify contract tests fail before implementing (T007-T010 before T011-T015)
- Constitutional requirement: Every function ≤20 lines (verified in T044)
- Type annotations required everywhere (verified in T041)
- Test coverage target: >80% (verified in T043)
- Black + isort applied before commit (T039-T040)

## Validation Checklist
*GATE: Checked before considering tasks complete*

- [ ] All contract interfaces have corresponding implementations
- [ ] All entities from data-model.md have implementation tasks
- [ ] All test scenarios from quickstart.md have integration tests
- [ ] Parallel tasks are truly independent (different files)
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task
- [ ] TDD order maintained (tests before implementation)
- [ ] Constitutional requirements addressed (20-line limit, types, formatting)

## Estimated Completion
- **Setup**: 6 tasks (T001-T006)
- **Contract Tests**: 4 tasks (T007-T010)
- **Core Implementation**: 5 tasks (T011-T015)
- **Unit Tests**: 11 tasks (T016-T026)
- **Integration Tests**: 6 tasks (T027-T032)
- **Examples & Docs**: 6 tasks (T033-T038)
- **Polish & Validation**: 9 tasks (T039-T047)
- **Total**: 47 tasks
