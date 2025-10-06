<!--
Sync Impact Report:
Version: 1.0.0 → 1.1.0
Modified principles: N/A
Added sections:
  - VI. Commit Discipline (new principle added to Core Principles)
  - Updated Development Workflow > Commit Requirements (new subsection)
Removed sections: N/A
Templates status:
  ✅ plan-template.md - Aligned (no changes needed)
  ✅ spec-template.md - Aligned (no changes needed)
  ✅ tasks-template.md - Should reflect phase-based commits in implementation tasks
  ⚠️  CLAUDE.md - Should mention commit discipline in development guidance
Follow-up TODOs:
  - Update tasks template to include commit checkpoints after each phase
  - Add commit discipline to agent guidance files
-->

# Swift-RL Constitution

## Core Principles

### I. Modular Design
Every function MUST be concise and focused on a single responsibility. Functions MUST
NOT exceed 20 lines of code. Complex operations MUST be decomposed into smaller,
composable functions. This ensures testability, readability, and maintainability.

**Rationale**: Short functions are easier to understand, test, and debug. The 20-line
limit forces developers to think about proper abstraction and separation of concerns,
leading to cleaner APIs and better code reuse.

### II. GPU-First Architecture
All tensor operations MUST be designed with GPU optimization in mind. Algorithms MUST
minimize CPU-GPU data transfers, maximize vectorization, and leverage PyTorch's
autograd efficiently. In-place operations SHOULD be used where appropriate to reduce
memory overhead.

**Rationale**: Swift-TD and Swift-SARSA are reinforcement learning algorithms that
benefit significantly from GPU acceleration. Proper GPU optimization is critical for
performance at scale. Poor GPU utilization can result in orders of magnitude slower
execution.

### III. Type Safety
All function signatures MUST include complete type annotations for parameters and
return values. Use Python 3.10+ type hints including `typing` module constructs
(`Optional`, `Union`, `Literal`, etc.) and PyTorch tensor type hints where applicable.

**Rationale**: Type annotations serve as inline documentation, enable static analysis
tools to catch bugs early, improve IDE support, and make the codebase more maintainable
as it grows.

### IV. Code Formatting (NON-NEGOTIABLE)
All code MUST be formatted using `black` (default settings) and imports MUST be sorted
using `isort` (profile: black). These tools MUST be run before every commit. No
exceptions.

**Rationale**: Consistent formatting eliminates bikeshedding, reduces diff noise in
code reviews, and ensures the codebase maintains a uniform style. The black + isort
combination is the de facto standard in the Python ecosystem.

### V. Algorithmic Clarity
Implementation MUST be straightforward and clearly reflect the underlying algorithm.
Avoid premature optimization that obscures the algorithm's logic. Code MUST be readable
by someone familiar with the algorithm's mathematical formulation.

**Rationale**: Swift-TD and Swift-SARSA are well-defined algorithms. The implementation
should serve as executable documentation. Clever optimizations that sacrifice clarity
MUST be justified and documented.

### VI. Commit Discipline
Commits MUST be created after completing each implementation phase. Each phase
(setup, tests, core implementation, integration, polish) MUST result in a separate
commit with a descriptive message. This enables better traceability, easier rollback,
and clearer project history.

**Rationale**: Phase-based commits create logical checkpoints in the development
process. When issues arise, developers can easily identify which phase introduced a
problem. Atomic commits per phase also make code review more manageable and facilitate
selective cherry-picking or reverting of changes.

## Code Quality Standards

### Testing
- Unit tests MUST cover all public functions
- Integration tests MUST verify algorithm correctness against known baselines
- Tests MUST include GPU execution paths when applicable
- Test coverage SHOULD be tracked and maintained above 80%

### Documentation
- All modules MUST have docstrings describing purpose and usage
- All public functions MUST have docstrings with Args, Returns, and Examples sections
- Mathematical formulations SHOULD be included in docstrings for algorithm
  implementations
- Complex tensor operations SHOULD include shape comments (e.g., `# (batch, features)`)

### Performance
- Tensor operations MUST be batched where possible
- Avoid Python loops over tensor dimensions; use vectorized operations
- Profile critical paths and document performance characteristics
- Memory allocation SHOULD be minimized in hot loops

## Development Workflow

### Pre-commit Requirements
1. Run `black .` to format all Python files
2. Run `isort .` to sort imports
3. Run type checker (e.g., `mypy`) to verify type annotations
4. Run test suite and ensure all tests pass
5. Verify no function exceeds 20 lines (excluding docstrings)

### Commit Requirements
Commits MUST be structured by implementation phase:
- **Setup phase**: Project structure, dependencies, configuration files
- **Test phase**: Contract tests, unit tests, integration tests
- **Implementation phase**: Core modules and functionality
- **Integration phase**: Module connections, end-to-end workflows
- **Polish phase**: Documentation, examples, formatting, validation

Each commit message MUST follow the format:
```
<type>(<scope>): <subject>

<optional body>
```

Where `<type>` is one of: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

Examples:
- `feat(setup): initialize project structure and dependencies`
- `test(swift-td): add contract tests for SwiftTD module`
- `feat(swift-td): implement SwiftTD temporal difference learning`
- `docs(examples): add standalone SwiftTD usage example`
- `chore(format): apply black and isort to all modules`

### Code Review Gates
- All function signatures properly typed
- All functions under 20 lines (excluding docstrings and blank lines)
- Black and isort applied
- Tests included for new functionality
- GPU optimization opportunities identified and addressed
- Algorithm correctness verified
- Commits organized by phase with clear messages

### Dependency Management
- Minimize external dependencies
- Pin PyTorch version explicitly
- Use `requirements.txt` for production dependencies
- Use `requirements-dev.txt` for development tools (black, isort, pytest, mypy)

## Governance

This constitution is the authoritative governance document for the Swift-RL project.
All code, documentation, and development practices MUST comply with these principles.

### Amendment Process
1. Proposed amendments MUST be documented with rationale
2. Amendments require consensus among maintainers
3. Version number MUST be incremented per semantic versioning
4. Migration plan MUST be provided for breaking changes

### Compliance
- All pull requests MUST be reviewed for constitutional compliance
- Reviewers MUST verify adherence to the 20-line function limit
- Reviewers MUST confirm black and isort have been applied
- Reviewers MUST check type annotations are complete
- Reviewers MUST verify commits are organized by phase
- Automated CI checks SHOULD enforce formatting and type checking

### Constitutional Violations
Code that violates constitutional principles MUST be rejected or refactored. If a
violation is necessary for a justified reason (e.g., performance critical path,
external API compatibility), it MUST be:
1. Documented with a clear explanation
2. Approved by at least two maintainers
3. Tracked as technical debt with a plan for resolution

**Version**: 1.1.0 | **Ratified**: 2025-10-05 | **Last Amended**: 2025-10-05
