# swfit-rl Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-05

## Active Technologies
- Python 3.10+ + PyTorch (latest stable), typing, numpy (for testing) (001-implement-https-rlj)

## Project Structure
```
src/
tests/
```

## Commands
cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style
Python 3.10+: Follow standard conventions

## Recent Changes
- 001-implement-https-rlj: Added Python 3.10+ + PyTorch (latest stable), typing, numpy (for testing)

<!-- MANUAL ADDITIONS START -->

## Constitutional Requirements

### Commit Discipline (Principle VI)
When implementing features, create commits after completing each phase:
- **Setup**: After initializing project structure and dependencies
- **Tests**: After writing contract/unit/integration tests
- **Implementation**: After completing core module functionality
- **Integration**: After connecting modules and workflows
- **Polish**: After documentation, formatting, and validation

**Commit message format**:
```
<type>(<scope>): <subject>
```
Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

**Examples**:
- `feat(setup): initialize project structure and dependencies`
- `test(swift-td): add contract tests for SwiftTD module`
- `feat(swift-td): implement SwiftTD temporal difference learning`

### Code Quality Requirements
- Functions MUST NOT exceed 20 lines (excluding docstrings)
- All functions MUST have complete type annotations
- Run `black` and `isort` before every commit
- Test coverage SHOULD be above 80%

<!-- MANUAL ADDITIONS END -->