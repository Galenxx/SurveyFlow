# Contributing to SurveyFlow

Thank you for your interest in contributing to SurveyFlow!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/SurveyFlow.git
cd SurveyFlow

# Install dependencies
pip install uv
uv sync --all-extras
cp .env.example .env
# Edit .env with your API keys

# Run linting and type checking
uv run ruff check .
uv run mypy .

# Run tests
uv run pytest -q
```

## Project Structure

Refer to the [Project Structure](README.md#project-structure) section in the README for a map of the codebase.

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Line length: 120 characters (enforced by `ruff`)
- Run `uv run ruff check --fix .` before committing

## Adding a New Node

To add a new node to the pipeline:

1. Create `nodes/your_node.py` with a function `your_node(state: SurveyState) -> SurveyState`
2. Add the node name and function to `graph/nodes.py`
3. Register it in `graph/builder.py` with `builder.add_node(...)` and the appropriate edge
4. Add tests in `tests/`

## Adding a New Benchmark System

1. Create a runner in `evaluation_citation/generators/your_system_runner.py`
2. Register it in `evaluation_citation/configs.py` under `SYSTEMS`
3. Implement the `run_detailed` interface (see existing runners for the signature)

## Pull Request Process

1. Fork the repository and create a branch from `main`
2. Run linting and tests locally
3. Open a pull request with a clear description of the changes
4. Ensure CI passes
5. Request review from a maintainer

## Reporting Issues

Please open an [issue](https://github.com/<your-username>/SurveyFlow/issues) with:
- A clear description of the problem
- Steps to reproduce
- Your environment (Python version, OS, etc.)
