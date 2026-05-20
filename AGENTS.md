# KoopmanRL Agent Guide

Use this note as the entry point before touching the repository. It points to the canonical module guides and highlights the conventions that apply everywhere.

## Read First
- `configurations/AGENTS.md` -
- `scripts/AGENTS.md` -
- `src/koopmanrl/AGENTS.md` -
- `tests/AGENTS.md` -

## Global Conventions
- Reinforcement learning core logic has to stay script-addressable through `python -m ..`.
- When operations are to be parallelized, strict priority should be given to [Ray](https://github.com/ray-project/ray) for parallelization.
- Outside of hyperparameter optimization, prefer a CleanRL single file-style syntax

## Repository Layout
```
configurations/          best hyperparameter configurations for the algorithms as JSON files
scripts/                 utility scripts which reproduce the results from the paper
src/koopmanrl/           core library including the two subfolders environments/, and koopman_tensor/
tests/                   regression tests and unit tests for the library's core functionality
```

## Working Checklist
1. Review the relevant AGENTS guide(s) and existing tests/examples for the feature you touch.
2. Prototype changes in modules or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (`tests/test_*.py`) alongside code changes.
4. Run the scoped pytest command (`uv run test -m ...`) before submitting.
5. Keep documentation edits minimal and aligned with the per-module format.

## Cursor Cloud specific instructions

### Environment
- Python 3.10 is required (`requires-python = "==3.10.*"`). The update script installs it via `uv python install 3.10` and then runs `uv sync`.
- All commands use `uv run` (e.g. `uv run pytest`, `uv run -m koopmanrl.<module>`). There is no separate venv activation step.
- This is a pure Python scientific computing package with zero external service dependencies (no databases, Docker, web servers, etc.).

### Lint
- `uv run pre-commit run --all-files` runs ruff, ruff-format, vulture, isort, and general file checks.

### Tests
- `uv run pytest tests/test_rl.py -v` runs all 20 RL algorithm tests (5 algorithms × 4 environments). Takes ~8 minutes.
- `uv run pytest tests/test_hparam_opt.py -v` runs Ray/Optuna hyperparameter optimization tests. These are slow (~10 min each) and use 16 CPU cores per trial.
- To run a single test: `uv run pytest tests/test_rl.py -k "test_lqr[LinearSystem-v0]" -v`

### Running algorithms
- Each algorithm is a standalone module: `uv run -m koopmanrl.<module> --env_id <EnvName> --total_timesteps <N>`.
- Use `--help` on any module for its full argument list.
- SKVI/SAKC write checkpoints to `./saved_models/` and TensorBoard logs to `./runs/`.

### Gotchas
- The `gym==0.23.1` dependency emits deprecation warnings about NumPy 2.0 and `env.seed()`. These are harmless and expected.
- The `koopmanrl_utils/` directory is a separate utility package for post-processing/visualization; it is not tested by the main test suite.
