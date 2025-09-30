# Repository Guidelines

## Project Structure & Module Organization
- `conf/`: Model and dataset configs organized by predictor (`conf/STGCN/METR_LA.conf`).
- `data/`: Archived datasets and helper assets; unzip locally before use and keep raw files out of version control.
- `lib/`: Training utilities covering argument parsing, loaders, metrics, and logging helpers.
- `model/`: GPT-ST and baselines with the entrypoint `model/Run.py`; checkpoints land under `model/SAVE/<DATASET>/`.

## Build, Test, and Development Commands
- Environment: `conda create -n gptst python=3.9.12 && conda activate gptst` prepares the interpreter.
- Dependencies: `pip install -r requirements.txt` syncs Python packages.
- Data prep: `cd data && unzip *.zip` expands bundled datasets for local runs.
- Execution: from `model/`, run `python Run.py -dataset PEMS08 -mode eval -model STGCN` for GPT-ST-enhanced evaluation. Swap dataset/mode/model to match experiments; append `-epochs 1 -batch_size 8` for smoke checks.

## Coding Style & Naming Conventions
- Use Python 3.9 with PEP 8 guidelines and 4-space indentation.
- Files and modules use `snake_case.py`; classes adopt `PascalCase`; functions and variables stay in `snake_case`.
- CLI flags follow the established pattern: GPT-ST options with single hyphen (e.g., `-dataset`), predictor-specific hyperparameters with double hyphen (e.g., `--hidden_dim`).

## Testing Guidelines
- Prefer lightweight verification: run targeted `Run.py` commands with small `-epochs` and `-batch_size` values.
- When adding automated tests, place them under `tests/` and rely on `pytest -q`, mocking small tensors for `lib/` utilities.
- Confirm training artifacts in `model/SAVE/<DATASET>/` and review logs for anomalies.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`) to mirror existing history.
- PRs should outline purpose, key code changes, reproduction commands, and—when relevant—metrics tables or log excerpts.
- Reference linked issues and note dataset variants impacted.

## Security & Configuration Tips
- Keep credentials out of configs; prefer environment variables or local `.env` files.
- Do not commit large raw datasets or checkpoints; provide scripts or download links instead.

## Agent-Specific Instructions
- Keep changes minimal, scoped, and consistent with directory layout and argparse patterns.
- Avoid broad refactors; focus on resolving the immediate task cleanly.
- Update documentation when CLI flags, commands, or workflows shift.
