# Repository Guidelines

## Project Structure & Module Organization
- `conf/` – Model and dataset configs (e.g., `conf/STGCN/METR_LA.conf`).
- `data/` – Datasets and auxiliary artifacts; unzip archives before use.
- `lib/` – Training utilities: parameter parsing, loaders, metrics, logging.
- `model/` – GPT‑ST and baselines; entrypoint `model/Run.py`; checkpoints in `model/SAVE/<DATASET>/`.

## Build, Test, and Development Commands
- Environment: `conda create -n gptst python=3.9.12 && conda activate gptst`
- Install deps: `pip install -r requirements.txt`
- Prepare data: `cd data && unzip *.zip`
- Run examples (from `model/`):
  - Eval GPT‑ST‑enhanced STGCN (PEMS08): `python Run.py -dataset PEMS08 -mode eval -model STGCN`
  - Baseline (NYC_TAXI): `python Run.py -dataset NYC_TAXI -mode ori -model CCRNN`
  - Pretrain GPT‑ST (NYC_BIKE): `python Run.py -dataset NYC_BIKE -mode pretrain`
- Smoke check: append `-epochs 1 -batch_size 8` to shorten runs.

## Coding Style & Naming Conventions
- Python 3.9; follow PEP 8 with 4‑space indentation.
- Names: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Configs: `conf/<MODEL>/<DATASET>.conf`, `conf/GPTST_pretrain/*.conf`.
- CLI: GPT‑ST flags use single hyphen (e.g., `-dataset`); predictor hyperparams use double hyphen (e.g., `--hidden_dim`).
- Keep style consistent; no enforced formatter/linter committed.

## Testing Guidelines
- No formal unit tests yet; favor quick runs over full training.
- Verify logs and checkpoints under `model/SAVE/<DATASET>/`.
- If adding tests, use `pytest` under `tests/` and mock small tensors for `lib/` utilities. Run with `pytest -q`.

## Commit & Pull Request Guidelines
- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- PRs include purpose, key changes, reproduction commands, linked issues, and metrics tables or logs for training‑related changes.

## Security & Configuration Tips
- Do not commit large raw data or new checkpoints; provide links/scripts instead.
- Keep secrets out of configs; prefer environment variables.

## Agent‑Specific Instructions
- Keep changes minimal and scoped; avoid broad refactors.
- Follow the existing directory layout and argparse patterns.
- Update `readme.md`/`README.md` if commands or flags change.

