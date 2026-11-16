# Contributing to SafeGuard

1. Create a feature branch from `main`.
2. Keep dataset/model binaries out of Git.
3. Run `pytest backend/tests tests -q` before pushing.
4. Run `ruff check src backend tests scripts` when Ruff is installed.
5. For frontend changes, run `npm run build` in `frontend/`.
6. Keep the canonical ontology unchanged unless the team deliberately changes the project scope and updates dataset mapping, tests and documentation together.
