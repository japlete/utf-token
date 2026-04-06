# Releasing `utf-token`

## Repository setup

Configure two GitHub environments:

- `testpypi`: trusted publishing configured against the TestPyPI project
- `pypi`: trusted publishing configured against the PyPI project, with manual approval enabled

Both environments should trust the repository's `release.yml` workflow.

## Local preflight

Run the same baseline checks locally before creating a release tag:

```bash
uv run ruff check
uv run ty check
uv run -m unittest discover -s tests
uv build --sdist --wheel
uvx --from twine twine check dist/*
```

The `dev` group is intentionally limited to build, lint, and test tooling so it stays compatible
with the full supported runtime range. Notebook and offline vocab-processing tools live in the
separate `offline` dependency group.

Optional local smoke test of the built wheel:

```bash
uv venv --python 3.10 .venv-smoke
uv pip install --python .venv-smoke/bin/python dist/*.whl
uv run --no-project --python .venv-smoke/bin/python --script scripts/smoke_installed_package.py
```

## CI coverage

`ci.yml` enforces four checks:

- lint and type checks on a recent interpreter
- source-tree unit tests on Python 3.10 and 3.14
- wheel smoke tests on `min-supported` and `latest-supported` dependency resolutions
- one sdist install smoke test

The wheel smoke tests compile runtime requirements from publishable package metadata with:

- `lowest-direct` resolution on Python 3.10
- default highest resolution on Python 3.14

This keeps the validation simple while still exercising both the oldest supported install lane
and the newest one.

## Release flow

1. Update `project.version` in `pyproject.toml`.
2. Commit the release changes.
3. Create and push a matching tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. The release workflow verifies that the tag matches `project.version`.
5. The workflow builds the artifacts once, validates them with `twine check`, and uploads them to TestPyPI.
6. A fresh environment installs the tagged version from TestPyPI and runs `scripts/smoke_installed_package.py`.
7. After the TestPyPI smoke test passes, approve the `pypi` environment and publish the exact same artifacts to PyPI.
