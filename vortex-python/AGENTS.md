# Python Binding Guidance

Applies to the Python bindings and their PyO3 implementation under `vortex-python/`.

Run the commands below from the repository root.

## Build and Development

`vortex-data` is a mixed Python/Rust package. `uv` manages the workspace environment, while
Maturin is the build backend that compiles `vortex-python/src/` into the `vortex._lib` extension.

Set up the workspace initially, and resync after dependency changes:

```bash
uv sync --all-packages
```

Python-only changes do not require an explicit extension rebuild. Run the affected tests through
`uv` as described below.

After changing PyO3 Rust code, `vortex-python/Cargo.toml`, or relevant Cargo features, rebuild the
extension before testing. For a single build-and-test cycle, force `uv` to reinstall `vortex-data`;
this invokes its Maturin backend:

```bash
uv run --all-packages --reinstall-package vortex-data pytest <changed-python-tests>
```

When iterating repeatedly on Rust code, rebuild the editable extension explicitly, then run as many
targeted test commands as needed:

```bash
(cd vortex-python && uv run maturin develop)
uv run --all-packages pytest <changed-python-tests>
```

Rerun `maturin develop` after each Rust change. For a feature-specific build, pass the feature to
Maturin, for example:

```bash
(cd vortex-python && uv run maturin develop --features <feature>)
```

Do not run both rebuild paths for the same test cycle. Before handing off broad Python binding
changes, run the canonical full check:

```bash
./vortex-python/check.sh
```

The script synchronizes the workspace, runs `maturin develop`, checks formatting, linting, and
types, builds and doctests the documentation, and runs the Python test suite.

## Testing

Run the narrow checks that match the files changed before broader Python suites:

```bash
python -m py_compile <changed-python-files>
uv run --all-packages pytest <changed-python-tests>
```

If Python docstrings, `docs/api/python/`, or Sphinx configuration change, also follow
`docs/AGENTS.md` and run the relevant clean Sphinx checks.

## Linting and Formatting

```bash
uv run basedpyright vortex-python
uv run ruff format --check <changed-python-files>
uv run ruff check <changed-python-files>
```

If PyO3 Rust files under `vortex-python/src/` change, include:

```bash
cargo +nightly fmt --check -p vortex-python
```

Always finish Python binding work with `git diff --check`.
