# Code Review Guidelines

In addition to checking for correctness, safety, and regressions, review changes against:

- [STYLE.md](STYLE.md) — Rust code style, documentation, type signatures, error handling, dispatch macros, and hot-loop accessors.
- [AGENTS.md — Tests](AGENTS.md#tests) — test structure, property tests, `.slt` preference, and test/monomorphization bounds.
- [AGENTS.md — Rust Code Style](AGENTS.md#rust-code-style) — keeping diffs focused and extending existing implementations where possible.
- [docs/specs/versioning/wire-contracts.md](docs/specs/versioning/wire-contracts.md) — when adding or changing serialized formats, layouts, or wire IDs.
- [.github/AGENTS.md](.github/AGENTS.md) — when editing GitHub Actions workflows or `.github/` configuration.
