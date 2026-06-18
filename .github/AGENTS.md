Prefer stable Rust for CI workflows. Only use nightly when a feature genuinely requires it.

Nightly is required for:
- `cargo fmt` (nightly formatting options)
- `-Z` flags: sanitizers (`-Zsanitizer=address`), miri (`-Zmiri-*`), publish (`-Zpublish-timeout`)
- `cargo-fuzz` (requires nightly)

Everything else (build, clippy, tests, docs, benchmarks, packaging) should use stable.

Nightly Rust toolchain is pinned via `NIGHTLY_TOOLCHAIN` env var in each workflow file — update all instances when changing the version.
For action inputs: `toolchain: ${{ env.NIGHTLY_TOOLCHAIN }}`
For shell commands: `cargo +$NIGHTLY_TOOLCHAIN ...`

All files under `.github/` are linted by `yamllint --strict -c .yamllint.yaml`.
