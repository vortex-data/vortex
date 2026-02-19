Nightly Rust toolchain is pinned via `NIGHTLY_TOOLCHAIN` env var in each workflow file — update all instances when changing the version.
For action inputs: `toolchain: ${{ env.NIGHTLY_TOOLCHAIN }}`
For shell commands: `cargo +$NIGHTLY_TOOLCHAIN ...`

All files under `.github/` are linted by `yamllint --strict -c .yamllint.yaml`.
