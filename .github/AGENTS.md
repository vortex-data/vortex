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

RunsOn CI jobs pick their AWS region from the run's trigger time, following the night in 8-hour
blocks: 20-04 UTC go to `eu-west-2`, 04-12 UTC to `us-east-1`, and 12-20 UTC to `ap-northeast-1`.
The hour comes from `github.event.pull_request.updated_at` or `github.event.head_commit.timestamp`,
matched with `contains(ts, 'T03')` for an exact hour or `contains(ts, 'T0')` for hours 00-09; checks
run in order and the first match wins. Runs with neither timestamp go to `us-east-1`. The
`CI_REGION` repository variable pins every CI job to one region. Copy the `region=` expression from
an existing CI job when adding one. Benchmark workflows always use `region=us-east-1`.
