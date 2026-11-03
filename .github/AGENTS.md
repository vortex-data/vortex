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

## Self-hosted runners and fork pull requests

Code from a fork pull request must never run on our self-hosted (runs-on) runners. These runners
run in our AWS account and share the S3 build cache with trusted builds.
`scripts/check_fork_pr_runners.py` enforces the rules below in CI; run it after changing a
workflow:

```bash
uv run --no-project --with pyyaml python scripts/check_fork_pr_runners.py
```

- A job may pick a self-hosted runner only when the run is trusted. Use this condition in
  `runs-on` and in job-level `if:`:

  ```yaml
  github.repository == 'vortex-data/vortex' &&
  (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
  ```

  Compare `head.repo.full_name`, not `head.repo.fork`. When the fork is deleted, `head.repo` is
  null, and `null == false` is true in workflow expressions, so a `fork == false` check passes.
- If the job has a GitHub-hosted fallback (`ubuntu-latest`, `windows-latest`), untrusted runs use
  it. If the job needs hardware that GitHub does not provide (GPU, arm64 test hosts, specific CPU
  families), add the condition to the job's `if:` so untrusted runs skip it.
- In steps and composite actions, check where the job runs with
  `runner.environment == 'self-hosted'`, not `github.repository`. For example, gate
  `runs-on/action` and `sccache --show-stats` on it.
- Benchmark workflows started by `action/bench-*` labels are the exception. Only maintainers can
  add these labels, so adding one is a decision to trust the pull request's code. The check exempts
  workflows whose `pull_request` trigger only lists the `labeled` type.
- Never use the `pull_request_target` event. It runs with secrets and a write token in the context
  of the fork pull request. The check rejects it.
