<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# IO changes integrated with the updated PR branch

The local IO work was backed up, then the checkout was fast-forwarded from `afeb2d1a4` to
PR #10339's `ji/planning-plan-planexec` head,
`e645cd9a97ccf55f6b9fc1ef2013b53ff8780389`, bringing in all 27 intervening commits. The IO
changes were reapplied and six overlapping files were reconciled before validation.

## Preserved behavior and conflict resolutions

- Upstream reads retain one registered consumer per announcement. Fetches consume those
  registrations, share pending reads, and release consumed bytes instead of retaining them
  for the whole split. Both inline and boxed fetch futures hold their own read reference until
  completion; a pruning withdrawal cannot cancel another consumer's fetch.
- Upstream announcements retain one registration per potential projection consumer and skip
  chunks excluded by the input selection. The projection-announcement control remains available.
- Upstream filtering retains its initial mask, dense predicate planning, and optional bounded
  projection prefetch. The new selection diagnostics coexist with that implementation.
- Both the upstream compute-timing events and detailed driver tracing remain available.
- A single projection morsel still publishes its own reads without a separate prefetch batch.
  Pruning withdrawal protects its selected ranges and aliases even when that batch is empty.
  Two regression cases cover distinct and aliased ranges.
- Seven existing Arc-clone lint errors in updated tests were corrected without changing behavior.

The local file-handle reuse, native-read admission and demanded-extent coalescing controls remain
opt-in. Previously rejected cost-based speculative coalescing and compute experiments are not
included in this IO patch. The updated branch's encoding kernels and split scheduling are retained.

## Local validation after integration

All commands below run against the integrated checkout. Build artifacts and temporary files use
the instance SSD at `/mnt/vortex-ssd/votex-4`; focused native test runs also use an SSD working
directory. The broad Rust library run enables V2 and all crate features.

| Check | Result |
|---|---|
| `vortex-file` library tests | 221 passed |
| `vortex-io` library tests | 197 passed |
| `vortex-layout` library tests | 582 passed |
| `vortex-scan` library tests | 57 passed |
| File scan IO tests with boxed fetches, ready fetches, and demanded/native32 settings | 15 passed per setting; 45 additional executions |
| Projection tests with withdrawal disabled and enabled | 51 passed per setting; 102 additional executions |
| Benchmark diagnostic extrema aggregation regression | 1 passed |
| Doctests for the four library crates | 1 passed; 3 existing examples ignored |
| Python benchmark/analyzer tests | 48 passed |
| All-target/all-feature Clippy for the four libraries and `datafusion-bench`, warnings denied | Passed |
| Pinned `nightly-2026-09-10` formatting for those five crates | Passed |
| Scoped Ruff lint, Ruff format and ty checks for the four tool/test files | Passed |
| Patch whitespace | Passed |

The broad Rust command is:

```bash
CARGO_TARGET_DIR=/mnt/vortex-ssd/votex-4/target \
TMPDIR=/mnt/vortex-ssd/votex-4/tmp VORTEX_SCAN_V2=1 \
cargo test --locked -j 8 \
  -p vortex-io -p vortex-file -p vortex-layout -p vortex-scan \
  --all-features --lib -- --test-threads=4
```

Clippy uses the same target and temporary directories:

```bash
cargo clippy --locked \
  -p vortex-io -p vortex-file -p vortex-layout -p vortex-scan -p datafusion-bench \
  --all-targets --all-features -- -D warnings
```

Logs, the original patch/file archive, and focused test commands are under
`/mnt/vortex-ssd/votex-4/results/integrate-pr10339-20261008T203936Z/`.
No workspace-wide tests, cross-platform checks or SQL benchmark rerun were performed during
this integration.

## Performance evidence remains tied to its measured source

[The IO-only report](REPORT_IO_ONLY.md) retains the cold SSD results and their sequential
attribution: handle reuse/native32 were measured together, while coalescing was isolated on
top of that bundle. The 22.95% suite reduction and four eight-round confirmations refer to
the frozen pre-integration executable, not this newer merged source or an untouched PR baseline.
No new latency reduction is claimed for this integration. Changes in upstream read lifetimes
and planning make a fresh benchmark necessary before transferring those figures to the new tip.
