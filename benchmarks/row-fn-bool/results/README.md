# Independent PR measurements

The comparison isolates this PR against `133aacdb7e` on Apple M4 Max, `aarch64-apple-darwin`.
The candidate source is `b17c6da787`. Subsequent commits only add evidence.
Both variants use rustc 1.98.0, LLVM 22.1.8, and the [documented protocol](../README.md).
[Environment](environment.json), binary hashes, raw samples, and [quartiles](summary.csv) are retained.

Partial-validity execution with a constant operand improves substantially in both positions.
Column-only execution regresses: accepted partial execution increases from 3,783.3 ns to 3,864.0 ns
with plain collection and from 3,777.4 ns to 3,975.9 ns with multiversioned collection.
Null-only rejection with two columns also regresses. Resolve or explicitly accept these costs
before merging the draft. No production workload improvement is established.

The table pools 36 observations per case and variant. `c0` means two columns, `c1` means constant
lhs, and `c2` means constant rhs. `dense` and `failure` are all-valid controls. `partial` accepts
dense evidence, while `retry` rejects only null payloads and retries the valid rows.

| Case | Rows | Before (ns) | After (ns) |
| --- | ---: | ---: | ---: |
| `bool_dense_c0_plain` | 16,384 | 3,634.9 | 3,632.1 |
| `bool_partial_c0_plain` | 16,384 | 3,783.3 | 3,864.0 |
| `bool_partial_c1_plain` | 16,384 | 10,356.4 | 2,847.0 |
| `bool_partial_c2_plain` | 16,384 | 10,927.5 | 2,857.2 |
| `bool_retry_c0_plain` | 16,384 | 14,861.3 | 15,263.3 |
| `bool_retry_c1_plain` | 16,384 | 22,533.5 | 14,724.1 |
| `bool_retry_c2_plain` | 16,384 | 22,147.3 | 14,256.0 |
| `bool_failure_c0_plain` | 16,384 | 3,553.0 | 3,589.0 |
| `bool_dense_c0_multi` | 16,384 | 3,771.6 | 3,761.2 |
| `bool_partial_c0_multi` | 16,384 | 3,777.4 | 3,975.9 |
| `bool_partial_c1_multi` | 16,384 | 10,352.4 | 2,836.0 |
| `bool_partial_c2_multi` | 16,384 | 10,968.0 | 2,995.6 |
| `bool_retry_c0_multi` | 16,384 | 14,780.5 | 15,591.3 |
| `bool_retry_c1_multi` | 16,384 | 22,213.2 | 14,706.6 |
| `bool_retry_c2_multi` | 16,384 | 22,287.8 | 14,350.8 |
| `bool_failure_c0_multi` | 16,384 | 3,707.2 | 3,688.0 |

## Compiler evidence

The [optimized LLVM IR and assembly](compiler/) come from the measured native binaries.
The baseline dense retry allocates one byte per row and stores Boolean bytes before packing.
Its two-column loop already vectorizes, while the constant fallback retains scalar comparisons
and byte stores. The candidate uses vector comparisons for both constant orientations and stores
packed words directly. These excerpts cover the plain collector specialization.

This confirms the changed collection path. It does not identify the cause of the column-only
regressions or establish anything about x86. The source does not change `collect_bool`, `pack.rs`,
or selected execution. The LLVM excerpts omit module-level metadata and are for inspection.
The [provenance file](compiler/provenance.json) records full artifact hashes and owning symbols.

## Correctness and limits

The isolated branch passes 175 RowFn tests. The new matrix covers empty and sliced arrays,
word boundaries, constants in both positions, both collector flags, all-valid, all-null, and
partial validity. Existing tests cover all-constant execution and observable failures.
A direct executor test confirms that decoding errors remain terminal. Allocation failure was
not injected. Rich row errors retain the existing construction and retry semantics.

`cargo clippy -p vortex-array --lib --features unstable_row_fns -- -D warnings` passes.
Formatting uses `nightly-2026-09-10`. Test, lint, and build logs are retained here.
No workspace-wide tests, doctests, all-feature linting, x86 runtime measurements, or end-to-end
queries ran for this split. The measurements do not establish cold-buffer behavior.
