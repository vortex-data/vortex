# Independent PR measurements

The comparison isolates this PR against `133aacdb7e` on Apple M4 Max, `aarch64-apple-darwin`.
The candidate source is `5da8598478`. Subsequent commits only add evidence.
Both variants use rustc 1.98.0, LLVM 22.1.8, and the [documented protocol](../README.md).
[Environment](environment.json), binary hashes, raw samples, and [quartiles](summary.csv) are retained.

Single-row inline decoding decreases from 91.5 ns to 25.2 ns, and external decoding decreases
from 108.3 ns to 35.2 ns. The change removes intermediate array construction while retaining the
same `VarBinViewData::validate_and_fix` call and sanitation contract. It does not reuse prior
validation evidence or remove the per-row string scan.

The first three pairs showed high variation at 1,024 inline rows, so three additional pairs ran
with the same binaries. Both sets remain in `run-*.csv` and `repeat-*.csv`. The table and summary
pool all 72 observations per case and variant. The pooled 1,024-row medians are 3,679.5 ns and
3,819.0 ns, with overlapping quartiles. The after distribution is wider. A stable large-batch
improvement is not established, and the cause of the variation remains unresolved.

| Case | Rows | Before (ns) | After (ns) |
| --- | ---: | ---: | ---: |
| `utf8_decode_inline` | 0 | 91.0 | 23.5 |
| `utf8_decode_inline` | 1 | 91.5 | 25.2 |
| `utf8_decode_external` | 1 | 108.3 | 35.2 |
| `utf8_decode_inline` | 64 | 311.1 | 248.9 |
| `utf8_decode_external` | 64 | 667.9 | 599.3 |
| `utf8_decode_inline` | 1,024 | 3,679.5 | 3,819.0 |
| `utf8_decode_external` | 1,024 | 9,181.7 | 9,101.7 |
| `utf8_decode_inline` | 16,384 | 57,280.6 | 58,174.1 |
| `utf8_decode_external` | 16,384 | 144,528.7 | 144,022.2 |

## Correctness and limits

The isolated branch passes 78 UTF-8 and VarBinView tests. Coverage includes empty and sliced
arrays, inline and external strings, arbitrary null payloads, invalid UTF-8, constants, and
repeated sanitation. The new parameterized test checks sanitized views across repeated decodes.

`cargo clippy -p vortex-array --lib --features unstable_row_fns -- -D warnings` passes.
Formatting uses `nightly-2026-09-10`. Test, lint, and build logs are retained here.
No workspace-wide tests, doctests, all-feature linting, x86 runtime measurements, or end-to-end
queries ran for this split. The measurements do not establish cold-buffer behavior.
