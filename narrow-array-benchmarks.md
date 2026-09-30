<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# NarrowArray benchmarks

Measured on 2026-09-23 against comparison PR #9948 at
`c215b0e423f558ef83ac0f99841160d70237a9b7`. All logical-i64 comparisons use the
same executable and kernels. The existing primitive comparison benchmarks and kernels
are unchanged by this draft.

The strongest measured compute benefit is i8 comparison. Smaller storage alone does not
make every operation faster: i32/i16 constant comparison and gathers that return i64
remain slower in this prototype. These are array microbenchmarks, not end-to-end queries.

## Buffer size

These are measured `ArrayRef::nbytes()` values for 1,048,576 non-null rows:

| Logical i64 representation | Buffer bytes | MiB | Reduction |
| --- | ---: | ---: | ---: |
| Primitive i64 | 8,388,608 | 8 | baseline |
| Narrow over i32 | 4,194,304 | 4 | 50% |
| Narrow over i16 | 2,097,152 | 2 | 75% |
| Narrow over i8 | 1,048,576 | 1 | 87.5% |

The same ratios were verified at 8,192 and 16,777,216 rows. NarrowArray owns no value
buffers; the child holds the narrower buffer. `nbytes()` does not count array objects,
statistics, allocator bookkeeping, or process RSS. Nullable arrays also retain validity
storage. No file-compression or serialized-size gain is claimed here.

## Comparisons at 1,048,576 rows

Pooled median time in microseconds; lower is better:

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| `x < 16` | 171.50 | 196.48 | 208.60 | 38.98 |
| `x == 16` | 179.35 | 206.44 | 220.19 | 41.67 |
| `x < y` | 296.02 | 243.25 | 256.94 | 49.17 |
| `x == y` | 305.44 | 244.58 | 257.90 | 49.69 |

For `<`, i8 storage is 4.40x faster against a constant and 6.02x faster against another
column. Across the three individual passes those ratios range from 4.33–4.41x and
6.03–7.02x, respectively. i32/i16 storage is about 15%/22% slower against a constant,
but improves column comparisons by about 1.22x/1.15x in the pooled measurements.

Plain native-width controls use the same child buffers as the wrappers:

| Storage | Native `< constant` (us) | Narrow `< constant` (us) | Native `< column` (us) | Narrow `< column` (us) |
| --- | ---: | ---: | ---: | ---: |
| i32 | 196.04 | 196.48 | 242.75 | 243.25 |
| i16 | 208.40 | 208.60 | 257.27 | 256.94 |
| i8 | 38.50 | 38.98 | 48.38 | 49.17 |

The wrapper's difference from a native array at the same width is small at this row count.
The 8-bit kernel optimization in the base PR is responsible for the large comparison
advantage; NarrowArray makes that kernel available while preserving logical i64.

## Selection and pipelines at 1,048,576 rows

Times in microseconds. Filter selects approximately 50% of rows. Take gathers N/8
indices in a deterministic pseudorandom order. Inputs, masks, and indices are prebuilt.
Every operation is executed to its final primitive or boolean result before stopping the timer.

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| Filter → i64 buffer | 304.85 | 227.27 | 156.02 | 342.69 |
| Take → i64 buffer | 215.77 | 331.96 | 308.33 | 321.40 |
| Compare → filter → i64 buffer | 458.10 | 402.62 | 342.27 | 382.17 |
| Filter → compare → bool bitmap | 379.71 | 211.58 | 156.58 | 264.73 |
| Take → compare → bool bitmap | 167.62 | 165.17 | 153.58 | 120.75 |

- Compare/filter with i8 storage is about 1.20x faster even with a final i64 result.
- i16 is the strongest filtering representation in this fixture: filter → i64 is 1.95x
  faster and filter → compare is 2.42x faster. i8 filter → i64 is about 12% slower.
- Take → compare benefits from i8 storage (1.39x), but take → i64 is about 49% slower.
  The current canonical fallback casts the narrow child to i64; the existing dictionary
  cast rule pushes that cast into all dictionary values before gathering. Avoiding that
  early widening is a follow-up optimization, not a benefit implemented by this draft.

## Encoding and canonical materialization

Times in microseconds for 1,048,576 rows:

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| Encode, fresh bounds scan | 212.96 | 384.04 | 355.71 | 334.15 |
| Materialize canonical i64 | 0.08 | 199.58 | 177.88 | 188.90 |

Encoding inputs require the indicated storage width: the common 0..31 pattern is shifted
by 0 for i8, 128 for i16, 32,768 for i32, and 2,147,483,648 for i64. The i64 column is
therefore the cost of discovering that narrowing is impossible, not a conversion baseline.
Each sample uses fresh array statistics so the bounds scan cannot disappear into a cache.
Input buffer generation and creation of the fresh array handle are outside the timer.

The i8 encoding cost is about 334 us. The pooled constant-comparison saving is about
133 us per pass: roughly three comparisons amortize conversion, or three to four using
individual-pass medians. Encoding solely for one comparison loses overall. Constructing
NarrowArray around an existing narrow child avoids conversion and the bounds scan.

Canonical materialization is an additional cost: about 189 us to widen i8 to i64 here.
The primitive-i64 materialization case is a no-op handle operation, not a buffer copy.
Arithmetic, aggregates, and mixed-storage-width comparisons have no specialized Narrow
kernel in this draft and may pay for widening.

## Row-count sweep

All times below are microseconds. These tables use `<`; equality and all native-width
controls are in the machine-readable results.

### 8,192 rows

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| `x < 16` | 2.21 | 2.54 | 2.62 | 1.29 |
| `x < y` | 2.71 | 2.42 | 2.42 | 1.08 |

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| FilterToI64 | 4.12 | 3.50 | 2.88 | 4.54 |
| TakeToI64 | 1.21 | 4.04 | 3.75 | 4.17 |
| CompareFilterToI64 | 7.46 | 5.83 | 5.29 | 5.88 |
| FilterCompare | 6.62 | 3.88 | 3.42 | 4.33 |
| TakeCompare | 2.25 | 3.29 | 3.21 | 3.12 |
| Encode | 2.88 | 4.40 | 3.96 | 3.79 |
| MaterializeI64 | 0.08 | 1.62 | 1.54 | 2.08 |

### 16,777,216 rows

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| `x < 16` | 3568.77 | 3942.46 | 4217.94 | 738.85 |
| `x < y` | 3905.52 | 2655.87 | 2815.94 | 544.52 |

| Operation | Primitive i64 | Narrow/i32 | Narrow/i16 | Narrow/i8 |
| --- | ---: | ---: | ---: | ---: |
| FilterToI64 | 4898.38 | 3663.04 | 2505.21 | 5478.21 |
| TakeToI64 | 9284.69 | 12349.31 | 11896.35 | 12101.79 |
| CompareFilterToI64 | 7328.42 | 6389.04 | 5343.48 | 6027.17 |
| FilterCompare | 6034.90 | 3367.04 | 2483.33 | 4225.44 |
| TakeCompare | 9435.10 | 8211.56 | 6757.50 | 3897.21 |
| Encode | 3439.96 | 6205.85 | 5742.08 | 5420.33 |
| MaterializeI64 | 0.25 | 3204.27 | 2862.83 | 3036.96 |

Small-array overhead matters: at 8K rows the i8 comparison gains are 1.71x/2.50x,
and take → compare is slower for every narrow width. Large-buffer selection and allocation
timings vary more between passes; consult the individual pass medians and quartiles rather
than treating the pooled values as precise latency guarantees.

## Method

- Apple M5 Max, aarch64, macOS 26.6.1 (25G76), Rust 1.98.0.
- Standard repository bench profile: optimized, 16 codegen units, LTO off. Repository
  `force-frame-pointers=yes`; no additional `RUSTFLAGS` or target-CPU override.
- Three sequential passes, 140 samples per case per pass. Representations rotate both their
  starting position and traversal direction on every round. No concurrent builds or other
  benchmarks were launched during measurement.
- The tables are pooled medians over 420 samples. Absolute times varied across passes;
  e.g. the 1M-row i64 constant comparison medians were 171, 223, and 141 us, while the
  corresponding narrow-i8 medians were 39, 52, and 32 us. Relative comparison benefits
  persisted. This was not a pinned-core or controlled-frequency experiment.
- All compute inputs are ordinary uncompressed PrimitiveArray buffers with non-null values
  in 0..31. Native-width controls expose that width as their logical type; every Narrow
  representation keeps logical i64. Pairs have matching storage widths on both sides.
- Timings include expression construction, execution, and result allocation. Input creation,
  conversion to the chosen input representation, execution-context creation, and destruction
  of the returned result are outside timing. Inputs are reused; no forced cold-cache flush.
- Comparison/operation results are checked against the i64 reference before timing. Encoding
  checks both value equality and output buffer size. Mask and index generation are excluded;
  compare → filter includes construction of the predicate mask.
- The public execution path may reorder or push operations through children. These pipeline
  measurements do not require a particular physical execution order.

Sources: [comparison matrix](vortex-array/benches/narrow_compare.rs),
[selection/encoding matrix](vortex-array/benches/narrow.rs).
[Machine-readable results](narrow-array-benchmark-results.json) contain every case, quartiles,
and individual-pass medians. Raw per-sample CSVs are in
`/private/tmp/narrow-{compare,operations}-run{1,2,3}.csv` on the measurement host.

Build command:

```sh
CARGO_TARGET_DIR=/Users/matt/Desktop/repo/vortex/target \
  cargo bench -p vortex-array --bench narrow --bench narrow_compare --no-run
```

The build prints the executable paths. The measured binaries were
`target/release/deps/narrow_compare-081029d9296a0cf7` and
`target/release/deps/narrow-729aa8d2feff4afe` under that target directory. Each was run with
`VORTEX_NARROW_INTERLEAVED=1` three times, sequentially, redirecting stdout to its CSV;
`narrow` also emits measured buffer bytes on stderr. Without that environment variable,
the same targets expose ordinary Divan benchmarks via `cargo bench`.

## Validation

- `cargo nextest run -p vortex-array -p vortex-fastlanes`: 4,121 passed, one existing skip.
- `cargo test --doc -p vortex-array`: 76 passed, 21 ignored.
- `cargo clippy -p vortex-array -p vortex-fastlanes --all-targets --all-features -- -D warnings`: passed.
- `cargo clippy --all-targets --all-features`: passed across the workspace.
- `cargo +nightly-2026-09-10 fmt --all --check`: passed.
- Three passes of both benchmark matrices: all result/size assertions passed.

The workspace build warned that clang-format was unavailable for a generated DuckDB header;
no C++ or CUDA sources were changed.

x86 performance and end-to-end query workloads have not been measured for this draft.
