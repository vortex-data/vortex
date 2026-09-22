# Reproduce the native probes

The standalone package uses public RowFn calls and an explicit wall-time timer.
It does not use CodSpeed registration as runtime evidence.
Run the commands from a Vortex checkout to inherit `.cargo/config.toml`.

## Build

```bash
CARGO_TARGET_DIR=/tmp/row-fn-probe \
  cargo rustc --manifest-path research/row-fn-engine/performance/probe/Cargo.toml \
  --release -- --emit=llvm-ir,asm,link
```

The manifest selects release optimization, 16 codegen units, and `lto = false`.
The measured target was `aarch64-apple-darwin`, with no `target-cpu` override.
Repository configuration adds `-C force-frame-pointers=yes`.
The compiler was rustc 1.98.0 with LLVM 22.1.8. The allocator was mimalloc 0.1.52.
The package lockfile retains the dependency versions.
[Environment and binary hashes](../results/environment.json) record the completed runs.

The benchmark creates its inputs, argument wrappers, session, and context before timing.
Each measured call includes output destruction.
Each case warms 200 calls, then doubles its repetition count until a block reaches two milliseconds.
It records 12 blocks per process. Three alternating process pairs produce 36 observations per case and variant.
Each binary calibrates independently. Raw rows include its exact repetition counts.
There is no CPU affinity or frequency lock.

## Run a family

```bash
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe utf8
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe bool
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe selected
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe cosine
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe invocation
RUST_BACKTRACE=0 /tmp/row-fn-probe/release/row-fn-performance-probe discard
RUST_BACKTRACE=1 /tmp/row-fn-probe/release/row-fn-performance-probe discard
```

An empty family argument runs every family. `error` measures a shallow error construction and destruction.
`discard` instruments construction inside nullable retry and also reports the instrumented outer call.
It uses 200 warm calls, then 12 blocks of 256 calls.
Each block requires successful outer results and exactly 256 rejected diagnostics.
The diagnostic interval excludes destruction and includes the internal timer cost.
Atomic counter updates are outside that interval.
Backtrace settings require separate processes because the runtime caches the setting.

`bool` uses deferred i64 predicates with plain and multiversioned collection.
It covers column/column, constant/column, and column/constant arrangements.
Its fixtures cover accepted dense evidence, accepted partial evidence, null-only rejection, and observable failure.
`selected` uses conservative input wrappers to force direct or filtered selected execution.
The wrappers are benchmark fixtures, not production element representations.
The mask invalidates one row in eight.
At zero or one row, universal empty/all-null handling can bypass the large-batch path.

## Compare two binaries

Build each source variant into its own target directory. Then run:

```bash
python3 research/row-fn-engine/performance/probe/run.py \
  /tmp/row-fn-before/release/row-fn-performance-probe \
  /tmp/row-fn-after/release/row-fn-performance-probe \
  bool /tmp/row-fn-results/bool

python3 research/row-fn-engine/performance/probe/summarize.py \
  'before=/tmp/row-fn-results/bool-before-*.csv' \
  'after=/tmp/row-fn-results/bool-after-*.csv'
```

`run.py` alternates process order and records binary hashes.
Both binaries warm each case internally.
Do not run other builds or timed processes during these comparisons.
`summarize.py` pools observations and reports medians and inclusive quartiles.
Quartiles describe dispersion. They are not confidence intervals.

## Reconstruct recorded variants

Start from commit `9314964c7a`. A Git archive into a temporary directory also works.
Copy this probe directory into that checkout.
Apply the following patches from the `results` directory.
Each row describes the complete patch set relative to the base commit.

| Binary label in environment.json | Patch set | Probe source snapshot |
| --- | --- | --- |
| baseline | None | probe-initial.rs |
| cosine | cosine.patch | probe-initial.rs |
| cosine-fast | cosine.patch, then cosine-fast.patch | probe-initial.rs |
| bool | cosine.patch and bool-dense.patch | probe-initial.rs |
| selected | cosine.patch and selected-experiment.patch | probe-initial.rs |
| control | None | probe-expanded.rs |
| narrow | cosine-narrow.patch and bool-dense.patch | probe-expanded.rs |
| utf8-before | bool-dense.patch | probe-expanded.rs |
| utf8 | bool-dense.patch and utf8.patch | probe-expanded.rs |
| diagnostic | bool-dense.patch and utf8.patch | probe-diagnostic.rs |
| final | bool-dense.patch and utf8.patch | probe/src/main.rs |

Copy the named snapshot to `probe/src/main.rs` before compilation.
The selected experiment patch includes its dense Boolean prerequisite.
The final dense patch also includes the corrected and expanded correctness tests.
Those test-only updates do not participate in release benchmark builds.

The initial snapshot measures cosine widths 2, 32, and 256.
The expanded snapshot adds widths 1, 3, 4, 8, and 16.
The diagnostic snapshot adds direct discarded-error instrumentation and isolated invocation probes.
The final source corrects the iterator control to use a native slice iterator.
Early `invocation_direct_iterator` observations used `Buffer::iter` and are excluded from the collector conclusion.
The final source also has a clarified timer comment. That comment does not change executable code.

## Recorded comparisons

| Raw file prefix | Before | After or measurement |
| --- | --- | --- |
| bool-cosine / bool-bool | cosine | bool |
| selected-bool / selected-selected | bool | selected |
| cosine-baseline / cosine-cosine | baseline | cosine |
| cosine-fast-baseline / cosine-fast-cosine-fast | baseline | cosine-fast |
| narrow-control / narrow-narrow | control | narrow |
| utf8-utf8-before / utf8-utf8 | utf8-before | utf8 |
| final-invocation | Within the final binary | Public-operation controls |
| final-discard-bt0 / final-discard-bt1 | Same final source | Backtraces disabled/enabled |

`baseline-bt0`, `error-bt1`, `retry-bt1`, `discard-bt*`, and `invocation-*` are earlier observations.
The follow-up report identifies the runs used for each conclusion.
The source snapshots retain the earlier fixture differences.

## Compiler evidence

The build command emits optimized LLVM IR, assembly, and the measured binary together.
Search every `.ll` file because the package uses 16 codegen units.
[Function excerpts and artifact hashes](../results/compiler/provenance.json) identify the inspected functions.
The complete artifacts remain in the target directories recorded there.
Excerpts omit unrelated functions and LLVM metadata definitions.

No cross-target compiler artifact counts as native runtime evidence.
The x86 measurements in the original research remain separate from these ARM results.
