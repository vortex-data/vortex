<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# A measurement program for an extracted engine

Measure the current implementation before changing the type system. Preserve those measurements as
adapter acceptance criteria. The [local probe](local-measurements.md) starts this process for one
canonical primitive path. It does not complete the matrix below.

## Required benchmark boundaries

Use the same row operation at four boundaries:

1. A direct loop over already decoded views into the required output representation.
2. The RowFn typed executor with the same views and output representation.
3. RowFn batch execution over arrays, including type dispatch, decoding, and output construction.
4. The native host function API, including any conversion and final materialization.

The first two isolate loop code generation. Their comparison needs an internal harness or a planned
prepared-executor API. The current public API starts at the third boundary. Do not expose private
implementation details solely to claim a smaller benchmark number.

Compare each host adapter with a native function in the same host. A Rust callback from DuckDB C++,
a C API callback, and a DataFusion Rust UDF have different batch entry costs. Treat FFI adapters as
separate configurations.

## Core matrix

| Dimension | Initial cases | What it resolves |
| --- | --- | --- |
| Batch size. | `0`, `1`, `8`, `64`, `128`, `1,024`, `2,048`, `16,384`, `262,144`. | Fixed cost, common batch sizes, throughput, and cache transitions. |
| Operation. | Add one, binary addition, comparison, UTF-8 length, substring, and geometry predicate. | Cheap lanes, packed output, variable-width access, and expensive callbacks. |
| Arity. | `0`, `1`, `2`, `4`, `5`, `12`. | Nullary behavior and the current inline argument-storage limit. |
| Input representation. | Canonical, constant, dictionary, compressed, sliced, and nested. | Decoding, flattening, wrappers, and pointer offsets. |
| Constants. | None, left only, right only, and all inputs. | Both broadcast orientations and constant folding. |
| Logical type. | Primitive, parameterized timestamp, fixed-size list, extension, and nested list. | Type metadata cost and parameter propagation. |
| Validity. | Non-nullable, nullable all-valid, all-null, and partial validity. | Metadata-only validity versus materialized masks. |
| Mask structure. | Random bits, alternating bits, long runs, and one selected row. | Traversal costs that density alone hides. |
| Error policy. | Infallible, immediate error, deferred no-error, null-only failure, and observable failure. | Successful execution, retry, and error materialization. |
| Output. | Native primitive, packed Boolean, string sink, fixed-size list sink, and extension label. | Construction cost and host buffer reuse. |
| Preparation. | No preparation, cheap constant work, and expensive constant parsing. | Cold binding, per-batch preparation, and reuse. |
| Ownership. | Reused arguments, fresh argument wrapper, reused context, and fresh context. | Caller costs versus batch-executor costs. |
| Target. | Apple ARM, Linux ARM, and deployed x86 feature sets. | Target-specific vectorization and allocator behavior. |
| Build. | Production profile first, then a named compiler or CGU/LTO control. | Compiler sensitivity without mixing configurations. |

Avoid the full Cartesian product. Start with one case per route, then expand dimensions that change
the selected algorithm. Confirm route reachability from the input and policy before interpreting a
benchmark name.

## Demand-mask matrix

A future definedness API adds a demand mask distinct from input validity. Use these initial cases:

| Demand and input state | Required observation |
| --- | --- |
| Empty demand over valid input. | Zero row callbacks. Record remaining bind, decode, and output costs. |
| Dense demand over valid input. | Equivalent output and cost to the ordinary dense route. |
| Sparse demand over valid input. | Only demanded rows are defined. Record output initialization separately. |
| Dense demand over partially valid input. | Strict null propagation and no errors from null payloads. |
| Sparse demand crossing partially valid input. | Compute the correct demand/validity intersection. |
| An error exists only outside demand. | No observable error from that row. |
| An error exists in a demanded valid row. | The declared error policy remains observable. |
| A decoder rejects data outside demand. | Demonstrate the chosen decoder contract, rather than assuming callback masking is enough. |
| An all-constant input has empty demand. | No constant preparation that violates the empty-demand contract. |
| A variable-width or nested result has sparse demand. | No reads or drops of uninitialized descriptors or child values. |

A benchmark must consume only output rows that its contract defines. Otherwise, it can turn a safe
partial-output API into an invalid experiment. Measure demand-mask composition as well as row savings.

## Controls and reporting

Construct fixtures outside timing. Keep encoding, nullability, buffer offsets, input values, and
output consumption identical between compared implementations. Fix the CPU, compiler, allocator,
profile, target features, and dependency versions for each comparison.

Validate output dtype, length, validity, values, and error behavior before timing. A baseline that
returns the right non-null values but loses nullability is not equivalent. Compare constant encoding
when representation preservation is part of the contract.

Warm all cases. Alternate comparison order and repeat across processes. Include an unchanged direct
control in each group. Preserve raw observations, paired differences, and paired ratios. Report their
medians and dispersion. Show absolute time and time per row beside percentages.

Record allocation requests and requested bytes in a separate instrumented build. Time the ordinary
allocator build. Measure peak live memory separately if it affects the design decision.

Collect optimized LLVM IR for the final target that owns the concrete row executor. Confirm the
important loop in assembly. Check vector width, scalar tails, calls in the loop, constant routing,
and output packing. A library-only artifact can omit the downstream monomorphization.

Measure binary text size and compile time separately. Static specialization can preserve fast loops
while increasing compiled code. The output mode, argument types, constant handling, and error policy
all contribute possible specializations. A larger binary is not itself evidence of runtime slowdown.

## What existing Vortex benchmarks establish

[`row_fn_output.rs`][row-bench] measures 16,384 rows through `execute_rows`. It covers primitive inputs with infallible and deferred Boolean outputs, constant orientations,
deferred `i64`, and filtered owned and sink outputs. Its synthetic `FilteredI64` input forces the fallback for partially valid input.
It does not provide a matched direct baseline or a size sweep.

[`binary_ops.rs`][binary-bench] includes end-to-end primitive operators, short and long batches,
constant orientations, nullable data, decimal output, and Boolean output. The timed body constructs
the binary expression and executes it. Its reused context differs from the fresh context fixture
in `row_fn_output`.

[`compare.rs`][compare-bench] exercises several array kinds through the binary API. It uses different
row counts and fixtures from `row_fn_output`. Subtracting results across these suites does not isolate
RowFn overhead.

The repository benchmark profile uses 16 codegen units and `lto = false`. Cargo applies local thin
LTO across those units. The release profile uses one codegen unit and `lto = "off"`, which disables LTO.
Keep those configurations distinct. See [the profile definitions][profiles] and
[Cargo LTO semantics](https://doc.rust-lang.org/cargo/reference/profiles.html#lto).

## Acceptance criteria for the extraction

The extraction needs these demonstrations before a portable performance claim:

- Dense typed primitive and Boolean loops retain suitable vectorized code on each supported target.
- Repeated bound calls avoid repeated semantic type resolution where the binding contract permits it.
- Constant inputs remain constant across each host boundary that supports that representation.
- Null-only errors and undemanded errors remain unobservable under the declared policies.
- Adapters report required normalization and copies explicitly.
- Native host and extracted RowFn functions receive equivalent fixtures and return equivalent outputs.
- The benchmark report separates fixed setup cost, size-dependent cost, and output conversion cost.

Set numeric budgets after the baseline matrix is available. A universal percentage budget hides the
tradeoff between single-row calls, vector-sized batches, and expensive row operations.

[row-bench]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/benches/row_fn_output.rs
[binary-bench]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/benches/binary_ops.rs
[compare-bench]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/benches/compare.rs
[profiles]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/Cargo.toml
