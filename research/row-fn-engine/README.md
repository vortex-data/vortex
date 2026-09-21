<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn as a portable library

**RowFn can become a portable function library, but replacing `DType` alone is insufficient.** The
useful boundary separates function semantics, typed row access, and host integration. A function can
then define one operation for every adapter that supports its semantic and storage requirements.

This is a research report against Vortex commit
[`96bd521eb0`](https://github.com/vortex-data/vortex/commit/96bd521eb0565555def2af7b8e97e96891728da6),
dated 2026-09-21. The proposed library and adapters are not implemented. Start with the findings
below, then follow the links for evidence and design details.

## Findings

### 1. A generic type system needs a semantic contract

An associated `Host::Type` can remove the direct Vortex dependency. It cannot tell one function how
to interpret every host's timestamps, decimals, extension types, or nested nullability.

The strongest candidate combines host-native types with a small, extensible semantic protocol.
Binding establishes supported type capabilities once. Typed readers and writers then handle the
batch without inspecting a logical type for every row.

This permits new hosts and type extensions. It does not make arbitrary unknown types executable
without a definition of their operations. Unsupported mappings remain explicit errors.

An [isolated Rust proof](type-system/compiled-proof.md) compiles one binder and row loop against two
independent adapter crates. It establishes the generic and borrowing mechanisms, not full host
integration. Read [type-system options and counterexamples](type-system/README.md) for the remaining
semantic questions.

### 2. Vortex dependencies extend through input, output, and evaluation

Current input decoders accept `ArrayRef` and `ExecutionCtx`. Output builders and sinks construct
Vortex arrays. Constants, validity, allocation, output labels, errors, serialization, and the scalar
function vtable add further ties.

The portable pieces are the typed row operation and much of the traversal machinery. Host adapters
need to own column access and output construction. Query optimizations and expression evaluation
remain separate integration layers.

Read the [current execution path and dependency inventory](current-system/README.md), then the
[proposed architecture](architecture.md).

### 3. Host integration has measurable boundaries

An arrow-rs adapter and a DataFusion UDF adapter can share column access. DataFusion still needs its
own binding and function metadata. Arrow transport alone does not provide that contract.

DuckDB's C scalar callback receives flattened inputs in the inspected version. A C++ adapter can
retain native vector information, with a different versioning contract. DataFusion's existing FFI
route also expands scalar arguments to arrays, unlike native Rust UDF calls.

The target can be one shared function definition with separately compiled adapters. One binary
that loads everywhere requires an additional ABI design.

Read the [Arrow, DataFusion, and DuckDB integration research](integrations/README.md).

### 4. Overhead is a set of costs, not one framework percentage

An exact description needs a specified entry point, input representation, execution path, output
contract, and baseline. Binding, argument construction, decoding, validity, row execution, and output
construction contribute different costs.

Source work counts and native timings answer different questions. The report separates them and
records the local experiment, controls, reproduction, and limitations. Those results concern the
current Vortex implementation, not the proposed portable library.

On Apple M4 Max, the measured `i64` function adds about 101 to 104 ns against a shared output
collector at 1 to 1,024 rows. At 16,384 rows, RowFn takes 2.53 us. The direct iterator takes 1.44 us,
and the direct shared collector takes 2.43 us. The collector difference therefore matters alongside
the batch wrapper. Reused-argument paths make the same allocation requests in this fixture.

Read the [performance model and measurement report](performance/README.md).

### 5. Demand and definedness need contracts separate from validity

Input demand says which results the caller needs. Output definedness says which results are
complete. Validity says which complete results are non-null. A computed null is defined. An
unrequested row is not necessarily null.

These are separate facts, but they do not require separate allocated bitmaps on every call.
Compact batches and successful full-batch execution can carry some guarantees implicitly.

The request must reach child evaluation and decoding. A final-loop mask cannot suppress earlier
errors. Dictionary rewrites and partial-result caches also need the correct row domain.

Read the [definedness model, examples, and evaluator changes](definedness/README.md).

## Recommended first experiment

Extract a small semantic binder and typed executor behind Vortex and plain-slice adapters. Then add
Arrow/DataFusion using the same function definitions. Use integer arithmetic, timestamp metadata,
UTF-8 output, and a fixed-size vector case to expose different boundaries.

Introduce demand through one conditional expression before fixing the public execution API. Keep
batch-level implementations available for functions that reuse whole buffers or exploit encodings.

The [next experiments](next-steps.md) define the evidence needed to accept each design decision.
The [prior-art comparison](prior-art.md) covers Velox and Substrait. The
[evidence guide](evidence.md) records the source and measurement boundaries.
