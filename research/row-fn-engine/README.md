<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn engine research

**RowFn can become a shared function library for Vortex and other engines.** The useful boundary
separates function semantics, typed row execution, and host adapters. Replacing `DType` is part of
that work. Decoding, output ownership, errors, and registration need the same treatment.

The recommendation is to prove this boundary with Vortex and Arrow before stabilizing an API.
DataFusion can then reuse the Arrow adapter. DuckDB needs a separate choice between its C callback
and a version-matched C++ adapter.

This tree contains findings and design proposals from 2026-09-21, updated for the Vortex source
baseline on 2026-09-25. The portable library and host adapters are not implemented. The
[recent changes](current-system/recent-changes.md) record completed RowFn improvements. The
[evidence record](evidence.md) separates the updated source findings from historical experiments.

## Start here

These pages contain the main argument. The other files provide supporting detail.

| Page | What it answers |
| --- | --- |
| [Recent changes](current-system/recent-changes.md) | Which research concerns have been addressed since September 21? |
| [Design](architecture.md) | What belongs in the library, and which API choices remain open? |
| [Performance](performance/README.md) | What was measured, and which costs need attention? |
| [Demand and completion](definedness/README.md) | How can conditionals avoid errors in rows they do not need? |
| [Next steps](next-steps.md) | Which changes and experiments resolve the remaining questions? |

## Main findings

**The type system can be generic.** Current row loops already operate on typed values and views.
An isolated [Rust experiment](type-system/compiled-proof.md) demonstrates shared dispatch and
borrowing across two adapter crates. A portable function still needs a semantic contract.
An `i64` timestamp, a decimal coefficient, and an ordinary integer are not interchangeable inputs.

**Outer nullability can move to the binding boundary.** The core can represent a semantic type and
its outer nullability separately. Nested child nullability and extension metadata must survive.
A small built-in type vocabulary is useful, but hosts must be able to reject unsupported types.
The [type analysis](type-system/README.md) compares this with an extensible capability design.

**Most reusable logic sits between host operations.** Typed traversal, constants, preparation,
deferred failure evidence, and sink initialization belong in the core. Adapters own column access,
allocation, output construction, and host registration. `OutputElement::Buffer` and `OutputBuffer`
already separate output storage from traversal, with execution-allocator support. Extraction can
build on that boundary. Explicit wrappers such as `VortexRowFn<F>` avoid the blanket-implementation
constraints. The [source inventory](current-system/README.md) maps the boundary and its safety
obligations.

**The original measurements separate several costs.** The ARM experiment isolates about 101 to
104 ns of additional batch work with a shared collector. It also finds a larger collector difference
at 16,384 rows. The x86 sweep reports different setup costs and exposes UTF-8 validation, retry,
and nullable execution costs. These are different experiments, not one combined benchmark. The
[performance summary](performance/README.md) keeps both baselines and their limits visible. These
timings predate direct packed Boolean retry, the UTF-8 decode change, and allocator and mask fixes.

**Demand, completion, and validity are different facts.** A caller requests rows. A successful call
completes those rows, including null results. Validity says which completed results are non-null.
Demand must reach child evaluation, decoding, and preparation before the row loop. A bitmap added
only to that loop is insufficient. The [definedness contract](definedness/contract.md) states the
required behavior and safe output representation.

**The row API needs a batch alternative.** Buffer reuse, dictionary transforms, and fused kernels
can require whole-column access. Keep that path under the same function semantics. Existing
RowFn also has a narrower contract than a general UDF: strict null propagation, fixed arity,
synchronous execution, and no null result from valid inputs. Those extensions need separate design.

## Supporting detail

| Topic | Detailed reading |
| --- | --- |
| Current framework | [Execution](current-system/execution.md), [contracts](current-system/contracts.md), [consumers](current-system/consumers.md), [design history](current-system/design-history.md). |
| Type design | [Alternatives](type-system/alternatives.md), [binding contract](type-system/portable-contract.md), [type inventory](type-system/dtype-inventory.md), [host mappings](type-system/mappings.md). |
| Extraction | [Dependencies](current-system/dependencies.md), [crate and trait choices](current-system/engine-boundary.md). |
| Host adapters | [Arrow, DataFusion, and DuckDB](integrations/README.md), [storage and ownership](integrations/storage-and-ownership.md). |
| Execution semantics | [Worked cases](definedness/worked-cases.md), [API alternatives](definedness/design-options.md), [current Vortex behavior](definedness/current-vortex.md). |
| Measurements | [ARM results](performance/local-measurements.md), [x86 results](performance/x86-measurements.md), [optimization candidates](performance/optimization-candidates.md). |
| Prior art | [Comparison and lessons](prior-art.md), including Velox, Arrow, DuckDB, DataFusion, ClickHouse, Polars, Presto, Spark, and Substrait. |
