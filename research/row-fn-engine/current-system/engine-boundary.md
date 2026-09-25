<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Crate and trait choices

[Design](../architecture.md) | [Current framework](README.md)

An associated `Engine` type can collect host operations at the batch boundary. It is a useful
extraction option, provided the row operation depends only on its typed inputs and output writer.
This page describes the option and the contracts that constrain it. It is not a compiled design.

## A possible crate structure

| Component | Responsibility |
| --- | --- |
| `row-fn-core` | Functions, typed views, tuples, visitors, planning, traversal, and failure evidence. |
| `row-fn-vortex` | Vortex decoding, batch operations, errors, registration, and persistence. |
| `row-fn-arrow` | Arrow readers, writers, fields, validity, and scalar operands. |
| `row-fn-datafusion` | `ScalarUDFImpl` wrapper over the Arrow adapter. |
| C or C++ shim | Batch calls, foreign ownership, error translation, and host registration. |

The core's buffer and lane-kernel dependencies remain a separate choice. A crate with no
`vortex-array` dependency can still depend on reusable Vortex memory utilities. A library with no
Vortex dependencies at all requires a wider extraction.

## An Engine interface

An engine interface needs associated column, scalar, context, error, selection, and type-system
representations. Its operations fall into four groups:

- Inspect column length, type, constant representation, and validity.
- Decode or lend typed input views, with stable owners and lengths.
- Allocate and finish typed output, including null and constant results.
- Apply host column operations such as filtering, metadata attachment, and explicit conversion.

The [dependency inventory](dependencies.md) maps each group to current code. A single trait keeps
the adapter boundary visible. Separate input and output capabilities avoid a large mandatory API
for hosts that support only a few domains.

The recommendation is to start with this responsibility split and prove the smallest useful trait
set. An exact method count or line-count estimate does not establish extraction feasibility.

## Couplings that need an explicit decision

**Errors.** A small core row error can convert into a host error at the batch boundary. Keeping host
errors throughout the executor preserves host diagnostics but exposes the host to generic code.
Both options need separate categories for binding, row-local, and infrastructure failures.
Deferred evidence must remain compact, with rich errors constructed only when required.

**Constants.** Current `batch_const` recognizes plain constants, masked constants, and extensions
over constant storage. The tensor decoder also handles those wrappers. Classifying each input once
in the Vortex adapter can centralize this policy. Extension-over-constant recognition still needs
the extension's semantic guarantee. It is not a universal storage rule.

**Context.** The current row module passes `ExecutionCtx` to array execution, validity resolution,
and scalar extraction. It also supplies the allocator for output payloads. The row arithmetic does
not inspect session services. An opaque associated context can preserve this boundary. The existing
allocation argument is separate from physical sink parameters, so host glue can provide resources
without expanding the row operation's API.

**Registration.** `ScalarFnId`, serialization, and `VortexSession` belong to Vortex expression
persistence. A core name can support diagnostics, but cross-host identity also needs a namespace
and semantic version. A name alone does not prove that two operations are equivalent.

**Blanket implementations.** In a separate adapter crate, `impl<F: RowFn> ScalarFnVTable for F`
violates the orphan rules. That crate owns neither the host trait nor the generic self type.
`VortexRowFn<F>` provides a local self type. It also permits custom host hooks alongside shared
row execution. Keeping an implementation in the trait-owning crate is another legal arrangement,
but retains the existing coherence constraint.

**Buffers and lane kernels.** `OutputElement::build_from` exposes `IndexedSource` in its public API.
The owner of that trait becomes part of the portable dependency surface. Reusing `vortex-compute`
and `vortex-buffer` preserves the current implementation. Extracting their required parts gives
other hosts a smaller dependency graph. Duplicating the loops creates another implementation whose
safety and generated code can drift.

**Output storage and reuse.** `OutputElement::Buffer` and `OutputBuffer` already let the output
implementation allocate storage, lend slots, and publish an array. Primitive publication reuses its
allocation. Generalize this boundary for host results instead of adding another collection layer.
Borrowed strings still need ownership of the input buffers. Existing `OutputSink::finish` does not
receive those owners. Sharing input payloads needs a separate ownership contract.

## Safety ownership after extraction

| Invariant | Responsible component |
| --- | --- |
| A decoded view has stable addressable indices and a retained owner. | Input adapter. |
| Every input view covers the validated row domain. | Executor, before unchecked access. |
| Selection indices are ordered as required and remain inside that domain. | Selection constructor and executor. |
| Distinct row handles address the intended output rows. | Sink implementation. |
| A write token belongs to the exact row that the callback initialized. | Sink API and callback contract. |
| Initialization remains intact until the callback returns its token. | The caller of `InitializedElement::write` or `InitializedRow::fill`. |
| Output slots retain their contents across views and permit safe abandonment. | `OutputBuffer` implementation. |
| Skipped rows contain safe placeholders before full-array export. | Executor and sink initializer. |
| An error or unwind can abandon every partially initialized prefix safely. | Output owner and executor. |
| UTF-8 unchecked access follows validation of the exact retained bytes. | String input adapter. |
| Nested buffers, offsets, and children satisfy the final host layout. | Output adapter. |

Trait generalization changes where these proofs cross a boundary. The implementation cannot assume
that moving the loops preserves every proof automatically. The [current contracts](contracts.md)
give the input, sink, and token rules with pinned source links.

## Migration order

1. Introduce type-use and ownership boundaries while Vortex remains the only production host.
2. Separate Vortex registration from typed execution with an explicit wrapper.
3. Exercise the real executor with another adapter and representative semantic types.
4. Move the proven boundary into crates, preserving the source shapes required by compiler evidence.
5. Stabilize the public traits after host conformance and performance measurements.

Performance fixes remain separate changes. A crate move does not establish that a cache, a new
collector, or a decoder rewrite preserves behavior.
