<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# A concrete extraction boundary

The proposed core operates on typed row views and row writers. Host adapters bind their logical
types, prepare those views, allocate outputs, and publish host arrays. This keeps Vortex ownership
and execution context out of the inner API.

This page gives the boundary implied by the current source. The [architecture proposal](../architecture.md)
and [next steps](../next-steps.md) place it in the wider research plan.

## Keep the row execution contract

The core can retain these existing concepts:

- Typed input tuples and borrowed row values.
- Constant-versus-column addressing.
- Preparation from constant operands.
- Dense and selected row traversal.
- Owned output and sink output.
- Immediate errors and deferred failure evidence.
- Output initialization and stable-view safety contracts.

These concepts already separate row computation from array metadata in the visitor and view
interfaces. Their implementations still name Vortex arrays, so extraction requires interface changes.
[Sources: visitor][visitor], [input view][input], [sink][sink].

The core can own a reference bitmap representation without requiring every host to copy masks into
it. Its traversal boundary needs a documented row domain and efficient views of the host selection.
A concrete mask adapter can resolve representation differences once per batch.

## Put column operations in adapters

Each host adapter owns input retention, decode or materialization, constant recognition, validity
access, filtering, output allocation, metadata attachment, and error translation. The Vortex adapter
also owns `ScalarFnVTable`, session registration, expression rewrites, and persistence.

The separation uses two lower-level contracts:

1. A decoded input owner lends a typed view whose addressable domain remains stable.
2. An output owner lends row writers and publishes a host result only after initialization succeeds.

The output side already has `OutputElement::Buffer` and `OutputBuffer`, which separate writable
slots from allocation and publication. They still name Vortex resource and result types. Generalize
those types while retaining the storage and initialization contract. A second adapter can use
ordinary Rust buffers without requiring the core to define a universal array object.
[Source: output buffer][output-buffer].

Keeping the host decoder separate matters for compressed input. A host can decode one vector, expose
a selection into existing storage, or apply a whole-column optimization before entering the row
core. Those choices preserve one row-kernel definition while changing physical execution.
[Sources: primitive decoder][primitive], [dictionary rewrite][dictionary].

## Make ownership explicit

The input owner must outlive every view and prepared value that borrows from it. The host can retain
an Arrow array, a Vortex decoded buffer, or another host's pinned storage behind that owner. The
portable core only needs the resulting typed view.

The output owner exposes its allocation and initialization policy before traversal. Current Vortex
outputs use the execution allocator, and primitive publication retains the same allocation. A host
adapter can use this boundary to write directly into host-owned buffers.
Borrowed string results need a retained owner or a copy into the output arena.

This design adds ownership work at the batch boundary, not a virtual call per row. Static generic
views and writers remain candidates for monomorphized loops. Whether a particular boundary preserves
vectorization requires measurement and compiler inspection.
[Source: existing loop structure][loop].

Allocation is a separate capability from type dispatch. An adapter must define alignment, memory
accounting, resource errors, and whether output buffers can retain input storage. A generic dtype
trait cannot answer those questions.

## Keep host optimizer facts distinct from row-loop policy

The semantic function description can declare strictness, determinism, and logical fallibility.
The prepared decoder can declare safe addressability and whether it can reject row payloads.
The chosen output path can declare whether writes are fallible.

These properties answer different questions. An optimizer asks whether it can evaluate additional
logical rows. A row executor asks whether it can decode and read particular payloads safely.
The [error contract](contracts.md) explains the distinction that the current Vortex flags already
expose.

The first extraction can preserve the current strict function class and its callback restrictions.
Explicit determinism and evaluation-count metadata are proposed additions. Null-producing functions,
variadic arguments, device kernels, and aggregates need additional contracts. They do not follow
automatically from generic types or generic arrays.

[Back to the overview](README.md).

[visitor]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
[input]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs
[sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs
[primitive]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/primitive.rs
[dictionary]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/arrays/dict/compute/rules.rs
[loop]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/sink.rs
[output-buffer]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs
