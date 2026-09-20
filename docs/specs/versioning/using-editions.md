# Using editions

To write files for an older deployment, select editions whose formats that deployment supports.
The [versioning overview](../versioning.md) explains the guarantee. This page shows the Rust
configuration API and the requirements to check on each side of a deployment.

## Configure a writer

A _session_ holds the registered implementations, edition declarations, and enabled editions. The
following function creates write options targeting `core2026.08.0`, whose recorded minimum reader
version is `0.84.0`:

```rust
use vortex::VortexSessionDefault;
use vortex::editions::CORE_2026_08_0;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexResult;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

fn writer_for_older_readers() -> VortexResult<VortexWriteOptions> {
    let session = VortexSession::default();
    session.enable_edition(CORE_2026_08_0)?;

    Ok(session.write_options())
}
```

Use the returned options' `write` method to write an array stream to an output. The
[Rust quickstart](../../getting-started/rust.rst) covers the input and I/O setup. The example uses
file support from the `vortex` crate. It does not require the consuming application to select the
same edition in its reader session.

The default session registers the standard implementations and edition declarations. It currently
enables `core2026.08.3`. Calling `enable_edition` replaces the enabled edition from the same family.
Set the selection before starting the write, which captures the permitted formats at that point.

An edition declaration describes permitted formats. Registering it does not install the code to
read or write those formats. When constructing a session without the defaults, register the required
implementations and declarations, then enable the target editions. Enabling an unregistered edition
returns an error. A selection that permits no components cannot serialize any edition-governed
component.

## Select optional features

An _edition family_ groups editions for related formats. The `core` family covers the default
writer's formats. Optional features have their own families, such as `tensor` and `zstd`, so they can
add formats without changing an application's `core` selection.

A writer selects at most one edition per family. Selecting `core2026.08.0` and `tensor2026.04.0`
permits every component in either edition. Within one family, a later edition includes all earlier
members. Across families, the selections are independent.

Check the [registry](editions.md#edition-registry) before enabling an optional family. For example,
`tensor2026.04.0` is a draft and has no frozen minimum reader version. Adding it does not extend
`core`'s frozen guarantee to the tensor formats. Both applications need the appropriate tensor
implementations.

An edition name such as `core2026.08.3` contains its family, year, month, and a number distinguishing
editions in that family and month. These are Vortex editions, separate from Rust language editions.

## Choose reader versions

For each selected frozen edition, find its recorded minimum version and its _origin_: the project
that supplies the component implementations. The `core` family's origin is `vortex`, so its
`min_library_version` refers to the shared Vortex Rust crate version. An independent plugin can name
a different origin with its own release numbers.

For editions with the same origin, use at least the highest recorded minimum. For different
origins, check each project separately. In both cases, register the implementations in the reader.
A sufficiently recent library without a required plugin is not enough.

For example, `core2026.08.0` records `0.84.0`, while `core2026.08.3` records `0.85.0`. A Vortex reader
using `0.85.0` with the required implementations meets either edition's requirements. The recorded
minimum covers every permitted format, including formats that an individual file does not use.
An older reader can sometimes read that file, but that is insufficient evidence that it supports the
writer's entire target edition.

## When writing fails

Edition checks apply to the actual serialized output, including child arrays, layouts, nested
extension dtypes, and stored aggregate functions. An array encoding can be permitted while one of
its children uses a forbidden encoding. The writer rejects that output too.

The default writer filters compression schemes by the formats they declare. A custom strategy or
compressor is responsible for constructing permitted representations. Selecting an edition does
not automatically reconfigure a custom strategy, and final checks still apply.

When a write fails because a format is forbidden, choose a permitted representation or strategy.
Alternatively, select a later edition after confirming that the readers meet its requirements.
[The decimal example](design.md#example-decimal-children) shows why an array's structure can require
a newer format even when its values appear suitable for an older one.

For custom or experimental output, `VortexWriteOptions::disable_editions()` disables the array,
layout, extension-dtype, and aggregate checks. It does not register missing implementations. Files
written this way have no edition compatibility guarantee, so producers and consumers must agree on
the required implementations themselves.

## When a reader reports an unknown ID

An unknown-ID error means that the reader has no registered implementation for that component.
Look up its kind and ID in the [registry](editions.md#edition-registry). The kind matters because an
array and a layout can share the same ID string while describing different formats.

- For a frozen edition, use at least the recorded minimum version of its origin and register the
  required optional module.
- For a draft edition, obtain a build that implements the component from the producer. A draft does
  not promise support in a published release.
- For a component absent from the registry, obtain its implementation from the producer and register
  it with the session.

Inspection and copying tools can use `allow_unknown` to retain the serialized data of unknown arrays,
layouts, and extension dtypes without interpreting it. Those objects are not available for ordinary
computation.

With `allow_unknown`, an unknown aggregate disables pruning for the affected zone-map layout. Its
data remains readable if the reader supports the other required formats. Without `allow_unknown`,
the unknown aggregate causes an error. Retaining unknown components or disabling pruning does not
establish full support for the file's formats.
