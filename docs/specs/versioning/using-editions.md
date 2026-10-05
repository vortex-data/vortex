# Using editions

To write files for an older Vortex version, select editions whose wire formats that version
supports. See [Versioning](../versioning.md) for the compatibility guarantee.

## Writer configuration

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
file support from the `vortex` crate.

The default session registers the standard implementations and edition declarations, and it
currently enables `core2026.08.3`. Calling `enable_edition` replaces the enabled edition from the
same family. Set the selection before starting the write, because that is when the writer captures
the permitted wire IDs.

An edition declaration describes permitted formats, but registering it does not install the code to
read or write those formats. When constructing a session without the defaults, register the required
implementations and declarations, then enable the target editions. See
[Registering plugins](../../developer-guide/internals/session.md#registering-plugins) for the
registration API. Enabling an unregistered edition returns an error, while a selection that permits
no components prevents the writer from serializing any component governed by editions.

## Edition families

An _edition family_ groups editions for related formats. The `core` family covers the default
writer's formats. Optional features have their own families, such as `tensor` and `zstd`, so they
can add formats without changing an application's `core` selection.

A writer selects at most one edition per family. Selecting `core2026.08.0` and `tensor2026.04.0`
permits every component in either edition. Within one family, a later edition includes all earlier
members. Across families, however, the selections are independent.

Check the [registry](editions.md#edition-registry) before enabling an optional family. For example,
`tensor2026.04.0` is a draft and has no frozen minimum reader version. Adding it does not extend
`core`'s frozen guarantee to the tensor formats. Both applications still need the appropriate tensor
implementations.

An edition name such as `core2026.08.3` contains its family, year, month, and a number
distinguishing editions in that family and month. These are Vortex editions, separate from Rust
language editions.

## Reader versions

For each selected frozen edition, find its recorded minimum version and its _origin_ in the
[registry](editions.md). The origin is the project that supplies the component implementations. The
`core` family's origin is `vortex`, so its `min_library_version` refers to the shared Vortex Rust
crate version. An independent plugin can name a different origin with its own release numbers.

For editions with the same origin, use at least the highest recorded minimum, whereas editions from
different origins require a separate version check for each project. In both cases, the reader must
register the required implementations, since meeting the version requirement alone does not make a
plugin available.

## Write errors

The writer rejects forbidden formats in arrays, children, layouts, nested extension dtypes, and
stored aggregate functions.

Selecting an edition restricts the permitted output, but does not automatically reconfigure a custom
strategy or compressor. Those implementations must therefore construct permitted representations
themselves. The default compressor's filtering is described in [Compression](design.md#compression).

When a write fails because a format is forbidden, choose a permitted representation or strategy.
Alternatively, select a later edition after confirming that the readers meet its requirements.
[The decimal example](design.md#example-decimal-children) shows why an array's structure can require
a newer format even when its values appear suitable for an older one.

For custom or experimental output, `VortexWriteOptions::disable_editions()` disables the array,
layout, extension-dtype, and aggregate checks. However, it does not register missing
implementations. Files written this way have no edition compatibility guarantee, so producers and
consumers must agree on the required implementations themselves.

## Unknown IDs

An unknown-ID error means that the reader has no registered implementation for that component. Look
up its kind and ID in the [registry](editions.md#edition-registry). The kind matters because an
array and a layout can share the same ID string while describing different formats.

- For a frozen edition, use at least the recorded minimum version of its origin and register the
  required optional module.
- For a draft edition, obtain a build that implements the component from the producer. A draft does
  not promise support in a published release.
- For a component absent from the registry, obtain its implementation from the producer and register
  it with the session.

Inspection and copying tools can use `allow_unknown` to retain the serialized data of unknown
arrays, layouts, and extension dtypes without interpreting it. However, retaining those objects does
not make them available for ordinary computation.

With `allow_unknown`, an unknown aggregate disables pruning for the affected zone-map layout, but
the layout's data remains readable if the reader supports the other required wire formats. Without
`allow_unknown`, the unknown aggregate causes an error.
