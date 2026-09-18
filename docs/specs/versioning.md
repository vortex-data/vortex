# Versioning

These docs explain how Vortex keeps files readable as its Rust implementation changes. They cover
how to write files for older deployments and how to add encoding formats without breaking existing
files.[^proof]

## The Vortex Rust library

Vortex's Rust library provides the array types and algorithms that an application uses to compress
and process data **in memory**. It also reads and writes files. The code is split across the
`vortex` crate and supporting crates such as `vortex-array` and `vortex-file`.

The _library version_, such as `0.85.0`, is the version shared by these crates when they are
published. Changes to the crates follow
[Rust's semantic versioning rules for crates](https://doc.rust-lang.org/cargo/reference/semver.html).
Published versions of the `vortex` crate are on
[crates.io](https://crates.io/crates/vortex/versions), with release notes on
[GitHub](https://github.com/vortex-data/vortex/releases).

In these docs, a _reader_ is the Vortex code that an application uses to read files. A _writer_
is the Vortex code it uses to write files. Each uses a particular crate version and the component
implementations registered by that application. One application can use both.

## Editions

When writing a file, an application selects an _edition_: a named set of formats it is allowed to
**serialize to disk**. To write files for an older deployment, select an edition whose formats
the Vortex code in that deployment can decode. The application writing the file still uses the
in-memory array types and algorithms provided by its own version of the Vortex crates.

Editions belong to _families_. The `core` family covers the default writer's formats, while optional
features can have their own families. Edition names use dates: `core2026.08.3` belongs to the `core`
family, `2026.08` gives its year and month, and `3` distinguishes editions in that family and month.
These are Vortex editions, separate from Rust language editions such as Rust 2024.

Once an edition is _frozen_, its permitted formats and reader requirements stay fixed. New formats
go into later editions.

## Array plugins

An _array plugin_ implements the read and write code for an array encoding. It can read an older
serialized format into the current in-memory array type, and write that type in an older format when
the array's structure allows it. **The serialized format and the Rust array type do not have to
change together.**

For example, a service can update its Vortex crates while keeping the edition it previously
selected for writing. The service uses the new crate version's array implementations, but writes
only formats permitted by that edition. Applications that could decode all those formats before
the update can still decode the output afterward, without updating their own Vortex crates.

**Encoding changes must be additive:** support for a new serialized format must preserve read
support for the old formats. The
[decimal encoding example](versioning/arrays-and-compression.md#example-decimal-children) shows
one array implementation supporting two serialized formats.

## Suggested reading order

After this overview, read the pages in this order:

1. [Using editions](versioning/using-editions.md): select formats for a writer and find the crate
   versions and plugins its readers need.
2. [Arrays and compression](versioning/arrays-and-compression.md): follow a decimal array through
   compression, serialization, and reading to see how one implementation supports multiple formats.
3. [Compatibility](versioning/compatibility.md): use the invariants and matrix to work through the
   combinations of old and new readers, writers, editions, and serialized formats.
4. [Edition lifecycle and registry](versioning/editions.md): add or revise a format, or look up an
   edition's components and minimum reader version. This page is also a reference to return to later.

```{toctree}
---
maxdepth: 1
hidden: true
---

versioning/using-editions
versioning/arrays-and-compression
versioning/compatibility
versioning/editions
```

The [implementation roadmap](versioning/arrays-and-compression.md#implementation-roadmap) covers
the remaining work, including configuring compression to produce formats permitted by the target
edition.

[^proof]: For a mathematical treatment, see the [formal proof of the versioning model (PDF)](../_static/versioning-proof.pdf).
