# Versioning and compatibility

This page is a high-level overview of Vortex's file compatibility guarantees. Read it alongside the
companion pages for the configuration details and the reasoning behind the design:

- [Using editions](versioning/using-editions.md) explains how to configure writers for older readers
  and diagnose compatibility errors.
- [Versioning design](versioning/design.md) explains how formats evolve, with a worked example and
  the invariants that preserve compatibility.
- [Compatibility matrix](versioning/compatibility.md) summarizes the combinations of writer version,
  edition, serialized format, and reader version.
- [Edition registry](versioning/editions.md) lists the formats and minimum versions for each edition.

Vortex guarantees **backward compatibility** for its frozen serialized formats: newer library
versions retain read support for files in those formats. Applications can upgrade their readers
without rewriting those files. Newer writers can also select formats that older readers support,
so the applications producing and consuming files do not need to upgrade together.

An _edition_ names a set of formats that a writer is allowed to put in a file. Once an edition is
[_frozen_](versioning/editions.md#freezing-an-edition), that set and its reader requirements stay fixed.
The first frozen edition is `core2025.05.0`, supported from version `0.36.0` of the Vortex Rust library.

This guarantee concerns file compatibility. The library's programming interfaces follow
[Rust's semantic versioning rules](https://doc.rust-lang.org/cargo/reference/semver.html), so an API
change can require application changes even when existing files remain readable.

## Writing for older readers

Consider two applications. A service writes files using Vortex `0.85.0`, while a query engine reads
them using Vortex `0.84.0`. These numbers identify the Rust library versions used by each application.

The service [selects `core2026.08.0` for writing](versioning/using-editions.md#writer-configuration).
That edition's recorded minimum reader version is `0.84.0`, so the query engine meets the version
requirement. With the required implementations
registered, it can read valid files successfully written within that edition's restrictions.

The service still uses its own library's array implementations and compression algorithms. Its
edition selection limits the formats it serializes. Improvements that preserve those formats do
not require an update to the query engine. The
[decimal example](versioning/design.md#example-decimal-children) shows how a newer implementation
can still write an older format.

If the service instead selects `core2026.08.3`, the edition permits additional formats and records
a minimum of `0.85.0`. The older query engine is no longer guaranteed to read every file the service
can produce. It can still read a particular file if that file uses only formats it supports.

The recorded minimum covers every format in the edition, including formats that a particular file
does not use.

## Compatibility requirements

**A successful write with edition checks enabled uses only permitted formats.** A reader that
supports all those formats can read the output. The reader must also understand the enclosing
[file format](file-format.md).

Meeting a [minimum library version](versioning/using-editions.md#reader-versions) is part of that
requirement. The application must also register the implementations that read the formats, including
any optional plugins. An independent plugin can have its own versions and compatibility policy.
Upgrading Vortex alone does not install it.

Edition selection does not guarantee that every input or custom writing strategy can produce a
permitted file. An array can need a format that the edition forbids, or a custom strategy can choose
an unsupported layout. The [write fails](versioning/using-editions.md#write-errors) when its
serialized output violates the selection.

[Draft editions](versioning/editions.md#draft-editions) have no frozen compatibility guarantee.
Custom formats written with edition checks disabled also fall outside the edition guarantee.

## Upgrades

The default Vortex session selects the newest frozen `core` edition, so a library upgrade can change
the default output permissions. To keep serving older readers, explicitly select an edition whose
requirements those readers meet. Change that selection when the readers can support the additional
formats.

Selecting an older edition affects writing. It does not prevent the same application from reading
newer formats that its registered implementations support.

```{toctree}
---
maxdepth: 1
hidden: true
---

versioning/using-editions
versioning/design
versioning/compatibility
versioning/editions
```
