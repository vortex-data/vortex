# Versioning and compatibility

Vortex retains read support for its frozen serialized formats as the library evolves. An application
can upgrade its Vortex reader without rewriting files that use those formats. Writers can also
restrict their output to formats that an older deployment supports, so the applications producing
and consuming files do not need to upgrade together.

An _edition_ names a set of formats that a writer is allowed to put in a file. Once an edition is
_frozen_, that set and its reader requirements stay fixed. The first frozen edition is
`core2025.05.0`, supported from version `0.36.0` of the Vortex Rust library. Later versions retain
read support for its formats and those of subsequent frozen editions.

This guarantee concerns file compatibility. The library's programming interfaces follow
[Rust's semantic versioning rules](https://doc.rust-lang.org/cargo/reference/semver.html), so an API
change can require application changes even when existing files remain readable.

## A newer writer and an older reader

Consider two applications. A service writes files using Vortex `0.85.0`, while a query engine reads
them using Vortex `0.84.0`. These numbers identify the Rust library versions used by each application.

The service selects `core2026.08.0` for writing. That edition's recorded minimum reader version is
`0.84.0`, so the query engine meets the version requirement. With the required implementations
registered, it can read valid files successfully written within that edition's restrictions.

The service still uses its own library's array implementations and compression algorithms. Its
edition selection limits the formats it serializes. Improvements that preserve those formats do
not require an update to the query engine.

If the service instead selects `core2026.08.3`, the edition permits additional formats and records
a minimum of `0.85.0`. The older query engine is no longer guaranteed to read every file the service
can produce. It can still read a particular file if that file uses only formats it supports.

An edition's minimum therefore answers a deployment question: which version supports *all* the
formats this writer is permitted to use? It is not necessarily the earliest version that can read
one particular file.

## What the guarantee requires

**A successful write with edition checks enabled uses only permitted formats.** A reader that
supports all those formats can read the output. The reader must also understand the enclosing
[file format](file-format.md).

Meeting a minimum library version is part of that requirement. The application must also register
the implementations that read the formats, including any optional plugins. An independent plugin
can have its own versions and compatibility policy. Upgrading Vortex alone does not install it.

Edition selection does not guarantee that every input or custom writing strategy can produce a
permitted file. An array can need a format that the edition forbids, or a custom strategy can choose
an unsupported layout. The write fails when its serialized output violates the selection.

Draft editions have no frozen compatibility guarantee. Custom formats written with edition checks
disabled also fall outside the edition guarantee. The [registry](versioning/editions.md) distinguishes
frozen editions from drafts and lists their formats and recorded minimum versions.

## Upgrading a deployment

An application upgrading its reader must retain the plugins needed by its existing files. The
updated implementations must continue to decode their frozen formats, including formats that
writers no longer choose.

An application upgrading its writer must also consider its consumers. The default Vortex session
selects the newest frozen `core` edition, so a library upgrade can change the default output
permissions. To keep serving older readers, explicitly select an edition whose requirements those
readers meet. Change that selection when the readers can support the additional formats.

Selecting an older edition affects writing. It does not prevent the same application from reading
newer formats that its registered implementations support.

## Configuration, design, and reference

- [Using editions](versioning/using-editions.md) shows how to configure a writer, check reader
  requirements, and diagnose incompatible output or missing implementations.
- [How Vortex evolves its formats](versioning/design.md) explains the separation between library
  releases, serialized formats, and editions through a worked example and compatibility tables.
- [Edition registry](versioning/editions.md) lists the formats and minimum versions, with instructions
  for introducing formats and maintaining edition records.

```{toctree}
---
maxdepth: 1
hidden: true
---

versioning/using-editions
versioning/design
versioning/editions
```
