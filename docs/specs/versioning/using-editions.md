# Using editions

To write files for another deployment, select editions whose formats that deployment can read.
This page explains how to configure that selection and find the crate versions and plugins
required to read the output.

## Selecting edition families

An _edition family_ groups editions for a related set of formats. For example, `core` covers the
default writer's formats, while `tensor` and `zstd` cover optional features. Keeping these in
separate families lets a writer enable an optional feature without changing its `core` selection.

**A writer selects at most one edition from each family.** Selecting `core2026.08.3` and
`tensor2026.04.0`, for example, permits every component in either edition. Within a family, later
editions include the components from all earlier editions, so selecting a later edition adds
formats to the permitted set.

## Components and wire IDs

Reading a file requires more than decoding its compressed arrays. The reader also needs to
understand how the file is laid out, how to interpret custom data types, and how to use any stored
summaries to skip irrelevant data. Editions cover each of these parts of the file.

An edition lists _components_, such as array encodings and file layouts. Each component has a kind
and an ID stored in the file, called its _wire ID_. **The kind and ID together identify the
component.** The array and layout encodings named `vortex.chunked`, for example, are distinct
components despite sharing the same ID string.

_Zone maps_ store summaries such as the minimum and maximum in a group of rows. A reader can use
these to skip a group when no value in it can match a filter. The _aggregate functions_ that compute
these summaries also have wire IDs for their serialized definitions.

| Kind | What the wire ID identifies |
|---|---|
| `array` | An array's serialized representation |
| `layout` | A node in the file's layout tree |
| `dtype` | An extension dtype: a custom logical data type in the schema |
| `aggregate` | An aggregate function stored in a zone map |

## Configuring a writer

A _session_ holds the writer's registered implementations and edition selection. The default
session from the `vortex` crate targets `core2026.08.3`. A session constructed without those defaults
needs its editions registered and enabled before writing.

_Registering_ an edition makes its declaration available to the session, including which components
it permits. _Enabling_ the edition selects those components for writing. **The component
implementations must be registered separately.** Enabling another edition from the same family
replaces the previous selection.

To write files for an older deployment, select a `core` edition whose recorded minimum does not
exceed that deployment's Vortex crate version. The deployment must also register the required
component implementations. Optional modules can enable their own families alongside `core`, such
as `tensor2026.04.0` for tensor support or `zstd2026.02.0` for Zstd buffer wrapping. If the selected
editions permit no components, the writer cannot serialize any edition-governed component.

For custom or experimental formats outside the edition declarations, the Rust writer provides
`disable_editions()`. This disables checks for arrays, layouts, extension dtypes, and aggregate
functions, while still requiring their implementations to be registered. Files written with these
checks disabled have **no edition compatibility guarantee**.

## Checks during writing

The final checks apply to what the writer actually serializes. A permitted array encoding can
contain child arrays with other encodings, so the writer must check those children too.

| Kind | Check |
|---|---|
| Arrays | Check the serializer's returned ID, then serialize and check its children recursively. |
| Layouts | Check every serialized layout ID. The layout strategy must use permitted layouts. |
| Extension dtypes | Check all extension dtypes in the schema, including nested ones, before writing bytes. |
| Aggregate functions | Check every function stored in a zone map against the edition and its format contract. |

A zone-map aggregate that the edition forbids causes the write to fail. Silently omitting it
changes which filters can use the configured zone map to skip rows. This differs from an aggregate
that does not apply to a column's data type: the writer omits that aggregate, so there is no
serialized component to check.

For example, `core2026.08.0` declares `min`, `max`, `bounded_min`, `bounded_max`, `nan_count`, and
`null_count`. It does not declare `sum` because zone maps do not store sums. File-level statistics
store sums in a fixed legacy field governed by the enclosing format's contract.

## Choosing a reader version

A frozen edition records the minimum version of the code needed to read all its components.
For example, version `0.85.0` of the Vortex crates implements decoding for every format in
`core2026.08.3`. It also retains decoding for the formats in `core2026.08.0`, whose recorded minimum
is `0.84.0`. The application reading the file must register those implementations in its session.

The edition family names an _origin_, the project that supplies its component implementations.
For `core`, the origin is `vortex`, so the edition's `min_library_version` refers to the shared
[Vortex Rust crate version](../versioning.md#the-vortex-rust-library). An independent plugin can
name a different origin with its own version numbers.

**Each origin has its own minimum version.** When selected editions share an origin, use a version
of that project's code at or above the highest recorded minimum. When the origins differ, check
each project separately. In particular, check an independent plugin's version even if the Vortex
crates already meet the `core` requirement. The reader must also register the required component
implementations, including any optional plugins.

The recorded minimum covers every format the edition permits, including ones that a particular
file does not use. An earlier crate version can therefore sometimes read that file, even though
it cannot read every file permitted by the edition.

## Unknown-component errors

An unknown-ID error means that the reader has no registered implementation for a component in the
file. Find its kind and ID in the [registry](editions.md#edition-registry):

1. For a frozen edition, use at least the recorded minimum version of its origin and register any
   required optional module.
2. For a draft edition, use a build that implements the component. The draft does not guarantee
   support in a published crate version. Ask the producer which build to use.
3. For a component absent from the registry, obtain its implementation from the producer and
   register it with the session.

Inspection and copying tools can use `allow_unknown` to preserve unknown arrays, layouts, and
extension dtypes. The reader retains their serialized data without interpreting it, so these
objects are not available for computation. An unknown aggregate disables the affected zone-map
pruning. Data reads remain correct, but they cannot use that aggregate to skip rows.

[Next: Arrays and compression](arrays-and-compression.md)
