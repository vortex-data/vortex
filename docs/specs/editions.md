# Editions

Vortex defines an ever-growing set of serialized formats for arrays and other durable objects.
Once a format is published, every future version of Vortex can read it. **Editions** group these
formats and record when each joined that compatibility guarantee.

An edition member is a **serialized format**: the ID written into the file, its metadata schema,
and the meaning of its buffers, children, options, or other payload. Array and layout encodings,
aggregate functions, and extension dtypes are different kinds of serialized format covered by the
same rule. The read-forever guarantee attaches to those bytes and their meaning, not to the
in-memory implementation that produced them.

The first edition, `core2025.05.0`, contains the stable serialized array formats that Vortex
`0.36.0` could write. This is the release from which the Vortex file format is considered stable.
Later `core` editions add stable formats released after that boundary. Editions are additive: each
edition contains every member of the preceding edition in its family, plus new members. For
example, selecting `core2026.07.0` and `unstable2026.06.0` enables stable formats released through
July 2026 and unstable formats released through June 2026. The most recent required Vortex release
among the selected editions is the earliest Vortex version that is guaranteed to read the file.

## What an edition contains

Every member of an edition is recorded with a **component kind**: `array`, `layout`, or `aggregate`. The kind is
recorded because ids are unique only within a kind — a layout named `vortex.flat` and an array encoding named
`vortex.flat` are different members — and because membership is resolved one kind at a time. The writer holds a
separate id set per kind and enforces each where that kind is written:

| Kind | Written | Enforced at |
| --- | --- | --- |
| `array` | every serialized array | array serialization context |
| `layout` | the footer's layout tree | layout serialization context |
| `aggregate` | zone maps in zoned layouts | the layout writer context |

**Writing a component outside the enabled editions fails the write, for every kind.** A zone map is only an
optimization, so a forbidden aggregate could in principle be dropped instead — but a file that silently prunes
worse than the writer was configured for is a bug you find in a benchmark six months later, not an error you can
act on. Violations surface at write time or not at all.

Aggregates are checked against the set the write would actually record: an aggregate that a column's dtype cannot
hold is not written, so it is not a violation either.

**A kind with no declared members is unrestricted.** An edition that declares no layouts makes no promise about
layouts, so the writer leaves them alone rather than forbidding all of them; declaring the first member of a kind is
what arms its filter. `core2026.08.0` declares the aggregates the default writer records in zone maps — `min`,
`max`, `bounded_min`, `bounded_max`, `nan_count`, `null_count` — so that filter is armed by default. `sum` is not
among them: a zone sum prunes nothing, so the writer records none. File-level statistics still carry a sum, which
this filter does not govern. A
session that registers components outside `core`, such as the spatial extension types, enables its own edition
family alongside `core`, and the writer may emit the union.

## Resolving an unknown-object error

If a read failed with an unknown ID for an array encoding, layout encoding, aggregate function, or
extension dtype and pointed you here, the reader encountered a serialized format it does not
support. Find the ID in the [registry](#edition-registry) below:

1. **The ID is listed under an edition.** The file is newer than your Vortex build. Upgrade to at
   least that edition's required Vortex release.
2. **The ID is not listed anywhere.** The file was written outside the editions system with a
   custom, third-party, or experimental format. Ask the producer how to read it, or register its
   implementation with your session. Tools that only inspect or relocate data can opt in to
   `allow_unknown`, which decodes unrecognised encodings into inert placeholders and disables
   pruning for zone maps whose aggregate functions it cannot resolve.

## Writing with an edition

By default, the writer targets a `core` edition that lags the latest Vortex release by a few
versions, leaving an adoption period before it writes new formats. Every file written under this
policy carries the read-forever guarantee. If serialization would emit a format ID outside the
target editions, the write fails immediately; edition violations never surface as someone else's
read error later.

The enabled editions are stored on the writer's Vortex session. Registering an edition makes its
declaration available to the session; enabling it separately allows the writer to emit its
members. Enabling another edition from the same family replaces the earlier selection.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older Vortex.
- **Opt in to additional edition families.** Families are independently versioned and additive:
  `core` exists today, and specialised families, for example for spatial formats, may be added
  later. A writer targets at most one edition per family and may emit any format in their union;
  each format belongs to exactly one family.

Lower-level sessions without an enabled-editions store opt out of editions entirely and can write
custom or experimental formats. A raw `with_allow_encodings` array-writer policy is another
explicit opt-out. Either choice gives up the standardization guarantee: only readers that know
those formats can read the files.

## How editions change

A published edition is frozen: neither its member list nor the meaning of any member ID may change.
New formats are staged in a **draft** edition and become guaranteed only when that draft is frozen
as the next edition; each registry entry records the edition it joined in. A format may later be
*deprecated*, meaning writers stop emitting it, but readers keep decoding it indefinitely.
Deprecation therefore never invalidates existing files.

## How serialized formats evolve

In-memory representations are unversioned implementation details. They may gain capabilities or be
replaced without changing an edition. Serialization plugin registries map between the two worlds:
on read, the plugin selected by a serialized ID constructs whichever in-memory representation the
reader prefers, and on write, the implementation chooses a serialized format that can represent
the value and is permitted by the target editions. Most in-memory representations have one format
and use the same ID in memory and on disk, but that is not required: several serialized IDs may
map to one in-memory representation. The edition check applies to the ID written into the file, so
it constrains exactly what the target reader will encounter.

### Compatible evolution keeps the ID

A serialized format may evolve under its existing ID only when the change is both **backward and
forward compatible**: an old reader must still interpret data from a new writer correctly, and a
new reader must still interpret data from an old writer correctly. Adding an optional field is
compatible only when old readers can safely ignore it and new readers have the correct default when
it is absent.

Compatible evolution may add accepted input, but it cannot change the meaning of bytes that
existing readers already accept. Removing or repurposing a field, redefining existing bytes, or
requiring information that old writers never emitted is incompatible.

### Incompatible evolution creates a new format

An incompatible revision is a **new serialized format** with a new ID, registry entry, and edition
membership. The old format remains in the registry and readable forever. The in-memory
representation does not need to fork: it can read and write both formats, choosing between them
according to the value and the target editions.

Name successive incompatible revisions as a version chain on the same base name:
`vortex.foo`, `vortex.foo_v2`, `vortex.foo_v3`. Do not use descriptively named successor variants.
This gives each format at most one successor, so its serialized history is a list rather than a
tree of competing revisions.

#### Example: multi-part decimals

`vortex.decimal_byte_parts` froze into `core2025.05.0` representing each decimal value as a single
signed integer child. Its metadata includes a `lower_part_count` field, but readers of the frozen
format require it to be zero. Suppose its in-memory representation gains support for wide decimals
as a signed most-significant part plus unsigned 64-bit lower parts:

- A single-part array still serializes as `vortex.decimal_byte_parts` with
  `lower_part_count = 0`, indistinguishable from files written before the change.
- An array with lower parts serializes as `vortex.decimal_byte_parts_v2`, staged in a draft
  edition.
- A new reader deserializes both IDs into the same in-memory representation. A reader that predates
  the second format reports an unknown-ID error for `vortex.decimal_byte_parts_v2`, rather than
  entering a decoder that was never taught about lower parts.

### Reading: deserialize into the current representation

Each serialized format that ships in a release remains readable forever. Its deserializer may
upgrade the data into the current in-memory representation instead of preserving a parallel legacy
representation. For example, a serialized `vortex.alp` array with interior patches is deserialized
as a `Patched` array wrapping a patch-free ALP array, and zone maps written before aggregate
descriptors existed, including whole `vortex.stats` layouts, deserialize into the same zone-map
machinery used by modern `vortex.zoned` layouts. There is no version negotiation at read time: the
reader resolves the serialized ID and deserializes it, or reports the relevant
[unknown-ID error](#resolving-an-unknown-object-error).

### Writing: select a permitted format

Writers choose a serialized format that can represent the current in-memory value and is permitted
by the target editions. This need not be the newest format: a value that the older frozen format
represents exactly may continue to use its older ID. If the preferred format is newer than the
target edition, the writer resolves the conflict in one of two ways:

1. **Translate.** If the value has a lossless translation to a permitted serialized format, emit
   that format. For example, a newer layout may re-emit its zone statistics using an older stats
   schema.
2. **Convert to canonical and recompress.** Otherwise, decompress the data to a canonical
   representation and recompress it with the configured compressors, filtered to the target
   editions. This is how arrays are handled today: the write pipeline normalizes each chunk,
   recursively executing an encoding outside the permitted set down to canonical, and then lets
   the edition-filtered compressor choose the final encoding.

Both paths run inside the ordinary write pipeline, so the configured compressors produce the final
bytes. If neither path can express the data within the target editions, the write fails rather than
emitting a file that the target reader cannot load.

### What this means for each kind

- **Arrays.** Enforced at write time today: the writer's array context only permits serialized
  array encodings from the enabled editions.
- **Layouts.** The layout strategy decides the layout tree at write time. Layout membership
  declares which serialized layout formats a target reader understands. Strategies must degrade
  to older structures, for example plain chunked data instead of newer auxiliary layouts, when
  targeting editions that predate them.
- **Aggregate functions.** Zone maps and file statistics serialize aggregate function IDs and
  their options. Writers targeting an edition without a function must omit it or translate to an
  older stats schema; readers handle an unknown aggregate under `allow_unknown` as described
  [above](#resolving-an-unknown-id-error), which is sound because dropping statistics only
  weakens pruning.
- **Extension dtypes.** Every serialized `DType`, including every file's schema, embeds the IDs and
  metadata of its extension dtypes. An extension dtype in durable data therefore needs the same
  guarantee as an array encoding. Readers resolve its ID against the session's dtype registry.

## Edition registry

Coming soon.
