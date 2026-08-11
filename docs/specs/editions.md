# Editions

Vortex defines an evergrowing set of serializable array encodings, once written this can be read back by any future
version of vortex. **Editions** are used to keep track of these encodings and talk about groups of encodings.

The first edition, `core2025.05.0`, contains the stable encodings that could be written by Vortex
`0.36.0`. This is the release from which the Vortex file format is considered stable. Later `core`
editions add stable encodings released after that compatibility boundary. Editions are additive so an edition that comes
after a previous one contains all the encodings from the previous one and more. The writer can be configured with a set
of different editions (for example, `core2026.07.0` and
`unstable2026.06.0` select stable encodings released through July 2026 and unstable encodings released through June
2026).

Editions can be used to constrain your minimum required vortex reader, since latest version over vortex across all
editions is the earliest version of vortex required to read that file.

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

If a read failed with an unknown ID (an encoding, layout, aggregate function, or extension dtype) and pointed you here,
the reader met an object it does not support:

1. **The ID is listed under an edition.** The file is newer than your Vortex build. Upgrade to at least that edition's
   required Vortex release and the file will read.
2. **The ID is not listed anywhere.** The file was written outside the editions system, with a custom, third-party, or
   experimental object. Ask the producer of the file how to read it, or register the object with your session before
   reading. Tools that only inspect or relocate data (rather than query it) can opt in to `allow_unknown`, which decodes
   unrecognised encodings into inert placeholders and disables pruning for zone maps whose aggregates it cannot resolve.

## Writing with an edition

The default the writer targets a `core` edition lagging the latest vortex release by a few version giving delay before
writing the latest vortex encodings to disk. Every file you write carries the read-forever guarantee. If a file would
contain an encoding outside the targeted edition, the write fails immediately; edition violations never surface as
someone else's read error later.

The enabled editions are stored on the writer's Vortex session. Registering an edition makes its declaration available
to the session; enabling it separately allows the writer to emit its encodings. Enabling another edition from the same
family replaces the earlier selection.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older Vortex.
- **Opt in to additional edition families.** Editions come in independently versioned, additive families — `core` today,
  with families for more specialised encoding groups (for example spatial encodings) possible later. A writer targets at
  most one edition per family and may emit any encoding in their union; each encoding belongs to exactly one family.

Lower-level sessions without an enabled-editions store opt out of editions entirely and can write custom or experimental
encodings. A raw `with_allow_encodings` writer policy is another explicit opt-out. Either choice gives up the
standardization guarantee — only readers that know those encodings can read the files.

## How editions change

A published edition is frozen — its member list never grows or shrinks, for any object kind. New objects are staged in a
**draft** edition and become guaranteed only when that draft is frozen as the next edition; each object's registry entry
records the edition it joined in. In the future an object may be *deprecated*, meaning writers stop emitting it — but
readers keep decoding it indefinitely, so deprecation never invalidates existing files.

## How serialized objects evolve

Editions name *which* objects a file may contain. This section defines how the serialized form of those objects — array
metadata, layout metadata, aggregate options, extension dtype metadata — is allowed to change over time.

An object may evolve in place, keeping its id and its registry entry, as long as every change is both **backward and
forward compatible**: a new reader must still understand data written by an old writer, and an old reader must still
understand data written by a new writer. Adding an optional field that older readers ignore and newer readers default
when absent is the canonical example.

A change that breaks either direction is not an evolution of the object — it is a **new object**. Removing or
repurposing a field, redefining what existing bytes mean, or requiring information older writers never emitted all fall
in this bucket, and each demands a new id, a new registry entry, and its own edition membership. The old object stays in
the registry and stays readable forever; writers stop emitting it only if it is deprecated. This is the rule that makes
the reading and writing halves below tractable: readers only ever accumulate compatible forms of an object, and writers
only ever have to choose between distinct objects the target edition does or does not guarantee.

### Reading: deserialize to the latest version

Vortex maintains exactly one in-memory representation per object: the latest one. Deserialization always targets it,
from **every serialized form that has ever existed**:

- A serialized form, once shipped in a release, stays readable **forever**. Deserializers accumulate historical forms;
  they never drop one.
- Old forms deserialize *into the latest in-memory version*, not into parallel legacy code paths. For example, zone maps
  written before aggregate descriptors existed (and whole
  `vortex.stats` layouts) deserialize into the same zone-map machinery that modern
  `vortex.zoned` layouts use; the reader upgrades on the way in.
- Consequently there is no version negotiation at read time: a reader either knows the object (and then reads all of its
  historical forms) or reports an unknown-object error covered [above](#resolving-an-unknown-object-error).

A compatible change therefore lands as *additional* accepted input: the new form joins the deserializer alongside every
earlier one, and its new fields are optional, so data written before they existed still deserializes. A change that
cannot be expressed that way is a new object, not a new form of this one.

### Writing: translate down, or convert to canonical and recompress

Writers emit the **newest serialized form permitted by the target editions**. An object (or object version) newer than
the target edition never leaks into the file; the writer resolves the conflict in one of two ways:

1. **Translate.** If the newer in-memory version has a defined translation to a serialized form the target edition
   guarantees, the writer emits the older form. This is preferred when the translation is lossless, e.g. re-emitting a
   newer layout's zone statistics using an older stats schema.
2. **Convert to canonical and recompress.** Otherwise the writer decompresses the data to a canonical representation and
   recompresses it using only the configured compressors, filtered to the target editions. This is how arrays are
   handled today: the write pipeline normalizes each chunk (recursively executing any encoding outside the permitted set
   down to canonical) and the configured compressor — default, BtrBlocks-style, or custom — is restricted to choose
   encodings from the enabled editions.

Both paths run inside the ordinary write pipeline, so the configured compressors are always what produces the final
bytes. If neither path can express the data inside the target editions, the write fails immediately rather than emitting
a file the target reader could not load.

### What this means per kind

- **Arrays.** Enforced at write time today. The writer's array context only permits encodings from the enabled editions;
  anything else is normalized to canonical and recompressed by the edition-filtered compressor.
- **Layouts.** The layout strategy decides the layout tree at write time. Layout membership declares which layout
  encodings (in their current serialized form) a target reader understands; strategies must degrade to older structures
  (e.g. plain chunked data instead of newer auxiliary layouts) when targeting editions that predate them.
- **Aggregations.** Zone maps and file statistics serialize aggregate function ids plus their options. Writers targeting
  an edition without a given aggregate must omit it or translate to an older stats schema; readers meeting an unknown
  aggregate disable pruning for that zone map (under `allow_unknown`) rather than failing the scan, since dropping
  statistics is always sound.
- **Extension dtypes.** Every serialized `DType` — including every file's schema — embeds the ids and metadata of the
  extension dtypes it uses, so an extension dtype in durable data needs the same guarantee as an encoding. Readers
  resolve ids against the session's dtype registry.

## Edition registry

Coming soon.. It will list each edition's members with their component kind, the edition they
joined in, and the Vortex release required to read them.
