# Editions

Vortex defines an evergrowing set of serializable array encodings, once written this can be read back by any future
version of vortex.
**Editions** are used to keep track of these encodings and talk about groups of encodings.

The first edition, `core2025.05.0`, contains the stable encodings that could be written by Vortex
`0.36.0`. This is the release from which the Vortex file format is considered stable. Later `core`
editions add stable encodings released after that compatibility boundary.
Editions are additive so an edition that comes after a previous one contains all the encodings from the previous one
and more.
The writer can be configured with a set of different editions (for example, `core2026.07.0` and
`unstable2026.06.0` select stable encodings released through July 2026 and unstable encodings
released through June 2026).

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

## Resolving an unknown-encoding error

If a read failed with an unknown encoding ID and pointed you here, the reader met an array encoding
it does not support. Find the encoding ID in the [registry](#edition-registry) below:

1. **The ID is listed under an edition.** The file is newer than your Vortex build. Upgrade to
   at least that edition's required Vortex release and the file will read.
2. **The ID is not listed anywhere.** The file was written outside the editions system, with a
   custom, third-party, or experimental encoding. Ask the producer of the file how to read it,
   or register the encoding with your session before reading. Tools that only inspect or
   relocate data (rather than query it) can opt in to `allow_unknown`, which decodes
   unrecognised encodings into inert placeholders.

## Writing with an edition

The default the writer targets a `core` edition lagging the latest vortex release by a few version giving delay before
writing the latest vortex encodings to disk.
Every file you write carries the read-forever guarantee. If a file would contain an encoding
outside the targeted edition, the write fails immediately; edition violations never surface as
someone else's read error later.

The enabled editions are stored on the writer's Vortex session. Registering an edition makes
its declaration available to the session; enabling it separately allows the writer to emit its
array encodings. Enabling another edition from the same family replaces the earlier selection.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older Vortex.
- **Opt in to additional edition families.** Editions come in independently versioned,
  additive families — `core` today, with families for more specialised encoding groups (for
  example spatial encodings) possible later. A writer targets at most one edition per family
  and may emit any encoding in their union; each member belongs to exactly one family.

Lower-level sessions without an enabled-editions store opt out of editions entirely and can write
custom or experimental encodings. A raw `with_allow_encodings` writer policy is another explicit
opt-out. Either choice gives up the standardization guarantee — only readers that know those
encodings can read the files.

## How editions change

A published edition is frozen — its member list never grows or shrinks. New members are
staged in a **draft** edition and become guaranteed only when that draft is frozen as the next
edition; each member's registry entry records its component kind and the edition it joined in.
Declaring the first member of a kind arms that kind's write-time filter, so a kind gains
enforcement at the edition that first declares one.
In the future an encoding may be *deprecated*, meaning writers stop emitting it — but readers
keep decoding it indefinitely, so deprecation never invalidates existing files.

## Edition registry

Coming soon.. It will list each edition's members with their component kind, the edition they
joined in, and the Vortex release required to read them.
