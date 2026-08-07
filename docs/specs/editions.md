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

## Resolving an unknown-encoding error

If a read failed with an unknown encoding ID and pointed you here, the reader met an encoding
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
encodings. Enabling another edition from the same family replaces the earlier selection.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older Vortex.
- **Opt in to additional edition families.** Editions come in independently versioned,
  additive families — `core` today, with families for more specialised encoding groups (for
  example spatial encodings) possible later. A writer targets at most one edition per family
  and may emit any encoding in their union; each encoding belongs to exactly one family.

Lower-level sessions without an enabled-editions store opt out of editions entirely and can write
custom or experimental encodings. A raw `with_allow_encodings` writer policy is another explicit
opt-out. Either choice gives up the standardization guarantee — only readers that know those
encodings can read the files.

## How editions change

A published edition is frozen — its encoding list never grows or shrinks. New encodings are
staged in a **draft** edition and become guaranteed only when that draft is frozen as the next
edition; each encoding's registry entry records the edition it joined in. In the future an
encoding may be *deprecated*, meaning writers stop emitting it — but readers keep decoding it
indefinitely, so deprecation never invalidates existing files.

Freezing an edition also freezes the *meaning* of each member: what a reader may assume about
bytes written under that ID never changes afterwards. An incompatible extension of a member is
a new serialized format with a new ID, not a mutation of the frozen one — see
[Extending a frozen format](#extending-a-frozen-format).

## Editions name serialized formats

An edition member is a **serialized format**: the encoding ID written into the file, together
with its metadata schema and the meaning of its buffers and children. The read-forever
guarantee attaches to those bytes. The in-memory encoding that produced them is an
implementation detail — it may evolve, gain capabilities, or be replaced entirely without
touching any edition.

The serialization plugin registry is the layer that maps between the two, in both directions:

- **On read**, a serialized format may be deserialized into whichever in-memory encoding the
  reader prefers. For example, a serialized `vortex.alp` array with interior patches is
  deserialized as a `Patched` array wrapping a patch-free ALP array: the file bytes are frozen,
  but the in-memory representation improved underneath them.
- **On write**, the plugin chooses which serialized format to emit for a given array. Most
  encodings have exactly one, so the serialized ID and the in-memory ID coincide — but nothing
  requires this, and one in-memory encoding may own several serialized formats.

Because the write-time edition check applies to the ID that is written into the file, an
edition constrains exactly what it guarantees: the bytes a reader will meet.

### Extending a frozen format

A frozen format never changes meaning. If an in-memory encoding learns to represent something
its serialized format cannot carry — even through a metadata field that was always present but
constrained to one value — writing it under the old ID would break every reader the edition
promised could read it. Such an extension is a **new serialized format with a new ID**, staged
in a draft edition like any other new member. The in-memory encoding does not fork; it gains a
second serialized format:

- Arrays the old format can represent keep serializing under the old ID, byte-identical,
  readable by every reader since the old edition froze.
- Arrays only the new format can represent serialize under the new ID, writable only once an
  edition containing it is enabled.
- Readers register one plugin for both IDs; both deserialize into the same in-memory encoding.

### Example: multi-part decimals

`vortex.decimal_byte_parts` froze into `core2025.05.0` representing each decimal value as a
single signed integer child. Its metadata reserves a `lower_part_count` field, but every
reader of the frozen format requires it to be zero — so that field cannot be used to extend
the format after the fact. Suppose the in-memory encoding grows support for wide decimals as
a signed most-significant part plus up to three unsigned 64-bit lower parts. Then:

- A single-part array still serializes as `vortex.decimal_byte_parts` with
  `lower_part_count = 0`, indistinguishable from files written before the change.
- An array carrying lower parts serializes as `vortex.decimal_byte_parts_wide`, a new format
  staged in a draft edition. A writer pinned to an edition without it cannot emit it; the
  compressor consults the enabled editions and only produces multi-part arrays when the new
  format is allowed, so the write-time check never fires as a surprise.
- A reader that supports the new format deserializes both IDs into the same in-memory
  encoding. A reader that predates it fails on `vortex.decimal_byte_parts_wide` with an
  unknown-encoding error pointing at the registry — the failure mode editions promise —
  rather than crashing inside a decoder that was never taught about lower parts.

## Edition registry

Coming soon..
