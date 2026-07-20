# Editions

An **edition** is a named grouping of encodings: it records exactly which encodings a Vortex
file may contain, and when each encoding joined the set. The current edition,
[`core2026.07.0`](#core2026070), groups the 32 encodings the default file writer emits today.

Editions make compatibility concrete. Because every edition is published and frozen, "written
under `core2026.07.0`" says precisely which encodings a reader must support — and every Vortex
release supports every edition published before it, so files written under an edition remain
readable, and queryable, by all future versions of Vortex.

Editions constrain writing, not reading: a reader needs no configuration and always
understands every published edition. The only time an edition surfaces on the read path is in
the error described next.

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

By default the writer targets the latest `core` edition — there is nothing to configure, and
every file you write carries the read-forever guarantee. If a file would contain an encoding
outside the targeted edition, the write fails immediately; edition violations never surface as
someone else's read error later.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older Vortex.
- **Opt in to additional edition families.** Editions come in independently versioned,
  additive families — `core` today, with families for more specialised encoding groups (for
  example spatial encodings) possible later. A writer targets at most one edition per family
  and may emit any encoding in their union; each encoding belongs to exactly one family.

You can also opt out of editions entirely to write custom or experimental encodings. Doing so
is an explicit choice that gives up the portability guarantee — only readers that know your
encodings can read those files.

Compression presets (the default writer, the compact preset, CUDA-oriented presets) choose
*within* the targeted edition, so switching presets never changes which readers can read a
file.

## How editions change

A published edition is frozen — its encoding list never grows or shrinks. New encodings are
staged in a **draft** edition and become guaranteed only when that draft is frozen as the next
edition; each encoding's registry entry records the edition it joined in. In the future an
encoding may be *deprecated*, meaning writers stop emitting it — but readers keep decoding it
indefinitely, so deprecation never invalidates existing files.

Editions are also distinct from the file format version (the `u16` tag in the file's
end-of-file marker): the format version describes the structure of the file container, the
edition describes which encodings may appear inside it.

## Edition registry

| Edition | Status | Frozen on | Encodings |
|---|---|---|---|
| [core2026.07.0](#core2026070) | current | 2026-07 | 32 |
| [core2026.10.0](#core2026100) | draft | — | 32 |

The registry is verified continuously: real `.vortex` files written by previous releases are
re-read and checked against known-good values, and published editions are pinned by tests so
no code change can silently alter a frozen set. The guarantee assumes a default-feature build
of Vortex; disabling default cargo features (e.g. `zstd`) removes the corresponding decoders.

### core2026.07.0

| | |
|---|---|
| **Status** | current |
| **Frozen on** | 2026-07 |
| **Required Vortex release** | the first release that includes this edition |
| **Encodings** | 32 |

| Encoding ID | Since |
|---|---|
| `fastlanes.bitpacked` | core2026.07.0 |
| `fastlanes.delta` | core2026.07.0 |
| `fastlanes.for` | core2026.07.0 |
| `fastlanes.rle` | core2026.07.0 |
| `vortex.alp` | core2026.07.0 |
| `vortex.alprd` | core2026.07.0 |
| `vortex.bool` | core2026.07.0 |
| `vortex.bytebool` | core2026.07.0 |
| `vortex.chunked` | core2026.07.0 |
| `vortex.constant` | core2026.07.0 |
| `vortex.datetimeparts` | core2026.07.0 |
| `vortex.decimal` | core2026.07.0 |
| `vortex.decimal_byte_parts` | core2026.07.0 |
| `vortex.dict` | core2026.07.0 |
| `vortex.ext` | core2026.07.0 |
| `vortex.fixed_size_list` | core2026.07.0 |
| `vortex.fsst` | core2026.07.0 |
| `vortex.list` | core2026.07.0 |
| `vortex.listview` | core2026.07.0 |
| `vortex.masked` | core2026.07.0 |
| `vortex.null` | core2026.07.0 |
| `vortex.pco` | core2026.07.0 |
| `vortex.primitive` | core2026.07.0 |
| `vortex.runend` | core2026.07.0 |
| `vortex.sequence` | core2026.07.0 |
| `vortex.sparse` | core2026.07.0 |
| `vortex.struct` | core2026.07.0 |
| `vortex.varbin` | core2026.07.0 |
| `vortex.varbinview` | core2026.07.0 |
| `vortex.variant` | core2026.07.0 |
| `vortex.zigzag` | core2026.07.0 |
| `vortex.zstd` | core2026.07.0 |

Not included, because they are experimental and their serialized forms may still change:
`vortex.onpair`, `vortex.zstd_buffers`, `vortex.patched`, `vortex.piecewise-sequence`, and
everything behind the `unstable_encodings` feature flag. Writing them requires opting out of
the edition.

### core2026.10.0

The draft staging area for the next `core` edition: no guarantee until frozen, contents may
change freely. No changes relative to [core2026.07.0](#core2026070) yet.
