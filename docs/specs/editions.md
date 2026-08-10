# Editions

Vortex defines an evergrowing set of serializable array encodings, once written this can be read back by any future
version of vortex.
**Editions** are used to keep track of these encodings and talk about groups of encodings.

Every member of an edition is recorded with a **component kind**, and today the only kind an edition declares is
`array`: an edition names the array encodings that may appear in a written array. Ids are unique per kind, not
globally, so a future layout named `vortex.flat` and an array encoding named `vortex.flat` are different members.
Resolution is always per kind — the writer asks for the enabled *array encoding* ids — which is what lets other
kinds (layouts, scalar functions, aggregate functions) join editions later without colliding with array encoding
ids or silently widening what a writer may emit.

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

A published edition is frozen — its member list never grows or shrinks. New encodings are
staged in a **draft** edition and become guaranteed only when that draft is frozen as the next
edition; each member's registry entry records the edition it joined in and its component kind.
In the future an encoding may be *deprecated*, meaning writers stop emitting it — but readers
keep decoding it indefinitely, so deprecation never invalidates existing files.

## Edition registry

Coming soon..
