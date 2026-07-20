# Editions

How Vortex [editions](/specs/editions) are defined, computed, and published. The public spec
describes what an edition *promises*; this page describes how the machinery works and what to
touch when stabilizing an encoding or cutting an edition.

## Single source of truth

Everything derives from two static data structures in the dependency-free `vortex-edition`
crate:

1. **The edition list** — `EDITIONS`: each edition's identifier
   (`EditionId { family, year, month, version }`) and a `draft` flag. Nothing else is
   stored: the freeze date is carried by the identifier itself (`core2026.07.0` was frozen
   in 2026-07), and everything further — status, lineage, the encoding sets, the required
   Vortex release — is derived.

2. **Per-encoding membership declarations** — `ENCODINGS`: each encoding declares the edition
   it has been a member of *since*:

   ```rust
   EncodingDecl {
       id: "vortex.fsst",
       since: CORE_2026_07_0,
       required_vortex_release: None,
   }
   ```

   An encoding with `since: E` is a member of `E` and every later edition of the same family.
   The membership edge also carries `required_vortex_release` — the earliest release able to
   read and execute this encoding, recorded from evidence such as compat-fixture history
   (`None` until recorded). Encoding IDs are globally unique across everything an edition
   can cover: when layout encodings join editions, their IDs must be distinct from array
   encoding IDs.
   The declarations currently live centrally in `vortex-edition/src/definitions.rs`; they are
   expected to migrate next to each encoding's vtable once the array plugin trait grows
   edition metadata.

Encoding IDs are deliberately **plain strings**, not references to encoding vtables, so the
crate depends on nothing and can be consumed by the writer, `xtask`, the compat-test suite,
and language bindings alike. An integration test in the file crate closes the loop by
asserting that every declared ID resolves to a registered, executable encoding in the default
session.

## Computation and validation

`vortex_edition::manifest::compute_manifests()` derives an `EditionManifest` per edition: the
resolved, sorted encoding set, plus derived metadata — `status` (`draft` if flagged;
otherwise the newest frozen edition per family is `current`, the rest `superseded`),
`supersedes` (the previous edition in the family), the freeze year-month (from the
identifier), and the required Vortex release. The delta against the superseded edition is
itself derived: the members whose `since` equals the edition's own id.

Computation fails loudly (it never produces silently odd output) on: duplicate encoding or
edition IDs, membership (`since`) references to unknown editions, editions out of
chronological order within a family (drafts must be newest), and malformed release strings
on membership edges.

Families partition the encoding space — every encoding belongs to exactly one family — so
the union of targeted editions is always unambiguous.

## Required Vortex release

An edition's required Vortex release — the earliest release guaranteed to read and execute
the full edition — is **never declared on the edition; it is inferred**, in three layers:

1. **The operational check needs no version number at all.** A binary supports edition `E`
   if and only if its own embedded edition list contains `E` marked frozen. Containment, not
   version comparison, is what the writer and any diagnostics actually test.

2. **When every membership edge records a per-encoding release**, the edition's requirement
   is derived as the **maximum over its members'** `required_vortex_release` values — the
   oldest release that can read every member. Per-edge values must themselves come from
   evidence (e.g. the compat-fixture history proving the encoding readable at that release),
   never from memory.

3. **While any edge is unrecorded**, the fallback is inferred from release history:

   > `required_vortex_release(E)` = the first release tag whose tree contains `E` as frozen.

   Mechanically: `git tag --contains <freeze-commit>`, filtered to release tags, minimum by
   version. Release tooling computes this once, after the release exists, and records it
   into the published JSON/docs artifacts — recorded inference, never a hand-authored claim.

Why is the fallback the freeze release, and why must earlier claims come from per-edge
evidence? Most of `core2026.07.0`'s encodings have been readable for many releases, so a
smaller number is genuinely possible — but old binaries are immutable, and CI going forward
can only ever prove "the current reader reads old files", never "an old reader handles this
newly named set". So an earlier requirement is only honest when each member's edge records a
release at which that encoding was demonstrably readable (the compat-fixture store is
exactly such a demonstration). Absent that evidence, the freeze release is the only provable
answer: the session cross-check test (every declared ID resolves to a registered, executable
encoding) runs in the same tree that freezes the edition, so any release containing the
frozen edition supports all of it by construction.

This is why the field is not stored on `Edition`: the membership edges (and, as a fallback,
the release history) carry all the information needed.

## The generation pipeline

```
vortex-edition statics ──compute_manifests()──▶ EditionManifest
        │                                            │
        │                                 serde_json │
        ▼                                            ▼
cargo xtask generate-editions ──▶ docs/specs/editions/<id>.json
                                              │
                                  re-parsed   │  (pages render from the JSON,
                                  from JSON   ▼   so page and JSON cannot disagree)
                                  docs/specs/editions/<id>.md
                                              +
                                  index block in docs/specs/editions.md
                                  (between the editions:index markers)
```

Run `cargo xtask generate-editions` after any change to the definitions. The command writes
one JSON file and one Markdown page per edition (drafts included, rendered with a warning
banner), and rewrites the edition index table and hidden toctree in `docs/specs/editions.md`
between the `<!-- editions:index:begin -->` / `<!-- editions:index:end -->` markers. All
generated files carry a do-not-edit header; hand edits will be overwritten. CI should check
the committed artifacts are up to date by re-running the generator and diffing.

The JSON files are the machine-readable contract for non-Rust consumers (Python/Java bindings,
external tools): same content as the pages, stable schema (`EditionManifest`).

## Freeze tests

Published editions are pinned by golden tests in `vortex-edition/src/tests.rs`: the exact
computed encoding set of each frozen edition is written out as a constant
(`FROZEN_CORE_2026_07_0`). Any change that alters a published edition's computed set —
editing a `since`, deleting a declaration, a refactor with surprising effect — fails CI. The
metadata is the source; the snapshot is the freeze.

## How to: stabilize an encoding

1. Ensure the encoding's serialized form is final and it has backward-compat fixtures
   (`vortex-test/compat-gen`).
2. Add its `EncodingDecl` to `ENCODINGS` with `since` set to the **current draft** edition
   (never a frozen one — the freeze test will catch you).
3. Run `cargo xtask generate-editions` and commit the regenerated artifacts. The encoding now
   appears on the draft edition's page, clearly marked as carrying no guarantee yet.

## How to: cut an edition

1. On the draft edition: set `draft: false`. (There is no date to record — the freeze date
   is the edition identifier itself, so the identifier must match the month of the freeze.)
2. Add the next draft edition to `EDITIONS` (e.g. the following quarter's identifier).
3. Add a `FROZEN_<ID>` golden test pinning the newly frozen set.
4. Run `cargo xtask generate-editions`; commit. Once the release shipping the edition is
   published, release tooling infers and records the required Vortex release (see
   [above](#required-vortex-release)), and compat fixtures are published for the newly added
   encodings.

## Not yet wired up

- **Writer enforcement**: deriving the file writer's allow-list (`ALLOWED_ENCODINGS` in
  `vortex-file/src/strategy.rs`) from `Edition::current()` instead of by hand, and a
  `with_edition(EditionId)` API on the write options. Compressor schemes will need to declare
  which array encodings they emit so the scheme pool — including cascades — can be filtered
  against the target edition automatically. (`register_default_encodings` in
  `vortex-file/src/lib.rs` is the long-standing seam for this.)
- **The session cross-check test** asserting every declared encoding ID resolves in the
  default session registries.
- **Release tooling** that infers `required_vortex_release` from release tags (first release
  containing the frozen edition) and records it into the published artifacts.
- **Reader error messages** linking to the published registry page.
- **Layout declarations**: layouts join editions as ordinary encodings (with IDs distinct
  from array encoding IDs), and deprecation metadata (deprecated-for-write in a new edition,
  with read-time warnings once tooling support lands).
- **Child-encoding closure validation** — checking that an edition guaranteeing an encoding
  also guarantees the children it can emit — was dropped from the minimal model; it may
  return once declarations migrate to the vtables, where children are authoritatively known.
- **Footer stamping**: recording the writer's target editions in the file footer for better
  diagnostics. The guarantee always derives from the encoding IDs actually present in the
  file, so the stamp would be informative only.
