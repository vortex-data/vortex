# Editions

:::{important}
An **edition** names the exact set of encodings a writer is permitted to put in a Vortex file.
Writing a file with edition `core2026.07.0` buys you a portability promise: every Vortex
release from that edition onward is guaranteed to read — and execute queries over — every
encoding in that edition, forever.
:::

This page is the canonical, published registry of Vortex editions. Error messages about
unrecognised encodings link here. If you were sent here by such an error, see
[Troubleshooting unknown encodings](#troubleshooting-unknown-encodings).

## The compatibility model

Editions constrain **writers**, never readers. A reader always understands the union of every
edition ever published, so reading a Vortex file requires no edition configuration at all; the
edition only surfaces when something has gone wrong.

Every edition is listed in the [registry](#edition-registry) below, and each edition has its
own page recording exactly which encodings it guarantees, together with a machine-readable
JSON definition. Once published, an edition is **frozen**: its sets never grow or shrink. New
encodings enter the promise only by cutting a new edition. (An edition may also exist as a
**draft** — the staging area for the next edition. Drafts carry no guarantee and may change
freely until frozen.)

Readers are **cumulative**: each Vortex release reads everything from every edition published
before it. Note that this is a promise about readers, not about editions themselves — an
encoding may eventually be *deprecated*, meaning writers stop emitting it and newer editions
no longer include it, but readers keep decoding it indefinitely. Deprecation is a warning,
never a read failure. (Deprecation machinery is reserved but not yet specified; no encoding is
currently deprecated.)

By default the writer targets the latest published `core` edition, and attempting to write an
encoding outside the targeted editions fails at **write** time — never at read time. You can
pin an older edition when files must remain readable by older deployments, opt in to
additional [edition families](#edition-families-and-names) when you need their encodings, or
opt out of editions entirely to write custom or experimental encodings, explicitly forfeiting
the portability promise.

### Editions and writer presets

Vortex ships more than one writer configuration — the regular compressor, the *compact*
preset (better ratios via Zstd/Pco-heavy schemes), and device-oriented presets such as the
CUDA-compatible one. These are **selection policies, not compatibility surfaces**: a preset
chooses which encodings to prefer *within* the target edition's sets, trading size, speed, and
decode locality, while the edition bounds what any preset may emit. Files written by the
regular and compact writers under the same edition are therefore equally portable, and
switching presets can never change which readers can read a file. If a preset's preferred
scheme would emit an encoding outside the target edition (for example, an experimental
encoding enabled by an unstable feature), edition enforcement excludes that scheme — the
preset degrades to its next-best choice rather than break the promise.

### Editions are not the file format version

The edition is distinct from the file format version (the `u16` version tag in the
[end-of-file marker](/specs/file-format)). The format version describes the *structure* of the
file container and footer and must match exactly; the edition describes the *contents* — which
encodings may appear inside a structurally valid file. The two evolve independently.

### Build features

The edition guarantee is stated for default-feature builds of Vortex. A build that disables
default cargo features may omit support for some edition encodings (for example, disabling the
`zstd` feature removes the `vortex.zstd` decoder). Such builds trade the edition guarantee for
a smaller binary.

## Edition families and names

Editions are named `family` + `year.month.version` — for example `core2026.07.0` — where the
family names an independently versioned group of encodings and the date records when the set
was frozen. Within a family, names order chronologically, so "my reader supports `core` up to
`core2026.07.0`" is meaningful at a glance. The trailing version distinguishes editions cut in
the same month and is normally `0`. Editions are expected to be cut rarely — a few times a
year at most, batching newly stabilized encodings — not on every release.

**Families are additive.** The `core` family covers the encodings the default writer emits;
other families extend it (for example, a future `geo` family for spatial encodings, or a
vendor family for encodings maintained outside this repository). A writer targets a *set* of
editions — at most one per family — and may emit any encoding in the union of their sets. The
read guarantee composes the same way: a reader can read a file if it supports each edition the
file was written under, and enabling an extra family for one file never affects the
portability of files written without it.

Every encoding belongs to exactly one family, so combining families never produces
conflicting definitions of the same encoding.

## Edition registry

Each edition links to its own page, which lists every encoding it guarantees and links a
machine-readable JSON definition of the same content for use by tools.

<!-- editions:index:begin -->
<!-- Generated by `cargo xtask generate-editions`. Do not edit by hand. -->

| Edition | Status | Frozen on | Encodings |
|---|---|---|---|
| [core2026.07.0](editions/core2026.07.0.md) | current | 2026-07 | 32 |
| [core2026.10.0](editions/core2026.10.0.md) | draft | — | 32 |

```{toctree}
---
maxdepth: 1
hidden: true
---

editions/core2026.07.0
editions/core2026.10.0
```
<!-- editions:index:end -->

## Verification

The edition promise is tested, not just declared: real `.vortex` files written by previous
releases are continuously re-read and checked against known-good values, and published
editions are pinned by tests so no code change can silently alter a frozen set. If you are
curious how editions are defined and enforced — or you maintain an encoding and want to add
it to an edition — see the [implementation notes](/developer-guide/internals/editions).

## Troubleshooting unknown encodings

A reader that encounters an encoding ID it does not recognise reports an error pointing at this
page. There are two possibilities:

1. **The file targets a newer edition than your Vortex build knows.** Check the registry above
   for the edition that introduced the encoding, and upgrade to at least that edition's
   required Vortex release.
2. **The file was written outside any edition** — with a custom, third-party, or experimental
   encoding. Ask the producer of the file, or register the encoding with your session before
   reading. Tools that only need to inspect or relocate data (rather than query it) can opt in
   to `allow_unknown`, which decodes unrecognised encodings into inert placeholder arrays.
