# Decimal and the type roadmap

`Decimal` is a built-in dtype with the shape of an extension type: an integer storage value plus a
scale. On `develop`, the decimal array does not keep to that shape. One decimal dtype can have six
physical forms, and `Decimal` duplicates much of `Primitive`. These notes describe the target
design that the maintainer settled on 2026-10-19, and they put the type work that it needs in
order. [`decisions.md`](../decisions.md) records the decisions and the open questions.

| Part | Content |
| --- | --- |
| [1. Current state](01-current-state.md) | How `Decimal` works on `develop`, and what is wrong |
| [2. Design](02-design.md) | The leaf, the integer view, `Narrow`, wide integers, FixedSizeBinary, and the open questions |
| [3. Roadmap](03-roadmap.md) | The dependency graph of the type work, the urgent changes, and the order of work |

## Summary

- The width of the decimal buffer does not depend on the precision. So one dtype has up to six
  physical forms, every kernel must handle mixed widths, and some paths panic.
- Decided: a decimal array stays a leaf with one buffer and validity. Each decimal dtype has
  exactly one canonical encoding, because the precision fixes the width of the buffer.
- Decided: a `Narrow` encoding stores a column as a child with a lower precision and the same
  scale. So narrow storage stays possible. These notes recommend one `Narrow` encoding for
  integers and decimals.
- Decimal kernels delegate to integer kernels through a zero-copy integer view of the buffer. Each
  decimal operation is an integer operation plus bookkeeping for the scale.
- The view needs the new primitive types `I128` and `I256` for precision 19 to 76. The leaf does
  not need them, so the fix for one canonical form can come first.
- Decided: Vortex adds `I128`, `I256`, `U128` and `U256`. None of them needs FixedSizeBinary.
- Decided: FixedSizeBinary becomes a core type for opaque bytes, such as UUIDs and hashes.
- The decimal file format does not change. Only decimal `Narrow` needs a new wire ID, behind an
  edition.
- Decided: `Decimal` stays a built-in type that behaves like a hardcoded extension type. A move to
  an extension type is an optional later step.

## Agent documents

- [`agents/report.md`](agents/report.md): the full report. It has file and line references, API
  sketches, the wire plan, the complete dependency graph with a reason for each edge, and the
  evidence log.
- [`agents/decimal-call-sites.md`](agents/decimal-call-sites.md): every decimal-specific file with
  line counts, every use of the width function, and every canonical dispatch site.
- [`agents/ptype-match-inventory.md`](agents/ptype-match-inventory.md): every `PType` dispatch
  macro use, every exhaustive `PType` match, and every copy of `PType` in other languages and
  formats.

## Evidence

The notes come from reading `develop` at `5fc5cb8`, RFC drafts, issues and pull requests. On
2026-10-19 the review also read the draft `Narrow` PRs and PR #10230. One throwaway Rust test
confirmed the `f32` cast disagreement and the `ext_storage` pushdown rules (probes R1 and R6 in the
extension notes). One short Python script checked byte order. Section 10 of the report lists the
claims that are not verified, which include every performance claim.
