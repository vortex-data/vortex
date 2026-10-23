# Decimal and the type roadmap

`Decimal` is a built-in dtype with the shape of an extension type: an integer storage value plus a
scale. On `develop`, the decimal array does not keep to that shape. One decimal dtype can have six
physical forms, and `Decimal` duplicates much of `Primitive`. These notes describe the target
design that the maintainer settled on 2026-10-19, 2026-10-22 and 2026-10-23, and they put the type
work that it needs in order. [`decisions.md`](../decisions.md) records the decisions and the open
questions. Issue [#10701](https://github.com/vortex-data/vortex/issues/10701) tracks the fix of the
canonical decimal encoding.

| Part | Content |
| --- | --- |
| [1. Current state](01-current-state.md) | How `Decimal` works on `develop`, and what is wrong |
| [2. Design](02-design.md) | The leaf, imports and Arrow, `Narrow`, DecimalByteParts, the file format, and the future work |
| [3. Roadmap](03-roadmap.md) | The dependency graph of the type work, the independent changes, and the order of work |
| [4. Implementation plan](04-implementation.md) | The PRs of the `Narrow` and canonical decimal work, their dependencies and waves, with one brief for each PR |

## Summary

- The width of the decimal buffer does not depend on the precision. So one dtype has up to six
  physical forms, every kernel must handle mixed widths, and some paths panic.
- Decided: a decimal array stays a leaf with one buffer and validity. Each decimal dtype has
  exactly one canonical encoding, because the precision fixes the width of the buffer. Each
  precision gets the smallest signed width that holds it.
- Decided: `DecimalArray::try_new` checks the width and every valid value. Imports copy into the
  canonical width, and they fail on a value outside the precision. File reads trust the precision.
- Decided: a `Narrow` encoding stores a column as a child with a lower precision and the same
  scale. The child must have a narrower canonical width. One `Narrow` encoding serves integers and
  decimals, and its kernels pass work to the child. An integer child keeps the signedness of its
  parent. The wire form is only for columns with a fixed logical dtype, and internal integer
  children keep changing their own dtype.
- Precision limits are powers of ten, and widths are powers of two. So a width does not give a
  precision: an `i32` holds 2,000,000,000, which needs precision 10. Each producer finds the child
  precision from the values.
- Decided: a reader finds an old narrow `vortex.decimal` node by its stored width. It scans the
  values for their max, and it returns a `Narrow` or a copy. Every narrowing copy writes zeros in
  null slots and fails on a value outside the precision.
- Decided: DecimalByteParts stays the compressed form. Operations run on its parts where they can,
  and it decodes to the canonical array of its own dtype. One compressor scheme builds it from the
  canonical array, so file contents stay the same.
- Decided: `vortex.narrow` stays in memory until the maintainer decides to write it. Then it
  enters one preview edition together with DecimalByteParts with lower parts. Only then does a
  decimal `Narrow` scheme narrow the decimal type first. Default writers use `core`, so until both
  formats enter `core`, they write wide decimals with values above 64 bits as plain `i256` arrays.
- Decided: the bar of no compute regression applies to the final state. Until the compressor
  narrows the decimal type first, decoding a compressed decimal runs at the full width in memory.
- Decided: the work lands as 22 PRs in four tracks. Each PR has a brief that a coding agent can
  implement alone. Generic structural encodings always sit above `Narrow`. A writer that cannot
  write `vortex.narrow` writes the canonical array, as a documented workaround. Which arrays the
  compressor narrows is deferred.
- Arrow has the same decimal model and the same gap at the top of each width. Its width is part
  of the type, and from precision 5 the Vortex widths match the smallest Arrow types.
- Decided: `vortex.decimal` keeps its wire ID, and the decimal file format does not change.
- Decided: `Decimal` stays a built-in type that behaves like a hardcoded extension type. A move to
  an extension type is an optional later step.
- Decided: Vortex adds `I128`, `I256`, `U128` and `U256`, and FixedSizeBinary becomes a core type.
  Their designs are future work, outside the canonical decimal fix. So are the integer view,
  decimal arithmetic across precisions, and casts that lose information. The recommendations for
  this future work are not reviewed.

## Agent documents

- [`agents/briefs/`](agents/briefs/): one brief for each PR of part 4, in four files.
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
2026-10-19 the review also read the draft `Narrow` PRs and PR #10230. On 2026-10-22 it read the
versioning spec, the array plugin and serialization code, and the DecimalByteParts code. It also
read the history of the decimal compression scheme, and PRs #10305 and #10314. On 2026-10-23 it
traced which statistics the compressors store on the DecimalByteParts parts, read the decimal
types of arrow-rs 59.3, and read the cast change of PR #10230. A second review on the same day
read when the reader attaches stored statistics, the DecimalByteParts constructor and compressor
scheme, the compat harness, and the edition families. A planning review on the same day had five
agents draft the PR briefs from the code at `b70206818f`, and the review checked their main claims
in the code. One throwaway Rust test confirmed
the `f32` cast disagreement and the `ext_storage` pushdown rules (probes R1 and R6 in the extension
notes). One short Python script checked byte order. Section 10 of the report lists the claims that
are not verified, which include every performance claim.
