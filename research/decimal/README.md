# Decimal and the type roadmap

`Decimal` is a built-in dtype with the shape of an extension type: an integer storage value plus a
scale. On `develop`, the decimal array does not keep to that shape. One decimal dtype can have six
physical forms, and `Decimal` duplicates much of `Primitive`. These notes propose an opaque
`Decimal` wrapper over an integer child, and they put the type work that it needs in order.

| Part | Content |
| --- | --- |
| [1. Current state](01-current-state.md) | How `Decimal` works on `develop`, and what is wrong |
| [2. Design](02-design.md) | The wrapper, the child type, wide integers, FixedSizeBinary, and the open questions |
| [3. Roadmap](03-roadmap.md) | The dependency graph of the type work, the urgent changes, and the order of work |

## Summary

- The width of the decimal buffer does not depend on the precision. So one dtype has up to six
  physical forms, every kernel must handle mixed widths, and some paths panic.
- Proposal: a decimal array is a wrapper with one child, the unscaled integers. The child dtype is
  a function of the precision. A narrower physical width becomes an encoding of the child.
- The child is an integer array, not a FixedSizeBinary array. Each decimal operation is an integer
  operation plus bookkeeping for the scale, and byte order is not integer order.
- Integer storage for precision 19 to 76 needs two new primitive types, `I128` and `I256`. They do
  not need FixedSizeBinary.
- FixedSizeBinary is still a good addition, on its own track, for opaque bytes such as UUIDs.
- The decimal file format does not change. The wrapper reads and writes the existing wire forms.
- `Decimal` stays a built-in type. A move into a well-known extension type in Rust is possible
  after the extension fixes land, and then it changes only the API.

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

The notes come from reading `develop` at `5fc5cb8`, RFC drafts, issues and pull requests. One
throwaway Rust test confirmed the `f32` cast disagreement (probe R1 in the extension notes). One
short Python script checked byte order. Section 10 of the report lists the claims that are not
verified, which include every performance claim.
