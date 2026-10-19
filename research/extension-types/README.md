# Extension types

Extension types are meant to work like Rust newtypes: an opaque wrapper over a storage type, with
behavior added on purpose. On `develop`, extension types inherit storage behavior in many places.
That causes wrong results, panics, and code paths that disagree with each other.

| Part | Content |
| --- | --- |
| [1. Why extension types must be opaque](01-why-opaque.md) | The argument for an opaque default |
| [2. Current problems](02-current-problems.md) | What is wrong on `develop` |
| [3. Design](03-design.md) | The proposed model, the alternatives, and the open questions |
| [4. Urgent changes](04-urgent-changes.md) | Fixes to make now, before the larger design |
| [5. Implementation](05-implementation.md) | Phases, order, tests and compatibility |

## Summary

- An extension type gives new meaning to storage values. A storage operation is correct for an
  extension type only when it agrees with that meaning, and only the extension type knows when it
  agrees. So the safe default is to inherit no semantic behavior.
- On `develop`, the same cast or comparison can return a correct value, a wrong value, or an error.
  The outcome depends on the encoding of the input, and on whether one row or the whole array is
  read.
- The proposal: extension types are opaque by default. Semantics are decided once, when an operation
  is bound, by a rewrite into storage operations. The kernels of generic operations, statistics
  and pruning see only storage.
- Each extension dtype declares the storage behavior that it inherits, such as equality and order.
  Operations across dtypes need explicit rules, and the design does not assume that they exist.
- The position of the wrapper relative to generic encodings is a separate decision, and it is
  open.
- Some fixes are urgent, because they cause wrong results or panics. They cover implicit casts
  through storage, comparisons against raw storage, storage checks, and panics on reads and on
  nullability changes.

## Agent documents

- [`agents/findings.md`](agents/findings.md): every finding, with file and line references,
  reproductions, observed and expected behavior, and fix sketches.
- [`agents/design-notes.md`](agents/design-notes.md): API sketches, invariants, rewrite rules, the
  derive table for each type, the validation model, and file lists for each phase.
- [`agents/probe.rs`](agents/probe.rs) and [`agents/probe_followup.rs`](agents/probe_followup.rs):
  the throwaway tests that reproduced the confirmed findings.

## Evidence

The findings come from reading `develop` at `5fc5cb8`, and from two throwaway tests in debug
builds. Findings marked "lead" in the agent catalog are not confirmed. No performance claim is
measured.

## Related

The [Decimal notes](../decimal/README.md) use `Decimal` as a test of this design. They also place
this plan in a larger graph of type work.
