# vortex-engine Rust style guide

Conventions that aren't enforced by `cargo fmt` or `cargo clippy` but
that the codebase follows. When in doubt, look at recent files in
`src/operators/` and `src/scheduler/` — they're the reference shape.

The rest of this document is what you should follow as you write or
edit Rust in this repo. **These are not suggestions.** They match the
shape of the existing codebase; mixing styles in the same file is
worse than picking a different style and applying it consistently.

## Imports

- **One import per `use` line.** No `use foo::{bar, baz, qux}`
  grouping. Each import is its own line, sorted alphabetically inside
  its group.
- **Three groups, separated by a blank line:** `std::*`, then external
  crates (incl. workspace crates like `vortex_*`), then `crate::*` /
  `super::*`. `cargo +nightly fmt` enforces this via
  `group_imports = "StdExternalCrate"`.
- **Always import the type, then use the short name.** Never write
  fully-qualified paths in function bodies, signatures, or type
  annotations. If `Runner` lives in `crate::queries::bench`, add
  `use crate::queries::bench::Runner;` at the top and write `Runner`
  in the code, not `crate::queries::bench::Runner`.
- **No re-imports of items already visible.** If a file uses
  `super::aggregate_common::scalar_to_array`, write
  `use super::aggregate_common::scalar_to_array;` once at the top —
  don't `use super::aggregate_common; ... aggregate_common::scalar_to_array(...)`.
- **Trait imports go with everything else.** No special `// for trait`
  comments; sort them alphabetically with the rest. If a trait is
  used only for its method, the `use` line is the contract that
  documents that.
- **Local `use` blocks inside functions are allowed only when** the
  imported name shadows another (rare) or when a generic helper takes
  one trait per call site. Default to top-of-file.

```rust
// Good
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::Mutex;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;

use super::aggregate_common::merge_partials;
use super::aggregate_common::scalar_to_array;

use crate::Batch;
use crate::Operator;
use crate::queries::bench::Runner;

// Bad — fully-qualified in body
runner: &crate::queries::bench::Runner,

// Bad — grouped braces
use vortex_array::aggregate_fn::{AggregateFnRef, fns::sum::Sum};
```

## Naming and visibility

- `pub` is for the crate's external API. `pub(crate)` is for items used
  across modules but not part of the public surface. `pub(super)` is
  for items shared between sibling modules under the same parent
  (e.g. `aggregate_common` helpers used by `aggregate`,
  `partial_aggregate`, `merge_aggregate`).
- Type names match their file: `Aggregate` lives in `aggregate.rs`,
  `PartialAggregate` in `partial_aggregate.rs`. Don't rename in the
  re-export.
- Functions that take a `usize` count argument should use a meaningful
  name: `worker_count`, `lane_count`, `n_inputs` — never just `n` or
  `count` at the API boundary.
- Sinks end in `*Sink` (`CollectI64Sink`, `ArrayCollectSink`). Sources
  don't have a suffix — they're whatever the data is (`Filter`,
  `LazyVortexFile`).

## Module organisation

- One operator per file in `src/operators/`, named after the operator
  type. Shared helpers live in a `*_common.rs` sibling, not in any
  one operator's file.
- `mod.rs` declares `mod foo;` and re-exports `pub use foo::Foo;` —
  nothing else. No glue code, no implementations.
- Layout-aware sources (anything that talks to vortex-file) go under
  `src/layouts/`. Format-agnostic operators go under `src/operators/`.
  Scheduler/runtime internals go under `src/scheduler/` and
  `src/drivers/`.
- Tests for a module live in a `#[cfg(test)] mod tests { ... }` block
  at the bottom of that module. Cross-module behaviour tests live in
  `tests/`.

## Doc comments

- Every `pub` item has a `///` doc comment. Crate root and every
  module gets a `//!` header explaining what lives there and why.
- The first sentence of a `///` comment is one line, ends in a
  period, and is a self-contained summary — that's what `cargo doc`
  shows in listings. Detail goes in subsequent paragraphs.
- Don't write doc comments that just restate the signature ("Returns
  the count"). If the name and types already say it, skip the comment.
- For non-obvious `pub` items, document **why** the API exists, not
  just what it does. Future-you doesn't need a function-name
  paraphrase; future-you needs the constraint that drove the design.
- Inline `//` comments inside function bodies are for non-obvious
  *why*. If the comment is "what" — delete it; the code already says
  what.

## Error handling

- Errors that originate inside the engine use `EngineError::message`
  with a short label that identifies the operator or phase
  (`"aggregate flush: {e}"`, `"merge combine_partials: {e}"`).
- Propagate with `?`. No `unwrap()` outside tests; no `expect()`
  outside tests except for invariants that genuinely cannot fail
  (channel-receiver-dropped, etc.) — and even then, prefer
  `.expect("descriptive message")` over `.unwrap()`.
- `EngineResult<T>` is the engine's `Result` alias. New public APIs
  return it.

## `unsafe`

- Avoid `unsafe` unless it's load-bearing. The current uses are:
  - Raw pointer plumbing into worker threads (`drivers/mod.rs`'s
    `EngineWorkerPool`), where the safety contract is documented on
    the surrounding type.
  - Vortex bridge code where the upstream API requires it.
- Every `unsafe` block has a `// SAFETY:` comment immediately above
  it explaining the invariant the caller is relying on.

## Comments inside operators

- Every `Operator` impl has a module-level `//!` comment explaining
  the operator's purpose, parallelism contract, and any non-obvious
  behaviour. Look at `src/operators/union.rs` and
  `src/operators/aggregate.rs` for the shape.
- Inside `run`, label the phases when there is more than one
  ("Phase 1: drain available batches and accumulate", "Phase 2:
  flush and seal"). The label gives a future reader a foothold; the
  steps within the phase don't need narration.

## Channels and capacity

- `ChannelBuffer::bounded_bytes(N)` for every connection. Pick N
  based on the expected per-batch size × pipelining depth, not "what
  works". Comment if a value is non-obvious.
- Don't paper over a channel-capacity panic with a bigger number —
  a panic means the producer/consumer pair has a back-pressure bug.
  See `docs/decisions/` and the channel-capacity design notes.

## Tests

- `#[test]` functions named `<scenario>_<expected_outcome>` —
  `lane_safety_matches_known_vtables`, not `test_lane_safety`.
- One assertion topic per test. If a test grows to assert three
  unrelated things, it should be three tests.
- `assert!`, `assert_eq!`, `assert_ne!` only — no `dbg!` outside
  active investigation.

## What `cargo fmt` already handles

Don't fight rustfmt. The repo uses the workspace's
`rustfmt.toml`-equivalent settings (import grouping, field
shorthand, 2024 edition style). Run `cargo +nightly fmt` before
committing.
