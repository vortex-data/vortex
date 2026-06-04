# string-skip changelog

## 2026-05-19 — Documentation review updates

This entry records documentation-only changes made after reviewing
`INTEGRATION_PLAN.md`, `README.md`, and the related OnPair/string-skip design
notes against the current Vortex codebase.

Changed:
- Clarified that Phase A is implemented and test-passing, but not yet
  CI-clean because clippy with `-D warnings` currently fails.
- Chose an OnPair-local `OnPairSkipIndexLayout` for v1 integration and kept
  `string-skip` as a Vortex-free algorithm crate.
- Replaced stale `ChunkStatsLayoutReader` / `FileStatsLayoutReader` integration
  language with the layout-tree approach based on `ZonedLayout`.
- Documented that final partial chunks must be supported by matching one skip
  entry to each actual data child, rather than assuming every chunk has exactly
  `row_block_size` rows.
- Moved explicit row sorting out of the critical B.1 path and into a separate
  follow-up milestone because it changes writer-stream semantics.
- Fixed the registry recommendation: an OnPair-local layout cannot be registered
  from `vortex-layout::LayoutSession::default`; it should be registered by an
  OnPair initialization hook called by `vortex-file::register_default_encodings`.
- Narrowed generic dictionary claims. The current bloom path is safe for OnPair
  v1's `u16`, lex-sorted, deterministic LPM dictionary with 16-byte max tokens;
  FSST needs a separate adapter/soundness story.
- Corrected `DictPresence` wording from "exact" to "sound necessary condition"
  for equality/prefix pruning.
- Corrected the BitFunnel-style ubiquitous-bigram soundness explanation:
  skipped bigrams are ignored as pruning evidence, not assumed truly present in
  every chunk.
- Recorded public-API and clippy cleanup as B.0 preflight work before adding
  stricter CI gates.

Validation observed during review:
- `cargo test -p string-skip` passed.
- `cargo clippy -p string-skip --all-targets -- -D warnings` failed on existing
  lint issues in `string-skip` library code.

## 2026-05-19 — Implementation execution plan

Added a concrete PR-by-PR execution plan to `INTEGRATION_PLAN.md`.

Changed:
- Split implementation into eight reviewable PR slices: Phase A cleanup, OnPair
  adapter, codec, writer, reader, registration/file integration, CI/benches, and
  cleanup follow-ups.
- Added explicit validation commands and merge criteria for each slice.
- Recorded ordering constraints that keep file-format work out of the adapter
  PR and keep generic layout work deferred until a second encoding adapter
  exists.
- Captured the collection-dependency decision point for fixing the current
  `std::collections` clippy failures while preserving the intended standalone
  shape of `string-skip`.

## 2026-05-19 — Implementation plan refinement

Refined the PR execution plan to remove avoidable ambiguity before coding
starts.

Changed:
- Made `string-skip` publishable by default because `vortex-onpair` is
  publishable and will depend on it at runtime.
- Chose direct `hashbrown` usage for `string-skip` instead of
  `vortex_utils::aliases`, preserving the crate's Vortex-free boundary while
  satisfying the workspace clippy policy.
- Added explicit execution rules and a PR dependency order.
- Split the layout shell/codec work into its own pre-registration PR so writer
  and reader changes have a stable layout identity to build on.
- Added reviewer-focus notes and sharper validation gates for partial chunks,
  async EOF ordering, expression translation soundness, future-version
  fallback, and old-file compatibility.
