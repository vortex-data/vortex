# string-skip → Vortex Integration Plan

**Status:** Phase A (standalone crate) implemented. `cargo test -p
string-skip` passes, but the crate is not CI-clean yet because
`cargo clippy -p string-skip --all-targets -- -D warnings` currently
fails. Phase B is the Vortex integration plan.

**Owner / SME:** [TBD]
**Estimated calendar time:** 4–5 weeks single-engineer after B.0
preflight issues are fixed
**Estimated LOC for Phase B:** ~2,050 new + ~30 LOC YAML changes

---

## Executive summary

We have implemented a standalone Rust crate, `string-skip`, that provides
chunk-level skip indexes for string columns. It supports range / equality /
prefix / suffix / substring / wildcard / length / null / set predicates with
sound (no false negative) pruning across 14 distinct query classes. The
algorithm is novel:
BitFunnel-style frequency-conscious code-bigram blooms with sound cover
enumeration for substring matching. Benchmarked across 18 real columns
(ClickBench, TPC-H, FineWeb, OnPair-paper datasets). No library equivalent
exists in the wild.

Phase A is intentionally Vortex-free, but it is not release/CI-ready until the
clippy blockers and public-API policy are resolved. Phase B should wire it into
Vortex as an **OnPair-local** layout encoding for v1, not as a generic
`string-skip` layout. A generic layout can be revisited after a second encoding
adapter exists.

---

## 1. Current state — Phase A (implemented)

| Component | Path | Notes |
|-----------|------|-------|
| Crate root | `encodings/string-skip/` | New workspace member |
| Algorithms | `src/{bloom,dict,presence,ubiq,tiers,prune}.rs` | DictPresence, HybridBloom, TieredBloom, UbiquitousBigrams, BigramTiers |
| Predicate AST | `src/pred.rs` | 16 variants covering all SQL string predicates |
| Per-chunk stats | `src/chunk_stats.rs` | min/max/length/null |
| Soundness tests | `tests/soundness.rs` | `proptest`, 29 unit + 2 property |
| Microbenches | `benches/bloom_probes.rs` | criterion, 5 groups |
| Example tool | `examples/queries_demo.rs` | parquet-backed end-to-end demo |
| Design doc | `DESIGN.md` | Algorithm details |
| README | `README.md` | API summary |

Dependencies: `memchr`, `bincode`, `serde`, `thiserror` (runtime); `proptest`,
`criterion`, `parquet`, `arrow-array`, `clap`, `rand`, `rand_xoshiro` (dev).

Test results:
- 29 unit tests + 2 property tests passing
- ~1000+ soundness verifications per `cargo test` (50 cases × 20 predicates)
- Microbenches: bloom probe ~12–17ns, dict_presence_is_set ~0.65ns

Preflight blockers before calling Phase A release-ready:
- `cargo clippy -p string-skip --all-targets -- -D warnings` fails on
  clippy lints and workspace-disallowed `std::collections::{HashMap, HashSet}`.
- The crate has no `public-api.lock`. Add one before relying on the repository
  public-API CI because v1 keeps `string-skip` publishable.
- Example/test warnings use deprecated `rand::Rng::gen_range` and contain
  unused test code; these need cleanup if examples/tests are linted with
  `-D warnings`.

---

## 2. Phase B — Vortex integration

### 2.1 Architecture overview

Mirror `ZonedLayout` (`vortex-layout/src/layouts/zoned/`), not
`FileStatsLayoutReader`. The latter is a *file-level* decorator;
chunk-level pruning belongs *inside* the layout tree.

For v1, add an OnPair-specific layout encoding, tentatively
`OnPairSkipIndexLayout`, in `vortex-onpair`. This keeps `string-skip` as a pure
algorithm crate and avoids pretending the layout is encoding-agnostic before
FSST or another adapter exists.

The layout wraps a chunked data layout. Two children:

- child 0 — `LayoutChildType::Transparent("data")`: the wrapped chunked
  data layout (typically `ChunkedLayout`).
- child 1 — `LayoutChildType::Auxiliary("skip")`: a `FlatLayout` holding
  the serialized skip-index blob.

The skip blob is a single versioned payload containing bincode-encoded data:

```rust
SkipIndexBlob {
    ubiq: UbiquitousBigrams,    // column-shared
    tiers: BigramTiers,         // column-shared
    dict_ref: DictRef,          // points into the OnPair column dict
    chunks: Vec<PerChunkSkip>,  // 1 entry per data chunk
}
PerChunkSkip {
    stats: ChunkStats,
    presence: DictPresence,
    bloom: Option<HybridBloom>,
    tiered: Option<TieredBloom>,
}
```

`chunks.len()` must match the number of actual data chunks, including the final
partial chunk. Fixed `row_block_size = 8192` is a useful default, but Vortex
layouts support arbitrary child row counts and the integration must preserve
that.

Total blob size is typically hundreds of KiB for millions of rows, but it must
be bounded by write options. High-diversity free-text columns can produce
multi-MiB metadata and should fall back to stats/presence-only or no skip index.

### 2.2 Phase B is split into milestones

| Milestone | Deliverable | LOC | Calendar |
|-----------|-------------|-----|----------|
| **B.0** | OnPair → `TokenDict` adapter + Phase A CI cleanup | ~120 | 2–3 days |
| **B.1** | `OnPairSkipIndexLayout` (read + write paths) | ~830 | 2–3 weeks |
| **B.2** | Versioning + back-compat | ~285 | 3–4 days |
| **B.3** | Optional sort flag / sorted metadata | ~355 | separate design or follow-up PR |
| **C.1** | CI gates, soundness fuzz extension, golden fixtures, benches | ~430 + 30 lines YAML | 3–5 days |
| **Total** | | **~2,050** | **4–5 weeks** |

### 2.3 Suggested first PR (de-risk before big work)

**Land B.0 only**: the OnPair → `TokenDict` adapter plus Phase A CI cleanup.
Add an integration test in `encodings/onpair/tests/` that builds skip indices on
deterministic OnPair-compressed data and verifies `chunk_might_match` is sound.
Real parquet columns should remain an optional/manual benchmark path unless a
small deterministic fixture is checked in. **No layout/file-format changes.**

This proves the standalone crate semantics carry through to real OnPair data
before committing to the layout work, while also making the crate compatible
with the repository's lint/public-API gates.

### 2.4 Implementation execution plan

Implement this as a sequence of reviewable PRs. Each PR must leave the tree in a
working state, keep its review question narrow, and avoid mixing algorithm
cleanup with file-format changes.

Execution rules:
- Keep `string-skip` publishable for v1. `vortex-onpair` is a published crate,
  so its runtime dependency must also be publishable unless `vortex-onpair`
  publishability is intentionally changed.
- Use a direct `hashbrown` dependency in `string-skip` rather than
  `vortex_utils::aliases`; this fixes the workspace clippy policy without
  making the algorithm crate depend on Vortex.
- Add `encodings/string-skip/public-api.lock` in PR 1 and refresh it whenever
  the public API changes.
- Do not touch `vortex-file` until PR 6.
- Do not register `OnPairSkipIndexLayout` until both writer and reader behavior
  are covered by tests.
- Do not add network-dependent parquet fixtures to PR-blocking tests.

Dependency order:

```text
PR 1 -> PR 2 -> PR 3 -> PR 4 -> PR 5 -> PR 6 -> PR 7 -> PR 8
```

PR 4 and PR 5 can be developed in parallel after PR 3 if they use disjoint
files, but they should merge in writer-before-reader order so reader tests can
round-trip through the real writer.

#### PR 1 — Phase A cleanup and release policy

Goal: make `string-skip` acceptable to the repository's normal Rust gates
without changing its algorithmic behavior.

Scope:
- Fix the current clippy failures in `chunk_stats.rs`, `prune.rs`, `tiers.rs`,
  and `ubiq.rs`.
- Replace `std::collections::{HashMap, HashSet}` in library code with
  `hashbrown::{HashMap, HashSet}` and add `hashbrown` to
  `encodings/string-skip/Cargo.toml`.
- Remove or gate deprecated `rand::Rng::gen_range` usage and unused test/example
  helpers that would fail under `-D warnings`.
- Add `encodings/string-skip/public-api.lock` through `./scripts/public-api.sh`.
- Do not disable crate publishing unless the release plan for `vortex-onpair`
  changes.

Validation:
- `cargo nextest run -p string-skip`
- `cargo clippy -p string-skip --all-targets -- -D warnings`
- `./scripts/public-api.sh`

Merge criteria:
- No Vortex integration code yet.
- No behavior changes unless accompanied by existing/new soundness coverage.
- Reviewer focus: lint/public-API hygiene only, not skip-index design.

#### PR 2 — OnPair adapter only

Goal: prove `string-skip` can consume OnPair dictionaries and code streams
without introducing a Vortex layout or file-format change.

Scope:
- Add `string-skip` as a runtime dependency of `vortex-onpair`.
- Add `encodings/onpair/src/skip_glue.rs` with `OnPairDictView<'a>`.
- Expose the module from `encodings/onpair/src/lib.rs` behind a small public or
  crate-private API, whichever is enough for tests and later layout code.
- Assert the OnPair invariants the bloom path relies on: lexicographic dict
  ordering, max token length ≤ 16 bytes, and code ids fitting in `u16`.
- Add deterministic tests in `encodings/onpair/tests/` that build OnPair data,
  feed `dv.dict_table`, `dv.codes`, and `dv.codes_offsets` into `string-skip`,
  and assert no false negatives across representative predicates.

Validation:
- `cargo nextest run -p string-skip`
- `cargo nextest run -p vortex-onpair`
- `cargo clippy -p string-skip -p vortex-onpair --all-targets --all-features -- -D warnings`

Merge criteria:
- No `vortex-file` changes.
- No layout registration.
- No network-dependent parquet fixtures.
- Reviewer focus: adapter zero-copy correctness and no false negatives.

#### PR 3 — Layout shell, skip payload codec, and metadata types

Goal: define the layout identity and persistent bytes before wiring read/write
paths, with back-compat behavior covered in isolation.

Scope:
- Add `encodings/onpair/src/skip_layout/mod.rs` with the layout id, metadata
  type, child names, and constructors. Do not register the layout yet.
- Add `encodings/onpair/src/skip_layout/codec.rs`.
- Define v1 metadata structs for column-level and per-chunk skip state.
- Add magic/version wrappers and `DecodeOutcome`.
- Add tests for current version, future version fallback, truncated payloads,
  and wrong magic inside an `OnPairSkipIndexLayout` payload.

Validation:
- `cargo nextest run -p vortex-onpair skip_layout`
- `cargo clippy -p vortex-onpair --all-targets -- -D warnings`

Merge criteria:
- Layout id is unique and OnPair-owned, for example
  `vortex.onpair.skip_index`.
- Codec round trips are deterministic.
- Unknown future versions return full-scan fallback state, not an error.
- Corrupt payloads inside a declared skip layout return an error.
- Reviewer focus: format stability and fallback/error boundaries.

#### PR 4 — Writer-side `OnPairSkipIndexStrategy`

Goal: write a data child plus auxiliary skip blob while preserving async
sequence semantics.

Scope:
- Add `encodings/onpair/src/skip_layout/writer.rs`.
- Implement `SkipIndexOptions` with conservative defaults and hard metadata
  limits.
- Build one `PerChunkSkip` for each actual stream chunk, including the final
  partial chunk.
- Accumulate skip metadata in stream order only; do not make correctness depend
  on parallel chunk-computation completion order.
- Mirror `ZonedStrategy::write_stream`: split EOF for data first, await the data
  child, then write the auxiliary skip payload with a `FlatLayout`.
- If a column is unsupported or exceeds metadata limits, fall back to the data
  child without emitting a skip layout.

Validation:
- `cargo nextest run -p vortex-onpair skip_layout`
- A focused round-trip test that writes with the strategy and verifies the
  auxiliary payload decodes and has `chunks.len() == data_child_count`.
- A focused partial-final-chunk test.
- `cargo clippy -p vortex-onpair --all-targets -- -D warnings`

Merge criteria:
- Reader pruning can still be absent; files must remain readable by falling
  back to the data child.
- No assumptions that all chunks have `row_block_size` rows.
- Reviewer focus: async sequence/EOF ordering and metadata budget behavior.

#### PR 5 — Reader-side pruning

Goal: make the layout useful by translating Vortex expressions into
`string_skip::Pred` and intersecting the skip mask with child pruning.

Scope:
- Add `encodings/onpair/src/skip_layout/reader.rs` and `expr.rs`.
- Implement lazy skip-blob loading and caching.
- Mirror `ChunkedReader` range mapping for row ranges and partial chunks.
- Support equality, ordered range, non-case-insensitive LIKE shapes, null,
  not-null, and `IN`/`ListContains` shapes that can be translated soundly.
- Treat unrecognized expressions as `ALL`, never as prunable.
- Combine skip pruning with the data child's pruning mask; skip pruning must
  only remove rows/chunks the data child would otherwise consider.

Validation:
- `cargo nextest run -p vortex-onpair skip_layout`
- Round-trip tests for `Eq`, prefix LIKE, contains LIKE, `BETWEEN`, `IS NULL`,
  `IS NOT NULL`, and mixed `AND`/`OR`.
- A negative test proving an unsupported predicate does not prune.
- A future-version payload test proving reader fallback keeps all candidate
  chunks.
- `cargo clippy -p vortex-onpair --all-targets -- -D warnings`

Merge criteria:
- No false negatives in deterministic tests.
- Reader combines skip pruning with the data child's existing pruning rather
  than replacing it.
- Reviewer focus: expression translation soundness, especially `OR` and
  unsupported subexpressions.

#### PR 6 — Registration and Vortex-file integration

Goal: make the layout available through normal file open/write paths while
keeping registration ownership in `vortex-onpair`.

Scope:
- Add `vortex_onpair::initialize(session)` to register the OnPair skip layout.
- Call it from `vortex-file::register_default_encodings` behind the existing
  `onpair` feature.
- Add a Vortex-file round-trip test that writes string data, reads it through
  normal file APIs, and observes chunk pruning for translated predicates.
- Refresh public API locks for touched public APIs.

Validation:
- `cargo nextest run -p vortex-onpair`
- `cargo nextest run -p vortex-file test_onpair`
- `./scripts/public-api.sh`
- `cargo clippy -p vortex-onpair -p vortex-file --all-targets --all-features -- -D warnings`

Merge criteria:
- Default sessions can read old files unchanged.
- New files with skip payloads degrade to full scan if the payload version is
  newer than the reader.
- Reviewer focus: registration ownership and old-file compatibility.

#### PR 7 — CI, benches, and fixtures

Goal: add durable regression coverage after the implementation shape has
settled.

Scope:
- Add the `string-skip` CodSpeed shard.
- Add or persist `proptest` regression seeds.
- Add release-mode soundness coverage sized for CI.
- Add golden pruning fixtures only if they are deterministic, small, and do not
  require network access.
- Keep real-data sweeps as `workflow_dispatch` or manual benchmark jobs unless
  the fixture is checked into the repository.

Validation:
- `cargo nextest run -p string-skip --release`
- `cargo bench -p string-skip --bench bloom_probes -- --list`
- Relevant CI workflow dry-runs or targeted local equivalents.

Merge criteria:
- CI cost is measured and acceptable.
- Golden fixture regeneration command is documented if fixtures are checked in.
- Reviewer focus: CI cost, determinism, and signal-to-noise.

#### PR 8 — Cleanup follow-ups

Goal: remove duplicated experimental paths only after the new path is proven.

Scope:
- Decide whether `encodings/onpair/src/skip.rs` should be deleted, retained as
  an example-only implementation, or reduced to compatibility glue.
- Move sort/sortedness work into a separate design if it is still desired.
- Revisit a generic encoding-shared layout only after a second encoding adapter
  exists.

Validation:
- Normal checks for files touched by the cleanup.

Merge criteria:
- Cleanup does not reduce test coverage for the implemented OnPair path.
- Reviewer focus: deleting dead code without hiding behavior changes.

---

## 3. Detailed milestone plans

### 3.1 B.0 — OnPair adapter

**Files:**
- `encodings/onpair/src/skip_glue.rs` (new)
- `encodings/onpair/Cargo.toml` add runtime dependency on `string-skip`
- `encodings/string-skip/public-api.lock` added through `./scripts/public-api.sh`

```rust
#[derive(Copy, Clone)]
pub struct OnPairDictView<'a> {
    dict_bytes: &'a [u8],
    dict_table: &'a [u64],
}

impl<'a> OnPairDictView<'a> {
    pub fn from_decode_view(dv: &DecodeView<'a>) -> Self { ... }
}

impl<'a> string_skip::TokenDict for OnPairDictView<'a> {
    fn len(&self) -> usize { self.dict_table.len() }
    fn token_bytes(&self, id: u16) -> &[u8] {
        let entry = self.dict_table[id as usize];
        let off = (entry >> 16) as usize;
        let len = (entry & 0xffff) as usize;
        &self.dict_bytes[off..off + len]
    }
}
```

Zero-copy, no allocation. Pass `dv.codes` and `dv.codes_offsets` directly
to `string-skip` builders (they take `&[u16]` / `&[u32]`).

**OnPair invariants confirmed compatible:**
- Dict is lex-sorted (`encodings/onpair/src/lpm.rs:6`)
- Max token size 16 bytes — relevant to the current substring cover
  enumeration, which uses a 16-byte maximum. This is safe for OnPair v1 but
  should become a trait/config value before claiming generic dictionary support.
- Dict size capped at 4096 for `dict-12` preset — fits in u16

**Tests:** round-trip vs direct `dv.dict_table` lookups; integration test using
small deterministic OnPair data and asserting no false negatives for equality,
prefix, substring, wildcard, length, null, and set predicates. Real-data
pruning-rate parity with `examples/queries.rs` belongs in a manual benchmark or
a small checked-in fixture, not a network-dependent CI test.

**CI cleanup:** make `cargo clippy -p string-skip --all-targets -- -D warnings`
pass before adding repository-wide gates. Current failures include
`unnecessary_map_or`, `collapsible_if`, `manual_repeat_n`, and
workspace-disallowed `std::collections::{HashMap, HashSet}`.

**Followup:** decide whether to delete `encodings/onpair/src/skip.rs`
(the in-tree skip code) after migration — out of scope for B.0.

### 3.2 B.0.x — FSST adapter (DEFERRED to v2)

FSST violates three `TokenDict` invariants:
1. Codes are `u8`, not `u16` (≤255 symbols)
2. Wire format mixes codes with literal-escape bytes — breaks LPM
   determinism that bloom soundness depends on
3. Symbol table is frequency-ordered, not lex-sorted

Mitigation requires a permutation + escape-aware code rewriting
(~150–250 LOC) plus a separate soundness story. **Defer**. If we want
FSST skip in v2, options are:
- (a) restrict FSST to `DictPresence` + `ChunkStats` only (no bigram bloom)
- (b) full escape-aware adapter with sentinel "skip-disabled" rows

### 3.3 B.1 — `OnPairSkipIndexLayout`

**Files (all new, in `encodings/onpair/src/skip_layout/`):**

| File | Purpose | LOC |
|------|---------|-----|
| `mod.rs` | `VTable` impl, `OnPairSkipIndexLayout` struct, `SkipIndexMetadata` | ~120 |
| `reader.rs` | `OnPairSkipIndexReader` implementing `LayoutReader` | ~180 |
| `writer.rs` | `OnPairSkipIndexStrategy` + `SkipIndexAccumulator` | ~250 |
| `codec.rs` | versioned bincode (de)serialization | ~60 |
| `expr.rs` | Vortex `Expression` → `string_skip::Pred` translator | ~220 |
| (planner glue) | `split_conjunction` walk + chunk-set intersection | ~80 |

Keep the `string-skip` crate free of `vortex-*` dependencies. The layout code
can depend on `vortex-layout`, `vortex-array`, and `string-skip` from
`vortex-onpair`.

**`OnPairSkipIndexReader` (read path):**

```rust
pub struct OnPairSkipIndexReader {
    layout: OnPairSkipIndexLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    chunk_offsets: Vec<u64>,                          // mirror ChunkedReader
    blob: OnceCell<Arc<SkipIndexBlob>>,               // lazy-loaded
    blob_loader: parking_lot::Mutex<Option<BoxFuture<'static, VortexResult<Arc<SkipIndexBlob>>>>>,
}

impl LayoutReader for OnPairSkipIndexReader {
    fn pruning_evaluation(&self, row_range, expr, mask) -> VortexResult<MaskFuture> {
        // 1. Always forward to data child.
        let data_eval = self.data_child()?.pruning_evaluation(row_range, expr, mask.clone())?;
        // 2. Translate `expr` → list of `(FieldPath, Pred)`.
        let Some(preds) = expr::to_skip_preds(expr) else { return Ok(data_eval); };
        // 3. Load blob (cheap, one segment).
        let blob_fut = self.load_blob();
        let chunk_ranges = self.chunk_ranges_in_mask(row_range); // include partial chunks
        Ok(MaskFuture::new(mask.len(), async move {
            let blob = blob_fut.await?;
            let mut builder = BitBufferMut::with_capacity(mask.len());
            for (chunk_idx, mask_slot) in chunk_ranges {
                let keep = preds.iter().all(|(_, p)|
                    chunk_might_match(p, &blob.state_for(chunk_idx)));
                builder.append_n(keep, mask_slot.len());
            }
            let skip_mask = Mask::from(builder.freeze());
            let combined = mask.bitand(&skip_mask);
            if combined.all_false() { Ok(combined) }
            else { Ok(combined.bitand(&data_eval.await?)) }
        }))
    }
}
```

Mirrors `ZonedReader::pruning_evaluation` at `vortex-layout/src/layouts/zoned/reader.rs:118-193`.
The chunk-to-mask mapping should mirror `ChunkedReader::ranges`, not assume a
fixed block length.

**`OnPairSkipIndexStrategy` (write path):**

```rust
pub struct OnPairSkipIndexStrategy {
    chunk_strategy: Arc<dyn LayoutStrategy>,
    skip_strategy: Arc<dyn LayoutStrategy>,
    options: SkipIndexOptions,
}

pub struct SkipIndexOptions {
    pub bloom_bits_per_row: usize,    // e.g. 16
    pub use_tiered: bool,
    pub ubiq_pct: u8,                 // e.g. 75
    pub max_dict_size: usize,
}

#[async_trait]
impl LayoutStrategy for OnPairSkipIndexStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let data_eof = eof.split_off(); // data segments before skip payload
        let acc = Arc::new(Mutex::new(SkipIndexAccumulator::new(&self.options)));
        let stream = wrap_stream_and_push_chunks(stream, Arc::clone(&acc))?;

        let data_layout = self
            .chunk_strategy
            .write_stream(ctx.clone(), Arc::clone(&segment_sink), stream, data_eof, session)
            .await?;

        let blob_bytes = encode_column_v1(&acc.lock().finish()?);
        let skip_stream = blob_bytes_to_single_array_stream(blob_bytes, eof.split_off())?;
        let skip_layout = self
            .skip_strategy
            .write_stream(ctx, segment_sink, skip_stream, eof, session)
            .await?;

        Ok(OnPairSkipIndexLayout::new(data_layout, skip_layout).into_layout())
    }
}
```

Mirrors `ZonedStrategy::write_stream` at `vortex-layout/src/layouts/zoned/writer.rs:60-181`.
Do not implement this as a synchronous writer shim: the current
`LayoutStrategy` API is async, sequence-pointer based, and sensitive to EOF
ordering.

**Expression → `Pred` translator (`expr.rs`):**

Vortex already has all the needed expression types as scalar fns:
- `Binary` (with `Operator::{Eq, NotEq, Lt, Lte, Gt, Gte, And, Or}`)
- `Between` (with strict/non-strict bounds)
- `Like` (with `LikeOptions { negated, case_insensitive }`)
- `IsNull`, `IsNotNull`, `Not`
- `ListContains` (for `IN (...)`)
- `GetItem` (column reference) + `Root` + `Literal`

Length predicates are supported by `string-skip::Pred`, but this repository
does not currently expose a string-length scalar function in the Vortex
expression layer. Treat length pushdown as out of scope for v1 unless that
expression exists before implementation starts.

```rust
pub fn pred_from_vortex_expr(expr: &Expression, column: &FieldName) -> Option<Pred>;
```

Dispatch table (key cases):

| Vortex shape | `Pred` |
|--------------|--------|
| `Binary(Eq, col(C), lit(s))` | `Eq(bytes(s))` |
| `Binary(Lt/Gt/Le/Ge, col, lit)` | corresponding range pred |
| swapped: `Binary(op, lit, col)` | re-dispatch via `Operator::swap` |
| `Between(col, lit_lo, lit_hi, NonStrict/NonStrict)` | `Between(lo, hi)` |
| `Like(col, lit("pattern"), { not negated, not ci })` | parse pattern via byte tokenizer |
| `IsNull(col)` | `Pred::IsNull` |
| `IsNotNull(col)` or `Not(IsNull(col))` | `Pred::IsNotNull` |
| `ListContains(lit(list), col)` | `Pred::InSet(elements)` |
| anything else | `None` (planner falls back to full scan) |

**LIKE pattern parser** — small byte tokenizer that classifies into 7 shapes:
`Eq | Prefix | Suffix | Contains | PrefixSuffix | SingleWildcard | MultiFragment`.
Implement SQL `\` escape handling here; the existing Vortex stats-only LIKE
parser handles only exact/prefix shapes.

**AND/OR composition** — `pred_from_vortex_expr` returns single `Pred`.
Composition done by planner:
```text
for clause in split_conjunction(expr):    // existing Vortex helper
    keep_set ∩= prune_clause(clause)

prune_clause(c):
  if c is Binary(Or, l, r): return prune_clause(l) ∪ prune_clause(r)
  if pred_from_vortex_expr(c, col).is_some(): return skip.prune(pred)
  return ALL                              // unrecognised → keep all
```

**B.1 implementation choices:**

1. **Encoding-shared vs encoding-local layout?** Use encoding-local
   `OnPairSkipIndexLayout` for v1; generic v2 if another encoding gets a
   skip-index adapter.
2. **Codec location?** Keep it next to `OnPairSkipIndexLayout`
   inside `vortex-onpair`, *not* inside the pure `string-skip` crate
   (keeps `string-skip` Vortex-free).
3. **Registry registration?** Because v1 is encoding-local, do not register it
   in `LayoutSession::default` from `vortex-layout` (that would couple
   `vortex-layout` to `vortex-onpair`). Add `vortex_onpair::initialize(session)`
   that registers the layout with `session.layouts()`, then call it from
   `vortex-file::register_default_encodings` behind the existing `onpair`
   feature.

### 3.4 B.2 — Versioning + back-compat

**Versioning (file: `encodings/onpair/src/skip_layout/codec.rs`, ~205 LOC):**

```text
Per-chunk skip-state blob:
+--------+--------+--------+--------+----------------------- ... ---+
| 'S'    | 'S'    | ver:u8 | rsv:u8 | bincode(ChunkSkipState_vN)    |
+--------+--------+--------+--------+----------------------- ... ---+

Column-level skip-meta blob:
+--------+--------+--------+--------+----------------------- ... ---+
| 'S'    | 'M'    | ver:u8 | rsv:u8 | bincode(ColumnSkipMeta_vN)    |
+--------+--------+--------+--------+----------------------- ... ---+
```

API:
```rust
pub fn encode_chunk_v1(state: &ChunkSkipState) -> Vec<u8>;
pub fn encode_column_v1(meta: &ColumnSkipMeta) -> Vec<u8>;

pub enum DecodeOutcome<T> {
    Ok(T),
    UnknownVersion { found: u8, known: u8 },   // forward-compat: skip pruning
}

pub fn decode_chunk(buf: &[u8]) -> VortexResult<DecodeOutcome<ChunkSkipState>>;
pub fn decode_column(buf: &[u8]) -> VortexResult<DecodeOutcome<ColumnSkipMeta>>;
```

**Back-compat rules:**

| Situation | Behaviour |
|-----------|-----------|
| No skip-index payload (old file) | Fall back to existing pruning, no error/warning |
| Unknown version > `KNOWN_VERSION` | `tracing::debug!` once; full-scan fallback |
| Older version < `KNOWN_VERSION` | Dispatch on `ver` → `decode_vN` (each version kept forever) |
| Magic mismatch inside an `OnPairSkipIndexLayout` payload | Hard error — corruption |

**Migration tests (~80 LOC):** no-payload old file, future-version payload,
truncated header, foreign-magic payload.

**File-format version (`vortex-file::VERSION`) is NOT bumped** —
skip-index versioning is encoding-local.

### 3.5 B.3 — Optional sort flag / sorted metadata

Sorting is not required for B.1 and should not block skip-index layout
integration. Treat it as a follow-up design because it changes writer semantics:
sorting must happen before `TableStrategy` splits columns, may require
materializing the input stream, and is awkward for the push-based `Writer` API.

If pursued, use an opt-in API similar to:

```rust
pub enum SortStrategy {
    Preserve,         // default
    Lexicographic,
    LengthThenLex,
    AutoDetect,       // record IsSorted flag iff already sorted; no reorder
}

// VortexWriteOptions gains:
sort_rows: Option<(FieldPath, SortStrategy)>,
```

Flow for `Lexicographic` / `LengthThenLex`: materialize the input stream,
compute a permutation for the chosen field, apply `take` to all columns,
re-stream in `row_block_size` chunks, and set sorted metadata. Flow for
`AutoDetect`: keep streaming and only record sortedness when already true.

Reuses:
- `vortex-array/src/aggregate_fn/fns/is_sorted/mod.rs` for auto-detection.
- `vortex-array/src/expr/stats/mod.rs` (`Stat::IsSorted` /
  `Stat::IsStrictSorted`) for existing sortedness stats.

Reader-side binary-search pruning over chunk min/max should be a separate
optimization after sortedness can be associated with a specific field path.

### 3.6 C.1 — Testing + CI

**Tier 1 (PR-blocking, runs on every PR):**

| Gate | Mechanism | Already exists? |
|------|-----------|-----------------|
| Unit + property tests (debug) | `ci.yml::rust-test-other` | Yes (auto-covers new crate) |
| Soundness fuzz (release, 1024 cases × 20 preds = ~20k checks) | New step in `rust-test-other` | **Add** |
| Clippy `-D warnings` | `ci.yml::rust-lint` | Yes |
| Nightly fmt | `ci.yml::rust-lint` | Yes |
| Doc tests | `ci.yml::rust-docs` | Yes (add doc examples) |
| `public-api.lock` | `ci.yml::public-api` | **Add** `string-skip` lock file |
| CodSpeed microbench shard | `codspeed.yml`: new shard | **Add** |
| Golden pruning fixtures (release) | New `tests/golden_pruning.rs` gated by `golden-fixture` feature | **Add** |

YAML change: `codspeed.yml` — add `{ shard: 9, name: "String skip", packages: "string-skip" }`.

**Tier 2 (nightly / on-demand):**

- Extended property tests: 10,000 cases (via `STRING_SKIP_HEAVY=1` env)
- Real-data sweep across ClickBench/FineWeb/TPC-H — `workflow_dispatch` only

**Soundness fuzz config (~50 LOC change):**

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(debug_assertions) { 50 } else { 1024 },
        max_shrink_iters: if cfg!(debug_assertions) { 100 } else { 4096 },
        failure_persistence: Some(Box::new(
            FileFailurePersistence::WithSource("regressions"))),
        .. ProptestConfig::default()
    })]
    #[test]
    fn soundness_chunk_might_match(...) { ... }
}
```

Regression seeds persist to `encodings/string-skip/tests/regressions/`.
1,024 cases × 20 predicates ≈ 20k evaluations per PR; detects bugs that
affect ≥0.025% of input space at 99% confidence.

**Golden pruning fixtures (~200 LOC + ~10 MB data):**

```
encodings/string-skip/
  goldenfiles/
    urls_1m.parquet                  # ~5 MB ClickBench URL top 1M
    hostnames_500k.parquet
    free_text_100k.parquet
    golden_pruning.json              # expected pruning rates per query class
  src/bin/regen-goldens.rs           # CLI to regenerate fixtures
  tests/golden_pruning.rs            # gated by golden-fixture feature
```

Tolerance ±2pp. Update process: `cargo run -p string-skip --bin regen-goldens`
when intentional algorithm changes ship.

**New criterion benches (~150 LOC):**

| Benchmark | What it measures |
|-----------|------------------|
| `chunk_build` | Per-chunk skip-index build cost |
| `chunk_probe_eq` / `_prefix` / `_contains` | End-to-end `chunk_might_match` per predicate class |
| `tokenize_needle` | Dict tokenizer throughput (MB/s) |

**CI cost added per PR:** ~5–6 minutes total.

---

## 4. Critical-path decisions

1. **Encoding-shared vs encoding-local layout?**
   - Decision for this plan: encoding-local `OnPairSkipIndexLayout` for v1.
2. **Codec location?**
   - Decision for this plan: `vortex-onpair` for now; hoist only if FSST or
     another encoding gets a v2 adapter.
3. **Sort default?**
   - Decision for this plan: off by default and not part of B.1. Any explicit
     sort API should be a follow-up design.
4. **Registry registration?**
   - Decision for this plan: register via `vortex_onpair::initialize(session)`,
     called from `vortex-file::register_default_encodings` behind the `onpair`
     feature.
5. **Drop existing `encodings/onpair/src/skip.rs`?**
   - Defer to a separate cleanup PR after B.0 lands.

---

## 5. Consolidated risks

| Risk | Severity | Mitigation |
|------|---------:|------------|
| `Cast` wrappers (utf8↔utf8) break translator | Med | Peel no-op casts in `match_col` helper |
| `SequencePointer` ordering: skip blob can't be read before data commits | High | Mirror `ZonedStrategy::eof.split_off()` pattern exactly |
| Final partial chunks mismatch skip blob entries | High | Build one skip entry per actual data child and mirror `ChunkedReader::ranges` |
| Sort materialization cost defeats streaming on huge datasets | Med | Document; `AutoDetect` provides cheap streaming path |
| `StatsCatalog` is per-field, not per-chunk | Low | Bypass it; our `Pred` translation calls `chunk_might_match` directly |
| Field-specific sortedness needs schema/API design | Med | Keep explicit sort support out of B.1 |
| Layout id collision | Low | Use a unique `vortex.onpair.skip_index` layout id and register once via OnPair initialization |
| Generic dictionary claims become unsound for long-token dicts | High | Keep v1 OnPair-only or add max-token-len to `TokenDict` before generalizing |
| Golden fixtures drift with intentional tuning | Low | `regen-goldens` CLI; require PR-description justification |
| `LayoutReader::pruning_evaluation` shape mismatch | Low | Returns `MaskFuture` — confirmed compatible |
| `Layout::metadata()` not suited for blob-sized skip data | Low | Use auxiliary `FlatLayout` segment (precedent: `ZonedLayout`) |

---

## 6. Done = ship criteria

**B.0 done when:**
- `OnPairDictView` adapter compiles and round-trips against `dv.dict_table`
- Integration test in `encodings/onpair/tests/` uses deterministic OnPair data
  and verifies no false negatives across the supported predicate classes
- Existing `encodings/onpair/examples/queries.rs` still runs (no regression)
- `cargo clippy -p string-skip --all-targets -- -D warnings` passes
- `encodings/string-skip/public-api.lock` exists and is refreshed

**B.1 done when:**
- `OnPairSkipIndexLayout` is registered through `vortex_onpair::initialize`
- Round-trip test: write Vortex file from string input → read Vortex file →
  assert chunk pruning happens
  for `Eq`, `Like 'p%'`, `Like '%x%'`, `BETWEEN`, `IS NULL`
- Soundness invariants hold under `proptest` × 1024 cases at release
- CodSpeed regression check passes (no >5% slowdown on hot paths)

**B.2 done when:**
- Reading a v0 (pre-skip-index) file works unchanged
- Reading a future-version skip-index payload degrades gracefully (debug log + full scan)
- Versioning byte layout has a "never-change-these-constants" test

**B.3 done when:**
- A separate sort design covers stream materialization, push-writer behavior,
  and field-specific sortedness metadata
- `VortexWriteOptions::with_sort_rows` is available only if the above design is accepted

**C.1 done when:**
- `codspeed.yml` includes `string-skip` shard
- `tests/regressions/` has at least one persisted seed (proves the mechanism works)
- `goldenfiles/golden_pruning.json` checked in with reproducible numbers
- `cargo nextest run -p string-skip --release` runs the 1024-case property test in <60s

---

## 7. Out of scope (deferred)

- FSST adapter (needs separate soundness design)
- DataFusion integration beyond what `vortex-datafusion` already provides
- Generic encoding-shared `SkipIndexLayout` (encoding-local v1 first)
- BitFunnel "higher rank rows" multiplexing (research direction)
- Ribbon/XOR filter backends (future perf optimization if blooms saturate)
- Anti-substring predicates (`NOT LIKE '%x%'`) — rarely prunes anyway
- Case-insensitive `ILIKE` — needs pre-folded bloom variant
- String length expression pushdown until Vortex exposes a suitable scalar fn

---

## 8. References

| Doc | Path |
|-----|------|
| Algorithm design | `encodings/string-skip/DESIGN.md` |
| Crate API | `encodings/string-skip/README.md` |
| Documentation changelog | `encodings/string-skip/CHANGELOG.md` |
| Standalone bench artifacts | `encodings/string-skip/benches/`, `examples/queries_demo.rs` |
| Original OnPair design | `encodings/onpair/SKIP_DESIGN.md` |
| OnPair paper | arXiv:2508.02280 |
| BitFunnel paper | SIGIR 2017 |

Source file references with line numbers are inlined throughout this doc.
