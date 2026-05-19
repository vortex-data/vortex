# string-skip → Vortex Integration Plan

**Status:** Phase A (standalone crate) shipped. Phase B (Vortex integration) planned in detail.

**Owner / SME:** [TBD]
**Estimated calendar time:** 4–5 weeks single-engineer
**Estimated LOC for Phase B:** ~2,050 new + ~30 LOC YAML changes

---

## Executive summary

We have shipped a standalone Rust crate, `string-skip`, that provides chunk-level
skip indexes for string columns. It supports range / equality / prefix / suffix
/ substring / wildcard / length / null / set predicates with sound (no false
negative) pruning across 14 distinct query classes. The algorithm is novel:
BitFunnel-style frequency-conscious code-bigram blooms with sound cover
enumeration for substring matching. Benchmarked across 18 real columns
(ClickBench, TPC-H, FineWeb, OnPair-paper datasets). No library equivalent
exists in the wild.

Phase A is portfolio-quality and can be released independently. Phase B wires
it into Vortex as a new layout encoding, sitting alongside (not on top of)
`ZonedLayout` and `ChunkedLayout`.

---

## 1. Current state — Phase A (shipped)

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

---

## 2. Phase B — Vortex integration

### 2.1 Architecture overview

Mirror `ZonedLayout` (`vortex-layout/src/layouts/zoned/`), not
`FileStatsLayoutReader`. The latter is a *file-level* decorator;
chunk-level pruning belongs *inside* the layout tree.

Add a new layout encoding `SkipIndexLayout` that wraps a chunked data
layout. Two children:

- child 0 — `LayoutChildType::Transparent("data")`: the wrapped chunked
  data layout (typically `ChunkedLayout`).
- child 1 — `LayoutChildType::Auxiliary("skip")`: a `FlatLayout` holding
  the serialized skip-index blob.

The skip blob is a single `bincode`-encoded payload:

```rust
SkipIndexBlob {
    ubiq: UbiquitousBigrams,    // column-shared
    tiers: BigramTiers,         // column-shared
    dict_ref: DictRef,          // points into the column dict
    chunks: Vec<PerChunkSkip>,  // 1 entry per data chunk
}
PerChunkSkip {
    stats: ChunkStats,
    presence: DictPresence,
    bloom: Option<HybridBloom>,
    tiered: Option<TieredBloom>,
}
```

Total blob size is hundreds of KiB for millions of rows — small enough to
load once and cache.

### 2.2 Phase B is split into milestones

| Milestone | Deliverable | LOC | Calendar |
|-----------|-------------|-----|----------|
| **B.0** | OnPair → `TokenDict` adapter | ~55 | 1–2 days |
| **B.1** | `SkipIndexLayout` (read + write paths) | ~830 | 2–3 weeks |
| **B.2** | Sort flag + versioning + back-compat | ~640 | 1 week (parallel with B.1) |
| **C.1** | CI gates, soundness fuzz extension, golden fixtures, benches | ~430 + 30 lines YAML | 3–5 days |
| **Total** | | **~2,050** | **4–5 weeks** |

### 2.3 Suggested first PR (de-risk before big work)

**Land B.0 only**: the OnPair → `TokenDict` adapter (~55 LOC) plus an integration
test in `encodings/onpair/tests/` that builds skip indices on a real parquet
column via the new crate and verifies pruning rates match
`examples/queries.rs`. **No layout/file-format changes.**

This is 1 day of work and proves the standalone crate semantics carry
through to real OnPair data before committing to the 4-week layout work.

---

## 3. Detailed milestone plans

### 3.1 B.0 — OnPair adapter

**File:** `encodings/onpair/src/skip_glue.rs` (new)

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
- Max token size 16 bytes — irrelevant to `TokenDict`
- Dict size capped at 4096 for `dict-12` preset — fits in u16

**Tests:** round-trip vs direct `dv.dict_table` lookups; integration test
loading ClickBench URL parquet and asserting `chunk_might_match` matches
the standalone `queries.rs` demo numbers.

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

### 3.3 B.1 — `SkipIndexLayout`

**Files (all new, in `encodings/string-skip/src/layout/`):**

| File | Purpose | LOC |
|------|---------|-----|
| `mod.rs` | `VTable` impl, `SkipIndexLayout` struct, `SkipIndexMetadata` | ~120 |
| `reader.rs` | `SkipIndexReader` implementing `LayoutReader` | ~180 |
| `writer.rs` | `SkipIndexStrategy` + `SkipIndexAccumulator` | ~250 |
| `blob.rs` | bincode (de)serialization | ~60 |
| `expr.rs` | Vortex `Expression` → `string_skip::Pred` translator | ~220 |
| (planner glue) | `split_conjunction` walk + chunk-set intersection | ~80 |

**`SkipIndexReader` (read path):**

```rust
pub struct SkipIndexReader {
    layout: SkipIndexLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    chunk_offsets: Vec<u64>,                          // mirror ChunkedReader
    blob: OnceCell<Arc<SkipIndexBlob>>,               // lazy-loaded
    blob_loader: parking_lot::Mutex<Option<BoxFuture<'static, VortexResult<Arc<SkipIndexBlob>>>>>,
}

impl LayoutReader for SkipIndexReader {
    fn pruning_evaluation(&self, row_range, expr, mask) -> VortexResult<MaskFuture> {
        // 1. Always forward to data child.
        let data_eval = self.data_child()?.pruning_evaluation(row_range, expr, mask.clone())?;
        // 2. Translate `expr` → list of `(FieldPath, Pred)`.
        let Some(preds) = expr::to_skip_preds(expr) else { return Ok(data_eval); };
        // 3. Load blob (cheap, one segment).
        let blob_fut = self.load_blob();
        let chunk_ranges = self.chunk_ranges_in_mask(row_range);
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

**`SkipIndexStrategy` (write path):**

```rust
pub struct SkipIndexStrategy {
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

impl LayoutStrategy for SkipIndexStrategy {
    fn write_stream(&self, stream, ctx, eof) -> VortexResult<LayoutRef> {
        let data_eof = eof.split_off();                                  // sequence: data first
        let mut acc = SkipIndexAccumulator::new(&self.options);
        let stream = stream.map(|chunk| { acc.push_chunk(&chunk)?; Ok(chunk) });
        let data_layout = self.chunk_strategy.write_stream(stream, ctx, data_eof)?;
        let blob = acc.finish()?;
        let blob_bytes = bincode::serialize(&blob)?;
        let skip_layout = self.skip_strategy.write_stream(/* blob_bytes */, ctx, eof)?;
        Ok(SkipIndexLayout::new(data_layout, skip_layout).into_layout())
    }
}
```

Mirrors `ZonedStrategy::write_stream` at `vortex-layout/src/layouts/zoned/writer.rs:60-181`.

**Expression → `Pred` translator (`expr.rs`):**

Vortex already has all the needed expression types as scalar fns:
- `Binary` (with `Operator::{Eq, NotEq, Lt, Lte, Gt, Gte, And, Or}`)
- `Between` (with strict/non-strict bounds)
- `Like` (with `LikeOptions { negated, case_insensitive }`)
- `IsNull`, `IsNotNull`, `Not`
- `ListContains` (for `IN (...)`)
- `GetItem` (column reference) + `Root` + `Literal`

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
| `Not(IsNull(col))` | `Pred::IsNotNull` |
| `ListContains(lit(list), col)` | `Pred::InSet(elements)` |
| anything else | `None` (planner falls back to full scan) |

**LIKE pattern parser** — small byte tokenizer that classifies into 7 shapes:
`Eq | Prefix | Suffix | Contains | PrefixSuffix | SingleWildcard | MultiFragment`.
Handles SQL `\` escape.

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

**Open decisions for B.1:**

1. **Encoding-shared vs encoding-local layout?** Recommend: encoding-local
   `OnPairSkipIndexLayout` for v1; generic v2 if another encoding gets a
   skip-index adapter.
2. **Codec location?** Recommend: live next to `OnPairSkipIndexLayout`
   inside `vortex-onpair`, *not* inside the pure `string-skip` crate
   (keeps `string-skip` Vortex-free).
3. **Registry registration?** Recommend: register
   `SkipIndexLayoutEncoding` in `LayoutSession::register` directly.

### 3.4 B.2 — Sort flag + versioning + back-compat

**Sort flag (file: `vortex-file/src/writer.rs`, ~120 LOC of glue):**

```rust
// new public enum
pub enum SortStrategy {
    Preserve,         // default
    Lexicographic,
    LengthThenLex,
    AutoDetect,       // record IsSorted flag iff already sorted; no reorder
}

// VortexWriteOptions gains:
sort_rows: Option<(FieldPath, SortStrategy)>,
```

Flow: `VortexWriteOptions::write` materializes the input stream → computes
permutation via existing `compare` kernels on chosen field → applies
`take` to all columns → re-streams in `row_block_size` chunks → sets
`Stat::IsSorted` / `Stat::IsStrictSorted` on `FileStatistics`.

Must happen **before** `TableStrategy` splits columns (since the sort
applies across all columns simultaneously).

Reuses:
- `vortex-array/src/aggregate_fn/fns/is_sorted/mod.rs:60` (auto-detect path)
- `vortex-array/src/expr/stats/mod.rs:51` (`Stat::IsSorted` / `IsStrictSorted`)

Reader-side: new `is_sorted_on(&FieldPath) -> bool` on `ChunkedReader`
that consults `FileStatistics`. Range pruning becomes binary search over
chunk min/max when sorted.

**Versioning (file: `encodings/onpair/src/skip/codec.rs`, ~205 LOC):**

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
    UnsupportedMagic,                          // not ours; ignore
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
| Magic mismatch | Hard error — corruption |

**Migration tests (~80 LOC):** no-payload, future-version, truncated header,
foreign-magic.

**File-format version (`vortex-file::VERSION`) is NOT bumped** —
skip-index versioning is encoding-local.

### 3.5 C.1 — Testing + CI

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

## 4. Critical-path decisions (need user input before code starts)

1. **Encoding-shared vs encoding-local layout?**
   - Recommended: encoding-local `OnPairSkipIndexLayout` for v1.
2. **Codec location?**
   - Recommended: `vortex-onpair` for now; hoist if FSST gets a v2 adapter.
3. **Sort default?**
   - Recommended: off by default (back-compat), opt-in via `VortexWriteOptions::with_sort_rows`.
4. **Registry registration?**
   - Recommended: directly in `LayoutSession::register`.
5. **Drop existing `encodings/onpair/src/skip.rs`?**
   - Defer to a separate cleanup PR after B.0 lands.

---

## 5. Consolidated risks

| Risk | Severity | Mitigation |
|------|---------:|------------|
| `Cast` wrappers (utf8↔utf8) break translator | Med | Peel no-op casts in `match_col` helper |
| `SequencePointer` ordering: skip blob can't be read before data commits | High | Mirror `ZonedStrategy::eof.split_off()` pattern exactly |
| Sort materialization cost defeats streaming on huge datasets | Med | Document; `AutoDetect` provides cheap streaming path |
| `StatsCatalog` is per-field, not per-chunk | Low | Bypass it; our `Pred` translation calls `chunk_might_match` directly |
| `Stat::SortedOn(field)` flatbuffer schema change | Med | Reuse per-array `Stat::IsSorted` + `FileStatistics` field; don't add new `Stat` variant |
| Layout id collision | Low | Reserve range; register in `LayoutSession` once |
| Golden fixtures drift with intentional tuning | Low | `regen-goldens` CLI; require PR-description justification |
| `LayoutReader::pruning_evaluation` shape mismatch | Low | Returns `MaskFuture` — confirmed compatible |
| `Layout::metadata()` not suited for blob-sized skip data | Low | Use auxiliary `FlatLayout` segment (precedent: `ZonedLayout`) |

---

## 6. Done = ship criteria

**B.0 done when:**
- `OnPairDictView` adapter compiles and round-trips against `dv.dict_table`
- Integration test in `encodings/onpair/tests/` loads ClickBench URL parquet
  and verifies pruning rates match `string-skip/examples/queries_demo.rs`
  within ±1pp
- Existing `encodings/onpair/examples/queries.rs` still runs (no regression)

**B.1 done when:**
- `SkipIndexLayout` registered in `LayoutSession`
- Round-trip test: write parquet → read parquet → assert chunk pruning happens
  for `Eq`, `Like 'p%'`, `Like '%x%'`, `BETWEEN`, `IS NULL`
- Soundness invariants hold under `proptest` × 1024 cases at release
- CodSpeed regression check passes (no >5% slowdown on hot paths)

**B.2 done when:**
- `VortexWriteOptions::with_sort_rows` available
- Reading a v0 (pre-skip-index) file works unchanged
- Reading a future-version skip-index payload degrades gracefully (debug log + full scan)
- Versioning byte layout has a "never-change-these-constants" test

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
- Variable chunk-size handling (assume fixed `row_block_size = 8192`)
- BitFunnel "higher rank rows" multiplexing (research direction)
- Ribbon/XOR filter backends (future perf optimization if blooms saturate)
- Anti-substring predicates (`NOT LIKE '%x%'`) — rarely prunes anyway
- Case-insensitive `ILIKE` — needs pre-folded bloom variant

---

## 8. References

| Doc | Path |
|-----|------|
| Algorithm design | `encodings/string-skip/DESIGN.md` |
| Crate API | `encodings/string-skip/README.md` |
| Standalone bench artifacts | `encodings/string-skip/benches/`, `examples/queries_demo.rs` |
| Original OnPair design | `encodings/onpair/SKIP_DESIGN.md` |
| OnPair paper | arXiv:2508.02280 |
| BitFunnel paper | SIGIR 2017 |

Source file references with line numbers are inlined throughout this doc.
