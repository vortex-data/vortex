# Point Functions

Point functions (`point_fn`) are the third class of operations in Vortex, sitting alongside
`scalar_fn` (element-wise) and `aggregate_fn` (reductions). They answer **point queries** on an
array: given a small input (a position, a value, or a few of each), produce a small output (a
scalar, an index, a search result) — *without materializing the array*.

## Motivation

`scalar_at(i)` is currently routed through `OperationsVTable::scalar_at`, and `search_sorted(v)`
is currently a generic blanket impl over `ArrayRef` that calls `execute_scalar` per probe
(see `vortex-array/src/search_sorted.rs:267-276`). Two consequences:

1. **Block-decoded encodings throw away work.** Every call to `scalar_at` on a `Pco`, `Fsst`,
   `Delta`, or `Zstd` array re-decodes the block containing that index. A `search_sorted` on a
   sorted compressed array performs `log(n)` probes, each decoding a fresh block. The TODO at
   `encodings/zstd/src/zstd_buffers.rs:463` ("maybe we should not support scalar_at, it is
   really slow") names this exact problem.

2. **No encoding-specific `search_sorted` pushdown.** The generic binary search calls
   `execute_scalar` per probe regardless of whether the encoding could answer the query in a
   fundamentally cheaper way (Dict by searching the dict directly; RunEnd by searching the
   values directly; Chunked by zone-map pruning).

3. **Per-probe `ExecutionCtx` construction.** `search_sorted.rs:269` calls
   `LEGACY_SESSION.create_execution_ctx()` *per binary-search probe*. For a 1M-row search,
   that is ~20 throwaway contexts per call.

`point_fn` is a peer subsystem to `scalar_fn` and `aggregate_fn` that fixes all three.

## Class membership

A point function is any operation matching the shape **`Array × O(1) input → O(1) output`**
that translates between value-space and position-space at a small number of points:

| direction | example | signature |
|---|---|---|
| position → value | `scalar_at` | `(arr, idx) → Scalar` |
| value → position | `search_sorted`, `rank`, `position_of` | `(arr, value) → SearchResult \| usize` |
| value-range → position-range | `range_search`, `count_in_range` | `(arr, lo, hi) → (idx, idx) \| usize` |
| positions → values | `take_scalars` (small N) | `(arr, &[idx]) → Vec<Scalar>` |

Members of the class share three structural properties:

1. **Sublinear cost target.** A point fn that degrades to `O(n)` on its input array is a failed
   point fn. This is the membership test: if an encoding cannot answer it in `o(n)` on
   compressed data (or at minimum reuse `o(n)` of work across calls), it does not belong here.
2. **Bounded I/O shape.** Inputs and outputs are `O(1)`-sized (or `O(k)` for batched
   variants), never array-shaped.
3. **Push through view encodings without materializing.** Every view encoding (Slice, Dict,
   RunEnd, Chunked, FoR, Sparse, Patched, Masked, Extension) must be able to rewrite a point
   fn call into a smaller point fn call on a child.

## Architecture

Three layers, cleanly separated:

```
┌──────────────────────────────────────────────────────────────┐
│ PointSession (public API)                                    │
│   know about workloads: batches, repeated access, prefetch   │
│   methods: scalar_at, search_sorted, range_search,           │
│            search_sorted_batch, take_scalars, rank,          │
│            position_of, count_in_range, …                    │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ PointRuntime / PointDispatch (the loop)                      │
│   caches (scalar + block), dispatches, recurses              │
│   one-shot variant (PointRuntime) stores no caches           │
│   session variant (PointSession) stores LRU caches           │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Kernels (stateless, per-encoding)                            │
│   ScalarAtKernel:     required                               │
│   SearchSortedKernel: required, has default (generic BS)     │
│   TakeScalarsKernel:  optional, default = loop               │
│   SearchSortedBatchKernel: optional, default = loop          │
└──────────────────────────────────────────────────────────────┘
```

### Kernel traits

Each kernel is its own trait, mirroring the layout of `scalar_fn/` (one trait per fn).

```rust
pub trait ScalarAtKernel<V: VTable> {
    fn execute<D: PointDispatch>(
        view: ArrayView<'_, V>,
        idx: usize,
        d: &mut D,
    ) -> VortexResult<Scalar>;
}

pub trait SearchSortedKernel<V: VTable> {
    fn execute<D: PointDispatch>(
        view: ArrayView<'_, V>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut D,
    ) -> VortexResult<SearchResult> {
        algorithms::generic_search_sorted(view, value, side, d)
    }
}
```

`TakeScalarsKernel` and `SearchSortedBatchKernel` follow the same shape with `&[usize]` /
`&[Scalar]` input and `Vec<_>` output, defaulting to a loop over the single-input kernel.

### `PointDispatch` trait

The single trait that kernels see. It is implemented by both `PointRuntime` (no cache) and
`PointSession` (with cache):

```rust
pub trait PointDispatch {
    fn scalar_at(&mut self, arr: &ArrayRef, idx: usize) -> VortexResult<Scalar>;
    fn search_sorted(&mut self, arr: &ArrayRef, v: &Scalar, side: SearchSortedSide)
        -> VortexResult<SearchResult>;

    /// HOF used by encodings that decode in blocks (Pco, Fsst, Delta, Zstd, BitPacked).
    /// Default impl just runs the decoder — no caching, no allocation.
    /// `PointSession` overrides to consult its block LRU.
    fn cached_block<B, F>(&mut self, _key: (ArrayId, BlockKey), decode: F)
        -> VortexResult<B>
    where
        B: Clone + Send + Sync + 'static,
        F: FnOnce() -> VortexResult<B>,
    {
        decode()
    }
}
```

### `PointRuntime` — one-shot, no cache stored

```rust
pub struct PointRuntime<'a> { ctx: &'a mut ExecutionCtx }
```

A single borrow. No allocation, no cache field. `cached_block` inherits the default no-op
impl, so encoding kernels that call it pay zero cost beyond the closure invocation.

### `PointSession` — caching, persists across calls

```rust
pub struct PointSession<'a> {
    ctx: &'a mut ExecutionCtx,
    scalar_cache: LruCache<(ArrayId, usize), Scalar>,
    block_cache:  LruCache<(ArrayId, BlockKey), Arc<dyn Any + Send + Sync>>,
}
```

Holds bounded LRUs for both scalar lookups and decoded blocks. The block cache key includes
an opaque `BlockKey` (per-encoding discriminator + block index) so multiple block-decoded
encodings can share the cache without collision.

### Public entry points on `ArrayRef`

```rust
impl ArrayRef {
    pub fn scalar_at(&self, idx: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar>;
    pub fn search_sorted(&self, v: &Scalar, side: SearchSortedSide,
                         ctx: &mut ExecutionCtx) -> VortexResult<SearchResult>;
    pub fn point_session<'a>(&'a self, ctx: &'a mut ExecutionCtx) -> PointSession<'a>;
}
```

One-shot methods construct a `PointRuntime` on the stack and discard it. `point_session`
returns a stateful object that caches across multiple calls.

## Per-encoding pushdown rules

Every encoding falls into one of these patterns. Most need only `ScalarAtKernel`; about a
third benefit from a custom `SearchSortedKernel`.

| pattern | encodings | `ScalarAtKernel` | `SearchSortedKernel` |
|---|---|---|---|
| Closed-form | `Constant`, `Null`, `Sequence` | O(1) inline | **override** O(1) |
| Direct read | `Primitive`, `Bool`, `ByteBool`, `VarBin`, `VarBinView`, `Decimal` | buffer read | default (generic BS) |
| Recursive view | `Slice`, `Masked`, `Extension`, `DateTime` | one `d.scalar_at(child, ...)` | **override** push to child |
| Multi-child structural | `Dict`, `RunEnd`, `Chunked` | bounce through 2 children | **override** structural shortcut |
| Monotonic transform | `FoR`, `decimal-byte-parts` | apply inverse to child value | **override** transform v, push |
| Patch overlay | `Sparse`, `Patched`, `Alp`, `BitPacked` | check patches first | default |
| Block decoded | `Pco`, `Delta`, `Fsst`, `Zstd`, `fastlanes/rle` | use `d.cached_block(...)` HOF | default |
| Non-monotonic | `ZigZag` | unwrap and recurse | default (cannot push) |
| Composite (no order) | `Struct`, `List`, `ListView`, `FixedSizeList`, `ParquetVariant` | compose from children | N/A |
| Selection (no order) | `Filter` | mask rank + recurse | default |
| Multi-part value | `datetime-parts` | combine parts | optional multi-stage override |

### Detailed pushdown rules — `SearchSortedKernel` overrides

- **Constant**: compare `v` to the constant, return `Found(0)` / `NotFound(0)` / `NotFound(len)`.
- **Sequence** (`start + i*step`): closed-form solve for `i = (v - start) / step`, check exact
  divisibility.
- **Slice**: search child, clamp result into `[offset, offset+len)`.
- **Extension** / **DateTime**: unwrap `v`, push to storage.
- **Dict** (sorted dict AND sorted codes): search small dict for `v` → code; then search codes
  for that code. `O(log dict + log n)` vs `O(log n × scalar_at)`.
- **RunEnd** (sorted `values`): search `values` for `v` → run index `r`; result position is
  `ends[r-1]` (or 0 if `r == 0`). `O(log num_runs)`.
- **Chunked** (cross-chunk monotonic): consult per-chunk min/max stats to find the candidate
  chunk; descend into that one chunk. Skips `O(num_chunks - 1)` work.
- **FoR**: subtract reference from `v`, push to encoded. `O(log n)` over smaller delta space.

### Detailed pushdown rules — leaves needing `cached_block`

- **Pco**, **Fsst**, **Delta**, **fastlanes/rle**: block-keyed cache, one decode per unique block.
- **Zstd**: whole-array "block" cached. Fixes the explicit TODO in `zstd_buffers.rs:463`.

## Procedures

Methods on `PointSession` that build on the kernels:

- `rank(v)` = `search_sorted(v, Right).to_index()`. Pure composition.
- `position_of(v)` = `search_sorted(v, Left).to_found()`. Pure composition.
- `search_range(lo, hi)` = two `search_sorted` calls. Pure composition; the session cache
  absorbs the shared descent work.
- `count_in_range(lo, hi)` = `let (l, h) = search_range(lo, hi); h - l`.

Procedures live on `PointSession` and never touch the vtable. If a future benchmark proves a
specific encoding (typically `Chunked`) needs a compound `search_range` shortcut that beats
the cache-amortized composition, the procedure can be promoted to its own optional kernel
with a per-encoding registry (the same pattern `scalar_fn` already uses) without disrupting
the existing API.

## Relationship to existing systems

- **`scalar_fn` / `aggregate_fn`**: `point_fn` is a peer. Same pattern (one trait per op,
  per-encoding overrides, optional defaults), different shape (point queries, not
  element-wise or reductions).
- **`Executable` / `execute()`**: `point_fn` is *not* a slot inside `execute`. The two systems
  share `ExecutionCtx` and the encoding registry but use different protocols. `execute` is
  iterative with the `ExecutionStep::ExecuteSlot` protocol for array-shaped output;
  `point_fn` is recursive with direct return for small outputs. The output-shape and
  call-frequency mismatch is the reason for the separation.
- **Parent kernels (`vortex-array/src/kernel.rs`)**: complementary. Parent kernels are for
  "child encoding handles its parent" fusion (the existing example in that module's
  documentation is RunEnd handling a Slice parent). Point fns are for "encoding answers a
  question about itself." A future iteration could add a `PointFnParentKernel` for
  fused-through-parent point queries; out of scope for the initial design.

## Migration plan

Staged across multiple PRs to keep each diff reviewable:

- **Phase 1 — Foundation:** new `point_fn/` module with traits, runtime, session,
  `cached_block` HOF. Port two encodings as proof-of-concept (`Slice` view + `Pco` block
  decoder). One benchmark demonstrating the perf win. No callers migrated; new API lives
  alongside `OperationsVTable::scalar_at` and the existing `SearchSorted` blanket.
- **Phase 2 — Structural encodings:** port `Dict`, `RunEnd`, `Chunked`, `Constant`,
  `Sequence`, `Extension`, `DateTime`, `Masked`, `FoR`. Add `SearchSortedKernel` overrides
  where they pay off. Per-encoding benches.
- **Phase 3 — Remaining block decoders:** port `Fsst`, `Delta`, `Zstd`, `Alp`,
  `fastlanes/rle`, `BitPacked`. The `Zstd` TODO is resolved. Benches showing the win on
  compressed search_sorted.
- **Phase 4 — Leaves:** port `Primitive`, `Bool`, `ByteBool`, `VarBin`, `VarBinView`,
  `Decimal`, `ZigZag`, `Sparse`, `Patched`. Mostly mechanical.
- **Phase 5 — Composites:** port `Struct`, `List`, `ListView`, `FixedSizeList`,
  `ParquetVariant`, `Filter`, `datetime-parts`. Composites are scalar_at-only (no
  `search_sorted` — no total order).
- **Phase 6 — Caller migration:** update `vortex-scan`, `vortex-layout`,
  `vortex-datafusion`, `vortex-duckdb` to call the new API. Mostly mechanical.
- **Phase 7 — Cleanup:** delete `OperationsVTable`, delete top-level `search_sorted.rs`,
  delete `Operations` shims in `array/erased.rs` and `array/mod.rs`.
- **Phase 8 — Comprehensive benchmark suite + report:** end-to-end perf numbers on
  TPC-H/ClickBench query subsets dominated by point queries.

Each phase is independently shippable. After Phase 1 the new API is callable; after Phase 6
all callers use it; after Phase 7 the old code is gone.

## Open questions

1. **`PointFnParentKernel`?** Should point fns participate in the existing parent-kernel
   infrastructure (`vortex-array/src/kernel.rs`) so that, e.g., `Slice(RunEnd(...))` can be
   answered via a fused parent kernel rather than the recursive `Slice → RunEnd` descent? The
   recursive path is already correct; this is purely an optimization that may or may not be
   measurable.
2. **Default LRU capacities?** `PointSession` needs sensible defaults for both the scalar
   cache and the block cache. Initial guess: scalar=64 entries, block=8 entries. Real
   numbers from Phase 1 benches.
3. **Promote `search_range` to a kernel?** Today it is a procedure. If Phase 2/3 benches show
   `Chunked.search_range` significantly beats two cached `search_sorted` calls (because
   interior chunks contribute trivially), promote it.
4. **Iterative scheduler?** The recursive descent is fine for the 2–4 level array trees Vortex
   sees in practice. If real workloads produce deeper nesting (e.g., heavily-chunked
   chunked-of-dict-of-runend stacks) we may need to bound recursion. Not part of the v1 design.
