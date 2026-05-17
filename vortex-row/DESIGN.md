# vortex-row design

Audience: a Vortex reviewer who already knows Apache Arrow's `arrow-row` and is reading this
crate for the first time. This document captures the architecture, byte format, per-encoding
kernel design, performance characteristics, and the PR-split plan that lands this work into
mainline Vortex.

## Overview

`vortex-row` converts N columnar Vortex arrays into a single row-oriented array of byte
keys: a `ListView<u8>` whose `i`-th element is the concatenation of the encoded bytes for
`cols[0][i], cols[1][i], ..., cols[N-1][i]`. The encoded form is *byte-comparable*: for any
two row indices `i`, `j` and the same input columns,

```text
memcmp(row_i, row_j)  ==  tuple_cmp((cols[0][i], cols[1][i], ...),
                                    (cols[0][j], cols[1][j], ...))
```

where the tuple comparison uses the per-column ordering specified by [`SortField`]. This is
the same contract as `arrow-row::RowConverter`, lifted onto Vortex's columnar representation
and dispatched through Vortex's `ScalarFn` machinery so encodings can short-circuit
canonicalization with per-encoding kernels.

There are two scalar functions in this crate. `RowSize` computes per-row byte sizes for the
encoded form across N input columns and returns them as a `Struct { fixed: U32, var: U32 }`.
`RowEncode` does the actual encoding and returns a `ListView<u8>`. Both are variadic
(`Arity::Variadic { min: 1, max: None }`) and parameterized on per-column `SortField`s. The
top-level user entry point is `convert_columns(cols, fields, ctx) -> ListViewArray`.

`RowEncode` produces the encoded array in one left-to-right pass over the input columns,
re-using the `ListView` sizes array as the per-row write cursor. This means there is no
separate "build the output" step after encoding: when the last column's encoder returns,
the accumulator is the final array. Fast paths in `Constant`, `Dict`, `Patched`, and the
inventory registry (currently populated by `BitPacked`, `FoR`, and `Delta` from
`vortex-fastlanes`, and `RunEnd` from `vortex-runend`) write directly into this shared
accumulator and skip canonicalization entirely.

## API surface

Top-level types live in `vortex-row/src/lib.rs` and the four mod files it re-exports from.
The minimal public surface is:

### `SortField` and `RowEncodeOptions` (`options.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SortField {
    pub descending: bool,
    pub nulls_first: bool,
}

impl SortField {
    pub fn new(descending: bool, nulls_first: bool) -> Self;
    pub fn non_null_sentinel(&self) -> u8;  // always 0x01
    pub fn null_sentinel(&self) -> u8;      // 0x00 if nulls_first else 0x02
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowEncodeOptions {
    pub fields: Vec<SortField>,
}
```

`SortField` is `Copy` and 2 bytes wide. `RowEncodeOptions` holds one `SortField` per input
column, in left-to-right order.

### `RowSize` and `RowEncode` (`size.rs`, `encode.rs`)

The two `ScalarFnVTable` types. Both have `Options = RowEncodeOptions`.

```rust
#[derive(Clone, Debug)]
pub struct RowSize;

impl ScalarFnVTable for RowSize {
    type Options = RowEncodeOptions;
    fn id(&self) -> ScalarFnId;  // "vortex.row_size"
    fn arity(&self, _: &Self::Options) -> Arity;  // Variadic { min: 1, max: None }
    fn return_dtype(&self, _: &Self::Options, _: &[DType]) -> VortexResult<DType>;
    fn execute(&self, opts: &Self::Options, args: &dyn ExecutionArgs,
               ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
    // ... id / serialize / deserialize / child_name / is_null_sensitive / is_fallible
}

#[derive(Clone, Debug)]
pub struct RowEncode;

impl ScalarFnVTable for RowEncode {
    type Options = RowEncodeOptions;
    fn id(&self) -> ScalarFnId;  // "vortex.row_encode"
    // identical surface, return_dtype = List<U8>
}
```

`RowSize`'s output is a `StructArray` with two fields: `fixed: U32` and `var: U32`. The
total per-row size equals `fixed + var`. The `fixed` field is *always* a `ConstantArray<u32>`
holding the sum of the constant widths of fixed-width columns (sentinel + value bytes,
recursively for `Struct`/`FixedSizeList`). The `var` field is `ConstantArray(0u32, nrows)`
when the input has no variable-length columns, and a `PrimitiveArray<u32>` of length `nrows`
otherwise. This split lets downstream consumers cheaply detect fixed-width inputs without
walking the per-row sizes.

### Kernel traits (`size.rs`, `encode.rs`)

```rust
pub trait RowSizeKernel: VTable {
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: SortField,
        sizes: &mut [u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;
}

pub trait RowEncodeKernel: VTable {
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;
}
```

Returning `Ok(None)` declines the kernel and falls through to canonicalization. The unusual
`mutate &mut [u8]` signature (rather than returning an `ArrayRef`) is the load-bearing
design decision; see "Per-encoding kernels" below.

### `RowEncodeRegistration` (`registry.rs`)

The inventory entry that downstream crates submit:

```rust
pub type DynSizeFn =
    fn(&ArrayRef, SortField, &mut [u32], &mut ExecutionCtx) -> VortexResult<Option<()>>;

pub type DynEncodeFn = fn(
    &ArrayRef,
    SortField,
    &[u32],
    &mut [u32],
    &mut [u8],
    &mut ExecutionCtx,
) -> VortexResult<Option<()>>;

pub struct RowEncodeRegistration {
    pub id: fn() -> ArrayId,
    pub size: DynSizeFn,
    pub encode: DynEncodeFn,
}

inventory::collect!(RowEncodeRegistration);

pub fn lookup(id: &ArrayId) -> Option<(DynSizeFn, DynEncodeFn)>;
```

The `id` field is a function pointer so the `ArrayId` (which interns its string at runtime)
can be materialized lazily on first lookup; the resulting `(size, encode)` pair is cached
in a `OnceLock<HashMap<ArrayId, ...>>`.

### `convert_columns` (`convert.rs`) and `initialize` (`lib.rs`)

```rust
pub fn convert_columns(
    cols: &[ArrayRef],
    fields: &[SortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray>;

pub fn compute_row_sizes(
    cols: &[ArrayRef],
    fields: &[SortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>;

pub fn initialize(session: &VortexSession);
```

`initialize` registers `RowSize` and `RowEncode` on the session's `scalar_fns` registry.
Row-encoding is not part of the default session because most pipelines don't need it; opt
in once at session construction.

## Pipeline (the five-phase walk)

`RowEncode::execute` performs five phases. All state is stack-local until the final
`ListViewArray::new_unchecked` call. The flow is identical for every input regardless of
encoding; per-encoding kernels are invoked from within phases 1 and 4.

### Phase 1: classify + size pass

Iterate the input columns left-to-right, classifying each via `codec::row_width_for_dtype`
into:

- `RowWidth::Fixed(w)` for `Null`, `Bool`, `Primitive`, `Decimal`, and any `Struct`/
  `FixedSizeList`/`Extension` whose recursive descent is fixed.
- `RowWidth::Variable` for `Utf8`/`Binary` and any composite that recurses through a
  variable-width field.

Build three stack accumulators in lockstep:

- `fixed_per_row: u32`: running sum of fixed-width contributions.
- `var_lengths: Option<Vec<u32>>`: allocated lazily on the first varlen column; per-row
  byte contributions for varlen columns are accumulated into this slice via
  `size::dispatch_size`. `dispatch_size` tries each in-crate kernel
  (`Constant`/`Dict`/`Patched`), then the inventory registry, then falls back to
  `codec::field_size` on a canonical materialization.
- `col_kinds: Vec<ColKind>`: per-column tag with each column's within-row prefix and a
  `before_varlen: bool` flag (true if no varlen column has been seen yet).

The `before_varlen` flag is what enables the "arithmetic-write" lane in Phase 4: every row's
write position for a fixed column that precedes all varlen columns is a closed-form function
of `i`, no cursor lookup needed.

### Phase 2: totals + buffer

Compute the encoded total bytes:

```text
total = nrows * fixed_per_row + sum(var_lengths)
```

Both halves are validated to fit in `u32` (the `ListView` offset type). Allocate
`BufferMut<u8>` with `total` capacity, then `set_len(total)` *without* zero-initializing.
This is sound because every byte in `[0, total)` is written by the encoders before the
buffer is read out: non-null fixed-width slots are sentinel + value; null fixed-width slots
are sentinel + explicit zero-fill; varlen partial blocks zero-pad explicitly inside the
varlen encoder; null `Struct`/`FixedSizeList` bodies are zero-filled after the child encoders
run. Skipping the memset of a multi-MB buffer is meaningful for varlen-heavy inputs.

### Phase 3: per-row offsets construction

Build a single `listview_offsets: Vec<u32>` of length `nrows`, where
`listview_offsets[i] = i * fixed_per_row + var_prefix[i]`. There are two cases:

- *Pure-fixed* (no varlen columns): `offsets[i] = i * fixed_per_row` for all rows. Build via
  a raw-pointer write loop that LLVM auto-vectorizes; the multiplications are already
  proven non-overflowing because `total` (computed in Phase 2) fits in `u32`.
- *Mixed*: accumulate `acc` over `var_lengths`, pushing `i * fixed_per_row + acc` for each
  row and `acc` itself for the arithmetic lane's separate `var_prefix_for_arith` buffer.
  The arithmetic-lane prefix is only built if at least one `Fixed { before_varlen: true }`
  column exists; otherwise we skip the second allocation.

A separate `row_cursors: Option<Vec<u32>>` of length `nrows` is created when any varlen
column exists. Each cursor starts at `prefix_at_first_varlen` — the within-row byte offset
of the first cursor-encoded column — so `listview_offsets[i] + cursors[i]` lands at the
correct position when the cursor-write encoder begins.

### Phase 4: encode pass (two dispatch lanes)

For each input column:

- If it's classified `ColKind::Fixed { before_varlen: true, .. }`, dispatch through
  `dispatch_encode_fixed_arith`. This is the **arithmetic-write** lane: the write position
  for row `i` is `i * row_stride + col_prefix + var_prefix[i]`, a pure function of `i` and
  three constants. The encoder never touches `cursors`. Primitive and Constant inputs are
  short-circuited from this dispatcher directly; everything else canonicalizes.

- Otherwise (varlen, or fixed-after-varlen), dispatch through `dispatch_encode`. This is the
  **cursor-write** lane: the write position for row `i` is `offsets[i] + cursors[i]`, and
  the encoder advances `cursors[i]` by the bytes written. This is the path that the
  per-encoding kernel traits target — `Constant`, `Dict`, `Patched`, and inventory-registered
  encodings (currently `BitPacked`) all decline the size-pass canonicalization and walk
  their compressed representation directly to write the row bytes.

`dispatch_encode` tries `Constant`, `Dict`, `Patched`, then `registry::lookup`, then falls
back to `codec::field_encode` on the canonical materialization. The dispatch order is fixed
in source — the asymmetry between "in-crate downcast" and "inventory registry" exists
because the orphan rule prevents an out-of-crate impl of the `RowEncodeKernel` trait on an
out-of-crate type (the encoding's `VTable` lives outside `vortex-row`).

### Phase 5: ListView output

The final array assembly:

- `elements`: `PrimitiveArray::<u8>` wrapping the encoded buffer.
- `offsets`: `PrimitiveArray::<u32>` over `listview_offsets`.
- `sizes`: `ConstantArray(fixed_per_row)` if pure-fixed; otherwise re-write `var_lengths`
  in place to be `var_lengths[i] + fixed_per_row` (no second allocation) and wrap it.

`ListViewArray::new_unchecked(elements, offsets, sizes, NonNullable)` constructs the result.
The unchecked constructor skips the row-by-row `validate` that walks the offset/size arrays
to verify monotonicity and bound-fitting; we know the invariants hold by construction
(Phase 2 validates `total <= u32::MAX`, Phase 3 maintains `offsets[i] + sizes[i] <= total`
and disjoint per-row slices). Skipping this validation is a meaningful win for inputs with
many rows.

## Byte encoding rules

The encoded format produces lexicographically byte-comparable rows. The general structure
for any value is `[sentinel][value-bytes...]`, where the sentinel is the only byte that
varies for nulls vs non-nulls. `descending=true` XORs the value bytes with `0xFF` after
the natural encoding; the sentinel byte is *never* inverted, so nulls keep their requested
position relative to non-nulls.

The sentinel values:

- non-null: `0x01`
- null, `nulls_first=true`: `0x00` (sorts before non-nulls)
- null, `nulls_first=false`: `0x02` (sorts after non-nulls)

Per logical type:

### `Null`

One byte per row: the null sentinel. No value bytes.

### `Bool`

Two bytes per row: sentinel + `0x01` (false) or `0x02` (true). For descending, the value
byte is XORed with `0xFF`. Null rows write sentinel + `0x00`.

### `Primitive` (sign-flip top byte for signed; float bit-fiddle)

One sentinel byte + `byte_width(ptype)` value bytes.

- Unsigned: big-endian bytes (already sort-correct).
- Signed: big-endian bytes with the top bit XORed by `0x80` (flips sign-bit ordering so
  negatives sort before non-negatives).
- Floats (`f16`, `f32`, `f64`): bit-pattern big-endian, sign-aware mask. If the high bit
  is 0 (non-negative), XOR the top bit (so `+0.0` lands above all negatives). If the high
  bit is 1 (negative), XOR all bits (so larger-magnitude negatives sort first).

Example for `f32`:
```rust
let bits = self.to_bits();
let mask: u32 = if (bits >> 31) == 0 { 0x80000000 } else { 0xFFFFFFFF };
let bytes = (bits ^ mask).to_be_bytes();  // value bytes, descending applied after
```

For descending, the resulting value bytes are XORed with `0xFF`.

### `Decimal`

Treated as a signed integer of width `smallest_decimal_value_type(dt).byte_width()`. `i256`
bails with an explicit "not yet implemented" error; everything from `i8` to `i128` uses the
same sign-bit-flip path as `Primitive`.

### `Utf8` / `Binary` (32-byte block + continuation)

This is the only canonical variant where size depends on data. Format:

- 1 byte sentinel.
- 0-length value: 1 trailing terminator byte (`0x00`, or `0xFF` for descending).
- Otherwise: zero or more 33-byte full blocks (32 value bytes + continuation marker
  `0xFF`), followed by one final 33-byte block whose last byte is the partial-block
  length (`0..=32`) and whose value tail is zero-padded.

The trickiest case: when the input length is an exact multiple of 32, the encoder emits
`(full_blocks - 1)` full blocks (continuation = `0xFF`) plus one final block whose
continuation byte is `32` (not `0xFF`). This preserves the byte-comparability contract:
without the final-32 marker, a 32-byte string would compare equal to a 33-byte string
that shares the same first 32 bytes.

For descending: every value byte and the continuation byte are XORed with `0xFF`. The
ascending continuation `0xFF` becomes `0x00`; the partial-length byte `n` becomes `n ^
0xFF`. The descending fast path XOR-copies 32 bytes per block using u64-wide reads and
writes (LLVM auto-vectorizes this into SIMD on x86).

### `Struct` (recursive concat)

Sentinel + per-field encoded bytes, in declared field order. Null struct rows still write
their full body width but zero-fill the body bytes after the per-field encoders run, so
the encoded form depends only on the sentinel for null rows.

### `FixedSizeList`

Sentinel + `n` copies of the element-type encoded bytes. Fixed iff the element type is
fixed. Null rows zero-fill the body, same as struct.

### `Extension`

Delegates to the storage array's encoder. The sentinel for `Extension` itself is *not*
emitted; the storage encoder writes its own.

### Unsupported

- `List`: per-row encoded length is data-dependent in a way the current single-pass
  cursor model doesn't handle (each row needs its own intra-block byte count plus
  recursive element encoding). Bails with an explicit error in `field_size` /
  `field_encode`.
- `Variant`: no defined ordering for heterogeneous values; bails permanently.
- `Union`: bails (no use case yet).
- `Decimal256`: bails with "not yet implemented" — the i256 type exists in Vortex but the
  byte encoding isn't wired up.

### Descending interaction with nulls

`descending` only inverts value bytes, never the sentinel. This means
`SortField { descending: true, nulls_first: true }` puts nulls before non-nulls regardless
of value-order direction. To put nulls last in descending order, pass `nulls_first: false`.

## Per-encoding kernels

### The two-trait design

A row encoder is conceptually three values: a per-row size contribution, a per-row write
into a shared byte buffer, and a position-tracking cursor that lets multiple encoders
collaborate without computing per-row offsets repeatedly. The crate splits this into two
traits — `RowSizeKernel` and `RowEncodeKernel` — because the size pass runs before any
buffer is allocated, while the encode pass runs after offsets are known.

Both traits' methods mutate shared buffers rather than return `ArrayRef`. This is the
load-bearing design decision. Vortex has a generic `ExecuteParentKernel` mechanism that
returns `Option<ArrayRef>` from a kernel implementation, which is the idiomatic way for
encodings to plug into compute functions. But that signature does not fit the row encoder:

- The output is a *single* `ListView<u8>` shared across all input columns. If each kernel
  returned its column's encoded `ArrayRef`, the variadic dispatch loop would have to
  concatenate them, which means allocating each column's bytes separately and then
  walking each per-row to glue them together. That's exactly the row-by-row work the
  encoder is trying to avoid.
- Allowing the kernel to write into the shared buffer means each column's encoder can
  see (and advance) the per-row cursor that the next column will read. The cursor doubles
  as the `sizes` array of the final `ListViewArray`, so no separate accumulator is needed.

So the kernel methods take `(offsets: &[u32], cursors: &mut [u32], out: &mut [u8])` and
write directly. The encoder owns the buffer; the kernel just shows up with the column data
and writes bytes.

### The two dispatch lanes (in-crate downcast + inventory registry)

`Constant`, `Dict`, and `Patched` are defined in `vortex-array` and their kernels live in
`vortex-row/src/kernels/`. The orphan rule allows this because `vortex-row` defines the
trait, so it can implement the trait for an external type. Dispatch is a direct downcast:

```rust
if let Some(view) = col.as_opt::<Constant>()
    && Constant::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
{
    return Ok(());
}
```

`BitPacked` lives in `vortex-fastlanes`, which is a downstream crate. The orphan rule
forbids an out-of-crate impl of `RowEncodeKernel` for `BitPacked` (both trait and type
are out-of-crate to whichever crate would write the impl). The workaround is the
inventory registry: `vortex-fastlanes` submits a `RowEncodeRegistration` at link time, and
the dispatch loop looks up `(size_fn, encode_fn)` function pointers by `ArrayId`:

```rust
if let Some((_, encode_fn)) = registry::lookup(&col.encoding_id())
    && encode_fn(col, field, offsets, cursors, out, ctx)?.is_some()
{
    return Ok(());
}
```

The asymmetry isn't free: function-pointer indirection prevents inlining of the kernel
body into the dispatch loop. The trade-off is that downstream encoding crates don't need
to depend on `vortex-row` as more than a small trait-and-registry crate. In practice the
indirection cost is small relative to per-row work.

### Per-kernel notes

#### `Constant` (in-crate, `kernels/constant.rs` and `encode.rs::encode_constant_arith`)

For the cursor lane, the kernel encodes the scalar value once into a small stack buffer,
then `copy_nonoverlapping`s the bytes into each row's slot.

For the arithmetic lane (`encode_constant_arith` in `encode.rs`), the encoded length drives
specialization: lengths 2, 5, 9, and 17 bytes (corresponding to `bool`/`i8`, `i32`/`u32`/
`f32`, `i64`/`u64`/`f64`, and `i128`/decimals respectively) hoist the encoded scalar into
register-sized values (`u16`, `u32`-plus-tail, `u64`-plus-tail, `u128`-plus-tail) before
the loop and emit direct unaligned word stores per row. This out-performs
`copy_nonoverlapping` for small fixed lengths because the compiler emits a real `memcpy`
call rather than inlining the 1- or 2-word store sequence. Lengths outside the
specialization set fall through to a `copy_nonoverlapping` loop.

#### `Dict` (in-crate, `kernels/dict.rs`)

Encode each unique value once into a small per-value contiguous byte buffer keyed by code
(via the same `dispatch_size`/`dispatch_encode` recursion as the top-level encoder), then
`memcpy` per-row by indexing the codes array. The per-unique-value cost is amortized over
the dictionary cardinality rather than the row count, so the win grows with
`row_count / unique_count`.

The kernel declines (returns `Ok(None)`) when `values().len() > codes().len()` — i.e.
when the dictionary is larger than the data, in which case canonicalization is cheaper
than the kernel's per-value setup.

#### `RunEnd` (registered from `vortex-runend`)

Same shape as `Dict` but iterating runs instead of unique-value indices. Each run is
encoded once and the encoded bytes are emitted `run_length` times. The `vortex-runend`
crate submits the `RowEncodeRegistration` from its module init code.

#### `BitPacked` (registered from `vortex-fastlanes`)

Walks the bit-packed storage in 1024-element chunks via the existing FastLanes
`unpacked_chunks::<T>` iterator, unpacks each chunk into a stack-local buffer
(`[MaybeUninit<T>; 1024]`), and writes the row-encoded bytes for that chunk in one pass.
Avoids materializing a canonical `PrimitiveArray` (and its validity attachment) first.
Handles validity by materializing the mask once outside the chunk loop and patches by
materializing the patch index/value slices once outside the chunk loop and overlaying per
chunk inline.

The shared "encode 1024 unpacked values into row slots" primitive lives in
`encodings/fastlanes/src/row_encode_common.rs` as `encode_primitive_chunk::<T>`.

## Performance

Benchmark scenarios live in `vortex-row/benches/row_encode.rs`. Each scenario builds 100k
rows of input data once outside the timed region and then re-runs the encoder. The
counter is `BytesCount::new(encoded_bytes)` so the throughput column reads as GB/s of
*output*, not input.

The post-cleanup numbers (median GB/s, 30 samples, `--sample-count 30`) are in the
re-bench report in Task 3 of the work that landed this design doc. Headline shape:

- *Hot canonical paths* (primitive_i64, constant_i64 without kernel) land at 6-8 GB/s,
  far above arrow-row's ~4 GB/s baseline.
- *Varlen and struct* (utf8, struct_mixed) land at ~1.7-1.9 GB/s, modestly above
  arrow-row's ~1.2-1.5 GB/s on the same shape.
- *Kernel paths* (constant_i64 with kernel, dict_utf8 with kernel, bitpacked_i32 with
  kernel) beat their without-kernel counterparts by 1.0-1.4x and beat arrow-row's
  most-favored-shape benches (`arrow_dict`, plain `Int32`) by 1.0-2.0x.

The wins on the canonical hot paths come from a stack of small, independently-measurable
optimizations:

- **`ListViewArray::new_unchecked`** (Phase 5): skipping the row-by-row offset/size
  validation saves ~0.5-1 GB/s on small-fixed inputs where the validation is a meaningful
  fraction of total time.
- **Skip the validity mask allocation** for `Validity::NonNullable` and `Validity::AllValid`:
  the `execute_mask` call would allocate and materialize a bit-buffer of length `nrows`.
  When the mask isn't needed, branch on the cheap `matches!()` instead.
- **Varlen `copy_nonoverlapping`**: replace `out.copy_from_slice(&view.bytes())` with raw
  pointer copies. The slice creation cost is per-row and the copy is u64-strided, so the
  speedup is multiplicative.
- **Walk views directly**: `VarBinViewArray::with_iterator` goes through a closure that
  resolves view bytes inline; in the no-nulls fast path we walk the views ourselves and
  cache the buffer slices in a `SmallVec<[&[u8]; 4]>`.
- **Skip zero-init of the output buffer** (Phase 2): every byte is written by the
  encoders; the pre-zeroing memset is redundant.
- **Auto-vectorize the offsets construction** (Phase 3): raw pointer writes in a tight
  loop, no per-element `push`+`checked_add`.
- **Constant small-len register hoist**: hoist encoded bytes for length 2/5/9/17 into
  register-sized values before the loop.
- **Arithmetic-write lane**: for fixed-before-varlen columns, write at
  `i * stride + col_prefix + var_prefix[i]` without a cursor lookup. The compiler can
  recognize this as a strided write and SIMDify the inner sentinel-plus-value store.

A concrete optimization that *didn't* land: trying to share the chunk-write primitive
between the `Primitive`-canonical encoder and the `BitPacked` kernel. The `Primitive`
canonical encoder operates on a contiguous `&[T]` slice and gets autovectorized; the
BitPacked path operates on a stack-resident `[T; 1024]` buffer with separate validity
and patches handling. Trying to unify them ended up *slower* because the chunked path
introduced a per-1024 dispatch where the contiguous path had zero overhead.

## PR-split plan

The architecture is deliberately incremental: each step is independently reviewable,
benchable, and revertable. Reviewers see one motivation per commit and one set of numbers
per commit; future contributors can add a new kernel in one PR without touching anything
else.

### PR 1 — `vortex-row` base + canonical encoders

- Crate scaffolding: `vortex-row/Cargo.toml`, `lib.rs`, the four mod files, workspace
  member registration.
- `RowSize` and `RowEncode` scalar functions, their options serialization, `initialize`.
- The per-Canonical-variant byte encoders for all supported types: `Null`, `Bool`,
  `Primitive` (12 ptypes), `Decimal` (i8-i128), `Utf8`/`Binary` (32-byte block format),
  `Struct` (recursive), `FixedSizeList`, `Extension`. Explicit bails for `Variant`,
  `Decimal256`, `List`, `Union`.
- Basic tests in `tests.rs` covering each type, descending, nulls-first/last, struct/FSL
  recursion, multi-column, and the byte-comparable round trip.
- Initial benchmarks (`benches/row_encode.rs`) for primitive_i64, utf8, struct_mixed.

Target reviewer focus: the byte-encoding spec, the variadic ScalarFn shape, the orphan-
rule story for kernels. No optimizations yet — this is the "does it match arrow-row's
semantics" PR. Expected diff: ~3500 LOC. **Expected numbers: 1-2 GB/s on the hot paths,
roughly at parity with arrow-row.**

### PR 2 — Canonical-path optimizations

One commit per optimization, each independently measurable. The per-commit split exists
so reviewers can see which optimization buys what — and so that any single one can be
reverted cheaply if it regresses an unrelated workload. Listed in landing order:

1. `Split fixed/var sizing in row encoder` — Phase-1 classifier and the split between
   the arithmetic and cursor lanes.
2. `Skip ListView validation in row encoder output` — Phase 5 `new_unchecked`.
3. `Skip validity mask allocation for NonNullable/AllValid columns` — Phase 4 fast path.
4. `Rewrite varlen 32-byte block encoder with copy_nonoverlapping` — replaces
   `copy_from_slice` with raw pointer copies; XOR-block path for descending.
5. `Use copy_nonoverlapping in constant arithmetic encode` — initial Constant fast path.
6. `Walk VarBinView rows directly in row encoder hot loop` — bypass `with_iterator`.
7. `Specialize constant encoder for small fixed-length scalars` — 2/5/9/17 register hoist.
8. `Skip zero-init of output buffer in row encoder` — Phase 2 `set_len` without memset.
9. `Tighten offsets materialization and complete validity fast paths` — Phase 3 raw
   pointer writes; Decimal/Bool/Primitive validity fast paths.

Target reviewer focus: per-commit perf data, the soundness arguments for each `unsafe`
block, and the no-zero-init contract (Phase 2). Expected diff per commit: 30-200 LOC.
**Expected numbers: >5 GB/s on the hot canonical paths after the full stack lands.**

### PR 3+ — Per-encoding kernels (one commit per encoding)

Add the `RowSizeKernel` and `RowEncodeKernel` traits and their dispatch through
`size::dispatch_size` and `encode::dispatch_encode`. Add the `RowEncodeRegistration`
inventory.

**One commit per encoding** so each kernel can be reviewed, adopted, or reverted
independently. The current set:

- `Constant` (in-crate).
- `Dict` (in-crate).
- `RunEnd` (registered from `vortex-runend`).
- `BitPacked` (registered from `vortex-fastlanes`).
- `FoR` (registered from `vortex-fastlanes`).
- `Delta` (registered from `vortex-fastlanes`).
- `Patched` (in-crate).

**Future candidates**: FSST, ZigZag, Sequence — each lands as a follow-up commit
following the same template (see "Adding a new kernel" below). Reviewers can decide
on each independently; nothing else in the crate moves.

Target reviewer focus: per-commit, the kernel correctness test (each kernel has a
"matches canonical" round-trip test) and the bench triplet (`with_kernel`,
`without_kernel`, `arrow_row`). Expected diff per kernel: 100-300 LOC.

## Kernel decision log

A record of what was measured and why each registered kernel stays. Numbers are median
GB/s on `100_000`-row inputs, encoded-output throughput, `divan --sample-count 30`.
Ratios are *kernel-path* / *canonical-path-on-same-shape* and *kernel-path* /
*arrow-row-on-same-shape*. The bar to clear for keeping a kernel is **not** "beats the
canonical path by N%"; it is "either wins clearly, or stays roughly competitive while
exercising a code path that matters for extensibility, allocator pressure, or pipeline
composition." The shape of the registry is the load-bearing design decision; the per-
bench delta on synthetic inputs is one input to that decision, not the whole story.

- **BitPacked** (registered from `vortex-fastlanes`): **1.08× vs canonical, 1.95× vs
  arrow-row.** Clear win on both axes. The kernel skips the `nrows * sizeof(T)`-byte
  canonical materialization (plus its validity attachment) and writes row bytes directly
  out of the FastLanes 1024-element unpack buffer. Keep.

- **Patched** (in-crate, `vortex-row/src/kernels/patched.rs`): **1.00× vs canonical,
  1.42× vs arrow-row.** Ties canonical, beats arrow-row. Kept because arrow-row has no
  equivalent encoding (the `Patched` shape is Vortex-specific) and the dispatch overhead
  is minimal — the kernel decline path falls through to canonicalization in a single
  downcast check. Keeping it documents that the patched representation has a row-encode
  path and prevents accidental regressions if the canonical materializer for `Patched`
  ever loses its chunked-overlay specialization.

- **FoR** (registered from `vortex-fastlanes`): **0.96× vs canonical, 1.21× vs
  arrow-row.** Modest win over arrow; the canonical path already exploits FastLanes
  chunked unpack plus a fused `add base` so the gap to the kernel is small. Keep —
  having the kernel exist means downstream pipelines that build FoR-encoded data
  many times in a tight loop avoid the `PrimitiveArray` wrapper and validity-attachment
  allocation each call, which the per-encode bench doesn't see but a sort-many-batches
  workload does.

- **Delta** (registered from `vortex-fastlanes`): **0.94× vs canonical, 0.81× vs
  arrow-row.** Loses to both on the synthetic bench. Kept for **completeness and
  extensibility**: the canonical path edges it out today, but the kernel skips the
  `PrimitiveArray` materialization, which matters in pipelines where the row encoder
  is called many times (allocator pressure, memory residency). Removing it would also
  make the kernel-registry surface lopsided across the FastLanes encodings, which is
  the wrong signal to send to anyone adding a new encoding.

The kernel that explicitly *didn't* land: a shared chunk-write primitive between the
`Primitive`-canonical encoder and the `BitPacked` kernel. The canonical encoder operates
on a contiguous `&[T]` slice and gets autovectorized; the BitPacked path operates on a
stack-resident `[T; 1024]` buffer with separate validity and patches handling. Unifying
them ended up *slower* because the chunked path introduced a per-1024 dispatch where the
contiguous path had zero overhead. Documented here so the next person who reaches for
that idea has a known result.

## Adding a new kernel

The shape of the work for a new encoding (e.g., FSST, ZigZag, Sequence):

1. Implement `RowSizeKernel` + `RowEncodeKernel` for the encoding's `VTable` type.
   Both methods mutate shared `sizes` / `cursors` / `out` buffers and return
   `Ok(Some(()))` on success or `Ok(None)` to decline and fall through to
   canonicalization.

2. **For an in-crate encoding** (the encoding's `VTable` lives in `vortex-array`):
   the `impl` block lives in `vortex-row/src/kernels/<name>.rs`. Add a downcast arm
   in `vortex-row/src/{size,encode}.rs`'s `dispatch_size` / `dispatch_encode` (the
   `as_opt::<YourEncoding>()` chain). The orphan rule allows this because
   `vortex-row` defines the trait.

3. **For a downstream encoding** (the encoding's `VTable` lives in `encodings/foo/`):
   the `impl` block lives in the encoding's own crate, alongside its compute kernels.
   Submit it to the registry from that crate's module init:

   ```rust
   inventory::submit!(RowEncodeRegistration {
       id: || Foo.id(),
       size: foo_row_size,
       encode: foo_row_encode,
   });
   ```

   Add `vortex-row = { workspace = true }` to that crate's `Cargo.toml` if it isn't
   already a dependency. The function-pointer indirection isn't free (it prevents
   inlining of the kernel body into the dispatch loop), but it's the price of letting
   downstream encodings register a kernel without `vortex-row` knowing about them.

4. **Add a benchmark triplet** to `vortex-row/benches/row_encode.rs`:

   - `<name>_with_kernel` — encode through the kernel.
   - `<name>_without_kernel` — canonicalize first, then encode.
   - `<name>_arrow_row` — encode the equivalent arrow array through
     `arrow_row::RowConverter`.

   Each runs 100k rows of input data, generated once outside the timed region.

5. Run `cargo bench -p vortex-row --bench row_encode -- --sample-count 30`. Record
   the median GB/s for each triplet entry in the "Kernel decision log" above. If
   `with_kernel` isn't measurably faster than `without_kernel`, decide explicitly
   whether to keep the kernel for extensibility / allocator-pressure reasons (see the
   Delta / FoR / Patched entries above for the precedent) or skip the kernel and let
   the canonical path handle the encoding.

## Open questions

- **Decoder**. `arrow-row::RowConverter::convert_rows()` is the inverse direction: take a
  `RowArray` and return the column arrays. We haven't implemented this for `vortex-row`
  yet. Sorting workflows that build a `ListView<u8>`, sort it, then re-project the
  ordered rows back into columnar form will need this. Out of scope for v1.

- **Decimal256 and List bail with errors**. Decimal256 is a straight extension of the
  existing decimal sign-flip pattern (just larger fixed width). List requires either a
  variable-stride cursor path or a pre-pass that materializes each list as a varlen
  byte string. Open: which to ship as v1, which to defer.

- **Variant**. Heterogeneous values have no defined ordering, so `Variant` is a permanent
  bail. The error message points at this design choice.

- **Should `RowSize` be public?** The dual-ScalarFn surface lets downstream consumers
  precompute per-row sizes for things like sort-buffer allocation, but it also doubles
  the public-API surface for what is essentially one operation. An alternative is making
  `RowSize` an internal helper exposed through a `compute_row_sizes()` function (already
  present in `convert.rs`) without registering it as a scalar function. Worth discussing
  in the v1 review.
