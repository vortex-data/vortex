# Sorted Merge Join with Offset-Value Coding — Design Notes

Three designs explored for producing order-preserving u64s ("ord numbers")
from arbitrary Vortex encodings, the inputs the merge driver consumes.
Each design lives in its own module under `vortex-array/src/`:

| Module          | Design                                | Status                    |
|-----------------|---------------------------------------|---------------------------|
| `ord_iter`      | `OrdIter` trait + chunked dispatch    | Converged design — ship this |
| `ord_direct`    | Hand-specialised per-encoding kernels | Baseline / reference      |
| `ord_memcmp`    | Materialize to bytes + memcmp merge   | Baseline / negative result |
| `ord_common`    | Shared source-column shapes + builders | Used by all three          |
| `ord_bench`     | Cross-design benchmark                | Reproduces the table below |

## The three designs

### 1. `ord_direct` — direct per-encoding kernels

A hand-written n-way merge function per encoding: `merge_n_way_prim`,
`merge_n_way_dict`, `merge_n_way_runend`, etc. Each knows the physical
shape and accesses it directly (codes for dict, cached run pointer for
run-end, etc.). No trait, no scratch buffer.

- Fastest per-row when the encoding's optimal access pattern is in-lined.
- Closed-set: adding a new encoding means adding a new top-level merge function.
- Multi-column composition is awkward — each combination wants its own function.

### 2. `ord_memcmp` — materialize then byte-compare

Every encoding produces a contiguous `Vec<u8>` of ord-bytes (sign-flipped
big-endian per primitive, escape-encoded varbin). The merge driver does
byte-slice comparison. One driver for everything.

- Uniform driver, simple to reason about.
- Materialization cost is O(N · stride) writes — dominates the pipeline
  for wide keys (varbin 1KB: 156 ns/row).
- Reasonable only if the byte buffer is amortised across multiple
  downstream operators.

### 3. `ord_iter` — the `OrdIter` trait (winner)

Each encoding implements:

```rust
pub trait OrdIter {
    fn ord_len(&self) -> usize;
    fn next_chunk<'s>(&mut self, max_rows: usize, scratch: &'s mut Scratch)
        -> Option<OrdChunk<'s>>;
    fn skip(&mut self, n: usize);
}

pub enum OrdChunk<'a> {
    Constant { value: u64, len: usize },
    RunEnd   { run_ends: &'a [u32], values: &'a [u64], len: usize },
    Dense    (&'a [u64]),
}
```

Each iter owns its cursor. Chunks borrow from caller-supplied scratch.
The structural variants let the merge driver bulk-emit through whole
Constant chunks and whole RunEnd runs. `skip(n)` advances without
producing values — the paper's duplicate-bypass shortcut.

Open extensibility: new encodings implement `OrdIter` (with a default
canonicalize-then-recurse fallback for any encoding that doesn't bother
to specialise — TODO: default impl not yet wired up).

## Benchmark — ord-number generation only

Pure cost of producing the order-preserving u64 stream for one side,
N=200,000 rows. Lower is better; ns is per logical row.

Run with `cargo test --release -p vortex-array ord_bench -- --ignored
--nocapture`.

### Primitive / Dict — uniform-shape encodings

```
                                  OrdIter    Direct    Memcmp(mat+drain)
Primitive i64 (sorted)              0.52      0.20         1.80
Dict 256 distinct                   0.29      0.18         1.74
Dict 16 distinct                    0.33      0.18         1.53
Dict 10K distinct                   0.31      0.18         1.49
Constant                            0.004     0.000        1.68
```

OrdIter is 1.5–3× slower than direct on uniform encodings — the price of
the trait/scratch indirection. Memcmp is 5–500× slower; materialization
is the bottleneck.

### RunEnd — structural encoding

```
                                  OrdIter    Direct(*)  Memcmp(mat+drain)
RunEnd 10 rows/run                  0.28     17.82         1.35
RunEnd 100 rows/run                 0.04     13.70         1.23
RunEnd 1000 rows/run                0.01      7.66         1.20
```

(*) Direct here uses the per-row binsearch accessor exposed for the bench;
the real `merge_n_way_runend` uses a cached run pointer and is faster (~1
ns/row). The bench shows direct's worst-case path.

**OrdIter wins by 30×–700× on RunEnd** because the `OrdChunk::RunEnd`
variant emits the run header structurally. The merge driver receives
e.g. `(run_ends, values)` and walks 200 runs instead of 200K rows.

### VarBin — variable-length keys

```
                                  OrdIter    Direct    Memcmp(mat+drain)
VarBin 50B (200K rows)              4.47      3.02        10.12
VarBin 200B (200K rows)             6.79      5.30       105.04
VarBin 1KB (20K rows)               7.07      6.12       156.35
```

OrdIter / Direct: both produce the first-8-byte prefix. ~1 ns/row gap is
the trait indirection.

Memcmp: scales linearly with key width because it copies every byte of
every row. **20× worse than OrdIter at 1KB keys.** This is the case where
materialization is structurally wrong.

### Multi-column + fallback

```
                                              OrdIter
Multi-col (2 i32 cols packed)                  0.37 ns/row
ClosureIter fallback (any encoding via Fn)     0.88 ns/row
PrimIter reference (specialised path)          0.30 ns/row
```

`MultiColI32Iter` packs two columns' values into one OVC u64 head (col0 in
the high 32 bits, col1 in the low). Same merge driver, no changes —
composition just becomes a different `next_chunk` body.

`ClosureIter` is the universal fallback: any new encoding satisfies
`OrdIter` by providing `Fn(usize) -> i64`. The fallback path is ~3×
slower than the specialised `PrimIter` but still under 1 ns/row — new
encodings work immediately at acceptable speed, can be optimised later.

## `skip` cost (`OrdIter` only)

```
                                              ns/row
PrimIter: skip(N/2) then drain rest            0.136
RunEndIter (100/run): skip(N/2) + drain        0.038
ConstantIter: skip(N/2) + drain                0.003
PrimIter: skip(99%) + drain                    0.007
PrimIter: skip(N) (no drain)                  ~0.000
ConstantIter: skip(N)                         ~0.000
RunEndIter (100/run): skip(N)                  0.015
```

`skip` is O(1) — sub-nanosecond. It just bumps the cursor; the next
`next_chunk` resumes from the new position with no materialization of
the skipped rows. This is the mechanism the merge driver uses to advance
through bulk-emitted runs and through duplicate-of-predecessor cases.

## Why `ord_iter` is the design to ship

The per-row generation cost is the metric that determines merge throughput
when the merge driver itself is well-tuned (the priority queue / loser
tree is fast at any reasonable n). `ord_iter`:

- Wins decisively on RunEnd (structural emit collapses runs).
- Wins decisively on wide VarBin (avoids the memcmp materialization).
- Loses by 1–3× on flat encodings (Primitive, Dict) — acceptable since
  these are already at <1 ns/row in absolute terms.
- Provides `skip(n)` for the bulk-advance shortcut.
- Composes recursively for nested types (Struct, List — design sketched,
  not yet implemented).
- Open to extension: new encodings ship with their own `impl OrdIter`.

The only case `ord_memcmp` might still be appropriate: when the ord-byte
buffer is reused across multiple operators (sort + dedup + merge join all
consuming the same materialised form). For an isolated SMJ, OrdIter is
strictly the right choice.

## Decision tree — which design where

| Situation | Pick |
|---|---|
| You can write one hand-tuned merge function per encoding combination, expect few encodings, no nested types | **`ord_direct`** — best per-row cost (<0.2 ns for primitives) |
| Sort key materialised once and consumed by multiple operators (sort + dedup + merge + agg sharing the byte buffer) | **`ord_memcmp`** — amortises the materialization cost |
| Sort key produced once and consumed once, by an SMJ over heterogeneous and possibly nested encodings | **`ord_iter`** — open extensibility, best on structural encodings, never penalises wide keys |
| New encoding type just added, no time to specialise yet | **`ord_iter` + `ClosureIter` fallback** — ~1 ns/row immediately, optimise later |

Default to `ord_iter` for any production SMJ path. The other two are
reference points / specialisations.

## Why `OrdIter` is the right shape

Three design principles that fall out of the benchmarks:

### 1. Encoding-aware shortcuts beat universal byte form

A universal materialise-to-bytes step makes the merge driver uniform but
*loses* the encoding's structural information. RunEnd materialised to
bytes writes `run_len` copies of each value; OrdIter's `OrdChunk::RunEnd`
variant emits one entry per run.

This is not a per-byte SIMD difference — it's an algorithmic difference.
At 200K rows with 100-row runs:
- memcmp: 200K materialised values × 8 bytes = 1.6 MB written
- ord_iter RunEnd: 2K run entries written

Two orders of magnitude less work. The chunk variant exposes the
encoding's shape so the merge driver doesn't have to rediscover it.

### 2. Specialisation is bounded — open extension is essential

`ord_direct`'s per-encoding merge functions are fast (0.2 ns/row for
primitives) but combinatorial: each new encoding multiplies the surface
area. For Vortex's ~15 built-in encodings + custom user encodings, this
is unmaintainable.

The trait with a default fallback (`ClosureIter`) gives every encoding a
working impl. Specialised impls are an optimisation layer on top, not a
correctness requirement. New encodings ship working at 1 ns/row; if a
benchmark says they need to be faster, drop in a specialised `impl`
without changing the merge driver.

### 3. Per-row dyn dispatch is unacceptable; per-chunk is free

A naive `&dyn Array` per-row dispatch costs ~5 ns/row (one virtual call
per value). That's worse than the entire OrdIter measurement on any
encoding.

The fix is chunked dispatch: pay one dyn call per chunk (default 1024
rows), amortise the cost to ~5 ns / 1024 = 0.005 ns/row. The inner loop
is statically typed over `&[u64]` scratch — the compiler vectorises it.

This is the chunking design choice everywhere: Arrow's compute kernels,
Velox vectors, DuckDB's vectorised engine. `OrdIter` follows the same
playbook.

## Cost model

Theoretical floors per row for each design:

| Operation | Cost |
|---|---|
| `memcmp` byte compare on a u64 row | ~0.5 ns (single SIMD op) |
| OVC `u64::cmp` | ~0.5 ns (single integer op) |
| One memory access (L1-resident) | ~1 ns |
| One indirect function call | ~5 ns |
| `Vec<u8>` write of 8 bytes (L1) | ~1 ns |

Per-row floor for each design at chunk_size=1024:

- **`ord_direct`** = 1 read + 1 OVC compute ≈ 1 ns
- **`ord_iter`** = 1 scratch write + 1 read for inner loop + amortised
  dyn call ≈ 2 ns per primitive row
- **`ord_memcmp`** = `mat_per_value` + read for merge ≈ 2 ns *per value*
  per row × stride / 8

`ord_memcmp` scales with key width; the other two don't (after the OVC
prefix). For wide keys, `ord_memcmp` is at the wrong end of an
asymptote.

Measured numbers (from the bench above) match the floors within ~2× —
the trait overhead is the residual gap on uniform encodings, paid back
by the structural shortcuts on RunEnd / Constant / wide VarBin.

## Alternatives considered and rejected

### Visitor pattern with closed enum (`ChunkRef::*`)

Sketched earlier in the design discussion. Same chunked-dispatch idea but
the chunk type is a closed enum:

```rust
enum ChunkRef<'a> {
    Primitive(...), Dict(...), RunEnd(...), Bytes(...), Constant(...),
    Generic(&'a dyn Array),
}
```

Rejected because: closed-set. Adding a new encoding requires modifying
the central enum — exactly the maintenance burden the trait avoids. Open
extension via downcasting is possible but reintroduces dyn complexity
without the benefits.

### Generic `fold` / `ArrayAccessor`

Vortex already has `ArrayAccessor<Item>` (`vortex-array/src/accessor.rs`)
yielding `Option<&Item>` per row. Rejected as the OVC entry point
because:
- Yields decoded values (string, list, etc.), not the order-preserving
  u64. Wrapping with a conversion reintroduces materialization.
- Per-row dispatch by default — no chunked amortisation.
- No way to expose structural shape (Constant, RunEnd) to the consumer.

`ArrayAccessor` is the right primitive for "do per-row work" — wrong one
for "produce ord-byte stream". OrdIter is the specialised version.

### Tournament tree / loser tree merge driver

The current `merge_n_way` does a linear-scan O(n) min-search per
emit-batch. A Graefe-style tree-of-losers gets O(log n).

Not yet implemented because at n ≤ 16 (typical n-way SMJ fan-in) the
linear-scan version is competitive: 8 u64 compares per emit-batch, all
hot in L1, branch-predicts well. For n > 32 a loser tree starts to
matter; deferring until a workload demands it.

The OrdIter trait is loser-tree-compatible — the tree consumes `&[u64]`
heads exactly like the linear-scan version. Migration is purely a
driver-side change.

### Window-function abstraction

Considered (and a sub-agent investigated). Rejected because:
- SQL window functions are 1-in-1-out. OVC's `skip(n)` and structural
  bulk-emit are 1-in-N-out and 0-in-1-out.
- Window functions assume a fixed partition. OVC compares against a
  *moving* predecessor that may live on a different side per emit.
- The "transformed iterator" framing is more accurate: `OrdIter` is a
  lending iterator that owns its cursor and yields structurally-shaped
  chunks. Not a window function.

## Limitations and open questions

Known gaps in the current `ord_iter` design that the doc tracks but the
prototype doesn't fully resolve:

1. **`cmp_full` for tie resolution.** When OVC u64 ties don't fully
   discriminate (long shared prefixes on VarBin, OVC tie between two
   sides at the same value), the driver currently can't fall back to a
   recursive compare. Needed for correctness on nested types
   (Struct / List). Sketched in chat as a recursive `cmp_full(&self, i,
   &dyn OrdIter, j) -> Ordering` method on the trait.
2. **Nested type impls.** `StructOrdIter` (horizontal: K field iters)
   and `ListOrdIter` (vertical: recursive over element iter) are
   designed but not coded.
3. **Real `Array` integration.** The modules use minimal stand-in
   structs (`ord_common::PrimI64`) rather than the full `Array` trait.
   Promoting is the path to operator integration — straightforward but
   requires understanding Vortex's vtable + canonicalize plumbing.
4. **Default trait impl via canonicalize.** Currently every encoding
   has an explicit `impl OrdIter`. The story is "new encodings can use
   the closure fallback or canonicalize then re-use a built-in iter."
   A default `next_chunk` that canonicalises + recurses would let any
   encoding work with zero boilerplate.
5. **Multi-column with offset packing.** `MultiColI32Iter` packs two
   columns into one u64 but doesn't expose which column first diverged
   (the OVC "offset"). For full multi-column OVC, the head u64 should
   pack `(offset, value_at_offset)` so the merge driver knows where to
   resume after a tie.

## Future work

1. **Default `OrdIter` impl that canonicalises** — gives every Vortex
   array a free working OrdIter even before a specialised impl exists.
2. **Multi-column composition** — `ColOrdIter<K>` wrapping K inner iters,
   packing first-diverging-column's value into the OVC u64. Sketched
   in chat, prototyped in earlier exploratory modules (now removed); not
   yet in `ord_iter`.
3. **Nested types** — `StructOrdIter` (horizontal composition over K
   field iters) and `ListOrdIter` (recursive over an element iter with
   list-lex cmp_full). Adds a `cmp_full` method for tie-break.
4. **Loser tree merge driver** — current driver does a linear-scan O(n)
   min-search per emit-batch; a Graefe-style tree-of-losers gets O(log n).
   Matters at n > ~16; not critical for typical SMJ.
5. **Wire `ord_iter` to real `vortex_array::Array`** — the modules
   currently use minimal stand-in structs (`ord_common::PrimI64` etc.).
   Promoting to the full `Array` trait is the path to operator integration.
