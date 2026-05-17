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
