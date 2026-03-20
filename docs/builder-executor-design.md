# Design: The Builder Executor

This document describes the iterative executor that drives `execute_into_builder`. It
mirrors the existing `execute_until` loop but writes into a `CanonicalWriter` instead of
producing intermediate arrays.

See [execute-into-builder-design.md](execute-into-builder-design.md) for the `CanonicalWriter`
trait, writer types, and the `get_init_bytes` primitive.

---

## The step enum

```rust
enum BuilderStep {
    /// This encoding finished writing into the writer.
    Done,

    /// Execute child[idx] into the SAME writer.
    /// After the child finishes, re-enter this encoding at phase+1.
    ///
    /// If `sub_writer` is Some(n), navigate to writer.child_writer(n) before
    /// entering the child. Used by StructEncoding to direct each field to its
    /// sub-writer.
    WriteChild {
        child_idx: usize,
        sub_writer: Option<usize>,
    },

    /// Materialize child[idx] via the existing iterative executor (execute_until).
    /// The result is placed back into the array via with_child.
    /// Re-enter this encoding at phase+1.
    MaterializeChild(usize, DonePredicate),
}
```

The two child-step types serve different purposes:

| | Data goes... | Parent sees result via... | Use when... |
|---|---|---|---|
| `WriteChild` | Into the shared output buffer | Reading/transforming the writer at the next phase | Child data flows sequentially into the output (same type, in-place transformable) |
| `MaterializeChild` | Into a temp `ArrayRef` | `array.child(idx)` at the next phase (placed by `with_child`) | Parent needs random-access into the child (gather, scatter, index lookup) |

---

## The VTable method

```rust
fn execute_into_builder(
    array: &Self::Array,
    writer: &mut dyn CanonicalWriter,
    phase: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BuilderStep>;
```

The encoding is **stateless**. It uses `phase` to determine what step it's on, and inspects
its children (which may have been replaced by `with_child` from a previous `MaterializeChild`)
to access materialized results.

Default implementation for encodings that don't specialize:

```rust
fn execute_into_builder(
    array: &Self::Array,
    writer: &mut dyn CanonicalWriter,
    _phase: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BuilderStep> {
    // Fall back to old executor, then write canonical result into writer.
    let canonical = array.to_array().execute::<Canonical>(ctx)?;
    writer.write(&canonical.into_array(), ctx)?;
    Ok(BuilderStep::Done)
}
```

---

## The executor loop

```rust
fn run_into_builder(
    array: ArrayRef,
    writer: &mut dyn CanonicalWriter,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mut current = array;
    let mut phase: usize = 0;

    // (parent_array, child_idx, parent_phase, pushed_sub_writer)
    let mut stack: Vec<(ArrayRef, usize, usize, bool)> = Vec::new();

    // Path of sub-writer indices for navigating StructWriter nesting.
    let mut writer_path: Vec<usize> = Vec::new();

    loop {
        // Run cross-encoding fusions (e.g. Dict+RunEnd → RunEnd with resolved values).
        // This mirrors execute_until's try_execute_parent call.
        // optimize() only runs reduce + reduce_parent; execute_parent is separate.
        loop {
            current = current.optimize()?;
            if let Some(rewritten) = try_execute_parent(&current, ctx)? {
                current = rewritten;
                continue; // re-optimize after rewrite
            }
            break;
        }

        let w = resolve_writer(writer, &writer_path);

        match current.vtable().execute_into_builder(&current, w, phase, ctx)? {
            BuilderStep::Done => {
                match stack.pop() {
                    None => return Ok(()),
                    Some((parent, _child_idx, parent_phase, pushed_sub)) => {
                        if pushed_sub {
                            writer_path.pop();
                        }
                        current = parent;
                        phase = parent_phase + 1;
                    }
                }
            }

            BuilderStep::WriteChild { child_idx, sub_writer } => {
                let child = current.nth_child(child_idx)?;
                // No optimize here — the top of the loop handles it.

                let pushed = sub_writer.is_some();
                stack.push((current, child_idx, phase, pushed));

                if let Some(sw) = sub_writer {
                    writer_path.push(sw);
                }

                current = child;
                phase = 0;
            }

            BuilderStep::MaterializeChild(child_idx, pred) => {
                let child = current.nth_child(child_idx)?;
                // Delegate to the EXISTING iterative executor.
                // This runs reduce, reduce_parent, execute_parent, execute —
                // the full optimization pipeline.
                let materialized = child.execute_until_pred(pred, ctx)?;
                current = current.with_child(child_idx, materialized)?;
                phase += 1;
            }
        }
    }
}

/// Navigate from the root writer to a nested sub-writer via a path of indices.
fn resolve_writer<'a>(
    root: &'a mut dyn CanonicalWriter,
    path: &[usize],
) -> &'a mut dyn CanonicalWriter {
    let mut w = root;
    for &idx in path {
        w = w.child_writer(idx).unwrap();
    }
    w
}
```

### How phase tracking works

- `WriteChild`: push `parent_phase` onto the stack. Child starts at `phase = 0`. On pop,
  `phase = parent_phase + 1`.
- `MaterializeChild`: increment `phase` inline (same loop iteration). The array is updated
  via `with_child` so the encoding can read the materialized child at the next phase.
- `Done`: pop the stack. If empty, return.

### Where optimizations run

| Step | Where optimizations happen |
|---|---|
| Top of loop (every iteration) | `optimize()` (reduce + reduce_parent) then `try_execute_parent` — runs on whatever `current` is, including after a `WriteChild` sets `current` to a child |
| `MaterializeChild` | Inside `execute_until_pred` — the old executor runs the full pipeline (reduce, reduce_parent, execute_parent, execute) |
| Between phases of the same encoding | No optimization on the parent array itself — the encoding controls its own phases |

The top-of-loop optimization is critical for **cross-encoding fusions** like Dict+RunEnd.
See the "Cross-encoding fusions" section below.

---

## The two execution systems

`execute_into_builder` does NOT replace the existing `execute` path. They coexist:

```
execute_into_builder                          execute (existing)
  Builder executor loop                         execute_until loop
  Writes into CanonicalWriter                   Returns ArrayRef
  For canonicalization                          For everything else
     │                                            (scalar pushdown,
     │                                             filter, cast, ...)
     │
     ├── WriteChild ─► stays in builder loop
     │
     └── MaterializeChild ─► delegates to execute_until ──►┘
```

`MaterializeChild` is the bridge. It drops into the old executor to get a temp array, then
hands it back to the builder loop. The old executor handles all the optimization passes.

---

## Cross-encoding fusions and `try_execute_parent`

Some optimizations require a child encoding to rewrite its parent. The canonical example is
**Dict+RunEnd fusion** (`RunEndTakeFrom` in `encodings/runend/src/compute/take_from.rs`).

### The Dict+RunEnd problem

Given `Dict(codes=RunEnd(ends, indices), values)`:

**Without fusion** (what happens if we skip `try_execute_parent`):
1. Dict's `execute_into_builder` does `MaterializeChild(CODES, PrimArray)`
2. The old executor expands RunEnd codes → flat `PrimitiveArray` (e.g. `[0,0,0,1,1,0,0]`)
3. Dict gathers: `output[i] = values[codes[i]]`

This expands the run-end encoding into a full-length array, losing the compression benefit.

**With fusion** (what `try_execute_parent` enables):
1. Top of builder loop: `try_execute_parent` on `Dict(RunEnd(...))`
2. RunEnd (child[0]) tells Dict: "I can resolve your dictionary" via `RunEndTakeFrom`
3. `RunEndTakeFrom` does: `RunEnd(ends, dict.values().take(runend.values()))` — resolving
   dictionary indices into actual values inside the RunEnd structure
4. Now `current = RunEnd(ends, resolved_values)` — a plain RunEnd, no Dict wrapper
5. RunEnd's `execute_into_builder` fires: materialize ends + resolved values, fill runs

Result: one decompression pass instead of two, and the run-end structure is preserved until
the final fill step.

### Why `optimize()` alone isn't enough

`optimize()` runs `reduce` + `reduce_parent` to fixpoint. These are metadata-only rewrites
(no `ExecutionCtx`, no data movement).

`RunEndTakeFrom` is an `execute_parent` kernel — it does real work (`dict.values().take(...)`)
and takes an `ExecutionCtx`. It can't be a `reduce_parent` rule. So `optimize()` misses it.

The builder loop must call `try_execute_parent` explicitly, just as `execute_until` does
at line 135 of `executor.rs`. The loop structure mirrors the existing executor:

```
execute_until loop:              builder loop:
  optimize()                       optimize()
  try_execute_parent()             try_execute_parent()  ← same!
  execute()                        execute_into_builder()
```

### Other execute_parent fusions

`RunEndTakeFrom` is the most impactful cross-encoding fusion for canonicalization, but the
same mechanism supports any `execute_parent` kernel. If a future encoding adds a fusion that
fires during canonicalization (not just operator pushdown), it will automatically work in
the builder loop.

The `execute_parent` kernels listed in the table at the end of this doc (Compare, Filter,
Mask, Slice, Take) primarily serve operator pushdown (e.g. `Filter(Dict(...))` → push filter
into dict codes). These fire during `MaterializeChild`'s `execute_until` call, not during
canonicalization. But `RunEndTakeFrom` is unique — it rewrites the parent array's encoding
structure during canonicalization, and needs `try_execute_parent` in the builder loop.

---

## Encoding patterns are composable

An encoding chooses **per child, per phase** whether to use `WriteChild` or `MaterializeChild`.
The "three patterns" from the design doc are just common combinations:

| Pattern | What it is | Phases |
|---|---|---|
| Direct decode | No children involved | phase 0: write into builder, Done |
| Child -> transform in-place | One child flows through builder | phase 0: WriteChild, phase 1: transform, Done |
| Materialize + combine | All children as temp arrays | phase 0..N-1: MaterializeChild each, phase N: combine into builder, Done |
| **Hybrid** | Mix of WriteChild and MaterializeChild | One child into builder, others materialized, combine in-place |

The hybrid pattern is the key insight: an encoding can write one child directly into the
output buffer via `WriteChild`, then materialize other children via `MaterializeChild`, then
combine everything in-place. This saves a temp allocation for the child that went directly
into the builder.

### When to use which

| The child's data needs to be... | Use | Temp allocation? |
|---|---|---|
| Written sequentially into the output, then transformed in-place | `WriteChild` | No |
| Read at random positions (gather, scatter, lookup by index) | `MaterializeChild` | Yes |

`WriteChild` works when the child's output type matches (or is transmutable to) the builder's
type, and the parent only needs a sequential pass over the data.

`MaterializeChild` is necessary when the parent needs to index into the child at arbitrary
positions determined by another child (e.g., Dict codes indexing into values, RunEnd values
indexed by run boundaries).

---

## Per-encoding examples

### BitPacking — Direct decode (phase 0 only)

```rust
fn execute_into_builder(array: &BitPackedArray, writer, phase: 0, ctx) -> BuilderStep {
    let pw = writer.downcast::<PrimitiveWriter>();
    let range = pw.get_init_bytes(array.len());
    unpack_into(array.packed(), range);
    apply_patches(range, array.patches(), ctx)?;
    range.finish();
    BuilderStep::Done
}
```

### FoR non-fused — WriteChild then transform

```rust
fn execute_into_builder(array: &FoRArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => BuilderStep::WriteChild { child_idx: 0, sub_writer: None },
        1 => {
            let pw = writer.downcast::<PrimitiveWriter>();
            pw.map_last_n(array.len(), |v| v.wrapping_add(&array.reference()));
            BuilderStep::Done
        }
    }
}
```

### ALP — WriteChild, transform, then materialize patches

```rust
fn execute_into_builder(array: &ALPArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => BuilderStep::WriteChild { child_idx: 0, sub_writer: None },
        1 => {
            let pw = writer.downcast::<PrimitiveWriter>();
            pw.map_last_n(array.len(), |int| alp_decode(int, array.exponents()));
            if array.has_patches() {
                BuilderStep::MaterializeChild(1, PrimitiveArray::matches)
            } else {
                BuilderStep::Done
            }
        }
        2 => BuilderStep::MaterializeChild(2, PrimitiveArray::matches),
        3 => {
            let indices = array.child(1).as_::<PrimitiveArray>();
            let values = array.child(2).as_::<PrimitiveArray>();
            apply_patches_to_writer(writer, indices, values);
            BuilderStep::Done
        }
    }
}
```

### DateTimeParts — Hybrid (WriteChild + MaterializeChild)

Days goes directly into the builder. Seconds and subseconds are materialized and added
in-place. Saves one temp allocation vs materializing all three.

```rust
fn execute_into_builder(array: &DateTimePartsArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        // Decode days directly into output buffer.
        0 => BuilderStep::WriteChild { child_idx: DAYS, sub_writer: None },

        // Days are in the buffer. Scale in-place, then request seconds.
        1 => {
            let pw = writer.downcast::<PrimitiveWriter<i64>>();
            pw.map_last_n(array.len(), |d| d * 86400 * divisor);
            BuilderStep::MaterializeChild(SECONDS, PrimitiveArray::matches)
        }

        // Seconds materialized. Add to buffer in-place, then request subseconds.
        2 => {
            let seconds = array.child(SECONDS).as_::<PrimitiveArray<i64>>();
            let pw = writer.downcast::<PrimitiveWriter<i64>>();
            pw.map_last_n_indexed(array.len(), |i, val| val + seconds[i] * divisor);
            BuilderStep::MaterializeChild(SUBSECONDS, PrimitiveArray::matches)
        }

        // Subseconds materialized. Add to buffer in-place. Done.
        3 => {
            let subseconds = array.child(SUBSECONDS).as_::<PrimitiveArray<i64>>();
            let pw = writer.downcast::<PrimitiveWriter<i64>>();
            pw.map_last_n_indexed(array.len(), |i, val| val + subseconds[i]);
            BuilderStep::Done
        }
    }
}
```

### Dict — Full materialize (both children need random access)

```rust
fn execute_into_builder(array: &DictArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => BuilderStep::MaterializeChild(CODES, PrimitiveArray::matches),
        1 => BuilderStep::MaterializeChild(VALUES, Canonical::matches),
        2 => {
            let codes = array.child(CODES).as_::<PrimitiveArray>();
            let values = array.child(VALUES);
            let pw = writer.downcast::<PrimitiveWriter>();
            let range = pw.get_init_bytes(array.len());
            for i in 0..array.len() {
                range.set_value(i, values.scalar_at(codes[i]));
            }
            range.finish();
            BuilderStep::Done
        }
    }
}
```

### Chunked — Sequential WriteChild per chunk

```rust
fn execute_into_builder(array: &ChunkedArray, writer, phase, ctx) -> BuilderStep {
    if phase < array.nchunks() {
        BuilderStep::WriteChild { child_idx: phase, sub_writer: None }
    } else {
        BuilderStep::Done
    }
}
```

### Struct — WriteChild per field with sub-writer navigation

```rust
fn execute_into_builder(array: &StructArray, writer, phase, ctx) -> BuilderStep {
    if phase < array.nfields() {
        // child_idx == phase: field 0 is child 0, field 1 is child 1, etc.
        // sub_writer == Some(phase): navigate to the field's sub-writer.
        BuilderStep::WriteChild { child_idx: phase, sub_writer: Some(phase) }
    } else {
        BuilderStep::Done
    }
}
```

The `sub_writer` field tells the executor to navigate to `writer.child_writer(phase)` before
entering the child. This way each struct field's encoding receives the correct leaf writer
(PrimitiveWriter, BoolWriter, etc.) rather than the top-level StructWriter.

---

## End-to-end trace: `Chunked<DateTimeParts<FoR<BP>, Sequence, Constant>>`

3 chunks, each is `DateTimeParts { days: FoR<BP<i64>>, seconds: Sequence, subseconds: Constant(0) }`.

Writer: `PrimitiveWriter<i64>` pre-allocated to total length.

```
current=Chunked  phase=0  stack=[]  writer_path=[]

── CHUNK 0 ──────────────────────────────────────────────────

→ WriteChild{0, None}                    // chunk 0
  optimize(chunk_0) → no rewrites
  push (Chunked, 0, 0, false)
  current=DateTimeParts  phase=0

  → WriteChild{DAYS, None}               // days child = FoR<BP<i64>>
    optimize(FoR<BP>) → no rewrites
    push (DTP, DAYS, 0, false)
    current=FoR<BP<i64>>  phase=0

    → WriteChild{0, None}                // FoR's encoded child = BP<i64>
      optimize(BP) → no rewrites
      push (FoR, 0, 0, false)
      current=BP<i64>  phase=0

      → decode into writer, Done          ← DATA LANDS IN OUTPUT BUFFER
      pop → current=FoR  phase=1

    → map_last_n(wrapping_add(ref)), Done ← FoR transforms in-place
    pop → current=DTP  phase=1

  → scale days: map_last_n(|d| d * 86400 * div)
    MaterializeChild(SECONDS, PrimArray)
    ┌─ OLD EXECUTOR ──────────────────┐
    │ Sequence(0, 1)                  │
    │ → PrimitiveArray<i64>           │
    └─────────────────────────────────┘
    with_child → phase=2

  → add seconds in-place: map_last_n_indexed(|i, v| v + sec[i] * div)
    MaterializeChild(SUBSECONDS, PrimArray)
    ┌─ OLD EXECUTOR ──────────────────┐
    │ Constant(0)                     │
    │ → PrimitiveArray<i64>           │
    └─────────────────────────────────┘
    with_child → phase=3

  → add subseconds in-place, Done
  pop → current=Chunked  phase=1

── CHUNK 1 ──────────────────────────────────────────────────

→ WriteChild{1, None}                    // chunk 1
  push (Chunked, 1, 1, false)
  current=DateTimeParts  phase=0
  ... same as chunk 0, writes into next region of buffer ...
  pop → current=Chunked  phase=2

── CHUNK 2 ──────────────────────────────────────────────────

→ WriteChild{2, None}                    // chunk 2
  ... same ...
  pop → current=Chunked  phase=3

→ Done, stack empty → return
```

**Max stack depth**: 3 (Chunked → DateTimeParts → FoR → BitPacking pops immediately).

**Allocations**: One pre-allocated output buffer. Two small temp arrays per chunk for
seconds/subseconds (via MaterializeChild). Days decoded directly into the output — zero
intermediate allocation for the largest child.

---

## Trace: `Chunked<Struct<FoR<BP<i32>>, ALP<FoR<BP<i64>>>>>`

2 chunks. Struct with two fields: an i32 column and an f64 column.

Writer: `StructWriter { PrimitiveWriter<i32>, PrimitiveWriter<f64> }`.

```
current=Chunked  phase=0  stack=[]  writer_path=[]

── CHUNK 0 ──────────────────────────────────────────────────

→ WriteChild{0, None}                           // chunk 0
  push (Chunked, 0, 0, false)
  current=Struct  phase=0
  writer_path=[]  writer=StructWriter

  → WriteChild{0, sub_writer: Some(0)}          // field 0 = FoR<BP<i32>>
    push (Struct, 0, 0, true)
    writer_path=[0]  writer=PrimitiveWriter<i32>
    current=FoR<BP<i32>>  phase=0

    → WriteChild{0, None}                       // FoR's encoded child
      push (FoR, 0, 0, false)
      writer_path=[0]
      current=BP<i32>  phase=0

      → decode into PrimitiveWriter<i32>, Done   ← i32 DATA INTO FIELD 0
      pop → FoR  phase=1

    → map_last_n(wrapping_add(ref)), Done
    pop → Struct  phase=1
    writer_path=[]                               ← sub_writer popped

  → WriteChild{1, sub_writer: Some(1)}          // field 1 = ALP<FoR<BP<i64>>>
    push (Struct, 1, 1, true)
    writer_path=[1]  writer=PrimitiveWriter<f64>
    current=ALP<FoR<BP<i64>>>  phase=0

    → WriteChild{0, None}                       // ALP's encoded child (FoR<BP<i64>>)
      push (ALP, 0, 0, false)
      writer_path=[1]
      current=FoR<BP<i64>>  phase=0

      → WriteChild{0, None}                     // FoR's encoded child
        push (FoR, 0, 0, false)
        writer_path=[1]
        current=BP<i64>  phase=0

        → decode into PrimitiveWriter<f64>, Done ← i64 BITS INTO FIELD 1
        pop → FoR  phase=1

      → map_last_n(wrapping_add(ref)), Done
      pop → ALP  phase=1

    → map_last_n(alp_decode), apply patches, Done
    pop → Struct  phase=2
    writer_path=[]

  → Done (nfields=2)
  pop → Chunked  phase=1

── CHUNK 1 ──────────────────────────────────────────────────

→ WriteChild{1, None}                           // chunk 1
  ... same pattern, data appends into both field writers ...
  pop → Chunked  phase=2

→ Done, stack empty → return
```

**Max stack depth**: 4 (Chunked → Struct → ALP → FoR → BP pops immediately).

**Writer path max depth**: 1 (one level of struct nesting).

**Allocations**: Two pre-allocated output buffers (one per struct field). Zero intermediate
arrays — all data decoded directly through the builder chain.

---

## Writer API additions

The `CanonicalWriter` trait needs one addition for sub-writer navigation:

```rust
pub trait CanonicalWriter: Send {
    // ... existing methods (as_any_mut, dtype, len, write, finish) ...

    /// Access a child writer for composite types.
    /// Only StructWriter overrides this. Returns None for leaf writers.
    fn child_writer(&mut self, idx: usize) -> Option<&mut dyn CanonicalWriter> {
        _ = idx;
        None
    }
}
```

The `PrimitiveWriter` needs one addition for hybrid-pattern encodings:

```rust
impl PrimitiveWriter {
    // ... existing methods (get_init_bytes, map_last_n, append_validity) ...

    /// Transform the last n values in-place with access to position index.
    /// Used by hybrid encodings that combine builder data with a materialized child.
    pub fn map_last_n_indexed(&mut self, n: usize, f: impl Fn(usize, T) -> T);
}
```

---

## Which encodings benefit from the hybrid pattern

| Encoding | Children | Hybrid opportunity | Saving |
|---|---|---|---|
| **DateTimeParts** | days, seconds, subseconds | WriteChild days, materialize seconds + subseconds | 1 fewer temp array (the largest child) |
| **Delta** | bases, deltas | WriteChild deltas, materialize bases (small) | 1 fewer temp (the large child) |
| **ALP-RD** | left_parts, right_parts | WriteChild right_parts (if same width as output), materialize left_parts | 1 fewer temp (type alignment required) |
| **Dict** | codes, values | Neither — gather is random-access on both | No hybrid benefit |
| **RunEnd** | ends, values | Neither — run-fill indexes both randomly | No hybrid benefit |
| **RLE (FL)** | values, indices, offsets | Neither — dictionary scatter is random-access | No hybrid benefit |
| **Sparse** | fill + patch indices + values | get_init_bytes + fill, materialize indices + values, scatter | No WriteChild — fill is scalar, not a child |

The hybrid pattern helps most when one child is large and sequential (days in DateTimeParts,
deltas in Delta) while the other children are small and accessed by index.

---

## Complete encoding reference

Every encoding in the codebase, with its builder executor strategy.

### Canonical types (already the target form — write buffers into the writer)

These encodings ARE canonical. Their `execute_into_builder` writes their own buffers
directly into the writer. No children to execute.

| Encoding | Crate | Writer | Strategy | Phases |
|---|---|---|---|---|
| **Primitive** | `vortex-array` | PrimitiveWriter | `get_init_bytes(n)` + `copy_from_slice` from buffer | 0: memcpy, Done |
| **Bool** | `vortex-array` | BoolWriter | Copy bits into `BitBufferMut` | 0: memcpy, Done |
| **Null** | `vortex-array` | NullWriter | Increment counter | 0: increment, Done |
| **Decimal** | `vortex-array` | DecimalWriter | Copy integer mantissa buffer | 0: memcpy, Done |
| **VarBinView** | `vortex-array` | VarBinViewWriter | `push_views(views, buffers)` — stash data buffers zero-copy | 0: push views, Done |
| **ListView** | `vortex-array` | ListWriter | Push shifted offsets/sizes, stash elements | 0: push parts, Done |
| **FixedSizeList** | `vortex-array` | FSLWriter | Stash elements | 0: stash, Done |
| **Extension** | `vortex-array` | inner writer | Delegate to storage array's `execute_into_builder` | 0: WriteChild(storage), Done |

All canonical types also push validity into the writer.

### Struct

| Encoding | Crate | Writer | Strategy |
|---|---|---|---|
| **Struct** | `vortex-array` | StructWriter | WriteChild per field with `sub_writer` navigation |

```rust
fn execute_into_builder(array: &StructArray, writer, phase, ctx) -> BuilderStep {
    if phase < array.nfields() {
        WriteChild { child_idx: phase, sub_writer: Some(phase) }
    } else {
        Done
    }
}
```

Validity is pushed at each phase. The `sub_writer` field routes each field to its
corresponding child writer inside the StructWriter.

### Container encodings (peel/restructure, don't decode)

| Encoding | Crate | Strategy | Phases |
|---|---|---|---|
| **Chunked** | `vortex-array` | WriteChild per chunk | 0..N-1: WriteChild(phase), N: Done |
| **Slice** | `vortex-array` | Slice the inner array, then WriteChild | 0: WriteChild(0) on `inner.slice(range)` |
| **Filter** | `vortex-array` | MaterializeChild + filter into writer | 0: MaterializeChild(child), 1: filter rows into writer |
| **Masked** | `vortex-array` | WriteChild + apply validity | 0: WriteChild(child), 1: apply mask to writer validity |
| **Shared** | `vortex-array` | Use default (execute via old executor, write result) | Default fallback |
| **ScalarFn** | `vortex-array` | Use default (execute the scalar function, write result) | Default fallback |

**Chunked** is the most important — it's the entry point for canonicalization:
```rust
fn execute_into_builder(array: &ChunkedArray, writer, phase, ctx) -> BuilderStep {
    if phase < array.nchunks() {
        WriteChild { child_idx: phase, sub_writer: None }
    } else {
        Done
    }
}
```

**Slice** wraps its inner child. The executor's `optimize()` will often reduce
`Slice(X)` before we enter, so many slice wrappers are eliminated before
`execute_into_builder` is even called.

**Masked** writes the child's data into the builder, then overlays the validity mask:
```rust
fn execute_into_builder(array: &MaskedArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => WriteChild { child_idx: 0, sub_writer: None },
        1 => {
            writer.apply_validity(array.mask());
            Done
        }
    }
}
```

**Filter** must materialize the child first (the output is a different size), then
write selected rows:
```rust
fn execute_into_builder(array: &FilterArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => MaterializeChild(0, Canonical::matches),
        1 => {
            let child = array.child(0); // now canonical
            let mask = array.mask();
            write_filtered_rows(child, mask, writer);
            Done
        }
    }
}
```

### Canonical-producing encodings (convert to a different canonical form)

| Encoding | Crate | Writer | Strategy | Phases |
|---|---|---|---|---|
| **VarBin** | `vortex-array` | VarBinViewWriter | Build views from offsets, share data buffer | 0: build views + push, Done |
| **List** | `vortex-array` | ListWriter | Compute sizes from adjacent offsets, push parts | 0: compute + push, Done |
| **Constant** | `vortex-array` | any | `get_init_bytes(n)` + fill with scalar | 0: fill, Done |

These are not compressed — they just need a format conversion to reach canonical form.
No child execution needed.

### Direct decode encodings (Pattern 1)

These own compressed bytes and decode straight into the writer. No children to execute
(patches are handled inline).

| Encoding | Crate | Writer | Phases | Notes |
|---|---|---|---|---|
| **BitPacking** | `fastlanes` | PrimitiveWriter | 0: unpack + patches, Done | `unpack_into_primitive_builder()` already writes to `UninitRange`. Patches materialized inline and scattered. |
| **FoR (fused)** | `fastlanes` | PrimitiveWriter | 0: fused unpack + ref, Done | `FoRStrategy` applies `wrapping_add(ref)` during unpack. Single pass. Patches shifted. |
| **Pco** | `pco` | PrimitiveWriter | 0: decompress pages, Done | Lazy — only needed pages. Size from metadata. |
| **Zstd** | `zstd` | PrimitiveWriter or VarBinViewWriter | 0: decompress frames, Done | Reconstructs array from decompressed bytes. Dtype determines writer type. |
| **ZstdBuffers** | `zstd` | any | 0: decompress buffers + rebuild, Done | Decompresses each buffer independently, rebuilds inner encoding, then writes. |
| **Sequence** | `sequence` | PrimitiveWriter | 0: generate values, Done | `get_init_bytes(n)` + fill `base + i * multiplier`. No input buffers. |
| **ByteBool** | `bytebool` | BoolWriter | 0: byte-to-bit conversion, Done | Convert each byte to a bit directly into `BitBufferMut`. |
| **FSST** | `fsst` | VarBinViewWriter | 0: decompress + push views, Done | Decompress code sequences via symbol table, bulk-allocate string buffer, push views. |

**BitPacking** with patches:
```rust
fn execute_into_builder(array: &BitPackedArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => {
            let pw = writer.downcast::<PrimitiveWriter>();
            let range = pw.get_init_bytes(array.len());
            unpack_into(array.packed(), range);
            if let Some(patches) = array.patches() {
                // Materialize patch indices + values inline (small arrays)
                let indices = patches.indices().execute::<PrimitiveArray>(ctx)?;
                let values = patches.values().execute::<PrimitiveArray>(ctx)?;
                for (idx, val) in indices.iter().zip(values.iter()) {
                    range.set_value(idx, val);
                }
            }
            unsafe { range.append_mask(array.validity_mask()?); }
            unsafe { range.finish(); }
            Done
        }
    }
}
```

Note: BitPacking materializes patch children inline (they're small). This avoids adding
phases for patches, keeping the common no-patches path at a single phase.

### WriteChild + transform encodings (Pattern 2)

One child writes into the builder, then the parent transforms those bytes in-place.

| Encoding | Crate | Writer | Child | Transform | Patches | Phases |
|---|---|---|---|---|---|---|
| **FoR (non-fused)** | `fastlanes` | PrimitiveWriter | encoded (child 0) | `wrapping_add(reference)` | From inner child, shifted | 0: WriteChild, 1: transform, Done |
| **ALP** | `alp` | PrimitiveWriter | encoded (child 0) | `alp_decode(int, e, f)` int->float | Float exceptions at sparse indices | 0: WriteChild, 1: transform + patches, Done |
| **ZigZag** | `zigzag` | PrimitiveWriter | encoded (child 0) | `zigzag_decode(u)` unsigned->signed | None | 0: WriteChild, 1: transform, Done |
| **DecimalByteParts** | `decimal-byte-parts` | DecimalWriter | msp (child 0) | None (bits ARE the mantissa) | None | 0: WriteChild, 1: wrap type, Done |

**ALP** — the most complex Pattern 2, with optional patch materialization:
```rust
fn execute_into_builder(array: &ALPArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        // Encoded child (e.g. FoR<BitPacked<i64>>) writes into builder.
        0 => WriteChild { child_idx: 0, sub_writer: None },
        // Transform integers to floats in-place, then request patches.
        1 => {
            writer.downcast::<PrimitiveWriter>()
                .map_last_n(array.len(), |int| alp_decode(int, array.exponents()));
            if array.has_patches() {
                MaterializeChild(1, PrimitiveArray::matches) // patch indices
            } else {
                Done
            }
        }
        2 => MaterializeChild(2, PrimitiveArray::matches), // patch values
        3 => {
            apply_patches_to_writer(writer, array.child(1), array.child(2));
            Done
        }
    }
}
```

The encoded child may be `FoR<BitPacked<i64>>` — it flows through the builder loop all
the way down to BitPacking, which decodes into the buffer. Then FoR adds the reference
in-place. Then ALP transmutes in-place. One buffer, three passes, zero intermediates.

**ZigZag**:
```rust
fn execute_into_builder(array: &ZigZagArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => WriteChild { child_idx: 0, sub_writer: None },
        1 => {
            writer.downcast::<PrimitiveWriter>()
                .map_last_n(array.len(), |u| zigzag_decode(u));
            Done
        }
    }
}
```

### Materialize + combine encodings (Pattern 3)

Multiple children materialized as temp arrays, then combined into the writer.

| Encoding | Crate | Writer | Children to materialize | Combine | Phases |
|---|---|---|---|---|---|
| **Dict** | `vortex-array` | any | codes, values | Gather: `output[i] = values[codes[i]]` | 0: Mat codes, 1: Mat values, 2: gather, Done |
| **RunEnd** | `runend` | any | ends, values | Fill runs: repeat each value for run length | 0: Mat ends, 1: Mat values, 2: fill, Done |
| **RLE (FL)** | `fastlanes` | PrimitiveWriter | values, indices, offsets | Dictionary scatter per 1024-element chunk | 0-2: Mat each, 3: scatter, Done |
| **Sparse** | `sparse` | any | patch_indices, patch_values | Fill default + scatter at indices | 0: Mat indices, 1: Mat values, 2: fill+scatter, Done |

**Dict**:
```rust
fn execute_into_builder(array: &DictArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => MaterializeChild(CODES, PrimitiveArray::matches),
        1 => MaterializeChild(VALUES, Canonical::matches),
        2 => {
            let codes = array.child(CODES).as_::<PrimitiveArray>();
            let values = array.child(VALUES);
            let range = writer.get_init_bytes(array.len());
            gather_into(codes, values, range);
            range.finish();
            Done
        }
    }
}
```

**Sparse** — the encoding IS patches:
```rust
fn execute_into_builder(array: &SparseArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => MaterializeChild(PATCH_INDICES, PrimitiveArray::matches),
        1 => MaterializeChild(PATCH_VALUES, Canonical::matches),
        2 => {
            let range = writer.get_init_bytes(array.len());
            range.fill(array.fill_value());
            let indices = array.child(PATCH_INDICES).as_::<PrimitiveArray>();
            let values = array.child(PATCH_VALUES);
            for (idx, val) in indices.iter().zip(values.iter()) {
                range.set_value(idx, val);
            }
            range.finish();
            Done
        }
    }
}
```

### Hybrid encodings (mix of WriteChild + MaterializeChild)

One child writes directly into the output buffer, others are materialized.

| Encoding | Crate | Writer | WriteChild | MaterializeChild | Phases |
|---|---|---|---|---|---|
| **DateTimeParts** | `datetime-parts` | PrimitiveWriter | days (child 0) | seconds, subseconds | 0: WC days, 1: scale + Mat sec, 2: add sec + Mat sub, 3: add sub, Done |
| **Delta** | `fastlanes` | PrimitiveWriter | deltas (child 1) | bases (child 0, small) | 0: WC deltas, 1: Mat bases, 2: undelta in-place, Done |
| **ALP-RD** | `alp` | PrimitiveWriter | right_parts (child 1, if same width) | left_parts (child 0) | 0: WC right, 1: Mat left, 2: combine in-place, Done |

**DateTimeParts** — days into builder, seconds + subseconds materialized:
```rust
fn execute_into_builder(array: &DateTimePartsArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => WriteChild { child_idx: DAYS, sub_writer: None },
        1 => {
            writer.downcast::<PrimitiveWriter<i64>>()
                .map_last_n(array.len(), |d| d * 86400 * divisor);
            MaterializeChild(SECONDS, PrimitiveArray::matches)
        }
        2 => {
            let seconds = array.child(SECONDS).as_::<PrimitiveArray<i64>>();
            writer.downcast::<PrimitiveWriter<i64>>()
                .map_last_n_indexed(array.len(), |i, val| val + seconds[i] * divisor);
            MaterializeChild(SUBSECONDS, PrimitiveArray::matches)
        }
        3 => {
            let sub = array.child(SUBSECONDS).as_::<PrimitiveArray<i64>>();
            writer.downcast::<PrimitiveWriter<i64>>()
                .map_last_n_indexed(array.len(), |i, val| val + sub[i]);
            Done
        }
    }
}
```

**Delta** — deltas are large (same size as output), bases are small:
```rust
fn execute_into_builder(array: &DeltaArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => WriteChild { child_idx: DELTAS, sub_writer: None },
        1 => MaterializeChild(BASES, PrimitiveArray::matches),
        2 => {
            let bases = array.child(BASES).as_::<PrimitiveArray>();
            let pw = writer.downcast::<PrimitiveWriter>();
            // Undelta + untranspose in-place using bases
            undelta_inplace(pw, bases, array.offset(), array.len());
            Done
        }
    }
}
```

**ALP-RD** — right_parts are u64 (same width as f64 output), left_parts need dict decode:
```rust
fn execute_into_builder(array: &ALPRDArray, writer, phase, ctx) -> BuilderStep {
    match phase {
        0 => WriteChild { child_idx: RIGHT_PARTS, sub_writer: None },
        1 => MaterializeChild(LEFT_PARTS, PrimitiveArray::matches),
        2 => {
            let left = array.child(LEFT_PARTS).as_::<PrimitiveArray<u16>>();
            let dict = array.left_parts_dictionary();
            let pw = writer.downcast::<PrimitiveWriter<u64>>();
            pw.map_last_n_indexed(array.len(), |i, right| {
                let left_decoded = dict[left[i] as usize] as u64;
                f64::to_bits(f64::from_bits((left_decoded << array.right_bit_width()) | right))
            });
            // Apply left_parts patches if present
            if array.has_patches() { ... }
            Done
        }
    }
}
```

### Summary table — all encodings

| Encoding | Pattern | Writer | WriteChild | MaterializeChild | Max phases |
|---|---|---|---|---|---|
| **Primitive** | canonical | PrimitiveWriter | — | — | 1 |
| **Bool** | canonical | BoolWriter | — | — | 1 |
| **Null** | canonical | NullWriter | — | — | 1 |
| **Decimal** | canonical | DecimalWriter | — | — | 1 |
| **VarBinView** | canonical | VarBinViewWriter | — | — | 1 |
| **ListView** | canonical | ListWriter | — | — | 1 |
| **FixedSizeList** | canonical | FSLWriter | — | — | 1 |
| **Extension** | canonical | inner writer | storage | — | 1 |
| **Struct** | container | StructWriter | all fields (sub_writer) | — | N+1 |
| **Chunked** | container | any | all chunks | — | N+1 |
| **Slice** | container | any | inner | — | 1 |
| **Masked** | container | any | child | — | 2 |
| **Filter** | container | any | — | child | 2 |
| **Shared** | default | any | — | — | 1 |
| **ScalarFn** | default | any | — | — | 1 |
| **VarBin** | conversion | VarBinViewWriter | — | — | 1 |
| **List** | conversion | ListWriter | — | — | 1 |
| **Constant** | direct | any | — | — | 1 |
| **BitPacking** | direct | PrimitiveWriter | — | — | 1 |
| **FoR (fused)** | direct | PrimitiveWriter | — | — | 1 |
| **Pco** | direct | PrimitiveWriter | — | — | 1 |
| **Zstd** | direct | varies | — | — | 1 |
| **ZstdBuffers** | direct | any | — | — | 1 |
| **Sequence** | direct | PrimitiveWriter | — | — | 1 |
| **ByteBool** | direct | BoolWriter | — | — | 1 |
| **FSST** | direct | VarBinViewWriter | — | — | 1 |
| **FoR (non-fused)** | WC + transform | PrimitiveWriter | encoded | — | 2 |
| **ALP** | WC + transform | PrimitiveWriter | encoded | patches (opt) | 4 |
| **ZigZag** | WC + transform | PrimitiveWriter | encoded | — | 2 |
| **DecimalByteParts** | WC + transform | DecimalWriter | msp | — | 2 |
| **Dict** | materialize | any | — | codes, values | 3 |
| **RunEnd** | materialize | any | — | ends, values | 3 |
| **RLE (FL)** | materialize | PrimitiveWriter | — | values, indices, offsets | 4 |
| **Sparse** | materialize | any | — | indices, values | 3 |
| **DateTimeParts** | hybrid | PrimitiveWriter | days | seconds, subseconds | 4 |
| **Delta** | hybrid | PrimitiveWriter | deltas | bases | 3 |
| **ALP-RD** | hybrid | PrimitiveWriter | right_parts | left_parts | 3+ |

### Reduce/reduce_parent rules by encoding

These rules fire during `optimize()` before `execute_into_builder` is called (for
`WriteChild`) or inside `execute_until` (for `MaterializeChild`).

| Encoding | reduce_parent rules | execute_parent kernels |
|---|---|---|
| **Primitive** | Cast, Mask | — |
| **Bool** | Cast, Mask | — |
| **Decimal** | Cast, Mask | — |
| **Constant** | Between, Cast, Compare, Filter, Mask, ScalarFn, Slice, Take | — |
| **Null** | Cast, Mask, Slice | — |
| **Struct** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **Dict** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **Chunked** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **VarBin** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **VarBinView** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **ListView** | Cast, Mask, Slice | — |
| **FixedSizeList** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **Extension** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **Filter** | Cast, Mask, Slice (own reduce) | — |
| **Slice** | Cast, Mask, Slice | — |
| **Masked** | Cast, Mask, Slice | — |
| **BitPacking** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **FoR** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **Delta** | Cast, Mask | — |
| **RLE (FL)** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **ALP** | Between, Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **ALP-RD** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **ZigZag** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **Pco** | Cast, Mask | — |
| **Zstd** | Cast, Mask | — |
| **FSST** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **Sparse** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **Sequence** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **RunEnd** | Cast, Mask | Compare, Filter, Mask, Slice, Take |
| **ByteBool** | Cast, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **DateTimeParts** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |
| **DecimalByteParts** | Cast, Filter, Mask, Slice | Compare, Filter, Mask, Slice, Take |

The `reduce_parent` rules (Cast, Mask, Slice, Filter) fire during `optimize()` at the top
of the builder loop. Most `execute_parent` kernels (Compare, Filter, etc.) serve operator
pushdown and fire during `MaterializeChild`'s `execute_until` call.

The exception is **`RunEndTakeFrom`** (RunEnd execute_parent on Dict parent) — it fires via
`try_execute_parent` at the top of the builder loop to fuse Dict+RunEnd before
`execute_into_builder` is called. See the "Cross-encoding fusions" section above.
