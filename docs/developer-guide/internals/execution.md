# Execution

Vortex defers computation wherever possible. Instead of immediately materializing intermediate
results, it represents them as arrays that still describe work to be done, such as `FilterArray`
and [`ScalarFnArray`](../../concepts/expressions.md). The actual computation happens later, when
the array is materialized -- either by a scan, a query engine, or an explicit `execute()` call.

## Why Defer

Deferring computation enables several optimizations that are not possible with eager evaluation:

- **Fusion** -- multiple operations can be reduced into fewer steps before any data is touched.
  For example, applying multiple arithmetic operations in sequence can be fused into a single
  operation.

- **Filter pushdown** -- when a `ScalarFnArray` appears inside a filter, the filter can be
  pushed through to the operation's children, avoiding materialization of rows that will be
  discarded.

- **GPU batching** -- deferred expression trees can be shipped to a GPU compute context in bulk.
  The GPU context can fuse the tree into a single kernel launch, reducing memory traffic and
  kernel launch overhead compared to eagerly executing each operation.

## The Executable Trait

Execution is driven by the `Executable` trait, which defines how to materialize an array into
a specific output type:

```rust
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}
```

The `ExecutionCtx` carries the session and accumulates a trace of execution steps for debugging.
Arrays can be executed into different target types:

- **`Canonical`** -- fully materializes the array into its canonical form.
- **`Columnar`** -- like `Canonical`, but with a variant for constant arrays to avoid
  unnecessary expansion.
- **Specific array types** (`PrimitiveArray`, `BoolArray`, `StructArray`, etc.) -- executes
  to canonical form and unwraps to the expected type, panicking if the dtype does not match.

## Constant Short-Circuiting

Executing to `Columnar` rather than `Canonical` enables an important optimization: if the
array is constant (a scalar repeated to a given length), execution returns the `ConstantArray`
directly rather than expanding it. This avoids allocating and filling a buffer with repeated
values.

```rust
pub enum Columnar {
    Canonical(Canonical),
    Constant(ConstantArray),
}
```

Almost all compute functions can make use of constant input values, and many query engines
support constant vectors directly, avoiding unnecessary expansion.

## Execution Overview

Execution has two closely related entry points:

- `ArrayRef::execute::<ArrayRef>` is the single-step executor. It tries `reduce`,
  `reduce_parent`, `execute_parent`, then `execute` once.
- `ArrayRef::execute_until<M>` is the matcher-driven loop used by `Canonical`, `Columnar`,
  and other target executors. It repeatedly interprets `ExecutionStep` until the current
  activation matches `M` or no further progress is possible.

`VTable::execute` never recursively descends into children on its own. Instead it returns an
`ExecutionResult` containing an `ExecutionStep` that tells `execute_until` what to do next.

The loop carries three mutable pieces of state:

- `current_array: ArrayRef` -- the array currently in focus.
- `current_builder: Option<Box<dyn ArrayBuilder>>` -- active only for the builder path.
  `AppendChild` appends detached children here, and `Done` finalizes the builder.
- `stack: Vec<StackFrame>` -- suspended parents from `ExecuteSlot`, including the detached
  slot index, its `DonePredicate`, and the parent builder that was active before focus moved
  into the child.

## The Four Layers

Encodings can contribute logic in four places. The single-step executor can touch all four.
The iterative `execute_until` loop revisits Layers 3 and 4 directly, using `ExecuteSlot`,
`AppendChild`, and `Done` to move focus around the tree.

### Layer 1: `reduce` -- self-rewrite rules

An encoding applies `ArrayReduceRule` rules to itself. These are structural simplifications
that look only at the array's own metadata and children types, not buffer contents.

Examples:
- A `FilterArray` with an all-true mask reduces to its child.
- A `FilterArray` with an all-false mask reduces to an empty canonical array.
- A `ScalarFnArray` whose children are all constants evaluates once and returns a `ConstantArray`.

### Layer 2: `reduce_parent` -- child-driven rewrite rules

Each child is given the opportunity to rewrite its parent via `ArrayParentReduceRule`. The child
matches on the parent's type via a `Matcher` and can return a replacement. This is still
metadata-only.

Examples:
- A `FilterArray` child of another `FilterArray` merges the two masks into one.
- A `PrimitiveArray` inside a `MaskedArray` absorbs the mask into its own validity field.
- A `DictArray` child of a `ScalarFnArray` pushes the scalar function into the dictionary
  values, applying the function to `N` unique values instead of `M >> N` total rows.
- A `RunEndArray` child of a `ScalarFnArray` pushes the function into the run values.

### Layer 3: `execute_parent` -- parent kernels

Each child is given the opportunity to execute its parent in a fused manner via
`ExecuteParentKernel`. Unlike reduce rules, parent kernels may read buffers and perform real
computation.

An encoding declares parent kernels by implementing `ExecuteParentKernel` and registering them
with the session-scoped kernel registry, specifying which parent types each kernel handles via a
`Matcher`:

```rust
pub trait ExecuteParentKernel<V: VTable> {
    type Parent: Matcher;  // which parent types this kernel handles

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,                   // the child
        parent: <Self::Parent as Matcher>::Match<'_>, // the matched parent
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(parent_id, Child, Kernel);
}
```

Examples:
- A `RunEndArray` child of a `SliceArray` performs a binary search on the run ends to produce a
  new `RunEndArray` with adjusted offsets, or a `ConstantArray` if the slice falls within a
  single run.
- A `PrimitiveArray` child of a `FilterArray` applies the filter mask directly over its buffer,
  producing a filtered `PrimitiveArray` in one pass.

### Layer 4: `execute` -- the encoding's own decode step

If no reduce rule or parent kernel handled the array, the encoding's `VTable::execute` method
is called. This is the encoding's chance to decode itself one step closer to canonical form.

Instead of recursively executing children inline, `execute` returns an `ExecutionResult`
containing an `ExecutionStep` that tells the scheduler what to do next:

```rust
pub enum ExecutionStep {
    /// Push the parent onto the stack, focus a single child, and resume the
    /// parent once that child matches the predicate.
    ExecuteSlot(usize, DonePredicate),

    /// Like ExecuteSlot, for the last child the parent waits for: once it is
    /// canonical, the parent only transforms its values row by row.
    ExecuteSpine(usize, DonePredicate),

    /// Detach a child, append it into the current activation's builder, and
    /// keep the parent as current_array for the next iteration.
    AppendChild(usize),

    /// Execution is complete. If a builder is active, it is finalized here.
    Done,
}
```

- `ExecuteSlot(i, pred)` detaches slot `i`, pushes the parent onto `stack`, and makes that
  child the new `current_array` until `pred` says it is done.
- `ExecuteSpine(i, pred)` behaves like `ExecuteSlot`, and marks the frame as fusable (see
  [Block Fusion](#block-fusion)).
- `AppendChild(i)` detaches slot `i`, appends that child into `current_builder`, and keeps
  the returned parent as `current_array` for the next iteration.
- `Done` finishes the current activation. If `current_builder` is active, the builder is
  finalized and its finished array becomes the result of this activation.

## The Execution Loop

The full `execute_until<M: Matcher>` loop uses an explicit work stack and an optional builder
to manage parent-child relationships without recursion.

```
execute_until<M>(root):
  current_array = root
  current_builder = None
  stack = []

  loop:
    ┌──────────────────────────────────────────────────────────────┐
    │ Step 1: is current_array "done"?                            │
    │   (matches M at the root, or the stack frame's              │
    │    DonePredicate inside ExecuteSlot)                        │
    ├──────────────────────┬───────────────────────────────────────┘
    │ yes                  │ no
    │                      │
    │  stack empty?        │  current_builder active?
    │  ├─ yes → return     │  ├─ yes → skip Step 2a / 2b
    │  └─ no  → pop frame, │  └─ no
    │     reattach child,  │
    │     restore builder, │
    │     loop             ▼
    │         ┌────────────────────────────────────────────┐
    │         │ Step 2a: execute_parent_for_child(         │
    │         │            stack.top.parent_array,         │
    │         │            current_array )                 │
    │         │ lookup session kernels by key:             │
    │         │   (parent.encoding_id(),                   │
    │         │    child.encoding_id())                    │
    │         ├────────────┬───────────────────────────────┘
    │         │ Some       │ None
    │         │            │
    │         │            ▼
    │         │  ┌─────────────────────────────────────────┐
    │         │  │ Step 2b: for each child:                │
    │         │  │   execute_parent_for_child(             │
    │         │  │     current_array, child )              │
    │         │  │ lookup session kernels by key:          │
    │         │  │   (parent.encoding_id(),                │
    │         │  │    child.encoding_id())                 │
    │         │  ├──────────┬──────────────────────────────┘
    │         │  │ Some     │ None
    │         │  │          │
    │         │  │          ▼
    │         │  │  ┌──────────────────────────────────────┐
    │         │  │  │ Step 3: current_array.execute()      │
    │         │  │  ├──────────────┬───────────────────────┘
    │         │  │  │              │
    │         │  │  │ ExecuteSlot(i, pred)
    │         │  │  │   -> push parent + builder
    │         │  │  │   -> current_array = child[i]
    │         │  │  │   -> current_builder = None
    │         │  │  │
    │         │  │  │ AppendChild(i)
    │         │  │  │   -> ensure current_builder
    │         │  │  │   -> child.append_to_builder(...)
    │         │  │  │   -> current_array = parent
    │         │  │  │
    │         │  │  │ Done
    │         │  │  │   -> finish current_builder if present
    │         │  │  │   -> otherwise use returned array
    │         ▼  ▼  ▼
    │    continue loop with rewritten or finished array
    └──────────────────────────────────────────────────────
```

Step 2a and Step 2b are skipped while `current_builder` is active. `AppendChild` partially
consumes `current_array`: some slots already live in the builder, so a parent rewrite would
observe inconsistent state and could discard accumulated builder data.

Both Step 2a and Step 2b route through `execute_parent_for_child`, which looks up
`(parent.encoding_id(), child.encoding_id())` in the session kernel snapshot.

## Block Fusion

Level-wise execution writes a full-length buffer at every level of an encoding tree, such as
FoR over FoR over BitPacked. Block fusion builds those levels into one pass over blocks of at most
1024 rows instead, whose intermediates stay in L1.

An encoding takes part with three pieces:

- `require_spine!` in `execute`, for the last child it waits for, once every other child it needs
  is executed. It promises that once that child is canonical, the encoding only transforms the
  child's values row by row, and that its validity is the child's. When the slot cannot fuse, it
  behaves exactly like `require_child!`.
- `VTable::block_op` for that slot: the transform of one block, which must produce exactly what
  `execute` produces.
- `VTable::block_source`, if the encoding can emit its own values a block at a time. An array
  with a block source is a *leaf*. A canonical primitive array is one: it lends its buffer.

The executor fuses in three steps:

1. **Check, going down.** When a parent asks for its spine with `ExecuteSpine`, the executor
   checks what does not need the slot executed: fusion is on, no builder is active, both arrays
   are primitive, non-empty and of one length, the parent has no `execute_parent` kernels of its
   own, and the parent above has none for it. A slot that fails runs as a plain `ExecuteSlot`.
2. **Build, going up.** When the child is ready, after its own `execute_parent` probes, the
   executor asks the parent for its op, once per frame, then the child for its block source. A
   leaf starts a block program, and the op of each fusable parent above is pushed onto it as its
   frame is popped. A child that is not a leaf yet executes as usual, and the op waits in the
   frame for the result.
3. **Drive.** As soon as the next parent cannot extend the program, or at the root, the executor
   runs it, in place of the parents it reached, and execution carries on with the result. A
   program never waits, so the executor holds it as a value, not as an array.

Fusion covers primitive arrays and extension types stored as primitives. It does not stream
validity: the program takes its leaf's. Disable it with
`ExecutionCtx::with_block_fusion(false)`. Traces show each `ExecuteSpine` or `ExecuteSlot`,
`fuse start`, `fuse extend` and `drive`, and how each op ran over the first block.

### Walkthrough: a Leaf and Two Ops

`FoR₁(FoR₂(FoR₃(BitPacked)))` fuses into one program. `FoR₃(BitPacked)` is the leaf: its block
source unpacks a FastLanes chunk and adds its reference in one pass. `FoR₂` and `FoR₁` are the
ops.

```
Going down: check each spine, build nothing
  FoR₁.execute → require_spine!(encoded)   check ✓ → frame { fuse }   ExecuteSpine
  FoR₂.execute → require_spine!(encoded)   check ✓ → frame { fuse }   ExecuteSpine
  current = FoR₃(BitPacked)

    stack:  FoR₂  fuse  ← top
            FoR₁  fuse

Going up (Step 2c): build
  FoR₂.block_op()        → add ref₂               the op first
  FoR₃.block_source()    → unpack + add ref₃      then the leaf     fuse start
  pop FoR₂, push its op                                             fuse extend ops=1
  FoR₁.block_op()        → add ref₁
  pop FoR₁, push its op                                             fuse extend ops=2
  stack empty → drive

Drive: one pass, 1024 rows at a time                                drive ops=2
  chunk k ──► unpack + ref₃ ──► scratch0 ──► + ref₂ ──► scratch0 ──► + ref₁ ──► output
                                (op 0: in place)         (op 1: the last op writes
                                                          the full-length output)
```

Level-wise execution writes three full-length buffers. The program writes one, and every
intermediate stays in one scratch block.

### When a Chain Does Not Form

A chain can stop at three points, each cheaper the earlier it stops:

```
a) Going down: the check fails, for example the parent has execute_parent kernels.
     Op.execute → require_spine!   check ✗ → frame { no fuse }        ExecuteSlot
     The child executes as usual. Going up, the frame is skipped with a field read.

b) Going up: the parent has no op, for example FoR with a zero reference.
     FoR(0).execute → require_spine!   check ✓ → frame { fuse }        ExecuteSpine
     FoR(0).block_op() → None → the frame turns off; no source is built
     FoR(5)(BitPacked) runs its own kernel, and FoR(0) returns the result.

c) Partly: a parent above cannot fuse, for example one that asks with require_child!.
     Pass.execute → require_child!                                    ExecuteSlot
     add.execute  → require_spine! ✓                                  ExecuteSpine
     add.execute  → require_spine! ✓                                  ExecuteSpine
     leaf + op + op, then the top frame is Pass → drive below it      drive ops=2
     Pass executes as usual over the driven result.
```

A child that is not a leaf yet, such as a Dict under FoR, does not stop the chain: the Dict
executes as usual, the op FoR built waits in its frame, and the Dict's canonical result is the
leaf.

## Incremental Execution

Execution is incremental: each call to `execute` moves the array one step closer to canonical
form, not necessarily all the way. This gives each child the opportunity to optimize before the
next iteration of execution.

For example, consider a `DictArray` whose codes are a sliced `RunEndArray`. Dict-RLE is a common
cascaded compression pattern with a fused decompression kernel, but the slice wrapper hides it:

```
dict:
  values: primitive(...)
  codes: slice(runend(...))    # Dict-RLE pattern hidden by slice
```

If execution jumped straight to canonicalizing the dict's children, it would expand the run-end
codes through the slice, missing the Dict-RLE optimization entirely. Incremental execution
avoids this:

1. First iteration: the slice `execute` returns `ExecuteSlot` for its `RunEndArray` child.
   Once that child is in focus, Step 2a looks up a registered parent kernel for the
   `(slice, runend)` pair before the child is forced toward canonical form.

2. Second iteration: the `RunEndArray` codes child now matches the Dict-RLE pattern. Its
   registered execute-parent kernel expands runs while performing dictionary lookups in a single
   pass, returning the canonical array directly.

## Walkthrough: Executing a RunEnd-Encoded Array

To make the execution flow concrete, here is a step-by-step trace of executing a
`RunEndArray` to `Canonical`:

```
Input:  RunEndArray { ends: [3, 7, 10], values: [A, B, C], len: 10 }
Goal:   Canonical (PrimitiveArray or similar)

Iteration 1:
  Step 1          → not done
  Step 2a         → skipped (root, no stacked parent)
  Step 2b         → None
  Step 3          → ends are not Primitive yet?
                    ExecuteSlot(0, Primitive::matches)
                    Stack: [(RunEnd, child_idx=0, Primitive::matches)]
                    Focus on: ends
                    current_builder = None

Iteration 2:
  Step 1          → done (ends already match Primitive)
                    Pop stack → replace child 0 in RunEnd

Iteration 3:
  Step 1          → not done
  Step 2a         → skipped (root again after the pop)
  Step 2b         → None
  Step 3          → values are not Canonical yet?
                    ExecuteSlot(1, AnyCanonical::matches)
                    Stack: [(RunEnd, child_idx=1, AnyCanonical::matches)]
                    Focus on: values

Iteration 4:
  Step 1          → done (values already match AnyCanonical)
                    Pop stack → replace child 1 in RunEnd

Iteration 5:
  Step 1          → not done
  Step 2a         → skipped (root)
  Step 2b         → None
  Step 3          → all children ready, decode runs:
                    [A, A, A, B, B, B, B, C, C, C]
                    Done → return PrimitiveArray

→ Result: PrimitiveArray [A, A, A, B, B, B, B, C, C, C]
```

## Walkthrough: Executing a Chunked Bool Array via `AppendChild`

`Chunked` uses the builder path for most dtypes. Instead of focusing one child as the new
`current_array`, it detaches one chunk at a time, appends it into `current_builder`, and keeps
the `ChunkedArray` itself as the active parent:

```
Input:  Chunked {
          chunks[0] = Bool[true, false],
          chunks[1] = Bool[false],
          chunks[2] = Bool[true, true],
        }
Goal:   Canonical BoolArray

Iteration 1:
  Step 1          → not done
  Step 2a         → skipped (root, no stacked parent)
  Step 2b         → None
  Step 3          → AppendChild(1)
                    create current_builder = BoolBuilder []
                    append chunks[0]
                    current_array = Chunked(next_builder_slot = 2)
                    current_builder = BoolBuilder [true, false]

Iteration 2:
  Step 1          → not done
  Step 2a / 2b    → skipped (builder active; current_array is partially consumed)
  Step 3          → AppendChild(2)
                    append chunks[1]
                    current_array = Chunked(next_builder_slot = 3)
                    current_builder = BoolBuilder [true, false, false]

Iteration 3:
  Step 1          → not done
  Step 2a / 2b    → skipped
  Step 3          → AppendChild(3)
                    append chunks[2]
                    current_array = Chunked(next_builder_slot = 4)
                    current_builder = BoolBuilder [true, false, false, true, true]

Iteration 4:
  Step 1          → not done
  Step 2a / 2b    → skipped
  Step 3          → Done
                    finish current_builder
                    result = BoolArray [true, false, false, true, true]

→ Result: BoolArray [true, false, false, true, true]
```

When `current_builder` is active, the array returned alongside `Done` is just the signal that
the parent activation has finished. The actual result comes from finalizing the builder.

## Implementing an Encoding: Where Does My Logic Go?

When adding a new encoding or optimizing an existing one, the key question is whether the
transformation needs to read buffer data:

| If you need to... | Put it in | Example |
|-------------------|-----------|---------|
| Rewrite the array by looking only at its own structure | `reduce` (Layer 1) | `FilterArray` removes itself when the mask is all true |
| Rewrite the parent by looking at your type and the parent's structure | `reduce_parent` (Layer 2) | `DictArray` pushes a scalar function into its values |
| Execute the parent's operation using your compressed representation | `execute_parent` / parent kernel (Layer 3) | `PrimitiveArray` applies a filter mask directly over its buffer |
| Decode yourself toward canonical form | `execute` (Layer 4) | `RunEndArray` expands runs into a `PrimitiveArray` |

Rules of thumb:

- Prefer `reduce` over `execute` when possible. Reduce rules are cheaper because they are
  metadata-only and run before any buffers are touched.
- Parent rules and parent kernels enable the "child sees parent" pattern. A child encoding often
  knows how to handle its parent's operation more efficiently than the parent knows how to handle
  the child.
- Treat `execute` as the fallback. If no reduce rule or parent kernel applies, the encoding
  decodes itself and uses `ExecuteSlot` or `AppendChild` to tell the scheduler what to do next.

## Executing Scalar Functions

When no reduce rule or parent kernel handles a `ScalarFnArray`, the executor calls its
`VTable::execute`, as for any other encoding. That passes the array's children as inputs to the
function's `ScalarFnVTable::execute_step`, which returns a `ScalarFnExecutionResult`:

- `ScalarFnExecutionResult::done(array)` -- the function's result.
- `ScalarFnExecutionResult::execute_input::<M>(i)` -- a request to execute input `i` until it
  matches the `Matcher` `M`. The `ScalarFnArray` turns this into `ExecuteSlot(i, M::matches)`,
  so the scheduler executes the input on its explicit stack and then calls `execute_step` again
  with the executed input in its place.

`Not` handles constant and canonical boolean inputs, and asks for anything else to be executed
to columnar form:

```rust
fn execute_step(
    &self,
    _options: &EmptyOptions,
    args: &dyn ExecutionArgs,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnExecutionResult> {
    match not_columnar(&args.get(0)?, args.row_count())? {
        Some(result) => Ok(ScalarFnExecutionResult::done(result)),
        None => Ok(ScalarFnExecutionResult::execute_input::<AnyColumnar>(0)),
    }
}
```

Because the scheduler executes inputs, a deep tree such as `not(not(not(chunked)))` runs without
recursion, and each input gets its reduce rules and parent kernels applied before it is forced.
Only request an input that does not already match `M`; such a request cannot make progress and
fails with an error.

### Migrating from `execute` to `execute_step`

`ScalarFnVTable::execute` is the older API. It runs the function over its inputs and executes
any input it cannot use inline. `execute_step` replaces it. Until every function is migrated,
`execute` stays a required method, and `execute_step` defaults to calling it and returning its
result as `done`.

Vortex only calls `execute` through that default. The iterative executor calls `execute_step`,
and `ScalarFnRef::execute`, used for example by `scalar_at` on a `ScalarFnArray`, is also
defined in terms of `execute_step`: on an input request it executes that input one step and
returns the function lazily over it as a new `ScalarFnArray`. So once a function implements
`execute_step`, its `execute` is never called.

To migrate a function:

1. Implement `execute_step`. Return `done` for inputs the function can compute over directly.
2. Where `execute` executed an input itself (for example
   `child.execute::<ArrayRef>(ctx)?` followed by a retry), return
   `execute_input::<M>(i)` with the matcher the function needs, such as `AnyColumnar` or
   `AnyCanonical`.
3. Replace the body of `execute` with `vortex_panic!`, as `Not` does.

Once every function is migrated, `execute` will be removed from the trait and `execute_step`
will become required.

## Future Work

The execution model is designed to support additional function types beyond scalar functions:

- **Aggregate functions** -- functions like `sum`, `min`, `max`, and `count` that reduce an
  array to a single value. These will follow a similar deferred pattern, with an `AggregateFnArray`
  capturing the operation and inputs until execution.

- **Window functions** -- functions that compute a value for each row based on a window of
  surrounding rows.

These extensions will use the same `Executable` trait and child-first optimization strategy,
allowing encodings to provide optimized implementations for specific aggregation patterns.
