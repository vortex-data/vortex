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

The `execute_until<M: Matcher>` method on `ArrayRef` drives execution. The scheduler is
iterative: it rewrites and executes arrays in small steps until the current array matches the
requested target form.

At a high level, each iteration works like this:

1. `optimize(current)` runs metadata-only rewrites to fixpoint:
   `reduce` lets an array simplify itself, and `reduce_parent` lets a child rewrite its parent.
2. If optimization does not finish execution, each child gets a chance to `execute_parent`,
   meaning "execute my parent's operation using my representation".
3. If no child can do that, the array's own `execute` method returns the next `ExecutionStep`.

This keeps execution iterative rather than recursive, and it gives optimization rules another
chance to fire after every structural or computational step.

## The Four Layers

The execution model has four layers, but they are not all invoked in the same way. Layers 1 and
2 make up `optimize`, which runs to fixpoint before and after execution steps. Layers 3 and 4
run only after optimization has stalled.

```
execute_until(root):
  current = optimize(root)             # Layers 1-2 to fixpoint

  loop:
    if current matches target:
      return / reattach to parent

    Layer 3: try execute_parent on each child
      if one succeeds:
        current = optimize(result)
        continue

    Layer 4: call execute(current)
      ExecuteChild(i, pred) -> focus child[i], then optimize
      Done                  -> current = optimize(result)
```

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

An encoding declares its parent kernels in a `ParentKernelSet`, specifying which parent types
each kernel handles via a `Matcher`:

```rust
pub trait ExecuteParentKernel<V: VTable> {
    type Parent: Matcher;  // which parent types this kernel handles

    fn execute_parent(
        &self,
        array: &V::Array,                          // the child
        parent: <Self::Parent as Matcher>::Match<'_>, // the matched parent
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
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
    /// Ask the scheduler to execute child[idx] until it matches the predicate,
    /// then replace the child and re-enter execution for this array.
    ExecuteChild(usize, DonePredicate),

    /// Execution is complete. The array in the ExecutionResult is the result.
    Done,
}
```

## The Execution Loop

The full `execute_until<M: Matcher>` loop uses an explicit work stack to manage
parent-child relationships without recursion:

```
execute_until<M>(root):
  stack = []
  current = optimize(root)

  loop:
    ┌─────────────────────────────────────────────────────┐
    │ Is current "done"?                                  │
    │   (matches M if at root, or matches the stack       │
    │    frame's DonePredicate if inside a child)         │
    ├──────────────────────┬──────────────────────────────┘
    │ yes                  │ no
    │                      │
    │  stack empty?        │  Already canonical?
    │  ├─ yes → return     │  ├─ yes → pop stack (can't make more progress)
    │  └─ no  → pop frame, │  └─ no  → continue to execution steps
    │     replace child,   │
    │     optimize, loop   │
    │                      ▼
    │         ┌────────────────────────────────────┐
    │         │  Try execute_parent on each child  │
    │         │  (Layer 3 parent kernels)          │
    │         ├────────┬───────────────────────────┘
    │         │ Some   │ None
    │         │        │
    │         │        ▼
    │         │  ┌─────────────────────────────────┐
    │         │  │  Call execute (Layer 4)         │
    │         │  │  Returns ExecutionResult        │
    │         │  ├────────┬────────────────────────┘
    │         │  │        │
    │         │  │  ExecuteChild(i, pred)?
    │         │  │  ├─ yes → push (array, i, pred)
    │         │  │  │        current = child[i]
    │         │  │  │        optimize, loop
    │         │  │  └─ Done → current = result
    │         │  │            loop
    │         │  │
    │         ▼  ▼
    │    optimize result, loop
    └──────────────────────────
```

Note that `optimize` runs after every transformation. This is what enables cross-step
optimizations: after a child is decoded, new `reduce_parent` rules may now match that were
previously blocked.

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

1. First iteration: the slice `execute_parent` (parent kernel on RunEnd for Slice) performs a
   binary search on run ends, returning a new `RunEndArray` with adjusted offsets.

2. Second iteration: the `RunEndArray` codes child now matches the Dict-RLE pattern. Its
   `execute_parent` provides a fused kernel that expands runs while performing dictionary
   lookups in a single pass, returning the canonical array directly.

## Walkthrough: Executing a RunEnd-Encoded Array

To make the execution flow concrete, here is a step-by-step trace of executing a
`RunEndArray` to `Canonical`:

```
Input:  RunEndArray { ends: [3, 7, 10], values: [A, B, C], len: 10 }
Goal:   Canonical (PrimitiveArray or similar)

Iteration 1:
  reduce?         → None (no self-rewrite rules match)
  reduce_parent?  → None (no parent, this is root)
  execute_parent? → None (no parent)
  execute         → ends are not Primitive yet?
                    ExecuteChild(0, Primitive::matches)
                    Stack: [(RunEnd, child_idx=0, Primitive::matches)]
                    Focus on: ends

Iteration 2:
  Current: ends array
  Already Primitive? → yes, done.
  Pop stack → replace child 0 in RunEnd, optimize.

Iteration 3:
  reduce?         → None
  reduce_parent?  → None
  execute_parent? → None
  execute         → values are not Canonical yet?
                    ExecuteChild(1, AnyCanonical::matches)
                    Stack: [(RunEnd, child_idx=1, AnyCanonical::matches)]
                    Focus on: values

Iteration 4:
  Current: values array
  Already Canonical? → yes, done.
  Pop stack → replace child 1 in RunEnd, optimize.

Iteration 5:
  reduce?         → None
  reduce_parent?  → None
  execute_parent? → None
  execute         → all children ready, decode runs:
                    [A, A, A, B, B, B, B, C, C, C]
                    Done → return PrimitiveArray

→ Result: PrimitiveArray [A, A, A, B, B, B, B, C, C, C]
```

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
  decodes itself and uses `ExecuteChild` to request child execution when needed.

## Future Work

The execution model is designed to support additional function types beyond scalar functions:

- **Aggregate functions** -- functions like `sum`, `min`, `max`, and `count` that reduce an
  array to a single value. These will follow a similar deferred pattern, with an `AggregateFnArray`
  capturing the operation and inputs until execution.

- **Window functions** -- functions that compute a value for each row based on a window of
  surrounding rows.

These extensions will use the same `Executable` trait and child-first optimization strategy,
allowing encodings to provide optimized implementations for specific aggregation patterns.
