# Execution

Vortex defers computation wherever possible. When an expression is applied to an array, the
result is not computed immediately. Instead, a `ScalarFnArray` is constructed that captures the
operation and its inputs as a new array node. The actual computation happens later, when the
array is materialized -- either by a scan, a query engine, or an explicit `execute()` call.

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

The `ExecutionCtx` carries the session and provides access to registered encodings during
execution. Arrays can be executed into different target types:

- **`Canonical`** -- fully materializes the array into its canonical form.
- **`Columnar`** -- like `Canonical`, but with a variant for constant arrays to avoid
  unnecessary expansion.
- **Specific array types** (`PrimitiveArray`, `BoolArray`, `StructArray`, etc.) -- executes
  to canonical form and unwraps to the expected type, panicking if the dtype does not match.

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

1. First iteration: the slice executes and returns a new `RunEndArray` whose offsets have been binary searched.

2. Second iteration: the `RunEndArray` codes child now matches the Dict-RLE pattern. Its
   `execute_parent` provides a fused kernel that expands runs while performing dictionary
   lookups in a single pass, returning the canonical array directly.

The execution loop runs until the array is canonical or constant:

1. **Child optimization** -- each child is given the opportunity to optimize its parent's
   execution by calling `execute_parent` on the child's vtable. If a child can handle the
   parent more efficiently, it returns the result directly.

2. **Incremental execution** -- if no child provides an optimized path, the array's own
   `execute` vtable method is called. This executes children one step and returns a new
   array that is closer to canonical form, or executes the array itself.

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

Almost all compute functions can make use of constant input values, and many query engines support constant vectors
avoiding unnecessary expansion.

## ScalarFnArray

A `ScalarFnArray` holds a scalar function (the operation to perform), a list of child arrays
(the inputs), and the expected output dtype and length. It is itself a valid Vortex array and
can be nested, sliced, and passed through the same APIs as any other array.

When an expression like `{x: $, y: $ + 1}` is applied to a bit-packed integer array, the
result is a tree of `ScalarFnArray` nodes rather than a materialized struct:

```
scalar_fn(struct.pack):
  children:
    - bitpacked(...)                    # x: passed through unchanged
    - scalar_fn(binary.add):            # y: deferred addition
        children:
          - bitpacked(...)              #    original array
          - constant(1)                 #    literal 1
```

Nothing is computed until the tree is executed.

## Future Work

The execution model is designed to support additional function types beyond scalar functions:

- **Aggregate functions** -- functions like `sum`, `min`, `max`, and `count` that reduce an
  array to a single value. These will follow a similar deferred pattern, with an `AggregateFnArray`
  capturing the operation and inputs until execution.

- **Window functions** -- functions that compute a value for each row based on a window of
  surrounding rows.

These extensions will use the same `Executable` trait and child-first optimization strategy,
allowing encodings to provide optimized implementations for specific aggregation patterns.
