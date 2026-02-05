# Lazy Execution

Vortex defers computation wherever possible. When an expression is applied to an array, the
result is not computed immediately. Instead, a `ScalarFnArray` is constructed that captures the
operation and its inputs as a new array node. The actual computation happens later, when the
array is materialized -- either by a scan, a query engine, or an explicit `execute()` call.

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

## Why Defer

Deferring computation enables several optimizations that are not possible with eager evaluation:

- **Fusion** -- multiple operations can be reduced into fewer steps before any data is touched.
  For example, applying multiple arithmetic operations in sequence can be fused into a single
  operation.

- **Constant folding** -- if all inputs to a `ScalarFnArray` are constants, the result is
  computed once at construction time and replaced with a `ConstantArray`.

- **Filter pushdown** -- when a `ScalarFnArray` appears inside a filter, the filter can be
  pushed through to the operation's children, avoiding materialization of rows that will be
  discarded. Note that we intend to explore cost-based filter pushdown to determine when pushing
  filters is actually beneficial.

- **GPU batching** -- deferred expression trees can be shipped to a GPU compute context in bulk.
  The GPU context can fuse the tree into a single kernel launch, reducing memory traffic and
  kernel launch overhead compared to eagerly executing each operation.

## Optimization

Optimization happens at two levels: expression-level and array-level.

### Expression Optimization

Before an expression is applied to an array, it is optimized in a loop that runs three passes
to convergence:

1. **Untyped simplification** -- rewrites that do not depend on input types. For example,
   masking with a literal `true` eliminates the mask operation entirely.

2. **Typed simplification** -- rewrites that use type information. For example, a cast from
   `u32` to `u32` is a no-op and can be eliminated.

3. **Abstract reduction** -- custom reduction rules defined by each expression vtable. These
   allow expression-specific rewrites such as flattening nested boolean conjunctions.

The loop repeats until no pass produces a change, or a maximum iteration count is reached.

### Array Optimization

After a `ScalarFnArray` is constructed, array-level reduction rules are applied. These operate
on the concrete array tree rather than the abstract expression tree:

- **Pack-to-struct fusion** -- a `ScalarFnArray` wrapping a pack expression is immediately
  reduced to a `StructArray`, avoiding an extra layer of indirection.

- **Constant folding** -- a `ScalarFnArray` whose children are all `ConstantArray`s is
  evaluated at construction time.

- **Filter pushdown** -- when a `ScalarFnArray` with a single non-constant input appears
  inside a `FilterArray`, the filter is pushed down to the input, preventing full
  materialization before filtering.

- **Parent reduction** -- child arrays can inspect their parent context and propose rewrites.
  This allows encodings to participate in cross-layer optimization.

## Execution

When a `ScalarFnArray` is finally executed (via `execute()` or `to_canonical()`), it evaluates
its scalar function on its children and returns the result as a canonical array. Children that
are themselves `ScalarFnArray`s are executed recursively, bottom-up.

For single-element access (`scalar_at`), each child is evaluated at the requested index and
the scalar function is applied to the resulting scalars, avoiding full array materialization.
