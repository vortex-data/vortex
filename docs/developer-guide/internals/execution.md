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

The execution loop runs until the array is canonical or constant. Each iteration runs
four steps in order, stopping as soon as one succeeds:

0. **Canonical check** -- if the array is already canonical, return it.

1. **`reduce`** -- call the array's own `VTable::reduce`. This is a metadata-only rewrite
   that does not read buffers or execute children. For example, `ChunkedVTable::reduce`
   unwraps a single-chunk chunked array into the chunk itself.

2. **`reduce_parent`** -- iterate over the array's children and call each child's
   `VTable::reduce_parent`. This lets a child encoding rewrite the parent using only
   metadata. For example, when a `ConstantArray` is the child of a `FilterArray`, it can
   replace the entire filter with a new constant of the filtered length -- no buffers
   touched.

3. **`execute_parent`** -- iterate over the array's children and call each child's
   `VTable::execute_parent`. This is like `reduce_parent` but may read buffers and execute
   sub-arrays. For example, when a `ChunkedArray` is the child of a `FilterArray`, it splits
   the mask across chunks and filters each one independently.

4. **`execute`** -- call the array's own `VTable::execute`. This is the fallback: execute
   children or decode the array one step toward canonical form.

Steps 1 and 2 are cheap metadata rewrites. Steps 3 and 4 may do real work. The separation
ensures that the cheapest optimizations always run first.

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

## Rules and Kernels

The four execution steps are the framework. The actual optimizations are implemented by
encoding authors through two mechanisms: **rules** (metadata-only rewrites that plug into
steps 1-2) and **kernels** (execution with buffer access that plug into step 3). Together,
they let each encoding define how it interacts with operations like filter, take, and slice
without modifying the operation itself.

### Lazy Operation Arrays

Operations in Vortex are lazy. Calling `array.filter(mask)` does not filter anything -- it
constructs a `FilterArray` wrapping the original array and the mask. The tree looks like:

```
filter(mask):
  child: <some encoding>(...)
```

Similarly, `array.take(indices)` builds a `DictArray`, and `array.slice(start, end)` builds
a `SliceArray`. Execution of the tree happens later, giving the rule and kernel system a
chance to optimize.

### The Reduce Traits: Metadata-Only Rewrites

A **reduce rule** rewrites an array using only its metadata and structure -- no buffer reads.
There are two flavors:

**`ArrayReduceRule<V>`** rewrites the array itself (step 1). The encoding implements
`VTable::reduce` and typically delegates to a `ReduceRuleSet`:

```rust
// ChunkedVTable::reduce: unwrap trivial chunked arrays
fn reduce(array: &ChunkedArray) -> VortexResult<Option<ArrayRef>> {
    Ok(match array.chunks.len() {
        0 => Some(Canonical::empty(array.dtype()).into_array()),
        1 => Some(array.chunks[0].clone()),
        _ => None,
    })
}
```

**`ArrayParentReduceRule<V>`** rewrites the *parent* from the child's perspective (step 2).
The child encoding implements rules keyed on the parent type via the `Matcher` trait:

```rust
pub trait ArrayParentReduceRule<V: VTable> {
    type Parent: Matcher;  // which parent type this rule applies to

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}
```

Rules are collected into a `ParentRuleSet` -- a static list attached to the encoding. When
the executor calls `reduce_parent` on a child, the child's vtable iterates its
`ParentRuleSet`, matching each rule against the parent's type. The first rule that returns
`Some` wins.

### The Kernel Traits: Execution with Buffer Access

When metadata-only rewrites are not enough, the encoding registers **kernels** that may read
buffers and execute sub-arrays. These plug into step 3 (`execute_parent`):

```rust
pub trait ExecuteParentKernel<V: VTable> {
    type Parent: Matcher;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
```

Like rules, kernels are collected into a `ParentKernelSet`. The difference is that kernels
receive an `ExecutionCtx` and are allowed to execute children, allocate buffers, and do real
work.

### Operation-Specific Traits

Each operation defines a pair of traits that encoding authors implement directly, plus
adaptors that wire them into the rule/kernel system:

| Operation | Reduce trait (metadata-only) | Kernel trait (buffer access) |
|-----------|------------------------------|------------------------------|
| Filter    | `FilterReduce`               | `FilterKernel`               |
| Take      | `TakeReduce`                 | `TakeExecute`                |
| Slice     | `SliceReduce`                | `SliceKernel`                |

For example, the filter traits:

```rust
pub trait FilterReduce: VTable {
    fn filter(array: &Self::Array, mask: &Mask) -> VortexResult<Option<ArrayRef>>;
}

pub trait FilterKernel: VTable {
    fn filter(array: &Self::Array, mask: &Mask, ctx: &mut ExecutionCtx)
        -> VortexResult<Option<ArrayRef>>;
}
```

Each trait has an adaptor struct (`FilterReduceAdaptor`, `FilterExecuteAdaptor`) that
converts it into an `ArrayParentReduceRule` or `ExecuteParentKernel`. The adaptors also
handle common preconditions -- for filter, they short-circuit all-true and all-false masks
before calling the encoding's implementation.

### Wiring It Up: ParentRuleSet and ParentKernelSet

An encoding registers its rules and kernels as static constants. Here is
`ConstantVTable`'s parent rule set:

```rust
const PARENT_RULES: ParentRuleSet<ConstantVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ConstantFilterRule),           // custom rule
    ParentRuleSet::lift(&CastReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&TakeReduceAdaptor(ConstantVTable)),
]);
```

And `ChunkedVTable`'s parent kernel set:

```rust
static PARENT_KERNELS: ParentKernelSet<ChunkedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ChunkedVTable)),
]);
```

The `lift` function converts a zero-sized rule/kernel struct into a dynamic trait object at
compile time. The vtable then delegates to these sets from the corresponding `VTable`
methods:

```rust
// In ChunkedVTable
fn reduce_parent(array: &ChunkedArray, parent: &ArrayRef, child_idx: usize)
    -> VortexResult<Option<ArrayRef>>
{
    PARENT_RULES.evaluate(array, parent, child_idx)
}

fn execute_parent(array: &ChunkedArray, parent: &ArrayRef, child_idx: usize, ctx: &mut ExecutionCtx)
    -> VortexResult<Option<ArrayRef>>
{
    PARENT_KERNELS.execute(array, parent, child_idx, ctx)
}
```

### Example: Filter on a ConstantArray

Consider filtering a constant array of value `42` with length 1000 by some boolean mask.
The array tree looks like:

```
filter(mask):          # len = mask.true_count()
  child: constant(42)  # len = 1000
```

Execution proceeds:

1. **`reduce`**: `FilterVTable::reduce` runs its own rules -- no match here.
2. **`reduce_parent`**: the executor iterates children. Child 0 is the `ConstantArray`.
   `ConstantVTable::reduce_parent` evaluates `PARENT_RULES`. The `ConstantFilterRule`
   matches `FilterArray` as the parent:

   ```rust
   fn reduce_parent(&self, child: &ConstantArray, parent: &FilterArray, _child_idx: usize)
       -> VortexResult<Option<ArrayRef>>
   {
       Ok(Some(ConstantArray::new(child.scalar.clone(), parent.len()).into_array()))
   }
   ```

   It returns `ConstantArray(42, len=mask.true_count())` -- the filtered result without
   reading a single buffer. Execution is done in one step.

### Example: Filter on a ChunkedArray

Now consider filtering a chunked array of three chunks. The tree looks like:

```
filter(mask):
  child: chunked([chunk0, chunk1, chunk2])
```

Execution proceeds:

1. **`reduce`**: no match.
2. **`reduce_parent`**: `ChunkedVTable`'s parent rules do not include a filter reduce
   (chunked filter needs buffer access to split the mask across chunks), so no match.
3. **`execute_parent`**: `ChunkedVTable`'s parent kernels include
   `FilterExecuteAdaptor(ChunkedVTable)`. It matches `FilterArray` as the parent and calls
   `ChunkedVTable`'s `FilterKernel::filter`:

   ```rust
   fn filter(array: &ChunkedArray, mask: &Mask, _ctx: &mut ExecutionCtx)
       -> VortexResult<Option<ArrayRef>>
   {
       // Split mask across chunk boundaries, filter each chunk independently
       let chunks = match mask_values.threshold_iter(selectivity_threshold) {
           MaskIter::Indices(indices) => filter_indices(array, indices),
           MaskIter::Slices(slices) => filter_slices(array, slices),
       }?;
       Ok(Some(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array()))
   }
   ```

   The result is a new `ChunkedArray` whose chunks have each been individually filtered.
   The per-chunk `filter` calls are themselves lazy -- they create `FilterArray` wrappers
   around each chunk, which will be resolved when those chunks are later executed.

### Example: Scalar Function Pushdown through Chunked

The rule system is not limited to filter/take/slice. Chunked arrays also define rules for
pushing scalar functions through their chunks. Consider:

```
scalar_fn(cast to f64):
  child: chunked([chunk0, chunk1])
```

During step 2, `ChunkedVTable::reduce_parent` matches `ChunkedUnaryScalarFnPushDownRule`.
This rule rewrites the tree into:

```
chunked:
  - scalar_fn(cast to f64):
      child: chunk0
  - scalar_fn(cast to f64):
      child: chunk1
```

The cast is pushed inside each chunk, where chunk-specific optimizations can apply. For
example, if `chunk0` is a `ConstantArray`, its own cast rule can fold the cast into the
scalar directly.

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
