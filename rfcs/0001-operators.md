# Arrays as a Logical Plan

This RFC proposes to turn Vortex arrays into logical query plans to support multiple modes of execution. Compute
functions become lazy and are modelled as arrays.

N.B. within Spiral we have often referred to this proposal as the “Operator Model”

* **Proposed**: @gatesn
* **Date**: September 18, 2025
* **Status**: In Review
* **Prototype
  **: [https://github.com/vortex-data/vortex/compare/ngates/operator](https://github.com/vortex-data/vortex/compare/ngates/operator?expand=1)

## Motivations

1. **Predictable Compute Cost**

Compute functions in Vortex currently take and return ArrayRefs, meaning they have full decision-making control (and
responsibility) over how much or little compute to actually perform.

For example, `filter(DictArray, Mask)` could fully evaluate the filter, or it could push down the filter over the
dictionary codes with an unknown cost.

2. **Query Planning**

Having a tree of arrays represent a logical execution plan allows us to implement alternate modes of compute. We
currently have three modes in mind:

- Batch Compute - operates over a full array, returning a canonical array.
- Pipelined Compute - operates over vectors of 1024 elements at a time, improving CPU cache locality getting much closer
  to the performance of fused decompression kernels.
- GPU Compute - self-explanatory

3. **Async Compute**

We would like to support async compute functions in the future, for example mapping strings through an HTTP API.
We also may want to move management of Vortex buffers into a centralized buffer pool, allowing us to spill-to-disk
or hold buffers on alternate devices (e.g. GPU memory).

For both of these use-cases, it's helpful to have our batch execution function be async.

## Vectors

As we are re-defining arrays to be a logical plan with logical children, we need a way to represent in-memory
canonicalized data. This is currently modelled using the `Canonical` enum, but this RFC proposes to replace this with a
new `Vector` enum holding recursively canonical vector data.

Similar to canonical there is one vector variant per Vortex DType. These variants use an in-memory format that is
heavily inspired by Arrow. The major difference, and the reason we define our own Vector types rather than reusing
Arrow arrays directly, is that we support arbitrary width auxiliary vectors such as those used for dictionar codes or
run-end offsets. We have found significant speed-ups when using the narrowest integer types possible for these vectors.

```rust
enum Vector {
    Null(NullVector),
    Bool(BoolVector),
    Primitive(PVector),
    // ...
}

enum PVector {
    U8(PrimitiveVector<u8>),
    // ...
}
```

Each vector has an associated _mutable_ version. These are owned structs modelled after the Tokio `BytesMut` type, the
same way `vortex-buffer` has `Buffer<T>` and `BufferMut<T>`. This design allows us to pass pre-allocated vectors to
compute kernels for in-place mutation. The primary use-case is to reduce copies when executing chunked arrays by
passing each chunk its own pre-allocated slice of a contiguous output vector.

```rust
enum VectorMut {
    Null(NullVectorMut),
    Bool(BoolVectorMut),
    Primitive(PVectorMut),
    // ...
}

impl VectorMut {
    fn freeze(self) -> Vector { ... }
    fn split_off(&mut self, at: usize) -> VectorMut { ... }
    fn unsplit(&mut self, other: VectorMut) -> VectorMut { ... }
}
```

## Arrays as a Logical Plan

Vortex implements a subset of functionality usually performed by a query engine. Specifically, we implement a highly
optimized physical Scan operation that supports pushdown of filters, projections, and scalar expressions. Another way
of looking at this, is that Vortex performs optimized evaluation of a scalar compute functions.

The new Vortex `Array` trait will subsume much of the functionality currently found in compute functions and
expressions. So in addition to the existing API, arrays will support push-down and pull-up optimizations, as well as
expose falsification transformations to construct pruning predicates.

As expected from a logical plan, all operations over arrays will have cost proportional to the size of the array tree
rather than the size of the data represented by the array.

The primary operation supported by arrays is `into_vector`, an operation which takes an optional selection mask and
returns an async kernel for resolving the array into a canonical vector.

### Exporting Compressed Arrays

Many query engines support partially compressed data (e.g. DuckDB, Velox) rather than fully decompressed Arrow (e.g.
DataFusion, Polars).

For these engines, we propose that the export logic inspects the root node of the array for compatible compression
codecs and deconstructs the array as required.

For example, DuckDB supports dictionary-encoded vectors. So we would try to downcast the root node as a DictArray, and
if successful, canonicalize the codes and values children as two separate executions.

## Expressions as an Abstract Plan

As part of this change, the `vortex-expr` crate can be significantly simplified.

* Each expression becomes an abstract node in a tree containing an `id`, `metadata`, and `children`. The leaves of
  this tree are represented by the placeholder expression called `scope` (often denoted as `$`).
* Expressions can be converted to arrays by first converting their children to arrays, then calling the `build`
  function of the array encoding indicated by the `id` of the expression, along with the serialized `metadata`. This
  process results in a bound array now with known `len` and `dtype`.
* All evaluation logic (for executing the expression), and analysis logic (for producing pruning predicates) will be
  moved to the `Array` trait.

Expressions simply provide a way of describing an unbound query over a root scope array and make up the user-facing
API of the scan operator.

## Layouts and Scanning

Vortex scans currently operate by converting a `Layout` into a `LayoutReader`. In practice, this is largely equivalent
to a current Vortex `Array`, except the execution function of the `LayoutReader` is async. In the new oeprator model,
arrays _are_ async, and so we can remove `LayoutReader` entirely in favor of regular Vortex arrays.

This change further allows us to simplify scanning logic and avoid the current expression partitioning. Instead of
splitting expressions and pushing the partitions down over the fields of a struct layout, we simply apply the expression
to a row-range of a layout to produce an array: `Layout::apply(Expr, RowRange) -> ArrayRef`. A `FlatLayout` would be
modelled as a node in the array tree allowing us to perform partial optimizations over the tree, such as pushing
`get_item` expressions through a `StructLayout`.

The scan would then find all `FlatLayout` nodes and resolve their segment IDs to load an array for each segment. These
nodes will be swapped out for the loaded arrays, the optimizer re-run to push compute into the flat layouts, and then
the root array can be evaluated.

## Optimizations

### Common Subtree Elimination

One of the primary benefits of this change is to avoid decompressing data multiple times during a scan. Currently, we
split a filter expression into conjuncts and run each one separately over the compressed columns, followed by a
projection expression. Each one of these evaluations can result in a full decompression of the same data.

For example, many queries in our benchmarks currently perform double string decompression if the string column appears
in both a filter and the projection.

I expect to see significant performance improvements from this change alone.

### Push-down and Pull-up Optimizations

Each array can offer both push-down and pull-up optimizations. We expose functions for arrays to optimize their
children, or optimize their parent assuming they are one of the children.

We always run reduce_children *before* reduce_parent.

Reduction functions are not recursive.

Note that we could perform child reduction during construction, but I think it’s better to allow for deferred and
optional optimization.

The APIs for reduction will look something like:

```rust
trait Array {
    /// Attempt to push down the array over its children,   
    fn reduce_children(&self) -> Result<Option<ArrayRef>>;

    /// Given a parent array where we are the n'th child (child_idx), return a
    /// possibly new parent array.
    fn reduce_parent(&self, parent: ArrayRef, child_idx: usize) -> Result<Option<ArrayRef>>;
}
```

**Example 1: Reduce Children - Constant Folding**

```
let array = ArithmeticArray:
    op: subtract
    lhs: Constant{value = 3}
    rhs: Constant{value = 1}

array.reduce_children() -> Constant{value = 2}
```

**Example 2: Reduce Children - Push Down**

```
let array = ArithmeticArray:
    op: subtract
    lhs: DictArray:
        codes: IntArray
        values: IntArray
    rhs: Constant{value = 1}

array.reduce_children() -> DictArray:
    codes: IntArray
    values: ArithmeticArray:
        op: subtract
        lhs: IntArray
        rhs: Constant{value = 1}
```

**Example 3: Reduce Parent - Pull Down**

```
let array = DictArray:
    codes: IntArray
    values: IntArray

let parent = ArithmeticArray:
    op: subtract
    lhs: array
    right: Constant{value = 1}

array.reduce_parent(parent, 0) -> DictArray:
    codes: IntArray
    values: ArithmeticArray:
        op: subtract
        lhs: IntArray
        rhs: Constant{value = 1}
```

Note that example 2 and 3 are the same, one is performed via reduce_children, and one is performed via reduce_parent.

Who implements the optimization rule is determined by who is aware of the properties of the other array.

For example, the types of operations that can be pushed down through a dictionary depends on whether the operation is
aware of null values, and then whether the dictionary has nulls only in its codes array, or in both codes and values.
For this reason, example 2 is unlikely to be implemented in practice since the push-down rules for dictionary arrays are
complex enough only to be known by the dictionary array itself.

### Definitions

```rust
trait Array {
    fn len(&self) -> usize;
    fn dtype(&self) -> &DType;

    ...

    // Inspection APIs for push-down / pull-up optimization
    fn children(&self) -> &[ArrayRef];
    fn with_children(self: Arc<dyn Self>, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;

    // Whether if the child is scalar w.r.t the current array, whether it's 
    // null-aware, other questions that optimizers may want to ask.
    fn child_info(&self, idx: usize) -> ChildInfo;

    ...

    // Downcast for different execution modes
    fn as_batch(&self) -> Option<&dyn BatchOperator>;
    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator>;
    fn as_gpu(&self) -> Option<&dyn GpuOperator>;
}

struct ChildInfo {
    // N.B. These properties will evolve as we attempt to implement optimization rules.
    aligned: bool, // Whether the child elements align 1:1 with the parent elements.
    propagates_null: bool, // Whether the child nulls align with the parent nulls.
}
```

## Array API

This change essentially collapses the API for an array down to just an async “into_vector” function. Almost everything
can be expressed via this function, and it means we can strip down the surface area required for array implementations.

For example, functions relating to validity can be removed and implemented using an IsNull operator pushed down over the
array tree and evaluated into a boolean vector. Even scalar_at could be implemented with a singleton selection array,
evaluated to a vector, and then converted to a scalar.

Given that this model involves many more array implementations than before (i.e. every compute function), it is
important that we keep the overhead of implementing a new array relatively low.

## Execution Modes

I will leave JIT execution aside for now, and discuss the other three execution modes: Batch, Pipelined, and GPU.

### Batch Execution

Batch execution computes over an entire array, taking canonicalized children and returning a canonicalized result.

We choose to make batch execution an async API to support async arrays in the future. This could be for several reasons,
including lazy or spilled buffer handles, or more conventional async expressions such as mapping each row through some
HTTP API.

```rust
trait Array {
    ...

    /// Even batch execution is optional, as some arrays may be pure placeholders
    /// and themselves have no evaluation logic.
    fn as_batch(&self) -> Option<&dyn BatchOperator>;
}

trait BatchOperator {
    fn bind(&self, ctx: BatchBindContext) -> VortexResult<Box<dyn BatchExecution>>;
}

struct BatchBindContext {
    children: Vec<Box<dyn BatchExecution>>
}

#[async_trait]
trait BatchExecution {
    /// Execute the array, producing a canonical vector result.
    async fn execute(self: Box<Self>, output: VectorMut) -> VortexResult<Vector>;
}
```

Batch execution is the primary mode for executing a canonicalization. The other modes of execution are optional
optimizations within a batch evaluation.

### Pipelined Execution

For deeply nested array trees there can be significant benefit to passing a small chunk of data through the entire
pipeline of array operators at once, keeping much of the data in the L1 CPU cache.

Early experiments have shown up to 4x improvements in performance.

Pipelines operate over vectors of 1024 elements at a time, until the final vector which may be shorter. They are not
allowed to perform any I/O or async operations.

The execution engine looks at the tree of arrays and finds subgraphs where all nodes support pipelined execution. These
subgraphs are collapsed into a single PipelineBatchExecution node that executes the subgraph step by step.

During construction, the pipeline goes through common sub-tree elimination, followed by a vector allocation stage. This
stage computes the minimal number of vectors required for execution based on the topological sort of the pipeline DAG (
essentially the same problem as register allocation in a compiler). By minimizing the number of intermediate vectors, we
can maximize our use of CPU caches. Where possible, pipeline operators will write directly into the output vector.

```rust
trait PipelinedOperator {
    fn bind(&self, ctx: &PipelineBindContext) -> VortexResult<Box<dyn Kernel>>;

    /// Returns the child indices of this operator that are passed to the kernel as input vectors.
    fn vector_children(&self) -> Vec<usize>;

    /// Returns the child indices of this operator that are passed to the kernel as batch inputs.
    fn batch_children(&self) -> Vec<usize>;

    /// Whether the operator can mutate its first child in-place to create
    /// its output.
    ///
    /// If true, the data for the operator's first child will be passed in via
    /// the mutable output buffer.
    fn in_place(&self) -> bool;
}

trait PipelineBindContext {
    fn children(&self) -> &[VectorId];
    fn batch_inputs(&self) -> &[BatchId];
}

trait Kernel {
    /// Step the execution of the pipeline.
    fn step(&mut self, ctx: &KernelCtx, out: &mut ViewMut) -> VortexResult<()>;
}

trait KernelCtx {
    fn vector(&self, id: VectorId) -> VectorRef<'_>;
    fn batch_input(&self, id: BatchId) -> &Canonical;
}
```

### GPU Execution

Similar to pipelined execution, the executor will find subgraphs of the array tree that are eligible for execution on
the GPU (ideally, we reach full compatibility and the entire graph can be moved).

During the bind phase, arrays buffers are moved over from CPU to GPU memory. Note that these buffers contain fully
compressed data since they live in the leaves of the compressed array tree. In the future, we should be able to hint to
the vortex-scan logic that buffers should be loaded directly from files/network into the GPU, avoiding the compressed
CPU buffers.

```rust
trait GpuOperator {
    fn bind(&self, ctx: &GpuBindContext) -> VortexResult<Box<dyn GpuExecution>>;
}

trait GpuExecution {
    ...
}
```

My understanding of GPU compute is still pretty slim, so would welcome suggestions here. But as I understand it, we have
a couple of options (note we only intend to support Nvidia GPUs for now):

- Produce compiled kernels for each operator, possibly via [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
- Leverage existing kernels from the RAPIDS ecosystem, e.g. via [cuDF](https://docs.rapids.ai/api/cudf/stable/)

## Selection Pushdown

One of the main contributors to the performance of Vortex is our ability to perform selection pushdown over compressed
data.

We currently do this in a very naive way…

```rust
if mask.density() < 0.2 {
array = expr.evaluate(array.filter(mask));
} else {
array = expr.evaluate(array).filter(mask);
}
```

I am also shocked at how well this has worked for us thus far.

In the operator model, the evaluation function takes an optional selection mask itself implemented as an Array.
Each operator can bind each of its children using the mask, using a projection of the mask, or using no mask. For
example:

```rust
trait Array {
    /// Create a kernel for resolving this array into a vector with the given selection mask.
    fn bind(&self, ctx: &dyn BindContext, selection: Option<&ArrayRef>) -> VortexResult<Box<dyn BatchKernel>>;
}
```

Some arrays may choose to construct multiple kernels for different selection densities, and then choose which kernel
to execute at runtime based on the density of the selection mask.

```rust
impl BatchKernel for FooArrayKernel {
    async fn execute(self: Box<Self>, out: VectorMut) -> VortexResult<Vector> {
        let mask = self.selection.execute(..).await?;
        if mask.density() < 0.2 {
            self.low_density.execute(out).await
        } else {
            self.high_density.execute(out).await
        }
    }
}
```

## Benchmarking

Because our current compute model in Vortex allows for an arbitrary amount of eager vs lazy compute, our benchmarks are
somewhat meaningless. Making a compute function lazy may improve benchmarks by an order of magnitude, only to result in
a slower canonicalization function later on.

With this new mode, all arrays produce canonical results, meaning benchmarks for individual operations are much more
stable and useful.

We will still need complex e2e benchmarks to analyze the effect of optimization passes.

## Migration Path

This is a very large change that touches all arrays and fundamentally changes the array trait. I propose we roughly:

1. Create a `vortex-vector` create holding the new recursively canonical `Vector` types.
2. Create a new `Array` trait modelling the logical plan as described in this RFC.
3. Create a new `Expression` struct modelling the abstract expression plan as described in this RFC.
3. Create new array implementations for each compute function, implementing the new `Array` trait.
4. Implement the `Layout::apply` function to convert layouts and expressions into arrays.
5. Implement an alternate scan operator that uses the new array and expression types.
6. Cut over the public array API to the new `Array` trait (these are largely compatible).

## Future Work

* **Physical Types** - In the future, I imagine the strict set of canonical vectors becomes too limiting for some  
  use-cases and we may have a physical type system attached to the array tree. In this world, a `PhysicalCast` array
  would allow defining the desired output vector and can be pushed down over the array tree as with other operators.
  This can help avoid intermediate conversions such as when reading a `VarBin` array from disk, performing a string
  operation over it (forcing a conversion to `VarBinView`), then returning an `arrow::VarBin` array as Arrow.
