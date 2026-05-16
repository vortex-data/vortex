# Lowering API

> **Status:** Accepted.
> **Progress:** Authoritative reference for the plan-time API — the
> `Operator` trait, `PhysicalPlan`, `LoweringCtx`, `PipelineTail`, and
> the lowered artifact. Implementing code lives in
> `src/physical_plan/plan.rs` and `src/physical_plan/lowering.rs`.
> **Open questions:** none.

The plan-time API is the surface a planner uses to hand the engine
a `PhysicalPlan` and have it lowered into a runnable pipeline DAG.
Three concepts: an `Operator` trait that operators implement, a
`LoweringCtx` that operators call into, and a `PipelineTail`
continuation that flows through `lower`.

## `Operator`

```rust
pub trait Operator: Send + Sync {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()>;
}
```

One method. The operator receives a `PipelineTail` (the
continuation representing the rest of the pipeline from this point
toward its sink) and does one of three things:

1. **Streaming**: `tail.prepend_transform(...)` and recurse into
   the child operator with the new tail.
2. **Source (leaf)**: `ctx.emit_pipeline(tail, ..., source)` to
   complete the tail. Ends the recursion.
3. **Pipeline-breaker or multi-input**: build per-input shared
   state, lower each input with a fresh tail that ends at a sink
   writing into the shared state and `publishes(barrier)`, then
   complete the outer tail with a source that reads the shared
   state and `depends_on(barrier)`.

`Operator` is `Send + Sync` so operator subtrees can be stored on
`Send`/`Sync` types and lowered from any thread.

## `PhysicalPlan`

```rust
pub struct PhysicalPlan { /* output: OutputContract, root: Box<dyn Operator> */ }

impl PhysicalPlan {
    pub fn new(output: OutputContract, root: Box<dyn Operator>) -> Self;

    pub fn output(&self) -> &OutputContract;
    pub fn root(&self) -> &dyn Operator;

    pub fn lower<E: SinkNode>(
        &self,
        ctx: &mut dyn LoweringCtx,
        output_domain: Domain,
        sink: E,
    ) -> BuildResult<()>;

    pub fn validate(&self) -> Result<(), PlanValidationError>;
}
```

A `PhysicalPlan` wraps the root `Operator` plus the
`OutputContract` describing the plan's overall output stream. The
caller picks a `SinkNode` for the result (e.g. `CollectSink`,
`CountDistinctI64`) and calls `plan.lower(ctx, output_domain,
sink)`. `lower` registers the output domain with the ctx and
recurses into the root.

## `LoweringCtx`

```rust
pub trait LoweringCtx {
    fn register_domain(&mut self, domain: Domain) -> BuildResult<()>;
    fn new_pipeline_barrier(&mut self) -> PipelineBarrier;
    fn emit_pipeline_dyn(
        &mut self,
        tail: PipelineTail,
        output_domain: Domain,
        output_contract: OutputContract,
        source: PipelineSource,
    ) -> BuildResult<PipelineId>;
}

pub trait LoweringCtxExt: LoweringCtx {
    fn emit_pipeline<E: SourceNode>(
        &mut self,
        tail: PipelineTail,
        output_domain: Domain,
        output_contract: OutputContract,
        source: E,
    ) -> BuildResult<PipelineId>;
}
impl<T: LoweringCtx + ?Sized> LoweringCtxExt for T {}
```

`LoweringCtx` is dyn-compatible so it can be passed through
`Operator::lower`'s `&mut dyn LoweringCtx`. `LoweringCtxExt` adds
the typed `emit_pipeline<E: SourceNode>` convenience.

Three calls cover the API:

- `register_domain(domain)` declares a domain that will appear in
  the lowered plan. Calling it twice with the same `DomainId` but
  different `Domain` definitions is a build error.
- `new_pipeline_barrier()` allocates a fresh `PipelineBarrier` id.
- `emit_pipeline(tail, output_domain, output_contract, source)`
  emits one pipeline by attaching `source` to `tail`. The
  source's declared output is validated against the tail's
  expected input.

## `PipelineTail`

```rust
pub struct PipelineTail { /* … */ }

impl PipelineTail {
    pub fn new<E: SinkNode>(input_domain: Domain, input_contract: OutputContract, sink: E) -> Self;

    pub fn prepend_transform<E: TransformNode>(
        self,
        input_domain: Domain,
        input_contract: OutputContract,
        transform: E,
    ) -> Self;

    pub fn depends_on(self, barrier: PipelineBarrier) -> Self;
    pub fn publishes(self, barrier: PipelineBarrier) -> Self;

    pub fn expected_input_domain(&self) -> &Domain;
    pub fn expected_input_contract(&self) -> &OutputContract;
    pub fn depends_on_set(&self) -> &BTreeSet<PipelineBarrier>;
    pub fn publishes_set(&self) -> &BTreeSet<PipelineBarrier>;
}
```

A tail accumulates transforms in reverse order toward the sink.
`prepend_transform` updates the tail's expected upstream input
shape, since the new transform now sits between the upstream and
the previous tail. `depends_on` and `publishes` accumulate the
barrier set that will be attached to the pipeline emitted from
this tail.

A tail is consumed by exactly one `ctx.emit_pipeline` call. After
that the tail moves into the pipeline.

## Lowered artifacts

```rust
pub struct LoweredPlan {
    pub pipelines: Vec<Pipeline>,
    pub domains:   BTreeMap<DomainId, Domain>,
}

pub struct Pipeline { /* … */ }
impl Pipeline {
    pub fn id(&self) -> PipelineId;
    pub fn source(&self) -> &PipelineSource;
    pub fn transforms(&self) -> &[PipelineTransform];
    pub fn sink(&self) -> &PipelineSink;
    pub fn depends_on(&self) -> &BTreeSet<PipelineBarrier>;
    pub fn publishes(&self) -> &BTreeSet<PipelineBarrier>;
}
```

A `LoweredPlan` is what `PipelineBuilder` produces:

```rust
pub struct PipelineBuilder { /* … */ }
impl PipelineBuilder {
    pub fn new() -> Self;
    pub fn into_plan(self) -> LoweredPlan;
}
impl LoweringCtx for PipelineBuilder { /* … */ }
```

A typical lowering:

```rust
let mut builder = PipelineBuilder::new();
plan.lower(&mut builder, output_domain, sink)?;
let lowered: LoweredPlan = builder.into_plan();
runtime::run_plan_blocking(lowered)?;
```

## Validation

`emit_pipeline` checks the source's output `Domain` and
`OutputContract` match the tail's expected input. A mismatch
returns `BuildError::OutputContractMismatch`.

`register_domain` checks for duplicate ids with diverging
definitions and returns `BuildError` if it sees one.

## Worked examples

### Streaming transform

```rust
impl Operator for ArrayPredicate {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            ArrayPredicateTransform::new(self.predicate.clone()),
        );
        self.input.lower(ctx, tail)
    }
}
```

### Pipeline-breaker (Sort)

```rust
impl Operator for Sort {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let runs = Arc::new(SortedRuns::default());
        let build_done = ctx.new_pipeline_barrier();

        self.input.lower(ctx,
            PipelineTail::new(self.input_domain.clone(),
                              self.input_contract.clone(),
                              SortSink::new(runs.clone(), self.keys.clone()))
                .publishes(build_done))?;

        ctx.register_domain(self.output_domain.clone())?;
        ctx.emit_pipeline(
            tail.depends_on(build_done),
            self.output_domain.clone(),
            self.output_contract.clone(),
            MergeKSource::new(runs, self.keys.clone()))?;
        Ok(())
    }
}
```

### Multi-input (SortedMergeJoin)

```rust
impl Operator for SortedMergeJoin {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let state = Arc::new(MergeJoinState::default());
        let left_done  = ctx.new_pipeline_barrier();
        let right_done = ctx.new_pipeline_barrier();

        self.left.lower(ctx,
            PipelineTail::new(self.left_domain.clone(),
                              self.left_contract.clone(),
                              MergeJoinSink::new(state.clone(), JoinSide::Left))
                .publishes(left_done))?;

        self.right.lower(ctx,
            PipelineTail::new(self.right_domain.clone(),
                              self.right_contract.clone(),
                              MergeJoinSink::new(state.clone(), JoinSide::Right))
                .publishes(right_done))?;

        ctx.register_domain(self.output_domain.clone())?;
        ctx.emit_pipeline(
            tail.depends_on(left_done).depends_on(right_done),
            self.output_domain.clone(),
            self.output_contract.clone(),
            MergeJoinSource::new(state))?;
        Ok(())
    }
}
```

Three pipelines, two barriers. The output pipeline cannot start
until both build barriers fire.

## See also

- [Runtime traits](runtime-traits.md) for `SourceNode`,
  `TransformNode`, `SinkNode`, `Batch`, and the ctx types.
- [Spawn primitives](spawn-primitives.md) for the offload
  primitives operators use inside their `poll_*` bodies.
- [Execution model](../concepts/execution-model.md) for the
  concept-level overview.
