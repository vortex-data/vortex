//! Pipeline lowering primitives.
//!
//! `Operator::lower(ctx, tail)` is the plan-time API. The `tail` is
//! a continuation that flows down toward the source(s); each
//! operator either prepends a transform onto it, completes it at a
//! source via `ctx.emit_pipeline`, or splits it (multi-input /
//! pipeline-breaker) by recursing into upstream operators with new
//! tails and then emitting an output pipeline that depends on
//! barriers the upstream pipelines publish.
//!
//! No row demand, no cancellation, no demand publishers.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Domain;
use crate::DomainId;
use crate::OutputContract;
use crate::physical_plan::abi::DynSinkNode;
use crate::physical_plan::abi::DynSourceNode;
use crate::physical_plan::abi::DynTransformNode;
use crate::physical_plan::abi::Parallelism;
use crate::physical_plan::abi::SinkNode;
use crate::physical_plan::abi::SourceNode;
use crate::physical_plan::abi::TransformNode;
use crate::physical_plan::abi::TypedSinkNode;
use crate::physical_plan::abi::TypedSourceNode;
use crate::physical_plan::abi::TypedTransformNode;
use crate::physical_plan::error::BuildError;
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::ids::PipelineBarrier;
use crate::physical_plan::ids::PipelineId;

/// One source instance carried by a lowered pipeline.
pub struct PipelineSource {
    output_domain: Domain,
    output_contract: OutputContract,
    node: Box<dyn DynSourceNode>,
}

impl PipelineSource {
    pub(crate) fn new<E>(output_domain: Domain, output_contract: OutputContract, node: E) -> Self
    where
        E: SourceNode,
    {
        Self {
            output_domain,
            output_contract,
            node: Box::new(TypedSourceNode::new(node)),
        }
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }

    pub fn output_contract(&self) -> &OutputContract {
        &self.output_contract
    }

    pub fn label(&self) -> &str {
        self.node.label()
    }

    pub fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    pub fn node(&self) -> &dyn DynSourceNode {
        self.node.as_ref()
    }
}

/// One sink instance terminating a lowered pipeline.
pub struct PipelineSink {
    input_domain: Domain,
    input_contract: OutputContract,
    node: Box<dyn DynSinkNode>,
}

impl PipelineSink {
    pub(crate) fn new<E>(input_domain: Domain, input_contract: OutputContract, node: E) -> Self
    where
        E: SinkNode,
    {
        Self {
            input_domain,
            input_contract,
            node: Box::new(TypedSinkNode::new(node)),
        }
    }

    pub fn input_domain(&self) -> &Domain {
        &self.input_domain
    }

    pub fn input_contract(&self) -> &OutputContract {
        &self.input_contract
    }

    pub fn label(&self) -> &str {
        self.node.label()
    }

    pub fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    pub fn node(&self) -> &dyn DynSinkNode {
        self.node.as_ref()
    }
}

/// One transform instance carried by a lowered pipeline.
pub struct PipelineTransform {
    node: Box<dyn DynTransformNode>,
}

impl PipelineTransform {
    pub(crate) fn new<E>(node: E) -> Self
    where
        E: TransformNode,
    {
        Self {
            node: Box::new(TypedTransformNode::new(node)),
        }
    }

    pub fn label(&self) -> &str {
        self.node.label()
    }

    pub fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    pub fn node(&self) -> &dyn DynTransformNode {
        self.node.as_ref()
    }
}

/// A `PipelineTail` represents "what remains of the pipeline from
/// this point down toward its sink." Operators receive a tail from
/// their downstream consumer and either prepend a transform onto it,
/// terminate it by attaching a source via `ctx.emit_pipeline`, or
/// split it by recursing into multiple inputs.
pub struct PipelineTail {
    expected_input_domain: Domain,
    expected_input_contract: OutputContract,
    depends_on: BTreeSet<PipelineBarrier>,
    publishes: BTreeSet<PipelineBarrier>,
    /// Transforms in *reverse* order (closest-to-sink first). When
    /// the tail is emitted, the runtime reverses this so the runtime
    /// pipeline reads source-to-sink.
    transforms_rev: Vec<PipelineTransform>,
    sink: PipelineSink,
}

impl PipelineTail {
    /// Create a fresh tail ending in `sink`. `input_domain` and
    /// `input_contract` describe the row stream the upstream
    /// eventually needs to provide.
    pub fn new<E>(input_domain: Domain, input_contract: OutputContract, sink: E) -> Self
    where
        E: SinkNode,
    {
        Self::from_sink(PipelineSink::new(input_domain, input_contract, sink))
    }

    pub(crate) fn from_sink(sink: PipelineSink) -> Self {
        Self {
            expected_input_domain: sink.input_domain().clone(),
            expected_input_contract: sink.input_contract().clone(),
            depends_on: BTreeSet::new(),
            publishes: BTreeSet::new(),
            transforms_rev: Vec::new(),
            sink,
        }
    }

    pub fn expected_input_domain(&self) -> &Domain {
        &self.expected_input_domain
    }

    pub fn expected_input_contract(&self) -> &OutputContract {
        &self.expected_input_contract
    }

    /// Prepend a synchronous transform; the caller declares the
    /// input shape it needs from upstream.
    pub fn prepend_transform<E>(
        mut self,
        input_domain: Domain,
        input_contract: OutputContract,
        transform: E,
    ) -> Self
    where
        E: TransformNode,
    {
        self.expected_input_domain = input_domain;
        self.expected_input_contract = input_contract;
        self.transforms_rev.push(PipelineTransform::new(transform));
        self
    }

    /// Declare that this pipeline must not start until `barrier` has
    /// been published.
    pub fn depends_on(mut self, barrier: PipelineBarrier) -> Self {
        self.depends_on.insert(barrier);
        self
    }

    /// Declare that completing this pipeline publishes `barrier`.
    pub fn publishes(mut self, barrier: PipelineBarrier) -> Self {
        self.publishes.insert(barrier);
        self
    }

    pub fn depends_on_set(&self) -> &BTreeSet<PipelineBarrier> {
        &self.depends_on
    }

    pub fn publishes_set(&self) -> &BTreeSet<PipelineBarrier> {
        &self.publishes
    }

    fn into_parts(
        self,
    ) -> (
        Domain,
        OutputContract,
        BTreeSet<PipelineBarrier>,
        BTreeSet<PipelineBarrier>,
        Vec<PipelineTransform>,
        PipelineSink,
    ) {
        (
            self.expected_input_domain,
            self.expected_input_contract,
            self.depends_on,
            self.publishes,
            self.transforms_rev,
            self.sink,
        )
    }
}

/// One fully-lowered pipeline.
pub struct Pipeline {
    id: PipelineId,
    source: PipelineSource,
    transforms: Vec<PipelineTransform>,
    sink: PipelineSink,
    depends_on: BTreeSet<PipelineBarrier>,
    publishes: BTreeSet<PipelineBarrier>,
}

impl Pipeline {
    pub fn id(&self) -> PipelineId {
        self.id
    }

    pub fn source(&self) -> &PipelineSource {
        &self.source
    }

    pub fn transforms(&self) -> &[PipelineTransform] {
        &self.transforms
    }

    pub fn sink(&self) -> &PipelineSink {
        &self.sink
    }

    pub fn depends_on(&self) -> &BTreeSet<PipelineBarrier> {
        &self.depends_on
    }

    pub fn publishes(&self) -> &BTreeSet<PipelineBarrier> {
        &self.publishes
    }

    pub(crate) fn take_source(self) -> (PipelineSource, Vec<PipelineTransform>, PipelineSink) {
        (self.source, self.transforms, self.sink)
    }
}

/// Trait operators implement to lower themselves into pipelines.
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

/// Convenience extension: emit a pipeline by providing a typed
/// `SourceNode`. Implemented for everything implementing the
/// dyn-compatible `LoweringCtx`.
pub trait LoweringCtxExt: LoweringCtx {
    fn emit_pipeline<E>(
        &mut self,
        tail: PipelineTail,
        output_domain: Domain,
        output_contract: OutputContract,
        source: E,
    ) -> BuildResult<PipelineId>
    where
        E: SourceNode,
    {
        let dyn_source =
            PipelineSource::new(output_domain.clone(), output_contract.clone(), source);
        self.emit_pipeline_dyn(tail, output_domain, output_contract, dyn_source)
    }
}

impl<T: LoweringCtx + ?Sized> LoweringCtxExt for T {}

/// The fully-lowered plan.
pub struct LoweredPlan {
    pub pipelines: Vec<Pipeline>,
    pub domains: BTreeMap<DomainId, Domain>,
}

/// Concrete `LoweringCtx` that captures emitted pipelines.
pub struct PipelineBuilder {
    pipelines: Vec<Pipeline>,
    domains: BTreeMap<DomainId, Domain>,
    next_barrier: usize,
    next_pipeline: usize,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            pipelines: Vec::new(),
            domains: BTreeMap::new(),
            next_barrier: 0,
            next_pipeline: 0,
        }
    }

    pub fn into_plan(self) -> LoweredPlan {
        LoweredPlan {
            pipelines: self.pipelines,
            domains: self.domains,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LoweringCtx for PipelineBuilder {
    fn register_domain(&mut self, domain: Domain) -> BuildResult<()> {
        let key = domain.id().clone();
        if let Some(existing) = self.domains.get(&key) {
            if existing != &domain {
                return Err(BuildError::message(format!(
                    "domain {:?} registered twice with different definitions",
                    key
                )));
            }
        } else {
            self.domains.insert(key, domain);
        }
        Ok(())
    }

    fn new_pipeline_barrier(&mut self) -> PipelineBarrier {
        let barrier = PipelineBarrier::from_index(self.next_barrier);
        self.next_barrier += 1;
        barrier
    }

    fn emit_pipeline_dyn(
        &mut self,
        tail: PipelineTail,
        output_domain: Domain,
        output_contract: OutputContract,
        source: PipelineSource,
    ) -> BuildResult<PipelineId> {
        // Validate that the source's declared output matches what
        // the tail expects (or what was last set via
        // `prepend_transform`).
        if source.output_domain() != tail.expected_input_domain() {
            return Err(BuildError::OutputContractMismatch {
                expected: format!("{:?}", tail.expected_input_domain().id()),
                actual: format!("{:?}", source.output_domain().id()),
            });
        }
        if source.output_contract() != tail.expected_input_contract() {
            return Err(BuildError::OutputContractMismatch {
                expected: format!("{:?}", tail.expected_input_contract()),
                actual: format!("{:?}", source.output_contract()),
            });
        }
        // The provided output_{domain,contract} are the *pipeline's*
        // overall output (matches the tail's expected input). Sanity
        // check.
        if &output_domain != source.output_domain() {
            return Err(BuildError::OutputContractMismatch {
                expected: format!("{:?}", source.output_domain().id()),
                actual: format!("{:?}", output_domain.id()),
            });
        }
        if &output_contract != source.output_contract() {
            return Err(BuildError::OutputContractMismatch {
                expected: format!("{:?}", source.output_contract()),
                actual: format!("{:?}", output_contract),
            });
        }

        let (_, _, depends_on, publishes, transforms_rev, sink) = tail.into_parts();
        let mut transforms = transforms_rev;
        transforms.reverse();

        let id = PipelineId::from_index(self.next_pipeline);
        self.next_pipeline += 1;

        self.pipelines.push(Pipeline {
            id,
            source,
            transforms,
            sink,
            depends_on,
            publishes,
        });
        Ok(id)
    }
}
