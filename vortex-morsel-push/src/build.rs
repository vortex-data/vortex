// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building an [`ExecPlan`] from a layout tree and a query.
//!
//! The plan is the immutable half of the design's split: one blueprint per scan. Each worker
//! instantiates one thread-local [`Arena`], whose node state survives IO suspension and is recycled
//! across that worker's morsels without crossing a thread boundary.
//!
//! Only the layouts and expression shapes named in the P1 scope are accepted. Anything else is a
//! build error rather than a silent fallback, so an unsupported query can never be timed as if
//! the prototype had executed it.

use std::ops::Range;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
use std::sync::atomic::Ordering;

use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_array::expr::and;
use vortex_array::expr::and_collect;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::chunked::Chunked;
use vortex_layout::layouts::flat::Flat;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::layouts::zoned::LegacyStats;
use vortex_layout::layouts::zoned::Zoned;

use crate::io::IoKey;
use crate::io::ProducerId;
use crate::node::ActivationTarget;
use crate::node::Arena;
use crate::node::DemandTarget;
use crate::node::InputPort;
use crate::node::Node;
use crate::node::NodeId;
use crate::node::Route;
use crate::nodes::ChunkedExec;
use crate::nodes::ConjunctExec;
use crate::nodes::ConjunctMode;
use crate::nodes::ConjunctSlot;
use crate::nodes::FilterExec;
use crate::nodes::FlatExec;
use crate::nodes::PushBatching;
use crate::nodes::StructExec;

/// The immutable blueprint of one node.
enum NodeSpec {
    Flat {
        layout: FlatLayout,
        root_offset: u64,
    },
    Chunked {
        chunk_offsets: Arc<[u64]>,
        children: Arc<[NodeId]>,
        dtype: DType,
    },
    Struct {
        names: FieldNames,
        children: Arc<[NodeId]>,
        push_batching: PushBatching,
        push_passthrough: bool,
    },
    Conjunct {
        slots: Vec<(NodeId, BoundExpression, Option<BoundExpression>)>,
        mode: ConjunctMode,
    },
    Filter {
        predicate: Option<NodeId>,
        projection: NodeId,
        expr: BoundExpression,
        dtype: DType,
        push_batching: PushBatching,
    },
}

/// A shared, immutable execution plan for one scan.
pub struct ExecPlan {
    nodes: Vec<NodeSpec>,
    /// Reverse child-to-parent routes, indexed by child node ID.
    routes: Vec<Option<Route>>,
    /// Forward parent-port-to-child edges, indexed by parent node ID and input port.
    inputs: Vec<Vec<Option<NodeId>>>,
    /// Every push source and the root-coordinate rows it can produce.
    sources: Vec<SourceActivation>,
    source_catalog: SourceCatalog,
    /// Push-only physical pipelines, indexed by their stable pipeline ID.
    topology: Arc<PhysicalTopology>,
    root: NodeId,
    output_dtype: DType,
    row_count: u64,
    /// Root-coordinate boundaries at which every column starts a fresh chunk, used as the
    /// default morsel cut.
    natural_splits: Vec<u64>,
}

#[derive(Debug)]
struct SourceCatalog {
    projection: Option<SourceIntervalGroup>,
    predicates: Vec<Option<SourceIntervalGroup>>,
    #[cfg(test)]
    lookup_probes: AtomicUsize,
}

#[derive(Debug)]
struct SourceIntervalGroup {
    by_start: Arc<[usize]>,
    max_end_tree: Arc<[u64]>,
    leaf_base: usize,
}

struct IntervalQuery<'a> {
    sources: &'a [SourceActivation],
    start: u64,
    upper: usize,
    out: &'a mut Vec<usize>,
}

impl SourceCatalog {
    fn new(sources: &[SourceActivation]) -> Self {
        let predicate_count = sources
            .iter()
            .filter_map(|source| match source.role.activation_target() {
                ActivationTarget::PredicateSlot(slot) => Some(slot + 1),
                ActivationTarget::Projection => None,
            })
            .max()
            .unwrap_or(0);
        let mut projection = Vec::new();
        let mut predicates = (0..predicate_count).map(|_| Vec::new()).collect::<Vec<_>>();
        for (index, source) in sources.iter().enumerate() {
            match source.role.activation_target() {
                ActivationTarget::PredicateSlot(slot) => predicates[slot].push(index),
                ActivationTarget::Projection => projection.push(index),
            }
        }
        Self {
            projection: SourceIntervalGroup::new(sources, projection),
            predicates: predicates
                .into_iter()
                .map(|indices| SourceIntervalGroup::new(sources, indices))
                .collect(),
            #[cfg(test)]
            lookup_probes: AtomicUsize::new(0),
        }
    }

    fn overlaps(
        &self,
        sources: &[SourceActivation],
        target: ActivationTarget,
        range: &Range<u64>,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        #[cfg(test)]
        self.lookup_probes.fetch_add(1, Ordering::Relaxed);
        let group = match target {
            ActivationTarget::PredicateSlot(slot) => {
                self.predicates.get(slot).and_then(Option::as_ref)
            }
            ActivationTarget::Projection => self.projection.as_ref(),
        };
        let Some(group) = group else {
            return;
        };
        let upper = group
            .by_start
            .partition_point(|&index| sources[index].root_range.start < range.end);
        let mut query = IntervalQuery {
            sources,
            start: range.start,
            upper,
            out,
        };
        group.collect_overlaps(1, 0, group.leaf_base, &mut query);
        out.sort_unstable();
    }

    fn overlaps_all(&self, sources: &[SourceActivation], range: &Range<u64>, out: &mut Vec<usize>) {
        out.clear();
        for group in self
            .predicates
            .iter()
            .filter_map(Option::as_ref)
            .chain(self.projection.iter())
        {
            let upper = group
                .by_start
                .partition_point(|&index| sources[index].root_range.start < range.end);
            let mut query = IntervalQuery {
                sources,
                start: range.start,
                upper,
                out,
            };
            group.collect_overlaps(1, 0, group.leaf_base, &mut query);
        }
        out.sort_unstable();
    }

    #[cfg(test)]
    fn reset_lookup_probes(&self) {
        self.lookup_probes.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn lookup_probes(&self) -> usize {
        self.lookup_probes.load(Ordering::Relaxed)
    }
}

impl SourceIntervalGroup {
    fn new(sources: &[SourceActivation], mut by_start: Vec<usize>) -> Option<Self> {
        if by_start.is_empty() {
            return None;
        }
        by_start.sort_unstable_by_key(|&index| (sources[index].root_range.start, index));
        let leaf_base = by_start.len().next_power_of_two();
        let mut max_end_tree = vec![0; leaf_base.saturating_mul(2)];
        for (position, &index) in by_start.iter().enumerate() {
            max_end_tree[leaf_base + position] = sources[index].root_range.end;
        }
        for node in (1..leaf_base).rev() {
            max_end_tree[node] = max_end_tree[node * 2].max(max_end_tree[node * 2 + 1]);
        }
        Some(Self {
            by_start: Arc::from(by_start),
            max_end_tree: Arc::from(max_end_tree),
            leaf_base,
        })
    }

    fn collect_overlaps(
        &self,
        node: usize,
        left: usize,
        right: usize,
        query: &mut IntervalQuery<'_>,
    ) {
        if left >= query.upper || self.max_end_tree[node] <= query.start {
            return;
        }
        if right - left == 1 {
            if let Some(&index) = self.by_start.get(left)
                && query.sources[index].root_range.end > query.start
            {
                query.out.push(index);
            }
            return;
        }
        let middle = left + (right - left) / 2;
        self.collect_overlaps(node * 2, left, middle, query);
        self.collect_overlaps(node * 2 + 1, middle, right, query);
    }
}

type StageLocation = (PipelineId, usize);
type CreditTargetRow = Arc<[Option<StageLocation>]>;
#[cfg(test)]
pub(crate) type TestPipelineDefinition = (Vec<(NodeId, Option<InputPort>)>, Option<Route>);

/// Immutable push topology shared by every worker and morsel runtime.
#[derive(Debug)]
pub(crate) struct PhysicalTopology {
    pipelines: Arc<[PhysicalPipeline]>,
    node_locations: Arc<[StageLocation]>,
    credit_targets: Arc<[CreditTargetRow]>,
    outgoing: Arc<[Option<OutgoingStage>]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OutgoingStage {
    pub(crate) pipeline: PipelineId,
    pub(crate) stage: usize,
    pub(crate) node: NodeId,
    pub(crate) input: InputPort,
    pub(crate) boundary: bool,
}

impl PhysicalTopology {
    pub(crate) fn pipelines(&self) -> &[PhysicalPipeline] {
        &self.pipelines
    }

    pub(crate) fn location(&self, node: NodeId) -> (PipelineId, usize) {
        self.node_locations[node as usize]
    }

    pub(crate) fn credit_target(
        &self,
        parent: NodeId,
        port: InputPort,
    ) -> Option<(PipelineId, usize)> {
        self.credit_targets
            .get(parent as usize)
            .and_then(|targets| targets.get(port.index()))
            .copied()
            .flatten()
    }

    pub(crate) fn outgoing(&self, node: NodeId) -> Option<OutgoingStage> {
        self.outgoing.get(node as usize).copied().flatten()
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        definitions: Vec<TestPipelineDefinition>,
        credit_targets: Vec<Vec<Option<(PipelineId, usize)>>>,
        node_count: usize,
    ) -> Arc<Self> {
        let pipelines = definitions
            .into_iter()
            .enumerate()
            .map(|(id, (stages, sink))| PhysicalPipeline {
                id: PipelineId::try_from(id).unwrap_or(PipelineId::MAX),
                stages: Arc::from(
                    stages
                        .into_iter()
                        .map(|(node, input)| PipelineStage { node, input })
                        .collect::<Vec<_>>(),
                ),
                sink,
            })
            .collect::<Vec<_>>();
        let mut locations = vec![(PipelineId::MAX, usize::MAX); node_count];
        for pipeline in &pipelines {
            for (stage, entry) in pipeline.stages.iter().enumerate() {
                locations[entry.node as usize] = (pipeline.id, stage);
            }
        }
        Arc::new(Self {
            outgoing: Arc::from(
                compiled_outgoing(&pipelines, &locations, node_count)
                    .vortex_expect("test physical topology is valid"),
            ),
            pipelines: Arc::from(pipelines),
            node_locations: Arc::from(locations),
            credit_targets: Arc::from(
                credit_targets
                    .into_iter()
                    .map(Arc::from)
                    .collect::<Vec<_>>(),
            ),
        })
    }
}

fn compiled_outgoing(
    pipelines: &[PhysicalPipeline],
    node_locations: &[StageLocation],
    node_count: usize,
) -> VortexResult<Vec<Option<OutgoingStage>>> {
    let mut outgoing = vec![None; node_count];
    for pipeline in pipelines {
        for (stage, pair) in pipeline.stages.windows(2).enumerate() {
            let next = pair[1];
            outgoing[pair[0].node as usize] = Some(OutgoingStage {
                pipeline: pipeline.id,
                stage: stage + 1,
                node: next.node,
                input: next
                    .input
                    .ok_or_else(|| vortex_err!("fused stage has no input"))?,
                boundary: false,
            });
        }
        if let Some((last, sink)) = pipeline.stages.last().zip(pipeline.sink) {
            let (target_pipeline, target_stage) = node_locations[sink.parent as usize];
            outgoing[last.node as usize] = Some(OutgoingStage {
                pipeline: target_pipeline,
                stage: target_stage,
                node: sink.parent,
                input: sink.port,
                boundary: true,
            });
        }
    }
    Ok(outgoing)
}

/// Stable index of a physical push pipeline in an [`ExecPlan`].
pub type PipelineId = u32;

/// A maximal push chain ending at a typed stateful boundary.
///
/// Single-input Chunked/Struct/conjunct operators and predicate-free filters remain fused in the
/// chain. True fan-in, ordering, and filter gates are statically bound sinks with their own drain
/// pipeline, so a worker never discovers the next operator with a route lookup.
#[derive(Clone, Debug)]
pub struct PhysicalPipeline {
    id: PipelineId,
    stages: Arc<[PipelineStage]>,
    sink: Option<Route>,
}

/// One operator invocation in a physical pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineStage {
    node: NodeId,
    input: Option<InputPort>,
}

impl PipelineStage {
    /// Logical node owned by this physical stage.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Typed input from the preceding stage; absent for the pipeline head.
    pub fn input(&self) -> Option<InputPort> {
        self.input
    }
}

impl PhysicalPipeline {
    /// Stable pipeline identity.
    pub fn id(&self) -> PipelineId {
        self.id
    }

    /// Operators invoked directly in stage order.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Statically bound downstream boundary input, or `None` for the root output pipeline.
    pub fn sink(&self) -> Option<Route> {
        self.sink
    }
}

/// A push source and the root-coordinate range in which it can be active.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceActivation {
    /// The source node to activate.
    pub node: NodeId,
    /// Root-coordinate rows the source can produce.
    pub root_range: Range<u64>,
    /// The logical demand stream that controls this source.
    pub role: SourceRole,
}

/// The logical demand stream controlling a push source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRole {
    /// A source in the projection subtree.
    Projection,
    /// A source in one predicate conjunct.
    Predicate {
        /// The conjunct's zero-based expression slot.
        slot: usize,
        /// Whether conjunct evaluation is cascaded or parallel.
        mode: ConjunctMode,
    },
}

impl SourceRole {
    /// The correctness-bearing activation target for sources with this role.
    pub fn activation_target(self) -> ActivationTarget {
        match self {
            Self::Projection => ActivationTarget::Projection,
            Self::Predicate { slot, .. } => ActivationTarget::PredicateSlot(slot),
        }
    }

    /// The optional demand-hint target for sources with this role.
    pub fn demand_target(self) -> DemandTarget {
        match self {
            Self::Projection => DemandTarget::Projection,
            Self::Predicate { slot, .. } => DemandTarget::PredicateSlot(slot),
        }
    }
}

impl ExecPlan {
    /// The root node of the plan.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The static route from `node` to its parent, or `None` for the plan root.
    pub fn route(&self, node: NodeId) -> Option<Route> {
        self.routes.get(node as usize).copied().flatten()
    }

    /// Every reverse child-to-parent route, indexed by child node ID.
    pub fn routes(&self) -> &[Option<Route>] {
        &self.routes
    }

    /// The child connected to one parent input port.
    ///
    /// Push credit acknowledgements use this inverse of [`ExecPlan::route`] to resume the child
    /// after its parent fully consumes one retained input head.
    pub fn input(&self, parent: NodeId, port: InputPort) -> Option<NodeId> {
        self.inputs
            .get(parent as usize)
            .and_then(|inputs| inputs.get(port.index()))
            .copied()
            .flatten()
    }

    /// The catalog of source nodes that may activate for a morsel.
    pub fn sources(&self) -> &[SourceActivation] {
        &self.sources
    }

    pub(crate) fn overlapping_source_indices(
        &self,
        target: ActivationTarget,
        range: &Range<u64>,
        out: &mut Vec<usize>,
    ) {
        self.source_catalog
            .overlaps(&self.sources, target, range, out);
    }

    pub(crate) fn overlapping_demand_source_indices(
        &self,
        target: DemandTarget,
        range: &Range<u64>,
        out: &mut Vec<usize>,
    ) {
        let target = match target {
            DemandTarget::PredicateSlot(slot) => ActivationTarget::PredicateSlot(slot),
            DemandTarget::Projection => ActivationTarget::Projection,
        };
        self.overlapping_source_indices(target, range, out);
    }

    pub(crate) fn source_io_use_at(
        &self,
        index: usize,
    ) -> Option<(NodeId, IoKey, Range<u64>, SourceRole)> {
        let source = self.sources.get(index)?;
        let NodeSpec::Flat { layout, .. } = &self.nodes[source.node as usize] else {
            return None;
        };
        Some((
            source.node,
            IoKey::Segment(layout.segment_id()),
            source.root_range.clone(),
            source.role,
        ))
    }

    pub(crate) fn overlapping_all_source_indices(&self, range: &Range<u64>, out: &mut Vec<usize>) {
        self.source_catalog.overlaps_all(&self.sources, range, out);
    }

    /// Push-only physical pipelines built with the logical plan.
    pub fn pipelines(&self) -> &[PhysicalPipeline] {
        self.topology.pipelines()
    }

    pub(crate) fn topology(&self) -> &Arc<PhysicalTopology> {
        &self.topology
    }

    /// Pipeline that drains values produced by `node`.
    pub fn pipeline_for_node(&self, node: NodeId) -> Option<(&PhysicalPipeline, usize)> {
        self.topology.pipelines.iter().find_map(|pipeline| {
            pipeline
                .stages
                .iter()
                .position(|stage| stage.node == node)
                .map(|stage| (pipeline, stage))
        })
    }

    /// Every I/O-backed push source with its exact key, root-coordinate extent, and role.
    ///
    /// Zero-field struct sources do not name I/O and are omitted. Keeping the node and role next
    /// to the key lets the scheduler apply optional demand hints without inferring ownership from
    /// ranges or from keys that may be shared by predicate and projection subtrees.
    pub fn source_io_uses(
        &self,
    ) -> impl Iterator<Item = (NodeId, IoKey, Range<u64>, SourceRole)> + '_ {
        self.sources.iter().filter_map(|source| {
            let NodeSpec::Flat {
                layout,
                root_offset,
            } = &self.nodes[source.node as usize]
            else {
                return None;
            };
            Some((
                source.node,
                IoKey::Segment(layout.segment_id()),
                *root_offset..*root_offset + layout.row_count(),
                source.role,
            ))
        })
    }

    /// The dtype the scan emits.
    pub fn output_dtype(&self) -> &DType {
        &self.output_dtype
    }

    /// The number of rows in the scanned layout.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub(crate) fn has_filter(&self) -> bool {
        matches!(
            &self.nodes[self.root as usize],
            NodeSpec::Filter {
                predicate: Some(_),
                ..
            }
        )
    }

    /// The union of every column's chunk boundaries, in root coordinates.
    pub fn natural_splits(&self) -> &[u64] {
        &self.natural_splits
    }

    /// Every flat node's stored unit and its root-coordinate row range, one entry per node.
    ///
    /// A segment referenced from two subtrees (a column in both filter and projection) appears
    /// once per referencing node, because each node registers its own use per morsel. This is
    /// the input to the shared-cell lease counts: the count for a unit is the number of
    /// (node, morsel) pairs whose ranges overlap.
    pub fn flat_uses(&self) -> impl Iterator<Item = (IoKey, Range<u64>)> + '_ {
        self.nodes.iter().filter_map(|spec| match spec {
            NodeSpec::Flat {
                layout,
                root_offset,
            } => Some((
                IoKey::Segment(layout.segment_id()),
                *root_offset..*root_offset + layout.row_count(),
            )),
            _ => None,
        })
    }

    /// The number of nodes in the plan.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Instantiate one worker's mutable arena from this blueprint.
    pub fn instantiate(&self) -> Arena {
        let nodes: Vec<Node> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, spec)| match spec {
                NodeSpec::Flat {
                    layout,
                    root_offset,
                } => Node::Flat(Box::new(FlatExec::new(
                    layout,
                    *root_offset,
                    ProducerId(u32::try_from(idx).unwrap_or(u32::MAX)),
                ))),
                NodeSpec::Chunked {
                    chunk_offsets,
                    children,
                    dtype,
                } => Node::Chunked(Box::new(ChunkedExec::new(
                    Arc::clone(chunk_offsets),
                    Arc::clone(children),
                    dtype.clone(),
                ))),
                NodeSpec::Struct {
                    names,
                    children,
                    push_batching,
                    push_passthrough,
                } => Node::Struct(Box::new(StructExec::new_with_push_batching(
                    names.clone(),
                    Arc::clone(children),
                    *push_batching,
                    *push_passthrough,
                ))),
                NodeSpec::Conjunct { slots, mode } => {
                    Node::Conjunct(Box::new(ConjunctExec::new_with_push_predicates(
                        slots
                            .iter()
                            .map(|(input, predicate, _)| ConjunctSlot {
                                input: *input,
                                predicate: predicate.clone(),
                            })
                            .collect(),
                        slots
                            .iter()
                            .map(|(_, _, push_predicate)| push_predicate.clone())
                            .collect(),
                        *mode,
                    )))
                }
                NodeSpec::Filter {
                    predicate,
                    projection,
                    expr,
                    dtype,
                    push_batching,
                } => Node::Filter(Box::new(FilterExec::new_with_push_batching(
                    *predicate,
                    *projection,
                    expr.clone(),
                    dtype.clone(),
                    *push_batching,
                ))),
            })
            .collect();
        let mut arena = Arena::new_compiled(nodes);
        arena.prepare_push_sidebands(self.inputs.iter().map(|inputs| inputs.len()));
        arena
    }
}

/// Build an execution plan for `layout` under `projection` and `filter`.
///
/// The expressions are *unbound*: each conjunct and the projection are re-bound against the
/// narrowed struct dtype of just the fields they reference, which is what lets a subtree read
/// only its own columns without any expression rewriting.
pub fn build_plan(
    layout: &LayoutRef,
    projection: &Expression,
    filter: Option<&Expression>,
    mode: ConjunctMode,
) -> VortexResult<ExecPlan> {
    let root_dtype = layout.dtype().clone();
    let root_fields = root_dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("the morsel executor requires a struct-rooted layout"))?
        .clone();
    if root_dtype.is_nullable() {
        vortex_bail!("the morsel executor does not support a nullable root struct");
    }
    if !layout.is::<Struct>() {
        vortex_bail!(
            "the morsel executor requires a struct root layout, got {}",
            layout.encoding_id()
        );
    }

    let mut builder = Builder {
        nodes: Vec::new(),
        layout: LayoutRef::clone(layout),
        root_fields,
        splits: Vec::new(),
    };

    // The filter: one subtree per conjunct, each over just that conjunct's fields.
    let predicate = match filter {
        None => None,
        Some(filter) => {
            let filter = optimize_ordered_filter(filter, &root_dtype)?;
            let conjuncts = split_conjuncts(&filter);
            let mut slots = Vec::with_capacity(conjuncts.len());
            for conjunct in conjuncts {
                let (input, bound, push_predicate) =
                    builder.build_scoped(&conjunct, PushBatching::Streaming, true)?;
                slots.push((input, bound, push_predicate));
            }
            Some(builder.push(NodeSpec::Conjunct { slots, mode }))
        }
    };

    // The projection.
    let push_batching = projection_push_batching(predicate.is_some());
    let (projection_input, projection_bound, _) =
        builder.build_scoped(projection, push_batching, false)?;
    let output_dtype = projection_bound.dtype().clone();
    let root = builder.push(NodeSpec::Filter {
        predicate,
        projection: projection_input,
        expr: projection_bound,
        dtype: output_dtype.clone(),
        push_batching,
    });

    let row_count = layout.row_count();
    let mut natural_splits = builder.splits;
    natural_splits.push(row_count);
    natural_splits.sort_unstable();
    natural_splits.dedup();
    natural_splits.retain(|&split| split > 0 && split <= row_count);

    let nodes = builder.nodes;
    let routes = reverse_routes(&nodes, root)?;
    let inputs = forward_inputs(nodes.len(), &routes)?;
    let sources = source_activations(&nodes, &routes, root, row_count)?;
    let source_catalog = SourceCatalog::new(&sources);
    let pipelines = physical_pipelines(&nodes, &routes, &sources)?;
    let topology = physical_topology(pipelines, &inputs, nodes.len())?;

    Ok(ExecPlan {
        nodes,
        routes,
        inputs,
        sources,
        source_catalog,
        topology,
        root,
        output_dtype,
        row_count,
        natural_splits,
    })
}

// A projection Struct is a morsel boundary: its field pipelines still run independently and
// directly, while the fan-in retains each field's ordered fragments without cross-field slicing.
fn projection_push_batching(_filtered: bool) -> PushBatching {
    PushBatching::Morsel
}

/// Optimize adjacent conjuncts without changing the query's declared cascade order.
///
/// The general between pass searches all conjunct pairs and can move a later bound ahead of an
/// intervening predicate. Applying it to adjacent pairs retains the useful range fusion while
/// keeping the execution policy explicit and stable.
fn optimize_ordered_filter(expr: &Expression, scope: &DType) -> VortexResult<Expression> {
    let declared = split_conjuncts(expr);
    let mut optimized = Vec::with_capacity(declared.len());
    let mut index = 0;
    while index < declared.len() {
        if let Some(next) = declared.get(index + 1) {
            let pair = and(declared[index].clone(), next.clone()).optimize_recursive(scope)?;
            let fused = pair
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<Between>())
                .is_some();
            if fused {
                optimized.push(pair);
                index += 2;
                continue;
            }
        }
        optimized.push(declared[index].optimize_recursive(scope)?);
        index += 1;
    }
    and_collect(optimized).ok_or_else(|| vortex_err!("filter optimization produced no predicate"))
}

fn physical_topology(
    pipelines: Vec<PhysicalPipeline>,
    inputs: &[Vec<Option<NodeId>>],
    node_count: usize,
) -> VortexResult<Arc<PhysicalTopology>> {
    let mut node_locations = vec![(PipelineId::MAX, usize::MAX); node_count];
    for pipeline in &pipelines {
        for (stage, physical_stage) in pipeline.stages.iter().enumerate() {
            let location = node_locations
                .get_mut(physical_stage.node as usize)
                .ok_or_else(|| vortex_err!("physical pipeline references unknown node"))?;
            if location.0 != PipelineId::MAX {
                vortex_bail!(
                    "node {} appears in more than one physical pipeline",
                    physical_stage.node
                );
            }
            *location = (pipeline.id, stage);
        }
    }
    if let Some((node, _)) = node_locations
        .iter()
        .enumerate()
        .find(|(_, location)| location.0 == PipelineId::MAX)
    {
        vortex_bail!("node {node} is absent from the physical topology");
    }
    let credit_targets = inputs
        .iter()
        .map(|ports| {
            Arc::from(
                ports
                    .iter()
                    .map(|child| child.map(|child| node_locations[child as usize]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let outgoing = compiled_outgoing(&pipelines, &node_locations, node_count)?;
    Ok(Arc::new(PhysicalTopology {
        pipelines: Arc::from(pipelines),
        node_locations: Arc::from(node_locations),
        credit_targets: Arc::from(credit_targets),
        outgoing: Arc::from(outgoing),
    }))
}

fn physical_pipelines(
    nodes: &[NodeSpec],
    routes: &[Option<Route>],
    sources: &[SourceActivation],
) -> VortexResult<Vec<PhysicalPipeline>> {
    let is_breaker = |node: NodeId| match &nodes[node as usize] {
        NodeSpec::Flat { .. } => false,
        NodeSpec::Chunked { children, .. } | NodeSpec::Struct { children, .. } => {
            children.len() != 1
        }
        NodeSpec::Conjunct { slots, .. } => slots.len() != 1,
        NodeSpec::Filter { predicate, .. } => predicate.is_some(),
    };
    let mut heads = sources.iter().map(|source| source.node).collect::<Vec<_>>();
    heads.extend(
        nodes
            .iter()
            .enumerate()
            .map(|(node, _)| NodeId::try_from(node).vortex_expect("exec plan exceeds u32 nodes"))
            .filter(|&node| is_breaker(node)),
    );
    heads.sort_unstable();
    heads.dedup();

    heads
        .into_iter()
        .enumerate()
        .map(|(id, head)| {
            let id = PipelineId::try_from(id)
                .map_err(|_| vortex_err!("exec plan exceeds u32 pipelines"))?;
            let mut stages = vec![PipelineStage {
                node: head,
                input: None,
            }];
            let mut node = head;
            let sink = loop {
                let Some(route) = routes[node as usize] else {
                    break None;
                };
                if is_breaker(route.parent) {
                    break Some(route);
                }
                stages.push(PipelineStage {
                    node: route.parent,
                    input: Some(route.port),
                });
                node = route.parent;
            };
            Ok(PhysicalPipeline {
                id,
                stages: Arc::from(stages),
                sink,
            })
        })
        .collect()
}

fn forward_inputs(
    node_count: usize,
    routes: &[Option<Route>],
) -> VortexResult<Vec<Vec<Option<NodeId>>>> {
    let mut inputs = vec![Vec::new(); node_count];
    for (child, route) in routes.iter().enumerate() {
        let Some(route) = route else {
            continue;
        };
        let parent_inputs = inputs
            .get_mut(route.parent as usize)
            .ok_or_else(|| vortex_err!("route references unknown parent {}", route.parent))?;
        if parent_inputs.len() <= route.port.index() {
            parent_inputs.resize(route.port.index() + 1, None);
        }
        let child = NodeId::try_from(child).vortex_expect("exec plan exceeds u32 nodes");
        if let Some(previous) = parent_inputs[route.port.index()].replace(child) {
            vortex_bail!(
                "push parent {} port {} connects both child {previous} and child {child}",
                route.parent,
                route.port.index()
            );
        }
    }
    Ok(inputs)
}

fn reverse_routes(nodes: &[NodeSpec], root: NodeId) -> VortexResult<Vec<Option<Route>>> {
    let mut routes = vec![None; nodes.len()];
    for (parent, spec) in nodes.iter().enumerate() {
        let parent = NodeId::try_from(parent).vortex_expect("exec plan exceeds u32 nodes");
        let mut register = |child: NodeId, port: usize| -> VortexResult<()> {
            let slot = routes
                .get_mut(child as usize)
                .ok_or_else(|| vortex_err!("node {parent} references unknown child {child}"))?;
            let route = Route {
                parent,
                port: InputPort::new(port)?,
            };
            if let Some(previous) = slot.replace(route) {
                vortex_bail!(
                    "push plan is not a tree: child {child} routes to both node {} and node {parent}",
                    previous.parent
                );
            }
            Ok(())
        };

        match spec {
            NodeSpec::Flat { .. } => {}
            NodeSpec::Chunked { children, .. } | NodeSpec::Struct { children, .. } => {
                for (port, &child) in children.iter().enumerate() {
                    register(child, port)?;
                }
            }
            NodeSpec::Conjunct { slots, .. } => {
                for (port, (child, ..)) in slots.iter().enumerate() {
                    register(*child, port)?;
                }
            }
            NodeSpec::Filter {
                predicate,
                projection,
                ..
            } => {
                if let Some(predicate) = predicate {
                    register(*predicate, 0)?;
                }
                // Filter ports stay stable whether or not the optional predicate exists.
                register(*projection, 1)?;
            }
        }
    }

    for (node, route) in routes.iter().enumerate() {
        let node = NodeId::try_from(node).vortex_expect("exec plan exceeds u32 nodes");
        if node == root {
            if route.is_some() {
                vortex_bail!("push plan root {root} unexpectedly has a parent route");
            }
        } else if route.is_none() {
            vortex_bail!("push plan node {node} is disconnected from root {root}");
        }
    }
    Ok(routes)
}

fn source_activations(
    nodes: &[NodeSpec],
    routes: &[Option<Route>],
    root: NodeId,
    row_count: u64,
) -> VortexResult<Vec<SourceActivation>> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(node, spec)| match spec {
            NodeSpec::Flat {
                layout,
                root_offset,
            } => Some((node, *root_offset..*root_offset + layout.row_count())),
            NodeSpec::Struct { children, .. } if children.is_empty() => Some((node, 0..row_count)),
            _ => None,
        })
        .map(|(node, root_range)| {
            let node = NodeId::try_from(node).vortex_expect("exec plan exceeds u32 nodes");
            Ok(SourceActivation {
                node,
                root_range,
                role: source_role(nodes, routes, node, root)?,
            })
        })
        .collect()
}

fn source_role(
    nodes: &[NodeSpec],
    routes: &[Option<Route>],
    source: NodeId,
    root: NodeId,
) -> VortexResult<SourceRole> {
    let mut node = source;
    let mut predicate = None;
    while node != root {
        let route = routes
            .get(node as usize)
            .copied()
            .flatten()
            .ok_or_else(|| vortex_err!("source {source} has no route to root {root}"))?;
        match nodes.get(route.parent as usize) {
            Some(NodeSpec::Conjunct { mode, .. }) => {
                let slot = route.port.index();
                if predicate.replace((slot, *mode)).is_some() {
                    vortex_bail!("source {source} crosses more than one conjunct node");
                }
            }
            Some(NodeSpec::Filter {
                predicate: root_predicate,
                ..
            }) => {
                return match route.port.index() {
                    0 if root_predicate.is_some() => {
                        let (slot, mode) = predicate.ok_or_else(|| {
                            vortex_err!(
                                "source {source} reaches the filter predicate without a conjunct slot"
                            )
                        })?;
                        Ok(SourceRole::Predicate { slot, mode })
                    }
                    1 => {
                        if predicate.is_some() {
                            vortex_bail!(
                                "source {source} crosses a conjunct before reaching projection"
                            );
                        }
                        Ok(SourceRole::Projection)
                    }
                    port => Err(vortex_err!(
                        "source {source} reaches filter root through invalid port {port}"
                    )),
                };
            }
            Some(_) => {}
            None => vortex_bail!(
                "source {source} route references unknown parent {}",
                route.parent
            ),
        }
        node = route.parent;
    }
    Err(vortex_err!(
        "source {source} reached root {root} without a filter demand role"
    ))
}

struct Builder {
    nodes: Vec<NodeSpec>,
    layout: LayoutRef,
    root_fields: StructFields,
    splits: Vec<u64>,
}

impl Builder {
    fn push(&mut self, spec: NodeSpec) -> NodeId {
        self.nodes.push(spec);
        NodeId::try_from(self.nodes.len() - 1).vortex_expect("exec plan exceeds u32 nodes")
    }

    /// Build the subtree for one expression: a struct over exactly the top-level fields the
    /// expression reads, plus that expression re-bound against the narrowed struct dtype.
    fn build_scoped(
        &mut self,
        expr: &Expression,
        push_batching: PushBatching,
        allow_predicate_passthrough: bool,
    ) -> VortexResult<(NodeId, BoundExpression, Option<BoundExpression>)> {
        let full = expr.bind(self.layout.dtype())?;
        let names = self.referenced_top_level_fields(&full)?;

        let dtypes = names
            .iter()
            .map(|name| {
                self.root_fields
                    .field(name)
                    .ok_or_else(|| vortex_err!("field {name} not found in the scan dtype"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let narrowed = DType::Struct(
            StructFields::new(FieldNames::from(names.clone()), dtypes.clone()),
            Nullability::NonNullable,
        );
        let bound = expr.bind(&narrowed)?;
        let push_predicate = allow_predicate_passthrough
            .then(|| direct_single_field_predicate(expr, &full, &names, &dtypes))
            .flatten();

        let mut children = Vec::with_capacity(names.len());
        for name in &names {
            let idx = self
                .root_fields
                .find(name)
                .ok_or_else(|| vortex_err!("field {name} not found in the scan dtype"))?;
            let field_layout = self.field_layout(idx)?;
            children.push(self.build_layout(&field_layout, 0)?);
        }

        let node = self.push(NodeSpec::Struct {
            names: FieldNames::from(names),
            children: Arc::from(children),
            push_batching,
            push_passthrough: push_predicate.is_some(),
        });
        Ok((node, bound, push_predicate))
    }

    /// The struct layout's child for field `idx`, accounting for the validity slot.
    fn field_layout(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.layout
            .slot(idx + 1)?
            .ok_or_else(|| vortex_err!("struct layout has no child for field {idx}"))
    }

    fn referenced_top_level_fields(&self, expr: &BoundExpression) -> VortexResult<Vec<FieldName>> {
        let paths = referenced_field_paths(expr)?;
        let mut names: Vec<FieldName> = Vec::new();
        let mut all = false;
        for path in paths.iter() {
            if path.is_root() {
                all = true;
                break;
            }
            match &path.parts()[0] {
                Field::Name(name) => {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
                other => vortex_bail!("unsupported field reference {other:?}"),
            }
        }
        if all {
            names = self.root_fields.names().iter().cloned().collect();
        }
        // Keep the scan dtype's field order so `select` and `pack` see the fields they expect.
        names.sort_by_key(|name| self.root_fields.find(name).unwrap_or(usize::MAX));
        Ok(names)
    }

    /// Build the subtree for one column, recording its chunk boundaries as natural splits.
    fn build_layout(&mut self, layout: &LayoutRef, root_offset: u64) -> VortexResult<NodeId> {
        if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
            let data = layout
                .slot(0)?
                .ok_or_else(|| vortex_err!("zoned layout has no data child"))?;
            return self.build_layout(&data, root_offset);
        }

        if layout.is::<Flat>() {
            self.splits.push(root_offset + layout.row_count());
            let flat = layout.as_::<Flat>().clone();
            return Ok(self.push(NodeSpec::Flat {
                layout: flat,
                root_offset,
            }));
        }

        if layout.is::<Chunked>() {
            let nchunks = layout.nchildren();
            let mut offsets = Vec::with_capacity(nchunks + 1);
            offsets.push(0u64);
            let mut children = Vec::with_capacity(nchunks);
            for idx in 0..nchunks {
                let child = layout
                    .slot(idx)?
                    .ok_or_else(|| vortex_err!("chunked layout has no child {idx}"))?;
                let offset = offsets[idx];
                children.push(self.build_layout(&child, root_offset + offset)?);
                offsets.push(offset + child.row_count());
            }
            return Ok(self.push(NodeSpec::Chunked {
                chunk_offsets: Arc::from(offsets),
                children: Arc::from(children),
                dtype: layout.dtype().clone(),
            }));
        }

        vortex_bail!(
            "the morsel executor supports flat and chunked columns only, got {} at row offset {}",
            layout.encoding_id(),
            root_offset
        )
    }
}

fn direct_single_field_predicate(
    expr: &Expression,
    full: &BoundExpression,
    names: &[FieldName],
    dtypes: &[DType],
) -> Option<BoundExpression> {
    let [name] = names else {
        return None;
    };
    let [dtype] = dtypes else {
        return None;
    };
    let paths = referenced_field_paths(full).ok()?;
    let expected = FieldPath::from_name(name.clone());
    if paths.iter().any(|path| path != &expected) || !paths.contains(&expected) {
        return None;
    }

    let needle = get_item(name.clone(), root());
    let rewritten = replace(expr.clone(), &needle, root());
    if rewritten == *expr {
        return None;
    }
    let bound = rewritten.bind(dtype).ok()?;
    matches!(bound.dtype(), DType::Bool(_)).then_some(bound)
}

/// Split a conjunction into its conjuncts, mirroring the V1 `FilterExpr` split.
fn split_conjuncts(expr: &Expression) -> Vec<Expression> {
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;

    let mut conjuncts = Vec::new();
    let mut pending = vec![expr.clone()];
    while let Some(expr) = pending.pop() {
        let is_and = expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And);
        if is_and {
            pending.extend(expr.children().iter().rev().cloned());
        } else {
            conjuncts.push(expr);
        }
    }
    conjuncts
}

/// The morsel row ranges for a plan, from its natural splits, coalesced to `target_rows`.
pub(crate) fn cut_morsels(splits: &[u64], target_rows: u64) -> Vec<Range<u64>> {
    let mut morsels = Vec::new();
    let mut start = 0u64;
    for &split in splits {
        if split <= start {
            continue;
        }
        if split - start >= target_rows {
            morsels.push(start..split);
            start = split;
        }
    }
    if let Some(&last) = splits.last()
        && last > start
    {
        morsels.push(start..last);
    }
    morsels
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::and;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::Between;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::NodeSpec;
    use super::PhysicalPipeline;
    use super::PushBatching;
    use super::SourceActivation;
    use super::SourceCatalog;
    use super::SourceRole;
    use super::direct_single_field_predicate;
    use super::forward_inputs;
    use super::optimize_ordered_filter;
    use super::physical_pipelines;
    use super::physical_topology;
    use super::projection_push_batching;
    use super::reverse_routes;
    use crate::node::ActivationTarget;

    fn range_scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["shipdate", "discount", "quantity"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn every_projection_uses_field_independent_morsel_batching() {
        assert_eq!(projection_push_batching(true), PushBatching::Morsel);
        assert_eq!(projection_push_batching(false), PushBatching::Morsel);
    }

    #[test]
    fn direct_predicate_accepts_repeated_nullable_top_level_field() -> VortexResult<()> {
        let field_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let scope = DType::Struct(
            StructFields::new(["x"].into(), vec![field_dtype.clone()]),
            Nullability::NonNullable,
        );
        let expr = and(
            gt_eq(get_item("x", root()), lit(1i32)),
            lt_eq(get_item("x", root()), lit(9i32)),
        );
        let full = expr.bind(&scope)?;
        let direct = direct_single_field_predicate(
            &expr,
            &full,
            &["x".into()],
            std::slice::from_ref(&field_dtype),
        )
        .ok_or_else(|| vortex_error::vortex_err!("eligible predicate was not fused"))?;

        assert!(matches!(direct.dtype(), DType::Bool(Nullability::Nullable)));
        assert!(direct.children().iter().any(|child| {
            child.is_root()
                || child
                    .children()
                    .iter()
                    .any(vortex_array::expr::BoundExpression::is_root)
        }));
        Ok(())
    }

    #[test]
    fn direct_predicate_rejects_nested_root_and_multiple_fields() -> VortexResult<()> {
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let nested_dtype = DType::Struct(
            StructFields::new(["y"].into(), vec![i32_dtype.clone()]),
            Nullability::NonNullable,
        );
        let nested_scope = DType::Struct(
            StructFields::new(["x"].into(), vec![nested_dtype.clone()]),
            Nullability::NonNullable,
        );
        let nested = gt(get_item("y", get_item("x", root())), lit(0i32));
        let nested_full = nested.bind(&nested_scope)?;
        assert!(
            direct_single_field_predicate(&nested, &nested_full, &["x".into()], &[nested_dtype],)
                .is_none()
        );

        let root_predicate = root();
        let root_full = root_predicate.bind(&nested_scope)?;
        assert!(
            direct_single_field_predicate(
                &root_predicate,
                &root_full,
                &["x".into()],
                std::slice::from_ref(&i32_dtype),
            )
            .is_none()
        );

        let multi_scope = range_scope();
        let multi = and(
            gt(get_item("discount", root()), lit(0i32)),
            gt(get_item("quantity", root()), lit(0i32)),
        );
        let multi_full = multi.bind(&multi_scope)?;
        assert!(
            direct_single_field_predicate(
                &multi,
                &multi_full,
                &["discount".into(), "quantity".into()],
                &[i32_dtype.clone(), i32_dtype],
            )
            .is_none()
        );
        Ok(())
    }

    fn is_between(expr: &vortex_array::expr::Expression) -> bool {
        expr.as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Between>())
            .is_some()
    }

    #[test]
    fn adjacent_q6_ranges_fuse_without_reordering() -> VortexResult<()> {
        let shipdate = || get_item("shipdate", root());
        let discount = || get_item("discount", root());
        let quantity = || get_item("quantity", root());
        let filter = and(
            and(
                gt_eq(shipdate(), lit(1994i32)),
                lt(shipdate(), lit(1995i32)),
            ),
            and(
                and(gt_eq(discount(), lit(5i32)), lt_eq(discount(), lit(7i32))),
                lt(quantity(), lit(24i32)),
            ),
        );

        let filter = optimize_ordered_filter(&filter, &range_scope())?;
        let conjuncts = super::split_conjuncts(&filter);

        assert_eq!(conjuncts.len(), 3);
        assert!(is_between(&conjuncts[0]));
        assert!(is_between(&conjuncts[1]));
        assert!(!is_between(&conjuncts[2]));
        assert!(conjuncts[0].to_string().contains("shipdate"));
        assert!(conjuncts[1].to_string().contains("discount"));
        assert!(conjuncts[2].to_string().contains("quantity"));
        Ok(())
    }

    #[test]
    fn nonadjacent_ranges_do_not_cross_an_intervening_conjunct() -> VortexResult<()> {
        let shipdate = || get_item("shipdate", root());
        let filter = and(
            and(
                gt_eq(shipdate(), lit(1994i32)),
                gt(get_item("quantity", root()), lit(0i32)),
            ),
            lt(shipdate(), lit(1995i32)),
        );

        let filter = optimize_ordered_filter(&filter, &range_scope())?;
        let conjuncts = super::split_conjuncts(&filter);

        assert_eq!(conjuncts.len(), 3);
        assert!(conjuncts.iter().all(|expr| !is_between(expr)));
        assert!(conjuncts[0].to_string().contains("shipdate"));
        assert!(conjuncts[1].to_string().contains("quantity"));
        assert!(conjuncts[2].to_string().contains("shipdate"));
        Ok(())
    }

    #[test]
    fn unsupported_pairs_and_null_simplifications_stay_separate() -> VortexResult<()> {
        let unsupported = and(
            gt_eq(get_item("shipdate", root()), lit(1994i32)),
            lt(get_item("discount", root()), lit(7i32)),
        );
        let unsupported = optimize_ordered_filter(&unsupported, &range_scope())?;
        let conjuncts = super::split_conjuncts(&unsupported);
        assert_eq!(conjuncts.len(), 2);
        assert!(conjuncts.iter().all(|expr| !is_between(expr)));

        let null = lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        )));
        let nullable_range = and(
            gt_eq(get_item("shipdate", root()), null),
            lt_eq(get_item("shipdate", root()), lit(1995i32)),
        );
        let nullable_range = optimize_ordered_filter(&nullable_range, &range_scope())?;
        let conjuncts = super::split_conjuncts(&nullable_range);
        assert_eq!(conjuncts.len(), 2);
        assert!(conjuncts.iter().all(|expr| !is_between(expr)));
        Ok(())
    }

    #[test]
    fn source_catalog_queries_overlaps_in_plan_order() {
        let sources = vec![
            SourceActivation {
                node: 0,
                root_range: 0..100,
                role: SourceRole::Projection,
            },
            SourceActivation {
                node: 1,
                root_range: 10..20,
                role: SourceRole::Projection,
            },
            SourceActivation {
                node: 2,
                root_range: 40..50,
                role: SourceRole::Predicate {
                    slot: 0,
                    mode: crate::nodes::ConjunctMode::Cascade,
                },
            },
            SourceActivation {
                node: 3,
                root_range: 25..35,
                role: SourceRole::Projection,
            },
        ];
        let catalog = SourceCatalog::new(&sources);
        let mut matches = Vec::new();

        catalog.overlaps(
            &sources,
            ActivationTarget::Projection,
            &(15..30),
            &mut matches,
        );
        assert_eq!(matches, [0, 1, 3]);

        catalog.overlaps(
            &sources,
            ActivationTarget::PredicateSlot(0),
            &(0..40),
            &mut matches,
        );
        assert!(matches.is_empty());
    }

    #[test]
    fn wide_source_catalog_uses_one_direct_target_lookup() {
        const SLOTS: usize = 128;
        let mut sources = (0..SLOTS)
            .map(|slot| SourceActivation {
                node: u32::try_from(slot).unwrap_or(u32::MAX),
                root_range: u64::try_from(SLOTS - slot).unwrap_or(u64::MAX)
                    ..u64::try_from(SLOTS - slot + 1).unwrap_or(u64::MAX),
                role: SourceRole::Predicate {
                    slot,
                    mode: crate::nodes::ConjunctMode::Cascade,
                },
            })
            .collect::<Vec<_>>();
        sources.push(SourceActivation {
            node: u32::try_from(SLOTS).unwrap_or(u32::MAX),
            root_range: 0..u64::try_from(SLOTS + 1).unwrap_or(u64::MAX),
            role: SourceRole::Projection,
        });
        let catalog = SourceCatalog::new(&sources);
        assert_eq!(catalog.predicates.len(), SLOTS);
        assert!(catalog.predicates.iter().all(Option::is_some));

        let mut matches = Vec::new();
        catalog.reset_lookup_probes();
        catalog.overlaps(
            &sources,
            ActivationTarget::PredicateSlot(SLOTS - 1),
            &(1..2),
            &mut matches,
        );
        assert_eq!(catalog.lookup_probes(), 1);
        assert_eq!(matches, [SLOTS - 1]);

        catalog.overlaps_all(&sources, &(0..u64::MAX), &mut matches);
        assert_eq!(matches, (0..=SLOTS).collect::<Vec<_>>());
    }

    #[test]
    fn unary_nodes_form_one_physical_pipeline_with_edge_sized_inputs() -> VortexResult<()> {
        let nodes = vec![
            NodeSpec::Struct {
                names: FieldNames::default(),
                children: Arc::from([]),
                push_batching: PushBatching::Streaming,
                push_passthrough: false,
            },
            NodeSpec::Struct {
                names: FieldNames::default(),
                children: Arc::from([0]),
                push_batching: PushBatching::Streaming,
                push_passthrough: false,
            },
            NodeSpec::Struct {
                names: FieldNames::default(),
                children: Arc::from([1]),
                push_batching: PushBatching::Streaming,
                push_passthrough: false,
            },
        ];
        let routes = reverse_routes(&nodes, 2)?;
        let sources = [SourceActivation {
            node: 0,
            root_range: 0..1,
            role: SourceRole::Projection,
        }];

        let pipelines = physical_pipelines(&nodes, &routes, &sources)?;

        assert_eq!(pipelines.len(), 1);
        assert_eq!(stage_nodes(&pipelines[0]), [0, 1, 2]);
        let inputs = forward_inputs(nodes.len(), &routes)?;
        assert_eq!(inputs.iter().map(Vec::len).sum::<usize>(), 2);
        let topology = physical_topology(pipelines, &inputs, nodes.len())?;
        let first = topology
            .outgoing(0)
            .ok_or_else(|| vortex_err!("first stage has no compiled target"))?;
        assert_eq!((first.pipeline, first.stage, first.node), (0, 1, 1));
        assert_eq!(first.input.index(), 0);
        assert!(!first.boundary);
        let second = topology
            .outgoing(1)
            .ok_or_else(|| vortex_err!("second stage has no compiled target"))?;
        assert_eq!((second.pipeline, second.stage, second.node), (0, 2, 2));
        assert!(!second.boundary);
        assert!(topology.outgoing(2).is_none());
        Ok(())
    }

    fn stage_nodes(pipeline: &PhysicalPipeline) -> Vec<u32> {
        pipeline.stages().iter().map(|stage| stage.node()).collect()
    }
}
