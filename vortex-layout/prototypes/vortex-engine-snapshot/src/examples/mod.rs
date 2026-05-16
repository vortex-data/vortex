use std::task::Context;
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::AsyncWorkId;
use crate::Batch;
use crate::Cardinality;
use crate::Cell;
use crate::ChannelBuffer;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineResult;
use crate::ExecutionMetrics;
use crate::BrokerId;
use crate::scheduler::FakeIoRequest;
use crate::InputPortId;
use crate::InputPortSpec;
use crate::InterestId;
use crate::InterestSpec;
use crate::LatencyClass;
use crate::MemoryReason;
use crate::Operator;
use crate::RowClass;
use crate::SimpleDelayBroker;
use crate::OperatorGraph;
use crate::OperatorNode;
use crate::OperatorSpec;
use crate::OutputPortSpec;
use crate::PreparedTask;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::ResourceKind;
use crate::ResourceSpec;
use crate::ResourceValue;
use crate::Row;
use crate::RowDemand;
use crate::ScheduleTrace;
use crate::TaskOptions;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExampleReport {
    pub output_rows: Vec<i64>,
    pub output_pairs: Vec<(i64, i64)>,
    pub metrics: ExecutionMetrics,
    pub trace: ScheduleTrace,
}

impl ExampleReport {
    fn from_sink(sink: Arc<Mutex<Vec<Row>>>, report: super::TaskReport) -> Self {
        let rows = sink.lock();
        let output_rows = rows
            .iter()
            .filter_map(|row| row.first().copied().flatten())
            .collect();
        let output_pairs = rows
            .iter()
            .filter_map(|row| {
                Some((
                    row.first().copied().flatten()?,
                    row.get(1).copied().flatten()?,
                ))
            })
            .collect();
        Self {
            output_rows,
            output_pairs,
            metrics: report.metrics,
            trace: report.trace,
        }
    }
}

pub fn basic_projection() -> EngineResult<ExampleReport> {
    let domain = domain("rows", 6);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(IntSource::new(
        "numbers",
        domain.clone(),
        rows1(&[1, 2, 3, 4, 5, 6]),
        Arc::clone(&metrics),
    )));
    let project = graph.add_operator(OperatorNode::new(ProjectOne::new(
        "project_double",
        domain.clone(),
        2,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        domain,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(project, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(project),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

/// Pre-built `OperatorGraph` plus its sink and metrics handles.
/// Used by drivers that want to run a graph without going through
/// the full example function.
pub struct BasicProjectionPipeline {
    pub graph: OperatorGraph,
    pub metrics: Arc<Mutex<ExecutionMetrics>>,
    pub sink_rows: Arc<Mutex<Vec<Row>>>,
}

/// Build a `source → project_double → collect` pipeline. The
/// operator labels are namespaced by `label_prefix` so multiple
/// parallel pipelines on the same metrics handle remain distinct.
pub fn basic_projection_pipeline(
    label_prefix: &str,
    inputs: &[i64],
    factor: i64,
) -> BasicProjectionPipeline {
    let row_domain = domain(
        &format!("{label_prefix}:rows"),
        u64_from_usize(inputs.len()),
    );
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(IntSource::new(
        format!("{label_prefix}:numbers"),
        row_domain.clone(),
        rows1(inputs),
        Arc::clone(&metrics),
    )));
    let project = graph.add_operator(OperatorNode::new(ProjectOne::new(
        format!("{label_prefix}:project_double"),
        row_domain.clone(),
        factor,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        format!("{label_prefix}:collect"),
        row_domain,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(project, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(project),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    BasicProjectionPipeline {
        graph,
        metrics,
        sink_rows,
    }
}

pub fn limit_backpropagation() -> EngineResult<ExampleReport> {
    let row_domain = domain("rows", 10);
    let limited = domain("limited", 3);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(IntSource::new(
        "numbers",
        row_domain.clone(),
        rows1(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        Arc::clone(&metrics),
    )));
    let limit = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_3",
        row_domain,
        limited.clone(),
        3,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(3),
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(limit, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(limit),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn filter_lateral_dontcare() -> EngineResult<ExampleReport> {
    let input = domain("input", 8);
    let filtered = domain("filtered", 3);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let keys = graph.add_operator(OperatorNode::new(IntSource::new(
        "keys",
        input.clone(),
        rows1(&[0, 1, 2, 3, 4, 5, 6, 7]),
        Arc::clone(&metrics),
    )));
    let predicate = graph.add_operator(OperatorNode::new(PredicateEval::new(
        "predicate_eval",
        input.clone(),
        |value| value % 2 == 0,
    )));
    let payload = graph.add_operator(OperatorNode::new(IntSource::new(
        "payload",
        input.clone(),
        rows1(&[100, 110, 120, 130, 140, 150, 160, 170]),
        Arc::clone(&metrics),
    )));
    let filter = graph.add_operator(OperatorNode::new(Filter::new(
        "filter",
        input,
        filtered.clone(),
        3,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        filtered,
        Some(3),
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(keys),
        vec![OperatorGraph::input(predicate, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(predicate),
        vec![OperatorGraph::input(filter, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(payload),
        vec![OperatorGraph::input(filter, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(filter),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn sorted_aggregate_limit() -> EngineResult<ExampleReport> {
    let input = domain("sorted", 10);
    let groups = domain("groups", 2);
    let limited = domain("limited_groups", 2);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let rows = vec![
        row(&[1, 10]),
        row(&[1, 20]),
        row(&[2, 5]),
        row(&[2, 7]),
        row(&[3, 100]),
        row(&[3, 200]),
        row(&[4, 1]),
        row(&[4, 2]),
        row(&[5, 3]),
        row(&[5, 4]),
    ];

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(
        IntSource::new("sorted_rows", input.clone(), rows, Arc::clone(&metrics)).with_batch_rows(1),
    ));
    let aggregate = graph.add_operator(OperatorNode::new(SortedAggregate::new(
        "sorted_aggregate",
        input,
        groups.clone(),
        2,
    )));
    let limit = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_groups",
        groups,
        limited.clone(),
        2,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(2),
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(aggregate, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(aggregate),
        vec![OperatorGraph::input(limit, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(limit),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn dynamic_filter_join() -> EngineResult<ExampleReport> {
    let build = domain("build", 4);
    let probe = domain("probe", 6);
    let probe_filtered = domain("probe_filtered", 3);
    let joined = domain("joined", 3);
    let projected = domain("projected", 3);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    graph.add_resource(ResourceSpec {
        id: "build_key_index".to_owned(),
        kind: ResourceKind::KeyIndex,
    });
    let build_source = graph.add_operator(OperatorNode::new(IntSource::new(
        "build_keys",
        build.clone(),
        vec![row(&[1, 0]), row(&[2, 1]), row(&[3, 2]), row(&[4, 3])],
        Arc::clone(&metrics),
    )));
    let build_index = graph.add_operator(OperatorNode::new(BuildKeyIndex::new(
        "build_index",
        build,
        "build_key_index",
    )));
    let probe_keys = graph.add_operator(OperatorNode::new(IntSource::new(
        "probe_keys",
        probe.clone(),
        rows1(&[0, 1, 5, 3, 3, 8]),
        Arc::clone(&metrics),
    )));
    let dynamic_predicate = graph.add_operator(OperatorNode::new(DynamicPredicate::new(
        "dynamic_predicate",
        probe.clone(),
        "build_key_index",
    )));
    let probe_payload = graph.add_operator(OperatorNode::new(IntSource::new(
        "probe_payload",
        probe.clone(),
        vec![
            row(&[0, 0]),
            row(&[1, 7]),
            row(&[5, 0]),
            row(&[3, 9]),
            row(&[3, 11]),
            row(&[8, 0]),
        ],
        Arc::clone(&metrics),
    )));
    let filter = graph.add_operator(OperatorNode::new(Filter::new(
        "probe_filter",
        probe,
        probe_filtered.clone(),
        3,
    )));
    let join = graph.add_operator(OperatorNode::new(JoinProbe::new(
        "join_probe",
        probe_filtered,
        joined.clone(),
        "build_key_index",
    )));
    let project = graph.add_operator(OperatorNode::new(LazyLeftProject::new(
        "left_payload_project",
        joined,
        projected.clone(),
        vec![1000, 2000, 3000, 4000],
        Arc::clone(&metrics),
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        projected,
        Some(3),
        Arc::clone(&sink_rows),
    )));

    graph.connect(
        OperatorGraph::output(build_source),
        vec![OperatorGraph::input(build_index, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(probe_keys),
        vec![OperatorGraph::input(dynamic_predicate, 0)],
        ChannelBuffer::dynamic_bytes(64, 4096, 4096),
    );
    graph.connect(
        OperatorGraph::output(dynamic_predicate),
        vec![OperatorGraph::input(filter, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(probe_payload),
        vec![OperatorGraph::input(filter, 0)],
        ChannelBuffer::dynamic_bytes(64, 4096, 4096),
    );
    graph.connect(
        OperatorGraph::output(filter),
        vec![OperatorGraph::input(join, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(join),
        vec![OperatorGraph::input(project, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(project),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let options = TaskOptions {
        max_turns: 10_000,
        memory_limit_bytes: 32,
        worker_count: 1,
    };
    let report = PreparedTask::prepare(graph, metrics, options)?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn fake_async_range_reads() -> EngineResult<ExampleReport> {
    let input = domain("async_rows", 6);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(AsyncRangeSource::new(
        "range_io",
        input.clone(),
        rows1(&[10, 20, 30, 40, 50, 60]),
        2,
        vec![1, 2, 3],
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        input,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn byte_accounted_memory_grants() -> EngineResult<ExampleReport> {
    let input = domain("byte_rows", 6);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(IntSource::new(
        "byte_source",
        input.clone(),
        rows1(&[1, 2, 3, 4, 5, 6]),
        Arc::clone(&metrics),
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        input,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::dynamic_bytes(64, 4096, 4096),
    );

    let options = TaskOptions {
        max_turns: 10_000,
        memory_limit_bytes: 32,
        worker_count: 1,
    };
    let report = PreparedTask::prepare(graph, metrics, options)?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn spmc_cse_requirement_merge() -> EngineResult<ExampleReport> {
    let input = domain("cse_rows", 6);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(IntSource::new(
        "cse_source",
        input.clone(),
        rows1(&[0, 1, 2, 4, 6, 5]),
        Arc::clone(&metrics),
    )));
    let left_sum = graph.add_operator(OperatorNode::new(RangeSumSink::new(
        "left_sum",
        input.clone(),
        0,
        3,
        Arc::clone(&sink_rows),
    )));
    let right_sum = graph.add_operator(OperatorNode::new(RangeSumSink::new(
        "right_sum",
        input,
        2,
        5,
        Arc::clone(&sink_rows),
    )));
    graph.connect_named(
        "cse_values",
        OperatorGraph::output(source),
        vec![
            OperatorGraph::input(left_sum, 0),
            OperatorGraph::input(right_sum, 0),
        ],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn late_dynamic_filter_cancellation() -> EngineResult<ExampleReport> {
    let probe = domain("late_probe", 6);
    let filtered = domain("late_filtered", 2);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    graph.add_resource(ResourceSpec {
        id: "late_suffix".to_owned(),
        kind: ResourceKind::Scalar,
    });
    let source = graph.add_operator(OperatorNode::new(AsyncRangeSource::new(
        "late_probe",
        probe.clone(),
        rows1(&[100, 200, 300, 400, 500, 600]),
        2,
        vec![1, 4, 64],
    )));
    let _suffix = graph.add_operator(OperatorNode::new(DelayedScalarResource::new(
        "late_suffix_build",
        "late_suffix",
        3,
        2,
    )));
    let filter = graph.add_operator(OperatorNode::new(LateSuffixFilter::new(
        "late_dynamic_filter",
        probe,
        filtered.clone(),
        "late_suffix",
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        filtered,
        Some(2),
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(filter, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(filter),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn parent_child_offsets_translation() -> EngineResult<ExampleReport> {
    // Two parent rows, each with a list of grandchild values. The
    // ListOffsetsOperator publishes contiguous one-to-many offsets.
    // GroupedAggregate consumes child values plus offsets and emits one row
    // per parent. A LIMIT 1 on the parent output shows back-propagation
    // through the offset relation: the second parent's child rows become
    // NotNeeded once the limit saturates.
    let parent = domain("parent_rows", 2);
    let child = domain("child_rows", 5);
    let aggregated = domain("parent_min", 2);
    let limited = domain("parent_min_limited", 1);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();

    let offsets = graph.add_operator(OperatorNode::new(ListOffsetsSource::new(
        "offsets",
        parent.clone(),
        // Parent 0 has children [0..2), parent 1 has children [2..5).
        vec![0, 2, 5],
    )));
    let child_values = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "child_values",
            child.clone(),
            rows1(&[10, 20, 7, 8, 9]),
            Arc::clone(&metrics),
        )
        .with_batch_rows(1),
    ));
    let aggregate = graph.add_operator(OperatorNode::new(ParentChildMin::new(
        "min_per_parent",
        parent,
        child,
        aggregated.clone(),
    )));
    let limit = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_parents",
        aggregated,
        limited.clone(),
        1,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(1),
        Arc::clone(&sink_rows),
    )));

    graph.connect(
        OperatorGraph::output(offsets),
        vec![OperatorGraph::input(aggregate, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(child_values),
        vec![OperatorGraph::input(aggregate, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(aggregate),
        vec![OperatorGraph::input(limit, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(limit),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

/// Three-level spiral query: `min(grandchild)` per parent, through two
/// nested ParentChild offset relations. Demonstrates that
/// back-propagation reaches the deepest source: with `Limit(1)` on the
/// parent output, the grandchild scan should only decode rows
/// belonging to the first parent's children.
///
/// Topology (2 parents, 4 children, 8 grandchildren):
///   parent 0 → children [c0, c1]
///     c0 → grandchildren [10, 20]
///     c1 → grandchildren [7, 8]
///   parent 1 → children [c2, c3]
///     c2 → grandchildren [30, 40]
///     c3 → grandchildren [50, 60]
pub fn parent_child_grandchild_offsets_translation() -> EngineResult<ExampleReport> {
    let parent = domain("parent_rows", 2);
    let child = domain("child_rows", 4);
    let grandchild = domain("grandchild_rows", 8);
    let child_min = domain("child_min", 4);
    let parent_min = domain("parent_min", 2);
    let limited = domain("parent_min_limited", 1);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();

    // child→grandchild offsets: c0=[0,2), c1=[2,4), c2=[4,6), c3=[6,8).
    let cg_offsets = graph.add_operator(OperatorNode::new(ListOffsetsSource::new(
        "cg_offsets",
        child.clone(),
        vec![0, 2, 4, 6, 8],
    )));
    let grandchild_values = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "grandchild_values",
            grandchild.clone(),
            rows1(&[10, 20, 7, 8, 30, 40, 50, 60]),
            Arc::clone(&metrics),
        )
        .with_batch_rows(1),
    ));
    let level1 = graph.add_operator(OperatorNode::new(ParentChildMin::new(
        "min_per_child",
        child.clone(),
        grandchild,
        child_min.clone(),
    )));

    // parent→child offsets: p0=[0,2), p1=[2,4).
    let pc_offsets = graph.add_operator(OperatorNode::new(ListOffsetsSource::new(
        "pc_offsets",
        parent.clone(),
        vec![0, 2, 4],
    )));
    let level2 = graph.add_operator(OperatorNode::new(ParentChildMin::new(
        "min_per_parent",
        parent,
        child_min,
        parent_min.clone(),
    )));

    let limit = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_parents",
        parent_min,
        limited.clone(),
        1,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(1),
        Arc::clone(&sink_rows),
    )));

    // child→grandchild wiring
    graph.connect(
        OperatorGraph::output(cg_offsets),
        vec![OperatorGraph::input(level1, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(grandchild_values),
        vec![OperatorGraph::input(level1, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    // parent→child wiring: level1's output (per-child min) feeds level2's values input
    graph.connect(
        OperatorGraph::output(pc_offsets),
        vec![OperatorGraph::input(level2, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(level1),
        vec![OperatorGraph::input(level2, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    // limit + sink
    graph.connect(
        OperatorGraph::output(level2),
        vec![OperatorGraph::input(limit, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(limit),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

/// Parameterized 3-level spiral query for benchmarking.
///
/// - `parent_count`: number of parent rows.
/// - `child_per_parent`: number of children per parent (uniform).
/// - `grandchild_per_child`: number of grandchildren per child (uniform).
/// - `limit`: cap on the number of parent output rows.
///
/// Grandchild values are a deterministic ramp `0..(parent_count *
/// child_per_parent * grandchild_per_child) as i64` so the test can
/// assert exact min values without a separate fixture.
pub fn spiral_three_level(
    parent_count: u64,
    child_per_parent: u64,
    grandchild_per_child: u64,
    limit: u64,
) -> EngineResult<ExampleReport> {
    let child_count = parent_count * child_per_parent;
    let grandchild_count = child_count * grandchild_per_child;

    let parent = domain("parent_rows", parent_count);
    let child = domain("child_rows", child_count);
    let grandchild = domain("grandchild_rows", grandchild_count);
    let child_min = domain("child_min", child_count);
    let parent_min = domain("parent_min", parent_count);
    let limited = domain("parent_min_limited", limit);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    // child→grandchild offsets: each child covers grandchild_per_child rows.
    let cg_offsets_vec: Vec<i64> = (0..=child_count)
        .map(|i| (i * grandchild_per_child) as i64)
        .collect();
    // parent→child offsets: each parent covers child_per_parent rows.
    let pc_offsets_vec: Vec<i64> = (0..=parent_count)
        .map(|i| (i * child_per_parent) as i64)
        .collect();
    let grandchild_vals: Vec<i64> = (0..grandchild_count as i64).collect();

    let mut graph = OperatorGraph::new();

    let cg_offsets = graph.add_operator(OperatorNode::new(ListOffsetsSource::new(
        "cg_offsets",
        child.clone(),
        cg_offsets_vec,
    )));
    let grandchild_values = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "grandchild_values",
            grandchild.clone(),
            rows1(&grandchild_vals),
            Arc::clone(&metrics),
        )
        .with_batch_rows(64),
    ));
    let level1 = graph.add_operator(OperatorNode::new(ParentChildMin::new(
        "min_per_child",
        child.clone(),
        grandchild,
        child_min.clone(),
    )));
    let pc_offsets = graph.add_operator(OperatorNode::new(ListOffsetsSource::new(
        "pc_offsets",
        parent.clone(),
        pc_offsets_vec,
    )));
    let level2 = graph.add_operator(OperatorNode::new(ParentChildMin::new(
        "min_per_parent",
        parent,
        child_min,
        parent_min.clone(),
    )));
    let limit_op = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_parents",
        parent_min,
        limited.clone(),
        limit,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(limit),
        Arc::clone(&sink_rows),
    )));

    let bb = |n: usize| ChannelBuffer::bounded_bytes(n);
    graph.connect(OperatorGraph::output(cg_offsets), vec![OperatorGraph::input(level1, 0)], bb(64 << 10));
    graph.connect(OperatorGraph::output(grandchild_values), vec![OperatorGraph::input(level1, 1)], bb(64 << 10));
    graph.connect(OperatorGraph::output(pc_offsets), vec![OperatorGraph::input(level2, 0)], bb(64 << 10));
    graph.connect(OperatorGraph::output(level1), vec![OperatorGraph::input(level2, 1)], bb(64 << 10));
    graph.connect(OperatorGraph::output(level2), vec![OperatorGraph::input(limit_op, 0)], bb(64 << 10));
    graph.connect(OperatorGraph::output(limit_op), vec![OperatorGraph::input(collect, 0)], bb(64 << 10));

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn presorted_merge_join() -> EngineResult<ExampleReport> {
    let left = domain("merge_left", 5);
    let right = domain("merge_right", 5);
    let joined = domain("merge_join", 3);
    let limited = domain("merge_join_limited", 2);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let left_scan = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "left_keys",
            left.clone(),
            vec![
                row(&[1, 100]),
                row(&[3, 300]),
                row(&[5, 500]),
                row(&[7, 700]),
                row(&[9, 900]),
            ],
            Arc::clone(&metrics),
        )
        .with_batch_rows(1),
    ));
    let right_scan = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "right_keys",
            right.clone(),
            vec![
                row(&[2, 22]),
                row(&[3, 33]),
                row(&[5, 55]),
                row(&[5, 56]),
                row(&[8, 88]),
            ],
            Arc::clone(&metrics),
        )
        .with_batch_rows(1),
    ));
    let join = graph.add_operator(OperatorNode::new(MergeJoin::new(
        "merge_join",
        left,
        right,
        joined.clone(),
    )));
    let limit = graph.add_operator(OperatorNode::new(Limit::new(
        "limit_join",
        joined,
        limited.clone(),
        2,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        limited,
        Some(2),
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(left_scan),
        vec![OperatorGraph::input(join, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(right_scan),
        vec![OperatorGraph::input(join, 1)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(join),
        vec![OperatorGraph::input(limit, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(limit),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

/// Pipelined I/O + CPU example.
///
/// A brokered source registers N interests, each with a `delay_turns`
/// latency. A CPU-bound transform sums incoming rows. The expected
/// behavior is that the scheduler admits broker proposals
/// pre-emptively in the same turn as CPU `Emit`/`Cpu` work, so the
/// I/O delay overlaps with the rest of the work rather than
/// serializing.
///
/// Validation: counts the number of broker submissions admitted
/// concurrently in single turns by inspecting the trace.
pub fn pipelined_broker_io() -> EngineResult<ExampleReport> {
    pipelined_broker_io_inner(None)
}

/// Variant that lets the caller swap in a custom `DriverIo`
/// implementation in place of the default `FakeDriverIo`.
/// Demonstrates the broker / driver substrate split is pluggable:
/// the same `SimpleDelayBroker` runs over any substrate, and the
/// driver's choice of substrate is what determines real-world I/O.
///
/// The substrate is passed in at drive time (`run_with_io`), not
/// owned by `PreparedTask`, so that `!Send` driver substrates can
/// stay pinned to one worker thread without leaking that bound
/// across engine state.
pub fn pipelined_broker_io_with_substrate(
    substrate: Box<dyn super::DriverIo + Send>,
) -> EngineResult<ExampleReport> {
    pipelined_broker_io_inner(Some(substrate))
}

fn pipelined_broker_io_inner(
    substrate: Option<Box<dyn super::DriverIo + Send>>,
) -> EngineResult<ExampleReport> {
    let domain = domain("brokered_rows", 6);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(BrokeredSource::new(
        "brokered_source",
        domain.clone(),
        BrokerId::from_index(0),
        // Six rows split into three pending interests of two rows each.
        vec![
            (DomainSpan::new(0, 2), Batch::from_values(0, [10, 20])),
            (DomainSpan::new(2, 2), Batch::from_values(2, [30, 40])),
            (DomainSpan::new(4, 2), Batch::from_values(4, [50, 60])),
        ],
        // Network-RTT latency: 16 turns per request in the broker's
        // simulator (approximate).
        4,
        LatencyClass::NetworkRtt,
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        domain,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let mut prepared = PreparedTask::prepare(graph, metrics, TaskOptions::default())?;
    prepared.register_broker(Box::new(SimpleDelayBroker::new(
        BrokerId::from_index(0),
        "range_broker",
    )));
    let report = match substrate {
        Some(mut io) => prepared.run_with_io(io.as_mut())?,
        None => prepared.run()?,
    };
    Ok(ExampleReport::from_sink(sink_rows, report))
}

pub fn lazy_vortex_batch_facade() -> EngineResult<ExampleReport> {
    let input = domain("vortex_rows", 3);
    let output = domain("vortex_projected", 3);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let sink_rows = Arc::new(Mutex::new(Vec::new()));

    let mut graph = OperatorGraph::new();
    let scan = graph.add_operator(OperatorNode::new(LazyVortexScan::new(
        "vortex_scan",
        input.clone(),
        vec![
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1000), Some(2000), Some(3000)],
        ],
        Arc::clone(&metrics),
    )));
    let project = graph.add_operator(OperatorNode::new(TakeLazyColumn::new(
        "take_payload",
        input,
        output.clone(),
        1,
        "vortex_scan",
        Arc::clone(&metrics),
    )));
    let collect = graph.add_operator(OperatorNode::new(CollectSink::new(
        "collect",
        output,
        None,
        Arc::clone(&sink_rows),
    )));
    graph.connect(
        OperatorGraph::output(scan),
        vec![OperatorGraph::input(project, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );
    graph.connect(
        OperatorGraph::output(project),
        vec![OperatorGraph::input(collect, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    Ok(ExampleReport::from_sink(sink_rows, report))
}

/// Demonstrates structural lane-splitting for a `Lanes { max }` operator.
///
/// A 2-lane sink reads from a single source channel. Each lane records
/// its observed lane index and per-lane update count into a shared
/// `LaneObservations` structure. The structural test asserts that:
///
/// - `init_local` runs once per lane (with the correct `lane.index`);
/// - both lanes receive `update` calls every turn until they finish;
/// - the sink as a whole only finishes once every lane has finished.
///
/// Per-lane channel routing is a follow-up; today the source's batches
/// are popped by whichever lane wins admission first. This example is
/// about the *structural* per-lane state machinery, not about input
/// fan-out.
pub fn lane_split_demo() -> EngineResult<LaneSplitReport> {
    let row_domain = domain("rows", 4);
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let observations = Arc::new(LaneObservations::default());

    let mut graph = OperatorGraph::new();
    let source = graph.add_operator(OperatorNode::new(
        IntSource::new(
            "numbers",
            row_domain.clone(),
            rows1(&[10, 20, 30, 40]),
            Arc::clone(&metrics),
        )
        .with_batch_rows(1),
    ));
    let sink = graph.add_operator(OperatorNode::new(LaneCounterSink::new(
        "lane_sink",
        row_domain,
        2,
        Arc::clone(&observations),
    )));
    graph.connect(
        OperatorGraph::output(source),
        vec![OperatorGraph::input(sink, 0)],
        ChannelBuffer::bounded_bytes(4096),
    );

    // The sink uses Workers { max: Some(2) }; worker_count must
    // match for both lanes to be created.
    let options = TaskOptions {
        worker_count: 2,
        ..TaskOptions::default()
    };
    let report = PreparedTask::prepare(graph, metrics, options)?.run()?;
    Ok(LaneSplitReport {
        observations,
        metrics: report.metrics,
        trace: report.trace,
    })
}

#[derive(Default)]
pub struct LaneObservations {
    /// Lane indices observed at `init_local` time, in init order.
    pub init_lane_indices: Mutex<Vec<usize>>,
    /// Per-lane update counts. Indexed by `lane.index`.
    pub update_counts: Mutex<Vec<u64>>,
    /// Per-lane run counts. Indexed by `lane.index`.
    pub run_counts: Mutex<Vec<u64>>,
    /// Per-lane batch-pop counts. Indexed by `lane.index`.
    pub batches_popped_by_lane: Mutex<Vec<u64>>,
}

pub struct LaneSplitReport {
    pub observations: Arc<LaneObservations>,
    pub metrics: ExecutionMetrics,
    pub trace: ScheduleTrace,
}

struct LaneCounterSink {
    label: String,
    domain: Domain,
    max_lanes: usize,
    observations: Arc<LaneObservations>,
}

struct LaneCounterState {
    lane: usize,
    finished: bool,
}

impl LaneCounterSink {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        max_lanes: usize,
        observations: Arc<LaneObservations>,
    ) -> Self {
        // Pre-size the per-lane counter vectors so lane indexing is
        // direct.
        *observations.update_counts.lock() = vec![0; max_lanes];
        *observations.run_counts.lock() = vec![0; max_lanes];
        *observations.batches_popped_by_lane.lock() = vec![0; max_lanes];
        Self {
            label: label.into(),
            domain,
            max_lanes,
            observations,
        }
    }
}

impl Operator for LaneCounterSink {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = LaneCounterState;

    fn spec(&self) -> OperatorSpec {
        // Lanes share the input channel and race to pop batches.
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],None,
        )
        .lanes(Some(self.max_lanes))
    }

    fn init_global(
        &self,
        _ctx: &mut crate::GlobalInitCtx<'_>,
    ) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        ctx: &mut crate::LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        self.observations
            .init_lane_indices
            .lock()
            .push(ctx.lane().index);
        Ok(LaneCounterState {
            lane: ctx.lane().index,
            finished: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let len = exact_len(&self.domain);
        let mut requirement = RequirementSet::default();
        requirement.require_span(DomainSpan::new(0, len));
        inputs[0] = requirement;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        self.observations.update_counts.lock()[local.lane] += 1;
        let port = InputPortId::from_index(0);
        let drainable = ctx.peek(port).is_some() || ctx.input_finished(port);
        let class = if drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        // Per-lane EV variation: lane 0 proposes a small bonus so
        // the scheduler tie-breaks it ahead of lane 1 on the first
        // batch; lane 1 is preferred on subsequent batches once
        // lane 0's local cursor has moved.
        let value = if local.lane == 0 {
            WorkValue::required(2)
        } else {
            WorkValue::required(1)
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        self.observations.run_counts.lock()[local.lane] += 1;
        if let Some(_batch) = ctx.pop(InputPortId::from_index(0)) {
            self.observations.batches_popped_by_lane.lock()[local.lane] += 1;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            local.finished = true;
            return Ok(WorkStatus::Finished);
        };
        Ok(WorkStatus::Made)
    }
}

fn domain(name: &str, len: u64) -> Domain {
    Domain::new(DomainId::new(name), Cardinality::Exact(len))
}

/// For sources without a simple cursor: pick the WorkValue band that
/// matches the highest demand band currently present in `output_req`.
/// Empty/Unknown requirement -> Required (propagation hasn't reached us
/// yet; default to running so it can catch up). Only-NotNeeded -> empty
/// (lowest priority slot for cleanup work). Otherwise: Required if any
/// row is Needed, else Candidate.
fn source_band_workvalue(output_req: &RequirementSet) -> WorkValue {
    let mut saw_required = false;
    let mut saw_candidate = false;
    let mut saw_not_needed = false;
    for iv in output_req.intervals() {
        match iv.demand {
            RowDemand::Needed => saw_required = true,
            RowDemand::Candidate => saw_candidate = true,
            RowDemand::NotNeeded => saw_not_needed = true,
            RowDemand::Unknown => {}
        }
    }
    if saw_required {
        WorkValue::required(1)
    } else if saw_candidate {
        WorkValue::candidate(1, 128)
    } else if saw_not_needed {
        WorkValue::empty()
    } else {
        // Empty requirement -- back-prop not yet arrived. Treat as
        // Required so propagation can catch up.
        WorkValue::required(1)
    }
}

fn rows1(values: &[i64]) -> Vec<Row> {
    values.iter().map(|value| vec![Some(*value)]).collect()
}

fn row(values: &[i64]) -> Row {
    values.iter().map(|value| Some(*value)).collect()
}

fn exact_len(domain: &Domain) -> u64 {
    match domain.cardinality() {
        Cardinality::Exact(len) => len,
        Cardinality::Unknown => 0,
    }
}

fn usize_from_u64(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

struct IntSource {
    label: String,
    domain: Domain,
    rows: Vec<Row>,
    metrics: Arc<Mutex<ExecutionMetrics>>,
    max_batch_rows: usize,
}

struct IntSourceState {
    cursor: u64,
    sealed: bool,
}

impl IntSource {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        rows: Vec<Row>,
        metrics: Arc<Mutex<ExecutionMetrics>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            rows,
            metrics,
            max_batch_rows: usize::MAX,
        }
    }

    fn with_batch_rows(mut self, max_batch_rows: usize) -> Self {
        self.max_batch_rows = max_batch_rows;
        self
    }
}

impl Operator for IntSource {
    type GlobalState = ();
    type LocalState = IntSourceState;

    fn spec(&self) -> OperatorSpec {
        let columns = self.rows.first().map_or(1, Vec::len);
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new("out", self.domain.clone(), columns)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(IntSourceState {
            cursor: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        // Honor the row-demand band for our next cursor position. The
        // scheduler relies on `WorkValue` distinguishing Required vs
        // Candidate to give witness-producers a 10^9x priority edge
        // over speculative source reads. See spec: "WorkValue must
        // reflect demand band of next row."
        let demand = ctx.output_requirement().row(state.cursor);
        let value = match demand {
            // NotNeeded rows: propose `empty()` rather than skipping
            // entirely. The source still needs at least one scheduling
            // slot to advance its cursor through NotNeeded rows and
            // seal; `empty()` makes that slot the lowest-priority
            // option so witness-producing work always wins. (Strictly
            // skipping the proposal here would leave the source
            // unsealed and quiesce the graph; see fix report.)
            RowDemand::NotNeeded => WorkValue::empty(),
            RowDemand::Needed => WorkValue::required(1),
            RowDemand::Candidate => WorkValue::candidate(1, 128),
            // Treat `Unknown` as required — back-prop hasn't reached
            // us yet, so default to running so propagation can catch up.
            RowDemand::Unknown => WorkValue::required(1),
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        let requirement = ctx.output_requirement();
        // Bulk-skip a contiguous NotNeeded span at the cursor in one
        // call. Per-row skipping returns WorkStatus::Made without
        // push/pop/seal and so doesn't flip the scheduler's progress
        // flag; a long NotNeeded run would quiesce the graph.
        let rows_len = u64_from_usize(self.rows.len());
        while state.cursor < rows_len
            && matches!(requirement.row(state.cursor), RowDemand::NotNeeded)
        {
            self.metrics.lock().add_source_rows_skipped(&self.label, 1);
            state.cursor += 1;
        }
        if usize_from_u64(state.cursor) == self.rows.len() {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        };
        let mut output_rows = Vec::new();
        let start = state.cursor;
        while let Some(source_row) = self.rows.get(usize_from_u64(state.cursor)) {
            let row_demand = requirement.row(state.cursor);
            match row_demand {
                RowDemand::NotNeeded => {
                    self.metrics.lock().add_source_rows_skipped(&self.label, 1);
                    state.cursor += 1;
                }
                RowDemand::Needed | RowDemand::Candidate => {
                    let mut row = Vec::with_capacity(source_row.len());
                    for (column, cell) in source_row.iter().enumerate() {
                        row.push(*cell);
                        self.metrics
                            .lock()
                            .add_source_value_read(&self.label, column, 1);
                    }
                    output_rows.push(row);
                    self.metrics.lock().add_source_rows_read(&self.label, 1);
                    state.cursor += 1;
                    if output_rows.len() >= self.max_batch_rows {
                        break;
                    }
                }
                RowDemand::Unknown => break,
            }
        };
        if !output_rows.is_empty() {
            if !ctx.has_capacity() {
                return Ok(WorkStatus::Made);
            };
            let _reservation = ctx
                .memory()
                .try_reserve(output_rows.len(), MemoryReason::OutputBatch)?;
            ctx.push(Batch::from_rows(start, output_rows),
            )?;
            return Ok(WorkStatus::Made);
        };
        if usize_from_u64(state.cursor) == self.rows.len() {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct ProjectOne {
    label: String,
    domain: Domain,
    factor: i64,
}

impl ProjectOne {
    fn new(label: impl Into<String>, domain: Domain, factor: i64) -> Self {
        Self {
            label: label.into(),
            domain,
            factor,
        }
    }
}

impl Operator for ProjectOne {
    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],
            Some(OutputPortSpec::new("out", self.domain.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        inputs[0] = output.clone();
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let rows = batch
                .into_rows()
                .into_iter()
                .map(|row| {
                    vec![
                        row.first()
                            .copied()
                            .flatten()
                            .map(|value| value * self.factor),
                    ]
                })
                .collect::<Vec<_>>();
            ctx.push(Batch::from_rows(0, rows),
            )?;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct CollectSink {
    label: String,
    domain: Domain,
    limit: Option<u64>,
    rows: Arc<Mutex<Vec<Row>>>,
}

impl CollectSink {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        limit: Option<u64>,
        rows: Arc<Mutex<Vec<Row>>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            limit,
            rows,
        }
    }
}

impl Operator for CollectSink {
    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],None,
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let len = self.limit.unwrap_or_else(|| exact_len(&self.domain));
        let mut requirement = RequirementSet::default();
        requirement.require_span(DomainSpan::new(0, len));
        if let Some(limit) = self.limit {
            let total = exact_len(&self.domain);
            if total > limit {
                requirement.not_needed_span(DomainSpan::new(limit, total - limit));
            }
        };
        inputs[0] = requirement;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let mut rows = self.rows.lock();
            let remaining = self
                .limit
                .map(|limit| usize_from_u64(limit.saturating_sub(u64_from_usize(rows.len()))))
                .unwrap_or(usize::MAX);
            rows.extend(batch.into_rows().into_iter().take(remaining));
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) || self.limit_reached() {
            *state = true;
            return Ok(WorkStatus::Finished);
        };
        if *state {
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

impl CollectSink {
    fn limit_reached(&self) -> bool {
        self.limit
            .is_some_and(|limit| u64_from_usize(self.rows.lock().len()) >= limit)
    }
}

struct Limit {
    label: String,
    input: Domain,
    output: Domain,
    limit: u64,
}

impl Limit {
    fn new(label: impl Into<String>, input: Domain, output: Domain, limit: u64) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            limit,
        }
    }
}

struct LimitState {
    emitted: u64,
    sealed: bool,
}

impl Operator for Limit {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = LimitState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.input.clone(), 1)],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(LimitState {
            emitted: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let mut requirement = RequirementSet::default();
        requirement.require_span(DomainSpan::new(0, self.limit));
        let total = exact_len(&self.input);
        if total > self.limit {
            requirement.not_needed_span(DomainSpan::new(self.limit, total - self.limit));
        };
        inputs[0] = requirement;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        if state.emitted >= self.limit {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        };
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let remaining = usize_from_u64(self.limit - state.emitted);
            let rows = batch
                .into_rows()
                .into_iter()
                .take(remaining)
                .collect::<Vec<_>>();
            let start = state.emitted;
            state.emitted += rows.len() as u64;
            ctx.push(Batch::from_rows(start, rows),
            )?;
            return Ok(WorkStatus::Made);
        }
        Ok(WorkStatus::Made)
    }
}

struct PredicateEval {
    label: String,
    domain: Domain,
    predicate: fn(i64) -> bool,
}

impl PredicateEval {
    fn new(label: impl Into<String>, domain: Domain, predicate: fn(i64) -> bool) -> Self {
        Self {
            label: label.into(),
            domain,
            predicate,
        }
    }
}

impl Operator for PredicateEval {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],
            Some(OutputPortSpec::new("out", self.domain.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let mut input = RequirementSet::default();
        let output = output.clone();
        for iv in output.intervals() {
            let span = DomainSpan::new(iv.start, iv.end - iv.start);
            match iv.demand {
                RowDemand::Needed => input.require_span(span),
                RowDemand::Candidate => input.candidate_span(span),
                RowDemand::NotNeeded => input.not_needed_span(span),
                RowDemand::Unknown => {}
            }
        };
        inputs[0] = input;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let start = batch.span().start();
            let rows = batch
                .into_rows()
                .into_iter()
                .map(|row| {
                    let value = row.first().copied().flatten().unwrap_or_default();
                    vec![Some(i64::from((self.predicate)(value)))]
                })
                .collect::<Vec<_>>();
            ctx.push(Batch::from_rows(start, rows),
            )?;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct Filter {
    label: String,
    input: Domain,
    output: Domain,
    expected_output: u64,
}

struct FilterState {
    predicates: BTreeMap<u64, bool>,
    selected: Vec<u64>,
    emitted: u64,
    sealed: bool,
}

impl Filter {
    fn new(label: impl Into<String>, input: Domain, output: Domain, expected_output: u64) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            expected_output,
        }
    }
}

impl Operator for Filter {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = FilterState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![
                InputPortSpec::new("values", self.input.clone(), 2),
                InputPortSpec::new("predicate", self.input.clone(), 1),
            ],
            Some(OutputPortSpec::new("out", self.output.clone(), 2)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(FilterState {
            predicates: BTreeMap::new(),
            selected: Vec::new(),
            emitted: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let output = output.clone();
        let needed = output.required_count_from_zero().max(self.expected_output);
        let mut values = RequirementSet::default();
        let mut predicate = RequirementSet::default();
        let total = exact_len(&self.input);
        let needed_usize = usize_from_u64(needed);
        if state.selected.len() < needed_usize {
            predicate.candidate_span(DomainSpan::new(0, total));
        };
        let cutoff = (state.selected.len() >= needed_usize && needed > 0)
            .then(|| state.selected[needed_usize - 1] + 1);
        for (row, _value) in &state.predicates {
            if cutoff.is_some_and(|cutoff| *row >= cutoff) {
                continue;
            };
            values.require_row(*row);
            predicate.require_row(*row);
        };
        if let Some(cutoff) = cutoff
            && total > cutoff
        {
            values.not_needed_span(DomainSpan::new(cutoff, total - cutoff));
            predicate.not_needed_span(DomainSpan::new(cutoff, total - cutoff));
        };
        inputs[0] = values;
        inputs[1] = predicate;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(1)) {
            let start = batch.span().start();
            for (offset, row) in batch.into_rows().into_iter().enumerate() {
                let ordinal = start + offset as u64;
                let passed = row.first().copied().flatten().unwrap_or_default() != 0;
                state.predicates.insert(ordinal, passed);
                if passed {
                    state.selected.push(ordinal);
                }
            };
            return Ok(WorkStatus::Made);
        };
        if let Some(batch) = ctx.peek(InputPortId::from_index(0)) {
            let all_known = (batch.span().start()..batch.span().end())
                .all(|ordinal| state.predicates.contains_key(&ordinal));
            if !all_known {
                return Ok(WorkStatus::Made);
            };
            let Some(batch) = ctx.pop(InputPortId::from_index(0)) else {
                return Ok(WorkStatus::Made);
            };
            let start = batch.span().start();
            let mut out = Vec::new();
            for (offset, row) in batch.into_rows().into_iter().enumerate() {
                let ordinal = start + offset as u64;
                if state.predicates.get(&ordinal).copied().unwrap_or_default() {
                    out.push(row);
                }
            };
            if !out.is_empty() {
                let start = state.emitted;
                state.emitted += out.len() as u64;
                ctx.push(Batch::from_rows(start, out),
                )?;
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0))
            && ctx.input_finished(InputPortId::from_index(1))
            && !state.sealed
        {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

// Additional operators for examples 4 and 5 follow the same small contract.

struct SortedAggregate {
    label: String,
    input: Domain,
    output: Domain,
    limit_groups: u64,
}

struct SortedAggregateState {
    current_group: Option<i64>,
    current_sum: i64,
    emitted: u64,
    consumed: u64,
    suffix_start: Option<u64>,
    sealed: bool,
}

impl SortedAggregate {
    fn new(label: impl Into<String>, input: Domain, output: Domain, limit_groups: u64) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            limit_groups,
        }
    }
}

impl Operator for SortedAggregate {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = SortedAggregateState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.input.clone(), 2)],
            Some(OutputPortSpec::new("out", self.output.clone(), 2)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(SortedAggregateState {
            current_group: None,
            current_sum: 0,
            emitted: 0,
            consumed: 0,
            suffix_start: None,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let mut input = RequirementSet::default();
        let total = exact_len(&self.input);
        if let Some(suffix_start) = state.suffix_start {
            input.not_needed_span(DomainSpan::new(suffix_start, total - suffix_start));
            input.candidate_span(DomainSpan::new(0, suffix_start));
        } else {
            input.candidate_span(DomainSpan::new(0, total));
        };
        inputs[0] = input;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            for row in batch.into_rows() {
                state.consumed += 1;
                let group = row.first().copied().flatten().unwrap_or_default();
                let value = row.get(1).copied().flatten().unwrap_or_default();
                if state.current_group.is_none() {
                    state.current_group = Some(group);
                    state.current_sum = value;
                    continue;
                };
                if state.current_group == Some(group) {
                    state.current_sum += value;
                    continue;
                };
                if state.emitted < self.limit_groups {
                    let out = vec![vec![state.current_group, Some(state.current_sum)]];
                    ctx.push(Batch::from_rows(state.emitted, out),
                    )?;
                    state.emitted += 1;
                };
                state.current_group = Some(group);
                state.current_sum = value;
                if state.emitted >= self.limit_groups {
                    state.suffix_start = Some(state.consumed);
                    return Ok(WorkStatus::Made);
                }
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            if state.emitted < self.limit_groups
                && let Some(group) = state.current_group
            {
                let out = vec![vec![Some(group), Some(state.current_sum)]];
                ctx.push(Batch::from_rows(state.emitted, out),
                )?;
                state.emitted += 1;
                return Ok(WorkStatus::Made);
            };
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct BuildKeyIndex {
    label: String,
    domain: Domain,
    resource: &'static str,
}

impl BuildKeyIndex {
    fn new(label: impl Into<String>, domain: Domain, resource: &'static str) -> Self {
        Self {
            label: label.into(),
            domain,
            resource,
        }
    }
}

impl Operator for BuildKeyIndex {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = BTreeMap<i64, Vec<usize>>;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 2)],None,
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(BTreeMap::new())
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let mut requirement = RequirementSet::default();
        requirement.require_span(DomainSpan::new(0, exact_len(&self.domain)));
        inputs[0] = requirement;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            for row in batch.into_rows() {
                let key = row.first().copied().flatten().unwrap_or_default();
                let raw_row_id = row.get(1).copied().flatten().unwrap_or_default();
                if let Ok(row_id) = usize::try_from(raw_row_id) {
                    state.entry(key).or_default().push(row_id);
                }
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            ctx.publish_resource(self.resource, ResourceValue::KeyIndex(state.clone()))?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct DynamicPredicate {
    label: String,
    domain: Domain,
    resource: &'static str,
}

impl DynamicPredicate {
    fn new(label: impl Into<String>, domain: Domain, resource: &'static str) -> Self {
        Self {
            label: label.into(),
            domain,
            resource,
        }
    }
}

impl Operator for DynamicPredicate {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("keys", self.domain.clone(), 1)],
            Some(OutputPortSpec::new("out", self.domain.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let mut input = RequirementSet::default();
        let output = output.clone();
        for iv in output.intervals() {
            let span = DomainSpan::new(iv.start, iv.end - iv.start);
            match iv.demand {
                RowDemand::Needed => input.require_span(span),
                RowDemand::Candidate => input.candidate_span(span),
                RowDemand::NotNeeded => input.not_needed_span(span),
                RowDemand::Unknown => {}
            }
        };
        inputs[0] = input;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        let Some(ResourceValue::KeyIndex(index)) = ctx.resource(self.resource) else {
            return Ok(WorkStatus::Made);
        };
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let start = batch.span().start();
            let rows = batch
                .into_rows()
                .into_iter()
                .map(|row| {
                    let key = row.first().copied().flatten().unwrap_or_default();
                    vec![Some(i64::from(index.contains_key(&key)))]
                })
                .collect::<Vec<_>>();
            ctx.push(Batch::from_rows(start, rows),
            )?;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct JoinProbe {
    label: String,
    input: Domain,
    output: Domain,
    resource: &'static str,
}

impl JoinProbe {
    fn new(
        label: impl Into<String>,
        input: Domain,
        output: Domain,
        resource: &'static str,
    ) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            resource,
        }
    }
}

impl Operator for JoinProbe {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("probe", self.input.clone(), 2)],
            Some(OutputPortSpec::new("join", self.output.clone(), 3)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        inputs[0] = output.clone();
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        let Some(ResourceValue::KeyIndex(index)) = ctx.resource(self.resource) else {
            return Ok(WorkStatus::Made);
        };
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let mut out = Vec::new();
            for row in batch.into_rows() {
                let key = row.first().copied().flatten().unwrap_or_default();
                let payload = row.get(1).copied().flatten().unwrap_or_default();
                if let Some(row_ids) = index.get(&key) {
                    for row_id in row_ids {
                        out.push(vec![Some(key), Some(payload), Some(*row_id as i64)]);
                    }
                }
            };
            if !out.is_empty() {
                ctx.push(Batch::from_rows(0, out))?;
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct LazyLeftProject {
    label: String,
    input: Domain,
    output: Domain,
    left_payload: Vec<i64>,
    metrics: Arc<Mutex<ExecutionMetrics>>,
}

impl LazyLeftProject {
    fn new(
        label: impl Into<String>,
        input: Domain,
        output: Domain,
        left_payload: Vec<i64>,
        metrics: Arc<Mutex<ExecutionMetrics>>,
    ) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            left_payload,
            metrics,
        }
    }
}

impl Operator for LazyLeftProject {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("join", self.input.clone(), 3)],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        inputs[0] = output.clone();
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let mut out = Vec::new();
            for row in batch.into_rows() {
                let probe_payload = row.get(1).copied().flatten().unwrap_or_default();
                let raw_row_id = row.get(2).copied().flatten().unwrap_or_default();
                if let Ok(row_id) = usize::try_from(raw_row_id)
                    && let Some(left) = self.left_payload.get(row_id)
                {
                    self.metrics
                        .lock()
                        .add_lazy_materialized_rows("left_payload", 1);
                    out.push(vec![Some(*left + probe_payload)]);
                }
            };
            ctx.push(Batch::from_rows(0, out))?;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct AsyncRangeSource {
    label: String,
    domain: Domain,
    rows: Vec<Row>,
    range_rows: usize,
    delays: Vec<usize>,
}

struct AsyncRangeSourceState {
    in_flight: BTreeMap<u64, AsyncWorkId>,
    completed: Vec<u64>,
    cancelled: Vec<u64>,
    sealed: bool,
}

impl AsyncRangeSource {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        rows: Vec<Row>,
        range_rows: usize,
        delays: Vec<usize>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            rows,
            range_rows,
            delays,
        }
    }

    fn range_span(&self, start: u64) -> DomainSpan {
        let total = u64_from_usize(self.rows.len());
        let remaining = total.saturating_sub(start);
        DomainSpan::new(start, remaining.min(u64_from_usize(self.range_rows)))
    }

    fn range_starts(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.rows.len())
            .step_by(self.range_rows)
            .map(u64_from_usize)
    }

    fn range_delay(&self, start: u64) -> usize {
        let index = usize_from_u64(start) / self.range_rows;
        self.delays.get(index).copied().unwrap_or(1)
    }

    fn range_batch(&self, span: DomainSpan) -> Batch {
        let start = usize_from_u64(span.start());
        let end = usize_from_u64(span.end()).min(self.rows.len());
        Batch::from_lazy_rows(span.start(), self.rows[start..end].to_vec())
    }

    fn range_has_useful_requirement(requirement: &RequirementSet, span: DomainSpan) -> bool {
        (span.start()..span.end()).any(|row| {
            matches!(
                requirement.row(row),
                RowDemand::Needed | RowDemand::Candidate
            )
        })
    }

    fn range_is_not_needed(requirement: &RequirementSet, span: DomainSpan) -> bool {
        (span.start()..span.end())
            .all(|row| requirement.row(row) == RowDemand::NotNeeded)
    }
}

impl Operator for AsyncRangeSource {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = AsyncRangeSourceState;

    fn spec(&self) -> OperatorSpec {
        let columns = self.rows.first().map_or(1, Vec::len);
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new("out", self.domain.clone(), columns)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(AsyncRangeSourceState {
            in_flight: BTreeMap::new(),
            completed: Vec::new(),
            cancelled: Vec::new(),
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        // Honor the highest demand band present in the output
        // requirement. Range-based sources have multiple in-flight
        // positions, so we summarize across intervals instead of
        // reading a single cursor. See `source_band_workvalue`.
        let value = source_band_workvalue(&ctx.output_requirement());
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        let requirement = ctx.output_requirement();
        let mut made_progress = false;

        for (start, id) in state.in_flight.clone() {
            let span = self.range_span(start);
            if Self::range_is_not_needed(&requirement, span) && ctx.cancel_async(id) {
                state.in_flight.remove(&start);
                state.cancelled.push(start);
                made_progress = true;
            }
        }

        for (start, id) in state.in_flight.clone() {
            let Some(batch) = ctx.take_async(id) else {
                continue;
            };
            state.in_flight.remove(&start);
            state.completed.push(start);
            let span = batch.span();
            if Self::range_is_not_needed(&requirement, span) {
                made_progress = true;
                continue;
            };
            if !ctx.has_capacity() {
                return Ok(WorkStatus::Made);
            };
            ctx.push(batch)?;
            made_progress = true;
        }

        for start in self.range_starts() {
            if state.in_flight.contains_key(&start)
                || state.completed.contains(&start)
                || state.cancelled.contains(&start)
            {
                continue;
            };
            let span = self.range_span(start);
            if !Self::range_has_useful_requirement(&requirement, span) {
                continue;
            };
            let batch = self.range_batch(span);
            let id = ctx.spawn_fake_io(FakeIoRequest {
                label: self.label.clone(),
                span,
                delay_turns: self.range_delay(start),
                bytes: batch.estimated_bytes(),
                batch,
            })?;
            state.in_flight.insert(start, id);
            made_progress = true;
        }

        let range_count = self.range_starts().count();
        if state.completed.len().saturating_add(state.cancelled.len()) == range_count
            && state.in_flight.is_empty()
        {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }

        if made_progress {
            return Ok(WorkStatus::Made);
        }
        Ok(WorkStatus::Made)
    }
}

struct RangeSumSink {
    label: String,
    domain: Domain,
    start: u64,
    end: u64,
    rows: Arc<Mutex<Vec<Row>>>,
}

struct RangeSumState {
    sum: i64,
    emitted: bool,
}

impl RangeSumSink {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        start: u64,
        end: u64,
        rows: Arc<Mutex<Vec<Row>>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            start,
            end,
            rows,
        }
    }
}

impl Operator for RangeSumSink {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = RangeSumState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],None,
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(RangeSumState {
            sum: 0,
            emitted: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let total = exact_len(&self.domain);
        let mut requirement = RequirementSet::default();
        requirement.require_span(DomainSpan::new(self.start, self.end - self.start));
        if self.start > 0 {
            requirement.not_needed_span(DomainSpan::new(0, self.start));
        };
        if total > self.end {
            requirement.not_needed_span(DomainSpan::new(self.end, total - self.end));
        };
        inputs[0] = requirement;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let start = batch.span().start();
            for (offset, row) in batch.into_rows().into_iter().enumerate() {
                let ordinal = start + u64_from_usize(offset);
                if (self.start..self.end).contains(&ordinal) {
                    state.sum += row.first().copied().flatten().unwrap_or_default();
                }
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !state.emitted {
            self.rows.lock().push(vec![Some(state.sum)]);
            state.emitted = true;
            return Ok(WorkStatus::Finished);
        };
        if state.emitted {
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct DelayedScalarResource {
    label: String,
    resource: &'static str,
    value: i64,
    delay_steps: usize,
}

impl DelayedScalarResource {
    fn new(
        label: impl Into<String>,
        resource: &'static str,
        value: i64,
        delay_steps: usize,
    ) -> Self {
        Self {
            label: label.into(),
            resource,
            value,
            delay_steps,
        }
    }
}

impl Operator for DelayedScalarResource {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = usize;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(self.label.clone(), Vec::new(), None)
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(0)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if *state < self.delay_steps {
            *state += 1;
            return Ok(WorkStatus::Made);
        };
        ctx.publish_resource(self.resource, ResourceValue::Scalar(self.value))?;
        Ok(WorkStatus::Finished)
    }
}

struct LateSuffixFilter {
    label: String,
    input: Domain,
    output: Domain,
    resource: &'static str,
}

struct LateSuffixFilterState {
    emitted: u64,
    sealed: bool,
}

impl LateSuffixFilter {
    fn new(
        label: impl Into<String>,
        input: Domain,
        output: Domain,
        resource: &'static str,
    ) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            resource,
        }
    }
}

impl Operator for LateSuffixFilter {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = LateSuffixFilterState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("probe", self.input.clone(), 1)],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(LateSuffixFilterState {
            emitted: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let total = exact_len(&self.input);
        let mut input = RequirementSet::default();
        let Some(ResourceValue::Scalar(suffix)) = ctx.resource(self.resource) else {
            input.candidate_span(DomainSpan::new(0, total));
            inputs[0] = input;
            return Ok(());
        };
        let suffix = u64::try_from(suffix).unwrap_or_default().min(total);
        input.require_span(DomainSpan::new(0, suffix));
        if total > suffix {
            input.not_needed_span(DomainSpan::new(suffix, total - suffix));
        };
        inputs[0] = input;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        let Some(ResourceValue::Scalar(suffix)) = ctx.resource(self.resource) else {
            return Ok(WorkStatus::Made);
        };
        let suffix = u64::try_from(suffix).unwrap_or_default();
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let start = batch.span().start();
            let mut out = Vec::new();
            for (offset, row) in batch.into_rows().into_iter().enumerate() {
                let ordinal = start + u64_from_usize(offset);
                if ordinal < suffix && ordinal.is_multiple_of(2) {
                    out.push(row);
                }
            };
            if !out.is_empty() {
                let start = state.emitted;
                state.emitted += u64_from_usize(out.len());
                ctx.push(Batch::from_rows(start, out),
                )?;
            };
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !state.sealed {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

struct LazyVortexScan {
    label: String,
    domain: Domain,
    columns: Vec<Vec<Cell>>,
    metrics: Arc<Mutex<ExecutionMetrics>>,
}

impl LazyVortexScan {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        columns: Vec<Vec<Cell>>,
        metrics: Arc<Mutex<ExecutionMetrics>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            columns,
            metrics,
        }
    }
}

impl Operator for LazyVortexScan {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new(
                "out",
                self.domain.clone(),
                self.columns.len(),
            )),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        // Honor the highest demand band present in the output
        // requirement. LazyVortexScan emits a single facade batch
        // covering its entire domain, so we summarize across
        // intervals rather than reading a per-row cursor.
        let value = source_band_workvalue(&ctx.output_requirement());
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if *state {
            return Ok(WorkStatus::Finished);
        };
        let requirement = ctx.output_requirement();
        if requirement.is_empty() {
            return Ok(WorkStatus::Made);
        };
        let batch = Batch::from_lazy_columns(0, self.columns.clone());
        self.metrics.lock().add_lazy_batch_emitted(&self.label);
        ctx.trace("lazy vortex batch facade emitted encoded columns");
        ctx.push(batch)?;
        ctx.seal()?;
        *state = true;
        Ok(WorkStatus::Finished)
    }
}

struct TakeLazyColumn {
    label: String,
    input: Domain,
    output: Domain,
    column: usize,
    source_label: &'static str,
    metrics: Arc<Mutex<ExecutionMetrics>>,
}

impl TakeLazyColumn {
    fn new(
        label: impl Into<String>,
        input: Domain,
        output: Domain,
        column: usize,
        source_label: &'static str,
        metrics: Arc<Mutex<ExecutionMetrics>>,
    ) -> Self {
        Self {
            label: label.into(),
            input,
            output,
            column,
            source_label,
            metrics,
        }
    }
}

impl Operator for TakeLazyColumn {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new(
                "in",
                self.input.clone(),
                self.column.saturating_add(1),
            )],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let output = output.clone();
        let mut input = RequirementSet::default();
        for iv in output.intervals() {
            let span = DomainSpan::new(iv.start, iv.end - iv.start);
            match iv.demand {
                RowDemand::Needed => input.require_span(span),
                RowDemand::Candidate => input.candidate_span(span),
                RowDemand::NotNeeded => input.not_needed_span(span),
                RowDemand::Unknown => {}
            }
        };
        inputs[0] = input;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let values = batch.column_values(self.column);
            self.metrics.lock().add_lazy_column_materialized(
                self.source_label,
                self.column,
                values.len(),
            );
            let rows = values.into_iter().map(|value| vec![value]).collect();
            ctx.trace("lazy vortex batch facade materialized projected column");
            ctx.push(Batch::from_rows(batch.span().start(), rows),
            )?;
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) && !*state {
            *state = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

/// Pre-sorted merge join operator.
///
/// Inputs are key-ordered streams; the operator emits one output row per
/// matched (left key, right key) pair, producing the right-side payload value.
/// The witness is implicit in the consumed cursor positions: requirement
/// translation maps a downstream `NotNeeded` suffix back to `NotNeeded` for
/// any left or right rows whose keys are strictly larger than the largest
/// emitted key.
struct MergeJoin {
    label: String,
    left: Domain,
    right: Domain,
    output: Domain,
}

struct MergeJoinState {
    emitted: u64,
    last_emitted_left: Option<u64>,
    last_emitted_right: Option<u64>,
    sealed: bool,
}

impl MergeJoin {
    fn new(label: impl Into<String>, left: Domain, right: Domain, output: Domain) -> Self {
        Self {
            label: label.into(),
            left,
            right,
            output,
        }
    }
}

impl Operator for MergeJoin {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = MergeJoinState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![
                InputPortSpec::new("left", self.left.clone(), 2),
                InputPortSpec::new("right", self.right.clone(), 2),
            ],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(MergeJoinState {
            emitted: 0,
            last_emitted_left: None,
            last_emitted_right: None,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let output = output.clone();
        let left_total = exact_len(&self.left);
        let right_total = exact_len(&self.right);
        let mut left = RequirementSet::default();
        let mut right = RequirementSet::default();

        // Conservative requirement: until the join's witness proves that no
        // more matches can be emitted, every input row may still contribute.
        // The witness is maintained by step(), which advances cursors as
        // matches commit; once downstream signals NotNeeded AND the join has
        // consumed input past the relevant rows, the suffix can be released.
        left.require_span(DomainSpan::new(0, left_total));
        right.require_span(DomainSpan::new(0, right_total));

        // If downstream marks any output rows NotNeeded, propagate
        // NotNeeded only for rows strictly beyond the consumed cursor on
        // each side. This is the proof side of the witness translation.
        let downstream_not_needed = output
            .intervals()
            .iter()
            .any(|iv| iv.demand == RowDemand::NotNeeded);
        if downstream_not_needed {
            if let Some(left_cursor) = state.last_emitted_left {
                let release_start = left_cursor + 1;
                if release_start < left_total {
                    left.not_needed_span(DomainSpan::new(
                        release_start,
                        left_total - release_start,
                    ));
                }
            };
            if let Some(right_cursor) = state.last_emitted_right {
                let release_start = right_cursor + 1;
                if release_start < right_total {
                    right.not_needed_span(DomainSpan::new(
                        release_start,
                        right_total - release_start,
                    ));
                }
            }
        }

        inputs[0] = left;
        inputs[1] = right;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        let left_finished = ctx.input_finished(InputPortId::from_index(0));
        let right_finished = ctx.input_finished(InputPortId::from_index(1));
        let left_peek = ctx.peek(InputPortId::from_index(0));
        let right_peek = ctx.peek(InputPortId::from_index(1));
        match (left_peek, right_peek) {
            (Some(left_batch), Some(right_batch)) => {
                let left_key = left_batch
                    .first_column_values()
                    .first()
                    .copied()
                    .unwrap_or_default();
                let right_key = right_batch
                    .first_column_values()
                    .first()
                    .copied()
                    .unwrap_or_default();
                let left_span = left_batch.span();
                let right_span = right_batch.span();
                match left_key.cmp(&right_key) {
                    std::cmp::Ordering::Less => {
                        ctx.pop(InputPortId::from_index(0));
                        Ok(WorkStatus::Made)
                    }
                    std::cmp::Ordering::Greater => {
                        ctx.pop(InputPortId::from_index(1));
                        Ok(WorkStatus::Made)
                    }
                    std::cmp::Ordering::Equal => {
                        if !ctx.has_capacity() {
                            return Ok(WorkStatus::Made);
                        };
                        let right_rows = right_batch.to_rows();
                        let right_value = right_rows
                            .first()
                            .and_then(|row| row.get(1).copied().flatten())
                            .unwrap_or_default();
                        let pair = vec![vec![Some(left_key), Some(right_value)]];
                        ctx.push(Batch::from_rows(state.emitted, pair),
                        )?;
                        state.emitted += 1;
                        state.last_emitted_left = Some(left_span.start());
                        state.last_emitted_right = Some(right_span.start());
                        ctx.pop(InputPortId::from_index(1));
                        Ok(WorkStatus::Made)
                    }
                }
            }
            _ => {
                if left_finished || right_finished {
                    state.sealed = true;
                    ctx.seal()?;
                    return Ok(WorkStatus::Finished);
                }
                Ok(WorkStatus::Made)
            }
        }
    }
}


/// `ListOffsetsSource` produces contiguous parent-to-child offset rows.
///
/// One output row per parent. Each row has two columns: `start` and `end`,
/// matching the `OffsetWitness` payload shape. The operator preserves the
/// parent `Domain` and is row-preserving over its declared output domain.
struct ListOffsetsSource {
    label: String,
    parent: Domain,
    /// Cumulative offsets, length = parent_count + 1. Offset[i] is the
    /// child start for parent i; offset[i+1] is the child end.
    offsets: Vec<i64>,
}

struct ListOffsetsSourceState {
    cursor: u64,
    sealed: bool,
}

impl ListOffsetsSource {
    fn new(label: impl Into<String>, parent: Domain, offsets: Vec<i64>) -> Self {
        Self {
            label: label.into(),
            parent,
            offsets,
        }
    }
}

impl Operator for ListOffsetsSource {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = ListOffsetsSourceState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new("offsets", self.parent.clone(), 2)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(ListOffsetsSourceState {
            cursor: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        // Honor the row-demand band for our next cursor position. The
        // scheduler relies on `WorkValue` distinguishing Required vs
        // Candidate to give witness-producers a 10^9x priority edge
        // over speculative source reads. See spec: "WorkValue must
        // reflect demand band of next row."
        let demand = ctx.output_requirement().row(state.cursor);
        let value = match demand {
            // NotNeeded rows: propose `empty()` rather than skipping
            // entirely. The source still needs at least one scheduling
            // slot to advance its cursor through NotNeeded rows and
            // seal; `empty()` makes that slot the lowest-priority
            // option so witness-producing work always wins. (Strictly
            // skipping the proposal here would leave the source
            // unsealed and quiesce the graph; see fix report.)
            RowDemand::NotNeeded => WorkValue::empty(),
            RowDemand::Needed => WorkValue::required(1),
            RowDemand::Candidate => WorkValue::candidate(1, 128),
            // Treat `Unknown` as required — back-prop hasn't reached
            // us yet, so default to running so propagation can catch up.
            RowDemand::Unknown => WorkValue::required(1),
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        let parent_count = exact_len(&self.parent);
        if state.cursor >= parent_count {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        };
        let requirement = ctx.output_requirement();
        // Bulk-skip a contiguous NotNeeded span at the cursor in one
        // call. Per-row skipping returns WorkStatus::Made without
        // push/pop/seal and so doesn't flip the scheduler's progress
        // flag; a long NotNeeded run would quiesce the graph.
        while state.cursor < parent_count
            && matches!(requirement.row(state.cursor), RowDemand::NotNeeded)
        {
            state.cursor += 1;
        }
        if state.cursor >= parent_count {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        };
        let row_req = requirement.row(state.cursor);
        let presence = row_req;
        if matches!(presence, RowDemand::Unknown) {
            return Ok(WorkStatus::Made);
        };
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        };
        let i = usize_from_u64(state.cursor);
        let start = self.offsets.get(i).copied().unwrap_or_default();
        let end = self.offsets.get(i + 1).copied().unwrap_or_default();
        let row = vec![vec![Some(start), Some(end)]];
        ctx.push(Batch::from_rows(state.cursor, row),
        )?;
        state.cursor += 1;
        Ok(WorkStatus::Made)
    }
}

/// `ParentChildMin` aggregates `min(child)` per parent through
/// contiguous parent-to-child offset evidence.
///
/// Inputs:
///   port 0: offset stream over D_parent, columns (start, end).
///   port 1: child value stream over D_child, column (value).
/// Output:
///   D_parent, column min(child).
///
/// Requirement translation: a `Required` parent row maps to `Required`
/// child range `[start, end)` once the corresponding offset row has been
/// observed. A `NotNeeded` parent row maps to `NotNeeded` child range.
/// Until the offset is known for a parent ordinal, the operator publishes
/// `Candidate` for the child range.
struct ParentChildMin {
    label: String,
    parent: Domain,
    child: Domain,
    output: Domain,
}

#[derive(Default)]
struct ParentChildMinState {
    offsets: Vec<(i64, i64)>,
    /// Position in the child input stream we have already consumed up to.
    child_consumed: u64,
    /// Running per-parent minimum.
    running_min: Vec<Option<i64>>,
    /// Per-parent emitted bit.
    emitted: Vec<bool>,
    sealed: bool,
}

impl ParentChildMin {
    fn new(label: impl Into<String>, parent: Domain, child: Domain, output: Domain) -> Self {
        Self {
            label: label.into(),
            parent,
            child,
            output,
        }
    }
}

impl Operator for ParentChildMin {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = ParentChildMinState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![
                InputPortSpec::new("offsets", self.parent.clone(), 2),
                InputPortSpec::new("values", self.child.clone(), 1),
            ],
            Some(OutputPortSpec::new("out", self.output.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(ParentChildMinState::default())
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let output = output.clone();
        let parent_total = exact_len(&self.parent);
        let child_total = exact_len(&self.child);

        let mut offset_req = RequirementSet::default();
        let mut value_req = RequirementSet::default();

        // The offset stream needs to cover any parent that may still emit a
        // group output. Translate the output requirement directly.
        for iv in output.intervals() {
            let span = DomainSpan::new(iv.start, iv.end - iv.start);
            match iv.demand {
                RowDemand::Needed => offset_req.require_span(span),
                RowDemand::Candidate => offset_req.candidate_span(span),
                RowDemand::NotNeeded => offset_req.not_needed_span(span),
                RowDemand::Unknown => {}
            }
        }

        // For each parent ordinal we already have offsets for, translate
        // its presence into the matching child span.
        for (parent_idx, (start, end)) in state.offsets.iter().enumerate() {
            let parent_row = u64_from_usize(parent_idx);
            let span_start = u64::try_from(*start).unwrap_or_default();
            let span_end = u64::try_from(*end).unwrap_or_default();
            let span_len = span_end.saturating_sub(span_start);
            if span_len == 0 {
                continue;
            };
            let span = DomainSpan::new(span_start, span_len);
            let presence = output.row(parent_row);
            match presence {
                RowDemand::Needed => value_req.require_span(span),
                RowDemand::Candidate => value_req.candidate_span(span),
                RowDemand::NotNeeded => value_req.not_needed_span(span),
                RowDemand::Unknown => value_req.candidate_span(span),
            }
        }
        // Decide what to publish for child rows past the last offset we
        // have observed. If every parent past the last known one is
        // NotNeeded by downstream, those child rows are NotNeeded too.
        // Otherwise publish Candidate so the source can run ahead.
        let known_end = state
            .offsets
            .last()
            .map(|(_, end)| u64::try_from(*end).unwrap_or_default())
            .unwrap_or(0);
        if known_end < child_total {
            let known_parent_count = u64_from_usize(state.offsets.len());
            let remaining_parents_all_not_needed = (known_parent_count..parent_total)
                .all(|parent_row| {
                    matches!(
                        output.row(parent_row),
                        RowDemand::NotNeeded
                    )
                })
                && parent_total > known_parent_count;
            let span = DomainSpan::new(known_end, child_total - known_end);
            if remaining_parents_all_not_needed {
                value_req.not_needed_span(span);
            } else {
                value_req.candidate_span(span);
            }
        }

        inputs[0] = offset_req;
        inputs[1] = value_req;
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        // Generic update for the prototype. Consumers (operators with
        // inputs that have data or are finished) propose `Emit` so
        // they win class priority over producers. Sources without
        // input drain pressure propose `Cpu`. Operators without
        // inputs at all (true sources) propose `Cpu` regardless.
        let num_inputs = ctx.input_count();
        let mut input_drainable = false;
        for i in 0..num_inputs {
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_some() || ctx.input_finished(port) {
                input_drainable = true;
                break;
            }
        };
        let class = if input_drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState, _work: WorkKey, ctx: &mut WorkCtx<'_>) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        };
        let parent_total = exact_len(&self.parent);
        // Drain any available offset batches.
        let mut offsets_changed = false;
        while let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            for row in batch.into_rows() {
                let start = row.first().copied().flatten().unwrap_or_default();
                let end = row.get(1).copied().flatten().unwrap_or_default();
                state.offsets.push((start, end));
                state.running_min.push(None);
                state.emitted.push(false);
            };
            offsets_changed = true;
        }
        // Drain available child batches; route each row to the matching
        // parent's running minimum, or release NotNeeded rows.
        let mut consumed_any = false;
        let value_input_requirement =
            ctx.input_requirement(InputPortId::from_index(1));
        while let Some(batch) = ctx.peek(InputPortId::from_index(1)) {
            let span = batch.span();
            // If the value row is marked NotNeeded by our own translation,
            // drain it without aggregating.
            if matches!(
                value_input_requirement.row(span.start()),
                RowDemand::NotNeeded
            ) {
                ctx.pop(InputPortId::from_index(1));
                consumed_any = true;
                continue;
            };
            // Ensure the batch's first row has a known parent group; if
            // not, wait for more offset rows before consuming.
            let first_parent_idx = state.offsets.iter().position(|(start, end)| {
                let s = u64::try_from(*start).unwrap_or_default();
                let e = u64::try_from(*end).unwrap_or_default();
                span.start() >= s && span.start() < e
            });
            if first_parent_idx.is_none() {
                // No matching offset yet; wait for more offset rows.
                break;
            };
            let values = batch.first_column_values();
            ctx.pop(InputPortId::from_index(1));
            consumed_any = true;
            // Map each row in the batch to its parent group via the
            // absolute row position; a batch can straddle a group
            // boundary when child batch sizes exceed group sizes.
            for (i, value) in values.iter().copied().enumerate() {
                let row_pos = span.start() + u64_from_usize(i);
                let parent_idx = state.offsets.iter().position(|(start, end)| {
                    let s = u64::try_from(*start).unwrap_or_default();
                    let e = u64::try_from(*end).unwrap_or_default();
                    row_pos >= s && row_pos < e
                });
                let Some(parent_index) = parent_idx else {
                    // Should not happen for rows past the first because
                    // offsets are contiguous, but stay safe.
                    state.child_consumed += 1;
                    continue;
                };
                state.child_consumed += 1;
                let entry = &mut state.running_min[parent_index];
                *entry = Some(match *entry {
                    Some(current) => current.min(value),
                    None => value,
                });
            };
        }
        // Emit any closed groups whose downstream demand is non-NotNeeded.
        for parent_index in 0..state.offsets.len() {
            if state.emitted[parent_index] {
                continue;
            };
            let parent_row = u64_from_usize(parent_index);
            let parent_presence = ctx
                .output_requirement()
                .row(parent_row)
                ;
            let group_end = u64::try_from(state.offsets[parent_index].1).unwrap_or_default();
            let group_done = state.child_consumed >= group_end;
            if !group_done {
                break;
            };
            if parent_presence == RowDemand::NotNeeded {
                state.emitted[parent_index] = true;
                continue;
            };
            if !ctx.has_capacity() {
                return Ok(WorkStatus::Made);
            };
            let value = state.running_min[parent_index];
            let out = vec![vec![value]];
            ctx.trace(format!(
                "parent_child_min emitted parent {parent_row}"
            ));
            ctx.push(Batch::from_rows(parent_row, out),
            )?;
            state.emitted[parent_index] = true;
        };
        let offsets_done = ctx.input_finished(InputPortId::from_index(0));
        let values_done = ctx.input_finished(InputPortId::from_index(1));
        let all_emitted = state.emitted.iter().all(|emitted| *emitted)
            && state.offsets.len() == usize_from_u64(parent_total);
        if (offsets_done && values_done) || all_emitted {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        };
        if offsets_changed || consumed_any {
            return Ok(WorkStatus::Made);
        }
        Ok(WorkStatus::Made)
    }
}

/// `BrokeredSource` registers each declared range as a broker
/// interest, then emits the completion batches when the broker
/// delivers them. Demonstrates concurrent broker submission with CPU
/// emission within one scheduler turn.
struct BrokeredSource {
    label: String,
    domain: Domain,
    broker: BrokerId,
    ranges: Vec<(DomainSpan, Batch)>,
    delay_turns: usize,
    latency_class: LatencyClass,
}

#[derive(Default)]
struct BrokeredSourceState {
    /// Range index → registered InterestId. Set when registered.
    interests: Vec<Option<InterestId>>,
    /// Range index → completed batch. Set when broker_take returns.
    completed: Vec<Option<Batch>>,
    /// Next range index to push downstream.
    next_emit: usize,
    sealed: bool,
}

impl BrokeredSource {
    fn new(
        label: impl Into<String>,
        domain: Domain,
        broker: BrokerId,
        ranges: Vec<(DomainSpan, Batch)>,
        delay_turns: usize,
        latency_class: LatencyClass,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            broker,
            ranges,
            delay_turns,
            latency_class,
        }
    }
}

impl Operator for BrokeredSource {
    fn propagation_depends_on_state(&self) -> bool { true }

    type GlobalState = ();
    type LocalState = BrokeredSourceState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new("out", self.domain.clone(), 1)),
        )
    }

    fn init_global(

        &self,

        _ctx: &mut crate::GlobalInitCtx<'_>,

    ) -> EngineResult<Self::GlobalState> {

        Ok(())

    }


    fn init_local(

        &self,

        _global: &Self::GlobalState,

        _ctx: &mut crate::LocalInitCtx<'_>,

    ) -> EngineResult<Self::LocalState> {
        Ok(BrokeredSourceState {
            interests: vec![None; self.ranges.len()],
            completed: vec![None; self.ranges.len()],
            next_emit: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    )  -> EngineResult<()> {
        if state.sealed {
            return Ok(());
        }
        // Register interests for ranges we haven't registered yet.
        for (idx, (span, batch)) in self.ranges.iter().enumerate() {
            if state.interests[idx].is_none() {
                let interest = ctx.broker_register(
                    self.broker,
                    InterestSpec {
                        label: format!("{}_range_{}", self.label, idx),
                        span: *span,
                        batch: batch.clone(),
                        bytes: batch.estimated_bytes(),
                        delay_turns: self.delay_turns,
                        row_class: RowClass::Required,
                        p_needed_x256: 256,
                        latency_class: self.latency_class,
                        unblock_rows: span.len(),
                    },
                );
                state.interests[idx] = Some(interest);
            }
        }
        // Absorb any completed broker results.
        while let Some(completion) = ctx.broker_take(self.broker) {
            // Find which range this corresponds to.
            for idx in 0..state.interests.len() {
                if state.interests[idx] == Some(completion.interest) {
                    state.completed[idx] = Some(completion.batch.clone());
                    break;
                }
            }
        }
        // Propose emit if the next range in order is completed.
        if state.next_emit < self.ranges.len()
            && state.completed[state.next_emit].is_some()
        {
            ctx.propose(WorkProposal::new(
                WorkKey::from_byte(1),
                WorkClass::Emit,
                WorkValue::required(self.ranges[state.next_emit].0.len()),
                WorkCost::small_emit(64),
                WorkConstraints::output_capacity(),
            ));
        } else if state.next_emit >= self.ranges.len() {
            ctx.propose(WorkProposal::new(
                WorkKey::from_byte(2),
                WorkClass::Seal,
                WorkValue::empty(),
                WorkCost::small_cpu(),
                WorkConstraints::none(),
            ));
        }
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        match work.tag() {
            1 => {
                if let Some(batch) = state.completed[state.next_emit].take() {
                    ctx.push(batch)?;
                    state.next_emit += 1;
                }
                Ok(WorkStatus::Made)
            }
            2 => {
                state.sealed = true;
                ctx.seal()?;
                Ok(WorkStatus::Finished)
            }
            _ => Ok(WorkStatus::Made),
        }
    }
}
