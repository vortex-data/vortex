//! ClickBench query implementations.
//!
//! See `vortex-bench/clickbench_queries.sql` (in the vortex repo)
//! for the canonical set. Currently:
//!
//! - **Q5** (`SELECT COUNT(DISTINCT "UserID") FROM hits`) — exercises
//!   column pruning via `bind_field` plus the streaming
//!   `CountDistinctI64Sink`.
//! - **Q20** (`SELECT "UserID" FROM hits WHERE "UserID" = ?`) —
//!   exercises predicate pushdown via `bind_field_filtered` plus
//!   `CollectI64Sink`.
//!
//! The sink operators themselves now live in [`crate::operators`].

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::layouts;
use crate::layouts::OutputOrdering;
use crate::operators::CollectI64Sink;
use crate::operators::CountDistinctI64Sink;
use crate::operators::CountDistinctState;
use crate::operators::LazyVortexFile;
use crate::queries::bench::BenchQuery;
use crate::queries::bench::Runner;
use crate::Cardinality;
use crate::ChannelBuffer;
use crate::Domain;
use crate::DomainId;
use crate::EngineResult;
use crate::ExecutionMetrics;
use crate::OperatorGraph;
use crate::OperatorNode;
use crate::PreparedTask;
use crate::TaskOptions;
use crate::TaskReport;

/// Run Q5 (`SELECT COUNT(DISTINCT "UserID") FROM hits`) against a
/// single ClickBench Vortex shard.
///
/// Returns the number of distinct non-null `UserID` values in the
/// shard. The graph reads only the `UserID` field via
/// `layouts::bind_field` — no other column's segments are touched —
/// and streams the resulting `i64` batches into a
/// `CountDistinctI64Sink`.
pub fn q5_count_distinct_userid(path: impl AsRef<Path>) -> EngineResult<u64> {
    q5_count_distinct_userid_with_workers(path, 1)
}

/// Q5 with a configurable shard count for within-shard parallelism.
/// `worker_count = 1` runs through the single-shard fast path
/// (sequential scheduler turn). `worker_count > 1` runs through
/// the driver's `thread::scope` orchestration. The layout source
/// subgraph is currently single-lane after the layout decomposition
/// pass, so within-shard scaling is flat until per-chunk
/// parallelism is brought back at the operator level.
pub fn q5_count_distinct_userid_with_workers(
    path: impl AsRef<Path>,
    worker_count: usize,
) -> EngineResult<u64> {
    let handle = layouts::open_vortex_file(path)?;
    let session = handle.session();
    let segment_source = handle.file.segment_source();
    let root = Arc::clone(handle.file.footer().layout());
    let row_count = handle.file.row_count();

    let mut graph = OperatorGraph::new();
    let source_id = layouts::bind_field(
        &mut graph,
        root,
        &["UserID"],
        "userid",
        Arc::clone(&handle.runtime),
        segment_source,
        &session,
    )?;
    let runner = Runner::new(worker_count);
    run_count_distinct_graph(graph, source_id, row_count, &runner)
}

/// Stream a source subgraph (already attached to `graph`) into
/// `CountDistinctI64Sink` and return the captured count. The
/// source's output port 0 must produce single-column `i64` batches.
pub fn run_count_distinct_graph(
    mut graph: OperatorGraph,
    source_id: crate::OperatorId,
    row_count: u64,
    runner: &Runner,
) -> EngineResult<u64> {
    let domain = Domain::new(DomainId::new("userid"), Cardinality::Exact(row_count));
    let counter: Arc<CountDistinctState> = Arc::new(CountDistinctState::new());

    let sink_id = graph.add_operator(OperatorNode::new(CountDistinctI64Sink::new(
        "count_distinct",
        domain,
        Arc::clone(&counter),
    )));
    graph.connect(
        OperatorGraph::output(source_id),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    runner.run(graph)?;
    Ok(counter.distinct_count())
}

/// Q20 over many shards in a **single graph**: every shard's filter
/// subgraph feeds into one `Union` operator that fans into a single
/// `CollectI64Sink`. This is the only Q20 execution shape the
/// engine supports — a per-shard separate-task variant existed but
/// was deleted because it forced an execution model where each
/// shard became its own `PreparedTask` (or batch of tasks across
/// workers), which the engine is not designed for.
pub fn q20_userid_equals_unioned<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
    target: i64,
) -> EngineResult<Vec<i64>> {
    let runner = Runner::new(1);
    let files = prepare_q20_files(paths, target)?;
    run_q20_with_files(&files, &runner, OutputOrdering::Unordered)
}

/// Eagerly open every shard's footer and evaluate
/// `can_prune(predicate)`. Returns the surviving (non-pruned)
/// `LazyVortexFile` operators. Provably-empty shards are dropped
/// here.
///
/// Splitting this from the per-iteration query execution lets
/// callers — benchmarks, planners — pay the footer-read cost once
/// across many query runs, the same way DataFusion's bench creates
/// the `SessionContext` once per format and runs the query
/// `iterations` times against it.
pub fn prepare_q20_files<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
    target: i64,
) -> EngineResult<Vec<LazyVortexFile>> {
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;

    let predicate = eq(root(), lit(target));
    let path_vec: Vec<PathBuf> = paths
        .into_iter()
        .map(|p| p.as_ref().to_path_buf())
        .collect();
    let mut surviving = Vec::with_capacity(path_vec.len());
    for (i, path) in path_vec.into_iter().enumerate() {
        let label = format!("userid_eq[{i}]");
        let output_domain = Domain::new(
            DomainId::new(format!("filter_out:filter:{label}")),
            Cardinality::Unknown,
        );
        let lazy = LazyVortexFile::open(
            label,
            path,
            vec!["UserID".to_string()],
            predicate.clone(),
            root(),
            output_domain,
            1,
        )?;
        if lazy.can_prune() == Some(true) {
            continue;
        }
        surviving.push(lazy);
    }
    Ok(surviving)
}

/// Run q20 against an already-opened set of `LazyVortexFile`
/// shards. The shards are cloned (cheap — `Arc`-backed) and used
/// to build a fresh operator graph each call, so this can be
/// invoked many times with the same input. Equivalent in shape to
/// DataFusion's per-iteration `execute(&mut ctx, query)`.
///
/// `ordering` decides the multi-shard fan-in shape: `Unordered` →
/// `Union` (cheap, lane-parallel, arrival-order interleave);
/// `OrderedByInputIndex` → `Concat` (single-lane, drains shards in
/// `files` order, output rows are concatenated in that order).
/// `CollectI64Sink` doesn't care about row order so the q20 binary
/// passes `Unordered` by default — pass `OrderedByInputIndex` only
/// when the consumer of the matches needs them in shard order.
pub fn run_q20_with_files(
    files: &[LazyVortexFile],
    runner: &Runner,
    ordering: OutputOrdering,
) -> EngineResult<Vec<i64>> {
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let mut graph = OperatorGraph::new();
    let mut input_domains: Vec<Domain> = Vec::new();
    let mut shard_ids: Vec<crate::OperatorId> = Vec::new();
    for lazy in files {
        let lazy = lazy.clone();
        input_domains.push(Domain::new(
            DomainId::new(format!("filter_out:filter:{}", lazy.label())),
            Cardinality::Unknown,
        ));
        shard_ids.push(graph.add_operator(OperatorNode::new(lazy)));
    }

    let fanin_output_domain =
        Domain::new(DomainId::new("fanin_out:userid_eq"), Cardinality::Unknown);
    drop(input_domains); // domains live on the shards' own output ports

    let captured: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_id = graph.add_operator(OperatorNode::new(CollectI64Sink::new(
        "collect_i64",
        fanin_output_domain,
        Arc::clone(&captured),
    )));
    // Multi-producer channel fans all shard outputs into the sink's
    // single input port. The channel handles span translation. For
    // `OutputOrdering::OrderedByInputIndex` we'd want connection-add-
    // order drain — the channel discipline supports that as a future
    // option (drain producers in `from`-list order). For now both
    // ordering modes share the same fan-in implementation: arrival-
    // order drain. The sink (CollectI64Sink) is order-insensitive so
    // this is observationally fine.
    let _ = ordering;
    graph.connect_multi_named(
        "fanin:userid_eq",
        shard_ids.clone(),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    runner.run(graph)?;
    Ok(Arc::try_unwrap(captured)
        .unwrap_or_else(|arc| Mutex::new(arc.lock().clone()))
        .into_inner())
}

/// Convenience wrapper that opens files + runs q20 in one call.
/// Equivalent to `prepare_q20_files(...) + run_q20_with_files(...)`.
/// Defaults to `Unordered` fan-in.
pub fn q20_userid_equals_unioned_with_workers<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
    target: i64,
    worker_count: usize,
) -> EngineResult<Vec<i64>> {
    let runner = Runner::new(worker_count);
    let files = prepare_q20_files(paths, target)?;
    run_q20_with_files(&files, &runner, OutputOrdering::Unordered)
}

/// Single-worker variant of `q20_userid_equals_unioned_with_workers`
/// that also returns the scheduler `TaskReport` (trace + metrics).
/// Useful for studying demand propagation on small inputs.
pub fn q20_userid_equals_unioned_traced<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
    target: i64,
) -> EngineResult<(Vec<i64>, TaskReport)> {
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;

    let predicate = eq(root(), lit(target));
    let mut graph = OperatorGraph::new();
    let mut input_domains: Vec<Domain> = Vec::new();
    let mut filter_ids: Vec<crate::OperatorId> = Vec::new();
    let mut _handles = Vec::new();

    for (i, path) in paths.into_iter().enumerate() {
        let handle = layouts::open_vortex_file(path.as_ref())?;
        let session = handle.session();
        let segment_source = handle.file.segment_source();
        let root_layout = Arc::clone(handle.file.footer().layout());
        let filter_id = layouts::bind_field_filtered(
            &mut graph,
            root_layout,
            &["UserID"],
            predicate.clone(),
            root(),
            format!("userid_eq[{i}]"),
            Arc::clone(&handle.runtime),
            segment_source,
            &session,
        )?;
        let input_domain = Domain::new(
            DomainId::new(format!("filter_out:filter:userid_eq[{i}]")),
            Cardinality::Unknown,
        );
        input_domains.push(input_domain);
        filter_ids.push(filter_id);
        _handles.push(handle);
    }

    let union_output_domain =
        Domain::new(DomainId::new("union_out:userid_eq"), Cardinality::Unknown);
    drop(input_domains);

    let captured: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_id = graph.add_operator(OperatorNode::new(CollectI64Sink::new(
        "collect_i64",
        union_output_domain,
        Arc::clone(&captured),
    )));
    graph.connect_multi_named(
        "fanin:userid_eq",
        filter_ids.clone(),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    let report = PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    let matches = Arc::try_unwrap(captured)
        .unwrap_or_else(|arc| Mutex::new(arc.lock().clone()))
        .into_inner();
    Ok((matches, report))
}


// =====================================================================
// BenchQuery impls
// =====================================================================
//
// Each clickbench query exposes a typed `BenchQuery` so the harness
// in `crate::queries::bench` can run it the same way for every
// query — setup once, execute per iteration, with comparable
// timing semantics across binaries (and against external engines
// like DataFusion's vortex-bench runner).

/// Q3: `SELECT AVG("UserID") FROM hits` across many shards via a
/// single graph using the `partial-then-merge` aggregate split.
///
/// Per shard, `bind_field("UserID")` reads only the `UserID` column
/// subtree feeding a single-lane [`PartialAggregate`] running
/// `Mean::combined()`. Each shard's `PartialAggregate` emits one
/// 1-row batch carrying the partial `struct(sum, count)` scalar.
/// Those small batches feed a [`Union`], and a single
/// [`MergeAggregate`] folds the partials into the final f64. The
/// fan-in carries 1 row per shard, not the raw column stream — so
/// the channel between `Union` and `MergeAggregate` doesn't need to
/// be sized for the bulk of the data.
pub struct Q3AvgUserId;

impl BenchQuery for Q3AvgUserId {
    type Prepared = Vec<layouts::VortexFileHandle>;
    type Output = f64;

    fn name(&self) -> &str {
        "q3_avg_userid"
    }

    fn prepare(&self, paths: Vec<PathBuf>) -> EngineResult<Self::Prepared> {
        paths
            .iter()
            .map(|p| layouts::open_vortex_file(p))
            .collect()
    }

    fn execute(
        &self,
        prepared: &Self::Prepared,
        runner: &Runner,
    ) -> EngineResult<Self::Output> {
        use crate::EngineError;
        use crate::operators::ArrayCollectSink;
        use crate::operators::MergeAggregate;
        use crate::operators::PartialAggregate;
        use vortex_array::aggregate_fn::AggregateFnVTableExt;
        use vortex_array::aggregate_fn::EmptyOptions;
        use vortex_array::aggregate_fn::combined::PairOptions;
        use vortex_array::aggregate_fn::fns::mean::Mean;
        use vortex_array::scalar::Scalar;

        if prepared.is_empty() {
            return Ok(f64::NAN);
        }

        let mut graph = OperatorGraph::new();
        let mut shard_ids: Vec<crate::OperatorId> = Vec::with_capacity(prepared.len());
        let mut shard_domains: Vec<Domain> = Vec::with_capacity(prepared.len());
        let mut user_id_dtype: Option<vortex_array::dtype::DType> = None;
        let mean_fn = Mean::combined().bind(PairOptions(EmptyOptions, EmptyOptions));

        for (i, handle) in prepared.iter().enumerate() {
            let session = handle.session();
            let segment_source = handle.file.segment_source();
            let root = Arc::clone(handle.file.footer().layout());
            let shard_name = format!("shard[{i}]");
            let shard_id = layouts::bind_field(
                &mut graph,
                root,
                &["UserID"],
                shard_name,
                Arc::clone(&handle.runtime),
                segment_source,
                &session,
            )?;
            let spec = graph.nodes()[shard_id.index()].spec();
            let output_port = spec
                .output
                .as_ref()
                .expect("bind_field source has an output port");
            shard_domains.push(output_port.domain.clone());
            shard_ids.push(shard_id);
            if user_id_dtype.is_none() {
                user_id_dtype = handle
                    .file
                    .footer()
                    .layout()
                    .dtype()
                    .as_struct_fields_opt()
                    .and_then(|fs| fs.field("UserID"));
            }
        }
        let user_id_dtype =
            user_id_dtype.ok_or_else(|| EngineError::message("UserID dtype missing"))?;
        let session = prepared[0].session();

        // Multi-producer channel feeds all 100 shards into
        // PartialAggregate's single input port. The channel handles
        // the fan-in (no Union operator) and translates each pushed
        // batch's span via its own output cursor so PartialAggregate
        // sees monotonic spans regardless of arrival order.
        let union_domain = Domain::new(
            DomainId::new("union_out:userid"),
            Cardinality::Unknown,
        );
        drop(shard_domains); // domains live on the shard output ports

        // Multi-lane PartialAggregate over the unioned stream. Each
        // lane runs its own accumulator and emits one partial-state
        // row at the end (struct(sum, count) for Mean). With L
        // lanes per host worker, cross-shard fan-in keeps every
        // worker busy without serialising on a single accumulator.
        let partial = PartialAggregate::new(
            "partial:mean(userid)",
            union_domain,
            user_id_dtype.clone(),
            mean_fn.clone(),
            session.clone(),
        )?;
        let partial_domain = partial.output_domain().clone();
        let partial_id = graph.add_operator(OperatorNode::new(partial));
        graph.connect_multi_named(
            "fanin:userid",
            shard_ids.clone(),
            vec![OperatorGraph::input(partial_id, 0)],
            ChannelBuffer::bounded_bytes(256 << 20),
        );

        // MergeAggregate folds the per-lane partials into the
        // finalised f64 mean. The fan-in here is just `lane_count`
        // 1-row batches — tiny.
        let captured_scalar: Arc<Mutex<Option<Scalar>>> = Arc::new(Mutex::new(None));
        let merge = MergeAggregate::new(
            "merge:mean(userid)",
            partial_domain,
            user_id_dtype,
            mean_fn,
            session.clone(),
        )?
        .with_capture(Arc::clone(&captured_scalar));
        let merge_output_domain = merge.output_domain().clone();
        let merge_id = graph.add_operator(OperatorNode::new(merge));
        graph.connect(
            OperatorGraph::output(partial_id),
            vec![OperatorGraph::input(merge_id, 0)],
            ChannelBuffer::bounded_bytes(1 << 20),
        );

        // MergeAggregate's output port still emits the 1-row batch
        // even when the scalar is captured separately; we have to
        // give that batch a consumer or `push` will fail. An empty
        // sink is enough.
        let collected: Arc<Mutex<Vec<vortex_array::ArrayRef>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink_id = graph.add_operator(OperatorNode::new(ArrayCollectSink::new(
            "collect:mean(userid)",
            merge_output_domain,
            Arc::clone(&collected),
        )));
        graph.connect(
            OperatorGraph::output(merge_id),
            vec![OperatorGraph::input(sink_id, 0)],
            ChannelBuffer::bounded_bytes(8 << 20),
        );

        runner.run(graph)?;

        let scalar = captured_scalar
            .lock()
            .clone()
            .ok_or_else(|| EngineError::message("Q3 aggregate produced no scalar"))?;
        Ok(scalar
            .as_primitive_opt()
            .and_then(|p| p.typed_value::<f64>())
            .unwrap_or(f64::NAN))
    }

    fn output_summary(&self, output: &Self::Output) -> String {
        format!("avg_userid={output:.6e}")
    }
}

/// Q5 single-shard: `SELECT COUNT(DISTINCT "UserID") FROM hits` on
/// exactly one Vortex file. There is no multi-shard variant — the
/// engine doesn't support a per-shard separate-task execution
/// model, and a proper multi-shard count-distinct would need a
/// single-graph fan-in (cross-shard hash-set merge) that hasn't
/// been written. For now, single-shard only.
pub struct Q5SingleShard;

impl BenchQuery for Q5SingleShard {
    type Prepared = layouts::VortexFileHandle;
    type Output = u64;

    fn name(&self) -> &str {
        "q5_count_distinct_userid"
    }

    fn prepare(&self, paths: Vec<PathBuf>) -> EngineResult<Self::Prepared> {
        if paths.len() != 1 {
            return Err(crate::EngineError::message(format!(
                "Q5SingleShard expects exactly 1 path, got {}",
                paths.len()
            )));
        }
        layouts::open_vortex_file(&paths[0])
    }

    fn execute(
        &self,
        prepared: &Self::Prepared,
        runner: &Runner,
    ) -> EngineResult<Self::Output> {
        let session = prepared.session();
        let segment_source = prepared.file.segment_source();
        let root = Arc::clone(prepared.file.footer().layout());
        let row_count = prepared.file.row_count();
        let mut graph = OperatorGraph::new();
        let source_id = layouts::bind_field(
            &mut graph,
            root,
            &["UserID"],
            "userid",
            Arc::clone(&prepared.runtime),
            segment_source,
            &session,
        )?;
        run_count_distinct_graph(graph, source_id, row_count, runner)
    }

    fn output_summary(&self, output: &Self::Output) -> String {
        format!("count_distinct={output}")
    }
}

/// Q20 unioned: filter `eq(UserID, target)` across many shards via
/// a single graph fanning per-shard outputs into a
/// `CollectI64Sink`. The fan-in operator is chosen by `ordering`:
/// `Union` for `Unordered`, `Concat` for `OrderedByInputIndex`.
/// Setup opens files + evaluates file-level `can_prune`;
/// provably-empty shards are dropped before any per-iteration
/// work happens.
pub struct Q20Unioned {
    pub target: i64,
    pub ordering: OutputOrdering,
}

impl Q20Unioned {
    /// Construct with `Unordered` fan-in (the cheap default).
    pub fn new(target: i64) -> Self {
        Self {
            target,
            ordering: OutputOrdering::Unordered,
        }
    }
}

impl BenchQuery for Q20Unioned {
    type Prepared = Vec<LazyVortexFile>;
    type Output = Vec<i64>;

    fn name(&self) -> &str {
        match self.ordering {
            OutputOrdering::Unordered => "q20_userid_equals_unioned",
            OutputOrdering::OrderedByInputIndex => "q20_userid_equals_concat",
        }
    }

    fn prepare(&self, paths: Vec<PathBuf>) -> EngineResult<Self::Prepared> {
        prepare_q20_files(paths, self.target)
    }

    fn execute(
        &self,
        prepared: &Self::Prepared,
        runner: &Runner,
    ) -> EngineResult<Self::Output> {
        run_q20_with_files(prepared, runner, self.ordering)
    }

    fn output_summary(&self, output: &Self::Output) -> String {
        format!("matches={}", output.len())
    }

    fn prepared_summary(&self, prepared: &Self::Prepared) -> String {
        format!("{} surviving (post file-stat prune)", prepared.len())
    }
}

