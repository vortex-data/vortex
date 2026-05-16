//! `LazyVortexFile`: a file-as-a-subroutine source operator.
//!
//! At bind time the engine knows only the file path, the filter
//! predicate, and the output port shape. No file-internal subgraph
//! exists yet. At runtime, on first `run`, the operator:
//!
//!   1. Opens the file's footer.
//!   2. Calls `VortexFile::can_prune(predicate)`. If the file's
//!      column statistics prove no row can match, the operator emits
//!      a seal and finishes — no segment reads, no inner graph
//!      built. This is the cheap path: ~footer-size I/O per file
//!      that gets pruned.
//!   3. Otherwise, builds an inner `OperatorGraph` containing the
//!      file's layout binding (via `bind_field_filtered`) wired to
//!      an `ArrayCollectSink`, prepares it, and runs the inner task
//!      to completion in-line. The captured arrays are then forwarded
//!      one batch per `run` call so the operator respects output
//!      capacity backpressure.
//!
//! ## Design notes
//!
//! - The inner task uses its own `FakeDriverIo` and is not visible to
//!   the outer task's scheduler. Cross-shard scheduling decisions
//!   (EV ranking across all 100 shards in q20) cannot reach inside.
//!   That's a deliberate trade-off for option (B): contained,
//!   no engine-graph mutation, minimal surface area. A full live-graph
//!   extension would expose the inner ops to the outer scheduler at
//!   the cost of a much larger engine surgery.
//!
//! - The operator is `Serial` parallelism. One file = one inner task.
//!   The outer scheduler dispatches the operator's single lane to a
//!   worker; that worker drives the inner task synchronously. With
//!   100 shards and 10 workers, ~10 file subtasks run concurrently.
//!
//! - The file footer is opened on first `run`, not on construction.
//!   This keeps the bind-time path cheap when the operator graph has
//!   many shards and we don't yet know which the scheduler will
//!   actually run (in principle a future scheduler could elect to
//!   skip late shards entirely).

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_file::FileStatistics;

use crate::layouts::VortexFileHandle;

use crate::Batch;
use crate::Cardinality;
use crate::ChannelBuffer;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineResult;
use crate::ExecutionMetrics;
use crate::GlobalInitCtx;
use crate::InputPortSpec;
use crate::LocalInitCtx;
use crate::Operator;
use crate::OperatorGraph;
use crate::OperatorNode;
use crate::OperatorSpec;
use crate::OutputPortSpec;
use crate::PreparedTask;
use crate::RequirementCtx;
use crate::RequirementSet;
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
use crate::layouts;
use crate::operators::ArrayCollectSink;

/// `Clone`-able so callers (benchmarks, planners) can open a file
/// once, pass references through, and rebuild the operator each
/// query iteration without re-doing the footer read. All fields
/// behind cheap clones — `handle` is an `Arc`, expressions are
/// `Arc`-backed, paths/labels are owned strings.
#[derive(Clone)]
pub struct LazyVortexFile {
    label: String,
    path: PathBuf,
    field_path: Vec<String>,
    predicate: Expression,
    projection: Expression,
    output_domain: Domain,
    output_columns: usize,
    /// File handle held from construction onwards. The footer is
    /// eagerly opened so an upstream logical planner can inspect
    /// `file_stats()` and `can_prune()` before scheduling — without
    /// re-opening the file on the runtime path. The handle is
    /// shared across construction/inspection/run so all paths agree
    /// on the same file state.
    handle: Arc<VortexFileHandle>,
    /// Result of `VortexFile::can_prune(predicate)` evaluated once
    /// at construction. `Some(true)` → file is provably empty for
    /// the predicate; `Some(false)` → cannot rule out matches;
    /// `None` → couldn't decide (e.g., stats missing). Cached so
    /// runtime doesn't re-evaluate.
    pre_pruned: Option<bool>,
}

pub struct LazyVortexFileState {
    /// `None` until the first `run` opens the file and decides
    /// whether to prune.
    captured: Option<Vec<ArrayRef>>,
    /// Index of the next captured array to emit downstream.
    next_chunk: usize,
    /// Cumulative output rows pushed; spans are issued consecutively
    /// starting at 0 since the output domain has unknown cardinality.
    output_cursor: u64,
    sealed: bool,
}

impl LazyVortexFile {
    /// Build a lazy file source. Opens the file footer immediately
    /// and evaluates `VortexFile::can_prune(predicate)` once so
    /// upstream callers (and a future logical planner) can inspect
    /// file-level statistics before the operator graph runs. The
    /// file's data subgraph is *not* bound here — that happens at
    /// runtime when (and if) `run` decides the file's data is
    /// needed.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        label: impl Into<String>,
        path: impl Into<PathBuf>,
        field_path: Vec<String>,
        predicate: Expression,
        projection: Expression,
        output_domain: Domain,
        output_columns: usize,
    ) -> EngineResult<Self> {
        let path = path.into();
        let handle = Arc::new(layouts::open_vortex_file(&path)?);
        let pre_pruned = handle.file.can_prune(&predicate).ok();
        Ok(Self {
            label: label.into(),
            path,
            field_path,
            predicate,
            projection,
            output_domain,
            output_columns,
            handle,
            pre_pruned,
        })
    }

    /// File-level statistics from the footer, exposed for upstream
    /// planner introspection. May be `None` for files written
    /// without statistics.
    pub fn file_stats(&self) -> Option<&FileStatistics> {
        self.handle.file.file_stats()
    }

    /// `Some(true)` iff this file is provably empty for the
    /// configured predicate (per `VortexFile::can_prune`). An
    /// upstream planner can drop the entire shard from the
    /// operator graph at construction time.
    pub fn can_prune(&self) -> Option<bool> {
        self.pre_pruned
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

impl Operator for LazyVortexFile {
    type GlobalState = ();
    type LocalState = LazyVortexFileState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            Vec::<InputPortSpec>::new(),
            Some(OutputPortSpec::new(
                "out",
                self.output_domain.clone(),
                self.output_columns,
            )),
        )
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(LazyVortexFileState {
            captured: None,
            next_chunk: 0,
            output_cursor: 0,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        if local.sealed {
            return Ok(());
        }
        // Two phases of work:
        //   * inner task hasn't run → one big Cpu chunk to open the
        //     file + decide prune + (optionally) execute the inner
        //     task.
        //   * inner task captured arrays → drain one batch per run
        //     call, gated on output capacity.
        let class = if local.captured.is_some() {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::candidate(0, 0),
            WorkCost::small_cpu(),
            WorkConstraints::output_capacity(),
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
        if local.sealed {
            return Ok(WorkStatus::Finished);
        }
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }
        // Phase 1: first call. Open the file, check pruning, run the
        // inner task if needed.
        if local.captured.is_none() {
            let captured = run_inner_file(self)?;
            local.captured = Some(captured);
        }
        // Phase 2: drain captured arrays into output batches one at
        // a time, respecting capacity.
        let captured = local
            .captured
            .as_mut()
            .expect("captured initialised in phase 1");
        if local.next_chunk >= captured.len() {
            local.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        let array = captured[local.next_chunk].clone();
        local.next_chunk += 1;
        let len = array.len() as u64;
        if len == 0 {
            return Ok(WorkStatus::Made);
        }
        let span = DomainSpan::new(local.output_cursor, len);
        local.output_cursor += len;
        ctx.push(Batch::from_array(span, array))?;
        Ok(WorkStatus::Made)
    }
}

/// Open the file, check pruning, and (if not pruned) run the inner
/// task to completion. Returns the captured arrays. An empty `Vec`
/// indicates the file was pruned at the footer level.
fn run_inner_file(op: &LazyVortexFile) -> EngineResult<Vec<ArrayRef>> {
    // The handle is opened at construction; reuse it. If the
    // construction-time `can_prune` decision was conclusive
    // (`Some(true)`), short-circuit without binding the inner graph.
    if op.pre_pruned == Some(true) {
        return Ok(Vec::new());
    }
    let handle = Arc::clone(&op.handle);
    let session = handle.session();
    let segment_source = handle.file.segment_source();
    let root_layout = Arc::clone(handle.file.footer().layout());
    let mut graph = OperatorGraph::new();
    let path_refs: Vec<&str> = op.field_path.iter().map(|s| s.as_str()).collect();
    let filter_id = layouts::bind_field_filtered(
        &mut graph,
        root_layout,
        &path_refs,
        op.predicate.clone(),
        op.projection.clone(),
        format!("{}.scan", op.label),
        Arc::clone(&handle.runtime),
        segment_source,
        &session,
    )?;

    // Sink domain: filtered-rows have unknown cardinality.
    let sink_domain = Domain::new(
        DomainId::new(format!("{}.sink", op.label)),
        Cardinality::Unknown,
    );
    let captured: Arc<Mutex<Vec<ArrayRef>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_id = graph.add_operator(OperatorNode::new(ArrayCollectSink::new(
        format!("{}.sink", op.label),
        sink_domain,
        Arc::clone(&captured),
    )));
    graph.connect(
        OperatorGraph::output(filter_id),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(64 << 20),
    );

    // Run the inner task synchronously.
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
    PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;

    // Hold the file handle alive until run completes. `handle` is
    // an `Arc` shared with the operator's persistent slot, so this
    // local clone simply releases here while the canonical reference
    // lives on `op.handle`.
    drop(handle);

    let arrays = std::mem::take(&mut *captured.lock());
    Ok(arrays)
}

