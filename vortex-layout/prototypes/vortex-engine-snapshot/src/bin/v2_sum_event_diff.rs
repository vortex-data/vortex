//! v2 ClickBench query: `SELECT SUM(EventTime - START_TS) FROM hits`.
//!
//! Single-plan version, dynamic shard submission through `Gather`.
//!
//! Plan shape:
//!
//! ```text
//! F64ScalarSink
//!   └── VortexAggregate<Sum>(accumulate=f64)            ← combine partials
//!         └── Gather<100>(target_concurrency=2*N)       ← cross-shard merge
//!               ├── Sum(i64) ← Sub(i64,START_TS) ← Cast(i64) ← Scan(file_0)
//!               ├── Sum(i64) ← Sub(i64,START_TS) ← Cast(i64) ← Scan(file_1)
//!               ...
//!               └── Sum(i64) ← Sub(i64,START_TS) ← Cast(i64) ← Scan(file_99)
//! ```
//!
//! The bench loop is now `for iter { build_plan(); run_plan(); }`. No
//! `blocking::unblock`, no per-shard outer fan-out — Gather owns child
//! submission inside the runtime.
//!
//! Per-shard SUM fits i64 (per-row diff × 1 M rows ≈ 1.2 e18, well
//! under i64::MAX). Cross-shard accumulation goes through the
//! combiner's f64 accumulator to handle the 100× total (~1.2 e20).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Poll;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_engine::Cardinality;
use vortex_engine::Domain;
use vortex_engine::DomainId;
use vortex_engine::DomainSpan;
use vortex_engine::EngineError;
use vortex_engine::EngineResult;
use vortex_engine::OutputContract;
use vortex_engine::physical_plan::DriverIo;
use vortex_engine::physical_plan::gather::{Gather, GatherInput};
use vortex_engine::physical_plan::vortex_aggregate::VortexAggregate;
use vortex_engine::physical_plan::vortex_scan::VortexScanSource;
use vortex_engine::physical_plan::{
    Batch, BuildResult, LocalInitRuntime, LoweringCtx, Operator as PlanOperator, OperatorPoll,
    PendingSend, PipelineBuilder, PipelineTail, SinkCtx, SinkNode, TransformCtx, TransformNode,
    TransformOutput, runtime,
};
use vortex_session::VortexSession;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const START_TS: i64 = 1_372_636_800_000_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shards: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let paths = collect_paths(shards);
    eprintln!(
        "v2_sum_event_diff: shards={} iters={iterations} START_TS={START_TS}",
        paths.len()
    );

    let session = default_session();
    let io = DriverIo::new(
        std::env::var("VORTEX_ENGINE_IO_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4),
    );
    let target_concurrency: usize = std::env::var("GATHER_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2 * 8);

    // Time only the physical-plan execution per iteration — building
    // the plan (which includes serially opening every file's Vortex
    // footer to learn its row_count / dtype) is the equivalent of
    // DataFusion's logical planning + Parquet metadata read, and isn't
    // what we're measuring here.
    let mut iter_times = Vec::new();
    for iter in 0..iterations {
        let captured: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
        let (t_build, t_run, total_count) =
            run_query(&paths, &session, Arc::clone(&io), target_concurrency, Arc::clone(&captured))
                .expect("query");
        let total_sum: f64 = captured.lock().unwrap().unwrap_or(0.0);
        iter_times.push(t_run);
        eprintln!(
            "  iter {iter}: run={t_run:?} (plan-build={t_build:?}) count={total_count} sum={total_sum:.6e}"
        );
    }

    let total: std::time::Duration = iter_times.iter().sum();
    let avg = total / iterations as u32;
    let min = iter_times.iter().min().unwrap();
    eprintln!("  avg: {avg:?}  min: {min:?}");
}

fn default_session() -> VortexSession {
    let session = VortexSession::default();
    if std::env::var("DISABLE_DICT_SUM").is_err() {
        vortex_engine::kernels::install(&session);
    }
    session
}

fn collect_paths(shards: usize) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = (0..shards)
        .map(|i| PathBuf::from(format!("{DATA_DIR}/hits_{i}.vortex")))
        .filter(|p| p.exists())
        .collect();
    paths.sort();
    paths
}

/// Build a single plan with all shards under a Gather and run it.
/// Returns the total row count (sum over shards' cardinalities).
fn run_query(
    paths: &[PathBuf],
    session: &VortexSession,
    io: Arc<DriverIo>,
    target_concurrency: usize,
    captured: Arc<Mutex<Option<f64>>>,
) -> EngineResult<(std::time::Duration, std::time::Duration, u64)> {
    let t_build_start = Instant::now();
    let i64_nullable = DType::Primitive(PType::I64, Nullability::Nullable);
    let f64_nullable = DType::Primitive(PType::F64, Nullability::Nullable);
    let mut total_rows: u64 = 0;
    let mut gather_children: Vec<GatherInput> = Vec::with_capacity(paths.len());

    for (idx, path) in paths.iter().enumerate() {
        // Open the source to learn the row count (used for the agg's
        // declared domain; the actual rows-read is the source's
        // responsibility at runtime).
        let source = VortexScanSource::open(format!("scan_{idx}"), path, "EventTime")?;
        let input_domain = source.output_domain().clone();
        let input_contract = source.output_contract().clone();
        let row_count = match input_domain.cardinality() {
            Cardinality::Exact(n) => n,
            Cardinality::Unknown => 0,
        };
        total_rows += row_count;

        // Per-shard: scan → cast-and-subtract → SumAgg.
        let subtract = SubtractI64Op {
            label: format!("sub_{idx}"),
            input_domain: input_domain.clone(),
            input_contract: input_contract.clone(),
            output_domain: input_domain.clone(),
            output_contract: OutputContract::new(i64_nullable.clone()),
            start_ts: START_TS,
            input: Box::new(source),
        };
        let shard_agg_domain = Domain::new(
            DomainId::new(format!("shard_agg_{idx}")),
            Cardinality::Exact(1),
        );
        let shard_sum_contract = OutputContract::new(i64_nullable.clone());
        let shard_agg = VortexAggregate::new(
            format!("sum_{idx}"),
            input_domain.clone(),
            OutputContract::new(i64_nullable.clone()),
            shard_agg_domain.clone(),
            shard_sum_contract.clone(),
            Sum,
            EmptyOptions,
            session.clone(),
            Box::new(subtract),
        )
        .with_accumulate_dtype(i64_nullable.clone());

        gather_children.push(GatherInput::new(
            Box::new(shard_agg),
            shard_agg_domain,
            shard_sum_contract,
        ));
    }

    // Combine: Gather collects all per-shard 1-row i64 batches, then
    // VortexAggregate<Sum> with f64 accumulator merges them into one
    // f64 scalar (f64 because cross-shard total overflows i64).
    let gather_out_domain = Domain::new(
        DomainId::new("gather_out"),
        Cardinality::Exact(paths.len() as u64),
    );
    let gather = Gather::new(
        "gather",
        gather_children,
        gather_out_domain.clone(),
        OutputContract::new(i64_nullable.clone()),
        target_concurrency,
    );
    let final_agg_domain = Domain::new(DomainId::new("final_agg"), Cardinality::Exact(1));
    let final_sum_contract = OutputContract::new(f64_nullable.clone());
    let final_agg = VortexAggregate::new(
        "final_sum",
        gather_out_domain,
        OutputContract::new(i64_nullable.clone()),
        final_agg_domain.clone(),
        final_sum_contract.clone(),
        Sum,
        EmptyOptions,
        session.clone(),
        Box::new(gather),
    )
    .with_accumulate_dtype(f64_nullable);

    let sink = F64ScalarSink {
        label: "sum_capture".to_string(),
        input_domain: final_agg_domain.clone(),
        input_contract: final_sum_contract.clone(),
        captured,
        input: Box::new(final_agg),
    };

    let mut builder = PipelineBuilder::new();
    sink.lower_as_root(&mut builder)
        .map_err(|e| EngineError::message(format!("lower: {e}")))?;
    let plan = builder.into_plan();
    let t_build = t_build_start.elapsed();
    let t_run_start = Instant::now();
    runtime::run_plan_blocking_with_io(plan, io)?;
    let t_run = t_run_start.elapsed();
    Ok((t_build, t_run, total_rows))
}

// -- Cast-to-i64 + subtract-const transform --------------------------

/// Plan-time op: cast input to plain i64 (canonicalises extensions),
/// then subtract a fixed i64 constant.
struct SubtractI64Op {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    output_domain: Domain,
    output_contract: OutputContract,
    start_ts: i64,
    input: Box<dyn PlanOperator>,
}

struct SubtractI64Node {
    label: String,
    start_ts: i64,
}

#[derive(Default)]
struct SubtractI64Local {
    pending: Option<Batch>,
    input_done: bool,
}

impl TransformNode for SubtractI64Node {
    type LocalState = SubtractI64Local;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(SubtractI64Local::default())
    }

    fn can_accept_input(&self, local: &Self::LocalState) -> bool {
        local.pending.is_none() && !local.input_done
    }

    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        let array = batch.array().clone();
        let len = array.len();
        let i64_nullable = DType::Primitive(PType::I64, Nullability::Nullable);
        // Cast to plain i64 (strips Timestamp extension; canonicalises
        // DateTimeParts to materialised i64 µs values).
        let array = array
            .cast(i64_nullable.clone())
            .map_err(|e| EngineError::message(format!("cast to i64: {e}")))?;
        let constant = ConstantArray::new(
            Scalar::from(self.start_ts)
                .cast(&i64_nullable)
                .map_err(|e| EngineError::message(format!("const cast: {e}")))?,
            len,
        )
        .into_array();
        let diff = array
            .binary(constant, Operator::Sub)
            .map_err(|e| EngineError::message(format!("subtract: {e}")))?;
        local.pending = Some(Batch::new(diff, batch.span()));
        Ok(())
    }

    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        local.input_done = true;
        Ok(())
    }

    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        if let Some(batch) = local.pending.take() {
            return Poll::Ready(Ok(TransformOutput::Batch(batch)));
        }
        if local.input_done {
            return Poll::Ready(Ok(TransformOutput::Finished));
        }
        Poll::Ready(Ok(TransformOutput::NeedInput))
    }
}

impl PlanOperator for SubtractI64Op {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let _ = (&self.output_domain, &self.output_contract);
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            SubtractI64Node {
                label: self.label.clone(),
                start_ts: self.start_ts,
            },
        );
        self.input.lower(ctx, tail)
    }
}

// -- Scalar capture sink ---------------------------------------------

struct F64ScalarSink {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    captured: Arc<Mutex<Option<f64>>>,
    input: Box<dyn PlanOperator>,
}

struct F64ScalarNode {
    label: String,
    captured: Arc<Mutex<Option<f64>>>,
}

#[derive(Default)]
struct F64ScalarLocal;

impl SinkNode for F64ScalarNode {
    type LocalState = F64ScalarLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(F64ScalarLocal)
    }

    fn poll_send(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()> {
        if let Some(batch) = send.take() {
            let array = batch.into_array();
            #[expect(deprecated)]
            let scalar = match array.scalar_at(0) {
                Ok(s) => s,
                Err(e) => {
                    return Poll::Ready(Err(EngineError::message(format!("scalar_at: {e}"))));
                }
            };
            *self.captured.lock().unwrap() = scalar.as_primitive().typed_value::<f64>();
        }
        Poll::Ready(Ok(()))
    }

    fn poll_finish(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
    ) -> OperatorPoll<()> {
        Poll::Ready(Ok(()))
    }
}

impl PlanOperator for F64ScalarSink {
    fn lower(&self, _ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        drop(tail);
        Err(vortex_engine::physical_plan::BuildError::message(
            "F64ScalarSink::lower should not be called directly; use lower_as_root",
        ))
    }
}

impl F64ScalarSink {
    fn lower_as_root(&self, ctx: &mut dyn LoweringCtx) -> BuildResult<()> {
        ctx.register_domain(self.input_domain.clone())?;
        let tail = PipelineTail::new(
            self.input_domain.clone(),
            self.input_contract.clone(),
            F64ScalarNode {
                label: self.label.clone(),
                captured: Arc::clone(&self.captured),
            },
        );
        self.input.lower(ctx, tail)
    }
}

#[allow(dead_code)]
fn _force_use() {
    drop(VortexSession::default());
    let _: DomainSpan = DomainSpan::new(0, 0);
}
