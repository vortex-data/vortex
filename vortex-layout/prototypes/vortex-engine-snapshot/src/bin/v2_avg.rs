//! v2 AVG over a configurable ClickBench column.
//!
//! Same shape as `v2_q3` but the column name is a CLI argument.
//! Designed to exercise the `(Dict, Sum)` kernel on low-cardinality
//! columns like `ResolutionWidth` (K≈150 values for ~1M rows per
//! chunk → kernel wins) vs. high-cardinality columns like `UserID`
//! where the density guard short-circuits and falls through to
//! canonical.
//!
//! Usage: `v2_avg <column> [shards=100] [iterations=3]`
//!
//! Set `DISABLE_DICT_SUM=1` to skip kernel registration entirely
//! (compare against canonical-only baseline).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Poll;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::combined::PairOptions;
use vortex_array::aggregate_fn::fns::mean::Mean;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_engine::Cardinality;
use vortex_engine::Domain;
use vortex_engine::DomainId;
use vortex_engine::EngineError;
use vortex_engine::EngineResult;
use vortex_engine::OutputContract;
use vortex_engine::physical_plan::vortex_aggregate::VortexAggregate;
use vortex_engine::physical_plan::vortex_scan::VortexScanSource;
use vortex_engine::physical_plan::{
    BuildResult, LocalInitRuntime, LoweringCtx, Operator, OperatorPoll, PendingSend,
    PipelineBuilder, PipelineTail, SinkCtx, SinkNode, runtime,
};
use vortex_session::VortexSession;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let column = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: v2_avg <column> [shards=100] [iterations=3]");
        std::process::exit(2);
    });
    let shards: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let paths = collect_paths(shards);
    eprintln!(
        "v2_avg: column={column} shards={} iters={iterations}",
        paths.len()
    );

    let session = default_session();

    let mut iter_times = Vec::new();
    for iter in 0..iterations {
        let t0 = Instant::now();
        let mut total_sum: f64 = 0.0;
        let mut total_count: u64 = 0;

        for path in &paths {
            let (sum, count) = run_shard(path, &column, &session).expect("shard");
            if let Some(s) = sum {
                total_sum += s;
            }
            total_count += count;
        }
        let dt = t0.elapsed();
        iter_times.push(dt);

        let avg = if total_count > 0 {
            total_sum / (total_count as f64)
        } else {
            f64::NAN
        };
        eprintln!("  iter {iter}: {dt:?}  count={total_count} avg({column})={avg:.6e}");
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

fn run_shard(
    path: &PathBuf,
    column: &str,
    session: &VortexSession,
) -> EngineResult<(Option<f64>, u64)> {
    let source = VortexScanSource::open(column, path, column)?;
    let input_domain = source.output_domain().clone();
    let input_contract = source.output_contract().clone();
    let row_count = match input_domain.cardinality() {
        Cardinality::Exact(n) => n,
        Cardinality::Unknown => 0,
    };
    let agg_domain = Domain::new(DomainId::new("agg"), Cardinality::Exact(1));

    // Cast to f64 at push time. `Dict::cast` rewrites only the K
    // dictionary values, so the Dict structure survives and the
    // `(Dict, Sum)` kernel still dispatches on Dict<codes, f64> via
    // the `sum_float` path. The cast is also what protects against
    // u64/i64 overflow for wide columns like UserID — for small
    // columns like ResolutionWidth it's just safety.
    let f64_nullable = DType::Primitive(PType::F64, Nullability::Nullable);
    let mean_contract = OutputContract::new(f64_nullable.clone());
    let mean_agg = VortexAggregate::new(
        format!("mean({column})"),
        input_domain.clone(),
        input_contract.clone(),
        agg_domain.clone(),
        mean_contract.clone(),
        Mean::combined(),
        PairOptions(EmptyOptions, EmptyOptions),
        session.clone(),
        Box::new(source),
    )
    .with_accumulate_dtype(f64_nullable);

    let mean_value: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let sink = F64ScalarSink {
        label: "mean_capture".to_string(),
        input_domain: agg_domain.clone(),
        input_contract: mean_contract.clone(),
        captured: Arc::clone(&mean_value),
        input: Box::new(mean_agg),
    };
    let mut builder = PipelineBuilder::new();
    sink.lower_as_root(&mut builder)
        .map_err(|e| EngineError::message(format!("lower mean: {e}")))?;
    let plan = builder.into_plan();
    runtime::run_plan_blocking(plan)?;
    let mean = *mean_value.lock().unwrap();
    let sum_estimate = mean.map(|m| m * (row_count as f64));
    Ok((sum_estimate, row_count))
}

struct F64ScalarSink {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    captured: Arc<Mutex<Option<f64>>>,
    input: Box<dyn Operator>,
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

impl Operator for F64ScalarSink {
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
}
