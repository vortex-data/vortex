//! Tiny benchmark for the v2 pipeline runtime.
//!
//! Usage: `v2_bench [left_rows] [right_rows] [iterations]`
//!
//! Default: 1,000,000 left × 1,000,000 right × 3 iterations.
//!
//! Runs a sorted-merge equi-join through the v2 plumbing. Keys are
//! interleaved so ~50% of rows on each side match.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_engine::Cardinality;
use vortex_engine::Domain;
use vortex_engine::DomainId;
use vortex_engine::OutputContract;
use vortex_engine::physical_plan::merge_join::SortedMergeJoin;
use vortex_engine::physical_plan::operators::{CollectSink, IntSource};
use vortex_engine::physical_plan::{PipelineBuilder, runtime};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let left_rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let right_rows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let batch_rows: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8192);

    eprintln!(
        "v2_bench: left={left_rows} right={right_rows} iters={iterations} batch_rows={batch_rows}"
    );

    // Generate keys with ~50% overlap. Left = 0, 2, 4, ..., right =
    // 0, 1, 2, ..., so half of left keys match every right key
    // exactly once.
    let left_keys: Vec<i64> = (0..left_rows as i64).map(|i| i * 2).collect();
    let right_keys: Vec<i64> = (0..right_rows as i64).collect();

    let contract = OutputContract::new(DType::Primitive(PType::I64, Nullability::NonNullable));

    let mut results = Vec::new();
    for iter in 0..iterations {
        let t0 = Instant::now();
        let rows = run_once(&left_keys, &right_keys, batch_rows, &contract);
        let dt = t0.elapsed();
        results.push(dt);
        eprintln!(
            "  iter {iter}: {dt:?}  output_rows={}",
            rows.len()
        );
    }

    let total: std::time::Duration = results.iter().sum();
    let avg = total / iterations as u32;
    let min = results.iter().min().unwrap();
    let throughput = ((left_rows + right_rows) as f64) / avg.as_secs_f64() / 1.0e6;
    eprintln!("  avg: {avg:?}  min: {min:?}  ~{throughput:.2}M rows/s (both sides)");
}

fn run_once(
    left_keys: &[i64],
    right_keys: &[i64],
    batch_rows: usize,
    contract: &OutputContract,
) -> Vec<i64> {
    let left_domain = Domain::new(
        DomainId::new("merge_left"),
        Cardinality::Exact(left_keys.len() as u64),
    );
    let right_domain = Domain::new(
        DomainId::new("merge_right"),
        Cardinality::Exact(right_keys.len() as u64),
    );
    // We don't know the output cardinality a priori; declare a generous bound.
    let joined = Domain::new(
        DomainId::new("merge_joined"),
        Cardinality::Exact((left_keys.len() + right_keys.len()) as u64),
    );
    let rows: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));

    let left = IntSource::new(
        "left_keys",
        left_domain.clone(),
        contract.clone(),
        left_keys.to_vec(),
    )
    .with_batch_rows(batch_rows);
    let right = IntSource::new(
        "right_keys",
        right_domain.clone(),
        contract.clone(),
        right_keys.to_vec(),
    )
    .with_batch_rows(batch_rows);
    let join = SortedMergeJoin::new(
        "bench_join",
        left_domain,
        contract.clone(),
        Box::new(left),
        right_domain,
        contract.clone(),
        Box::new(right),
        joined.clone(),
        contract.clone(),
    );
    let sink = CollectSink::new(
        "collect",
        joined,
        contract.clone(),
        Arc::clone(&rows),
        Box::new(join),
    );

    let mut builder = PipelineBuilder::new();
    sink.lower_as_root(&mut builder).expect("lower");
    let plan = builder.into_plan();
    runtime::run_plan_blocking(plan).expect("run");

    let collected = rows.lock().unwrap().clone();
    collected
}
