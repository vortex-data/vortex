//! v2 spiral with plan-time push-down — the equivalent of
//! backprop's `limited` run, but expressed as a static narrowing
//! of source inputs at plan time rather than a runtime back-prop
//! mechanism.
//!
//! This binary computes the grandchild row range needed for
//! `limit` parents from the static offsets, and constructs each
//! source over just that range. Same algorithm as v2_spiral; less
//! data flowing through the pipelines.
//!
//! Usage:
//!   v2_spiral_pushdown [parents] [child/parent] [grandchild/child] [limit] [iters]
//! Default: 100 100 100 1 5
//!
//! Reports wall-clock for the pushed-down query plus the count of
//! grandchild rows actually materialised.

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
use vortex_engine::physical_plan::limit::Limit;
use vortex_engine::physical_plan::operators::{CollectSink, IntSource};
use vortex_engine::physical_plan::parent_child_min::ParentChildMin;
use vortex_engine::physical_plan::{PipelineBuilder, runtime};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parents: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let child_per_parent: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let grandchild_per_child: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let limit: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iterations: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);

    let children_total = parents * child_per_parent;
    let grandchildren_total = children_total * grandchild_per_child;

    // Plan-time push-down: only the first `limit` parents' worth of
    // child and grandchild rows are visited.
    let parents_used = limit;
    let children_used = parents_used * child_per_parent;
    let grandchildren_used = children_used * grandchild_per_child;

    eprintln!(
        "v2_spiral_pushdown: parents={parents}/{parents_used} \
         child/parent={child_per_parent} \
         grandchild/child={grandchild_per_child} \
         limit={limit} iters={iterations}"
    );
    eprintln!(
        "  grandchild rows (full / pushed): {grandchildren_total} / {grandchildren_used} \
         (reduction {:.1}×)",
        (grandchildren_total as f64) / (grandchildren_used.max(1) as f64)
    );

    let contract = OutputContract::new(DType::Primitive(PType::I64, Nullability::NonNullable));

    // Only construct the data we'll actually scan.
    let cg_offsets_pushed: Vec<i64> = (0..=children_used)
        .map(|i| (i * grandchild_per_child) as i64)
        .collect();
    let pc_offsets_pushed: Vec<i64> = (0..=parents_used)
        .map(|i| (i * child_per_parent) as i64)
        .collect();
    let grandchild_vals_pushed: Vec<i64> = (0..grandchildren_used as i64).collect();

    let mut results = Vec::new();
    for iter in 0..iterations {
        let t0 = Instant::now();
        let rows = run_once(
            parents_used,
            children_used,
            grandchildren_used,
            limit,
            &cg_offsets_pushed,
            &pc_offsets_pushed,
            &grandchild_vals_pushed,
            &contract,
        );
        let dt = t0.elapsed();
        results.push(dt);
        eprintln!("  iter {iter}: {dt:?}  output_rows={}", rows.len());
        if iter == 0 {
            eprintln!("    first output: {:?}", rows.first());
        }
    }

    let total: std::time::Duration = results.iter().sum();
    let avg = total / iterations as u32;
    let min = results.iter().min().unwrap();
    eprintln!("  avg: {avg:?}  min: {min:?}");
    eprintln!(
        "  effective throughput: ~{:.2}M grandchild rows/s (full domain)",
        (grandchildren_total as f64) / avg.as_secs_f64() / 1.0e6
    );
}

fn run_once(
    parents_used: u64,
    children_used: u64,
    grandchildren_used: u64,
    limit: u64,
    cg_offsets: &[i64],
    pc_offsets: &[i64],
    grandchild_vals: &[i64],
    contract: &OutputContract,
) -> Vec<i64> {
    let parent = domain("parent_rows", parents_used);
    let child = domain("child_rows", children_used);
    let grandchild = domain("grandchild_rows", grandchildren_used);
    let child_min = domain("child_min", children_used);
    let parent_min = domain("parent_min", parents_used);
    let limited = domain("parent_min_limited", limit);
    let rows: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));

    let cg_off = IntSource::new(
        "cg_offsets",
        child.clone(),
        contract.clone(),
        cg_offsets.to_vec(),
    )
    .with_batch_rows(1024);
    let gv = IntSource::new(
        "grandchild_values",
        grandchild.clone(),
        contract.clone(),
        grandchild_vals.to_vec(),
    )
    .with_batch_rows(8192);
    let level1 = ParentChildMin::new(
        "min_per_child",
        child.clone(),
        contract.clone(),
        Box::new(cg_off),
        grandchild,
        contract.clone(),
        Box::new(gv),
        child_min.clone(),
        contract.clone(),
    )
    .with_batch_rows(1024);

    let pc_off = IntSource::new(
        "pc_offsets",
        parent.clone(),
        contract.clone(),
        pc_offsets.to_vec(),
    )
    .with_batch_rows(1024);
    let level2 = ParentChildMin::new(
        "min_per_parent",
        parent.clone(),
        contract.clone(),
        Box::new(pc_off),
        child_min,
        contract.clone(),
        Box::new(level1),
        parent_min.clone(),
        contract.clone(),
    )
    .with_batch_rows(1024);

    let lim = Limit::new(
        "limit_parents",
        parent_min,
        contract.clone(),
        limit,
        Box::new(level2),
    );
    let sink = CollectSink::new(
        "collect",
        limited,
        contract.clone(),
        Arc::clone(&rows),
        Box::new(lim),
    );

    let mut builder = PipelineBuilder::new();
    sink.lower_as_root(&mut builder).expect("lower");
    let plan = builder.into_plan();
    runtime::run_plan_blocking(plan).expect("run");

    rows.lock().unwrap().clone()
}

fn domain(name: &str, len: u64) -> Domain {
    Domain::new(DomainId::new(name), Cardinality::Exact(len))
}
