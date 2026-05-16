//! ClickBench Q20 driver for DataFusion physical-plan execute.
//!
//! Pre-builds the physical plans (so SQL parsing and logical
//! optimization happen outside the timed region), then runs
//! `datafusion_physical_plan::collect` over every shard in a tight
//! loop. Suitable for `samply` profiling — the hot stacks are
//! DataFusion + Vortex, with no engine code in the way.
//!
//! Usage:
//!   cargo build --release --example q20_df
//!   samply record ./target/release/examples/q20_df [shard_count] [iters]
//!
//! Defaults: 100 shards, 5 iterations.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow_schema::Schema;
use datafusion::prelude::SessionContext;
use vortex::VortexSessionDefault;
use vortex_datafusion::v2::VortexTable;
use vortex_file::OpenOptionsSessionExt;
use vortex_session::VortexSession;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const TARGET: i64 = 435090932899640449;

fn paths(n: usize) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(DATA_DIR)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("vortex"))
        .collect();
    out.sort();
    out.truncate(n);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shard_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let paths = paths(shard_count);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio rt");

    let plans = rt.block_on(async {
        let mut plans = Vec::new();
        for path in &paths {
            let session = VortexSession::default();
            let file = session
                .open_options()
                .open_path(path.clone())
                .await
                .expect("open");
            let arrow_schema =
                Arc::new(file.dtype().to_arrow_schema().expect("arrow schema"));
            let data_source = file.data_source().expect("data source");
            let table = Arc::new(VortexTable::new(
                data_source,
                session,
                arrow_schema as Arc<Schema>,
            ));
            let ctx = SessionContext::new();
            ctx.register_table("hits", table).expect("register");
            let df = ctx
                .sql(&format!(
                    "SELECT \"UserID\" FROM hits WHERE \"UserID\" = {TARGET}"
                ))
                .await
                .expect("sql");
            let plan = df.create_physical_plan().await.expect("plan");
            let task_ctx = ctx.task_ctx();
            plans.push((plan, task_ctx));
        }
        plans
    });

    println!(
        "q20_df: shards={shard_count} iters={iters} target={TARGET}",
    );

    let mut times = Vec::with_capacity(iters);
    let mut last_count = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let count = rt.block_on(async {
            use datafusion_physical_plan::ExecutionPlan;
            use datafusion_physical_plan::collect;
            let mut total = 0;
            for (plan, task_ctx) in &plans {
                let plan: Arc<dyn ExecutionPlan> = Arc::clone(plan);
                let batches =
                    collect(plan, Arc::clone(task_ctx)).await.expect("execute");
                for b in &batches {
                    total += b.num_rows();
                }
            }
            total
        });
        times.push(start.elapsed());
        last_count = count;
    }
    times.sort();
    println!(
        "matches={last_count} min={:?} median={:?} max={:?}",
        times[0],
        times[iters / 2],
        times[iters - 1]
    );
}
