//! DataFusion baseline for `SELECT SUM(EventTime - START_TS) FROM hits`.
//!
//! Mirrors the `v2_sum_event_diff` binary so we can compare a
//! DataFusion + VortexTable run against the engine's pipeline. SQL
//! parsing and physical-plan creation happen outside the timed
//! region (same as `q20_df`).
//!
//! Usage:
//!   cargo build --release --example sum_eventtime_df
//!   ./target/release/examples/sum_eventtime_df [shards=100] [iters=5]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow_schema::Schema;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::Int64Array;
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use vortex::VortexSessionDefault;
use vortex_datafusion::v2::VortexTable;
use vortex_file::OpenOptionsSessionExt;
use vortex_session::VortexSession;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const START_TS: i64 = 1_372_636_800_000_000; // µs, matches v2_sum_event_diff

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


    // SUM(EventTime − START_TS): cast Timestamp[µs] to BIGINT (raw
    // µs storage), subtract the µs constant, then SUM. Same algebra
    // the engine computes.
    let sql = format!(
        "SELECT SUM(CAST(\"EventTime\" AS BIGINT) - {START_TS}) FROM hits"
    );

    // Pre-open the files and build the tables ONCE outside the
    // timed region. We rebuild physical plans per iter because
    // DataFusion 53's RepartitionExec retains state across executions
    // and panics on re-run with multi-partition plans.
    let tables = rt.block_on(async {
        let mut tables = Vec::new();
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
            let table: Arc<dyn TableProvider> = Arc::new(VortexTable::new(
                data_source,
                session,
                arrow_schema as Arc<Schema>,
            ));
            tables.push(table);
        }
        tables
    });

    println!(
        "sum_eventtime_df: shards={shard_count} iters={iters} START_TS={START_TS}"
    );

    let mut times = Vec::with_capacity(iters);
    let mut last_total: i128 = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let total = rt.block_on(async {
            use datafusion_physical_plan::ExecutionPlan;
            use datafusion_physical_plan::collect;
            // Rebuild plans this iter.
            let mut plans = Vec::with_capacity(tables.len());
            for table in &tables {
                let mut cfg = datafusion::execution::config::SessionConfig::new();
                if let Some(p) = std::env::var("DF_PARTITIONS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    cfg = cfg.with_target_partitions(p);
                }
                let ctx = SessionContext::new_with_config(cfg);
                ctx.register_table("hits", Arc::clone(table)).expect("register");
                let df = ctx.sql(&sql).await.expect("sql");
                let plan = df.create_physical_plan().await.expect("plan");
                let task_ctx = ctx.task_ctx();
                plans.push((plan, task_ctx));
            }
            // Set SHARDS_PARALLEL=1 to fire all shards concurrently
            // via tokio::spawn (each plan also parallelises internally
            // per its target_partitions setting).
            if std::env::var("SHARDS_PARALLEL").is_ok() {
                let mut handles = Vec::with_capacity(plans.len());
                for (plan, task_ctx) in plans {
                    handles.push(tokio::spawn(async move {
                        let plan: Arc<dyn ExecutionPlan> = plan;
                        collect(plan, task_ctx).await.expect("execute")
                    }));
                }
                let mut total: i128 = 0;
                for h in handles {
                    let batches = h.await.expect("join");
                    for b in &batches {
                        let col = b.column(0);
                        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                            for i in 0..arr.len() {
                                if !arr.is_null(i) {
                                    total += arr.value(i) as i128;
                                }
                            }
                        }
                    }
                }
                total
            } else {
                let mut total: i128 = 0;
                for (plan, task_ctx) in plans {
                    let plan: Arc<dyn ExecutionPlan> = plan;
                    let batches =
                        collect(plan, task_ctx).await.expect("execute");
                    for b in &batches {
                        let col = b.column(0);
                        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                            for i in 0..arr.len() {
                                if !arr.is_null(i) {
                                    total += arr.value(i) as i128;
                                }
                            }
                        }
                    }
                }
                total
            }
        });
        times.push(start.elapsed());
        last_total = total;
    }
    times.sort();
    println!(
        "sum(EventTime-START_TS)={last_total}  min={:?}  median={:?}  max={:?}",
        times[0],
        times[iters / 2],
        times[iters - 1]
    );
}
