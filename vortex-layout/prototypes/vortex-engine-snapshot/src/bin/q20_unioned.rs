//! ClickBench Q20 — single-graph Union variant.
//!
//! Usage: `q20_unioned [shards] [workers] [iterations]`. Default
//! 100 / 1 / 1.
//!
//! The fan-in is always `Union`: q20's consumer is
//! `CollectI64Sink`, a commutative bag-of-rows aggregator that
//! doesn't care about row order. There is no scenario where
//! ordered output is correct or useful for this query.
//!
//! Setup (parallel file opens + file-stat prune) is timed once.
//! Per-iteration query execution is timed individually. Matches
//! `datafusion-bench`'s `setup(format)` + `execute(query)` shape.

use vortex_engine::queries::bench::BenchHarness;
use vortex_engine::queries::bench::vortex_files_in;
use vortex_engine::queries::clickbench::Q20Unioned;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const TARGET: i64 = 435090932899640449;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shards: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let workers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let paths = vortex_files_in(DATA_DIR, shards);
    BenchHarness::run_and_print(&Q20Unioned::new(TARGET), paths, workers, iterations)
        .expect("q20 unioned");
}
