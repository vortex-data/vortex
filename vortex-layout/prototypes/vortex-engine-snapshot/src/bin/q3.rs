//! ClickBench Q3 (`SELECT AVG("UserID") FROM hits`), evaluated via
//! the partial-then-merge aggregate split: per-shard `bind_field` →
//! `Union` → multi-lane `PartialAggregate(Mean)` → single-lane
//! `MergeAggregate`.
//!
//! Usage: `q3 [shards] [workers] [iterations]`. Default 100 / 1 / 1.
//!
//! Same harness shape as `q20_unioned`: parallel footer opens
//! during setup (untimed); each timed iteration builds the full
//! graph fresh and reads the finalised f64 scalar via
//! `MergeAggregate::with_capture`.

use vortex_engine::queries::bench::BenchHarness;
use vortex_engine::queries::bench::vortex_files_in;
use vortex_engine::queries::clickbench::Q3AvgUserId;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shards: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let workers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let paths = vortex_files_in(DATA_DIR, shards);
    BenchHarness::run_and_print(&Q3AvgUserId, paths, workers, iterations).expect("q3");
}
