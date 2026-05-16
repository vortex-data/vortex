//! ClickBench Q5 (`SELECT COUNT(DISTINCT "UserID") FROM hits`),
//! single shard.
//!
//! Usage: `q5 [shard_index] [workers] [iterations]`. Default
//! 0 / 1 / 1.
//!
//! There is no multi-shard variant — see
//! [`Q5SingleShard`](vortex_engine::queries::clickbench::Q5SingleShard)
//! for why.

use vortex_engine::queries::bench::BenchHarness;
use vortex_engine::queries::bench::vortex_files_in;
use vortex_engine::queries::clickbench::Q5SingleShard;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shard_index: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let workers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut paths = vortex_files_in(DATA_DIR, usize::MAX);
    if shard_index >= paths.len() {
        eprintln!(
            "shard_index {shard_index} out of range ({} shards available)",
            paths.len()
        );
        std::process::exit(2);
    }
    let path = paths.remove(shard_index);
    BenchHarness::run_and_print(&Q5SingleShard, vec![path], workers, iterations)
        .expect("q5 single shard");
}
