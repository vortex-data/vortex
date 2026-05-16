//! Q20 implemented through Vortex's own ScanBuilder API. This is
//! the apples-to-apples physical-execution baseline: same scan
//! abstraction DataFusion uses internally, but with no SQL parse,
//! no logical-plan optimisation, no DataFusion record-batch
//! pipeline, and no parallelism.
//!
//! Selecting projection = `select(["UserID"], root())` and filter =
//! `eq(get("UserID"), lit(TARGET))` against the FULL clickbench
//! struct schema (since the scan is on the file root, not a single
//! field), so this hits Vortex's own column-projection +
//! predicate-pushdown path the same way DataFusion does.

use std::path::PathBuf;
use std::time::Instant;

use futures::TryStreamExt;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const TARGET: i64 = 435090932899640449;

fn run_one(path: &std::path::Path, runtime: &CurrentThreadRuntime) -> usize {
    use vortex::VortexSessionDefault;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    let session = VortexSession::default().with_handle(runtime.handle());
    let path = path.to_path_buf();
    let s2 = session.clone();

    let predicate = eq(get_item("UserID", root()), lit(TARGET));
    let projection = select(["UserID"], root());

    let total = runtime.block_on(async move {
        let file = s2.open_options().open_path(path).await.expect("open");
        let stream = file
            .scan()
            .expect("scan")
            .with_filter(predicate)
            .with_projection(projection)
            .into_array_stream()
            .expect("stream");
        let arrays: Vec<_> = stream
            .try_collect()
            .await
            .expect("collect");
        arrays.iter().map(|a| a.len()).sum::<usize>()
    });
    total
}

fn main() {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(DATA_DIR)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vortex"))
        .collect();
    paths.sort();
    eprintln!("running q20_vortex_scan over {} shard(s)", paths.len());

    let start = Instant::now();
    let mut total = 0;
    for path in &paths {
        let runtime = CurrentThreadRuntime::new();
        total += run_one(path, &runtime);
    }
    let elapsed = start.elapsed();
    println!("matches={total} elapsed={elapsed:?}");
}
