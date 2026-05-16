//! Q20 implemented WITHOUT the engine. Calls Vortex's `LayoutReader`
//! directly. Tests whether the engine actually contributes to perf,
//! or whether the win is just "we skipped DataFusion's overhead."

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use vortex_array::MaskFuture;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_layout::Layout;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutRef;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const TARGET: i64 = 435090932899640449;

fn struct_field_subtree(layout: &dyn Layout, field: &str) -> Option<LayoutRef> {
    for idx in 0..layout.nchildren() {
        if let LayoutChildType::Field(name) = layout.child_type(idx)
            && name.as_ref() == field
        {
            return layout.child(idx).ok();
        }
    }
    None
}

fn run_one(path: &std::path::Path, runtime: &Arc<CurrentThreadRuntime>) -> usize {
    use vortex::VortexSessionDefault;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    let session = VortexSession::default().with_handle(runtime.handle());
    let path = path.to_path_buf();
    let s2 = session.clone();
    let file = runtime
        .block_on(async move { s2.open_options().open_path(path).await })
        .expect("open");

    let segment_source = file.segment_source();
    let layout = Arc::clone(file.footer().layout());
    let userid = struct_field_subtree(layout.as_ref(), "UserID").expect("UserID");
    let reader = userid
        .new_reader("userid".into(), segment_source, &session)
        .expect("reader");

    let row_count = reader.row_count();
    let predicate = eq(root(), lit(TARGET));
    let projection = root();
    let mask_in = MaskFuture::new_true(usize::try_from(row_count).unwrap());

    let row_range = 0..row_count;
    let filter_fut = reader
        .filter_evaluation(&row_range, &predicate, mask_in)
        .expect("filter");
    let proj_fut = reader
        .projection_evaluation(&row_range, &projection, filter_fut)
        .expect("project");

    let array = runtime.block_on(proj_fut).expect("array");
    array.len()
}

fn main() {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(DATA_DIR)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vortex"))
        .collect();
    paths.sort();
    eprintln!("running q20_direct over {} shard(s)", paths.len());

    let start = Instant::now();
    let mut total = 0;
    for path in &paths {
        // Fresh runtime per shard, matching what our engine wrapper does.
        let runtime = Arc::new(CurrentThreadRuntime::new());
        total += run_one(path, &runtime);
    }
    let elapsed = start.elapsed();
    println!("matches={total} elapsed={elapsed:?}");
}
