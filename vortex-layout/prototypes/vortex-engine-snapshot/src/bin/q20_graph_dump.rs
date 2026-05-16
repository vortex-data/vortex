//! Dump the q20 unioned operator graph for inspection.
//!
//! Usage: `cargo run --release --bin q20_graph_dump [shards]`

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_engine::Cardinality;
use vortex_engine::ChannelBuffer;
use vortex_engine::Domain;
use vortex_engine::DomainId;
use vortex_engine::OperatorGraph;
use vortex_engine::OperatorId;
use vortex_engine::OperatorNode;
use vortex_engine::layouts;
use vortex_engine::operators::CollectI64Sink;

const DATA_DIR: &str =
    "/Users/ngates/git/vortex/vortex-bench/data/clickbench_partitioned/vortex-file-compressed";
const TARGET: i64 = 435090932899640449;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shard_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(DATA_DIR)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vortex"))
        .collect();
    paths.sort();
    paths.truncate(shard_count);

    let predicate = eq(root(), lit(TARGET));
    let mut graph = OperatorGraph::new();
    let mut input_domains: Vec<Domain> = Vec::new();
    let mut filter_ids: Vec<OperatorId> = Vec::new();
    let mut _handles = Vec::new();

    for (i, path) in paths.into_iter().enumerate() {
        let handle = layouts::open_vortex_file(&path).expect("open");
        let session = handle.session();
        let segment_source = handle.file.segment_source();
        let root_layout = Arc::clone(handle.file.footer().layout());
        let filter_id = layouts::bind_field_filtered(
            &mut graph,
            root_layout,
            &["UserID"],
            predicate.clone(),
            root(),
            format!("userid_eq[{i}]"),
            Arc::clone(&handle.runtime),
            segment_source,
            &session,
        )
        .expect("bind");
        let input_domain = Domain::new(
            DomainId::new(format!("filter_out:filter:userid_eq[{i}]")),
            Cardinality::Unknown,
        );
        input_domains.push(input_domain);
        filter_ids.push(filter_id);
        _handles.push(handle);
    }

    let union_output_domain =
        Domain::new(DomainId::new("union_out:userid_eq"), Cardinality::Unknown);
    drop(input_domains);
    let captured: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_id = graph.add_operator(OperatorNode::new(CollectI64Sink::new(
        "collect_i64",
        union_output_domain,
        Arc::clone(&captured),
    )));
    graph.connect_multi_named(
        "fanin:userid_eq",
        filter_ids.clone(),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    print!("{}", graph.dump());
}
