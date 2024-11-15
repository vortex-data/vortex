use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{load_datasets, run_tpch_query, tpch_queries, Format};
use divan::{Bencher, Divan};
use itertools::Itertools;
use tokio::runtime::Builder;

#[divan::bench()]
fn vortex_compressed(bencher: Bencher) {
    let runtime = Builder::new_current_thread()
        .thread_name("benchmark-tpch")
        .enable_all()
        .build()
        .unwrap();

    // Run TPC-H data gen.
    let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();

    let vortex_compressed_ctx = runtime
        .block_on(load_datasets(
            &data_dir,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ))
        .unwrap();

    let queries = tpch_queries().collect_vec();

    bencher.bench(|| {
        runtime.block_on(run_tpch_query(
            &vortex_compressed_ctx,
            &queries[0].1,
            queries[0].0,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ))
    })
}

fn main() {
    // Run registered benchmarks.
    Divan::default()
        .sample_size(1)
        .sample_count(3)
        .run_benches()
}
