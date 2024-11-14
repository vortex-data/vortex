use bench_vortex::clickbench::{clickbench_queries, HITS_SCHEMA};
use bench_vortex::data_downloads::download_data;
use bench_vortex::tpch::execute_query;
use bench_vortex::{clickbench, IdempotentPath};
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::prelude::SessionContext;
use tokio::runtime::Builder;

fn benchmark(c: &mut Criterion) {
    let runtime = Builder::new_current_thread().enable_all().build().unwrap();
    let basepath = "clickbench".to_data_path();

    let parquet_file = download_data(
        basepath.join("hits.parquet"),
        "https://datasets.clickhouse.com/hits_compatible/hits.parquet",
    );

    let session_context = SessionContext::new();
    let context = session_context.clone();
    runtime.block_on(async move {
        clickbench::register_vortex_file(
            &context,
            "hits",
            parquet_file.as_path(),
            &HITS_SCHEMA,
            true,
        )
        .await
        .unwrap();
    });

    let mut group = c.benchmark_group("clickbench");

    for (idx, query) in clickbench_queries().into_iter() {
        let context = session_context.clone();
        group.bench_function(format!("q-{}", idx + 1), |b| {
            b.to_async(&runtime)
                .iter(|| async { execute_query(&context, &query).await.unwrap() });
        });
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
