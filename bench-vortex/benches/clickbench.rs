#![feature(exit_status_error)]

use std::path::PathBuf;
use std::process::Command;

use bench_vortex::clickbench::{clickbench_queries, HITS_SCHEMA};
use bench_vortex::data_downloads::download_data;
use bench_vortex::{clickbench, execute_query, idempotent, IdempotentPath};
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::prelude::SessionContext;
use tokio::runtime::Builder;

fn benchmark(c: &mut Criterion) {
    let runtime = Builder::new_current_thread().enable_all().build().unwrap();
    let basepath = "clickbench".to_data_path();

    let raw_data = download_data(
        basepath.join("hits.parquet"),
        "https://datasets.clickhouse.com/hits_compatible/hits.parquet",
    );

    let output_path = basepath.join("processed.parquet");

    let final_parquet_path = idempotent(&output_path, |output_path| {
        println!("Fixing parquet file");
        let command = format!(
            "COPY (SELECT * REPLACE 
                (epoch_ms(EventTime * 1000) AS EventTime, \
                epoch_ms(ClientEventTime * 1000) AS ClientEventTime, \
                epoch_ms(LocalEventTime * 1000) AS LocalEventTime, \
                    DATE '1970-01-01' + INTERVAL (EventDate) DAYS AS EventDate) \
            FROM read_parquet('{}')) TO '{}' (FORMAT 'parquet');",
            raw_data.to_str().unwrap(),
            output_path.to_str().unwrap()
        );
        Command::new("duckdb")
            .arg("-c")
            .arg(command)
            .status()?
            .exit_ok()?;

        anyhow::Ok(PathBuf::from(output_path))
    })
    .unwrap();

    println!("Registering Vortex file from {final_parquet_path:?}");

    let session_context = SessionContext::new();
    let context = session_context.clone();
    runtime.block_on(async move {
        clickbench::register_vortex_file(
            &context,
            "hits",
            final_parquet_path.as_path(),
            &HITS_SCHEMA,
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

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark
);
criterion_main!(benches);
