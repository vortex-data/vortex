#![feature(exit_status_error)]

use std::path::PathBuf;
use std::process::Command;

use bench_vortex::clickbench::{clickbench_queries, HITS_SCHEMA};
use bench_vortex::{clickbench, execute_query, idempotent, IdempotentPath};
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::prelude::SessionContext;
use tokio::runtime::Builder;

fn benchmark(c: &mut Criterion) {
    let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
    let basepath = "clickbench".to_data_path();

    // The clickbench-provided file is missing some higher-level type info, so we reprocess it
    // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.
    for idx in 0..100 {
        let output_path = basepath.join(format!("hits_{idx}.parquet"));
        idempotent(&output_path, |output_path| {
            eprintln!("Fixing parquet file {idx}");
            let command = format!(
                "
                SET home_directory='/home/ci-runner/';
                INSTALL HTTPFS;
                COPY (SELECT * REPLACE
                    (epoch_ms(EventTime * 1000) AS EventTime, \
                    epoch_ms(ClientEventTime * 1000) AS ClientEventTime, \
                    epoch_ms(LocalEventTime * 1000) AS LocalEventTime, \
                        DATE '1970-01-01' + INTERVAL (EventDate) DAYS AS EventDate) \
                FROM read_parquet('https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{idx}.parquet', binary_as_string=True)) TO '{}' (FORMAT 'parquet');",
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
    }

    let session_context = SessionContext::new();
    let context = session_context.clone();
    runtime.block_on(async move {
        clickbench::register_vortex_file(&context, "hits", basepath.as_path(), &HITS_SCHEMA)
            .await
            .unwrap();
    });

    let mut group = c.benchmark_group("clickbench");

    for (idx, query) in clickbench_queries().into_iter() {
        let context = session_context.clone();
        group.bench_function(format!("q-{:02}", idx), |b| {
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
