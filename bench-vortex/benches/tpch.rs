use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{load_datasets, run_tpch_query, tpch_queries, EXPECTED_ROW_COUNTS};
use bench_vortex::Format;
use criterion::{criterion_group, criterion_main, Criterion};
use tokio::runtime::Builder;

fn benchmark(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .thread_name("benchmark-tpch")
        .enable_all()
        .build()
        .unwrap();

    // Run TPC-H data gen.
    let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();

    let vortex_ctx = runtime
        .block_on(load_datasets(
            &data_dir,
            Format::InMemoryVortex {
                enable_pushdown: true,
            },
        ))
        .unwrap();
    let arrow_ctx = runtime
        .block_on(load_datasets(&data_dir, Format::Arrow))
        .unwrap();
    let parquet_ctx = runtime
        .block_on(load_datasets(&data_dir, Format::Parquet))
        .unwrap();
    let vortex_compressed_ctx = runtime
        .block_on(load_datasets(
            &data_dir,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ))
        .unwrap();

    for (q, sql_queries) in tpch_queries() {
        let expected_row_count = EXPECTED_ROW_COUNTS[q];
        let mut group = c.benchmark_group(format!("tpch_q{q}"));
        group.sample_size(10);

        group.bench_function("vortex-in-memory-pushdown", |b| {
            b.to_async(&runtime).iter(|| async {
                let row_count = run_tpch_query(
                    &vortex_ctx,
                    &sql_queries,
                    q,
                    Format::InMemoryVortex {
                        enable_pushdown: true,
                    },
                )
                .await;
                assert_eq!(expected_row_count, row_count, "Mismatched row count {row_count} instead of {expected_row_count} in query {q} for in memory pushdown format");

            })
        });

        group.bench_function("arrow", |b| {
            b.to_async(&runtime).iter(|| async {
                let row_count = run_tpch_query(&arrow_ctx, &sql_queries, q, Format::Arrow).await;
                assert_eq!(expected_row_count, row_count, "Mismatched row count {row_count} instead of {expected_row_count} in query {q} for arrow format");
            })
        });

        group.bench_function("parquet", |b| {
            b.to_async(&runtime).iter(|| async {
                let row_count = run_tpch_query(&parquet_ctx, &sql_queries, q, Format::Parquet).await;
                assert_eq!(expected_row_count, row_count, "Mismatched row count {row_count} instead of {expected_row_count} in query {q} for parquet format");
            })
        });

        group.bench_function("vortex-file-compressed", |b| {
            b.to_async(&runtime).iter(|| async {
                let row_count = run_tpch_query(
                    &vortex_compressed_ctx,
                    &sql_queries,
                    q,
                    Format::OnDiskVortex {
                        enable_compression: true,
                    },
                )
                .await;
                assert_eq!(expected_row_count, row_count, "Mismatched row count {row_count} instead of {expected_row_count} in query {q} for on disk compressed format");
            })
        });
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
