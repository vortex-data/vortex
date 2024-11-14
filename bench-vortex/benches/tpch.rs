use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{load_datasets, run_tpch_query, tpch_queries, Format};
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

    let vortex_no_pushdown_ctx = runtime
        .block_on(load_datasets(
            &data_dir,
            Format::InMemoryVortex {
                enable_pushdown: false,
            },
        ))
        .unwrap();
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
    let vortex_uncompressed_ctx = runtime
        .block_on(load_datasets(
            &data_dir,
            Format::OnDiskVortex {
                enable_compression: false,
            },
        ))
        .unwrap();

    for (q, sql_queries) in tpch_queries() {
        let mut group = c.benchmark_group(format!("tpch_q{q}"));
        group.sample_size(10);

        group.bench_function("vortex-in-memory-no-pushdown", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(
                    &vortex_no_pushdown_ctx,
                    &sql_queries,
                    q,
                    Format::InMemoryVortex {
                        enable_pushdown: false,
                    },
                )
                .await;
            })
        });

        group.bench_function("vortex-in-memory-pushdown", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(
                    &vortex_ctx,
                    &sql_queries,
                    q,
                    Format::InMemoryVortex {
                        enable_pushdown: true,
                    },
                )
                .await;
            })
        });

        group.bench_function("arrow", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(&arrow_ctx, &sql_queries, q, Format::Arrow).await;
            })
        });

        group.bench_function("parquet", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(&parquet_ctx, &sql_queries, q, Format::Parquet).await;
            })
        });

        group.bench_function("vortex-file-compressed", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(
                    &vortex_compressed_ctx,
                    &sql_queries,
                    q,
                    Format::OnDiskVortex {
                        enable_compression: true,
                    },
                )
                .await;
            })
        });

        group.bench_function("vortex-file-uncompressed", |b| {
            b.to_async(&runtime).iter(|| async {
                run_tpch_query(
                    &vortex_uncompressed_ctx,
                    &sql_queries,
                    q,
                    Format::OnDiskVortex {
                        enable_compression: false,
                    },
                )
                .await;
            })
        });
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
