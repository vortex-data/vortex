use std::sync::{Arc, LazyLock};

use arrow_array::builder::{StringBuilder, UInt32Builder};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use criterion::measurement::Measurement;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use datafusion::common::Result as DFResult;
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::execution::memory_pool::human_readable_size;
use datafusion::functions_aggregate::count::count_distinct;
use datafusion::logical_expr::lit;
use datafusion::prelude::{col, DataFrame, SessionContext};
use vortex::aliases::hash_set::HashSet;
use vortex::compress::CompressionStrategy;
use vortex::dict::DictEncoding;
use vortex::encoding::EncodingRef;
use vortex::fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex::sampling_compressor::compressors::bitpacked::BITPACK_WITH_PATCHES;
use vortex::sampling_compressor::compressors::delta::DeltaCompressor;
use vortex::sampling_compressor::compressors::dict::DictCompressor;
use vortex::sampling_compressor::compressors::r#for::FoRCompressor;
use vortex::sampling_compressor::compressors::CompressorRef;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::{Array, Context};
use vortex_datafusion::memory::{VortexMemTable, VortexMemTableOptions};

pub static CTX: LazyLock<Context> = LazyLock::new(|| {
    Context::default().with_encodings([
        &BitPackedEncoding as EncodingRef,
        &DictEncoding,
        &FoREncoding,
        &DeltaEncoding,
    ])
});

pub static COMPRESSORS: LazyLock<HashSet<CompressorRef<'static>>> = LazyLock::new(|| {
    [
        &BITPACK_WITH_PATCHES as CompressorRef<'static>,
        &DictCompressor,
        &FoRCompressor,
        &DeltaCompressor,
    ]
    .into()
});

fn toy_dataset_arrow() -> RecordBatch {
    // 64,000 rows of string and numeric data.
    // 8,000 values of first string, second string, third string, etc.

    let names = [
        "Alexander",
        "Anastasia",
        "Archibald",
        "Bartholomew",
        "Benjamin",
        "Christopher",
        "Elizabeth",
        "Gabriella",
    ];

    let mut col1 = StringBuilder::with_capacity(640_000, 64_000_000);
    let mut col2 = UInt32Builder::with_capacity(640_000);
    for i in 0..640_000 {
        col1.append_value(names[i % 8]);
        col2.append_value(u32::try_from(i).unwrap());
    }

    let col1 = col1.finish();
    let col2 = col2.finish();

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("names", DataType::Utf8, false),
            Field::new("scores", DataType::UInt32, false),
        ])),
        vec![Arc::new(col1), Arc::new(col2)],
    )
    .unwrap()
}

fn toy_dataset_vortex(compress: bool) -> Array {
    let uncompressed = toy_dataset_arrow().try_into().unwrap();

    if !compress {
        return uncompressed;
    }

    println!(
        "uncompressed size: {:?}",
        human_readable_size(uncompressed.nbytes())
    );
    let compressor: &dyn CompressionStrategy = &SamplingCompressor::new(COMPRESSORS.clone());
    let compressed = compressor.compress(&uncompressed).unwrap();
    println!(
        "vortex compressed size: {:?}",
        human_readable_size(compressed.nbytes())
    );
    compressed
}

fn filter_agg_query(df: DataFrame) -> DFResult<DataFrame> {
    // SELECT SUM(scores) FROM table WHERE scores >= 3000 AND scores <= 4000
    df.filter(col("scores").gt_eq(lit(3_000)))?
        .filter(col("scores").lt_eq(lit(4_000)))?
        .aggregate(vec![], vec![count_distinct(col("names"))])
}

fn measure_provider<M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    session: &SessionContext,
    table: Arc<dyn TableProvider>,
) {
    group.bench_function("planning", |b| {
        b.to_async(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        )
        .iter(|| async {
            // Force physical planner to execute on our TableProvider.
            filter_agg_query(black_box(session).read_table(table.clone()).unwrap())
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
        });
    });

    group.bench_function("exec", |b| {
        b.to_async(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        )
        .iter(|| async {
            // Force full query execution with .collect()
            filter_agg_query(black_box(session).read_table(table.clone()).unwrap())
                .unwrap()
                .collect()
                .await
                .unwrap();
        });
    });
}

fn bench_arrow<M: Measurement>(mut group: BenchmarkGroup<M>, session: &SessionContext) {
    let arrow_dataset = toy_dataset_arrow();
    let arrow_table =
        Arc::new(MemTable::try_new(arrow_dataset.schema(), vec![vec![arrow_dataset]]).unwrap());

    measure_provider(&mut group, session, arrow_table);
}

fn bench_vortex<M: Measurement>(
    mut group: BenchmarkGroup<M>,
    session: &SessionContext,
    enable_pushdown: bool,
    compress: bool,
) {
    let vortex_dataset = toy_dataset_vortex(compress);
    let vortex_table = Arc::new(VortexMemTable::new(
        vortex_dataset,
        VortexMemTableOptions::default().with_pushdown(enable_pushdown),
    ));

    measure_provider(&mut group, session, vortex_table);
}

fn bench_datafusion(c: &mut Criterion) {
    bench_arrow(c.benchmark_group("arrow"), &SessionContext::new());

    // compress=true, pushdown enabled
    bench_vortex(
        c.benchmark_group("vortex-pushdown-compressed"),
        &SessionContext::new(),
        true,
        true,
    );

    // compress=false, pushdown enabled
    bench_vortex(
        c.benchmark_group("vortex-pushdown-uncompressed"),
        &SessionContext::new(),
        true,
        false,
    );

    // compress=true, pushdown disabled
    bench_vortex(
        c.benchmark_group("vortex-nopushdown-compressed"),
        &SessionContext::new(),
        false,
        true,
    );

    // compress=false, pushdown disabled
    bench_vortex(
        c.benchmark_group("vortex-nopushdown-uncompressed"),
        &SessionContext::new(),
        false,
        false,
    );
}

criterion_group!(benches, bench_datafusion);
criterion_main!(benches);
