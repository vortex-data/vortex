use bench_vortex::compress::bench::{CompressMeasurements, benchmark_compress};
use bench_vortex::datasets::BenchmarkDataset;
use bench_vortex::datasets::public_bi_data::PBIDataset::{
    AirlineSentiment, Arade, Bimbo, CMSprovider, Euro2016, Food, HashTags,
};
use bench_vortex::datasets::struct_list_of_ints::StructListOfInts;
use bench_vortex::datasets::taxi_data::TaxiData;
use bench_vortex::datasets::tpch_l_comment::{TPCHLCommentCanonical, TPCHLCommentChunked};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::{Format, default_env_filter, feature_flagged_allocator, setup_logger};
use clap::Parser;
use indicatif::ProgressBar;
use regex::Regex;
use tokio::runtime::{Builder, Runtime};
use vortex::arrays::ChunkedArray;
use vortex::builders::builder_with_capacity;
use vortex::{Array, ArrayExt};

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(long)]
    datasets: Option<String>,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
}

fn main() {
    let args = Args::parse();

    let filter = default_env_filter(args.verbose);
    setup_logger(filter);

    let runtime = match args.threads {
        Some(0) => panic!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .expect("Failed building the Runtime");

    compress(
        runtime,
        args.iterations,
        args.datasets.map(|d| Regex::new(&d).unwrap()),
        args.formats,
        args.display_format,
    );
}

fn compress(
    runtime: Runtime,
    iterations: usize,
    datasets_filter: Option<Regex>,
    formats: Vec<Format>,
    display_format: DisplayFormat,
) {
    let structlistofints = vec![
        StructListOfInts::new(10, 1000, 1),
        StructListOfInts::new(100, 1000, 1),
        StructListOfInts::new(1000, 1000, 1),
        StructListOfInts::new(10, 1000, 50),
        StructListOfInts::new(100, 1000, 50),
        StructListOfInts::new(1000, 1000, 50),
    ];
    let datasets: Vec<&dyn BenchmarkDataset> = [
        &TaxiData as &dyn BenchmarkDataset,
        &AirlineSentiment,
        &Arade,
        &Bimbo,
        &CMSprovider,
        // Corporations, // duckdb thinks ' is a quote character but its used as an apostrophe
        // CityMaxCapita, // 11th column has F, M, and U but is inferred as boolean
        &Euro2016,
        &Food,
        &HashTags,
        // Hatred, // panic in fsst_compress_iter
        // TableroSistemaPenal, // thread 'main' panicked at bench-vortex/benches/compress_benchmark.rs:224:42: called `Result::unwrap()` on an `Err` value: expected type: {column00=utf8?, column01=i64?, column02=utf8?, column03=f64?, column04=i64?, column05=utf8?, column06=utf8?, column07=utf8?, column08=utf8?, column09=utf8?, column10=i64?, column11=i64?, column12=utf8?, column13=utf8?, column14=i64?, column15=i64?, column16=utf8?, column17=utf8?, column18=utf8?, column19=utf8?, column20=i64?, column21=utf8?, column22=utf8?, column23=utf8?, column24=utf8?, column25=i64?, column26=utf8?} but instead got {column00=utf8?, column01=i64?, column02=i64?, column03=i64?, column04=i64?, column05=utf8?, column06=i64?, column07=i64?, column08=i64?, column09=utf8?, column10=ext(vortex.date, ExtMetadata([4]))?, column11=ext(vortex.date, ExtMetadata([4]))?, column12=utf8?, column13=utf8?, column14=utf8?, column15=i64?, column16=i64?, column17=utf8?, column18=utf8?, column19=utf8?, column20=utf8?, column21=utf8?}
        // YaleLanguages, // 4th column looks like integer but also contains Y
        &TPCHLCommentChunked,
        &TPCHLCommentCanonical,
    ]
    .into_iter()
    .chain(structlistofints.iter().map(|d| d as &dyn BenchmarkDataset))
    .filter(|d| {
        datasets_filter.is_none()
            || datasets_filter
                .as_ref()
                .is_some_and(|ds| ds.is_match(d.name()))
    })
    .collect();

    let progress = ProgressBar::new((datasets.len() * formats.len() * 2) as u64);

    let measurements = datasets
        .into_iter()
        .map(|dataset_handle| {
            benchmark_compress(
                &runtime,
                &progress,
                &formats,
                iterations,
                dataset_handle.name(),
                || {
                    let vx_array =
                        runtime.block_on(async { dataset_handle.to_vortex_array().await });
                    ChunkedArray::from_iter(vx_array.as_::<ChunkedArray>().chunks().iter().map(
                        |chunk| {
                            let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());
                            chunk.append_to_builder(builder.as_mut()).unwrap();
                            builder.finish()
                        },
                    ))
                    .into_array()
                },
            )
        })
        .collect::<CompressMeasurements>();

    progress.finish();

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements.throughputs, &formats, RatioMode::Throughput).unwrap();
            render_table(
                measurements.ratios,
                if formats.contains(&Format::OnDiskVortex) {
                    &[Format::OnDiskVortex]
                } else {
                    &[]
                },
                RatioMode::Throughput,
            )
            .unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements.throughputs).unwrap();
            print_measurements_json(measurements.ratios).unwrap();
        }
    }
}
