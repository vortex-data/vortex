mod tokio_runtime;

use core::cell::LazyCell;
use core::str::FromStr;
use core::sync::atomic::{AtomicBool, Ordering};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::{env, fs};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Schema;
use bench_vortex::data_downloads::BenchmarkDataset;
use bench_vortex::public_bi_data::BenchmarkDatasets;
use bench_vortex::public_bi_data::PBIDataset::*;
use bench_vortex::taxi_data::taxi_data_parquet;
use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::{fetch_taxi_data, tpch};
use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use futures::StreamExt;
use log::LevelFilter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use regex::Regex;
use simplelog::*;
use tokio::runtime::Runtime;
use vortex::array::{ChunkedArray, StructArray};
use vortex::dtype::field::Field;
use vortex::error::VortexResult;
use vortex::file::{LayoutContext, LayoutDeserializer, VortexFileWriter, VortexReadBuilder};
use vortex::sampling_compressor::compressors::fsst::FSSTCompressor;
use vortex::sampling_compressor::{SamplingCompressor, ALL_ENCODINGS_CONTEXT};
use vortex::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

use crate::tokio_runtime::TOKIO_RUNTIME;

#[derive(serde::Serialize)]
struct GenericBenchmarkResults<'a> {
    name: &'a str,
    value: f64,
    unit: &'a str,
    range: f64,
}

static LOG_INITIALIZED: AtomicBool = AtomicBool::new(false);

fn ensure_dir_exists(dir: &str) -> std::io::Result<()> {
    let path = Path::new(dir);
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn chunked_to_vec_record_batch(chunked: ChunkedArray) -> (Vec<RecordBatch>, Arc<Schema>) {
    let chunks_vec = chunked.chunks().collect::<Vec<_>>();

    if chunks_vec.is_empty() {
        panic!("empty chunks");
    }

    let batches = chunks_vec
        .iter()
        .map(|x| RecordBatch::try_from(x.clone()).unwrap())
        .collect::<Vec<_>>();
    let schema = batches[0].schema();
    (batches, schema)
}

#[inline(never)]
fn parquet_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    compression: Compression,
    buf: &mut Vec<u8>,
) -> usize {
    let mut buf = Cursor::new(buf);
    let writer_properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(writer_properties)).unwrap();
    for batch in batches {
        writer.write(&batch).unwrap();
    }
    writer.flush().unwrap();
    let n_bytes = writer.bytes_written();
    writer.close().unwrap();
    n_bytes
}

#[inline(never)]
fn parquet_decompress_read(buf: bytes::Bytes) -> usize {
    let builder = ParquetRecordBatchReaderBuilder::try_new(buf).unwrap();
    let reader = builder.build().unwrap();
    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch.unwrap().get_array_memory_size()
    }
    nbytes
}

fn parquet_compressed_written_size(array: &ArrayData, compression: Compression) -> usize {
    let chunked = ChunkedArray::maybe_from(array).unwrap();
    let (batches, schema) = chunked_to_vec_record_batch(chunked);
    parquet_compress_write(batches, schema, compression, &mut Vec::new())
}

#[inline(never)]
fn vortex_compress_write(
    runtime: &Runtime,
    compressor: &SamplingCompressor<'_>,
    array: &ArrayData,
    buf: &mut Vec<u8>,
) -> VortexResult<u64> {
    async fn async_write(array: &ArrayData, cursor: &mut Cursor<&mut Vec<u8>>) -> VortexResult<()> {
        let mut writer = VortexFileWriter::new(cursor);

        writer = writer.write_array_columns(array.clone()).await?;
        writer.finalize().await?;
        Ok(())
    }

    let compressed = compressor.compress(array, None)?.into_array();
    let mut cursor = Cursor::new(buf);

    runtime.block_on(async_write(&compressed, &mut cursor))?;

    Ok(cursor.position())
}

#[inline(never)]
fn vortex_decompress_read(runtime: &Runtime, buf: Bytes) -> VortexResult<Vec<ArrayRef>> {
    async fn async_read(buf: Bytes) -> VortexResult<Vec<ArrayRef>> {
        let builder: VortexReadBuilder<_> = VortexReadBuilder::new(
            buf,
            LayoutDeserializer::new(
                ALL_ENCODINGS_CONTEXT.clone(),
                LayoutContext::default().into(),
            ),
        );

        let mut batches = vec![];
        let mut stream = builder.build().await?;
        while let Some(batch) = stream.next().await {
            batches.push(batch?.into_arrow()?);
        }
        Ok(batches)
    }

    runtime.block_on(async_read(buf))
}

fn vortex_compressed_written_size(
    runtime: &Runtime,
    compressor: &SamplingCompressor<'_>,
    array: &ArrayData,
) -> VortexResult<u64> {
    vortex_compress_write(runtime, compressor, array, &mut Vec::new())
}

fn benchmark_compress<F, U>(
    c: &mut Criterion,
    compressor: &SamplingCompressor<'_>,
    make_uncompressed: F,
    sample_size: usize,
    measurement_time: Option<Duration>,
    bench_name: &str,
) where
    F: Fn() -> U,
    U: AsRef<ArrayData>,
{
    // if no logging is enabled, enable it
    if !LOG_INITIALIZED.swap(true, Ordering::SeqCst) {
        TermLogger::init(
            env::var("RUST_LOG")
                .ok()
                .and_then(|s| LevelFilter::from_str(&s).ok())
                .unwrap_or(LevelFilter::Off),
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )
        .unwrap();
    }

    ensure_dir_exists("benchmarked-files").unwrap();
    let runtime = &TOKIO_RUNTIME;
    let uncompressed = make_uncompressed();
    let uncompressed_size = uncompressed.as_ref().nbytes();
    let mut compressed_size = 0;

    {
        let mut group = c.benchmark_group("compress time");
        group.sample_size(sample_size);
        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        measurement_time.map(|t| group.measurement_time(t));
        group.bench_function(bench_name, |b| {
            b.iter_with_large_drop(|| {
                compressed_size = black_box(
                    vortex_compressed_written_size(runtime, compressor, uncompressed.as_ref())
                        .unwrap(),
                );
            });
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("parquet_rs-zstd compress time");
        group.sample_size(sample_size);
        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        measurement_time.map(|t| group.measurement_time(t));

        group.bench_function(bench_name, |b| {
            let chunked = ChunkedArray::try_from(uncompressed.as_ref().clone()).unwrap();
            let (batches, schema) = chunked_to_vec_record_batch(chunked);

            b.iter_with_large_drop(|| {
                black_box(parquet_compress_write(
                    batches.clone(),
                    schema.clone(),
                    Compression::ZSTD(ZstdLevel::default()),
                    &mut Vec::new(),
                ));
            });
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("decompress time");
        group.sample_size(sample_size);
        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        measurement_time.map(|t| group.measurement_time(t));

        let buffer = LazyCell::new(|| {
            let mut buf = Vec::new();
            vortex_compress_write(runtime, compressor, uncompressed.as_ref(), &mut buf).unwrap();
            Bytes::from(buf)
        });

        group.bench_function(bench_name, |b| {
            b.iter_with_large_drop(|| {
                black_box(vortex_decompress_read(runtime, buffer.clone()).unwrap());
            });
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("parquet_rs-zstd decompress time");
        group.sample_size(sample_size);
        group.throughput(Throughput::Bytes(uncompressed_size as u64));
        measurement_time.map(|t| group.measurement_time(t));

        let buffer = LazyCell::new(|| {
            let chunked = ChunkedArray::try_from(uncompressed.as_ref().clone()).unwrap();
            let (batches, schema) = chunked_to_vec_record_batch(chunked);
            let mut buf = Vec::new();
            parquet_compress_write(
                batches.clone(),
                schema.clone(),
                Compression::ZSTD(ZstdLevel::default()),
                &mut buf,
            );
            bytes::Bytes::from(buf)
        });

        group.bench_function(bench_name, |b| {
            b.iter_with_large_drop(|| {
                black_box(parquet_decompress_read(buffer.clone()));
            });
        });
        group.finish();
    }

    if env::var("BENCH_VORTEX_RATIOS")
        .ok()
        .map(|x| Regex::new(&x).unwrap().is_match(bench_name))
        .unwrap_or(false)
    {
        let vortex_nbytes =
            vortex_compressed_written_size(runtime, compressor, uncompressed.as_ref()).unwrap();

        let parquet_zstd_nbytes = parquet_compressed_written_size(
            uncompressed.as_ref(),
            Compression::ZSTD(ZstdLevel::default()),
        );

        eprintln!(
            "{}",
            serde_json::to_string(&GenericBenchmarkResults {
                name: &format!("vortex:parquet-zstd size/{}", bench_name),
                value: (vortex_nbytes as f64) / (parquet_zstd_nbytes as f64),
                unit: "ratio",
                range: 0.0,
            })
            .unwrap()
        );

        eprintln!(
            "{}",
            serde_json::to_string(&GenericBenchmarkResults {
                name: &format!("vortex:raw size/{}", bench_name),
                value: (compressed_size as f64) / (uncompressed_size as f64),
                unit: "ratio",
                range: 0.0,
            })
            .unwrap()
        );

        eprintln!(
            "{}",
            serde_json::to_string(&GenericBenchmarkResults {
                name: &format!("vortex size/{}", bench_name),
                value: compressed_size as f64,
                unit: "bytes",
                range: 0.0,
            })
            .unwrap()
        );
    }
}

fn yellow_taxi_trip_data(c: &mut Criterion) {
    taxi_data_parquet();
    benchmark_compress(
        c,
        &SamplingCompressor::default(),
        fetch_taxi_data,
        10,
        None,
        "taxi",
    );
}

fn public_bi_benchmark(c: &mut Criterion) {
    for dataset_handle in [
        AirlineSentiment,
        Arade,
        Bimbo,
        CMSprovider,
        // Corporations, // duckdb thinks ' is a quote character but its used as an apostrophe
        // CityMaxCapita, // 11th column has F, M, and U but is inferred as boolean
        Euro2016,
        Food,
        HashTags,
        // Hatred, // panic in fsst_compress_iter
        // TableroSistemaPenal, // thread 'main' panicked at bench-vortex/benches/compress_benchmark.rs:224:42: called `Result::unwrap()` on an `Err` value: expected type: {column00=utf8?, column01=i64?, column02=utf8?, column03=f64?, column04=i64?, column05=utf8?, column06=utf8?, column07=utf8?, column08=utf8?, column09=utf8?, column10=i64?, column11=i64?, column12=utf8?, column13=utf8?, column14=i64?, column15=i64?, column16=utf8?, column17=utf8?, column18=utf8?, column19=utf8?, column20=i64?, column21=utf8?, column22=utf8?, column23=utf8?, column24=utf8?, column25=i64?, column26=utf8?} but instead got {column00=utf8?, column01=i64?, column02=i64?, column03=i64?, column04=i64?, column05=utf8?, column06=i64?, column07=i64?, column08=i64?, column09=utf8?, column10=ext(vortex.date, ExtMetadata([4]))?, column11=ext(vortex.date, ExtMetadata([4]))?, column12=utf8?, column13=utf8?, column14=utf8?, column15=i64?, column16=i64?, column17=utf8?, column18=utf8?, column19=utf8?, column20=utf8?, column21=utf8?}
        // YaleLanguages, // 4th column looks like integer but also contains Y
    ] {
        let dataset = BenchmarkDatasets::PBI(dataset_handle);

        benchmark_compress(
            c,
            &SamplingCompressor::default(),
            || dataset.to_vortex_array().unwrap(),
            10,
            None,
            dataset_handle.dataset_name(),
        );
    }
}

fn tpc_h_l_comment(c: &mut Criterion) {
    let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();
    let rt = &TOKIO_RUNTIME;
    let lineitem_vortex = rt.block_on(tpch::load_table(
        data_dir,
        "lineitem",
        &tpch::schema::LINEITEM,
    ));

    let compressor = SamplingCompressor::default().excluding(&FSSTCompressor);
    let compressor_fsst = SamplingCompressor::default();

    let comment_chunks = ChunkedArray::try_from(lineitem_vortex)
        .unwrap()
        .chunks()
        .map(|chunk| {
            StructArray::try_from(chunk)
                .unwrap()
                .project(&[Field::from("l_comment")])
                .unwrap()
                .into_array()
        })
        .collect::<Vec<_>>();
    let comment_dtype = comment_chunks[0].dtype().clone();
    let comments = ChunkedArray::try_new(comment_chunks, comment_dtype)
        .unwrap()
        .into_array();

    benchmark_compress(
        c,
        &compressor,
        || &comments,
        10,
        None,
        "TPC-H l_comment chunked without fsst",
    );

    benchmark_compress(
        c,
        &compressor_fsst,
        || &comments,
        10,
        None,
        "TPC-H l_comment chunked",
    );

    let comments_canonical = comments
        .into_canonical()
        .unwrap()
        .into_struct()
        .unwrap()
        .into_array();
    let dtype = comments_canonical.dtype().clone();
    let comments_canonical_chunked =
        ChunkedArray::try_new(vec![comments_canonical], dtype).unwrap();

    benchmark_compress(
        c,
        &compressor_fsst,
        || &comments_canonical_chunked,
        10,
        Some(Duration::new(15, 0)),
        "TPC-H l_comment canonical",
    );
}

criterion_group!(
    benches,
    yellow_taxi_trip_data,
    public_bi_benchmark,
    tpc_h_l_comment,
);
criterion_main!(benches);
