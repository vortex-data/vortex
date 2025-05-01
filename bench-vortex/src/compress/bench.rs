use std::borrow::Cow;
use std::cell::LazyCell;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use indicatif::ProgressBar;
use parquet::basic::{Compression, ZstdLevel};
use tokio::runtime::Runtime;
use vortex::arrays::ChunkedArray;
use vortex::builders::builder_with_capacity;
use vortex::error::VortexUnwrap;
use vortex::{Array, ArrayExt};

use crate::Format;
use crate::bench_run::run;
use crate::compress::chunked_to_vec_record_batch;
use crate::compress::parquet::{parquet_compress_write, parquet_decompress_read};
use crate::compress::vortex::{vortex_compress_write, vortex_decompress_read};
use crate::datasets::Dataset;
use crate::measurements::{CompressionTimingMeasurement, CustomUnitMeasurement};

#[derive(Default)]
pub struct CompressMeasurements {
    pub timings: Vec<CompressionTimingMeasurement>,
    pub ratios: Vec<CustomUnitMeasurement>,
}

impl Extend<CompressMeasurements> for CompressMeasurements {
    fn extend<T: IntoIterator<Item = CompressMeasurements>>(&mut self, iter: T) {
        iter.into_iter().for_each(|measurement| {
            self.timings.extend(measurement.timings);
            self.ratios.extend(measurement.ratios);
        })
    }
}

impl FromIterator<CompressMeasurements> for CompressMeasurements {
    fn from_iter<T: IntoIterator<Item = CompressMeasurements>>(iter: T) -> Self {
        let mut into_iter = iter.into_iter();
        match into_iter.next() {
            None => CompressMeasurements::default(),
            Some(mut ms) => {
                ms.extend(into_iter);
                ms
            }
        }
    }
}

pub fn benchmark_compress(
    runtime: &Runtime,
    progress: &ProgressBar,
    formats: &[Format],
    iterations: usize,
    dataset_handle: &dyn Dataset,
) -> CompressMeasurements {
    let bench_name = dataset_handle.name();
    tracing::info!("Running {bench_name} benchmark");

    let vx_array = runtime.block_on(async { dataset_handle.to_vortex_array().await });
    let uncompressed =
        ChunkedArray::from_iter(vx_array.as_::<ChunkedArray>().chunks().iter().map(|chunk| {
            let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());
            chunk.append_to_builder(builder.as_mut()).vortex_unwrap();
            builder.finish()
        }))
        .into_array();

    let compressed_size = AtomicU64::default();

    let mut ratios = Vec::new();
    let mut timings = Vec::new();
    let mut vortex_compress_time = None;
    let mut vortex_decompress_time = None;
    let mut parquet_compress_time = None;
    let mut parquet_decompress_time = None;

    if formats.contains(&Format::OnDiskVortex) {
        let time = run(runtime, iterations, || async {
            compressed_size.store(
                vortex_compress_write(&uncompressed, &mut Vec::new())
                    .await
                    .unwrap(),
                Ordering::SeqCst,
            );
        });
        vortex_compress_time = Some(time);
        timings.push(CompressionTimingMeasurement {
            name: format!("compress time/{}", bench_name),
            time,
            format: Format::OnDiskVortex,
        });
        progress.inc(1);

        let compressed_size_f64 = compressed_size.load(Ordering::SeqCst) as f64;
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex size/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("bytes"),
            value: compressed_size_f64,
        });
    }

    if formats.contains(&Format::Parquet) {
        let parquet_compressed_size = AtomicU64::default();
        let chunked = uncompressed.as_::<ChunkedArray>().clone();
        let (batches, schema) = chunked_to_vec_record_batch(chunked);
        let time = run(runtime, iterations, || async {
            parquet_compressed_size.store(
                parquet_compress_write(
                    batches.clone(),
                    schema.clone(),
                    Compression::ZSTD(ZstdLevel::default()),
                    &mut Vec::new(),
                ) as u64,
                Ordering::SeqCst,
            );
        });
        parquet_compress_time = Some(time);
        timings.push(CompressionTimingMeasurement {
            name: format!("compress time/{}", bench_name),
            time,
            format: Format::Parquet,
        });

        progress.inc(1);
        let parquet_compressed_size = parquet_compressed_size.into_inner();
        ratios.push(CustomUnitMeasurement {
            name: format!("parquet-zstd size/{}", bench_name),
            // unlike timings, ratios have a single column vortex
            format: Format::OnDiskVortex,
            unit: Cow::from("bytes"),
            value: parquet_compressed_size as f64,
        });
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd size/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: compressed_size.load(Ordering::SeqCst) as f64 / parquet_compressed_size as f64,
        });
    }

    if formats.contains(&Format::OnDiskVortex) {
        let buffer = LazyCell::new(|| {
            let mut buf = Vec::new();
            runtime
                .block_on(vortex_compress_write(&uncompressed, &mut buf))
                .unwrap();
            Bytes::from(buf)
        });
        // Force materialization of the lazy cell so it's not invoked from within the async benchmark function
        LazyCell::force(&buffer);

        let time = run(runtime, iterations, || async {
            vortex_decompress_read(buffer.clone()).await.unwrap()
        });
        vortex_decompress_time = Some(time);
        timings.push(CompressionTimingMeasurement {
            name: format!("decompress time/{}", bench_name),
            time,
            format: Format::OnDiskVortex,
        });
        progress.inc(1);
    }

    if formats.contains(&Format::Parquet) {
        let buffer = LazyCell::new(|| {
            let chunked = uncompressed.as_::<ChunkedArray>().clone();
            let (batches, schema) = chunked_to_vec_record_batch(chunked);
            let mut buf = Vec::new();
            parquet_compress_write(
                batches,
                schema,
                Compression::ZSTD(ZstdLevel::default()),
                &mut buf,
            );
            Bytes::from(buf)
        });
        LazyCell::force(&buffer);

        let time = run(runtime, iterations, || async {
            parquet_decompress_read(buffer.clone());
        });
        parquet_decompress_time = Some(time);
        timings.push(CompressionTimingMeasurement {
            name: format!("decompress time/{}", bench_name),
            time,
            format: Format::Parquet,
        });
        progress.inc(1);
    }

    if let Some((vortex, parquet)) = vortex_compress_time.zip(parquet_compress_time) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd ratio compress time/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex.as_nanos() as f64 / parquet.as_nanos() as f64,
        });
    }

    if let Some((vortex, parquet)) = vortex_decompress_time.zip(parquet_decompress_time) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd ratio decompress time/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex.as_nanos() as f64 / parquet.as_nanos() as f64,
        });
    }

    CompressMeasurements { timings, ratios }
}
