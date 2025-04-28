use std::borrow::Cow;
use std::cell::LazyCell;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use indicatif::ProgressBar;
use parquet::basic::{Compression, ZstdLevel};
use rust_lapper::{Interval, Lapper};
use tokio::runtime::Runtime;
use vortex::arrays::ChunkedArray;
use vortex::builders::builder_with_capacity;
use vortex::error::VortexUnwrap;
use vortex::{Array, ArrayExt, ArrayVisitorExt};

use crate::Format;
use crate::bench_run::run;
use crate::compress::chunked_to_vec_record_batch;
use crate::compress::parquet::{parquet_compress_write, parquet_decompress_read};
use crate::compress::vortex::{vortex_compress_write, vortex_decompress_read};
use crate::datasets::Dataset;
use crate::measurements::{CustomUnitMeasurement, ThroughputMeasurement};

#[derive(Default)]
pub struct CompressMeasurements {
    pub throughputs: Vec<ThroughputMeasurement>,
    pub ratios: Vec<CustomUnitMeasurement>,
}

impl Extend<CompressMeasurements> for CompressMeasurements {
    fn extend<T: IntoIterator<Item = CompressMeasurements>>(&mut self, iter: T) {
        iter.into_iter().for_each(|measurement| {
            self.throughputs.extend(measurement.throughputs);
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

    let uncompressed_size = uncompressed_bytes(&vx_array);
    let compressed_size = AtomicU64::default();

    let mut ratios = Vec::new();
    let mut throughputs = Vec::new();

    if formats.contains(&Format::OnDiskVortex) {
        throughputs.push(ThroughputMeasurement {
            name: format!("compress time/{}", bench_name),
            bytes: uncompressed_size as u64,
            time: run(runtime, iterations, || async {
                compressed_size.store(
                    vortex_compress_write(&uncompressed, &mut Vec::new())
                        .await
                        .unwrap(),
                    Ordering::SeqCst,
                );
            }),
            format: Format::OnDiskVortex,
        });
        progress.inc(1);

        let compressed_size_f64 = compressed_size.load(Ordering::SeqCst) as f64;
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:raw size/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: compressed_size_f64 / (uncompressed_size as f64),
        });
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
        throughputs.push(ThroughputMeasurement {
            name: format!("compress time/{}", bench_name),
            bytes: uncompressed_size as u64,
            time: run(runtime, iterations, || async {
                parquet_compressed_size.store(
                    parquet_compress_write(
                        batches.clone(),
                        schema.clone(),
                        Compression::ZSTD(ZstdLevel::default()),
                        &mut Vec::new(),
                    ) as u64,
                    Ordering::SeqCst,
                );
            }),
            format: Format::Parquet,
        });

        progress.inc(1);
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd size/{}", bench_name),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: compressed_size.load(Ordering::SeqCst) as f64
                / parquet_compressed_size.into_inner() as f64,
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

        throughputs.push(ThroughputMeasurement {
            name: format!("decompress time/{}", bench_name),
            bytes: uncompressed_size as u64,
            time: run(runtime, iterations, || async {
                vortex_decompress_read(buffer.clone()).await.unwrap()
            }),
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

        throughputs.push(ThroughputMeasurement {
            name: format!("decompress time/{}", bench_name),
            bytes: uncompressed_size as u64,
            time: run(runtime, iterations, || async {
                parquet_decompress_read(buffer.clone());
            }),
            format: Format::Parquet,
        });
        progress.inc(1);
    }

    CompressMeasurements {
        throughputs,
        ratios,
    }
}

// Count total bytes all buffers use for the array, only
// counting unique memory regions once.
fn uncompressed_bytes(vx_array: &dyn Array) -> usize {
    let mut intervals = Vec::new();

    for array in vx_array.depth_first_traversal() {
        for buffer in array.buffers() {
            let slice: &[u8] = buffer.inner().as_ref();
            let start = slice.as_ptr() as usize;
            intervals.push(Interval {
                start,
                stop: start + slice.len(),
                val: true,
            });
        }
    }
    Lapper::new(intervals).cov()
}
