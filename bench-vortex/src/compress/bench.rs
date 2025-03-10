use std::borrow::Cow;
use std::cell::LazyCell;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use indicatif::ProgressBar;
use parquet::basic::{Compression, ZstdLevel};
use tokio::runtime::Runtime;
use vortex::arrays::ChunkedArray;
use vortex::nbytes::NBytes;
use vortex::{ArrayExt, ArrayRef};

use crate::Format;
use crate::bench_run::run;
use crate::compress::chunked_to_vec_record_batch;
use crate::compress::parquet::{parquet_compress_write, parquet_decompress_read};
use crate::compress::vortex::{vortex_compress_write, vortex_decompress_read};
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

pub fn benchmark_compress<F>(
    runtime: &Runtime,
    progress: &ProgressBar,
    formats: &[Format],
    iterations: usize,
    bench_name: &str,
    make_uncompressed: F,
) -> CompressMeasurements
where
    F: Fn() -> ArrayRef,
{
    let uncompressed = make_uncompressed();
    let uncompressed_size = uncompressed.nbytes();
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
                batches.clone(),
                schema.clone(),
                Compression::ZSTD(ZstdLevel::default()),
                &mut buf,
            );
            Bytes::from(buf)
        });

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
