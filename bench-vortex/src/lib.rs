#![feature(exit_status_error)]

use std::env::temp_dir;
use std::fs::{create_dir_all, File};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use itertools::Itertools;
use lazy_static::lazy_static;
use log::LevelFilter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use vortex::aliases::hash_set::HashSet;
use vortex::array::ChunkedArray;
use vortex::arrow::FromArrowType;
use vortex::compress::CompressionStrategy;
use vortex::{Array, Context, IntoArray};
use vortex_dtype::DType;
use vortex_fastlanes::DeltaEncoding;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::alp_rd::ALPRDCompressor;
use vortex_sampling_compressor::compressors::bitpacked::BITPACK_WITH_PATCHES;
use vortex_sampling_compressor::compressors::date_time_parts::DateTimePartsCompressor;
use vortex_sampling_compressor::compressors::dict::DictCompressor;
use vortex_sampling_compressor::compressors::fsst::FSSTCompressor;
use vortex_sampling_compressor::compressors::r#for::FoRCompressor;
use vortex_sampling_compressor::compressors::roaring_bool::RoaringBoolCompressor;
use vortex_sampling_compressor::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use vortex_sampling_compressor::compressors::sparse::SparseCompressor;
use vortex_sampling_compressor::compressors::CompressorRef;
use vortex_sampling_compressor::SamplingCompressor;

use crate::data_downloads::FileType;
use crate::reader::BATCH_SIZE;
use crate::taxi_data::taxi_data_parquet;

pub mod data_downloads;
pub mod parquet_utils;
pub mod public_bi_data;
pub mod reader;
pub mod taxi_data;
pub mod tpch;
pub mod vortex_utils;

lazy_static! {
    pub static ref CTX: Arc<Context> = Arc::new(
        Context::default()
            .with_encodings(SamplingCompressor::default().used_encodings())
            .with_encoding(&DeltaEncoding)
    );
    pub static ref COMPRESSORS: HashSet<CompressorRef<'static>> = [
        &ALPCompressor as CompressorRef<'static>,
        &ALPRDCompressor,
        &DictCompressor,
        &BITPACK_WITH_PATCHES,
        &FoRCompressor,
        &FSSTCompressor,
        &DateTimePartsCompressor,
        &DEFAULT_RUN_END_COMPRESSOR,
        &RoaringBoolCompressor,
        &SparseCompressor
    ]
    .into();
}

/// Creates a file if it doesn't already exist.
/// NB: Does NOT modify the given path to ensure that it resides in the data directory.
pub fn idempotent<T, E, P: IdempotentPath + ?Sized>(
    path: &P,
    f: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<PathBuf, E> {
    let data_path = path.to_data_path();
    if !data_path.exists() {
        let temp_location = path.to_temp_path();
        let temp_path = temp_location.as_path();
        f(temp_path)?;
        std::fs::rename(temp_path, &data_path).unwrap();
    }
    Ok(data_path)
}

pub async fn idempotent_async<T, E, F, P>(
    path: &P,
    f: impl FnOnce(PathBuf) -> F,
) -> Result<PathBuf, E>
where
    F: Future<Output = Result<T, E>>,
    P: IdempotentPath + ?Sized,
{
    let data_path = path.to_data_path();
    if !data_path.exists() {
        let temp_location = path.to_temp_path();
        f(temp_location.clone()).await?;
        std::fs::rename(temp_location.as_path(), &data_path).unwrap();
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
    fn to_temp_path(&self) -> PathBuf;
}

impl IdempotentPath for str {
    fn to_data_path(&self) -> PathBuf {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join(self);
        if !path.parent().unwrap().exists() {
            create_dir_all(path.parent().unwrap()).unwrap();
        }
        path
    }

    fn to_temp_path(&self) -> PathBuf {
        let temp_dir = temp_dir().join(uuid::Uuid::new_v4().to_string());
        if !temp_dir.exists() {
            create_dir_all(temp_dir.clone()).unwrap();
        }
        temp_dir.join(self)
    }
}

impl IdempotentPath for PathBuf {
    fn to_data_path(&self) -> PathBuf {
        if !self.parent().unwrap().exists() {
            create_dir_all(self.parent().unwrap()).unwrap();
        }
        self.to_path_buf()
    }

    fn to_temp_path(&self) -> PathBuf {
        let temp_dir = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        if !temp_dir.exists() {
            create_dir_all(temp_dir.clone()).unwrap();
        }
        temp_dir.join(self.file_name().unwrap())
    }
}

pub fn setup_logger(level: LevelFilter) {
    TermLogger::init(
        level,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )
    .unwrap();
}

pub fn fetch_taxi_data() -> Array {
    let file = File::open(taxi_data_parquet()).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.with_batch_size(BATCH_SIZE).build().unwrap();

    let schema = reader.schema();
    ChunkedArray::try_new(
        reader
            .into_iter()
            .map(|batch_result| batch_result.unwrap())
            .map(Array::try_from)
            .map(Result::unwrap)
            .collect_vec(),
        DType::from_arrow(schema),
    )
    .unwrap()
    .into_array()
}

pub fn compress_taxi_data() -> Array {
    let uncompressed = fetch_taxi_data();
    let compressor: &dyn CompressionStrategy = &SamplingCompressor::new(COMPRESSORS.clone());

    compressor.compress(&uncompressed).unwrap()
}

pub struct CompressionRunStats {
    schema: DType,
    total_compressed_size: Option<u64>,
    compressed_sizes: Vec<u64>,
    file_type: FileType,
    file_name: String,
}

impl CompressionRunStats {
    pub fn to_results(&self, dataset_name: String) -> Vec<CompressionRunResults> {
        let DType::Struct(st, _) = &self.schema else {
            unreachable!()
        };

        self.compressed_sizes
            .iter()
            .zip_eq(st.names().iter().zip_eq(st.dtypes().iter()))
            .map(
                |(&size, (column_name, column_type))| CompressionRunResults {
                    dataset_name: dataset_name.clone(),
                    file_name: self.file_name.clone(),
                    file_type: self.file_type.to_string(),
                    column_name: (**column_name).to_string(),
                    column_type: column_type.to_string(),
                    compressed_size: size,
                    total_compressed_size: self.total_compressed_size,
                },
            )
            .collect::<Vec<_>>()
    }
}

pub struct CompressionRunResults {
    pub dataset_name: String,
    pub file_name: String,
    pub file_type: String,
    pub column_name: String,
    pub column_type: String,
    pub compressed_size: u64,
    pub total_compressed_size: Option<u64>,
}

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::ops::Deref;
    use std::sync::Arc;

    use arrow_array::{ArrayRef as ArrowArrayRef, StructArray as ArrowStructArray};
    use log::LevelFilter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use vortex::arrow::FromArrowArray;
    use vortex::compress::CompressionStrategy;
    use vortex::{Array, IntoCanonical};
    use vortex_sampling_compressor::SamplingCompressor;

    use crate::taxi_data::taxi_data_parquet;
    use crate::{compress_taxi_data, setup_logger, COMPRESSORS};

    #[ignore]
    #[test]
    fn compression_ratio() {
        setup_logger(LevelFilter::Debug);
        _ = compress_taxi_data();
    }

    #[ignore]
    #[test]
    fn round_trip_arrow() {
        let file = File::open(taxi_data_parquet()).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.with_limit(1).build().unwrap();

        for record_batch in reader.map(|batch_result| batch_result.unwrap()) {
            let struct_arrow: ArrowStructArray = record_batch.into();
            let arrow_array: ArrowArrayRef = Arc::new(struct_arrow);
            let vortex_array = Array::from_arrow(arrow_array.clone(), false);
            let vortex_as_arrow = vortex_array.into_canonical().unwrap().into_arrow().unwrap();
            assert_eq!(vortex_as_arrow.deref(), arrow_array.deref());
        }
    }

    // Ignoring since Struct arrays don't currently support equality.
    // https://github.com/apache/arrow-rs/issues/5199
    #[ignore]
    #[test]
    fn round_trip_arrow_compressed() {
        let file = File::open(taxi_data_parquet()).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.with_limit(1).build().unwrap();
        let compressor: &dyn CompressionStrategy = &SamplingCompressor::new(COMPRESSORS.clone());

        for record_batch in reader.map(|batch_result| batch_result.unwrap()) {
            let struct_arrow: ArrowStructArray = record_batch.into();
            let arrow_array: ArrowArrayRef = Arc::new(struct_arrow);
            let vortex_array = Array::from_arrow(arrow_array.clone(), false);

            let compressed = compressor.compress(&vortex_array).unwrap();
            let compressed_as_arrow = compressed.into_canonical().unwrap().into_arrow().unwrap();
            assert_eq!(compressed_as_arrow.deref(), arrow_array.deref());
        }
    }
}
