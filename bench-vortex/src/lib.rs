#![feature(exit_status_error)]

use std::clone::Clone;
use std::env::temp_dir;
use std::fs::{create_dir_all, File};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use arrow_array::{RecordBatch, RecordBatchReader};
use blob::SlowObjectStoreRegistry;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::cache_unit::{DefaultFileStatisticsCache, DefaultListFilesCache};
use datafusion::execution::object_store::DefaultObjectStoreRegistry;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_physical_plan::{collect, ExecutionPlan};
use itertools::Itertools;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::{Rng, SeedableRng as _};
use serde::Serialize;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use vortex::array::{ChunkedArray, ListArray, PrimitiveArray, StructArray};
use vortex::arrow::FromArrowType;
use vortex::compress::CompressionStrategy;
use vortex::dtype::{DType, Nullability, PType, StructDType};
use vortex::encodings::fastlanes::DeltaEncoding;
use vortex::error::VortexResult;
use vortex::sampling_compressor::{SamplingCompressor, ALL_ENCODINGS_CONTEXT};
use vortex::validity::Validity;
use vortex::{Array, ContextRef, IntoArray};

use crate::data_downloads::FileType;
use crate::reader::BATCH_SIZE;
use crate::taxi_data::taxi_data_parquet;

pub mod blob;
pub mod clickbench;
pub mod data_downloads;
pub mod display;
pub mod parquet_utils;
pub mod public_bi_data;
pub mod reader;
pub mod taxi_data;
pub mod tpch;
pub mod vortex_utils;

#[macro_export]
macro_rules! feature_flagged_allocator {
    () => {
        cfg_if::cfg_if! {
            if #[cfg(feature = "mimalloc")] {
                #[global_allocator]
                static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
            } else if #[cfg(feature = "jemalloc")] {
                #[global_allocator]
                static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
            }
        }
    };
}

pub static CTX: LazyLock<ContextRef> = LazyLock::new(|| {
    Arc::new(
        (*(ALL_ENCODINGS_CONTEXT.clone()))
            .clone()
            .with_encoding(DeltaEncoding::vtable()),
    )
});

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Format {
    Csv,
    Arrow,
    Parquet,
    InMemoryVortex,
    OnDiskVortex,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Csv => write!(f, "csv"),
            Format::Arrow => write!(f, "arrow"),
            Format::Parquet => write!(f, "parquet"),
            Format::InMemoryVortex => {
                write!(f, "in_memory_vortex")
            }
            Format::OnDiskVortex => {
                write!(f, "on_disk_vortex(compressed=true)")
            }
        }
    }
}

impl Format {
    pub fn name(&self) -> String {
        match self {
            Format::Csv => "csv".to_string(),
            Format::Arrow => "arrow".to_string(),
            Format::Parquet => "parquet".to_string(),
            Format::InMemoryVortex => "vortex-in-memory".to_string(),
            Format::OnDiskVortex => "vortex-file-compressed".to_string(),
        }
    }
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

pub fn setup_logger(_filter: EnvFilter) {
    // #[cfg(not(feature = "tracing"))]
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_file(true)
        .with_level(true)
        .with_line_number(true)
        .with_env_filter(_filter)
        .init();
}

pub fn default_env_filter(is_verbose: bool) -> EnvFilter {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_e) => {
            let default_level = if is_verbose {
                LevelFilter::TRACE
            } else {
                LevelFilter::INFO
            };

            EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy()
        }
    }
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
    CompressionStrategy::compress(&SamplingCompressor::default(), &fetch_taxi_data()).unwrap()
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
            .zip_eq(st.names().iter().zip_eq(st.fields()))
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

pub async fn execute_query(ctx: &SessionContext, query: &str) -> VortexResult<Vec<RecordBatch>> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    let physical_plan = state.create_physical_plan(&plan).await?;
    let result = collect(physical_plan.clone(), state.task_ctx()).await?;
    Ok(result)
}

pub async fn physical_plan(
    ctx: &SessionContext,
    query: &str,
) -> anyhow::Result<Arc<dyn ExecutionPlan>> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    Ok(state.create_physical_plan(&plan).await?)
}

#[derive(Clone, Debug)]
pub struct Measurement {
    pub query_idx: usize,
    pub time: Duration,
    pub format: Format,
    pub dataset: String,
}

#[derive(Serialize)]
pub struct JsonValue {
    pub name: String,
    pub unit: String,
    pub value: u128,
    pub commit_id: String,
}

pub static GIT_COMMIT_ID: LazyLock<String> = LazyLock::new(|| {
    String::from_utf8(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_string()
});

impl Measurement {
    pub fn to_json(&self) -> JsonValue {
        let name = format!(
            "{dataset}_q{query_idx:02}/{format}",
            dataset = self.dataset,
            format = self.format.name(),
            query_idx = self.query_idx
        );

        JsonValue {
            name,
            unit: "ns".to_string(),
            value: self.time.as_nanos(),
            commit_id: GIT_COMMIT_ID.to_string(),
        }
    }
}

pub fn get_session_with_cache(emulate_object_store: bool) -> SessionContext {
    let registry = if emulate_object_store {
        Arc::new(SlowObjectStoreRegistry::default()) as _
    } else {
        Arc::new(DefaultObjectStoreRegistry::new()) as _
    };

    let file_static_cache = Arc::new(DefaultFileStatisticsCache::default());
    let list_file_cache = Arc::new(DefaultListFilesCache::default());

    let cache_config = CacheManagerConfig::default()
        .with_files_statistics_cache(Some(file_static_cache))
        .with_list_files_cache(Some(list_file_cache));

    let rt = RuntimeEnvBuilder::new()
        .with_cache_manager(cache_config)
        .with_object_store_registry(registry)
        .build_arc()
        .expect("could not build runtime environment");

    SessionContext::new_with_config_rt(SessionConfig::default(), rt)
}

/// Creates a randomly generated struct array, where each field is a list of
/// i64 of size one.
pub fn generate_struct_of_list_of_ints_array(
    num_columns: u32,
    rows: u32,
    chunk_count: u32,
) -> VortexResult<ChunkedArray> {
    let int_dtype = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));
    let list_of_ints_dtype = DType::List(int_dtype.clone(), Nullability::Nullable);
    let struct_dtype: Arc<StructDType> = Arc::new(
        (0..num_columns)
            .map(|col_idx| (col_idx.to_string(), list_of_ints_dtype.clone()))
            .collect(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let rows_per_chunk = (rows / chunk_count).max(1u32);
    let arrays = (0..rows)
        .step_by(rows_per_chunk as usize)
        .map(|starting_row| rows_per_chunk.min(rows - starting_row))
        .map(|chunk_row_count| {
            let fields = (0u32..num_columns)
                .map(|_| {
                    let elements = PrimitiveArray::from_iter(
                        (0u32..chunk_row_count).map(|_| rng.gen::<i64>()),
                    );
                    let offsets = PrimitiveArray::from_iter(0u32..=chunk_row_count);
                    ListArray::try_new(
                        elements.into_array(),
                        offsets.into_array(),
                        Validity::AllValid,
                    )
                    .map(|a| a.into_array())
                })
                .collect::<VortexResult<Vec<_>>>()?;
            StructArray::try_new(
                struct_dtype.names().clone(),
                fields,
                chunk_row_count as usize,
                Validity::NonNullable,
            )
            .map(|a| a.into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;

    ChunkedArray::try_new(
        arrays,
        DType::Struct(struct_dtype.clone(), Nullability::NonNullable),
    )
}
