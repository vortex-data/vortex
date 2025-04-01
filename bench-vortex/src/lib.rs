#![feature(exit_status_error)]

use std::clone::Clone;
use std::env::temp_dir;
use std::fmt::Display;
use std::fs::create_dir_all;
use std::future::Future;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock};

use arrow_array::RecordBatch;
use blob::SlowObjectStoreRegistry;
use clap::ValueEnum;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::cache_unit::{DefaultFileStatisticsCache, DefaultListFilesCache};
use datafusion::execution::object_store::DefaultObjectStoreRegistry;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_physical_plan::{ExecutionPlan, collect};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use rand::{Rng, SeedableRng as _};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use url::Url;
use vortex::Array;
use vortex::arrays::{ChunkedArray, ListArray, PrimitiveArray, StructArray};
use vortex::dtype::{DType, Nullability, PType, StructDType};
use vortex::error::VortexResult;
use vortex::validity::Validity;

pub mod bench_run;
pub mod blob;
pub mod clickbench;
pub mod compress;
pub mod datasets;
pub mod display;
pub mod measurements;
pub mod metrics;
pub mod parquet_reader;
pub mod random_access;
pub mod tpch;

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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, ValueEnum)]
pub enum Format {
    #[clap(name = "csv")]
    Csv,
    #[clap(name = "arrow")]
    Arrow,
    #[clap(name = "parquet")]
    Parquet,
    #[clap(name = "in-memory-vortex")]
    InMemoryVortex,
    #[clap(name = "vortex")]
    OnDiskVortex,
}

impl Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Format {
    pub fn name(&self) -> &'static str {
        match self {
            Format::Csv => "csv",
            Format::Arrow => "arrow",
            Format::Parquet => "parquet",
            Format::InMemoryVortex => "vortex-in-memory",
            Format::OnDiskVortex => "vortex-file-compressed",
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

pub fn setup_logger(filter: EnvFilter) {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_file(true)
        .with_level(true)
        .with_line_number(true)
        .with_env_filter(filter)
        .with_ansi(std::io::stderr().is_terminal())
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

pub async fn execute_query(
    ctx: &SessionContext,
    query: &str,
) -> VortexResult<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    let physical_plan = state.create_physical_plan(&plan).await?;
    let result = collect(physical_plan.clone(), state.task_ctx()).await?;
    Ok((result, physical_plan))
}

pub async fn execute_physical_plan(
    ctx: &SessionContext,
    plan: Arc<dyn ExecutionPlan>,
) -> VortexResult<Vec<RecordBatch>> {
    let result = collect(plan.clone(), ctx.state().task_ctx()).await?;
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
                        (0u32..chunk_row_count).map(|_| rng.random::<i64>()),
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

pub fn make_object_store(
    df: &SessionContext,
    source: &Url,
) -> anyhow::Result<Arc<dyn ObjectStore>> {
    match source.scheme() {
        "s3" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let s3 = Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()
                    .unwrap(),
            );
            df.register_object_store(&Url::parse(&format!("s3://{}/", bucket_name))?, s3.clone());
            Ok(s3)
        }
        "gs" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let gcs = Arc::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()
                    .unwrap(),
            );
            df.register_object_store(&Url::parse(&format!("gs://{}/", bucket_name))?, gcs.clone());
            Ok(gcs)
        }
        _ => {
            let fs = Arc::new(LocalFileSystem::default());
            df.register_object_store(&Url::parse("file:/")?, fs.clone());
            Ok(fs)
        }
    }
}
