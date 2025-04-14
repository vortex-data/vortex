#![feature(exit_status_error)]

use std::clone::Clone;
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
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use url::Url;
use vortex::error::VortexResult;

pub mod bench_run;
pub mod blob;
pub mod clickbench;
pub mod compress;
pub mod conversions;
pub mod datasets;
pub mod display;
pub mod measurements;
pub mod metrics;
pub mod public_bi;
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

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq)]
pub enum Engine {
    #[default]
    Vortex,
    #[clap(name = "datafusion")]
    DataFusion,
    #[clap(name = "duckdb")]
    DuckDB,
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Engine::DataFusion => write!(f, "DataFusion"),
            Engine::DuckDB => write!(f, "DuckDB"),
            Engine::Vortex => write!(f, "Vortex"),
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
    let temp_path = temp_download_filepath();
    if !data_path.exists() {
        f(temp_path.as_path())?;
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
    let temp_path = temp_download_filepath();
    if !data_path.exists() {
        f(temp_path.clone()).await?;
        std::fs::rename(temp_path, &data_path).unwrap();
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
}

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

pub fn temp_download_filepath() -> PathBuf {
    data_dir().join(format!("download_{}.file", uuid::Uuid::new_v4()))
}

impl IdempotentPath for str {
    fn to_data_path(&self) -> PathBuf {
        let path = data_dir().join(self);
        if !path.parent().unwrap().exists() {
            create_dir_all(path.parent().unwrap()).unwrap();
        }
        path
    }
}

impl IdempotentPath for PathBuf {
    fn to_data_path(&self) -> PathBuf {
        if !self.parent().unwrap().exists() {
            create_dir_all(self.parent().unwrap()).unwrap();
        }
        self.to_path_buf()
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
