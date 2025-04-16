use std::fs;
use std::process::Command;
use std::sync::{Arc, LazyLock};

use arrow_array::RecordBatch;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::cache_unit::{DefaultFileStatisticsCache, DefaultListFilesCache};
use datafusion::execution::object_store::DefaultObjectStoreRegistry;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::error::VortexResult;

use crate::blob::SlowObjectStoreRegistry;

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

/// Write execution plan details to files for benchmarks
///
/// Creates 2 plan files for each query execution:
/// - A detailed plan with full structure
/// - A condensed plan with metrics and schema
pub fn write_execution_plan(
    query_idx: usize,
    format: crate::Format,
    dataset_name: &str,
    execution_plan: &std::sync::Arc<dyn datafusion::physical_plan::execution_plan::ExecutionPlan>,
) {
    use datafusion::physical_plan::display::DisplayableExecutionPlan;

    fs::write(
        format!("{dataset_name}_{format}_q{query_idx:02}.plan"),
        format!("{:#?}", execution_plan),
    )
    .expect("Unable to write file");

    fs::write(
        format!("{dataset_name}_{format}_q{query_idx:02}.short.plan"),
        format!(
            "{}",
            DisplayableExecutionPlan::with_full_metrics(execution_plan.as_ref())
                .set_show_schema(true)
                .set_show_statistics(true)
                .indent(true)
        ),
    )
    .expect("Unable to write file");
}
