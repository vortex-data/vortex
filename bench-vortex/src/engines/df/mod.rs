// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::process::Command;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use anyhow::Result;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::cache_unit::{DefaultFileStatisticsCache, DefaultListFilesCache};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::GetExt;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex_datafusion::VortexFormatFactory;

use crate::Engine;
pub use crate::Format;
use crate::engines::{QueryEngine, QueryMetrics};

pub struct DataFusionCtx {
    pub execution_plans: Vec<(usize, Arc<dyn ExecutionPlan>)>,
    pub metrics: Vec<(
        usize,
        Format,
        Vec<datafusion::physical_plan::metrics::MetricsSet>,
    )>,

    pub session: SessionContext,
    pub emit_plan: bool,
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

pub fn get_session_context(disable_datafusion_cache: bool) -> SessionContext {
    let mut rt_builder = RuntimeEnvBuilder::new();

    if !disable_datafusion_cache {
        let file_static_cache = Arc::new(DefaultFileStatisticsCache::default());
        let list_file_cache = Arc::new(DefaultListFilesCache::default());
        let cache_config = CacheManagerConfig::default()
            .with_files_statistics_cache(Some(file_static_cache))
            .with_list_files_cache(Some(list_file_cache));
        rt_builder = rt_builder.with_cache_manager(cache_config);
    }

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new();

    let mut session_state_builder = SessionStateBuilder::new()
        .with_config(SessionConfig::default())
        .with_runtime_env(rt)
        .with_default_features();

    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            Arc::new(DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(Arc::new(factory));
    }

    SessionContext::new_with_state(session_state_builder.build())
}

pub fn make_object_store(df: &SessionContext, source: &Url) -> Result<Arc<dyn ObjectStore>> {
    match source.scheme() {
        "s3" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let s3 = Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()
                    .unwrap(),
            );
            df.register_object_store(&Url::parse(&format!("s3://{bucket_name}/"))?, s3.clone());
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
            df.register_object_store(&Url::parse(&format!("gs://{bucket_name}/"))?, gcs.clone());
            Ok(gcs)
        }
        _ => {
            let fs = Arc::new(LocalFileSystem::default());
            df.register_object_store(&Url::parse("file:/")?, fs.clone());
            Ok(fs)
        }
    }
}

impl DataFusionCtx {
    /// Execute a query and return batches and execution plan
    ///
    /// This is the internal method used by the QueryEngine trait implementation.
    /// Prefer using the trait method instead.
    pub async fn execute_query_internal(
        &self,
        query: &str,
    ) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
        execute_query(&self.session, query).await
    }
}

pub async fn execute_query(
    ctx: &SessionContext,
    query: &str,
) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
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
    format: Format,
    dataset_name: &str,
    execution_plan: &dyn ExecutionPlan,
) {
    fs::write(
        format!("{dataset_name}_{format}_q{query_idx:02}.plan"),
        format!("{execution_plan:#?}"),
    )
    .expect("Unable to write file");

    fs::write(
        format!("{dataset_name}_{format}_q{query_idx:02}.short.plan"),
        format!(
            "{}",
            DisplayableExecutionPlan::with_metrics(execution_plan)
                .set_show_schema(true)
                .tree_render()
        ),
    )
    .expect("Unable to write file");
}

#[async_trait]
impl QueryEngine for DataFusionCtx {
    async fn execute_query(&mut self, query: &str) -> Result<QueryMetrics> {
        let start = Instant::now();
        let (batches, execution_plan) = execute_query(&self.session, query).await?;
        let duration = start.elapsed();

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();

        Ok(QueryMetrics {
            duration,
            row_count,
            execution_plan: Some(execution_plan),
        })
    }

    fn engine_type(&self) -> Engine {
        Engine::DataFusion
    }

    fn should_emit_plan(&self) -> bool {
        self.emit_plan
    }

    fn execution_plans(&self) -> &[(usize, Arc<dyn ExecutionPlan>)] {
        &self.execution_plans
    }

    fn metrics(
        &self,
    ) -> &[(
        usize,
        Format,
        Vec<datafusion_physical_plan::metrics::MetricsSet>,
    )] {
        &self.metrics
    }

    fn add_execution_data(
        &mut self,
        query_idx: usize,
        plan: Arc<dyn ExecutionPlan>,
        format: Format,
        metrics: Vec<datafusion_physical_plan::metrics::MetricsSet>,
    ) {
        self.execution_plans.push((query_idx, plan));
        self.metrics.push((query_idx, format, metrics));
    }

    fn as_datafusion_session(&self) -> Option<&SessionContext> {
        Some(&self.session)
    }
}
