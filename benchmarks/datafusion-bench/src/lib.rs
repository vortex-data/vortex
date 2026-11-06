// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod any_groups_limit;
pub mod metrics;
mod mixed_distinct;
mod topk_distinct;
pub mod tracer;

use std::sync::Arc;

use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;
use vortex_utils::parallelism::get_available_parallelism;

/// The environment variables that override the defaults of [`session_config`].
const TARGET_PARTITIONS_VAR: &str = "DATAFUSION_EXECUTION_TARGET_PARTITIONS";
const SKIP_PARTIAL_RATIO_VAR: &str =
    "DATAFUSION_EXECUTION_SKIP_PARTIAL_AGGREGATION_PROBE_RATIO_THRESHOLD";

/// The DataFusion configuration from the environment, with two defaults for the aggregations of
/// the benchmarks when the environment does not set them:
///
/// - One and a half partitions per core. A partition that waits on its input leaves its core to
///   another one, which keeps the cores busy through the hash aggregations.
/// - A partial aggregation stops when it has more than one group per five rows. A partial
///   aggregation over keys of high cardinality, such as `UserID`, costs a hash table update per
///   row and reduces the rows little, so the final aggregation can do the work alone.
fn session_config() -> datafusion::common::Result<SessionConfig> {
    let mut config = SessionConfig::from_env()?;

    if std::env::var_os(TARGET_PARTITIONS_VAR).is_none() {
        let cores = get_available_parallelism().unwrap_or(1);
        config = config.with_target_partitions(cores * 3 / 2);
    }

    if std::env::var_os(SKIP_PARTIAL_RATIO_VAR).is_none() {
        config
            .options_mut()
            .execution
            .skip_partial_aggregation_probe_ratio_threshold = 0.2;
    }

    Ok(config)
}

#[expect(clippy::expect_used)]
pub fn get_session_context() -> SessionContext {
    let mut rt_builder = RuntimeEnvBuilder::new();

    rt_builder = rt_builder.with_cache_manager(CacheManagerConfig::default());

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new().with_options(vortex_table_options());

    let mut session_state_builder = SessionStateBuilder::new()
        .with_config(session_config().expect("shouldn't fail"))
        .with_runtime_env(rt)
        .with_default_features()
        .with_optimizer_rule(Arc::new(any_groups_limit::AnyGroupsLimit))
        .with_optimizer_rule(Arc::new(topk_distinct::TopKDistinct))
        .with_optimizer_rule(Arc::new(mixed_distinct::MixedDistinctToGroupBy));

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

pub fn make_object_store(
    session: &SessionContext,
    source: &Url,
) -> anyhow::Result<Arc<dyn ObjectStore>> {
    match source.scheme() {
        "s3" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let s3 = Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()?,
            );
            session.register_object_store(
                &Url::parse(&format!("s3://{bucket_name}/"))?,
                Arc::<object_store::aws::AmazonS3>::clone(&s3),
            );
            Ok(s3)
        }
        "gs" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let gcs = Arc::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()?,
            );
            session.register_object_store(
                &Url::parse(&format!("gs://{bucket_name}/"))?,
                Arc::<object_store::gcp::GoogleCloudStorage>::clone(&gcs),
            );
            Ok(gcs)
        }
        _ => {
            let fs = Arc::new(LocalFileSystem::default());
            session
                .register_object_store(&Url::parse("file:/")?, Arc::<LocalFileSystem>::clone(&fs));
            Ok(fs)
        }
    }
}

pub fn format_to_df_format(format: Format, scan_v2: bool) -> anyhow::Result<Arc<dyn FileFormat>> {
    let mut options = vortex_table_options();
    options.scan_v2 = scan_v2;
    Ok(match format {
        Format::Csv => Arc::new(CsvFormat::default()) as _,
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexSpatialNative => {
            Arc::new(VortexFormat::new_with_options(SESSION.clone(), options))
        }
        Format::ArrowIpc | Format::OnDiskDuckDB | Format::Lance | Format::LitData => {
            anyhow::bail!("Format {format} cannot be turned into a DataFusion `FileFormat`")
        }
    })
}

fn vortex_table_options() -> VortexTableOptions {
    let mut opts = VortexTableOptions::default();

    opts.predicate_pushdown = true;
    opts.projection_pushdown = true;

    opts
}
