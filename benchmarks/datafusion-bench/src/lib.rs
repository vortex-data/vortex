// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod metrics;
pub mod tracer;

use std::sync::Arc;

use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::arrow::ArrowFormat;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::cache::DefaultListFilesCache;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::file_statistics_cache::DefaultFileStatisticsCache;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::scan::ScanScheduler;
use vortex::scan::ScanSchedulerConfig;
use vortex::scan::ScanSchedulerSessionExt;
use vortex::session::VortexSession;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;

#[expect(clippy::expect_used)]
pub fn get_session_context() -> SessionContext {
    let mut rt_builder = RuntimeEnvBuilder::new();

    let file_static_cache = Arc::new(DefaultFileStatisticsCache::default());
    let list_file_cache = Arc::new(DefaultListFilesCache::default());
    let cache_config = CacheManagerConfig::default()
        .with_file_statistics_cache(Some(file_static_cache))
        .with_list_files_cache(Some(list_file_cache));
    rt_builder = rt_builder.with_cache_manager(cache_config);

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new()
        .with_session(
            vortex_session_from_env().expect("invalid Vortex benchmark scan scheduler env"),
        )
        .with_options(vortex_table_options());

    let mut session_state_builder = SessionStateBuilder::new()
        .with_config(SessionConfig::from_env().expect("shouldn't fail"))
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

pub fn format_to_df_format(format: Format) -> anyhow::Result<Arc<dyn FileFormat>> {
    Ok(match format {
        Format::Csv => Arc::new(CsvFormat::default()) as _,
        Format::Arrow => Arc::new(ArrowFormat),
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact => Arc::new(VortexFormat::new_with_options(
            vortex_session_from_env()?,
            vortex_table_options(),
        )),
        Format::OnDiskDuckDB | Format::Lance => {
            anyhow::bail!("Format {format} cannot be turned into a DataFusion `FileFormat`")
        }
    })
}

fn vortex_session_from_env() -> anyhow::Result<VortexSession> {
    let session = SESSION.clone();
    let Ok(mode) = std::env::var("VORTEX_SCAN_SCHEDULER") else {
        return Ok(session);
    };
    let config = scan_scheduler_config_from_env()?;
    Ok(match mode.as_str() {
        "unbounded" => session.with_unbounded_scan_scheduler(),
        "shared" | "global" => session.with_scan_scheduler(Arc::new(ScanScheduler::new(config))),
        "per-query" | "per-scan" => session.with_new_scan_scheduler_per_scan(config),
        other => anyhow::bail!(
            "Invalid VORTEX_SCAN_SCHEDULER={other}; expected unbounded, shared, or per-query"
        ),
    })
}

fn scan_scheduler_config_from_env() -> anyhow::Result<ScanSchedulerConfig> {
    let read_byte_budget = std::env::var("VORTEX_SCAN_MAX_READ_BYTES")
        .ok()
        .map(|value| {
            value.parse::<u64>().map_err(|e| {
                anyhow::anyhow!("invalid scan scheduler read byte budget {value}: {e}")
            })
        })
        .transpose()?;

    Ok(match read_byte_budget {
        Some(bytes) => ScanSchedulerConfig::default().with_read_byte_budget(Some(bytes)),
        None => ScanSchedulerConfig::default(),
    })
}

fn vortex_table_options() -> VortexTableOptions {
    VortexTableOptions {
        projection_pushdown: true,
        predicate_pushdown: true,
        ..Default::default()
    }
}
