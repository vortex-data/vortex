// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod latency_store;
pub mod metrics;
pub mod tracer;

use std::sync::Arc;
use std::time::Duration;

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

use crate::latency_store::LatencyStore;

#[expect(clippy::expect_used)]
pub fn get_session_context() -> SessionContext {
    let mut rt_builder = RuntimeEnvBuilder::new();

    rt_builder = rt_builder.with_cache_manager(CacheManagerConfig::default());

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new().with_options(vortex_table_options());

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
            let fs: Arc<dyn ObjectStore> = match local_get_latency()? {
                // Emulate a remote store's per-request latency over local files, so IO
                // scheduling can be compared where requests are expensive.
                Some(latency) => Arc::new(LatencyStore::new(LocalFileSystem::default(), latency)),
                None => Arc::new(LocalFileSystem::default()),
            };
            session.register_object_store(&Url::parse("file:/")?, Arc::clone(&fs));
            Ok(fs)
        }
    }
}

/// The latency `VORTEX_BENCH_GET_LATENCY_MS` adds to every local object store GET, if set.
fn local_get_latency() -> anyhow::Result<Option<Duration>> {
    std::env::var("VORTEX_BENCH_GET_LATENCY_MS")
        .ok()
        .map(|ms| Ok(Duration::from_secs_f64(ms.parse::<f64>()? / 1000.0)))
        .transpose()
}

pub fn format_to_df_format(format: Format) -> anyhow::Result<Arc<dyn FileFormat>> {
    Ok(match format {
        Format::Csv => Arc::new(CsvFormat::default()) as _,
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexSpatialNative => Arc::new(
            VortexFormat::new_with_options(SESSION.clone(), vortex_table_options()),
        ),
        Format::ArrowIpc | Format::OnDiskDuckDB | Format::Lance => {
            anyhow::bail!("Format {format} cannot be turned into a DataFusion `FileFormat`")
        }
    })
}

fn vortex_table_options() -> VortexTableOptions {
    let mut opts = VortexTableOptions::default();

    opts.predicate_pushdown = true;
    opts.projection_pushdown = true;
    opts.morsel_scan = std::env::var("VORTEX_DF_MORSEL_SCAN").is_ok_and(|v| v == "1");

    opts
}
