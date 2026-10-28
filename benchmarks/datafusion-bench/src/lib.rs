// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod metrics;
pub mod tracer;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub mod uring;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub mod memory;

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
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexSpatialNative => {
            let format = VortexFormat::new_with_options(SESSION.clone(), vortex_table_options());
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            let format = match uring::factory()? {
                Some(factory) => format.with_vortex_reader_factory(factory),
                None => format,
            };
            Arc::new(format)
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
    opts.scan_v2 = std::env::var("VORTEX_SCAN_V2").is_ok_and(|value| value == "1");
    opts.scan_v2_share_plans = std::env::var("VORTEX_SCAN_V2_SHARE_PLANS")
        .map_or(true, |value| value == "1");
    opts.scan_v2_io_cache_bytes = std::env::var("VORTEX_SCAN_V2_IO_CACHE_BYTES")
        .ok().map(|value| value.parse().expect("VORTEX_SCAN_V2_IO_CACHE_BYTES must be an integer"))
        .unwrap_or(0);
    opts.scan_v2_chunk_rows = std::env::var("VORTEX_SCAN_V2_CHUNK_ROWS")
        .ok().map(|value| value.parse().expect("VORTEX_SCAN_V2_CHUNK_ROWS must be an integer"));
    opts.scan_v2_consumer_mapping = std::env::var("VORTEX_SCAN_V2_CONSUMER_MAPPING")
        .as_deref() == Ok("1");
    opts.scan_v2_consumer_mapping_fixed_width_only =
        std::env::var("VORTEX_SCAN_V2_CONSUMER_MAPPING_FIXED_WIDTH_ONLY").as_deref() == Ok("1");
    opts.scan_concurrency = std::env::var("VORTEX_SCAN_CONCURRENCY")
        .ok().map(|value| value.parse().expect("VORTEX_SCAN_CONCURRENCY must be an integer"));

    opts
}
