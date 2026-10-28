// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod cpu_log;
pub mod get_counter;
pub mod metrics;
pub mod page_cache;
pub mod run_rests;
#[cfg(unix)]
mod t4_cache;
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
use get_counter::GetCounter;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex_bench::Format;
use vortex_bench::Opts;
use vortex_bench::SESSION;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;
use vortex_datafusion::reader::VortexReaderFactory;

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

/// The store serving `source`, registered with `session`; its GETs counted by `gets` if given,
/// for a store over the network.
pub fn make_object_store(
    session: &SessionContext,
    source: &Url,
    gets: Option<&Arc<GetCounter>>,
) -> anyhow::Result<Arc<dyn ObjectStore>> {
    let counted = |store: Arc<dyn ObjectStore>| match gets {
        Some(gets) => gets.wrap(store),
        None => store,
    };
    match source.scheme() {
        "s3" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let s3 = counted(Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()?,
            ));
            session.register_object_store(
                &Url::parse(&format!("s3://{bucket_name}/"))?,
                Arc::clone(&s3),
            );
            Ok(s3)
        }
        "gs" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let gcs = counted(Arc::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()?,
            ));
            session.register_object_store(
                &Url::parse(&format!("gs://{bucket_name}/"))?,
                Arc::clone(&gcs),
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

/// t4 in front of `store`, which serves `data_url`, if `--opt t4-policy` asks for it: the
/// factory that reads every Vortex file through t4 ([`t4_cache`]).
pub fn t4_in_front(
    session: &SessionContext,
    opts: &Opts,
    data_url: &Url,
    store: Arc<dyn ObjectStore>,
) -> anyhow::Result<Option<Arc<dyn VortexReaderFactory>>> {
    #[cfg(unix)]
    return t4_cache::t4_in_front(session, opts, data_url, store);
    #[cfg(not(unix))]
    {
        let _ = (session, data_url, store);
        anyhow::ensure!(opts.get("t4-policy").is_none(), "t4 runs on unix only");
        Ok(None)
    }
}

pub fn format_to_df_format(
    format: Format,
    scan_v2: bool,
    reader_factory: Option<Arc<dyn VortexReaderFactory>>,
) -> anyhow::Result<Arc<dyn FileFormat>> {
    let mut options = vortex_table_options();
    options.scan_v2 = scan_v2;
    Ok(match format {
        Format::Csv => Arc::new(CsvFormat::default()) as _,
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexSpatialNative => {
            let mut vortex = VortexFormat::new_with_options(SESSION.clone(), options);
            if let Some(reader_factory) = reader_factory {
                vortex = vortex.with_vortex_reader_factory(reader_factory);
            }
            Arc::new(vortex)
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
