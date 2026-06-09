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
use datafusion::execution::cache::cache_unit::DefaultFileStatisticsCache;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::io::object_store::FileLocation;
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
        .with_files_statistics_cache(Some(file_static_cache))
        .with_list_files_cache(Some(list_file_cache));
    rt_builder = rt_builder.with_cache_manager(cache_config);

    let rt = rt_builder
        .build_arc()
        .expect("could not build runtime environment");

    let factory = VortexFormatFactory::new().with_options(VortexTableOptions {
        projection_pushdown: true,
        ..Default::default()
    });

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
    match FileLocation::resolve(source.as_str())? {
        FileLocation::Remote { store, .. } => {
            let authority = &source[url::Position::BeforeHost..url::Position::AfterHost];
            let base = Url::parse(&format!("{}://{authority}/", source.scheme()))?;
            session.register_object_store(&base, Arc::clone(&store));
            Ok(store)
        }
        FileLocation::Local(_) => {
            let fs = Arc::new(LocalFileSystem::default());
            session
                .register_object_store(&Url::parse("file:/")?, Arc::<LocalFileSystem>::clone(&fs));
            Ok(fs)
        }
    }
}

pub fn format_to_df_format(format: Format) -> Arc<dyn FileFormat> {
    match format {
        Format::Csv => Arc::new(CsvFormat::default()) as _,
        Format::Arrow => Arc::new(ArrowFormat),
        Format::Parquet => Arc::new(ParquetFormat::new()),
        Format::OnDiskVortex | Format::VortexCompact => {
            Arc::new(VortexFormat::new(SESSION.clone()))
        }
        Format::OnDiskDuckDB | Format::Lance => {
            unimplemented!("Format {format} cannot be turned into a DataFusion `FileFormat`")
        }
    }
}
