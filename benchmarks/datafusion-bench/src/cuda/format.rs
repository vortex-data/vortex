// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-accelerated Vortex file format for DataFusion.

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::internal_datafusion_err;
use datafusion_common::not_impl_err;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::LexRequirement;
use datafusion_physical_plan::ExecutionPlan;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;

use super::source::CudaVortexSource;

/// CUDA-accelerated Vortex file format for DataFusion.
///
/// This wraps the standard `VortexFormat` but uses `CudaVortexSource` for execution.
pub struct CudaVortexFormat {
    /// The underlying VortexFormat for schema inference and statistics.
    inner: VortexFormat,
    session: VortexSession,
}

impl Debug for CudaVortexFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaVortexFormat").finish()
    }
}

impl CudaVortexFormat {
    /// Create a new CUDA-accelerated Vortex format.
    pub fn new(session: VortexSession) -> Self {
        Self {
            inner: VortexFormat::new(session.clone()),
            session,
        }
    }
}

#[async_trait]
impl FileFormat for CudaVortexFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        None
    }

    fn get_ext(&self) -> String {
        VORTEX_FILE_EXTENSION.to_string()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> DFResult<String> {
        match file_compression_type.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok(self.get_ext()),
            _ => Err(DataFusionError::Internal(
                "Vortex does not support file level compression.".into(),
            )),
        }
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> DFResult<SchemaRef> {
        // Delegate to inner VortexFormat
        self.inner.infer_schema(state, store, objects).await
    }

    async fn infer_stats(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        // Delegate to inner VortexFormat
        self.inner
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        file_scan_config: FileScanConfig,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Get the source from the config and replace with our CUDA source
        let mut source = file_scan_config
            .file_source()
            .as_any()
            .downcast_ref::<CudaVortexSource>()
            .cloned()
            .ok_or_else(|| internal_datafusion_err!("Expected CudaVortexSource"))?;

        source = source
            .with_file_metadata_cache(state.runtime_env().cache_manager.get_file_metadata_cache());

        let conf = FileScanConfigBuilder::from(file_scan_config)
            .with_source(Arc::new(source))
            .build();

        Ok(DataSourceExec::from_data_source(conf))
    }

    async fn create_writer_physical_plan(
        &self,
        _input: Arc<dyn ExecutionPlan>,
        _state: &dyn Session,
        _conf: datafusion_datasource::file_sink_config::FileSinkConfig,
        _order_requirements: Option<LexRequirement>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        not_impl_err!("CudaVortexFormat does not support writing")
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        Arc::new(CudaVortexSource::new(table_schema, self.session.clone()))
    }
}
