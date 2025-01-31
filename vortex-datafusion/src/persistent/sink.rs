use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::datasource::physical_plan::FileSinkConfig;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::insert::DataSink;
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use futures::{StreamExt, TryStreamExt};
use rand::distributions::{Alphanumeric, DistString};
use vortex_array::arrow::FromArrowType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_file::{VortexWriteOptions, VORTEX_FILE_EXTENSION};
use vortex_io::ObjectStoreWriter;

pub struct VortexSink {
    config: FileSinkConfig,
}

impl VortexSink {
    pub fn new(config: FileSinkConfig) -> Self {
        Self { config }
    }
}

impl std::fmt::Debug for VortexSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexSink").finish()
    }
}

impl DisplayAs for VortexSink {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "VortexSink")
            }
        }
    }
}

#[async_trait]
impl DataSink for VortexSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
    ) -> datafusion_common::error::Result<u64> {
        let object_store = context
            .runtime_env()
            .object_store(&self.config.object_store_url)?;

        let base_output_path = &self.config.table_paths[0];

        let single_file_output =
            !base_output_path.is_collection() && base_output_path.file_extension().is_some();

        let path = if single_file_output {
            base_output_path.prefix().to_owned()
        } else {
            let filename = Alphanumeric.sample_string(&mut rand::thread_rng(), 16);
            base_output_path
                .prefix()
                .child(format!("{filename}.{}", VORTEX_FILE_EXTENSION))
        };

        let vortex_writer = ObjectStoreWriter::new(object_store, path).await?;

        // TODO(adam): This is a temporary hack
        let row_counter = Arc::new(AtomicU64::new(0));

        let dtype = DType::from_arrow(data.schema());
        let stream = data
            .map_err(VortexError::from)
            .map(|rb| rb.and_then(Array::try_from))
            .map_ok(|rb| {
                row_counter.fetch_add(rb.len() as u64, Ordering::SeqCst);
                rb
            });

        let stream = ArrayStreamAdapter::new(dtype, stream);

        let _ = VortexWriteOptions::default()
            .write(vortex_writer, stream)
            .await?;

        Ok(row_counter.load(Ordering::SeqCst))
    }
}
