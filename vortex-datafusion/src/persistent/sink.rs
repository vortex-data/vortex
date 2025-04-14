use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::runtime::SpawnedTask;
use datafusion::datasource::file_format::write::demux::DemuxedStreamReceiver;
use datafusion::datasource::physical_plan::{FileSink, FileSinkConfig};
use datafusion_common::DataFusionError;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::insert::DataSink;
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use futures::StreamExt;
use object_store::ObjectStore;
use tokio_stream::wrappers::ReceiverStream;
use vortex_array::TryIntoArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_file::VortexWriteOptions;
use vortex_io::{ObjectStoreWriter, VortexWrite};

pub struct VortexSink {
    config: FileSinkConfig,
    schema: SchemaRef,
}

impl VortexSink {
    pub fn new(config: FileSinkConfig, schema: SchemaRef) -> Self {
        Self { config, schema }
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

    /// Returns the sink schema
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
    ) -> datafusion_common::Result<u64> {
        FileSink::write_all(self, data, context).await
    }
}

#[async_trait]
impl FileSink for VortexSink {
    fn config(&self) -> &FileSinkConfig {
        &self.config
    }

    async fn spawn_writer_tasks_and_join(
        &self,
        _context: &Arc<TaskContext>,
        demux_task: SpawnedTask<datafusion_common::Result<()>>,
        mut file_stream_rx: DemuxedStreamReceiver,
        object_store: Arc<dyn ObjectStore>,
    ) -> datafusion_common::Result<u64> {
        // This is a hack
        let row_counter = Arc::new(AtomicU64::new(0));

        // TODO(adamg):
        // 1. We only write only file at a time
        // 2. We can probably be better at signaling how much memory we're consuming (potentially when reading too), see ParquetSink::spawn_writer_tasks_and_join.
        while let Some((path, rx)) = file_stream_rx.recv().await {
            let writer = ObjectStoreWriter::new(object_store.clone(), path).await?;

            let stream = ReceiverStream::new(rx).map(|rb| {
                row_counter.fetch_add(rb.num_rows() as u64, Ordering::Relaxed);
                rb.try_into_array()
            });
            let dtype = DType::from_arrow(self.config.output_schema.as_ref());
            let stream_adapter = ArrayStreamAdapter::new(dtype, stream);

            let mut writer = VortexWriteOptions::default()
                .write(writer, stream_adapter)
                .await?;

            writer.shutdown().await?;
        }

        demux_task
            .join_unwind()
            .await
            .map_err(DataFusionError::ExecutionJoin)??;

        Ok(row_counter.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::datasource::DefaultTableSource;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values};
    use tempfile::TempDir;

    use crate::persistent::{VortexFormatFactory, register_vortex_format_factory};

    #[tokio::test]
    async fn test_insert_into() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::default_config();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex 
                LOCATION '{}/*';",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        let my_tbl = session.table("my_tbl").await.unwrap();

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal("hello".into()),
                Expr::Literal(42_i32.into()),
            ]],
        };

        let tbl_provider = session.table_provider("my_tbl").await.unwrap();

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(tbl_provider)),
            datafusion_expr::dml::InsertOp::Append,
        )
        .unwrap()
        .build()
        .unwrap();

        session
            .execute_logical_plan(logical_plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        session
            .sql("INSERT INTO my_tbl VALUES ('world', 24);")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        my_tbl.clone().show().await.unwrap();

        assert_eq!(
            session
                .table("my_tbl")
                .await
                .unwrap()
                .count()
                .await
                .unwrap(),
            2
        );
    }
}
