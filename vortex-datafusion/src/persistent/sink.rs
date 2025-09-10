// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_datasource::file_sink_config::{FileSink, FileSinkConfig};
use datafusion_datasource::sink::DataSink;
use datafusion_datasource::write::demux::DemuxedStreamReceiver;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexResult;
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamAdapter;

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
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
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
    ) -> DFResult<u64> {
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
        demux_task: SpawnedTask<DFResult<()>>,
        mut file_stream_rx: DemuxedStreamReceiver,
        object_store: Arc<dyn ObjectStore>,
    ) -> DFResult<u64> {
        // This is a hack
        let row_counter = Arc::new(AtomicU64::new(0));

        let mut file_write_tasks: JoinSet<DFResult<Path>> = JoinSet::new();

        // TODO(adamg):
        // 1. We can probably be better at signaling how much memory we're consuming (potentially when reading too), see ParquetSink::spawn_writer_tasks_and_join.
        while let Some((path, rx)) = file_stream_rx.recv().await {
            let row_counter = row_counter.clone();
            let object_store = object_store.clone();
            let dtype = DType::from_arrow(self.config.output_schema.clone());

            // We need to spawn work because there's a dependency between the different files. If one file has too many batches buffered,
            // the demux task might deadlock itself.
            file_write_tasks.spawn(async move {
                let stream = ReceiverStream::new(rx).map(move |rb| {
                    row_counter.fetch_add(rb.num_rows() as u64, Ordering::Relaxed);
                    VortexResult::Ok(ArrayRef::from_arrow(rb, false))
                });

                let stream_adapter = ArrayStreamAdapter::new(dtype, stream);

                VortexWriteOptions::default()
                    .write_object_store(&object_store, &path, stream_adapter)
                    .await
                    .map_err(|e| {
                        DataFusionError::Execution(format!("Failed to write Vortex file: {e}"))
                    })?;

                Ok(path)
            });
        }

        while let Some(result) = file_write_tasks.join_next().await {
            match result {
                Ok(path) => {
                    let path = path?;
                    tracing::info!(path = %path, "Successfully written file");
                }
                Err(e) => {
                    if e.is_panic() {
                        std::panic::resume_unwind(e.into_panic());
                    } else {
                        unreachable!();
                    }
                }
            }
        }

        demux_task
            .join_unwind()
            .await
            .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))??;

        Ok(row_counter.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::arrow::array::{Int8Array, RecordBatch};
    use datafusion::datasource::DefaultTableSource;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values};
    use datafusion::prelude::SessionContext;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::file_format::format_as_file_type;
    use rstest::rstest;
    use tempfile::TempDir;

    use crate::persistent::{VortexFormatFactory, register_vortex_format_factory};

    #[tokio::test]
    async fn test_insert_into() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::new();

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        let my_tbl = session.table("my_tbl").await.unwrap();

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal(ScalarValue::new_utf8view("hello"), None),
                Expr::Literal(42_i32.into(), None),
            ]],
        };

        let tbl_provider = session.table_provider("my_tbl").await.unwrap();

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(tbl_provider)),
            datafusion::logical_expr::dml::InsertOp::Append,
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

    /// Reproduction by <https://github.com/vortex-data/vortex/issues/4315>.
    #[rstest]
    #[case(1000, 1)]
    #[case(40_961, 4)]
    #[case(1_000_000, 4)]
    #[tokio::test]
    async fn test_write_large_batch(
        #[case] entries: usize,
        #[case] expected_files: usize,
    ) -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        let factory = VortexFormatFactory::new();

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![0i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new())),
            Default::default(),
            vec![],
        )?
        .build()?;

        session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        // Validate the output by reading back the written files
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        assert_eq!(result.len(), 1);
        let count_batch = &result[0];
        assert_eq!(count_batch.num_rows(), 1);

        let count_value = count_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {} entries, but found {}",
            entries, count_value
        );

        let all_data = session
            .sql("SELECT a FROM written_data")
            .await?
            .collect()
            .await?;

        let mut total_rows = 0;
        for batch in all_data {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                assert_eq!(
                    col.value(i),
                    0i8,
                    "Expected value 0 at row {}, but found {}",
                    total_rows + i,
                    col.value(i)
                );
            }
            total_rows += batch.num_rows();
        }

        assert_eq!(
            total_rows, entries,
            "Total rows read ({}) doesn't match expected entries ({})",
            total_rows, entries
        );

        let read_dir = std::fs::read_dir(dir.path())?;
        assert_eq!(
            read_dir.count(),
            expected_files,
            "Expected {expected_files} files for {entries} values"
        );

        Ok(())
    }
}
