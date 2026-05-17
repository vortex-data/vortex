// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_common_runtime::JoinSet;
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::file_sink_config::FileSink;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSink;
use datafusion_datasource::write::demux::DemuxedStreamReceiver;
use datafusion_datasource::write::get_writer_schema;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::metrics::MetricsSet;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use tokio_stream::wrappers::ReceiverStream;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::session::VortexSession;

pub struct VortexSink {
    config: FileSinkConfig,
    schema: SchemaRef,
    session: VortexSession,
}

impl VortexSink {
    pub fn new(config: FileSinkConfig, schema: SchemaRef, session: VortexSession) -> Self {
        Self {
            config,
            schema,
            session,
        }
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
        let mut file_write_tasks: JoinSet<DFResult<(Path, WriteSummary)>> = JoinSet::new();

        // TODO(adamg):
        // 1. We can probably be better at signaling how much memory we're consuming (potentially when reading too), see ParquetSink::spawn_writer_tasks_and_join.
        while let Some((path, rx)) = file_stream_rx.recv().await {
            let session = self.session.clone();
            let object_store = Arc::clone(&object_store);
            let writer_schema = get_writer_schema(&self.config);
            let dtype = session
                .arrow()
                .from_arrow_schema(&writer_schema)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to derive Vortex DType from writer schema: {e}")
                })?;

            // We need to spawn work because there's a dependency between the different files. If one file has too many batches buffered,
            // the demux task might deadlock itself.
            let arrow_session = session.clone();
            let import_schema = Arc::clone(&writer_schema);
            file_write_tasks.spawn(async move {
                let stream = ReceiverStream::new(rx).map(move |rb| {
                    arrow_session
                        .arrow()
                        .from_arrow_record_batch(rb, &import_schema)
                });

                let stream_adapter = ArrayStreamAdapter::new(dtype, stream);

                let mut object_writer = ObjectStoreWrite::new(object_store, &path)
                    .await
                    .map_err(|e| exec_datafusion_err!("Failed to create ObjectStoreWrite: {e}"))?;

                let summary = session
                    .write_options()
                    .write(&mut object_writer, stream_adapter)
                    .await
                    .map_err(|e| exec_datafusion_err!("Failed to write Vortex file: {e}"))?;

                object_writer
                    .shutdown()
                    .await
                    .map_err(|e| exec_datafusion_err!("Failed to shutdown Vortex writer: {e}"))?;

                Ok((path, summary))
            });
        }

        let results = drain_writers_with_cleanup(file_write_tasks, demux_task).await?;

        let mut row_count = 0;
        for (path, summary) in results {
            row_count += summary.row_count();
            tracing::info!(path = %path, "Successfully written file");
        }

        Ok(row_count)
    }
}

/// Drain a `JoinSet` of writer tasks to completion, then await the demux task.
///
/// Unlike a naive `?` inside the join loop, this never aborts in-flight writer
/// tasks when the first error appears. It captures the first writer error,
/// continues draining the remaining tasks (so every spawned write either
/// completes or surfaces its own error), logs subsequent writer errors at
/// warn level, and only then awaits the demux task. If demux failed and no
/// writer error was captured, the demux error is surfaced; otherwise the
/// first writer error wins.
///
/// On success returns the per-task results in completion order.
async fn drain_writers_with_cleanup<T: 'static>(
    mut file_write_tasks: JoinSet<DFResult<T>>,
    demux_task: SpawnedTask<DFResult<()>>,
) -> DFResult<Vec<T>> {
    let mut successes: Vec<T> = Vec::new();
    let mut first_writer_err: Option<DataFusionError> = None;

    while let Some(result) = file_write_tasks.join_next().await {
        match result {
            Ok(Ok(value)) => successes.push(value),
            Ok(Err(e)) => {
                if first_writer_err.is_none() {
                    first_writer_err = Some(e);
                } else {
                    tracing::warn!(error = %e, "additional writer task error suppressed");
                }
            }
            Err(e) => {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                } else {
                    // Cancelled join errors should not occur because we never
                    // call `abort_all` on this JoinSet, but surface them rather
                    // than panic.
                    let join_err = DataFusionError::ExecutionJoin(Box::new(e));
                    if first_writer_err.is_none() {
                        first_writer_err = Some(join_err);
                    } else {
                        tracing::warn!(error = %join_err, "additional writer task error suppressed");
                    }
                }
            }
        }
    }

    let demux_result = demux_task
        .join_unwind()
        .await
        .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))
        .and_then(|inner| inner);

    if let Some(err) = first_writer_err {
        if let Err(demux_err) = demux_result {
            tracing::warn!(error = %demux_err, "demux task error suppressed");
        }
        return Err(err);
    }

    demux_result?;

    Ok(successes)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion::arrow::array::Int8Array;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::assert_batches_sorted_eq;
    use datafusion::datasource::DefaultTableSource;
    use datafusion::logical_expr::Expr;
    use datafusion::logical_expr::LogicalPlan;
    use datafusion::logical_expr::LogicalPlanBuilder;
    use datafusion::logical_expr::Values;
    use datafusion::logical_expr::dml::InsertOp;
    use datafusion_common::ScalarValue;
    use datafusion_common::exec_datafusion_err;
    use datafusion_common_runtime::JoinSet;
    use datafusion_common_runtime::SpawnedTask;
    use datafusion_datasource::file_format::format_as_file_type;
    use futures::TryStreamExt;
    use rstest::rstest;

    use super::drain_writers_with_cleanup;
    use crate::common_tests::TestSessionContext;
    use crate::persistent::VortexFormatFactory;

    #[tokio::test]
    async fn test_insert_into_sql() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/';",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl VALUES ('hello', 1), ('world', 2);")
            .await?
            .collect()
            .await?;

        let batches = ctx
            .session
            .sql("SELECT * from my_tbl")
            .await?
            .collect()
            .await?;

        assert_batches_sorted_eq!(
            &[
                "+-------+----+",
                "| c1    | c2 |",
                "+-------+----+",
                "| hello | 1  |",
                "| world | 2  |",
                "+-------+----+",
            ],
            &batches
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_into_logical_plan() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/';",
            )
            .await?;

        let my_tbl = ctx.session.table("my_tbl").await?;

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal(ScalarValue::new_utf8view("hello"), None),
                Expr::Literal(42_i32.into(), None),
            ]],
        };

        let tbl_provider = ctx.session.table_provider("my_tbl").await?;

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(Arc::clone(&tbl_provider))),
            InsertOp::Append,
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let batches = ctx.session.read_table(tbl_provider)?.collect().await?;

        assert_batches_sorted_eq!(
            [
                "+-------+----+",
                "| c1    | c2 |",
                "+-------+----+",
                "| hello | 42 |",
                "+-------+----+",
            ],
            &batches
        );

        Ok(())
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
        let ctx = TestSessionContext::default();

        let data = ctx.session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![0i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new())),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let result = ctx
            .session
            .sql("SELECT COUNT(*) as count FROM '/table/'")
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

        let all_data = ctx
            .session
            .sql("SELECT a FROM '/table/'")
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

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(
            file_metas.len(),
            expected_files,
            "Expected {expected_files} files for {entries} values"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_partitioned() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        let _unused = ctx
            .session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                PARTITIONED BY (c1);",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl (c1, c2) VALUES ('world', 24), ('world', 25), ('hello', 42);")
            .await?
            .collect()
            .await?;

        let table = ctx.session.table("my_tbl").await?;
        assert_eq!(table.count().await?, 3);

        let location = object_store::path::Path::parse("table/")?;
        let file_metas = ctx
            .store
            .list(Some(&location))
            .try_collect::<Vec<_>>()
            .await?;

        for meta in file_metas.into_iter() {
            let location = meta.location;
            assert!(
                location.prefix_matches(&"c1=hello".into())
                    || location.prefix_matches(&"c1=world".into())
            );
        }

        Ok(())
    }

    /// Spawn three writer tasks (one fails quickly, two succeed slowly) and
    /// verify that the drain helper waits for every spawned task to complete
    /// before returning the first writer error. On the pre-fix code path, an
    /// early `?` would have dropped the slow tasks before they could bump the
    /// shared counter; with the drain-on-error semantics all three must run.
    #[tokio::test]
    async fn test_drain_writers_with_cleanup_drains_all_on_failure() -> anyhow::Result<()> {
        let completed = Arc::new(AtomicUsize::new(0));

        let mut joinset: JoinSet<datafusion_common::Result<u64>> = JoinSet::new();

        let c = Arc::clone(&completed);
        joinset.spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            c.fetch_add(1, Ordering::SeqCst);
            Err(exec_datafusion_err!("writer A failed"))
        });

        for value in [100u64, 200u64] {
            let c = Arc::clone(&completed);
            joinset.spawn(async move {
                tokio::time::sleep(Duration::from_millis(80)).await;
                c.fetch_add(1, Ordering::SeqCst);
                Ok(value)
            });
        }

        let demux: SpawnedTask<datafusion_common::Result<()>> =
            SpawnedTask::spawn(async { Ok(()) });

        let err = drain_writers_with_cleanup(joinset, demux)
            .await
            .expect_err("expected the writer error to propagate");
        assert!(
            err.to_string().contains("writer A failed"),
            "unexpected error: {err}"
        );

        assert_eq!(
            completed.load(Ordering::SeqCst),
            3,
            "expected all three writer tasks to run to completion, observed {}",
            completed.load(Ordering::SeqCst)
        );

        Ok(())
    }

    /// When every writer succeeds but the demux task fails, the helper must
    /// surface the demux error rather than silently swallowing it.
    #[tokio::test]
    async fn test_drain_writers_with_cleanup_surfaces_demux_error() -> anyhow::Result<()> {
        let mut joinset: JoinSet<datafusion_common::Result<u64>> = JoinSet::new();
        joinset.spawn(async { Ok(1u64) });
        joinset.spawn(async { Ok(2u64) });

        let demux: SpawnedTask<datafusion_common::Result<()>> =
            SpawnedTask::spawn(async { Err(exec_datafusion_err!("demux failed")) });

        let err = drain_writers_with_cleanup(joinset, demux)
            .await
            .expect_err("expected the demux error to propagate");
        assert!(
            err.to_string().contains("demux failed"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    /// On full success, the helper returns the collected per-task values.
    #[tokio::test]
    async fn test_drain_writers_with_cleanup_returns_values_on_success() -> anyhow::Result<()> {
        let mut joinset: JoinSet<datafusion_common::Result<u64>> = JoinSet::new();
        joinset.spawn(async { Ok(10u64) });
        joinset.spawn(async { Ok(20u64) });
        joinset.spawn(async { Ok(30u64) });

        let demux: SpawnedTask<datafusion_common::Result<()>> =
            SpawnedTask::spawn(async { Ok(()) });

        let mut values = drain_writers_with_cleanup(joinset, demux).await?;
        values.sort_unstable();
        assert_eq!(values, vec![10, 20, 30]);

        Ok(())
    }

    /// When both a writer task and the demux task fail, the writer error wins
    /// (it is the more actionable signal — demux can't make progress when its
    /// consumer dies).
    #[tokio::test]
    async fn test_drain_writers_with_cleanup_prefers_writer_error() -> anyhow::Result<()> {
        let mut joinset: JoinSet<datafusion_common::Result<u64>> = JoinSet::new();
        joinset.spawn(async { Err(exec_datafusion_err!("writer boom")) });
        joinset.spawn(async { Ok(5u64) });

        let demux: SpawnedTask<datafusion_common::Result<()>> =
            SpawnedTask::spawn(async { Err(exec_datafusion_err!("demux boom")) });

        let err = drain_writers_with_cleanup(joinset, demux)
            .await
            .expect_err("expected the writer error to propagate");
        assert!(
            err.to_string().contains("writer boom"),
            "expected writer error to win, got: {err}"
        );

        Ok(())
    }
}
