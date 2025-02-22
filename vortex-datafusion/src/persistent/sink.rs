use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_schema::SchemaRef;
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
use vortex_array::TryIntoArray;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_file::{VortexWriteOptions, VORTEX_FILE_EXTENSION};
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
            .map(|rb| rb.and_then(|rb| rb.try_into_array()))
            .map_ok(|rb| {
                row_counter.fetch_add(rb.len() as u64, Ordering::SeqCst);
                rb
            });

        let stream = ArrayStreamAdapter::new(dtype, stream);

        let mut writer = VortexWriteOptions::default()
            .write(vortex_writer, stream)
            .await?;

        writer.flush().await?;
        writer.shutdown().await?;

        Ok(row_counter.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values};
    use tempfile::TempDir;

    use crate::persistent::{register_vortex_format_factory, VortexFormatFactory};

    #[tokio::test]
    async fn test_insert_into() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::default_config();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let df = session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex 
                LOCATION '{}/*';",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        assert_eq!(df.clone().count().await.unwrap(), 0);
        let my_tbl = session.table("my_tbl").await.unwrap();

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal("hello".into()),
                Expr::Literal(42_i32.into()),
            ]],
        };

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            my_tbl.schema().as_arrow(),
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
            .sql("INSERT INTO my_tbl VALUES ('hello', 42::INT);")
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
