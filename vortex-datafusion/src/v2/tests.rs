// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion::assert_batches_eq;
use datafusion::prelude::SessionContext;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::buffer::ByteBufferMut;
use vortex::expr::Expression;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use super::VortexDataSource;
use super::source::VortexDataSourceBuilder;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;

async fn vortex_source() -> anyhow::Result<VortexDataSource> {
    let batch =
        arrow_array::record_batch!(("a", Int32, vec![1, 2, 3]), ("b", Int32, vec![30, 10, 20]))?;
    vortex_source_from_batch(batch).await
}

async fn vortex_source_from_batch(batch: RecordBatch) -> anyhow::Result<VortexDataSource> {
    Ok(vortex_source_builder(batch).await?.build().await?)
}

async fn vortex_source_builder(batch: RecordBatch) -> anyhow::Result<VortexDataSourceBuilder> {
    let session = VortexSession::default();
    let schema = batch.schema();
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buffer)?;
    Ok(VortexDataSource::builder(file.data_source()?, session).with_arrow_schema(schema))
}

#[tokio::test]
async fn projection_filters_use_scan_schema() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("b", 1)), "b"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                Operator::Modulo,
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "odd",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let supported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("b", 0)),
        Operator::Gt,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(15)))),
    ));
    let unsupported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("odd", 1)),
        Operator::Eq,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
    ));

    let pushed =
        projected.try_pushdown_filters(vec![supported, unsupported], &ConfigOptions::new())?;
    assert!(matches!(
        pushed.filters.as_slice(),
        [PushedDown::Yes, PushedDown::No]
    ));
    let updated = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not added to the source"))?;
    let plan = Arc::new(DataSourceExec::new(updated)) as Arc<dyn ExecutionPlan>;
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert_batches_eq!(
        [
            "+----+-----+",
            "| b  | odd |",
            "+----+-----+",
            "| 30 | 1   |",
            "| 20 | 1   |",
            "+----+-----+",
        ],
        &batches
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn projection_preserves_arrow_types(
    #[values(false, true)] fallback: bool,
) -> anyhow::Result<()> {
    let batch = arrow_array::record_batch!(
        ("a", Int32, vec![1, 2, 3]),
        ("s", Utf8, vec![Some("one"), None, Some("three")])
    )?;
    let source = vortex_source_from_batch(batch).await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("s", 1)), "text"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                if fallback {
                    Operator::Modulo
                } else {
                    Operator::Plus
                },
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "n",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = Arc::new(DataSourceExec::new(projected)) as Arc<dyn ExecutionPlan>;
    let expected_schema = plan.schema();
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    for batch in &batches {
        assert_eq!(batch.schema(), expected_schema);
    }
    let expected = if fallback {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![1, 0, 1])
        )?
    } else {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![3, 4, 5])
        )?
    };
    let actual = concat_batches(&expected_schema, &batches)?;
    assert_eq!(actual.columns(), expected.columns());
    Ok(())
}

struct CountingConverter {
    calls: Arc<AtomicUsize>,
    inner: DefaultExpressionConverter,
}

impl ExpressionConverter for CountingConverter {
    fn try_convert(
        &self,
        expression: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.try_convert(expression, schema)
    }
}

#[rstest]
#[tokio::test]
async fn expression_converter_survives_pushdown_updates(
    #[values(false, true)] custom: bool,
) -> anyhow::Result<()> {
    let batch = arrow_array::record_batch!(("a", Int32, vec![Some(1), None, Some(3)]))?;
    let calls = Arc::new(AtomicUsize::new(0));
    let builder = vortex_source_builder(batch).await?;
    let builder = if custom {
        builder.with_expression_converter(Arc::new(CountingConverter {
            calls: Arc::clone(&calls),
            inner: DefaultExpressionConverter::default(),
        }))
    } else {
        builder
    };
    let source = builder.build().await?;
    let source = source
        .with_fetch(None)
        .ok_or_else(|| anyhow!("fetch update was not supported"))?;
    let projection = ProjectionExprs::from(vec![ProjectionExpr::new(
        Arc::new(df_expr::BinaryExpr::new(
            Arc::new(df_expr::Column::new("a", 0)),
            Operator::Plus,
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
        )),
        "incremented",
    )]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    assert_eq!(calls.load(Ordering::Relaxed), usize::from(custom));
    let projected = projected
        .repartitioned(2, 0, None)?
        .ok_or_else(|| anyhow!("repartitioning was not supported"))?;
    let filter: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("incremented", 0)),
        Operator::Gt,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
    ));
    let pushed = projected.try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert_eq!(calls.load(Ordering::Relaxed), 2 * usize::from(custom));
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let updated = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not added to the source"))?;
    let plan = Arc::new(DataSourceExec::new(updated)) as Arc<dyn ExecutionPlan>;
    let context = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, context.task_ctx()).await?;
    assert_batches_eq!(
        [
            "+-------------+",
            "| incremented |",
            "+-------------+",
            "| 4           |",
            "+-------------+",
        ],
        &batches
    );
    Ok(())
}
