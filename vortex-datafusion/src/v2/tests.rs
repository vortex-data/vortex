// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::assert_batches_eq;
use datafusion::prelude::SessionContext;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
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
use vortex::array::dtype::DType;
use vortex::array::dtype::FieldPath;
use vortex::array::expr::stats::Precision as VortexPrecision;
use vortex::array::expr::stats::Stat;
use vortex::array::scalar::Scalar;
use vortex::array::stats::StatsSet;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::scan::DataSource as ScanDataSource;
use vortex::scan::DataSourceScanRef;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use super::VortexDataSource;

async fn vortex_source() -> anyhow::Result<VortexDataSource> {
    let batch =
        arrow_array::record_batch!(("a", Int32, vec![1, 2, 3]), ("b", Int32, vec![30, 10, 20]))?;
    vortex_source_from_batch(batch).await
}

async fn vortex_source_from_batch(batch: RecordBatch) -> anyhow::Result<VortexDataSource> {
    let session = VortexSession::default();
    let schema = batch.schema();
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buffer)?;
    Ok(VortexDataSource::builder(file.data_source()?, session)
        .with_arrow_schema(schema)
        .build()
        .await?)
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

struct StatisticsSource {
    dtype: DType,
}

#[async_trait]
impl ScanDataSource for StatisticsSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> VortexPrecision<u64> {
        VortexPrecision::Exact(3)
    }

    async fn scan(&self, _request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        vortex_bail!("statistics-only test source")
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        if field_path != &FieldPath::from_name("a") {
            return Ok(StatsSet::default());
        }
        let exact = |value: Scalar| {
            VortexPrecision::Exact(value.into_value().expect("non-null fixture statistic"))
        };
        Ok([
            (Stat::Min, exact(Scalar::from(1_i32))),
            (Stat::Max, exact(Scalar::from(3_i32))),
            (Stat::NullCount, exact(Scalar::from(1_u64))),
        ]
        .into_iter()
        .collect())
    }
}

async fn source_with_exact_statistics() -> anyhow::Result<VortexDataSource> {
    let session = VortexSession::default();
    let batch = arrow_array::record_batch!(("a", Int32, vec![Some(1), None, Some(3)]))?;
    let array = session
        .arrow()
        .from_arrow_record_batch(batch.clone(), &batch.schema())?;
    Ok(VortexDataSource::builder(
        Arc::new(StatisticsSource {
            dtype: array.dtype().clone(),
        }),
        session,
    )
    .build()
    .await?)
}

#[rstest]
#[case(false, None, Precision::Exact(3), true)]
#[case(false, Some(3), Precision::Exact(3), true)]
#[case(false, Some(1), Precision::Exact(1), false)]
#[case(false, Some(0), Precision::Exact(0), false)]
#[case(true, None, Precision::Inexact(3), false)]
#[case(true, Some(1), Precision::Inexact(1), false)]
#[tokio::test]
async fn subset_statistics_follow_filter_and_fetch(
    #[case] filter: bool,
    #[case] fetch: Option<usize>,
    #[case] rows: Precision<usize>,
    #[case] exact_columns: bool,
) -> anyhow::Result<()> {
    let original = source_with_exact_statistics().await?;
    let source: Arc<dyn DataSource> = if filter {
        let filter: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
            Arc::new(df_expr::Column::new("a", 0)),
            Operator::Gt,
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        original
            .try_pushdown_filters(vec![filter], &ConfigOptions::new())?
            .updated_node
            .ok_or_else(|| anyhow!("comparison was not pushed down"))?
    } else {
        Arc::new(original.clone())
    };
    let source = source
        .with_fetch(fetch)
        .ok_or_else(|| anyhow!("fetch was not accepted"))?;
    let statistics = source.partition_statistics(None)?;
    assert_eq!(statistics.num_rows, rows);
    assert_eq!(statistics.column_statistics.len(), 1);
    let column = &statistics.column_statistics[0];
    assert_eq!(column.min_value.is_exact(), Some(exact_columns));
    assert_eq!(column.max_value.is_exact(), Some(exact_columns));
    assert_eq!(column.null_count.is_exact(), Some(exact_columns));
    assert_eq!(
        column.min_value.get_value(),
        Some(&ScalarValue::Int32(Some(1)))
    );
    assert_eq!(
        column.max_value.get_value(),
        Some(&ScalarValue::Int32(Some(3)))
    );
    assert_eq!(column.null_count.get_value(), Some(&1));
    assert_eq!(
        original.partition_statistics(None)?.num_rows,
        Precision::Exact(3)
    );
    Ok(())
}

#[tokio::test]
async fn projected_statistics_retain_filter_and_fetch_state() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let filter: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("a", 0)),
        Operator::Gt,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
    ));
    let filtered = source
        .try_pushdown_filters(vec![filter], &ConfigOptions::new())?
        .updated_node
        .ok_or_else(|| anyhow!("comparison was not pushed down"))?;
    let fetched = filtered
        .with_fetch(Some(1))
        .ok_or_else(|| anyhow!("fetch was not accepted"))?;
    let projection = ProjectionExprs::from(vec![ProjectionExpr::new(
        Arc::new(df_expr::Column::new("b", 1)),
        "renamed",
    )]);
    let projected = fetched
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    assert_eq!(projected.fetch(), Some(1));
    let statistics = projected.partition_statistics(None)?;
    assert_eq!(statistics.num_rows, Precision::Inexact(1));
    assert_eq!(statistics.column_statistics.len(), 1);
    assert_eq!(statistics.column_statistics[0].min_value, Precision::Absent);

    let no_fetch = projected
        .with_fetch(None)
        .ok_or_else(|| anyhow!("fetch was not removed"))?;
    assert_eq!(
        no_fetch.partition_statistics(None)?.num_rows,
        Precision::Inexact(3)
    );
    assert_eq!(
        fetched.partition_statistics(None)?.num_rows,
        Precision::Inexact(1)
    );
    assert_eq!(
        source.partition_statistics(None)?.num_rows,
        Precision::Exact(3)
    );
    Ok(())
}
