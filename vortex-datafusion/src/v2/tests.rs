// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use arrow_array::ArrayRef;
use arrow_array::Decimal128Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use datafusion::assert_batches_eq;
use datafusion::prelude::SessionContext;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::cast;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use futures::TryStreamExt;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::buffer::ByteBufferMut;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
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
async fn projection_statistics_follow_reordered_and_aliased_columns(
    #[values(false, true)] fallback: bool,
) -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let initial = source.partition_statistics(None)?;
    assert_eq!(
        initial.column_statistics[0].min_value,
        Precision::Exact(ScalarValue::Int32(Some(1)))
    );
    assert_eq!(
        initial.column_statistics[1].max_value,
        Precision::Exact(ScalarValue::Int32(Some(30)))
    );
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("b", 1)), "renamed_b"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                if fallback {
                    Operator::Modulo
                } else {
                    Operator::Plus
                },
                df_expr::lit(2_i32),
            )),
            "computed",
        ),
        ProjectionExpr::new(Arc::new(df_expr::Column::new("a", 0)), "renamed_a"),
    ]);
    let projected = source.try_swapping_with_projection(&projection)?.unwrap();
    let statistics = projected.partition_statistics(None)?;
    assert_eq!(
        statistics.column_statistics,
        vec![
            initial.column_statistics[1].clone(),
            ColumnStatistics::new_unknown(),
            initial.column_statistics[0].clone(),
        ]
    );
    let pushed = projected
        .try_pushdown_filters(
            vec![Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("renamed_b", 0)),
                Operator::Gt,
                df_expr::lit(15_i32),
            ))],
            &ConfigOptions::new(),
        )?
        .updated_node
        .unwrap();
    let filtered = pushed.partition_statistics(None)?;
    assert_eq!(filtered.num_rows, Precision::Inexact(3));
    assert_eq!(
        filtered.column_statistics[0].min_value,
        Precision::Inexact(ScalarValue::Int32(Some(10)))
    );
    assert_eq!(
        filtered.column_statistics[2].max_value,
        Precision::Inexact(ScalarValue::Int32(Some(3)))
    );
    assert!(
        projected
            .try_swapping_with_projection(&ProjectionExprs::from(vec![
                ProjectionExpr::new(Arc::new(df_expr::Column::new("computed", 1)), "first"),
                ProjectionExpr::new(Arc::new(df_expr::Column::new("computed", 1)), "second"),
            ]))?
            .is_none()
    );
    let repeated = projected
        .try_swapping_with_projection(&ProjectionExprs::from(vec![
            ProjectionExpr::new(Arc::new(df_expr::Column::new("renamed_a", 2)), "a_again"),
            ProjectionExpr::new(
                Arc::new(df_expr::Column::new("computed", 1)),
                "computed_again",
            ),
            ProjectionExpr::new(Arc::new(df_expr::Column::new("renamed_b", 0)), "b_again"),
        ]))?
        .unwrap();
    assert_eq!(
        repeated.partition_statistics(None)?.column_statistics,
        vec![
            initial.column_statistics[0].clone(),
            ColumnStatistics::new_unknown(),
            initial.column_statistics[1].clone(),
        ]
    );
    let plan = Arc::new(DataSourceExec::new(repeated)) as Arc<dyn ExecutionPlan>;
    let schema = plan.schema();
    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;
    let actual = concat_batches(&schema, &batches)?;
    let computed = if fallback {
        vec![1, 0, 1]
    } else {
        vec![3, 4, 5]
    };
    let expected = arrow_array::record_batch!(
        ("a_again", Int32, vec![1, 2, 3]),
        ("computed_again", Int32, computed),
        ("b_again", Int32, vec![30, 10, 20])
    )?;
    assert_eq!(actual.columns(), expected.columns());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn projection_preserves_arrow_types(
    #[values(false, true)] fallback: bool,
    #[values(false, true)] string_view: bool,
) -> anyhow::Result<()> {
    let batch = arrow_array::record_batch!(
        ("a", Int32, vec![1, 2, 3]),
        ("s", Utf8, vec![Some("one"), None, Some("three")])
    )?;
    let batch = if string_view {
        RecordBatch::try_from_iter([
            ("a", Arc::clone(batch.column(0))),
            ("s", cast(batch.column(1), &DataType::Utf8View)?),
        ])?
    } else {
        batch
    };
    let source = vortex_source_from_batch(batch).await?;
    let expected_type = if string_view {
        DataType::Utf8View
    } else {
        DataType::Utf8
    };
    let initial = source.partition_statistics(None)?;
    for bound in [
        &initial.column_statistics[1].min_value,
        &initial.column_statistics[1].max_value,
    ] {
        assert_eq!(
            bound.get_value().map(ScalarValue::data_type),
            Some(expected_type.clone())
        );
    }
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
    assert_eq!(
        projected.partition_statistics(None)?.column_statistics[0],
        initial.column_statistics[1]
    );
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
    let expected = RecordBatch::try_new(
        Arc::clone(&expected_schema),
        vec![
            cast(expected.column(0), &expected_type)?,
            Arc::clone(expected.column(1)),
        ],
    )?;
    let actual = concat_batches(&expected_schema, &batches)?;
    assert_eq!(actual.columns(), expected.columns());
    Ok(())
}

#[tokio::test]
async fn statistics_preserve_arrow_decimal_width() -> anyhow::Result<()> {
    let amounts = Decimal128Array::from(vec![Some(123_i128), None, Some(-456)])
        .with_precision_and_scale(10, 2)?;
    let batch = RecordBatch::try_from_iter([("amount", Arc::new(amounts) as ArrayRef)])?;
    let source = vortex_source_from_batch(batch).await?;
    let statistics = source.partition_statistics(None)?;
    assert_eq!(
        statistics.column_statistics[0].min_value.get_value(),
        Some(&ScalarValue::Decimal128(Some(-456), 10, 2)),
    );
    assert_eq!(
        statistics.column_statistics[0].max_value.get_value(),
        Some(&ScalarValue::Decimal128(Some(123), 10, 2)),
    );
    Ok(())
}

#[tokio::test]
async fn statistics_follow_filter_and_fetch() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let filter = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("a", 0)),
        Operator::Gt,
        df_expr::lit(1_i32),
    ));
    let filtered = source
        .try_pushdown_filters(vec![filter], &ConfigOptions::new())?
        .updated_node
        .expect("filter accepted");
    let limited = filtered.with_fetch(Some(1)).expect("fetch accepted");
    assert_eq!(
        limited.partition_statistics(None)?.num_rows,
        Precision::Inexact(1)
    );
    let rejected = limited.try_pushdown_filters(
        vec![
            df_expr::lit(false),
            Arc::new(DynamicFilterPhysicalExpr::new(vec![], df_expr::lit(false))),
        ],
        &ConfigOptions::new(),
    )?;
    assert!(matches!(
        rejected.filters.as_slice(),
        [PushedDown::No, PushedDown::No]
    ));
    assert!(rejected.updated_node.is_none());
    Ok(())
}

#[rstest]
#[case(Operator::And, vec![30, 20], vec![30], vec![20])]
#[case(Operator::Or, vec![30, 10, 20], vec![30, 20], vec![30, 20])]
#[tokio::test]
async fn dynamic_filters_refresh_projected_bounds_and_keep_residuals(
    #[case] operator: Operator,
    #[case] unrestricted: Vec<i32>,
    #[case] above_bound: Vec<i32>,
    #[case] equal_bound: Vec<i32>,
) -> anyhow::Result<()> {
    let source = vortex_source()
        .await?
        .try_swapping_with_projection(&ProjectionExprs::from(vec![ProjectionExpr::new(
            Arc::new(df_expr::Column::new("b", 1)),
            "value",
        )]))?
        .unwrap();
    let column = Arc::new(df_expr::Column::new("value", 0)) as Arc<dyn PhysicalExpr>;
    let above = |value| {
        Arc::new(df_expr::BinaryExpr::new(
            Arc::clone(&column),
            Operator::Gt,
            df_expr::lit(value),
        )) as Arc<dyn PhysicalExpr>
    };
    let bound = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&column)],
        df_expr::lit(true),
    ));
    let pushed = source.try_pushdown_filters(
        vec![Arc::new(df_expr::BinaryExpr::new(
            above(15_i32),
            operator,
            Arc::clone(&bound) as Arc<dyn PhysicalExpr>,
        ))],
        &ConfigOptions::new(),
    )?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::No]));
    let source = pushed.updated_node.unwrap();
    assert!(source.with_fetch(Some(1)).is_none());
    let mut visited = false;
    source.apply_expressions(&mut |expression| {
        expression.apply(|expr| {
            visited |= expr.is::<DynamicFilterPhysicalExpr>();
            Ok(TreeNodeRecursion::Continue)
        })
    })?;
    assert!(visited);
    // An unsupported dynamic predicate must not erase the static AND restriction.
    let unsupported = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::BinaryExpr::new(
            Arc::clone(&column),
            Operator::Modulo,
            df_expr::lit(2_i32),
        )),
        Operator::Eq,
        df_expr::lit(1_i32),
    ));
    for (predicate, expected) in [
        (df_expr::lit(true), unrestricted.clone()),
        (above(25_i32), above_bound),
        (
            Arc::new(df_expr::BinaryExpr::new(
                Arc::clone(&column),
                Operator::Eq,
                df_expr::lit(20_i32),
            )) as Arc<dyn PhysicalExpr>,
            equal_bound,
        ),
        (unsupported as Arc<dyn PhysicalExpr>, unrestricted),
    ] {
        let stream = source.open(0, SessionContext::new().task_ctx())?;
        let schema = stream.schema();
        bound.update(predicate)?;
        let batch = concat_batches(&schema, &stream.try_collect::<Vec<_>>().await?)?;
        assert_eq!(
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .as_ref(),
            expected
        );
    }
    Ok(())
}
