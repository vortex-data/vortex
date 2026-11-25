// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test that projection pushdown works correctly with deeply nested structs.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::buffer::NullBuffer;
use datafusion_common::Result as DFResult;
use datafusion_common::assert_batches_eq;
use datafusion_common::create_array;
use datafusion_expr::col;
use datafusion_functions::expr_fn::get_field;
use datafusion_functions::expr_fn::upper;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::projection::ProjectionExprs;
use parking_lot::Mutex;
use rstest::rstest;
use vortex::expr::Expression;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;
use crate::convert::ProcessedProjection;

/// Schema: {id: Int64, outer: {inner: {leaf: Utf8, value: Int64}, extra: Int64}}
fn make_nested_batch(parent_nulls: bool) -> RecordBatch {
    let leaf_array: ArrowArrayRef = create_array!(Utf8, vec![Some("a"), Some("b"), Some("c")]);
    let value_array: ArrowArrayRef = create_array!(Int64, vec![10i64, 20, 30]);

    let inner_struct: ArrowArrayRef = Arc::new(StructArray::new(
        vec![
            Field::new("leaf", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
        ]
        .into(),
        vec![leaf_array, value_array],
        parent_nulls.then(|| NullBuffer::from(vec![true, true, false])),
    ));

    let extra_array: ArrowArrayRef = create_array!(Int64, vec![100i64, 200, 300]);

    let outer_struct: ArrowArrayRef = Arc::new(StructArray::new(
        vec![
            Field::new("inner", inner_struct.data_type().clone(), true),
            Field::new("extra", DataType::Int64, true),
        ]
        .into(),
        vec![inner_struct, extra_array],
        parent_nulls.then(|| NullBuffer::from(vec![true, false, true])),
    ));

    let id_array: ArrowArrayRef = create_array!(Int64, vec![1i64, 2, 3]);
    RecordBatch::try_from_iter(vec![("id", id_array), ("outer", outer_struct)]).unwrap()
}

/// Test projecting a leaf field from a deeply nested struct (root.outer.inner.leaf).
#[rstest]
#[tokio::test]
async fn test_nested_struct_leaf_projection(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    let batch = make_nested_batch(false);
    ctx.write_arrow_batch("files/nested.vortex", &batch).await?;

    let schema = batch.schema();
    let provider = ctx
        .table_provider("nested_tbl", "/files/", schema.as_ref().clone())
        .await?;

    let table = ctx.session.read_table(provider)?;

    let result = table
        .select(vec![
            col("id"),
            get_field(get_field(col("outer"), "inner"), "leaf").alias("leaf"),
        ])?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+----+------+",
            "| id | leaf |",
            "+----+------+",
            "| 1  | a    |",
            "| 2  | b    |",
            "| 3  | c    |",
            "+----+------+",
        ],
        &result
    );

    Ok(())
}

/// Test projecting a mid-level struct from a nested struct (root.outer.inner).
#[rstest]
#[tokio::test]
async fn test_nested_struct_mid_level_projection(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    let batch = make_nested_batch(false);
    ctx.write_arrow_batch("files/nested.vortex", &batch).await?;

    let schema = batch.schema();
    let provider = ctx
        .table_provider("nested_tbl", "/files/", schema.as_ref().clone())
        .await?;

    let table = ctx.session.read_table(provider)?;

    let result = table
        .select(vec![
            col("id"),
            get_field(col("outer"), "inner").alias("inner"),
        ])?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+----+----------------------+",
            "| id | inner                |",
            "+----+----------------------+",
            "| 1  | {leaf: a, value: 10} |",
            "| 2  | {leaf: b, value: 20} |",
            "| 3  | {leaf: c, value: 30} |",
            "+----+----------------------+",
        ],
        &result
    );

    Ok(())
}

#[tokio::test]
async fn test_nested_struct_null_parent_projection() -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(true);
    let batch = make_nested_batch(true);
    ctx.write_arrow_batch("files/nested.vortex", &batch).await?;
    let provider = ctx
        .table_provider("nested_tbl", "/files/", batch.schema().as_ref().clone())
        .await?;

    let inner = get_field(col("outer"), "inner");
    let result = ctx
        .session
        .read_table(provider)?
        .select(vec![
            col("id"),
            get_field(inner.clone(), "leaf").alias("leaf"),
            get_field(inner.clone(), "value").alias("value"),
            inner.alias("inner"),
        ])?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+----+------+-------+----------------------+",
            "| id | leaf | value | inner                |",
            "+----+------+-------+----------------------+",
            "| 1  | a    | 10    | {leaf: a, value: 10} |",
            "| 2  |      |       |                      |",
            "| 3  |      |       |                      |",
            "+----+------+-------+----------------------+",
        ],
        &result
    );
    Ok(())
}

/// The default converter, recording the scan projection of each projection that it keeps in
/// DataFusion.
#[derive(Default)]
struct RawScanRecorder {
    inner: DefaultExpressionConverter,
    scan_projections: Mutex<Vec<String>>,
}

impl ExpressionConverter for RawScanRecorder {
    fn try_convert(&self, expr: &PhysicalExprRef, schema: &Schema) -> DFResult<Option<Expression>> {
        self.inner.try_convert(expr, schema)
    }

    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let processed = self
            .inner
            .no_pushdown_projection(source_projection, input_schema)?;
        self.scan_projections
            .lock()
            .push(processed.scan_projection.to_string());
        Ok(processed)
    }
}

/// DataFusion flattens the nested field access of a query before the file scan sees it, so the
/// projection that stays in DataFusion reads only the leaf of each struct.
#[rstest]
#[tokio::test]
async fn test_nested_struct_leaf_projection_reads_only_leaf(
    #[values(false, true)] parent_nulls: bool,
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let converter = Arc::new(RawScanRecorder::default());
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new()
            .with_options(VortexTableOptions {
                projection_pushdown: true,
                scan_v2,
                ..Default::default()
            })
            .with_expression_converter(Arc::<RawScanRecorder>::clone(&converter)),
    ));

    let batch = make_nested_batch(parent_nulls);
    ctx.write_arrow_batch("files/nested.vortex", &batch).await?;
    let provider = ctx
        .table_provider("nested_tbl", "/files/", batch.schema().as_ref().clone())
        .await?;

    let result = ctx
        .session
        .read_table(provider)?
        .select(vec![
            col("id"),
            upper(get_field(get_field(col("outer"), "inner"), "leaf")).alias("leaf"),
        ])?
        .collect()
        .await?;

    let scan_projections = converter.scan_projections.lock().clone();
    assert!(!scan_projections.is_empty());
    for scan_projection in scan_projections {
        assert_eq!(
            scan_projection,
            "pack(id: $.id, outer: pack(inner: $.outer.inner{leaf}))"
        );
    }

    let (second_row, third_row) = if parent_nulls {
        ("| 2  |      |", "| 3  |      |")
    } else {
        ("| 2  | B    |", "| 3  | C    |")
    };
    assert_batches_eq!(
        [
            "+----+------+",
            "| id | leaf |",
            "+----+------+",
            "| 1  | A    |",
            second_row,
            third_row,
            "+----+------+",
        ],
        &result
    );

    Ok(())
}
