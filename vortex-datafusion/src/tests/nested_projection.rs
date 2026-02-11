// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test that projection pushdown works correctly with deeply nested structs.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion_common::assert_batches_eq;
use datafusion_common::create_array;
use datafusion_expr::col;
use datafusion_functions::expr_fn::get_field;
use rstest::rstest;

use crate::common_tests::TestSessionContext;

/// Schema: {id: Int64, outer: {inner: {leaf: Utf8, value: Int64}, extra: Int64}}
fn make_nested_batch() -> RecordBatch {
    let leaf_array: ArrowArrayRef = create_array!(Utf8, vec![Some("a"), Some("b"), Some("c")]);
    let value_array: ArrowArrayRef = create_array!(Int64, vec![10i64, 20, 30]);

    let inner_struct: ArrowArrayRef = Arc::new(StructArray::new(
        vec![
            Field::new("leaf", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
        ]
        .into(),
        vec![leaf_array, value_array],
        None,
    ));

    let extra_array: ArrowArrayRef = create_array!(Int64, vec![100i64, 200, 300]);

    let outer_struct: ArrowArrayRef = Arc::new(StructArray::new(
        vec![
            Field::new("inner", inner_struct.data_type().clone(), true),
            Field::new("extra", DataType::Int64, true),
        ]
        .into(),
        vec![inner_struct, extra_array],
        None,
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

    let batch = make_nested_batch();
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

    let batch = make_nested_batch();
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
