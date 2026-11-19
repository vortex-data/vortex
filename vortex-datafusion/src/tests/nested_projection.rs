// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test that projection pushdown works correctly with deeply nested structs.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::buffer::NullBuffer;
use datafusion_common::assert_batches_eq;
use datafusion_common::create_array;
use datafusion_expr::col;
use datafusion_expr::registry::FunctionRegistry;
use datafusion_functions::expr_fn::get_field;
use datafusion_functions_nested::planner::FieldAccessPlanner;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::dtype::Nullability;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::session::VortexSession;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;
use crate::tests::RecordingConverter;

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

/// A projection that mixes nested fields with an unsupported function over one of them reads only
/// those fields in the scan, once each, and DataFusion applies the function to the scan output.
///
/// The fields follow the converter's nested field access, as they do when the scan evaluates the
/// whole projection: a null `outer` or `inner` makes the fields under it null.
#[rstest]
#[tokio::test]
async fn test_mixed_projection_reads_needed_fields(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let converter = Arc::new(RecordingConverter::new(VortexSession::default()));
    let options = VortexTableOptions {
        projection_pushdown: true,
        scan_v2,
        ..Default::default()
    };
    let factory = VortexFormatFactory::new()
        .with_options(options)
        .with_expression_converter(Arc::clone(&converter) as _);
    let mut ctx = TestSessionContext::new_with_factory(Arc::new(factory));
    // The workspace builds DataFusion without its nested expressions, which plan SQL field access.
    ctx.session
        .register_expr_planner(Arc::new(FieldAccessPlanner))?;
    ctx.write_arrow_batch("files/nested.vortex", &make_nested_batch(true))
        .await?;
    ctx.session
        .sql("CREATE EXTERNAL TABLE nested STORED AS vortex LOCATION '/files/'")
        .await?;

    let batches = ctx
        .session
        .sql(
            "SELECT id, upper(\"outer\"['inner']['leaf']) AS shout, \
             \"outer\"['inner']['leaf'] AS leaf, \"outer\"['extra'] AS extra \
             FROM nested ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+-------+------+-------+",
            "| id | shout | leaf | extra |",
            "+----+-------+------+-------+",
            "| 1  | A     | a    | 100   |",
            "| 2  |       |      |       |",
            "| 3  |       |      | 300   |",
            "+----+-------+------+-------+",
        ],
        &batches
    );

    let outer = get_item("outer", root());
    let expected = pack(
        [
            ("id", get_item("id", root())),
            (
                "__vortex_expr_0",
                get_item("leaf", get_item("inner", outer.clone())),
            ),
            ("__vortex_expr_1", get_item("extra", outer)),
        ],
        Nullability::NonNullable,
    );
    let scan_projections = converter.scan_projections.lock();
    assert!(!scan_projections.is_empty());
    for projection in scan_projections.iter() {
        assert_eq!(projection, &expected);
    }
    Ok(())
}
