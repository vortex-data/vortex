// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test that checks we can evolve schemas in a compatible way across files.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::datatypes::UInt16Type;
use datafusion::arrow::datatypes::UInt32Type;
use datafusion::assert_batches_sorted_eq;
use datafusion_common::assert_batches_eq;
use datafusion_common::create_array;
use datafusion_common::record_batch;
use datafusion_expr::col;
use datafusion_expr::lit;
use datafusion_functions::expr_fn::get_field;
use rstest::rstest;

use crate::common_tests::TestSessionContext;

#[rstest]
#[tokio::test]
async fn test_filter_with_schema_evolution(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // file1 only contains field "a"
    ctx.write_arrow_batch(
        "files/file1.vortex",
        &record_batch!(("a", Utf8, vec![Some("one"), Some("two"), Some("three")]))?,
    )
    .await?;

    // file2 only contains field "b"
    ctx.write_arrow_batch(
        "files/file2.vortex",
        &record_batch!(("b", Utf8, vec![Some("four"), Some("five"), Some("six")]))?,
    )
    .await?;

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl \
                STORED AS vortex  \
                LOCATION '/files/'",
        )
        .await?;

    let table = ctx.session.table("my_tbl").await?;

    // Table schema contains both fields
    assert_eq!(
        table.schema().as_arrow(),
        &Schema::new(vec![
            Field::new("a", DataType::Utf8View, true),
            Field::new("b", DataType::Utf8View, true),
        ])
    );

    // Filter the result to only ones with a column, i.e. only file1
    let result = table.filter(col("a").is_not_null())?.collect().await?;

    let expected = [
        "+-------+---+",
        "| a     | b |",
        "+-------+---+",
        "| one   |   |",
        "| three |   |",
        "| two   |   |",
        "+-------+---+",
    ];
    assert_batches_sorted_eq!(expected, &result);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_filter_schema_evolution_order(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // file1 only contains field "a"
    ctx.write_arrow_batch(
        "files/file1.vortex",
        &record_batch!(("a", Int32, vec![Some(1), Some(3), Some(5)]))?,
    )
    .await?;

    // file2 containing fields "b" and "a", where "a" needs to be upcast at scan time.
    ctx.write_arrow_batch(
        "files/file2.vortex",
        &record_batch!(
            ("b", Utf8, vec![Some("two"), Some("four"), Some("six")]),
            ("a", Int16, vec![Some(2), Some(4), Some(6)])
        )?,
    )
    .await?;

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl (a INT, b STRING) \
                STORED AS vortex  \
                LOCATION '/files/'",
        )
        .await?;

    let table = ctx.session.table("my_tbl").await?;

    // Table schema contains both fields
    assert_eq!(
        table.schema().as_arrow(),
        &Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ])
    );

    // Filter referencing the b column, which only appears in file2
    let result = table
        .clone()
        .filter(col("b").eq(lit("two")))?
        .collect()
        .await?;

    assert_batches_eq!(
        &[
            "+---+-----+",
            "| a | b   |",
            "+---+-----+",
            "| 2 | two |",
            "+---+-----+",
        ],
        &result
    );

    // Filter on the "a" column, which has different types for each file
    let result = table.filter(col("a").gt_eq(lit(3i16)))?.collect().await?;

    // a field: present in both files
    // b field: only present in file2, file1 fills with nulls
    assert_batches_sorted_eq!(
        &[
            "+---+------+",
            "| a | b    |",
            "+---+------+",
            "| 3 |      |",
            "| 4 | four |",
            "| 5 |      |",
            "| 6 | six  |",
            "+---+------+",
        ],
        &result
    );

    Ok(())
}

/// Test for correct schema evolution behavior in the presence of nested struct fields.
/// We use a hypothetical schema of some observability data with "wide records", struct columns
/// with nullable payloads that may or may not be present for every file.
#[rstest]
#[tokio::test]
async fn test_filter_schema_evolution_struct_fields(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    fn make_metrics(
        hostname: &str,
        uptime: Vec<i64>,
        instance: Option<Vec<Option<&str>>>,
    ) -> RecordBatch {
        let values_array: ArrowArrayRef = create_array!(Int64, uptime);
        let payload_array = if let Some(tags) = instance {
            let tags_array: ArrowArrayRef = create_array!(Utf8, tags);
            Arc::new(StructArray::new(
                vec![
                    Field::new("uptime", DataType::Int64, true),
                    Field::new("instance", DataType::Utf8, true),
                ]
                .into(),
                vec![values_array, tags_array],
                None,
            ))
        } else {
            Arc::new(StructArray::new(
                vec![Field::new("uptime", DataType::Int64, true)].into(),
                vec![values_array],
                None,
            ))
        };

        let len = payload_array.len();
        let hostname_array = create_array!(Utf8, vec![Some(hostname); len]);

        let payload_type = payload_array.data_type().clone();
        let hostname_type = hostname_array.data_type().clone();

        RecordBatch::from(StructArray::new(
            vec![
                Field::new("hostname", hostname_type, true),
                Field::new("payload", payload_type, true),
            ]
            .into(),
            vec![hostname_array, payload_array],
            None,
        ))
    }

    let host01 = make_metrics("host01.local", vec![1, 2, 3, 4], None);
    let host02 = make_metrics(
        "host02.local",
        vec![10, 20, 30, 40],
        // host02 has new logging code which adds the new "instance" nested field in its payload
        Some(vec![Some("c6i"), Some("c6i"), Some("m5"), Some("r5")]),
    );

    // Write metrics files to storage
    ctx.write_arrow_batch("files/host01.vortex", &host01)
        .await?;
    ctx.write_arrow_batch("files/host02.vortex", &host02)
        .await?;

    let read_schema = host02.schema();

    let provider = ctx
        .table_provider("tbl", "/files/", Arc::clone(&read_schema))
        .await?;

    let table = ctx.session.read_table(provider)?;

    // Table schema contains both fields
    assert_eq!(table.schema().as_arrow(), read_schema.as_ref(),);

    // Scan all the records, NULLs are filled in for nested optional fields.
    let full_scan = table.clone().collect().await?;

    assert_batches_sorted_eq!(
        &[
            "+--------------+-----------------------------+",
            "| hostname     | payload                     |",
            "+--------------+-----------------------------+",
            "| host01.local | {uptime: 1, instance: }     |",
            "| host01.local | {uptime: 2, instance: }     |",
            "| host01.local | {uptime: 3, instance: }     |",
            "| host01.local | {uptime: 4, instance: }     |",
            "| host02.local | {uptime: 10, instance: c6i} |",
            "| host02.local | {uptime: 20, instance: c6i} |",
            "| host02.local | {uptime: 30, instance: m5}  |",
            "| host02.local | {uptime: 40, instance: r5}  |",
            "+--------------+-----------------------------+",
        ],
        &full_scan
    );

    // run a filter that touches both the payload.uptime AND the payload.instance nested fields
    let filtered_scan = table
        .filter(
            // payload.instance = 'c6i' OR payload.uptime < 10
            // We need to perform filtering over nested columns which don't exist in every
            // file type.
            get_field(col("payload"), "instance")
                .eq(lit("c6i"))
                .or(get_field(col("payload"), "uptime").lt(lit(10))),
        )?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        &[
            "+--------------+-----------------------------+",
            "| hostname     | payload                     |",
            "+--------------+-----------------------------+",
            "| host01.local | {uptime: 1, instance: }     |",
            "| host01.local | {uptime: 2, instance: }     |",
            "| host01.local | {uptime: 3, instance: }     |",
            "| host01.local | {uptime: 4, instance: }     |",
            "| host02.local | {uptime: 10, instance: c6i} |",
            "| host02.local | {uptime: 20, instance: c6i} |",
            "+--------------+-----------------------------+",
        ],
        &filtered_scan
    );

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_schema_evolution_struct_of_dict(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // First file
    let struct_fields = Fields::from(vec![
        Field::new_dictionary("a", DataType::UInt16, DataType::Utf8, true),
        Field::new_dictionary("b", DataType::UInt16, DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(DictionaryArray::<UInt16Type>::from_iter(["x1", "y1", "x1"])),
            Arc::new(DictionaryArray::<UInt16Type>::from_iter(["p1", "p1", "q1"])),
        ],
        None,
    );

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(struct_fields),
            true,
        )])),
        vec![Arc::new(struct_array)],
    )?;

    ctx.write_arrow_batch("files/file1.vortex", &batch).await?;

    // Second file
    let struct_fields = Fields::from(vec![
        Field::new_dictionary("a", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("b", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("c", DataType::UInt32, DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["x2", "y2", "x2"])),
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["p2", "p2", "q2"])),
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["a2", "b2", "c2"])),
        ],
        None,
    );

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(struct_fields.clone()),
            true,
        )])),
        vec![Arc::new(struct_array)],
    )?;

    ctx.write_arrow_batch("files/file2.vortex", &batch).await?;

    let read_schema = batch.schema();

    let provider = ctx
        .table_provider("tbl", "/files/", Arc::clone(&read_schema))
        .await?;

    let table = ctx.session.read_table(provider)?;

    assert_eq!(table.schema().as_arrow(), read_schema.as_ref());

    let full_scan = table.clone().collect().await?;

    assert_batches_sorted_eq!(
        &[
            "+-----------------------+",
            "| my_struct             |",
            "+-----------------------+",
            "| {a: x1, b: p1, c: }   |",
            "| {a: x1, b: q1, c: }   |",
            "| {a: x2, b: p2, c: a2} |",
            "| {a: x2, b: q2, c: c2} |",
            "| {a: y1, b: p1, c: }   |",
            "| {a: y2, b: p2, c: b2} |",
            "+-----------------------+",
        ],
        &full_scan
    );

    let filter =
        get_field(col("my_struct"), "a")
            .eq(lit("x1"))
            .or(get_field(col("my_struct"), "a").eq(lit("x2")));
    // run a filter that touches both the payload.uptime AND the payload.instance nested fields

    let filtered_scan = table.filter(filter)?.collect().await?;

    assert_eq!(filtered_scan[0].schema(), read_schema);

    assert_batches_sorted_eq!(
        &[
            "+-----------------------+",
            "| my_struct             |",
            "+-----------------------+",
            "| {a: x1, b: p1, c: }   |",
            "| {a: x1, b: q1, c: }   |",
            "| {a: x2, b: p2, c: a2} |",
            "| {a: x2, b: q2, c: c2} |",
            "+-----------------------+",
        ],
        &filtered_scan
    );

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_schema_evolution_struct_field_order(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // File1: labels = {region, service} - service at position 1
    let file1_labels: ArrowArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![
            Field::new("region", DataType::Utf8, true),
            Field::new("service", DataType::Utf8, true),
        ]),
        vec![
            create_array!(Utf8, vec![Some("us-east"), Some("us-west")]),
            create_array!(Utf8, vec![Some("api"), Some("api")]),
        ],
        None,
    ));

    ctx.write_arrow_batch(
        "reorder/file1.vortex",
        &RecordBatch::try_from_iter([("labels", file1_labels)])?,
    )
    .await?;

    // File2: labels = {service, instance, job} - service at position 0
    let file2_labels: ArrowArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![
            Field::new("service", DataType::Utf8, true),
            Field::new("instance", DataType::Utf8, true),
            Field::new("job", DataType::Utf8, true),
        ]),
        vec![
            create_array!(Utf8, vec![Some("api"), Some("api")]),
            create_array!(Utf8, vec![Some("host-0"), Some("host-1")]),
            create_array!(Utf8, vec![Some("scraper"), Some("scraper")]),
        ],
        None,
    ));
    ctx.write_arrow_batch(
        "reorder/file2.vortex",
        &RecordBatch::try_from_iter([("labels", file2_labels)])?,
    )
    .await?;

    let target_schema = Arc::new(Schema::new(vec![Field::new(
        "labels",
        DataType::Struct(Fields::from(vec![
            Field::new("region", DataType::Utf8, true),
            Field::new("service", DataType::Utf8, true),
            Field::new("instance", DataType::Utf8, true),
            Field::new("job", DataType::Utf8, true),
        ])),
        true,
    )]));

    let table = ctx.table_provider("tbl", "/reorder", target_schema).await?;

    let result = ctx
        .session
        .read_table(table)?
        .select(vec![
            get_field(col("labels"), "region").alias("region"),
            get_field(col("labels"), "service").alias("service"),
            get_field(col("labels"), "instance").alias("instance"),
            get_field(col("labels"), "job").alias("job"),
        ])?
        .collect()
        .await?;

    assert_batches_sorted_eq!(
        [
            "+---------+---------+----------+---------+",
            "| region  | service | instance | job     |",
            "+---------+---------+----------+---------+",
            "| us-east | api     |          |         |",
            "| us-west | api     |          |         |",
            "|         | api     | host-0   | scraper |",
            "|         | api     | host-1   | scraper |",
            "+---------+---------+----------+---------+",
        ],
        &result
    );

    Ok(())
}

/// Test that complex projection expressions (arithmetic) work correctly
/// with both projection pushdown enabled and disabled.
#[rstest]
#[tokio::test]
async fn test_projection_expressions(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    ctx.write_arrow_batch(
        "files/data.vortex",
        &record_batch!(
            ("a", Int32, vec![Some(1), Some(2), Some(3)]),
            ("b", Int32, vec![Some(10), Some(20), Some(30)])
        )?,
    )
    .await?;

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl \
                STORED AS vortex  \
                LOCATION '/files/'",
        )
        .await?;

    let table = ctx.session.table("my_tbl").await?;

    // Test arithmetic projection: a + b * 2
    let result = table
        .select(vec![(col("a") + col("b") * lit(2)).alias("computed")])?
        .collect()
        .await?;

    assert_batches_eq!(
        &[
            "+----------+",
            "| computed |",
            "+----------+",
            "| 21       |",
            "| 42       |",
            "| 63       |",
            "+----------+",
        ],
        &result
    );

    Ok(())
}

/// Test that Dictionary columns are preserved correctly when scanning with a defined schema.
/// This reproduces an issue from the polarsignals benchmark where Dictionary(UInt32, Utf8)
/// columns were being returned as Utf8View.
#[rstest]
#[tokio::test]
async fn test_dictionary_column_type_preservation(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // Create a batch with Dictionary columns (like polarsignals schema)
    let dict_fields = Fields::from(vec![
        Field::new_dictionary("producer", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sample_type", DataType::UInt32, DataType::Utf8, false),
    ]);

    let producer_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "agent", "agent", "agent",
    ]));
    let sample_type_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "samples", "samples", "samples",
    ]));

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(dict_fields.to_vec())),
        vec![producer_array, sample_type_array],
    )?;

    ctx.write_arrow_batch("files/data.vortex", &batch).await?;

    // Create table with explicit schema that expects Dictionary types
    let table_schema = batch.schema();
    let provider = ctx
        .table_provider("tbl", "/files/", Arc::clone(&table_schema))
        .await?;

    let table = ctx.session.read_table(provider)?;

    // Verify the schema matches
    assert_eq!(table.schema().as_arrow(), table_schema.as_ref());

    // Query and verify the result schema preserves Dictionary types
    let result = table
        .filter(col("producer").eq(lit("agent")))?
        .collect()
        .await?;

    assert!(!result.is_empty(), "Expected results from query");

    // Check that the result schema preserves Dictionary types
    let result_schema = result[0].schema();
    assert_eq!(
        result_schema.field(0).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        "producer column should preserve Dictionary type"
    );
    assert_eq!(
        result_schema.field(1).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        "sample_type column should preserve Dictionary type"
    );

    Ok(())
}

/// Test that nested struct fields with Dictionary types are preserved correctly.
/// This reproduces the polarsignals benchmark issue where accessing `labels.comm`
/// (a Dictionary field inside a struct) returns Utf8View instead of Dictionary.
#[rstest]
#[tokio::test]
async fn test_nested_struct_dictionary_type_preservation(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // Create a struct with Dictionary fields (like polarsignals labels)
    let labels_fields = Fields::from(vec![
        Field::new_dictionary("comm", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("node", DataType::UInt32, DataType::Utf8, true),
    ]);

    let comm_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "proc_a", "proc_b", "proc_a",
    ]));
    let node_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "node_1", "node_1", "node_2",
    ]));

    let labels_struct = StructArray::new(labels_fields.clone(), vec![comm_array, node_array], None);

    // Add other columns like in polarsignals
    let value_array = create_array!(Int64, vec![Some(100i64), Some(200), Some(300)]);
    let producer_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "agent", "agent", "agent",
    ]));

    let schema = Arc::new(Schema::new(vec![
        Field::new("labels", DataType::Struct(labels_fields.clone()), false),
        Field::new("value", DataType::Int64, false),
        Field::new_dictionary("producer", DataType::UInt32, DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(labels_struct), value_array, producer_array],
    )?;

    ctx.write_arrow_batch("files/data.vortex", &batch).await?;

    let provider = ctx
        .table_provider("tbl", "/files/", Arc::clone(&schema))
        .await?;
    let table = ctx.session.read_table(provider)?;

    // Query that projects a nested struct field (like in polarsignals Q0)
    let result = table
        .clone()
        .filter(col("producer").eq(lit("agent")))?
        .select(vec![
            col("value"),
            get_field(col("labels"), "comm").alias("comm"),
        ])?
        .collect()
        .await?;

    assert!(!result.is_empty(), "Expected results from query");

    // The nested dictionary field should preserve its type
    let result_schema = result[0].schema();
    assert_eq!(
        result_schema.field(1).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        "labels.comm should preserve Dictionary type, got {:?}",
        result_schema.field(1).data_type()
    );

    Ok(())
}

/// Test reproducing the polarsignals benchmark schema with multiple dictionary columns
/// and filters on dictionary columns.
#[rstest]
#[tokio::test]
async fn test_polarsignals_like_schema(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // Create labels struct with dictionary fields
    let labels_fields = Fields::from(vec![Field::new_dictionary(
        "comm",
        DataType::UInt32,
        DataType::Utf8,
        true,
    )]);

    let comm_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "proc_a", "proc_b", "proc_a",
    ]));

    let labels_struct = StructArray::new(labels_fields.clone(), vec![comm_array], None);

    // Create multiple dictionary columns like polarsignals
    let value_array = create_array!(Int64, vec![Some(1i64), Some(2), Some(3)]);
    let producer_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "agent", "agent", "agent",
    ]));
    let sample_type_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "samples", "samples", "samples",
    ]));
    let sample_unit_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "count", "count", "count",
    ]));
    let period_type_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "cpu", "cpu", "cpu",
    ]));
    let period_unit_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "nanoseconds",
        "nanoseconds",
        "nanoseconds",
    ]));

    let schema = Arc::new(Schema::new(vec![
        Field::new("labels", DataType::Struct(labels_fields.clone()), false),
        Field::new("value", DataType::Int64, false),
        Field::new_dictionary("producer", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sample_type", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sample_unit", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("period_type", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("period_unit", DataType::UInt32, DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(labels_struct),
            value_array,
            producer_array,
            sample_type_array,
            sample_unit_array,
            period_type_array,
            period_unit_array,
        ],
    )?;

    ctx.write_arrow_batch("files/data.vortex", &batch).await?;

    let provider = ctx
        .table_provider("tbl", "/files/", Arc::clone(&schema))
        .await?;
    let table = ctx.session.read_table(provider)?;

    // Query like polarsignals Q0: filter on multiple dictionary columns, project value and labels.comm
    let result = table
        .filter(col("producer").eq(lit("agent")))?
        .filter(col("sample_type").eq(lit("samples")))?
        .filter(col("sample_unit").eq(lit("count")))?
        .filter(col("period_type").eq(lit("cpu")))?
        .filter(col("period_unit").eq(lit("nanoseconds")))?
        .select(vec![
            col("value"),
            get_field(col("labels"), "comm").alias("comm"),
        ])?
        .collect()
        .await?;

    assert!(!result.is_empty(), "Expected results from query");

    // Verify result values
    assert_batches_eq!(
        &[
            "+-------+--------+",
            "| value | comm   |",
            "+-------+--------+",
            "| 1     | proc_a |",
            "| 2     | proc_b |",
            "| 3     | proc_a |",
            "+-------+--------+",
        ],
        &result
    );

    Ok(())
}

/// Test using SQL to create an external table (closer to how benchmarks work).
/// This tests that Dictionary column types are preserved when using ListingTable.
#[rstest]
#[tokio::test]
async fn test_external_table_dictionary_columns(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // Create a simple batch with dictionary columns
    let producer_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "agent", "agent", "agent",
    ]));
    let sample_type_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "samples", "samples", "samples",
    ]));
    let value_array = create_array!(Int64, vec![Some(1i64), Some(2), Some(3)]);

    let schema = Arc::new(Schema::new(vec![
        Field::new_dictionary("producer", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sample_type", DataType::UInt32, DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![producer_array, sample_type_array, value_array],
    )?;

    ctx.write_arrow_batch("files/data.vortex", &batch).await?;

    // Use SQL to create external table (like the benchmark does)
    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE stacktraces \
                STORED AS vortex \
                LOCATION '/files/'",
        )
        .await?;

    // Query with filter on dictionary column and projection
    let result = ctx
        .session
        .sql(
            "SELECT value, sample_type FROM stacktraces \
             WHERE producer = 'agent' AND sample_type = 'samples'",
        )
        .await?
        .collect()
        .await?;

    assert!(!result.is_empty(), "Expected results from query");

    assert_batches_eq!(
        &[
            "+-------+-------------+",
            "| value | sample_type |",
            "+-------+-------------+",
            "| 1     | samples     |",
            "| 2     | samples     |",
            "| 3     | samples     |",
            "+-------+-------------+",
        ],
        &result
    );

    Ok(())
}

/// Test using SQL to access struct fields with dictionary types (like polarsignals Q0).
/// This reproduces the polarsignals benchmark error where `labels.comm` returns
/// Utf8View instead of Dictionary(UInt32, Utf8).
#[rstest]
#[tokio::test]
async fn test_sql_struct_field_dictionary_type(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);

    // Create labels struct with dictionary fields (like polarsignals)
    let labels_fields = Fields::from(vec![
        Field::new_dictionary("comm", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("node", DataType::UInt32, DataType::Utf8, true),
    ]);

    let comm_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "proc_a", "proc_b", "proc_a",
    ]));
    let node_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "node_1", "node_1", "node_2",
    ]));

    let labels_struct = StructArray::new(labels_fields.clone(), vec![comm_array, node_array], None);

    // Add other columns like in polarsignals
    let value_array = create_array!(Int64, vec![Some(1i64), Some(2), Some(3)]);
    let producer_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "agent", "agent", "agent",
    ]));
    let sample_type_array: ArrowArrayRef = Arc::new(DictionaryArray::<UInt32Type>::from_iter([
        "samples", "samples", "samples",
    ]));

    let schema = Arc::new(Schema::new(vec![
        Field::new("labels", DataType::Struct(labels_fields.clone()), false),
        Field::new("value", DataType::Int64, false),
        Field::new_dictionary("producer", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sample_type", DataType::UInt32, DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(labels_struct),
            value_array,
            producer_array,
            sample_type_array,
        ],
    )?;

    ctx.write_arrow_batch("files/data.vortex", &batch).await?;

    // Create table provider with explicit schema (like the benchmark)
    let provider = ctx.table_provider("stacktraces", "/files/", schema).await?;
    ctx.session.register_table("stacktraces", provider)?;

    // Query like polarsignals Q0: filter on dictionary columns, project struct field
    let result = ctx
        .session
        .sql(
            "SELECT value, labels.comm FROM stacktraces \
             WHERE producer = 'agent' AND sample_type = 'samples'",
        )
        .await?
        .collect()
        .await?;

    assert!(!result.is_empty(), "Expected results from query");

    // Verify the result values
    assert_batches_eq!(
        [
            "+-------+--------------------------+",
            "| value | stacktraces.labels[comm] |",
            "+-------+--------------------------+",
            "| 1     | proc_a                   |",
            "| 2     | proc_b                   |",
            "| 3     | proc_a                   |",
            "+-------+--------------------------+",
        ],
        &result
    );

    // Verify that labels.comm preserves Dictionary type
    let result_schema = result[0].schema();
    assert_eq!(
        result_schema.field(1).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        "labels.comm should preserve Dictionary type, got {:?}",
        result_schema.field(1).data_type()
    );

    Ok(())
}
