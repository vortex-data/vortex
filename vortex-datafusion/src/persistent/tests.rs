// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use insta::assert_snapshot;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::editions::CORE_2026_08_3;
use vortex::editions::EditionSessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::Handle;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::session::VortexSession;

use crate::VortexFormatFactory;
use crate::common_tests::TestSessionContext;

fn make_session(
    object_store: Arc<dyn ObjectStore>,
    repartition_file_scans: bool,
) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());

    let config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_scans(repartition_file_scans)
        .with_repartition_file_min_size(0);
    let mut state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://").unwrap(), object_store);

    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }

    SessionContext::new_with_state(state.build()).enable_url_table()
}

async fn count_query_partitions(ctx: &SessionContext, sql: &str) -> anyhow::Result<usize> {
    let explain = ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    let marker = "DataSourceExec: file_groups={";
    let start = plan
        .find(marker)
        .ok_or_else(|| anyhow!("EXPLAIN plan did not contain a DataSourceExec"))?
        + marker.len();
    let partitions = plan[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();

    Ok(partitions.parse()?)
}

fn batch_values(batches: &[RecordBatch]) -> Vec<i32> {
    let mut values = Vec::with_capacity(batches.iter().map(|batch| batch.num_rows()).sum());

    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("value column should be Int32");
        values.extend(array.values().iter().copied());
    }

    values
}

#[rstest]
#[tokio::test]
async fn test_query_file(#[values(Some(1), None)] limit: Option<usize>) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    let session = VortexSession::default();

    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::try_new(
        ["strings", "numbers"].into(),
        vec![strings, numbers],
        8,
        Validity::NonNullable,
    )?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"test.vortex".into()).await?;

    let summary = session
        .write_options()
        .write(&mut writer, st.into_array().to_array_stream())
        .await?;

    writer.shutdown().await?;

    assert_eq!(summary.row_count(), 8);

    let read_row_count = ctx
        .session
        .sql("SELECT * from '/test.vortex'")
        .await?
        .limit(0, limit)?
        .count()
        .await?;

    assert_eq!(read_row_count, limit.unwrap_or(8));

    Ok(())
}

#[tokio::test]
async fn test_addition_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_data VALUES (0), (1), (2), (3), (4)")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql("SELECT a, a + 5 as five, a + 6 as six FROM written_data WHERE a + 5 > 7")
        .await?
        .collect()
        .await?;

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---+------+-----+
        | a | five | six |
        +---+------+-----+
        | 3 | 8    | 9   |
        | 4 | 9    | 10  |
        +---+------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_octet_length_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(true);

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_strings \
                    (s VARCHAR NOT NULL) \
                STORED AS vortex \
                LOCATION '/strings/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_strings VALUES ('a'), ('é'), ('abcd'), ('')")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql(
            "SELECT s, octet_length(s) AS len \
             FROM written_strings \
             WHERE octet_length(s) > 1 \
             ORDER BY s",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(
        result[0].schema().field_with_name("len")?.data_type(),
        &DataType::Int32
    );
    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +------+-----+
        | s    | len |
        +------+-----+
        | abcd | 4   |
        | é    | 2   |
        +------+-----+
        ");

    Ok(())
}

/// Lambda parameters are indexed past the projection's input columns, so splitting a projection
/// that contains lambdas must not reshape the scan output.
#[tokio::test]
async fn test_lambda_projection_with_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(true);
    datafusion_functions_nested::register_all(&mut *ctx.session.state_ref().write())?;

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_lists \
                    (id INT NOT NULL, xs INT[]) \
                STORED AS vortex \
                LOCATION '/lists/'",
        )
        .await?;

    ctx.session
        .sql(
            "INSERT INTO written_lists VALUES \
                (1, make_array(1, NULL, 2)), \
                (2, make_array(0, 0))",
        )
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("SET datafusion.sql_parser.dialect = 'duckdb'")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql(
            "SELECT id, \
                    array_sum(array_transform(xs, lambda x: x IS NOT NULL)) AS n_valid, \
                    array_sum(xs) AS total, \
                    array_transform(xs, lambda x: x + 1) AS incremented \
             FROM written_lists \
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +----+---------+-------+-------------+
        | id | n_valid | total | incremented |
        +----+---------+-------+-------------+
        | 1  | 2.0     | 3.0   | [2, , 3]    |
        | 2  | 2.0     | 0.0   | [1, 1]      |
        +----+---------+-------+-------------+
        ");

    Ok(())
}

#[tokio::test]
async fn create_table_ordered_by() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    // Vortex
    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl_vx \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                WITH ORDER (c1 ASC)
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('air', 5), ('balloon', 42)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('zebra', 5)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('texas', 2000), ('alabama', 2000)")
        .await?
        .collect()
        .await?;

    let df = ctx
        .session
        .sql("SELECT * FROM my_tbl_vx ORDER BY c1 ASC limit 3")
        .await?;

    let physical_plan = ctx
        .session
        .state()
        .create_physical_plan(df.logical_plan())
        .await?;

    insta::assert_snapshot!(DisplayableExecutionPlan::new(physical_plan.as_ref())
                .tree_render().to_string(), @r"
        ┌───────────────────────────┐
        │  SortPreservingMergeExec  │
        │    --------------------   │
        │     c1 ASC NULLS LAST     │
        │                           │
        │          limit: 3         │
        └─────────────┬─────────────┘
        ┌─────────────┴─────────────┐
        │       DataSourceExec      │
        │    --------------------   │
        │          files: 3         │
        │       format: vortex      │
        └───────────────────────────┘
        ");

    let r = df.collect().await?;

    insta::assert_snapshot!(pretty_format_batches(&r)?.to_string(), @r"
        +---------+------+
        | c1      | c2   |
        +---------+------+
        | air     | 5    |
        | alabama | 2000 |
        | balloon | 42   |
        +---------+------+
        ");

    Ok(())
}

/// Doc example: demonstrates creating, writing, reading, and filtering a Vortex table.
#[tokio::test]
async fn doc_example() -> anyhow::Result<()> {
    // [setup]
    use std::sync::Arc;

    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_common::GetExt;
    use object_store::memory::InMemory;

    use crate::VortexFormatFactory;

    let factory = Arc::new(VortexFormatFactory::new());
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_file_formats(vec![factory])
        .build();
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    // [setup]

    // Register an in-memory object store for the test.
    let store = Arc::new(InMemory::new());
    ctx.register_object_store(&url::Url::try_from("file://").unwrap(), store);

    // [create]
    ctx.sql(
        "CREATE EXTERNAL TABLE my_table \
                (name VARCHAR NOT NULL, age INT NOT NULL) \
            STORED AS vortex \
            LOCATION '/demo/'",
    )
    .await?;
    // [create]

    // [write]
    ctx.sql(
        "INSERT INTO my_table VALUES \
                ('Alice', 30), ('Bob', 25), ('Charlie', 35), ('Diana', 28)",
    )
    .await?
    .collect()
    .await?;
    // [write]

    // [query]
    let result = ctx
        .sql("SELECT name, age FROM my_table WHERE age > 28 ORDER BY age")
        .await?
        .collect()
        .await?;
    // [query]

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---------+-----+
        | name    | age |
        +---------+-----+
        | Alice   | 30  |
        | Charlie | 35  |
        +---------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_repartitioned_scan_matches_non_repartitioned_for_uneven_splits() -> anyhow::Result<()>
{
    let store = Arc::new(InMemory::new()) as _;
    let session = VortexSession::default();
    let path = object_store::path::Path::parse("/split-aligned-repartition.vortex")?;

    let chunk_1_len = 2_000;
    let chunk_2_len = 5_000;
    let chunk_3_len = 6_000;
    let row_count = chunk_1_len + chunk_2_len + chunk_3_len;

    let chunk_1 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(0_i32..chunk_1_len).into_array()],
        usize::try_from(chunk_1_len)?,
        Validity::NonNullable,
    )?;
    let chunk_2 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(chunk_1_len..(chunk_1_len + chunk_2_len)).into_array()],
        usize::try_from(chunk_2_len)?,
        Validity::NonNullable,
    )?;
    let chunk_3 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter((chunk_1_len + chunk_2_len)..row_count).into_array()],
        usize::try_from(chunk_3_len)?,
        Validity::NonNullable,
    )?;
    let table = ChunkedArray::from_iter([
        chunk_1.into_array(),
        chunk_2.into_array(),
        chunk_3.into_array(),
    ])
    .into_array();
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&flat),
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
    ));

    let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(&mut writer, table.into_array().to_array_stream())
        .await?;
    writer.shutdown().await?;

    let reader = Arc::new(ObjectStoreReadAt::new(
        Arc::clone(&store),
        path.clone(),
        Handle::find().expect("tokio runtime should be available in tests"),
    ));
    let vxf = session
        .open_options()
        .with_file_size(summary.size())
        .open_read(reader)
        .await?;
    let split_ranges = vxf.splits()?;
    let split_lengths = split_ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect::<Vec<_>>();

    assert!(split_ranges.len() > 1);
    assert!(
        split_lengths
            .windows(2)
            .any(|window| window[0] != window[1])
    );

    let serial_ctx = make_session(Arc::clone(&store), false);
    let repartitioned_ctx = make_session(Arc::clone(&store), true);
    let repartitioned_partitions = count_query_partitions(
        &repartitioned_ctx,
        "SELECT value FROM '/split-aligned-repartition.vortex'",
    )
    .await?;

    assert!(repartitioned_partitions > 1);

    let serial = serial_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let repartitioned = repartitioned_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let serial_values = batch_values(&serial);
    let repartitioned_values = batch_values(&repartitioned);
    let expected = (0_i32..row_count).collect::<Vec<_>>();

    assert_eq!(serial_values, expected);
    assert_eq!(repartitioned_values, serial_values);

    Ok(())
}

/// Roundtrip an `arrow.uuid` extension column through a Vortex file: write the column directly
/// via the session-aware Arrow→Vortex conversion, then `SELECT *` and assert both the field
/// metadata and the underlying values survive the trip.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    let session = VortexSession::default();
    session.enable_edition(CORE_2026_08_3)?;

    let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    uuid_field.try_with_extension_type(Uuid)?;
    let schema = Arc::new(Schema::new(vec![uuid_field]));

    let uuids = FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(uuids)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT * FROM '/uuid.vortex'")
        .await?
        .collect()
        .await?;

    assert!(
        result[0]
            .schema_ref()
            .field(0)
            .has_valid_extension_type::<Uuid>()
    );

    assert_batches_sorted_eq!(
        [
            "+----------------------------------+",
            "| id                               |",
            "+----------------------------------+",
            "| 30313233343536373839616263646566 |",
            "| 66656463626139383736353433323130 |",
            "+----------------------------------+",
        ],
        &result
    );

    Ok(())
}

/// Same as [`arrow_uuid_extension_roundtrip`] but with the `arrow.uuid` field nested inside a
/// top-level `Struct`, exercising recursive session-aware Field/Schema inference: if any layer
/// falls back to the non-plugin canonical path, the inner field loses its extension metadata.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip_nested_struct() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::Array;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::array::StructArray as ArrowStructArray;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    let session = VortexSession::default();
    session.enable_edition(CORE_2026_08_3)?;

    let mut inner_uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    inner_uuid_field.try_with_extension_type(Uuid)?;
    let payload_fields = Fields::from(vec![inner_uuid_field]);
    let payload_field = Field::new("payload", DataType::Struct(payload_fields.clone()), false);
    let schema = Arc::new(Schema::new(vec![payload_field]));

    let uuids: Arc<dyn Array> = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);
    let payload_array = ArrowStructArray::new(payload_fields, vec![uuids], None);
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(payload_array)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer =
        ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid_struct.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT payload FROM '/uuid_struct.vortex'")
        .await?
        .collect()
        .await?;

    let read_payload = result[0].schema_ref().field(0);
    let DataType::Struct(read_inner) = read_payload.data_type() else {
        panic!(
            "expected Struct payload, got {:?}",
            read_payload.data_type()
        );
    };
    assert!(read_inner[0].has_valid_extension_type::<Uuid>());

    assert_batches_sorted_eq!(
        [
            "+----------------------------------------+",
            "| payload                                |",
            "+----------------------------------------+",
            "| {id: 30313233343536373839616263646566} |",
            "| {id: 66656463626139383736353433323130} |",
            "+----------------------------------------+",
        ],
        &result
    );

    Ok(())
}

/// Writes `files` Vortex files of a nullable `a` column under `/dyn/` and registers them as `t`.
///
/// File `f` holds `f * 100 + i` for `i` in `0..100`, with nulls wherever `i % 17 == 0`.
async fn register_dynamic_filter_table(ctx: &TestSessionContext, files: i32) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    for file in 0..files {
        let values = (0..100).map(|i| (i % 17 != 0).then_some(file * 100 + i));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter(values))],
        )?;
        ctx.write_arrow_batch(format!("dyn/{file}.vortex"), &batch)
            .await?;
    }
    ctx.session
        .sql("CREATE EXTERNAL TABLE t (a INT) STORED AS vortex LOCATION '/dyn/'")
        .await?;
    Ok(())
}

/// Runs `sql` and renders its single output column as comma-separated values.
async fn query_values(ctx: &TestSessionContext, sql: &str) -> anyhow::Result<String> {
    let batches = ctx.session.sql(sql).await?.collect().await?;
    let mut values = vec![];
    for batch in &batches {
        let column = batch.column(0);
        for row in 0..column.len() {
            values.push(if column.is_null(row) {
                "NULL".to_owned()
            } else {
                array_value_to_string(column, row)?
            });
        }
    }
    Ok(values.join(", "))
}

/// The scan accepts TopK dynamic filters and still returns the right rows, including nulls.
#[rstest]
#[case("ORDER BY a DESC NULLS LAST LIMIT 3", "499, 498, 497")]
#[case("ORDER BY a ASC NULLS LAST LIMIT 3", "1, 2, 3")]
#[case("ORDER BY a DESC NULLS FIRST LIMIT 3", "NULL, NULL, NULL")]
#[case("ORDER BY a ASC NULLS FIRST OFFSET 29 LIMIT 3", "NULL, 1, 2")]
#[tokio::test]
async fn topk_dynamic_filter_pushdown(
    #[case] order: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    register_dynamic_filter_table(&ctx, 5).await?;

    let query = format!("SELECT a FROM t {order}");
    let plan = ctx
        .session
        .sql(&query)
        .await?
        .create_physical_plan()
        .await?;
    let plan_str = DisplayableExecutionPlan::new(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        plan_str.contains("predicate: DynamicFilter"),
        "expected the scan to accept the dynamic filter:\n{plan_str}"
    );

    assert_eq!(query_values(&ctx, &query).await?, expected);
    Ok(())
}

/// Hash join dynamic filters on the probe side keep the join result correct, whether or not
/// their bounds are selective enough to be applied.
#[rstest]
// 151 and 17 are null in the table, and the bounds [17, 352] span most files.
#[case("(150), (151), (352), (17)", "150, 352")]
// Bounds [150, 152] skip all but one file's worth of values, so they are applied.
#[case("(150), (152)", "150, 152")]
#[tokio::test]
async fn hash_join_dynamic_filter_pushdown(
    #[case] build_values: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    register_dynamic_filter_table(&ctx, 5).await?;

    let query =
        format!("SELECT t.a FROM t JOIN (VALUES {build_values}) AS b(x) ON t.a = b.x ORDER BY t.a");
    assert_eq!(query_values(&ctx, &query).await?, expected);
    Ok(())
}

/// When the file's type differs from the table's, the expression adapter wraps the dynamic
/// filter's columns in casts. The filter must still be handled rather than rejected by the scan.
#[rstest]
#[case("ORDER BY a DESC LIMIT 3", "499, 498, 497")]
#[case("ORDER BY a ASC LIMIT 3", "1, 2, 3")]
#[tokio::test]
async fn topk_dynamic_filter_with_cast_column(
    #[case] order: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    // Files store `a` as INT while this table declares BIGINT.
    register_dynamic_filter_table(&ctx, 5).await?;
    ctx.session
        .sql("CREATE EXTERNAL TABLE t_wide (a BIGINT) STORED AS vortex LOCATION '/dyn/'")
        .await?;

    let query = format!("SELECT a FROM t_wide WHERE a IS NOT NULL {order}");
    assert_eq!(query_values(&ctx, &query).await?, expected);
    Ok(())
}

/// A TopK over a column that isn't declared sorted is pushed down inexactly: the scan reads the
/// most promising files first, and the TopK still produces exactly the right rows.
#[rstest]
#[case("ORDER BY a DESC LIMIT 3", "499, 498, 497", true)]
#[case("ORDER BY a ASC LIMIT 3", "1, 2, 3", false)]
#[case("ORDER BY a DESC, b ASC LIMIT 2", "499, 498", true)]
#[tokio::test]
async fn topk_inexact_sort_pushdown(
    #[case] order: &str,
    #[case] expected: &str,
    #[case] reverse_splits: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));
    // Disjoint value ranges, written so that file names don't follow value order. Values are
    // shuffled within each file so that the files have no known ordering.
    for file in [3, 0, 4, 1, 2] {
        let a = Int32Array::from_iter_values((0..100).map(|i| file * 100 + i * 37 % 100));
        let b = Int32Array::from_iter_values((0..100).map(|i| i % 7));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(a), Arc::new(b)])?;
        ctx.write_arrow_batch(format!("sorted/{file}.vortex"), &batch)
            .await?;
    }
    ctx.session
        .sql("CREATE EXTERNAL TABLE s (a INT NOT NULL, b INT NOT NULL) STORED AS vortex LOCATION '/sorted/'")
        .await?;

    let query = format!("SELECT a FROM s WHERE a > 0 {order}");
    let plan = ctx
        .session
        .sql(&query)
        .await?
        .create_physical_plan()
        .await?;
    let plan_str = DisplayableExecutionPlan::new(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        plan_str.contains("read_order: ["),
        "expected an inexact sort pushdown:\n{plan_str}"
    );
    assert_eq!(
        plan_str.contains("reverse_splits"),
        reverse_splits,
        "{plan_str}"
    );

    assert_eq!(query_values(&ctx, &query).await?, expected);
    Ok(())
}

/// Writes five files of a sorted `a` column and an unsorted `b` column under `/ordered/` and
/// registers them as `o`. Each file in `null_files` starts with five nulls in `a`. With
/// `overlapping`, the files' value ranges overlap.
async fn register_ordered_table(
    ctx: &TestSessionContext,
    nullable: bool,
    null_files: &[i32],
    overlapping: bool,
) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, nullable),
        Field::new("b", DataType::Int32, false),
    ]));
    let stride = if overlapping { 10 } else { 100 };
    for file in [3, 0, 4, 1, 2] {
        let has_nulls = null_files.contains(&file);
        let a = (0..100).map(|i| (!has_nulls || i >= 5).then_some(file * stride + i));
        let b = Int32Array::from_iter_values((0..100).map(|i| i % 7));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter(a)), Arc::new(b)],
        )?;
        ctx.write_arrow_batch(format!("ordered/{file}.vortex"), &batch)
            .await?;
    }
    let a_type = if nullable { "INT" } else { "INT NOT NULL" };
    ctx.session
        .sql(&format!(
            "CREATE EXTERNAL TABLE o (a {a_type}, b INT NOT NULL) STORED AS vortex LOCATION '/ordered/'"
        ))
        .await?;
    Ok(())
}

async fn physical_plan_string(ctx: &TestSessionContext, sql: &str) -> anyhow::Result<String> {
    let plan = ctx.session.sql(sql).await?.create_physical_plan().await?;
    Ok(DisplayableExecutionPlan::new(plan.as_ref())
        .indent(true)
        .to_string())
}

/// Files whose columns the writer proved sorted declare that ordering, so a query sorting by it
/// over files with disjoint ranges needs no sort at all.
#[rstest]
#[case::not_null(false, &[], "ORDER BY a LIMIT 3", "output_ordering=[a@0 ASC NULLS LAST]", "0, 1, 2")]
#[case::nullable_without_nulls(true, &[], "ORDER BY a LIMIT 3", "output_ordering=[a@0 ASC NULLS LAST]", "0, 1, 2")]
#[tokio::test]
async fn inferred_file_ordering_removes_sort(
    #[case] nullable: bool,
    #[case] null_files: &[i32],
    #[case] order: &str,
    #[case] ordering: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    register_ordered_table(&ctx, nullable, null_files, false).await?;

    let query = format!("SELECT a, b FROM o {order}");
    let plan_str = physical_plan_string(&ctx, &query).await?;
    assert!(plan_str.contains(ordering), "{plan_str}");
    assert!(!plan_str.contains("SortExec"), "{plan_str}");

    assert_eq!(query_values(&ctx, &query).await?, expected);
    Ok(())
}

/// Queries the inferred file orderings cannot answer without sorting still sort correctly:
/// columns with nulls, a descending sort, and files whose ranges overlap.
#[rstest]
// Each file's nulls come first, but files read one after another would interleave them with
// values, so files with nulls must not declare an ordering.
#[case::nulls_first(true, &[0, 1, 2, 3, 4], false, "ORDER BY a NULLS FIRST OFFSET 25 LIMIT 3", "5, 6, 7")]
#[case::nulls_last(true, &[0, 1, 2, 3, 4], false, "ORDER BY a NULLS LAST LIMIT 3", "5, 6, 7")]
#[case::descending(false, &[], false, "ORDER BY a DESC LIMIT 3", "499, 498, 497")]
#[case::mixed_null_files(true, &[0], false, "ORDER BY a NULLS FIRST LIMIT 7", "NULL, NULL, NULL, NULL, NULL, 5, 6")]
#[case::overlapping(false, &[], true, "ORDER BY a LIMIT 5", "0, 1, 2, 3, 4")]
#[case::overlapping_desc(false, &[], true, "ORDER BY a DESC LIMIT 4", "139, 138, 137, 136")]
#[tokio::test]
async fn inferred_file_ordering_keeps_results_correct(
    #[case] nullable: bool,
    #[case] null_files: &[i32],
    #[case] overlapping: bool,
    #[case] order: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    register_ordered_table(&ctx, nullable, null_files, overlapping).await?;

    assert_eq!(
        query_values(&ctx, &format!("SELECT a FROM o {order}")).await?,
        expected
    );
    Ok(())
}
