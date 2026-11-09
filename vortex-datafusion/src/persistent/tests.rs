// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::RunArray;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::array::UInt32Array;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Field;
use datafusion::arrow::datatypes::Int32Type;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::datatypes::UInt32Type;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_common::Statistics;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::stats::Precision;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::StatisticsArgs;
use datafusion_physical_plan::StatisticsContext;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;
use insta::assert_snapshot;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::editions::CORE_2026_08_3;
use vortex::editions::EditionSessionExt;
use vortex::editions::UNSTABLE;
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

use crate::VortexFormat;
use crate::VortexFormatFactory;
use crate::VortexScanConfig;
use crate::VortexSource;
use crate::common_tests::TestSessionContext;
use crate::persistent::segment_cache::DEFAULT_SEGMENT_CACHE_SIZE_BYTES;
use crate::persistent::segment_cache::SEGMENT_CACHE_HITS;
use crate::persistent::segment_cache::SEGMENT_CACHE_MISSES;
use crate::persistent::segment_cache::SEGMENT_CACHE_STORES;

fn make_session(
    object_store: Arc<dyn ObjectStore>,
    repartition_file_scans: bool,
    scan_v2: bool,
) -> SessionContext {
    let factory = Arc::new(
        VortexFormatFactory::new().with_options(crate::VortexTableOptions {
            scan_v2,
            ..Default::default()
        }),
    );

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

    let batches = ctx
        .session
        .sql("SELECT * from '/test.vortex'")
        .await?
        .limit(0, limit)?
        .collect()
        .await?;

    let mut seen = std::collections::BTreeSet::new();
    for batch in &batches {
        assert_eq!(batch.num_columns(), 2);
        for row in 0..batch.num_rows() {
            let number = datafusion_common::ScalarValue::try_from_array(batch.column(1), row)?;
            let datafusion_common::ScalarValue::UInt32(Some(number)) = number else {
                panic!("expected non-null UInt32");
            };
            assert!((1..=8).contains(&number));
            assert!(seen.insert(number), "duplicate row {number}");
            let expected = ["ab", "foo", "bar", "baz"][(number as usize - 1) % 4];
            assert_eq!(
                datafusion_common::ScalarValue::try_from_array(batch.column(0), row)?,
                datafusion_common::ScalarValue::Utf8View(Some(expected.to_owned()))
            );
        }
    }
    assert_eq!(seen.len(), limit.unwrap_or(8));

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

#[rstest]
#[tokio::test]
async fn test_repartitioned_scan_matches_non_repartitioned_for_uneven_splits(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as _;
    let session = VortexSession::default();
    let path = Path::parse("/split-aligned-repartition.vortex")?;

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

    let serial_ctx = make_session(Arc::clone(&store), false, scan_v2);
    let repartitioned_ctx = make_session(Arc::clone(&store), true, scan_v2);
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

/// A V2 scan of a local store reads through the native file reader, resolving paths under the
/// store's prefix, and returns every row once whether or not the file is split into byte ranges.
#[rstest]
#[tokio::test]
async fn test_local_store_scan_reads_byte_range_partitions(
    #[values(false, true)] scan_v2: bool,
    #[values(false, true)] repartition: bool,
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let store: Arc<dyn ObjectStore> = Arc::new(
        object_store::local::LocalFileSystem::new_with_prefix(dir.path())?,
    );
    let session = VortexSession::default();
    let row_count = 50_000_i32;
    let table = ChunkedArray::from_iter((0..5).map(|chunk| {
        let start = chunk * (row_count / 5);
        StructArray::try_new(
            ["value"].into(),
            vec![Buffer::from_iter(start..start + row_count / 5).into_array()],
            usize::try_from(row_count / 5).expect("chunk length fits usize"),
            Validity::NonNullable,
        )
        .expect("valid struct")
        .into_array()
    }))
    .into_array();
    let path = object_store::path::Path::parse("local-prefix.vortex")?;
    let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
    session
        .write_options()
        .write(&mut writer, table.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let ctx = make_session(store, repartition, scan_v2);
    let sql = "SELECT value FROM '/local-prefix.vortex' ORDER BY value";
    if repartition {
        assert!(count_query_partitions(&ctx, sql).await? > 1);
    }
    let batches = ctx.sql(sql).await?.collect().await?;

    assert_eq!(batch_values(&batches), (0..row_count).collect::<Vec<_>>());
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

#[rstest]
#[case::modulo("id, a", "CAST(a % 2 AS BIGINT) = 0")]
#[case::in_list("id, a IN (1, 4, 5, 6, 7, NULL) AS member", "TRUE")]
#[case::not_in("id, a", "a NOT IN (1, 4, 5, 6, 7, 8)")]
#[case::null_list("id, a", "a NOT IN (1, 4, 5, 6, 7, NULL)")]
#[case::column_list("id, a", "a IN (b, 4, 5, 6, 7, 8)")]
// Overflow equivalence is deferred until Vortex matches DataFusion's arithmetic semantics.
#[case::add("id, a + CAST(1 AS INT) AS n", "id > 1")]
#[case::subtract("id, a - CAST(1 AS INT) AS n", "id > 1")]
#[case::multiply("id, a * CAST(2 AS INT) AS n", "id > 1")]
#[case::arithmetic_filter("id", "b * CAST(2 AS INT) = 4")]
#[case::narrow_cast("id, CAST(a AS TINYINT) AS n", "TRUE")]
#[case::case_branch("id, CASE WHEN b <> 0 THEN 12 / b ELSE 0 END AS n", "TRUE")]
#[case::case_filter("id", "CASE WHEN b <> 0 THEN 12 / b ELSE 0 END = 6")]
#[case::case_else("id, CASE WHEN b = 0 THEN 0 ELSE 12 / b END AS n", "TRUE")]
#[case::case_later_condition(
    "id, CASE WHEN b = 0 THEN 0 WHEN 12 / b > 1 THEN 12 / b ELSE 1 END AS n",
    "TRUE"
)]
#[case::case_no_else("id, CASE WHEN b <> 0 THEN 12 / b END AS n", "TRUE")]
#[case::case_selected_error("id, CASE WHEN b = 0 THEN 12 / b ELSE 0 END AS n", "TRUE")]
#[case::not("id, NOT (a IN (1, 2, NULL)) AS n", "NOT (a = b)")]
#[case::projection_alias("id, a + CAST(1 AS INT) AS a, abs(a) AS b", "id > 1")]
#[case::unprojected("id", "CAST(a AS VARCHAR) = '1'")]
#[case::decimal(
    "id, CAST(a AS DECIMAL(12, 2)) + CAST(b AS DECIMAL(12, 2)) AS n",
    "TRUE"
)]
#[tokio::test]
async fn test_predicate_memtable_oracle(
    #[case] projection: &str,
    #[case] predicate: &str,
    #[values(false, true)] pushdown: bool,
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let options = crate::VortexTableOptions {
        scan_v2,
        projection_pushdown: pushdown,
        predicate_pushdown: pushdown,
        ..Default::default()
    };
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(options),
    ));
    let batch = arrow_array::record_batch!(
        ("id", Int32, vec![0, 1, 2, 3, 4, 5]),
        (
            "a",
            Int32,
            vec![
                Some(i32::MAX),
                Some(i32::MIN),
                Some(1),
                Some(2),
                None,
                Some(4)
            ]
        ),
        ("b", Int32, vec![0, 0, 1, 2, 0, 2])
    )?;
    ctx.write_arrow_batch("oracle.vortex", &batch).await?;
    let actual_table = ctx
        .table_provider("actual", "/oracle.vortex", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("actual", actual_table)?;
    ctx.session.register_table(
        "oracle",
        Arc::new(datafusion::datasource::MemTable::try_new(
            batch.schema(),
            vec![vec![batch]],
        )?),
    )?;
    let actual = ctx
        .session
        .sql(&format!(
            "SELECT {projection} FROM actual WHERE {predicate} ORDER BY id"
        ))
        .await?;
    let expected = ctx
        .session
        .sql(&format!(
            "SELECT {projection} FROM oracle WHERE {predicate} ORDER BY id"
        ))
        .await?;
    assert_eq!(actual.schema().as_arrow(), expected.schema().as_arrow());
    let schema = Arc::new(expected.schema().as_arrow().clone());
    match (actual.collect().await, expected.collect().await) {
        (Ok(actual), Ok(expected)) => {
            let actual = concat_batches(&schema, &actual)?;
            let expected = concat_batches(&schema, &expected)?;
            assert_eq!(actual, expected);
        }
        (Err(_), Err(_)) => {}
        (actual, expected) => anyhow::bail!(
            "Vortex/MemTable mismatch for {projection} WHERE {predicate}: {actual:?} / {expected:?}"
        ),
    }
    Ok(())
}

/// The [`VortexSource`] of `plan` when it is a Vortex scan.
fn vortex_source(plan: &dyn ExecutionPlan) -> Option<&VortexSource> {
    plan.downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<VortexScanConfig>())
        .and_then(|config| {
            config
                .file_scan_config()
                .file_source
                .downcast_ref::<VortexSource>()
        })
}

/// Returns whether `plan` filters the rows of a Vortex scan in DataFusion and whether a Vortex
/// scan accepted a predicate.
fn filter_placement(plan: &dyn ExecutionPlan) -> (bool, bool) {
    let (in_datafusion, in_scan, _) = vortex_filter_placement(plan);
    (in_datafusion, in_scan)
}

/// Like [`filter_placement`], and also whether `plan` contains a Vortex scan.
fn vortex_filter_placement(plan: &dyn ExecutionPlan) -> (bool, bool, bool) {
    let source = vortex_source(plan);
    let mut has_scan = source.is_some();
    let mut in_scan = source.is_some_and(|source| source.predicate().is_some());
    let mut in_datafusion = false;
    for child in plan.children() {
        let (child_in_datafusion, child_in_scan, child_has_scan) =
            vortex_filter_placement(child.as_ref());
        in_datafusion |= child_in_datafusion;
        in_scan |= child_in_scan;
        has_scan |= child_has_scan;
    }
    in_datafusion |= has_scan && plan.downcast_ref::<FilterExec>().is_some();
    (in_datafusion, in_scan, has_scan)
}

/// Regression test for #10634: string comparisons on `REE<Dict<Utf8>>` columns are pushed into
/// the scan and return the same rows as DataFusion.
#[rstest]
#[case::eq("ree_dict = 'x'")]
#[case::not_eq("ree_dict <> 'x'")]
#[case::or("ree_dict = 'x' OR ree_dict = 'y'")]
#[case::in_list("ree_dict IN ('y', 'unused')")]
#[case::and("ree_dict = 'x' AND id > 1")]
#[tokio::test]
async fn test_ree_dict_predicate_memtable_oracle(
    #[case] predicate: &str,
    #[values(false, true)] pushdown: bool,
) -> anyhow::Result<()> {
    let options = crate::VortexTableOptions {
        projection_pushdown: pushdown,
        predicate_pushdown: pushdown,
        ..Default::default()
    };
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(options),
    ));
    // Rows: x, x, NULL, y, y, x. The dictionary also holds a value no key references.
    let values = DictionaryArray::<UInt32Type>::try_new(
        UInt32Array::from(vec![Some(0), None, Some(1), Some(0)]),
        Arc::new(StringArray::from(vec!["x", "y", "unused"])),
    )?;
    let ree = RunArray::<Int32Type>::try_new(&Int32Array::from(vec![2, 3, 5, 6]), &values)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ree_dict", ree.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from_iter_values(0..6)), Arc::new(ree)],
    )?;
    ctx.write_arrow_batch("ree_dict.vortex", &batch).await?;
    let actual_table = ctx
        .table_provider("actual", "/ree_dict.vortex", schema.as_ref().clone())
        .await?;
    ctx.session.register_table("actual", actual_table)?;
    ctx.session.register_table(
        "oracle",
        Arc::new(datafusion::datasource::MemTable::try_new(
            schema,
            vec![vec![batch]],
        )?),
    )?;

    let query = |table: &str| format!("SELECT id FROM {table} WHERE {predicate} ORDER BY id");
    let actual = ctx.session.sql(&query("actual")).await?;
    let plan = actual.clone().create_physical_plan().await?;
    assert_eq!(
        filter_placement(plan.as_ref()),
        (!pushdown, pushdown),
        "{}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );
    let actual = actual.collect().await?;
    let expected = ctx.session.sql(&query("oracle")).await?.collect().await?;
    assert_eq!(batch_values(&actual), batch_values(&expected));
    Ok(())
}

/// The rows each Vortex scan in `plan` returned, after its predicate.
fn vortex_scan_rows(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
    let mut rows = Vec::new();
    if vortex_source(plan.as_ref()).is_some() {
        rows.extend(plan.metrics().and_then(|metrics| metrics.output_rows()));
    }
    for child in plan.children() {
        rows.extend(vortex_scan_rows(child));
    }
    rows
}

/// A hash join's dynamic filter reaches a V2 scan as bounds and a hash table lookup, routed by
/// hash to each build partition for a partitioned join, and widened by `key IS NULL` for a
/// null-equal join. The scan returns exactly the probe rows that the join can match.
#[rstest]
#[tokio::test]
async fn test_hash_join_dynamic_filter_filters_scan_rows_exactly(
    #[values(false, true)] partitioned: bool,
    #[values(false, true)] null_equal: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(crate::VortexTableOptions {
            scan_v2: true,
            ..Default::default()
        }),
    ));
    let threshold = if partitioned { 0 } else { 1 << 30 };
    for setting in [
        "datafusion.execution.target_partitions = 4".to_string(),
        // Build sides above this many distinct keys test membership with a hash table lookup.
        "datafusion.optimizer.hash_join_inlist_pushdown_max_distinct_values = 1".to_string(),
        format!("datafusion.optimizer.hash_join_single_partition_threshold = {threshold}"),
        format!("datafusion.optimizer.hash_join_single_partition_threshold_rows = {threshold}"),
    ] {
        ctx.session
            .sql(&format!("SET {setting}"))
            .await?
            .collect()
            .await?;
    }

    // Probe keys span the build keys' bounds, so only the membership test rejects rows.
    let probe_keys = (0..4096)
        .map(|key| (key % 11 != 5).then_some(key))
        .collect::<Vec<_>>();
    let build_keys = (0..4096)
        .step_by(97)
        .map(Some)
        .chain(null_equal.then_some(None))
        .collect::<Vec<_>>();
    let probe = arrow_array::record_batch!(
        ("k", Int32, probe_keys.clone()),
        ("v", Int32, (0..4096).collect::<Vec<i32>>())
    )?;
    ctx.write_arrow_batch("probe.vortex", &probe).await?;
    let table = ctx
        .table_provider("probe", "/probe.vortex", probe.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("probe", table)?;
    let build = arrow_array::record_batch!(("k", Int32, build_keys.clone()))?;
    ctx.session.register_table(
        "build",
        Arc::new(datafusion::datasource::MemTable::try_new(
            build.schema(),
            vec![vec![build]],
        )?),
    )?;

    let condition = if null_equal {
        "probe.k IS NOT DISTINCT FROM build.k"
    } else {
        "probe.k = build.k"
    };
    let plan = ctx
        .session
        .sql(&format!(
            "SELECT probe.v FROM build JOIN probe ON {condition}"
        ))
        .await?
        .create_physical_plan()
        .await?;
    let batches =
        datafusion_physical_plan::collect(Arc::clone(&plan), ctx.session.task_ctx()).await?;

    let expected = probe_keys
        .iter()
        .filter(|key| match key {
            Some(key) => build_keys.contains(&Some(*key)),
            None => null_equal,
        })
        .count();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        expected
    );
    assert_eq!(
        vortex_scan_rows(&plan),
        [expected],
        "{}",
        DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
    );
    Ok(())
}

/// Predicates whose conversion relies on a rewrite return the same rows as DataFusion, and are
/// removed from DataFusion's `FilterExec` when pushed down.
#[rstest]
#[case::subquery_eq("a = (SELECT max(a) - 2 FROM oracle)")]
#[case::subquery_cast_range(
    "CAST(a AS BIGINT) >= (SELECT min(a) FROM oracle) AND CAST(a AS BIGINT) <= (SELECT max(a) - 1 FROM oracle)"
)]
#[case::subquery_null("a = (SELECT max(a) FROM oracle WHERE a > 100)")]
#[case::int_to_decimal("CAST(a AS DECIMAL(12, 2)) > CAST(2.5 AS DECIMAL(12, 2))")]
#[case::utf8_to_view("arrow_cast(s, 'Utf8View') = '2412'")]
#[case::coalesce_int("coalesce(a, 0) = 0")]
#[case::coalesce_string("coalesce(s, 'none') IN ('none', '2412')")]
#[case::or_equalities("a = 1 OR a = 4 OR a = 4")]
#[case::repeated_disjunction("(a = 2 OR a = 4) AND (a = 2 OR a = 4 OR a = 4)")]
#[case::implied_disjunction("a IN (1, 2, 4) AND (a = 4 OR a = 1)")]
#[case::substr_in("substr(s, 1, 5) IN ('24128', '76232', 'a%c_e', 'äöü12')")]
#[case::substr_not_in("substr(s, 1, 5) NOT IN ('24128', 'äöü12')")]
#[case::substr_eq("substr(s, 1, 4) = '2412'")]
#[case::substr_or("substr(s, 1, 3) = 'äöü' OR substr(s, 1, 3) = 'a%c'")]
#[tokio::test]
async fn test_rewritten_predicate_memtable_oracle(
    #[case] predicate: &str,
    #[values(false, true)] pushdown: bool,
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let options = crate::VortexTableOptions {
        scan_v2,
        projection_pushdown: pushdown,
        predicate_pushdown: pushdown,
        ..Default::default()
    };
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(options),
    ));
    let batch = arrow_array::record_batch!(
        ("id", Int32, vec![0, 1, 2, 3, 4, 5]),
        (
            "a",
            Int32,
            vec![Some(1), Some(2), None, Some(4), Some(5), Some(2)]
        ),
        (
            "s",
            Utf8,
            vec![
                Some("24128-1"),
                Some("2412"),
                None,
                Some("a%c_ef"),
                Some("äöü12"),
                Some("76232")
            ]
        )
    )?;
    ctx.write_arrow_batch("rewritten.vortex", &batch).await?;
    let actual_table = ctx
        .table_provider(
            "actual",
            "/rewritten.vortex",
            batch.schema().as_ref().clone(),
        )
        .await?;
    ctx.session.register_table("actual", actual_table)?;
    ctx.session.register_table(
        "oracle",
        Arc::new(datafusion::datasource::MemTable::try_new(
            batch.schema(),
            vec![vec![batch]],
        )?),
    )?;

    let query = |table: &str| format!("SELECT id FROM {table} WHERE {predicate} ORDER BY id");
    let actual = ctx.session.sql(&query("actual")).await?;
    let plan = actual.clone().create_physical_plan().await?;
    let (in_datafusion, in_scan) = filter_placement(plan.as_ref());
    assert_eq!(
        (in_datafusion, in_scan),
        (!pushdown, pushdown),
        "{}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );
    let actual = actual.collect().await?;
    let expected = ctx.session.sql(&query("oracle")).await?.collect().await?;
    assert_eq!(batch_values(&actual), batch_values(&expected));
    Ok(())
}

async fn write_cache_test_file(
    store: &Arc<dyn ObjectStore>,
    path: &object_store::path::Path,
    offset: i32,
) -> anyhow::Result<()> {
    let row_count = 40_000_i32;
    let names = ["alpha", "beta", "gamma", "delta"];
    let table = StructArray::try_new(
        ["value", "name"].into(),
        vec![
            Buffer::from_iter(offset..offset + row_count).into_array(),
            VarBinArray::from(
                (0..row_count)
                    .map(|i| names[usize::try_from(i % 4).expect("index fits usize")])
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ],
        usize::try_from(row_count)?,
        Validity::NonNullable,
    )?;
    let session = VortexSession::default();
    let mut writer = ObjectStoreWrite::new(Arc::clone(store), path).await?;
    session
        .write_options()
        .write(&mut writer, table.into_array().to_array_stream())
        .await?;
    writer.shutdown().await?;
    Ok(())
}

/// Runs `sql`, returning the first column's values and the sum of each segment cache counter.
async fn run_with_cache_metrics(
    ctx: &SessionContext,
    sql: &str,
) -> anyhow::Result<(Vec<i32>, [usize; 3])> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let batches = datafusion_physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let metrics = crate::metrics::VortexMetricsFinder::find_all(plan.as_ref());
    let count = |name: &str| {
        metrics
            .iter()
            .filter_map(|set| set.sum_by_name(name))
            .map(|value| value.as_usize())
            .sum::<usize>()
    };
    Ok((
        batch_values(&batches),
        [
            count(SEGMENT_CACHE_HITS),
            count(SEGMENT_CACHE_MISSES),
            count(SEGMENT_CACHE_STORES),
        ],
    ))
}

/// V2 scans of a table keep its zone maps and dictionaries across queries, until a file changes.
#[rstest]
#[case::enabled(DEFAULT_SEGMENT_CACHE_SIZE_BYTES)]
#[case::disabled(0)]
#[tokio::test]
async fn test_segment_cache_reused_across_queries(#[case] size: usize) -> anyhow::Result<()> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = object_store::path::Path::parse("/cached/data.vortex")?;
    write_cache_test_file(&store, &path, 0).await?;

    let factory = Arc::new(
        VortexFormatFactory::new().with_options(crate::VortexTableOptions {
            scan_v2: true,
            segment_cache_size_bytes: size,
            ..Default::default()
        }),
    );
    let mut state = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(2))
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://")?, Arc::clone(&store));
    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }
    let ctx = SessionContext::new_with_state(state.build());
    ctx.sql("CREATE EXTERNAL TABLE cached STORED AS vortex LOCATION '/cached/'")
        .await?
        .collect()
        .await?;

    let sql = "SELECT value FROM cached WHERE value >= 35000 AND name <> 'beta' ORDER BY value";
    let expected = |offset: i32| {
        (35_000..40_000 + offset)
            .filter(|value| (value - offset) % 4 != 1)
            .collect::<Vec<_>>()
    };

    let (cold, [hits, misses, stores]) = run_with_cache_metrics(&ctx, sql).await?;
    assert_eq!(cold, expected(0));
    assert_eq!(hits, 0);
    if size == 0 {
        assert_eq!((misses, stores), (0, 0));
    } else {
        assert!(misses > 0);
        assert_eq!(stores, misses);
    }

    let (warm, [hits, misses, _]) = run_with_cache_metrics(&ctx, sql).await?;
    assert_eq!(warm, expected(0));
    if size == 0 {
        assert_eq!((hits, misses), (0, 0));
    } else {
        assert!(hits > 0);
        assert_eq!(misses, 0);
    }

    // A rewritten file is a new cache entry: its segments are read, not the old file's. The
    // listing cache would otherwise keep the old file's metadata.
    write_cache_test_file(&store, &path, 1).await?;
    if let Some(listing) = ctx.runtime_env().cache_manager.get_list_files_cache() {
        listing.clear();
    }
    let (rewritten, [hits, misses, _]) = run_with_cache_metrics(&ctx, sql).await?;
    assert_eq!(rewritten, expected(1));
    assert_eq!(hits, 0);
    assert_eq!(misses > 0, size > 0);
    Ok(())
}

/// Finds the statistics of the only `DataSourceExec` in `plan`.
fn scan_statistics(plan: &Arc<dyn ExecutionPlan>) -> anyhow::Result<Arc<Statistics>> {
    if plan.downcast_ref::<DataSourceExec>().is_some() {
        return Ok(StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())?);
    }
    match plan.children().as_slice() {
        [child] => scan_statistics(child),
        _ => anyhow::bail!("expected a single scan"),
    }
}

/// A pushed predicate removes the `FilterExec`, so the scan itself estimates the rows it keeps.
#[rstest]
#[case::no_filter("TRUE", Precision::Exact(1_000))]
// Interval analysis over the exact bounds [0, 999] keeps 100 of 1000 values.
#[case::range("k < 100", Precision::Inexact(100))]
// Interval analysis does not support `LIKE`, so the default 20% selectivity applies.
#[case::default_selectivity("s LIKE 'a%'", Precision::Inexact(200))]
#[case::contradiction("k < 10 AND k > 20", Precision::Inexact(0))]
#[tokio::test]
async fn pushed_filter_scales_scan_statistics(
    #[case] predicate: &str,
    #[case] expected_rows: Precision<usize>,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    let batch = arrow_array::record_batch!(
        ("id", Int64, (0..1_000).collect::<Vec<i64>>()),
        ("k", Int64, (0..1_000).collect::<Vec<i64>>()),
        (
            "s",
            Utf8,
            (0..1_000).map(|i| format!("{i}")).collect::<Vec<_>>()
        )
    )?;
    ctx.write_arrow_batch("filtered.vortex", &batch).await?;
    let table = ctx
        .table_provider("t", "/filtered.vortex", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("t", table)?;

    let plan = ctx
        .session
        .sql(&format!("SELECT id FROM t WHERE {predicate}"))
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        filter_placement(plan.as_ref()),
        (false, predicate != "TRUE")
    );
    let statistics = scan_statistics(&plan)?;
    assert_eq!(statistics.num_rows, expected_rows);
    assert_eq!(statistics.column_statistics.len(), 1);
    Ok(())
}

/// Scans report the in-memory size of the columns they read, including variable-width ones.
#[rstest]
#[case::fixed_width("id", Some(Precision::Exact(8_000)))]
#[case::variable_width("s", None)]
#[case::mixed("id, s", None)]
#[tokio::test]
async fn scan_reports_projected_byte_size(
    #[case] projection: &str,
    #[case] fixed_width_total: Option<Precision<usize>>,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    let batch = arrow_array::record_batch!(
        ("id", Int64, (0..1_000).collect::<Vec<i64>>()),
        (
            "s",
            Utf8,
            (0..1_000)
                .map(|i| format!("value-{i:>20}"))
                .collect::<Vec<_>>()
        ),
        (
            "unread",
            Utf8,
            (0..1_000).map(|i| format!("{i:>100}")).collect::<Vec<_>>()
        )
    )?;
    ctx.write_arrow_batch("sized.vortex", &batch).await?;
    let table = ctx
        .table_provider("t", "/sized.vortex", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("t", table)?;

    let plan = ctx
        .session
        .sql(&format!("SELECT {projection} FROM t"))
        .await?
        .create_physical_plan()
        .await?;
    let statistics = scan_statistics(&plan)?;
    let column_sizes = statistics
        .column_statistics
        .iter()
        .map(|column| match column.byte_size {
            Precision::Inexact(size) if size > 0 => Ok(size),
            other => Err(anyhow!("expected an inexact byte size, got {other:?}")),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let expected_total =
        fixed_width_total.unwrap_or_else(|| Precision::Inexact(column_sizes.iter().sum()));
    assert_eq!(statistics.total_byte_size, expected_total);
    Ok(())
}

/// Writes a file of a column with 5,000 distinct values among 20,000 rows, a nullable column with
/// one distinct value, and a constant column, and returns its inferred DataFusion statistics.
async fn distinct_count_statistics(
    session: VortexSession,
    nested: bool,
) -> anyhow::Result<Statistics> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let rows = 20_000;
    let keys = PrimitiveArray::from_iter((0..rows).map(|i| i % 5_000)).into_array();
    let sparse = PrimitiveArray::from_option_iter((0..rows).map(|i| (i % 2 == 0).then_some(1i32)))
        .into_array();
    let constant = ConstantArray::new(7i32, 20_000).into_array();
    let array = StructArray::try_new(
        ["keys", "sparse", "constant"].into(),
        vec![keys, sparse, constant],
        20_000,
        Validity::NonNullable,
    )?
    .into_array();

    let path = Path::from("ndv.vortex");
    let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
    let options = session.write_options();
    let options = if nested {
        options.include_nested_statistics()
    } else {
        options
    };
    options.write(&mut writer, array.to_array_stream()).await?;
    writer.shutdown().await?;

    let ctx = SessionContext::new();
    let state = ctx.state();
    let format = VortexFormat::new(session);
    let object = store.head(&path).await?;
    let schema = format
        .infer_schema(&state, &store, std::slice::from_ref(&object))
        .await?;
    Ok(format.infer_stats(&state, &store, schema, &object).await?)
}

/// A file with a distinct-count sketch reports an inexact distinct count, capped at the non-null
/// rows; a file without one reports none, beyond the exact count of a constant column.
#[rstest]
#[case::sketch(true, true)]
#[case::core_edition(false, true)]
#[case::legacy_statistics(true, false)]
#[tokio::test]
async fn infer_stats_reports_approx_distinct_count(
    #[case] unstable: bool,
    #[case] nested: bool,
) -> anyhow::Result<()> {
    let session = VortexSession::default();
    if unstable {
        session.enable_edition(UNSTABLE)?;
    }
    let stats = distinct_count_statistics(session, nested).await?;
    let [keys, sparse, constant] = stats.column_statistics.as_slice() else {
        panic!("expected three columns");
    };

    if unstable && nested {
        let Precision::Inexact(estimate) = keys.distinct_count else {
            panic!(
                "expected an inexact distinct count, got {:?}",
                keys.distinct_count
            );
        };
        assert!(estimate.abs_diff(5_000) <= 250, "estimate {estimate}");
        assert_eq!(sparse.distinct_count, Precision::Inexact(1));
    } else {
        assert_eq!(keys.distinct_count, Precision::Absent);
        assert_eq!(sparse.distinct_count, Precision::Absent);
    }
    assert_eq!(constant.distinct_count, Precision::Exact(1));
    Ok(())
}
