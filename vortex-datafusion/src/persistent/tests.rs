// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
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
    dbg!(&ctx.store);

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
