// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int16Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_common::ScalarValue;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::Literal;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::mask::Mask;
use vortex::session::VortexSession;

use super::CountBy;
use super::count_chunk;
use super::count_rows;
use crate::VortexCountByPushdown;
use crate::VortexFormatFactory;

fn session(store: &Arc<InMemory>, count_by: bool) -> SessionContext {
    session_with_config(
        store,
        count_by,
        SessionConfig::new().with_target_partitions(4),
    )
}

fn session_with_config(
    store: &Arc<InMemory>,
    count_by: bool,
    config: SessionConfig,
) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(
            &Url::try_from("file://").unwrap(),
            Arc::clone(store) as Arc<dyn ObjectStore>,
        );
    if count_by {
        builder = builder.with_physical_optimizer_rule(Arc::new(VortexCountByPushdown));
    }
    if let Some(file_formats) = builder.file_formats() {
        file_formats.push(factory as _);
    }
    SessionContext::new_with_state(builder.build())
}

const CREATE_HITS: &str = "CREATE EXTERNAL TABLE hits \
    (url VARCHAR NOT NULL, referer VARCHAR, source SMALLINT NOT NULL, width INT NOT NULL, \
    opt INT, score DOUBLE NOT NULL) \
    STORED AS vortex LOCATION '/hits/'";

/// Writes two files of rows with few distinct strings, so the writer dictionary-encodes them.
async fn write_hits(store: &Arc<InMemory>) -> anyhow::Result<()> {
    let writer = session(store, false);
    writer.sql(CREATE_HITS).await?;
    for offset in [0, 50_000] {
        writer
            .sql(&format!(
                "INSERT INTO hits SELECT \
                    'https://example.com/' || CAST((value * 7) % 300 AS VARCHAR), \
                    CASE WHEN value % 11 = 0 THEN NULL \
                        ELSE 'https://ref.example/' || CAST(value % 40 AS VARCHAR) END, \
                    CAST(value % 5 AS SMALLINT), \
                    CAST(value % 1000 AS INT), \
                    CASE WHEN value % 3 = 0 THEN NULL ELSE CAST(value % 7 AS INT) END, \
                    CAST(value % 10 AS DOUBLE) / 2 \
                FROM generate_series({offset}, {} ) AS t(value)",
                offset + 49_999
            ))
            .await?
            .collect()
            .await?;
    }
    Ok(())
}

async fn run(ctx: &SessionContext, sql: &str) -> anyhow::Result<(String, String)> {
    let plan = pretty_format_batches(&ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?)?
        .to_string();
    let result = pretty_format_batches(&ctx.sql(sql).await?.collect().await?)?.to_string();
    Ok((plan, result))
}

/// The pushed-down partial count gives exactly DataFusion's result, and is used where expected.
#[rstest]
#[case::dictionary_key(
    "SELECT url, COUNT(*) AS c FROM hits WHERE width > 100 GROUP BY url ORDER BY url",
    true
)]
#[case::dictionary_and_integer_keys(
    "SELECT source, url, COUNT(*) AS c FROM hits GROUP BY source, url ORDER BY source, url",
    true
)]
#[case::computed_key(
    "SELECT source, CASE WHEN source = 0 THEN url ELSE '' END AS src, COUNT(*) AS c FROM hits \
     GROUP BY source, src ORDER BY source, src",
    true
)]
#[case::nullable_key(
    "SELECT referer, COUNT(*) AS c FROM hits GROUP BY referer ORDER BY referer NULLS FIRST",
    true
)]
#[case::integer_keys(
    "SELECT source, width, COUNT(*) AS c FROM hits WHERE width < 50 GROUP BY source, width \
     ORDER BY source, width",
    true
)]
#[case::constant_key(
    "SELECT 'all' AS k, COUNT(*) AS c FROM hits WHERE width > 100 GROUP BY 1",
    false
)]
#[case::other_aggregate(
    "SELECT url, SUM(width) AS s FROM hits GROUP BY url ORDER BY url",
    false
)]
#[case::filter_above_scan(
    "SELECT source, COUNT(*) AS c FROM hits WHERE width % 3 = 0 GROUP BY source ORDER BY source",
    false
)]
#[case::nullable_integer_key(
    "SELECT opt, COUNT(*) AS c FROM hits GROUP BY opt ORDER BY opt NULLS FIRST",
    true
)]
#[case::float_key(
    "SELECT score, COUNT(*) AS c FROM hits GROUP BY score ORDER BY score",
    false
)]
#[case::count_of_column(
    "SELECT source, COUNT(referer) AS c FROM hits GROUP BY source ORDER BY source",
    false
)]
#[case::count_distinct(
    "SELECT source, COUNT(DISTINCT url) AS c FROM hits GROUP BY source ORDER BY source",
    false
)]
#[case::count_filter(
    "SELECT source, COUNT(*) FILTER (WHERE width > 500) AS c FROM hits GROUP BY source \
     ORDER BY source",
    false
)]
#[case::rollup(
    "SELECT source, url, COUNT(*) AS c FROM hits GROUP BY ROLLUP (source, url) \
     ORDER BY source NULLS FIRST, url NULLS FIRST",
    false
)]
#[case::volatile_key(
    "SELECT source + CAST(random() * 0 AS SMALLINT) AS k, COUNT(*) AS c FROM hits GROUP BY 1 \
     ORDER BY k",
    false
)]
#[tokio::test]
async fn count_by_matches_datafusion(
    #[case] sql: &str,
    #[case] pushed_down: bool,
) -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new());
    write_hits(&store).await?;

    let expected = session(&store, false);
    expected.sql(CREATE_HITS).await?;
    let (_, expected) = run(&expected, sql).await?;

    let actual = session(&store, true);
    actual.sql(CREATE_HITS).await?;
    let (plan, actual) = run(&actual, sql).await?;

    assert_eq!(plan.contains("VortexCountBy:"), pushed_down, "{plan}");
    assert_eq!(actual, expected);
    Ok(())
}

/// A file split across partitions counts each part's rows once.
#[tokio::test]
async fn count_by_matches_datafusion_over_split_files() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new());
    write_hits(&store).await?;
    let sql = "SELECT url, source, COUNT(*) AS c FROM hits GROUP BY url, source \
               ORDER BY url, source";
    let split = || {
        SessionConfig::new()
            .with_target_partitions(8)
            .with_repartition_file_scans(true)
            .with_repartition_file_min_size(0)
    };

    let expected = session_with_config(&store, false, split());
    expected.sql(CREATE_HITS).await?;
    let (_, expected) = run(&expected, sql).await?;

    let actual = session_with_config(&store, true, split());
    actual.sql(CREATE_HITS).await?;
    let (plan, actual) = run(&actual, sql).await?;

    assert!(plan.contains("VortexCountBy:"), "{plan}");
    assert_eq!(actual, expected);
    Ok(())
}

/// Rows a residual filter kept after the scan each count once, as the grouping expressions'
/// values over them.
#[test]
fn count_rows_counts_each_row_once() -> anyhow::Result<()> {
    let scan_schema = Arc::new(Schema::new(vec![Field::new(
        "source",
        DataType::Int16,
        false,
    )]));
    let rows = RecordBatch::try_new(
        Arc::clone(&scan_schema),
        vec![Arc::new(Int16Array::from(vec![3, 1, 3]))],
    )?;
    let count_by = CountBy {
        schema: Arc::new(Schema::new(vec![
            Field::new("source", DataType::Int16, false),
            Field::new("count", DataType::Int64, false),
        ])),
        group_by: vec![Arc::new(Column::new("source", 0))],
    };

    let expected = RecordBatch::try_new(
        Arc::clone(&count_by.schema),
        vec![
            Arc::new(Int16Array::from(vec![3, 1, 3])),
            Arc::new(Int64Array::from(vec![1, 1, 1])),
        ],
    )?;
    assert_eq!(count_rows(&rows, &count_by)?, expected);
    Ok(())
}

/// A table can opt out of the pushdown.
#[tokio::test]
async fn count_by_pushdown_can_be_disabled_per_table() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new());
    write_hits(&store).await?;
    let ctx = session(&store, true);
    ctx.sql(&CREATE_HITS.replace(
        "STORED AS vortex",
        "STORED AS vortex OPTIONS (count_by_pushdown 'false')",
    ))
    .await?;
    let (plan, _) = run(&ctx, "SELECT url, COUNT(*) AS c FROM hits GROUP BY url").await?;
    assert!(!plan.contains("VortexCountBy:"), "{plan}");
    Ok(())
}

/// Rows of a result, as `(grouping values, count)` with the count last.
async fn rows(ctx: &SessionContext, sql: &str) -> anyhow::Result<Vec<(String, i64)>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut rows = Vec::new();
    for batch in batches {
        let count = batch.num_columns() - 1;
        let counts = batch
            .column(count)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("count is not Int64"))?;
        let keys = (0..batch.num_rows())
            .map(|row| {
                (0..count)
                    .map(|column| array_value_to_string(batch.column(column), row))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| values.join("|"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.extend(keys.into_iter().zip(counts.values().iter().copied()));
    }
    Ok(rows)
}

/// The largest counts the pushed-down top-k returns are DataFusion's, and each is a group's
/// exact total. Groups tied at the cutoff may differ.
#[rstest]
#[case::dictionary_key("url", "", 5)]
#[case::computed_key(
    "source, CASE WHEN source = 0 THEN referer ELSE '' END AS src, url",
    "WHERE width > 10",
    20
)]
#[case::offset("referer", "", 3)]
#[case::nullable_key("referer", "WHERE width < 500", 7)]
#[tokio::test]
async fn top_count_matches_datafusion(
    #[case] keys: &str,
    #[case] filter: &str,
    #[case] limit: usize,
) -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new());
    write_hits(&store).await?;
    let group_by = (1..=keys.split(", ").count())
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let top = format!(
        "SELECT {keys}, COUNT(*) AS c FROM hits {filter} GROUP BY {group_by} ORDER BY c DESC LIMIT {limit}"
    );
    let all = format!("SELECT {keys}, COUNT(*) AS c FROM hits {filter} GROUP BY {group_by}");

    let expected = session(&store, false);
    expected.sql(CREATE_HITS).await?;
    let expected_top = rows(&expected, &top).await?;
    let totals = rows(&expected, &all)
        .await?
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

    let actual = session(&store, true);
    actual.sql(CREATE_HITS).await?;
    let (plan, _) = run(&actual, &top).await?;
    assert!(plan.contains("VortexTopCountExec"), "{plan}");
    let actual_top = rows(&actual, &top).await?;

    let counts = |rows: &[(String, i64)]| rows.iter().map(|(_, count)| *count).collect::<Vec<_>>();
    assert_eq!(counts(&actual_top), counts(&expected_top));
    for (key, count) in &actual_top {
        assert_eq!(totals.get(key), Some(count), "{key}");
    }
    Ok(())
}

/// A chunk's dictionary and integer columns count their rows by codes, one row per distinct
/// combination, and the grouping expressions see those representative rows.
#[test]
fn count_chunk_counts_dictionary_codes() -> anyhow::Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    let values = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
    let codes = PrimitiveArray::new(buffer![0u8, 1, 0, 2, 0, 1], Validity::NonNullable);
    let url = DictArray::try_new(codes.into_array(), values)?.into_array();
    let source = PrimitiveArray::new(buffer![1i16, 1, 1, 1, 2, 1], Validity::NonNullable);
    let chunk = StructArray::from_fields(&[("url", url), ("source", source.into_array())])?;

    let scan_fields = Fields::from(vec![
        Field::new("url", DataType::Utf8View, false),
        Field::new("source", DataType::Int16, false),
    ]);
    let scan_schema = Arc::new(Schema::new(scan_fields.clone()));
    let count_by = CountBy {
        schema: Arc::new(Schema::new(vec![
            Field::new("url", DataType::Utf8View, false),
            Field::new("source", DataType::Int16, false),
            Field::new("count", DataType::Int64, false),
        ])),
        group_by: vec![
            Arc::new(Column::new("url", 0)),
            Arc::new(Column::new("source", 1)),
        ],
    };

    let batch = count_chunk(
        chunk.into_array(),
        &count_by,
        &Field::new_struct("", scan_fields, false),
        &scan_schema,
        &mut ctx,
    )?;

    let expected = RecordBatch::try_new(
        Arc::clone(&count_by.schema),
        vec![
            Arc::new(StringViewArray::from(vec!["a", "b", "c", "a"])),
            Arc::new(Int16Array::from(vec![1, 1, 1, 2])),
            Arc::new(Int64Array::from(vec![2, 2, 1, 1])),
        ],
    )?;
    assert_eq!(batch, expected);
    Ok(())
}

/// A scan of no columns, as for a constant grouping key, counts all of a chunk's rows as one
/// group.
#[test]
fn count_chunk_counts_a_scan_of_no_columns() -> anyhow::Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    let chunk = StructArray::new_fieldless_with_len(5);
    let scan_schema = Arc::new(Schema::empty());
    let count_by = CountBy {
        schema: Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("count", DataType::Int64, false),
        ])),
        group_by: vec![Arc::new(Literal::new(ScalarValue::from("all")))],
    };

    let batch = count_chunk(
        chunk.into_array(),
        &count_by,
        &Field::new_struct("", Fields::empty(), false),
        &scan_schema,
        &mut ctx,
    )?;

    let expected = RecordBatch::try_new(
        Arc::clone(&count_by.schema),
        vec![
            Arc::new(StringArray::from(vec!["all"])),
            Arc::new(Int64Array::from(vec![5])),
        ],
    )?;
    assert_eq!(batch, expected);
    Ok(())
}

/// Representative rows of a filtered chunk with a nullable dictionary are the filtered rows'
/// values, though they are taken from the codes decoded for counting.
#[test]
fn count_chunk_takes_representatives_of_filtered_rows() -> anyhow::Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    let values = VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some("c")]).into_array();
    let codes = PrimitiveArray::new(buffer![2u16, 0, 1, 0, 2, 1, 0], Validity::NonNullable);
    let referer = DictArray::try_new(codes.into_array(), values)?.into_array();
    let width = PrimitiveArray::new(
        buffer![7i32, 5, 5, 5, 7, 5, 9],
        Validity::from_iter([true, true, true, true, true, true, true]),
    );
    let chunk = StructArray::from_fields(&[("referer", referer), ("width", width.into_array())])?
        .into_array()
        .filter(Mask::from_iter([
            false, true, true, true, true, true, false,
        ]))?;

    let scan_fields = Fields::from(vec![
        Field::new("referer", DataType::Utf8View, true),
        Field::new("width", DataType::Int32, true),
    ]);
    let scan_schema = Arc::new(Schema::new(scan_fields.clone()));
    let count_by = CountBy {
        schema: Arc::new(Schema::new(vec![
            Field::new("referer", DataType::Utf8View, true),
            Field::new("width", DataType::Int32, true),
            Field::new("count", DataType::Int64, false),
        ])),
        group_by: vec![
            Arc::new(Column::new("referer", 0)),
            Arc::new(Column::new("width", 1)),
        ],
    };

    let batch = count_chunk(
        chunk,
        &count_by,
        &Field::new_struct("", scan_fields, false),
        &scan_schema,
        &mut ctx,
    )?;

    let expected = RecordBatch::try_new(
        Arc::clone(&count_by.schema),
        vec![
            Arc::new(StringViewArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(Int32Array::from(vec![5, 5, 7])),
            Arc::new(Int64Array::from(vec![2, 2, 1])),
        ],
    )?;
    assert_eq!(batch, expected);
    Ok(())
}
