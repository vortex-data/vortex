// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int16Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringViewArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_physical_expr::expressions::Column;
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
use vortex::session::VortexSession;

use super::CountBy;
use super::count_chunk;
use crate::VortexCountByPushdown;
use crate::VortexFormatFactory;

fn session(store: &Arc<InMemory>, count_by: bool) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());
    let mut builder = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(4))
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
    (url VARCHAR NOT NULL, referer VARCHAR, source SMALLINT NOT NULL, width INT NOT NULL) \
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
                    CAST(value % 1000 AS INT) \
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
#[case::other_aggregate(
    "SELECT url, SUM(width) AS s FROM hits GROUP BY url ORDER BY url",
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

    assert_eq!(plan.contains("VortexCountByExec"), pushed_down, "{plan}");
    assert_eq!(actual, expected);
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
