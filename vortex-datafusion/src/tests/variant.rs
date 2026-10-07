// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `variant_get` over shredded Variant columns in Vortex files.

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_common::assert_batches_eq;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use rstest::rstest;

use crate::common_tests::TestSessionContext;
use crate::variant::variant_get_udf;

/// `{id: Int64, data: Variant}` with `kind` and `commit.collection` shredded.
fn make_batch() -> anyhow::Result<RecordBatch> {
    let json: ArrowArrayRef = Arc::new(StringArray::from(vec![
        r#"{"kind": "commit", "commit": {"collection": "post", "rev": 1}}"#,
        r#"{"kind": "identity"}"#,
        r#"{"kind": "commit", "commit": {"collection": 7}}"#,
        r#"{"kind": "commit", "commit": {"collection": "like"}}"#,
        r#"[1, 2]"#,
    ]));
    let schema = ShreddedSchemaBuilder::new()
        .with_path("kind", &DataType::Utf8)?
        .with_path("commit.collection", &DataType::Utf8)?
        .build();
    let variant = shred_variant(&json_to_variant(&json)?, &schema)?;
    let id: ArrowArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
    Ok(RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", DataType::Int64, false),
            variant.field("data"),
        ])),
        vec![id, ArrowArrayRef::from(variant)],
    )?)
}

#[rstest]
#[tokio::test]
async fn variant_get_reads_paths(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);
    ctx.session.register_udf(variant_get_udf());

    let batch = make_batch()?;
    ctx.write_arrow_batch("files/variant.vortex", &batch)
        .await?;
    let provider = ctx
        .table_provider("t", "/files/", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("t", provider)?;

    let sql = "SELECT id, variant_get(data, 'commit.collection', 'Utf8') AS collection, \
               variant_get(data, 'commit.rev', 'Int64') AS rev \
               FROM t WHERE variant_get(data, 'kind', 'Utf8') = 'commit' ORDER BY id";
    let result = ctx.session.sql(sql).await?.collect().await?;
    assert_batches_eq!(
        [
            "+----+------------+-----+",
            "| id | collection | rev |",
            "+----+------------+-----+",
            "| 1  | post       | 1   |",
            "| 3  |            |     |",
            "| 4  | like       |     |",
            "+----+------------+-----+",
        ],
        &result
    );

    if projection_pushdown {
        let plan = ctx
            .session
            .sql(&format!("EXPLAIN {sql}"))
            .await?
            .collect()
            .await?;
        let plan = pretty_format_batches(&plan)?.to_string();
        let scan = plan
            .lines()
            .find(|line| line.contains("DataSourceExec"))
            .ok_or_else(|| anyhow::anyhow!("no scan in plan:\n{plan}"))?;
        assert!(
            scan.contains("projection=[") && scan.contains("variant_get(data@"),
            "variant_get must be pushed into the scan:\n{plan}"
        );
    }
    Ok(())
}
