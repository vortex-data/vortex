// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `variant_get` over Vortex files with a shredded Variant column.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_common::assert_batches_eq;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use rstest::rstest;

use crate::common_tests::TestSessionContext;
use crate::variant::VariantGetUdf;

/// A `data` Variant column with `a` shredded as an integer and `b.c` as a string. One document
/// holds a string `a`, which stays in the residual.
fn variant_batch() -> anyhow::Result<RecordBatch> {
    let documents: ArrowArrayRef = Arc::new(StringArray::from(vec![
        r#"{"a": 1, "b": {"c": "x"}, "e": "keep"}"#,
        r#"{"a": 2, "b": {"c": "y"}}"#,
        r#"{"a": "three", "b": {"c": "y"}, "e": "more"}"#,
        r#"[1, 2]"#,
    ]));
    let shredding = DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new(
            "b",
            DataType::Struct(Fields::from(vec![Field::new("c", DataType::Utf8, true)])),
            true,
        ),
    ]));
    let variant = shred_variant(&json_to_variant(&documents)?, &shredding)?;
    let field = variant.field("data");
    Ok(RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![field])),
        vec![ArrowArrayRef::from(variant)],
    )?)
}

async fn variant_table(projection_pushdown: bool) -> anyhow::Result<TestSessionContext> {
    let ctx = TestSessionContext::new(projection_pushdown);
    let batch = variant_batch()?;
    ctx.write_arrow_batch("files/variant.vortex", &batch)
        .await?;
    let provider = ctx
        .table_provider("docs", "/files/", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("docs", provider)?;
    ctx.session.register_udf(VariantGetUdf::udf());
    Ok(ctx)
}

#[rstest]
#[tokio::test]
async fn variant_get_projects_and_filters(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = variant_table(projection_pushdown).await?;

    let result = ctx
        .session
        .sql(
            "SELECT variant_get(data, 'a', 'Int64') AS a, \
                    variant_get(data, 'a', 'Utf8') AS a_str, \
                    variant_get(data, '$.e', 'Utf8') AS e \
             FROM docs \
             WHERE variant_get(data, 'b.c', 'Utf8') = 'y' \
             ORDER BY e NULLS FIRST",
        )
        .await?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+---+-------+------+",
            "| a | a_str | e    |",
            "+---+-------+------+",
            "| 2 |       |      |",
            "|   | three | more |",
            "+---+-------+------+",
        ],
        &result
    );
    Ok(())
}

#[tokio::test]
async fn variant_get_pushes_into_vortex_scan() -> anyhow::Result<()> {
    let ctx = variant_table(true).await?;

    let plan = ctx
        .session
        .sql(
            "EXPLAIN SELECT variant_get(data, 'b.c', 'Utf8') AS c FROM docs \
             WHERE variant_get(data, 'a', 'Int64') > 1",
        )
        .await?
        .collect()
        .await?;
    let plan = pretty_format_batches(&plan)?.to_string();
    let scan = plan
        .lines()
        .find(|line| line.contains("DataSourceExec"))
        .ok_or_else(|| anyhow::anyhow!("no DataSourceExec in plan:\n{plan}"))?;
    assert!(
        scan.contains("projection=[variant_get(data@0, b.c, Utf8)")
            && scan.contains("predicate: variant_get(data@0, a, Int64) > 1"),
        "variant_get was not pushed into the scan:\n{plan}"
    );
    Ok(())
}
