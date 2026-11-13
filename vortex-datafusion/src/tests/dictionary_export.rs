// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end checks that exporting dictionary-encoded strings as Arrow dictionaries, see
//! [`VortexTableOptions::dictionary_export`], changes no query result.

use std::sync::Arc;

use arrow_array::Decimal128Array;
use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::arrow::util::pretty::pretty_format_batches;
use rstest::rstest;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;

const SALES_ROWS: i32 = 30_000;
const STORES: i32 = 12;

fn sales_batch() -> VortexResult<RecordBatch> {
    let sk = Int32Array::from_iter_values(0..SALES_ROWS);
    let store =
        StringArray::from_iter_values((0..SALES_ROWS).map(|i| format!("store_{}", i % STORES)));
    // Nullable, with a few hundred distinct values so its codes need more than eight bits.
    let item = StringArray::from_iter(
        (0..SALES_ROWS).map(|i| (i % 97 != 0).then(|| format!("item_{:03}", (i * 7) % 300))),
    );
    let qty = Int32Array::from_iter_values((0..SALES_ROWS).map(|i| i % 50));
    let price = Decimal128Array::from_iter_values((0..SALES_ROWS).map(|i| i128::from(i % 10_000)))
        .with_precision_and_scale(7, 2)?;
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("sk", DataType::Int32, false),
            Field::new("store", DataType::Utf8, false),
            Field::new("item", DataType::Utf8, true),
            Field::new("qty", DataType::Int32, false),
            Field::new("price", DataType::Decimal128(7, 2), false),
        ])),
        vec![
            Arc::new(sk),
            Arc::new(store),
            Arc::new(item),
            Arc::new(qty),
            Arc::new(price),
        ],
    )?)
}

fn stores_batch() -> VortexResult<RecordBatch> {
    let name = StringArray::from_iter_values((0..STORES).map(|i| format!("store_{i}")));
    let state =
        StringArray::from_iter_values((0..STORES).map(|i| ["CA", "NY", "TX"][(i % 3) as usize]));
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_state", DataType::Utf8, false),
        ])),
        vec![Arc::new(name), Arc::new(state)],
    )?)
}

async fn session(dictionary_export: bool, scan_v2: bool) -> VortexResult<TestSessionContext> {
    let options = VortexTableOptions {
        scan_v2,
        projection_pushdown: true,
        predicate_pushdown: true,
        dictionary_export,
        ..Default::default()
    };
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(options),
    ));
    let sales = sales_batch()?;
    // Two files, so schema inference reconciles the dictionary columns of both.
    for (idx, rows) in [
        (0, 0..SALES_ROWS as usize / 2),
        (1, SALES_ROWS as usize / 2..SALES_ROWS as usize),
    ] {
        ctx.write_arrow_batch(
            format!("sales/part{idx}.vortex").as_str(),
            &sales.slice(rows.start, rows.len()),
        )
        .await
        .map_err(|e| vortex_err!("{e}"))?;
    }
    ctx.write_arrow_batch("stores/part0.vortex", &stores_batch()?)
        .await
        .map_err(|e| vortex_err!("{e}"))?;
    for (table, location) in [("sales", "/sales/"), ("stores", "/stores/")] {
        ctx.session
            .sql(&format!(
                "CREATE EXTERNAL TABLE {table} STORED AS vortex LOCATION '{location}'"
            ))
            .await
            .map_err(|e| vortex_err!("{e}"))?;
    }
    Ok(ctx)
}

async fn query(ctx: &TestSessionContext, sql: &str) -> VortexResult<String> {
    let batches = ctx
        .session
        .sql(sql)
        .await
        .map_err(|e| vortex_err!("planning {sql}: {e}"))?
        .collect()
        .await
        .map_err(|e| vortex_err!("executing {sql}: {e}"))?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

async fn physical_plan(ctx: &TestSessionContext, sql: &str) -> VortexResult<String> {
    query(ctx, &format!("EXPLAIN FORMAT INDENT {sql}")).await
}

#[rstest]
#[tokio::test]
async fn dictionary_columns_are_inferred(#[values(false, true)] scan_v2: bool) -> VortexResult<()> {
    let ctx = session(true, scan_v2).await?;
    let schema = ctx
        .session
        .table("sales")
        .await
        .map_err(|e| vortex_err!("{e}"))?
        .schema()
        .as_arrow()
        .clone();
    let dict = |values| DataType::Dictionary(Box::new(DataType::UInt32), Box::new(values));
    assert_eq!(
        schema.field_with_name("store")?.data_type(),
        &dict(DataType::Utf8View)
    );
    assert_eq!(
        schema.field_with_name("item")?.data_type(),
        &dict(DataType::Utf8View)
    );
    assert_eq!(schema.field_with_name("sk")?.data_type(), &DataType::Int32);

    let plain = session(false, scan_v2).await?;
    let schema = plain
        .session
        .table("sales")
        .await
        .map_err(|e| vortex_err!("{e}"))?
        .schema()
        .as_arrow()
        .clone();
    assert_eq!(
        schema.field_with_name("store")?.data_type(),
        &DataType::Utf8View
    );
    Ok(())
}

#[rstest]
#[case::filter_eq("SELECT count(*), sum(qty) FROM sales WHERE store = 'store_3'")]
#[case::filter_neq("SELECT count(*) FROM sales WHERE store <> 'store_3' AND item <> 'item_007'")]
#[case::filter_range("SELECT count(*) FROM sales WHERE store > 'store_5' AND item <= 'item_100'")]
#[case::filter_in("SELECT count(*), sum(qty) FROM sales WHERE store IN ('store_1', 'store_2')")]
#[case::filter_like("SELECT count(*) FROM sales WHERE item LIKE 'item_1%'")]
#[case::filter_not_like(
    "SELECT count(*) FROM sales WHERE item NOT LIKE '%5' AND store LIKE 'store_1_'"
)]
#[case::filter_is_null("SELECT count(*) FROM sales WHERE item IS NULL")]
#[case::filter_substr("SELECT count(*) FROM sales WHERE substr(item, 6, 1) IN ('1', '2')")]
#[case::filter_mixed(
    "SELECT sum(qty) FROM sales WHERE store = 'store_4' AND qty > 10 AND price > 50"
)]
#[case::group_single("SELECT store, count(*), sum(qty) FROM sales GROUP BY store ORDER BY store")]
#[case::group_multi(
    "SELECT store, item, count(*), sum(price) FROM sales GROUP BY store, item ORDER BY store, item"
)]
#[case::group_mixed_types(
    "SELECT item, qty, count(*) FROM sales WHERE qty < 3 GROUP BY item, qty ORDER BY item, qty"
)]
#[case::count_distinct("SELECT count(DISTINCT item), count(DISTINCT store) FROM sales")]
#[case::join_on_string(
    "SELECT s_state, count(*), sum(qty) FROM sales JOIN stores ON store = s_name \
     GROUP BY s_state ORDER BY s_state"
)]
#[case::self_join_on_dictionaries(
    "SELECT count(*) FROM (SELECT DISTINCT store, item FROM sales WHERE qty = 1) a \
     JOIN (SELECT DISTINCT store, item FROM sales WHERE qty = 2) b \
     ON a.store = b.store AND a.item = b.item"
)]
#[case::sort("SELECT store, item, sk FROM sales ORDER BY item DESC NULLS LAST, store, sk LIMIT 25")]
#[case::top_k_by_string("SELECT item FROM sales WHERE item IS NOT NULL ORDER BY item LIMIT 5")]
#[case::substr_group(
    "SELECT substr(item, 1, 6) AS prefix, count(*) FROM sales GROUP BY prefix ORDER BY prefix"
)]
#[case::string_functions(
    "SELECT upper(store) || '-' || coalesce(item, 'none') AS k, length(store), count(*) \
     FROM sales GROUP BY k, length(store) ORDER BY k LIMIT 30"
)]
#[case::case_when(
    "SELECT sum(CASE WHEN store = 'store_1' THEN qty WHEN item LIKE '%9' THEN 2 * qty ELSE 0 END), \
     count(CASE WHEN store IN ('store_2', 'store_3') THEN 1 END) FROM sales"
)]
#[case::case_projection(
    "SELECT CASE WHEN qty > 25 THEN store ELSE item END AS k, count(*) FROM sales \
     GROUP BY k ORDER BY k NULLS FIRST LIMIT 20"
)]
#[case::union_with_plain_strings(
    "SELECT n FROM (SELECT store AS n FROM sales WHERE sk < 5 \
     UNION ALL SELECT s_name AS n FROM stores) ORDER BY n"
)]
#[case::window(
    "SELECT store, total, rank() OVER (ORDER BY total DESC) AS r FROM \
     (SELECT store, sum(qty) AS total FROM sales GROUP BY store) ORDER BY r, store"
)]
#[case::min_max("SELECT min(store), max(item), min(item) FROM sales")]
#[tokio::test]
async fn dictionary_export_matches_plain(
    #[case] sql: &str,
    #[values(false, true)] scan_v2: bool,
) -> VortexResult<()> {
    let plain = session(false, scan_v2).await?;
    let dict = session(true, scan_v2).await?;
    assert_eq!(query(&dict, sql).await?, query(&plain, sql).await?, "{sql}");
    Ok(())
}

/// Comparing a dictionary column with a literal stays in the scan, as it does for `Utf8View`.
#[rstest]
#[case::eq("SELECT sk FROM sales WHERE store = 'store_3'")]
#[case::in_list("SELECT sk FROM sales WHERE store IN ('store_1', 'store_2')")]
#[case::like("SELECT sk FROM sales WHERE item LIKE 'item_1%'")]
#[case::range("SELECT sk FROM sales WHERE item > 'item_100'")]
#[case::is_null("SELECT sk FROM sales WHERE item IS NULL")]
#[tokio::test]
async fn dictionary_filters_are_pushed_down(
    #[case] sql: &str,
    #[values(false, true)] scan_v2: bool,
) -> VortexResult<()> {
    let plain = session(false, scan_v2).await?;
    let dict = session(true, scan_v2).await?;
    let plain_plan = physical_plan(&plain, sql).await?;
    let dict_plan = physical_plan(&dict, sql).await?;
    assert!(!plain_plan.contains("FilterExec"), "{plain_plan}");
    assert!(!dict_plan.contains("FilterExec"), "{dict_plan}");
    Ok(())
}

/// The scan hands DataFusion dictionaries, and the batches of every split of one Vortex
/// dictionary share its exported values.
#[rstest]
#[tokio::test]
async fn dictionary_values_are_shared(#[values(false, true)] scan_v2: bool) -> VortexResult<()> {
    // Enough rows for several scan splits over the one dictionary of the file.
    const ROWS: usize = 300_000;
    let options = VortexTableOptions {
        scan_v2,
        dictionary_export: true,
        ..Default::default()
    };
    let ctx = TestSessionContext::new_with_factory(Arc::new(
        VortexFormatFactory::new().with_options(options),
    ));
    let store = StringArray::from_iter_values((0..ROWS).map(|i| format!("store_{}", i % 7)));
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "store",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(store)],
    )?;
    ctx.write_arrow_batch("big/part0.vortex", &batch)
        .await
        .map_err(|e| vortex_err!("{e}"))?;
    ctx.session
        .sql("CREATE EXTERNAL TABLE big STORED AS vortex LOCATION '/big/'")
        .await
        .map_err(|e| vortex_err!("{e}"))?;
    let batches = ctx
        .session
        .sql("SELECT store FROM big")
        .await
        .map_err(|e| vortex_err!("{e}"))?
        .collect()
        .await
        .map_err(|e| vortex_err!("{e}"))?;
    let values = batches
        .iter()
        .map(|batch| {
            batch
                .column(0)
                .as_dictionary_opt::<UInt32Type>()
                .map(|dict| Arc::as_ptr(dict.values()).cast::<()>().addr())
                .ok_or_else(|| vortex_err!("not a dictionary: {}", batch.column(0).data_type()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let mut distinct = values.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 1, "{} batches", values.len());
    Ok(())
}
