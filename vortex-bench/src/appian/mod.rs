// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Appian benchmark.
//!
//! Mirrors the queries from DuckDB's in-tree `benchmark/appian_benchmarks` suite. Upstream
//! ships the data as a single `.duckdb` blob (~593 MB); we download it once and shell out
//! to the `duckdb` CLI to project each table into Parquet, lowercasing column names along
//! the way. `data-gen` then handles every other format from those Parquet files.
//!
//! ## Identifier case
//!
//! The upstream `.duckdb` blob preserves camelCase column names (`orderItem_quantity`,
//! `address_customerId`, ...) and capitalized table names (`CustomerView`). The Appian
//! queries reference those identifiers unquoted, which would break under DataFusion's
//! default `enable_ident_normalization=true` (parser lowercases identifier references
//! while the Parquet schema and registered table names preserve case → field-not-found).
//!
//! The conversion below lowercases every column at COPY time, and the table names in
//! [`TABLES`] are already lowercase. Both engines then resolve the verbatim camelCase
//! queries the same way: DataFusion lowercases the query identifiers and matches them
//! against the lowercased Parquet schema, while DuckDB's case-insensitive unquoted
//! identifier resolution makes the original case irrelevant.

use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;
use glob::Pattern;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::datasets::data_downloads::download_data;
use crate::utils::file::resolve_data_url;

/// Upstream `.duckdb` blob; pinned to the URL hard-coded into DuckDB's
/// `benchmark/appian_benchmarks/appian.benchmark.in`.
const UPSTREAM_BLOB_URL: &str = "https://blobs.duckdb.org/data/appian_benchmark_data.duckdb";

/// Table names from DuckDB's `appian.benchmark.in` template in upstream case. Ordering
/// must match [`TABLES`] so each upstream source maps to its lowercased Parquet output.
const UPSTREAM_TABLES: &[&str] = &[
    "AddressView",
    "CategoryView",
    "CreditCardView",
    "CustomerView",
    "OrderItemNovelty_Update",
    "OrderItemView",
    "OrderView",
    "ProductView",
    "TaxRecordView",
];

/// Lowercased table names registered with the query engines. Matches the output Parquet
/// file names produced by [`AppianBenchmark::generate_base_data`].
const TABLES: &[&str] = &[
    "addressview",
    "categoryview",
    "creditcardview",
    "customerview",
    "orderitemnovelty_update",
    "orderitemview",
    "orderview",
    "productview",
    "taxrecordview",
];

/// Eight join-heavy queries copied verbatim from
/// `duckdb/duckdb:benchmark/appian_benchmarks/queries/q0[1-8].sql`.
const QUERIES: &[&str] = &[
    // q01 — three-way left join, group by state, sum order-item quantities.
    "SELECT address_state AS g0, sum(orderItem_quantity) AS p0
FROM CustomerView c
LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
GROUP BY address_state
ORDER BY address_state
LIMIT 500",
    // q02 — eight-CTE breadth-first aggregation across the whole schema.
    "SELECT
    a.address_state AS g0,
    t1rp1 AS g1,
    t2rp1 AS g2,
    max(t5rp1) AS p0,
    avg(t8rp1 * t8rp2) AS p1,
    max(t6rp1) AS p2,
    count(c.customer_priority) AS p3,
    coalesce(avg(t7rp1), 0.0) AS p4
FROM CustomerView c
LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
LEFT OUTER JOIN TaxRecordView t ON a.address_id = t.taxRecord_addressId
LEFT OUTER JOIN (
        SELECT sum(creditCard_cvv) AS t1rp1, c.customer_id AS t1pk
        FROM CustomerView c
        LEFT OUTER JOIN CreditCardView cc ON c.customer_id = cc.creditCard_customerId
        GROUP BY c.customer_id
    ) t1 ON c.customer_id = t1.t1pk
LEFT OUTER JOIN (
        SELECT min(p.product_likes) AS t2rp1, c.customer_id AS t2pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        WHERE ca.category_seasonal = TRUE
        GROUP BY c.customer_id
    ) t2 ON c.customer_id = t2.t2pk
LEFT OUTER JOIN (
        SELECT max(o.order_subShipments) AS t5rp1, c.customer_id AS t5pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        GROUP BY c.customer_id
    ) t5 ON c.customer_id = t5pk
LEFT OUTER JOIN (
        SELECT max(coalesce(oi.orderItem_weight, 1)) AS t6rp1, c.customer_id AS t6pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        WHERE o.order_serverId IN (1, 3, 5)
        GROUP BY c.customer_id
    ) t6 ON c.customer_id = t6pk
LEFT OUTER JOIN (
        SELECT count(ca.category_seasonal) AS t7rp1, c.customer_id AS t7pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        WHERE ca.category_perishable = TRUE
        GROUP BY c.customer_id
    ) t7 ON c.customer_id = t7pk
LEFT OUTER JOIN (
        SELECT
            sum(creditCard_zip) AS t8rp1,
            sum(creditCard_lastChargeAmount) AS t8rp2,
            c.customer_id AS t8pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN CreditCardView cc ON o.order_creditCardNumber = cc.creditCard_number
        GROUP BY c.customer_id
    ) t8 ON c.customer_id = t8pk
WHERE t.taxRecord_value > 149670.0
GROUP BY a.address_state, t1rp1, t2rp1
ORDER BY g0, p0, p1
LIMIT 500",
    // q03 — many-way star join with a CASE expression over a date diff.
    "SELECT
    c.customer_priority AS g0,
    t1rp1 AS g1,
    t.taxRecord_bracket AS g2,
    sum(oi.orderItem_weight) AS p0,
    max(ca.category_demandScore) AS p1,
    max(ca.category_auditDate) AS p2,
    CAST(avg(ca.category_valuation) AS int) AS p3,
    sum(t1rp2) AS p4,
    sum(
        CASE
            WHEN p.product_inventoryLastOrderedOn - ca.category_auditDate > 300 THEN 1
            WHEN p.product_inventoryLastOrderedOn - ca.category_auditDate > 150 THEN 10
            WHEN p.product_inventoryLastOrderedOn - ca.category_auditDate > 0 THEN 100
            ELSE 1000
        END +(c.customer_priority * a.address_zone)) AS p5
FROM OrderItemView oi
LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
LEFT OUTER JOIN CreditCardView cc ON o.order_creditCardNumber = cc.creditCard_number
LEFT OUTER JOIN CustomerView c ON o.order_customerId = c.customer_id
LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
LEFT OUTER JOIN TaxRecordView t ON a.address_id = t.taxRecord_addressId
LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
LEFT OUTER JOIN (
        SELECT
            min(cc.creditCard_expirationDate) AS t1rp1,
            sum(cc.creditCard_lastChargeAmount) AS t1rp2,
            c.customer_id AS t1pk
        FROM CustomerView c
        LEFT OUTER JOIN CreditCardView cc ON c.customer_id = cc.creditCard_customerId
        GROUP BY c.customer_id
    ) t1 ON c.customer_id = t1pk
WHERE cc.creditCard_lastChargeAmount > 90.0 AND p.product_price > 34.0
GROUP BY c.customer_priority, t1rp1, t.taxRecord_bracket
ORDER BY p1, p3, g2
LIMIT 500",
    // q04 — category-rooted fan-out with four parallel sub-aggregations.
    "SELECT
    t2rp1 AS g0,
    t3rp1 AS g1,
    t4rp1 AS g2,
    CAST(avg(cc.creditCard_lastChargeAmount) AS int) AS p0,
    min(cc.creditCard_lastChargeTimestamp) AS p1,
    count(DISTINCT (cc.creditCard_holder)) AS p2
FROM CategoryView ca
LEFT OUTER JOIN ProductView p ON ca.category_name = p.product_categoryName
LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
LEFT OUTER JOIN CreditCardView cc ON o.order_creditCardNumber = cc.creditCard_number
LEFT OUTER JOIN (
        SELECT sum(taxRecord_bracket) AS t1rp1, ca.category_name AS t1pk
        FROM CategoryView ca
        LEFT OUTER JOIN ProductView p ON ca.category_name = p.product_categoryName
        LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
        LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
        LEFT OUTER JOIN CustomerView c ON o.order_customerId = c.customer_id
        LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
        LEFT OUTER JOIN TaxRecordView t ON a.address_id = t.taxRecord_addressId
        GROUP BY ca.category_name
    ) t1 ON ca.category_name = t1pk
LEFT OUTER JOIN (
        SELECT max(p.product_likes) AS t2rp1, ca.category_name AS t2pk
        FROM CategoryView ca
        LEFT OUTER JOIN ProductView p ON ca.category_name = p.product_categoryName
        GROUP BY ca.category_name
    ) t2 ON ca.category_name = t2pk
LEFT OUTER JOIN (
        SELECT sum(oi.orderItem_productGroup) AS t3rp1, ca.category_name AS t3pk
        FROM CategoryView ca
        LEFT OUTER JOIN ProductView p ON ca.category_name = p.product_categoryName
        LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
        WHERE oi.orderItem_weight > 15.0
        GROUP BY ca.category_name
    ) t3 ON ca.category_name = t3pk
LEFT OUTER JOIN (
        SELECT max(cc.creditCard_zip) AS t4rp1, ca.category_name AS t4pk
        FROM CategoryView ca
        LEFT OUTER JOIN ProductView p ON ca.category_name = p.product_categoryName
        LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
        LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
        LEFT OUTER JOIN CreditCardView cc ON o.order_creditCardNumber = cc.creditCard_number
        GROUP BY ca.category_name
    ) t4 ON ca.category_name = t4pk
WHERE t1rp1 > 6
GROUP BY t2rp1, t3rp1, t4rp1
ORDER BY g1, p2
LIMIT 500",
    // q05 — tax-record rooted query with a timestamp-bound subquery.
    "SELECT t.taxRecord_rate AS g0, t2rp1 AS g1, min(c.customer_balance) AS p0
FROM TaxRecordView t
LEFT OUTER JOIN AddressView a ON t.taxRecord_addressId = a.address_id
LEFT OUTER JOIN CustomerView c ON a.address_customerId = c.customer_id
LEFT OUTER JOIN (
        SELECT min(o.order_placedOn) AS t1rp1, t.taxRecord_id AS t1pk
        FROM TaxRecordView t
        LEFT OUTER JOIN AddressView a ON t.taxRecord_addressId = a.address_id
        LEFT OUTER JOIN OrderView o ON a.address_customerId = o.order_customerId
        GROUP BY t.taxRecord_id
    ) t1 ON t.taxRecord_id = t1pk
LEFT OUTER JOIN (
        SELECT sum(p.product_price * oi.orderItem_quantity) AS t2rp1, t.taxRecord_id AS t2pk
        FROM TaxRecordView t
        LEFT OUTER JOIN AddressView a ON t.taxRecord_addressId = a.address_id
        LEFT OUTER JOIN OrderView o ON a.address_customerId = o.order_customerId
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        GROUP BY t.taxRecord_id
    ) t2 ON t.taxRecord_id = t2pk
WHERE t1rp1 > '2020-01-14 12:12:30.0'
GROUP BY t.taxRecord_rate, t2rp1
ORDER BY p0
LIMIT 500",
    // q06 — product-rooted with cascading CASE buckets over CTE outputs.
    "SELECT
    t1rp2 AS g0,
    sum(t1rp3) / sum(t1rp4) AS p0,
    sum(
        CASE
            WHEN t1rp5 > 1 THEN 1
            WHEN t2rp1 > 20200 THEN 2
            WHEN t1rp6 > 15 THEN 3
            WHEN t3rp1 > 150 THEN 4
            ELSE 5
        END) AS p1
FROM ProductView p
LEFT OUTER JOIN (
        SELECT
            avg(a.address_valuation) AS t1rp1,
            sum(a.address_zone) AS t1rp2,
            sum(a.address_zone) AS t1rp3,
            count(a.address_zone) AS t1rp4,
            avg(o.order_serverId) AS t1rp5,
            avg(c.customer_balance) AS t1rp6,
            p.product_id AS t1pk
        FROM ProductView p
        LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
        LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
        LEFT OUTER JOIN AddressView a ON o.order_customerId = a.address_customerId
        LEFT OUTER JOIN CustomerView c ON o.order_customerId = c.customer_id
        GROUP BY p.product_id
    ) t1 ON p.product_id = t1pk
LEFT OUTER JOIN (
        SELECT min(a.address_zip) AS t2rp1, p.product_id AS t2pk
        FROM ProductView p
        LEFT OUTER JOIN OrderItemView oi ON p.product_id = oi.orderItem_productId
        LEFT OUTER JOIN OrderView o ON oi.orderItem_orderId = o.order_id
        LEFT OUTER JOIN AddressView a ON o.order_customerId = a.address_customerId
        WHERE a.address_state IN ('PA', 'CA', 'VA', 'MA', 'ME', 'MD', 'CO', 'MO')
        GROUP BY p.product_id
    ) t2 ON p.product_id = t2pk
LEFT OUTER JOIN (
        SELECT ca.category_warehouseSqft AS t3rp1, p.product_id AS t3pk
        FROM ProductView p
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        WHERE ca.category_seasonal = TRUE
    ) t3 ON p.product_id = t3pk
WHERE t1rp1 > 10000.0
GROUP BY t1rp2
ORDER BY p0
LIMIT 500",
    // q07 — customer-rooted with derived divisions and an IN filter on CC fields.
    "SELECT
    t1rp1 AS g0,
    t2rp1 AS g1,
    c.customer_age AS g2,
    c.customer_balance AS g3,
    count(c.customer_name) AS p0,
    sum(c.customer_age) AS p1
FROM CustomerView c
LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
LEFT OUTER JOIN TaxRecordView t ON a.address_id = t.taxRecord_addressId
LEFT OUTER JOIN (
        SELECT avg(oi.orderItem_weight) AS t1rp1, c.customer_id AS t1pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN CreditCardView cc ON o.order_creditCardNumber = cc.creditCard_number
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        WHERE creditCard_cvv IN (113, 115, 117, 119, 121)
        GROUP BY c.customer_id
    ) t1 ON c.customer_id = t1pk
LEFT OUTER JOIN (
        SELECT
            avg((oi.orderItem_quantity * p.product_price) /(oi.orderItem_weight + oi.orderItem_sku)) AS t2rp1,
            c.customer_id AS t2pk
        FROM CustomerView c
        LEFT OUTER JOIN OrderView o ON c.customer_id = o.order_customerId
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        WHERE ca.category_name IN ('Pet', 'Food', 'Game', 'Software')
        GROUP BY c.customer_id
    ) t2 ON c.customer_id = t2pk
WHERE t.taxRecord_bracketThreshold IN (22, 24, 27, 29)
GROUP BY t1rp1, t2rp1, c.customer_age, c.customer_balance
ORDER BY p0, p1
LIMIT 500",
    // q08 — credit-card-rooted query with six parallel sub-aggregations.
    "SELECT
    t4rp1 AS g0,
    t5rp1 AS g1,
    sum(creditCard_lastChargeAmount) AS p0,
    min(t6rp1) AS p1,
    sum(t3rp2) AS p2
FROM CreditCardView cc
LEFT OUTER JOIN (
        SELECT min(order_id) AS t1rp1, creditCard_number AS t1pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        WHERE order_slaProbability > 0.125
        GROUP BY creditCard_number
    ) t1 ON cc.creditCard_number = t1pk
LEFT OUTER JOIN (
        SELECT sum(orderItem_weight) AS t2rp1, creditCard_number AS t2pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        GROUP BY creditCard_number
    ) t2 ON cc.creditCard_number = t2pk
LEFT OUTER JOIN (
        SELECT
            min(address_zip) AS t3rp1,
            sum(taxRecord_bracketThreshold) AS t3rp2,
            creditCard_number AS t3pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        LEFT OUTER JOIN CustomerView c ON o.order_customerId = c.customer_id
        LEFT OUTER JOIN AddressView a ON c.customer_id = a.address_customerId
        LEFT OUTER JOIN TaxRecordView t ON a.address_id = t.taxRecord_addressId
        GROUP BY creditCard_number
    ) t3 ON cc.creditCard_number = t3pk
LEFT OUTER JOIN (
        SELECT sum(product_price) AS t4rp1, creditCard_number AS t4pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        WHERE orderItem_weight < 25.0
        GROUP BY creditCard_number
    ) t4 ON cc.creditCard_number = t4pk
LEFT OUTER JOIN (
        SELECT sum(category_regulationProbability) AS t5rp1, creditCard_number AS t5pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        GROUP BY creditCard_number
    ) t5 ON cc.creditCard_number = t5pk
LEFT OUTER JOIN (
        SELECT min(product_inventoryLastOrderedOn) AS t6rp1, creditCard_number AS t6pk
        FROM CreditCardView cc
        LEFT OUTER JOIN OrderView o ON cc.creditCard_number = o.order_creditCardNumber
        LEFT OUTER JOIN OrderItemView oi ON o.order_id = oi.orderItem_orderId
        LEFT OUTER JOIN ProductView p ON oi.orderItem_productId = p.product_id
        LEFT OUTER JOIN CategoryView ca ON p.product_categoryName = ca.category_name
        WHERE product_price < 200.0
        GROUP BY creditCard_number
    ) t6 ON cc.creditCard_number = t6pk
WHERE t1rp1 > 10000 OR t2rp1 > 15 OR t3rp1 > 20200
GROUP BY t4rp1, t5rp1
ORDER BY p0, p1, p2
LIMIT 500",
];

/// Benchmark over the [Appian benchmark suite from DuckDB][upstream].
///
/// [upstream]: https://github.com/duckdb/duckdb/tree/main/benchmark/appian_benchmarks
pub struct AppianBenchmark {
    data_url: Url,
}

impl AppianBenchmark {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }

    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url(use_remote_data_dir.as_deref(), "appian")?;
        Ok(Self { data_url })
    }

    fn base_dir(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!(
                "Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"
            ))
    }

    fn parquet_dir(&self) -> anyhow::Result<PathBuf> {
        Ok(self.base_dir()?.join(Format::Parquet.name()))
    }
}

#[async_trait::async_trait]
impl Benchmark for AppianBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(QUERIES.iter().map(|s| s.to_string()).enumerate().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let parquet_dir = self.parquet_dir()?;
        std::fs::create_dir_all(&parquet_dir)?;

        // Idempotency: if every target Parquet is already in place, do nothing.
        if TABLES
            .iter()
            .all(|t| parquet_dir.join(format!("{t}.parquet")).exists())
        {
            info!(
                "appian: {} Parquet shards already present in {}",
                TABLES.len(),
                parquet_dir.display(),
            );
            return Ok(());
        }

        // Download the upstream `.duckdb` blob into the dataset cache directory.
        let blob_path = self.base_dir()?.join("appian_benchmark_data.duckdb");
        let blob = download_data(blob_path, UPSTREAM_BLOB_URL).await?;

        // DuckDB SQL can't use a query result as a projection list, so build per-table
        // lowercased projections in Rust, then run all nine `COPY`s in a single subprocess.
        let projections = discover_projections(&blob)?;
        let mut script = format!("ATTACH '{}' AS src (READ_ONLY);\n", blob.display());
        for (i, &upstream) in UPSTREAM_TABLES.iter().enumerate() {
            let projection = projections
                .iter()
                .find(|(t, _)| t == upstream)
                .map(|(_, p)| p.as_str())
                .with_context(|| format!("no columns reported for upstream table {upstream}"))?;
            let out_path = parquet_dir.join(format!("{}.parquet", TABLES[i]));
            script.push_str(&format!(
                "COPY (SELECT {projection} FROM src.\"{upstream}\") TO '{}' (FORMAT PARQUET);\n",
                out_path.display(),
            ));
        }

        let output = Command::new("duckdb").arg("-c").arg(&script).output()?;
        if !output.status.success() {
            bail!(
                "duckdb appian COPY failed: stdout={:?} stderr={:?}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        info!(
            "appian base data generated in {} ({} Parquet shards)",
            parquet_dir.display(),
            TABLES.len(),
        );
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Appian
    }

    fn dataset_name(&self) -> &str {
        "appian"
    }

    fn dataset_display(&self) -> String {
        "appian".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        TABLES
            .iter()
            .map(|name| TableSpec::new(name, None))
            .collect()
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}

/// Run a single `duckdb` invocation that returns, for each upstream Appian table, a
/// projection string of the form `"OrigName" AS "origname", ...` so the `COPY` statements
/// below can lowercase every column name without enumerating them by hand.
fn discover_projections(blob: &std::path::Path) -> anyhow::Result<Vec<(String, String)>> {
    // `chr(31)` (unit separator) keeps `table_name` and the projection list distinct in
    // the single-column `-list` output without colliding with `|` (list separator) or
    // `,` (projection delimiter).
    let sql = format!(
        "ATTACH '{}' AS src (READ_ONLY); \
         SELECT table_name || chr(31) || \
                string_agg('\"' || column_name || '\" AS \"' || lower(column_name) || '\"', ', ' ORDER BY column_index) \
         FROM duckdb_columns() \
         WHERE database_name = 'src' \
         GROUP BY table_name;",
        blob.display(),
    );
    let output = Command::new("duckdb")
        .arg("-noheader")
        .arg("-list")
        .arg("-c")
        .arg(&sql)
        .output()?;
    if !output.status.success() {
        bail!(
            "duckdb column discovery failed: stdout={:?} stderr={:?}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    let stdout = String::from_utf8(output.stdout)?;
    Ok(stdout
        .lines()
        .filter_map(|line| {
            line.split_once('\x1f')
                .map(|(t, p)| (t.to_owned(), p.to_owned()))
        })
        .collect())
}
