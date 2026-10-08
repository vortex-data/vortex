// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The V1 scan (`ScanBuilder` over a `LayoutReader`) on queries over a lineitem-like file, for
//! comparison with the same queries on the exec graph in `query_exec.rs`.
//!
//! Both benchmarks write the same table through the default write strategy (BtrBlocks
//! compression, dictionary encoding, zoned statistics, 8K row blocks coalesced to 1MB segments)
//! into an in-memory buffer, open it as a `VortexFile`, and run the same filter and projection
//! over the file's natural splits. Only the executor differs. The table and the queries are
//! defined by the same code in both files.

#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use futures::StreamExt;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_btrblocks::CompressionSession;
use vortex_buffer::ByteBufferMut;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&FILES);
    divan::main();
}

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<CompressionSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    session
});

/// Rows in the table.
const ROWS: usize = 1 << 21;
/// Rows per chunk of the written stream. The writer repartitions these into its own blocks.
const CHUNK: usize = 1 << 16;

/// Days per year of ship dates, which are clustered by row order with some jitter, so zone
/// statistics can prune a date range.
const DAYS_PER_YEAR: i32 = 365;
const YEARS: i32 = 7;

/// A lineitem-like table, as plain columns.
struct Table {
    orderkey: Vec<i64>,
    partkey: Vec<i64>,
    quantity: Vec<i64>,
    extendedprice: Vec<f64>,
    discount: Vec<f64>,
    shipdate: Vec<i32>,
    returnflag: Vec<&'static str>,
    shipmode: Vec<&'static str>,
    comment: Vec<String>,
}

const SHIPMODES: [&str; 7] = ["AIR", "AIR REG", "FOB", "MAIL", "RAIL", "SHIP", "TRUCK"];
const RETURNFLAGS: [&str; 3] = ["A", "N", "R"];

fn table() -> Table {
    let mut rng = StdRng::seed_from_u64(42);
    let mut t = Table {
        orderkey: Vec::with_capacity(ROWS),
        partkey: Vec::with_capacity(ROWS),
        quantity: Vec::with_capacity(ROWS),
        extendedprice: Vec::with_capacity(ROWS),
        discount: Vec::with_capacity(ROWS),
        shipdate: Vec::with_capacity(ROWS),
        returnflag: Vec::with_capacity(ROWS),
        shipmode: Vec::with_capacity(ROWS),
        comment: Vec::with_capacity(ROWS),
    };
    let total_days = DAYS_PER_YEAR * YEARS;
    for row in 0..ROWS {
        t.orderkey.push((row / 4) as i64 * 2 + 1);
        t.partkey.push(rng.random_range(1..200_000));
        let quantity = rng.random_range(1..=50);
        t.quantity.push(quantity);
        t.extendedprice
            .push(quantity as f64 * rng.random_range(900.0..100_000.0_f64).round() / 100.0);
        t.discount.push(rng.random_range(0..=10) as f64 / 100.0);
        let base = (row as i64 * total_days as i64 / ROWS as i64) as i32;
        t.shipdate
            .push((base + rng.random_range(-45..=45)).clamp(0, total_days - 1));
        t.returnflag.push(RETURNFLAGS[rng.random_range(0..3)]);
        t.shipmode.push(SHIPMODES[rng.random_range(0..7)]);
        t.comment.push(format!(
            "{:016x}{:08x}",
            rng.random::<u64>(),
            rng.random::<u32>()
        ));
    }
    t
}

fn columns(t: &Table, range: std::ops::Range<usize>) -> ArrayRef {
    StructArray::from_fields(&[
        (
            "l_orderkey",
            PrimitiveArray::from_iter(t.orderkey[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_partkey",
            PrimitiveArray::from_iter(t.partkey[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_quantity",
            PrimitiveArray::from_iter(t.quantity[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_extendedprice",
            PrimitiveArray::from_iter(t.extendedprice[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_discount",
            PrimitiveArray::from_iter(t.discount[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_shipdate",
            PrimitiveArray::from_iter(t.shipdate[range.clone()].iter().copied()).into_array(),
        ),
        (
            "l_returnflag",
            VarBinViewArray::from_iter_str(t.returnflag[range.clone()].iter()).into_array(),
        ),
        (
            "l_shipmode",
            VarBinViewArray::from_iter_str(t.shipmode[range.clone()].iter()).into_array(),
        ),
        (
            "l_comment",
            VarBinViewArray::from_iter_str(t.comment[range].iter()).into_array(),
        ),
    ])
    .expect("struct")
    .into_array()
}

/// How a file variant is written.
#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    zone_maps: bool,
    /// Rows per block. The default strategy uses 8192 and coalesces blocks into 1MB segments;
    /// the small-block variants coalesce nothing, so a split is one block of every column.
    row_block: usize,
    coalesce: bool,
    /// Whether the file is written to disk and read through the file IO path, rather than kept
    /// in memory and read by slicing a buffer.
    on_disk: bool,
}

/// Writes the table through the default strategy, configured by `variant`, into memory or to
/// a file on disk, and opens it.
fn write_file(t: &Table, variant: Variant) -> VortexFile {
    let chunks = (0..ROWS / CHUNK).map(|i| columns(t, i * CHUNK..(i + 1) * CHUNK));
    let array = ChunkedArray::from_iter(chunks).into_array();
    let mut strategy = WriteStrategyBuilder::from_session(&SESSION)
        .with_zone_maps(variant.zone_maps)
        .with_row_block_size(variant.row_block);
    if !variant.coalesce {
        strategy = strategy.with_data_block_target_bytes(None);
    }
    let strategy = strategy.build();
    let mut buf = ByteBufferMut::empty();
    RUNTIME
        .block_on(
            SESSION
                .write_options()
                // A bare session enables no edition, which would reject the zoned statistics
                // aggregates. The layout strategy is still the default one.
                .disable_editions()
                .with_strategy(strategy)
                .write(&mut buf, array.to_array_stream()),
        )
        .expect("write");
    if variant.on_disk {
        // The directory lives as long as the file is benchmarked.
        let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
        let path = dir.path().join(format!("{}.vortex", variant.name));
        std::fs::write(&path, buf.as_ref()).expect("write file");
        return RUNTIME
            .block_on(SESSION.open_options().open_path(&path))
            .expect("open path");
    }
    SESSION.open_options().open_buffer(buf).expect("open")
}

/// The file variants.
///
/// `zoned` is what the default strategy writes. `plain` drops the zone maps, so a comparison with
/// an executor that does not prune has nothing to prune on either side and no statistics to
/// evaluate. The `plain-*` variants also shrink the blocks and coalesce nothing, so a split is
/// a few hundred or a few thousand rows: the work per split becomes small enough that what is
/// measured is each executor's own cost per split and per node, on the same layouts and plans.
const VARIANTS: [Variant; 6] = [
    Variant {
        name: "zoned",
        zone_maps: true,
        row_block: 8192,
        coalesce: true,
        on_disk: false,
    },
    Variant {
        name: "plain",
        zone_maps: false,
        row_block: 8192,
        coalesce: true,
        on_disk: false,
    },
    Variant {
        name: "plain-2k",
        zone_maps: false,
        row_block: 2048,
        coalesce: false,
        on_disk: false,
    },
    Variant {
        name: "plain-256",
        zone_maps: false,
        row_block: 256,
        coalesce: false,
        on_disk: false,
    },
    // The default layout on disk: every segment is read through the file IO path, so what is
    // measured includes issuing, awaiting and copying real reads (served from the page cache).
    Variant {
        name: "zoned-disk",
        zone_maps: true,
        row_block: 8192,
        coalesce: true,
        on_disk: true,
    },
    Variant {
        name: "plain-disk",
        zone_maps: false,
        row_block: 8192,
        coalesce: true,
        on_disk: true,
    },
];

/// A query: a filter, a projection, and the rows it keeps, from the plain columns.
struct Query {
    name: &'static str,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    expected: usize,
}

impl std::fmt::Debug for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

fn col(name: &str) -> vortex_array::expr::Expression {
    get_item(name, root())
}

/// The queries, bound to the file's dtype.
///
/// - `q6`: a date range, a discount range, and a quantity bound, projecting price and discount.
///   The date range is a year of seven, and ship dates are clustered by row, so zone statistics
///   can prune most blocks of the filter columns.
/// - `dict`: a dictionary-encoded string equality and a quantity bound, projecting an integer,
///   a float, and a high-cardinality string. Nothing is prunable.
/// - `project`: no filter, projecting an integer, an integer, and a dictionary string.
fn queries(t: &Table, dtype: &DType) -> Vec<Query> {
    let year = 3 * DAYS_PER_YEAR..4 * DAYS_PER_YEAR;
    let q6 = and(
        and(
            and(
                gt_eq(col("l_shipdate"), lit(year.start)),
                lt(col("l_shipdate"), lit(year.end)),
            ),
            and(
                gt_eq(col("l_discount"), lit(0.05_f64)),
                lt_eq(col("l_discount"), lit(0.07_f64)),
            ),
        ),
        lt(col("l_quantity"), lit(24_i64)),
    );
    let q6_rows = (0..ROWS)
        .filter(|&r| {
            year.contains(&t.shipdate[r])
                && (0.05..=0.07).contains(&t.discount[r])
                && t.quantity[r] < 24
        })
        .count();
    // The same shape as q6 with the date range replaced by a part key range of the same
    // selectivity. Part keys are uniform, so zone statistics cannot prune it.
    let keys = 1..200_000 / YEARS as i64;
    let q6_flat = and(
        and(
            and(
                gt_eq(col("l_partkey"), lit(keys.start)),
                lt(col("l_partkey"), lit(keys.end)),
            ),
            and(
                gt_eq(col("l_discount"), lit(0.05_f64)),
                lt_eq(col("l_discount"), lit(0.07_f64)),
            ),
        ),
        lt(col("l_quantity"), lit(24_i64)),
    );
    let q6_flat_rows = (0..ROWS)
        .filter(|&r| {
            keys.contains(&t.partkey[r])
                && (0.05..=0.07).contains(&t.discount[r])
                && t.quantity[r] < 24
        })
        .count();
    let dict = and(
        eq(col("l_shipmode"), lit("AIR")),
        gt(col("l_quantity"), lit(25_i64)),
    );
    let dict_rows = (0..ROWS)
        .filter(|&r| t.shipmode[r] == "AIR" && t.quantity[r] > 25)
        .count();
    vec![
        Query {
            name: "q6",
            filter: Some(q6.bind(dtype).expect("bind")),
            projection: select(FieldNames::from(["l_extendedprice", "l_discount"]), root())
                .bind(dtype)
                .expect("bind"),
            expected: q6_rows,
        },
        Query {
            name: "q6_flat",
            filter: Some(q6_flat.bind(dtype).expect("bind")),
            projection: select(FieldNames::from(["l_extendedprice", "l_discount"]), root())
                .bind(dtype)
                .expect("bind"),
            expected: q6_flat_rows,
        },
        Query {
            name: "dict",
            filter: Some(dict.bind(dtype).expect("bind")),
            projection: select(
                FieldNames::from(["l_orderkey", "l_extendedprice", "l_comment"]),
                root(),
            )
            .bind(dtype)
            .expect("bind"),
            expected: dict_rows,
        },
        Query {
            name: "project",
            filter: None,
            projection: select(
                FieldNames::from(["l_orderkey", "l_quantity", "l_shipmode"]),
                root(),
            )
            .bind(dtype)
            .expect("bind"),
            expected: ROWS,
        },
    ]
}

static FILES: LazyLock<Vec<(&'static str, VortexFile, Vec<Query>)>> = LazyLock::new(|| {
    let t = table();
    VARIANTS
        .iter()
        .map(|variant| {
            let file = write_file(&t, *variant);
            let queries = queries(&t, file.dtype());
            (variant.name, file, queries)
        })
        .collect()
});

/// Every query on every file variant, as `query@variant`.
fn cases() -> Vec<&'static str> {
    FILES
        .iter()
        .flat_map(|(variant, _, queries)| {
            queries
                .iter()
                .map(move |q| &*Box::leak(format!("{}@{variant}", q.name).into_boxed_str()))
        })
        .collect()
}

fn lookup(case: &str) -> (&'static VortexFile, &'static Query) {
    let (name, variant) = case.split_once('@').expect("case");
    let (_, file, queries) = FILES.iter().find(|(v, ..)| *v == variant).expect("variant");
    (
        file,
        queries.iter().find(|q| q.name == name).expect("query"),
    )
}

/// One V1 scan of the query over the file's natural splits, drained on the session's runtime.
#[divan::bench(args = cases())]
fn v1(bencher: Bencher, case: &str) {
    let (file, query) = lookup(case);
    bencher
        .counter(ItemsCount::new(query.expected))
        .bench_local(|| {
            let rows = RUNTIME.block_on(async {
                let mut stream = file
                    .scan()
                    .expect("scan")
                    .with_some_filter(query.filter.clone())
                    .with_projection(query.projection.clone())
                    .with_split_by(SplitBy::Layout)
                    .into_array_stream()
                    .expect("stream");
                let mut rows = 0;
                while let Some(batch) = stream.next().await {
                    rows += batch.expect("batch").len();
                }
                rows
            });
            assert_eq!(rows, query.expected);
        });
}
