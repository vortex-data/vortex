// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exec graph (`vortex_layout::plan::exec`) on queries over a lineitem-like file, for
//! comparison with the same queries on the V1 scan in `query_v1.rs`.
//!
//! Both benchmarks write the same table through the default write strategy (BtrBlocks
//! compression, dictionary encoding, zoned statistics, 8K row blocks coalesced to 1MB segments)
//! into an in-memory buffer, open it as a `VortexFile`, and run the same filter and projection
//! over the file's natural splits. Only the executor differs. The table and the queries are
//! defined by the same code in both files.
//!
//! Here the file's layout is lowered to a plan, the filter and the projection are each pushed
//! into it by the plan optimizer, and every split runs the filter plan over all its rows to a
//! mask, then the projection plan over the rows the mask keeps, as the V1 scan does. Reads go
//! through the file's own segment source and are awaited as the graph publishes them. Zone
//! statistics are not used: the plan reads a zoned layout's data child whole.

#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use futures::future::join_all;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
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
use vortex_buffer::ByteBufferMut;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::plan::EvalPlan;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::exec::DecodeCache;
use vortex_layout::plan::exec::ExecGraph;
use vortex_layout::plan::exec::ExecOutput;
use vortex_layout::plan::exec::ExecState;
use vortex_layout::plan::lower;
use vortex_layout::plan::optimize;
use vortex_layout::scan::scan_builder::referenced_field_masks;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&FILE);
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

/// Writes the table through the default strategy into memory and opens it.
fn write_file(t: &Table) -> VortexFile {
    let chunks = (0..ROWS / CHUNK).map(|i| columns(t, i * CHUNK..(i + 1) * CHUNK));
    let array = ChunkedArray::from_iter(chunks).into_array();
    let strategy = WriteStrategyBuilder::from_session(&SESSION).build();
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
    SESSION.open_options().open_buffer(buf).expect("open")
}

/// A query: a filter, a projection, and the rows it keeps, from the plain columns.
struct Query {
    name: &'static str,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    expected: usize,
}

fn col(name: &str) -> vortex_array::expr::Expression {
    get_item(name, root())
}

/// The queries, bound to the file's dtype. See `query_v1.rs`.
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

/// A query lowered to plans: the filter plan, if any, and the projection plan.
struct Planned {
    name: &'static str,
    filter: Option<PlanRef>,
    projection: PlanRef,
    expected: usize,
    splits: Vec<std::ops::Range<u64>>,
}

fn plan(file: &VortexFile, query: &Query) -> Planned {
    let plan = lower(file.footer().layout()).expect("lower");
    let filter = query.filter.clone().map(|filter| {
        optimize(
            EvalPlan::try_new(filter, plan.clone())
                .expect("filter plan")
                .into_plan(),
        )
        .expect("optimize filter")
    });
    let projection = optimize(
        EvalPlan::try_new(query.projection.clone(), plan)
            .expect("projection plan")
            .into_plan(),
    )
    .expect("optimize projection");
    Planned {
        name: query.name,
        filter,
        projection,
        expected: query.expected,
        splits: splits(file, query),
    }
}

/// The natural splits of a query: the chunk boundaries of the columns it reads, as the V1 scan
/// computes them for the same filter and projection.
fn splits(file: &VortexFile, query: &Query) -> Vec<std::ops::Range<u64>> {
    let reader = file.layout_reader().expect("reader");
    let masks = referenced_field_masks(&query.projection, query.filter.as_ref()).expect("masks");
    SplitBy::Layout
        .splits(reader.as_ref(), &(0..file.row_count()), &masks)
        .expect("splits")
        .windows(2)
        .map(|w| w[0]..w[1])
        .collect()
}

static FILE: LazyLock<(VortexFile, Vec<Planned>)> = LazyLock::new(|| {
    let t = table();
    let file = write_file(&t);
    let planned = queries(&t, file.dtype())
        .iter()
        .map(|query| plan(&file, query))
        .collect();
    (file, planned)
});

fn query_names() -> Vec<&'static str> {
    FILE.1.iter().map(|q| q.name).collect()
}

/// Drives one graph over `rows` of `plan` to completion, awaiting reads from `source` as the
/// graph publishes them, and hands each root array to `sink`. Segments decoded by earlier
/// graphs sharing `decoded` are reused, as a V1 reader reuses them across its splits.
fn drive(
    source: &Arc<dyn SegmentSource>,
    plan: &PlanRef,
    rows: std::ops::Range<u64>,
    mask: Mask,
    decoded: &DecodeCache,
    mut sink: impl FnMut(ArrayRef),
) {
    let mut graph =
        ExecGraph::try_new(SESSION.clone(), plan, rows, mask, 0, decoded.clone()).expect("graph");
    loop {
        match graph.state() {
            ExecState::Done => return,
            ExecState::NeedsCompute => match graph.compute().expect("compute") {
                ExecOutput::Piece(array) => sink(array),
                ExecOutput::NeedsIO(batch) => {
                    let reads = batch
                        .iter()
                        .map(|request| source.request(request.segment_id))
                        .collect::<Vec<_>>();
                    let results = RUNTIME.block_on(join_all(reads));
                    for (request, result) in batch.into_iter().zip(results) {
                        graph
                            .set_io_result(request.id, result.expect("read"))
                            .expect("deliver");
                    }
                }
                ExecOutput::Yield => {}
            },
            ExecState::Waiting => unreachable!("every read is answered as it is published"),
        }
    }
}

/// One exec graph run of the query over the file's natural splits: per split, the filter plan
/// to a mask, then the projection plan under it.
#[divan::bench(args = query_names())]
fn exec(bencher: Bencher, name: &str) {
    let (file, planned) = &*FILE;
    let query = planned.iter().find(|q| q.name == name).expect("query");
    let splits = &query.splits;
    let source = file.segment_source();
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(query.expected))
        .bench_local(|| {
            let decoded = DecodeCache::default();
            let mut rows = 0;
            for split in splits {
                let len = (split.end - split.start) as usize;
                let mask = match &query.filter {
                    None => Mask::new_true(len),
                    Some(filter) => {
                        let mut predicate = Vec::new();
                        drive(
                            &source,
                            filter,
                            split.clone(),
                            Mask::new_true(len),
                            &decoded,
                            |array| predicate.push(array),
                        );
                        ChunkedArray::try_new(predicate, filter.dtype().clone())
                            .expect("predicate")
                            .into_array()
                            .execute::<Mask>(&mut ctx)
                            .expect("mask")
                    }
                };
                if mask.all_false() {
                    continue;
                }
                drive(
                    &source,
                    &query.projection,
                    split.clone(),
                    mask,
                    &decoded,
                    |array| rows += array.len(),
                );
            }
            assert_eq!(rows, query.expected);
        });
}
