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
#![expect(clippy::print_stdout)]
#![allow(dead_code)]

use std::ops::BitAnd;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use bit_vec::BitVec;
use futures::StreamExt;
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
use vortex_array::builtins::ArrayBuiltins;
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
use vortex_layout::plan::EvalPlan;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::QueryPlan;
use vortex_layout::plan::exec::DecodeCache;
use vortex_layout::plan::exec::ExecGraph;
use vortex_layout::plan::exec::ExecOutput;
use vortex_layout::plan::exec::ExecState;
use vortex_layout::plan::lower;
use vortex_layout::plan::optimize;
use vortex_layout::scan::filter::FilterExpr;
use vortex_layout::scan::scan_builder::referenced_field_masks;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Segment reads the exec graph issued in the current run.
static READS: AtomicUsize = AtomicUsize::new(0);

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
const ROWS: usize = 1 << 18;
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
}

/// Writes the table through the default strategy, configured by `variant`, into memory and
/// opens it.
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
    SESSION.open_options().open_buffer(buf).expect("open")
}

/// The file variants.
///
/// `zoned` is what the default strategy writes. `plain` drops the zone maps, so a comparison with
/// an executor that does not prune has nothing to prune on either side and no statistics to
/// evaluate. The `plain-*` variants also shrink the blocks and coalesce nothing, so a split is
/// a few hundred or a few thousand rows: the work per split becomes small enough that what is
/// measured is each executor's own cost per split and per node, on the same layouts and plans.
const VARIANTS: [Variant; 4] = [
    Variant {
        name: "zoned",
        zone_maps: true,
        row_block: 8192,
        coalesce: true,
    },
    Variant {
        name: "plain",
        zone_maps: false,
        row_block: 8192,
        coalesce: true,
    },
    Variant {
        name: "plain-2k",
        zone_maps: false,
        row_block: 2048,
        coalesce: false,
    },
    Variant {
        name: "plain-256",
        zone_maps: false,
        row_block: 256,
        coalesce: false,
    },
];

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

/// A query lowered to plans: one plan per filter conjunct, the whole filter as one plan, and
/// the projection plan, with the splits the V1 scan would use for it.
struct Planned {
    name: &'static str,
    filter: Option<BoundExpression>,
    conjuncts: Vec<PlanRef>,
    whole_filter: Option<PlanRef>,
    projection: PlanRef,
    projection_expr: BoundExpression,
    expected: usize,
    splits: Vec<std::ops::Range<u64>>,
}

fn optimized(expression: BoundExpression, plan: &PlanRef) -> PlanRef {
    optimize(
        EvalPlan::try_new(expression, plan.clone())
            .expect("eval plan")
            .into_plan(),
    )
    .expect("optimize")
}

fn plan(file: &VortexFile, query: &Query) -> Planned {
    let plan = lower(file.footer().layout()).expect("lower");
    let conjuncts = query
        .filter
        .as_ref()
        .map(|filter| {
            FilterExpr::new(filter.clone())
                .conjuncts()
                .iter()
                .map(|conjunct| optimized(conjunct.clone(), &plan))
                .collect()
        })
        .unwrap_or_default();
    Planned {
        name: query.name,
        filter: query.filter.clone(),
        conjuncts,
        whole_filter: query.filter.clone().map(|f| optimized(f, &plan)),
        projection: optimized(query.projection.clone(), &plan),
        projection_expr: query.projection.clone(),
        expected: query.expected,
        splits: splits(file, query),
    }
}

impl Planned {
    /// Plans the query as one [`QueryPlan`] over `source`, the file's lowered layout, with the
    /// splits to run it over, as a V1 scan prepares on every scan over its shared reader tree.
    fn build(&self, file: &VortexFile, source: &PlanRef) -> (PlanRef, Vec<std::ops::Range<u64>>) {
        let query = QueryPlan::try_new(
            self.filter.clone(),
            self.projection_expr.clone(),
            source.clone(),
        )
        .expect("query plan")
        .into_plan();
        (
            query,
            layout_splits(file, &self.projection_expr, self.filter.as_ref()),
        )
    }
}

/// The natural splits of a query: the chunk boundaries of the columns it reads, as the V1 scan
/// computes them for the same filter and projection.
fn splits(file: &VortexFile, query: &Query) -> Vec<std::ops::Range<u64>> {
    layout_splits(file, &query.projection, query.filter.as_ref())
}

fn layout_splits(
    file: &VortexFile,
    projection: &BoundExpression,
    filter: Option<&BoundExpression>,
) -> Vec<std::ops::Range<u64>> {
    let reader = file.layout_reader().expect("reader");
    let masks = referenced_field_masks(projection, filter).expect("masks");
    SplitBy::Layout
        .splits(reader.as_ref(), &(0..file.row_count()), &masks)
        .expect("splits")
        .windows(2)
        .map(|w| w[0]..w[1])
        .collect()
}

/// A file variant, its layout lowered to a plan once, and its planned queries.
struct Fixture {
    name: &'static str,
    file: VortexFile,
    source: PlanRef,
    planned: Vec<Planned>,
}

static FILES: LazyLock<Vec<Fixture>> = LazyLock::new(|| {
    let t = table();
    let wanted = std::env::var("VARIANT").unwrap_or_else(|_| "plain-256".to_string());
    VARIANTS
        .iter()
        .filter(|variant| variant.name == wanted)
        .map(|variant| {
            let file = write_file(&t, *variant);
            let planned = queries(&t, file.dtype())
                .iter()
                .map(|query| plan(&file, query))
                .collect();
            let source = lower(file.footer().layout()).expect("lower");
            Fixture {
                name: variant.name,
                file,
                source,
                planned,
            }
        })
        .collect()
});

/// Every query on every file variant, as `query@variant`.
fn cases() -> Vec<&'static str> {
    FILES
        .iter()
        .flat_map(|fixture| {
            fixture
                .planned
                .iter()
                .map(move |q| &*Box::leak(format!("{}@{}", q.name, fixture.name).into_boxed_str()))
        })
        .collect()
}

fn lookup(case: &str) -> (&'static Fixture, &'static Planned) {
    let (name, variant) = case.split_once('@').expect("case");
    let fixture = FILES.iter().find(|f| f.name == variant).expect("variant");
    (
        fixture,
        fixture
            .planned
            .iter()
            .find(|q| q.name == name)
            .expect("query"),
    )
}

/// Drives one graph over `rows` of `plan` to completion, awaiting reads from `source` as the
/// graph publishes them, and hands each root array to `sink`. Segments decoded by earlier
/// graphs sharing `decoded` are reused.
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
                    READS.fetch_add(batch.len(), Ordering::Relaxed);
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

/// Runs `plan`, a boolean plan, over every row of `rows`, and returns what it produces: one lazy
/// array per piece, in row order.
fn predicate(
    source: &Arc<dyn SegmentSource>,
    plan: &PlanRef,
    rows: std::ops::Range<u64>,
    decoded: &DecodeCache,
) -> Vec<ArrayRef> {
    let len = (rows.end - rows.start) as usize;
    let mut pieces = Vec::new();
    drive(source, plan, rows, Mask::new_true(len), decoded, |array| {
        pieces.push(array)
    });
    pieces
}

/// The selected fraction at or above which a conjunct runs over the whole piece, as the V1
/// flat reader's threshold.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Evaluates one conjunct under `mask` exactly as the V1 readers do: the predicate is applied
/// to every row of each flat piece, then either filtered to the selected rows before it is
/// executed, when few are selected, or executed whole and intersected with the mask, and the
/// pieces' masks are concatenated as the V1 chunked reader concatenates its chunks'.
fn evaluate_conjunct(
    source: &Arc<dyn SegmentSource>,
    plan: &PlanRef,
    rows: std::ops::Range<u64>,
    mask: &Mask,
    decoded: &DecodeCache,
    ctx: &mut vortex_array::ExecutionCtx,
) -> Mask {
    let mut offset = 0;
    let mut masks = Vec::new();
    for piece in predicate(source, plan, rows, decoded) {
        let mask = mask.slice(offset..offset + piece.len());
        offset += piece.len();
        masks.push(if mask.density() < EXPR_EVAL_THRESHOLD {
            let conjunct = piece
                .filter(mask.clone())
                .expect("filter")
                .fill_null(false)
                .expect("fill_null")
                .execute::<Mask>(ctx)
                .expect("mask");
            mask.intersect_by_rank(&conjunct)
        } else {
            let conjunct = piece
                .fill_null(false)
                .expect("fill_null")
                .execute::<Mask>(ctx)
                .expect("mask");
            mask.bitand(&conjunct)
        });
    }
    assert_eq!(offset, mask.len(), "the pieces tile the split");
    if masks.len() == 1 {
        return masks.remove(0);
    }
    Mask::from_iter(masks)
}

/// How the exec graph evaluates a query's filter.
#[derive(Clone, Copy)]
enum Algorithm {
    /// The whole predicate as one plan over every row of the split, then one mask. Segments
    /// decoded once per query.
    Whole,
    /// The V1 scan's algorithm: conjuncts one at a time in the order V1's own `FilterExpr`
    /// chooses, each narrowing the mask as the V1 flat reader does. Segments decoded once per
    /// query.
    Conjuncts,
    /// `Conjuncts`, with segments decoded once per graph, as the V1 reader decodes them once
    /// per split and per expression that reads them.
    ConjunctsRedecode,
    /// One `Query` plan per split: conjuncts and projection as one graph.
    Query,
    /// `Query` with one decode cache for the run, advanced a generation per split, so a segment
    /// spanning consecutive splits is decoded once and dropped once the splits have passed it.
    /// The V1 scan decodes such a segment once per split.
    QueryStreaming,
}

fn run(
    file: &VortexFile,
    query: &Planned,
    algorithm: Algorithm,
    source: &PlanRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> usize {
    let segments = file.segment_source();
    let shared = DecodeCache::default();
    let cache = || match algorithm {
        Algorithm::Whole | Algorithm::Conjuncts => shared.clone(),
        Algorithm::ConjunctsRedecode | Algorithm::Query => DecodeCache::default(),
        Algorithm::QueryStreaming => shared.clone(),
    };
    // A fresh scheduler per run, as every V1 scan starts with no selectivity history.
    let scheduler = query.filter.clone().map(FilterExpr::new);
    let mut rows = 0;
    if matches!(algorithm, Algorithm::Query | Algorithm::QueryStreaming) {
        let (plan, splits) = query.build(file, source);
        for split in splits {
            let len = (split.end - split.start) as usize;
            let cache = cache();
            drive(
                &segments,
                &plan,
                split,
                Mask::new_true(len),
                &cache,
                |array| rows += array.len(),
            );
            cache.next_generation();
        }
        return rows;
    }
    for split in &query.splits {
        let len = (split.end - split.start) as usize;
        let mut mask = Mask::new_true(len);
        match algorithm {
            Algorithm::Query | Algorithm::QueryStreaming => unreachable!("handled above"),
            Algorithm::Whole => {
                if let Some(filter) = &query.whole_filter {
                    let pieces = predicate(&segments, filter, split.clone(), &cache());
                    mask = ChunkedArray::try_new(pieces, filter.dtype().clone())
                        .expect("predicate")
                        .into_array()
                        .execute::<Mask>(ctx)
                        .expect("mask");
                }
            }
            Algorithm::Conjuncts | Algorithm::ConjunctsRedecode => {
                if let Some(scheduler) = &scheduler {
                    let mut remaining = BitVec::from_elem(query.conjuncts.len(), true);
                    while let Some(idx) = scheduler.next_conjunct(&remaining) {
                        remaining.set(idx, false);
                        if mask.all_false() {
                            break;
                        }
                        let input = mask.true_count();
                        mask = evaluate_conjunct(
                            &segments,
                            &query.conjuncts[idx],
                            split.clone(),
                            &mask,
                            &cache(),
                            ctx,
                        );
                        scheduler.report_selectivity(idx, mask.true_count() as f64 / input as f64);
                    }
                }
            }
        }
        if mask.all_false() {
            continue;
        }
        drive(
            &segments,
            &query.projection,
            split.clone(),
            mask,
            &cache(),
            |array| rows += array.len(),
        );
    }
    rows
}

#[inline(never)]
fn run_v1(file: &VortexFile, query: &Planned) -> usize {
    RUNTIME.block_on(async {
        let mut stream = file
            .scan()
            .expect("scan")
            .with_some_filter(query.filter.clone())
            .with_projection(query.projection_expr.clone())
            .with_split_by(SplitBy::Layout)
            .into_array_stream()
            .expect("stream");
        let mut rows = 0;
        while let Some(batch) = stream.next().await {
            rows += batch.expect("batch").len();
        }
        rows
    })
}

#[inline(never)]
fn run_exec(file: &VortexFile, query: &Planned, algorithm: Algorithm, source: &PlanRef) -> usize {
    let mut ctx = SESSION.create_execution_ctx();
    run(file, query, algorithm, source, &mut ctx)
}

fn main() {
    let only = std::env::var("ONLY").unwrap_or_else(|_| "both".to_string());
    let wanted = std::env::var("QUERY").unwrap_or_else(|_| "project".to_string());
    let algorithm = match std::env::var("ALGO").as_deref() {
        Ok("whole") => Algorithm::Whole,
        Ok("redecode") => Algorithm::ConjunctsRedecode,
        Ok("query") => Algorithm::Query,
        Ok("stream") => Algorithm::QueryStreaming,
        _ => Algorithm::Conjuncts,
    };
    let Fixture {
        name: variant,
        file,
        source,
        planned,
    } = &FILES[0];
    let query = planned.iter().find(|q| q.name == wanted).expect("query");
    println!("{variant} {wanted} splits={}", query.splits.len());
    {
        let start = std::time::Instant::now();
        let (built, splits) = query.build(file, source);
        println!(
            "plan build (lower+optimize+splits) {}ms, {} splits",
            start.elapsed().as_secs_f64() * 1e3,
            splits.len()
        );
        if std::env::var("PLAN").is_ok() {
            println!("{}", built.display_tree());
        }
    }
    for _ in 0..2 {
        if only != "v1" {
            let start = std::time::Instant::now();
            READS.store(0, Ordering::Relaxed);
            let rows = run_exec(file, query, algorithm, source);
            println!("exec reads={}", READS.load(Ordering::Relaxed));
            println!("exec rows={rows} {}ms", start.elapsed().as_secs_f64() * 1e3);
        }
        if only != "exec" {
            let start = std::time::Instant::now();
            let rows = run_v1(file, query);
            println!("v1   rows={rows} {}ms", start.elapsed().as_secs_f64() * 1e3);
        }
    }
}
