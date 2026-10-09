// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for running a plan as pipelines.
//!
//! A struct of chunked columns is written to in-memory segments once, as setup. Each iteration
//! scans the lowered plan, answers its reads synchronously from memory, and drains the output.
//! That measures everything a scan does per split (compiling pipelines, scheduling, port
//! hand-offs, decoding, struct assembly) with no IO latency in the way.
//!
//! The shapes vary the two things that drive a scan's cost: how many pipelines it builds
//! (columns times chunks) and how much of the data is selected. A second group runs the same
//! shapes over hand-built sources that decode nothing, so what is left is the scheduling.

#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::serde::SerializeOptions;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_layout::LayoutRef;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::plan::ConcatPlan;
use vortex_layout::plan::PackPlan;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::QueryPlan;
use vortex_layout::plan::lower;
use vortex_layout::plan::pipeline::Scan;
use vortex_layout::plan::pipeline::Split;
use vortex_layout::plan::pipeline::Turn;
use vortex_layout::plan::pipeline::synthetic::RowSource;
use vortex_layout::plan::pipeline::synthetic::empty_segment;
use vortex_layout::plan::pipeline::synthetic::row_dtype;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::array_session().with::<LayoutSession>());

/// Rows in every column.
const ROWS: usize = 1 << 16;

/// In-memory segments, standing in for the IO service.
#[derive(Default)]
struct Store {
    segments: Vec<BufferHandle>,
}

impl Store {
    fn flat(&mut self, array: &ArrayRef) -> LayoutRef {
        let ctx = ArrayContext::empty();
        let buffers = array
            .serialize(
                &ctx,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )
            .expect("serialize");
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let segment_id = SegmentId::from(self.segments.len() as u32);
        self.segments.push(BufferHandle::new_host(bytes.freeze()));
        FlatLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.to_ids()),
        )
        .into_layout()
    }

    fn chunked(&mut self, array: &ArrayRef, chunks: usize) -> LayoutRef {
        let size = array.len() / chunks;
        let children = (0..chunks)
            .map(|i| self.flat(&array.slice(i * size..(i + 1) * size).expect("slice")))
            .collect();
        ChunkedLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            layout_children(children),
        )
        .into_layout()
    }
}

/// A struct of `columns` i64 columns, each in `chunks` chunks, lowered to a plan.
fn fixture(columns: usize, chunks: usize) -> (Arc<Store>, PlanRef) {
    let mut store = Store::default();
    let mut fields = Vec::with_capacity(columns);
    let mut layouts = Vec::with_capacity(columns);
    for column in 0..columns {
        let values =
            PrimitiveArray::from_iter((0..ROWS as i64).map(|row| row * column as i64)).into_array();
        layouts.push(store.chunked(&values, chunks));
        fields.push((format!("c{column}"), values));
    }
    let dtype = StructArray::from_fields(
        &fields
            .iter()
            .map(|(name, values)| (name.as_str(), values.clone()))
            .collect::<Vec<_>>(),
    )
    .expect("struct")
    .dtype()
    .clone();
    let layout = StructLayout::new(ROWS as u64, dtype, layouts).into_layout();
    let plan = lower(&layout).expect("lower");
    (Arc::new(store), plan)
}

/// Scans `splits` of `plan` to completion, answering each read with `answer` at once, and
/// returns the rows produced.
fn drive(plan: &PlanRef, splits: Vec<Split>, answer: impl Fn(usize) -> BufferHandle) -> usize {
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), splits).expect("scan");
    let mut rows = 0;
    loop {
        match scan.step().expect("step") {
            Turn::Read(read) => scan
                .deliver(read.id, answer(*read.segment_id as usize))
                .expect("deliver"),
            Turn::Output(_, array) => rows += array.len(),
            Turn::Waiting => unreachable!("every read is answered as it is asked for"),
            Turn::Done => return rows,
        }
    }
}

/// Which rows a split asks for.
#[derive(Clone, Copy, Debug)]
enum Selection {
    /// Every row: the projection-only case.
    All,
    /// One row in sixteen: a selective filter's projection.
    Sparse,
    /// Rows in the middle chunk only: most chunks are never read.
    OneChunk,
}

impl Selection {
    fn mask(self, chunks: usize) -> Mask {
        match self {
            Self::All => Mask::new_true(ROWS),
            Self::Sparse => Mask::from_iter((0..ROWS).map(|row| row % 16 == 0)),
            Self::OneChunk => {
                let size = ROWS / chunks;
                let start = (chunks / 2) * size;
                Mask::from_indices(ROWS, start..start + size)
            }
        }
    }
}

/// Columns and chunks per column, so nodes per graph is roughly their product.
const SHAPES: &[(usize, usize)] = &[(1, 1), (8, 1), (8, 16), (64, 16)];

#[divan::bench(args = SHAPES, consts = [0, 1, 2])]
fn split<const SEL: usize>(bencher: Bencher, shape: (usize, usize)) {
    let (columns, chunks) = shape;
    let selection = [Selection::All, Selection::Sparse, Selection::OneChunk][SEL];
    let (store, plan) = fixture(columns, chunks);
    let mask = selection.mask(chunks);
    let selected = mask.true_count();
    bencher
        .counter(ItemsCount::new(selected * columns))
        .bench_local(|| {
            let splits = vec![Split {
                rows: 0..ROWS as u64,
                mask: mask.clone(),
            }];
            let rows = drive(&plan, splits, |segment| store.segments[segment].clone());
            assert_eq!(rows, selected);
        });
}

/// Rows per hand-built source. Small, so the cost per node, not per row, is what is measured.
const SOURCE_ROWS: u64 = 16;

/// The runtime's own overhead, isolated: a tree of hand-built sources that emit tiny arrays
/// without decoding anything, under the real Concat and Pack operators. With IO, every source
/// requests a read that is answered from memory as soon as it is asked for, so read routing
/// and delivery wakeups are measured too. Items are sources, so the rate is pipelines run per
/// second.
#[divan::bench(args = SHAPES, consts = [false, true])]
fn scheduling<const IO: bool>(bencher: Bencher, shape: (usize, usize)) {
    let (columns, chunks) = shape;
    let chunk_rows = SOURCE_ROWS;
    let rows = chunk_rows as usize * chunks;
    let fields = StructFields::from_iter((0..columns).map(|i| (format!("c{i}"), row_dtype())));
    let plan = PackPlan::try_new(
        fields,
        Nullability::NonNullable,
        rows as u64,
        (0..columns)
            .map(|column| {
                let sources = (0..chunks)
                    .map(|chunk| {
                        let segment = IO.then(|| SegmentId::from((column * chunks + chunk) as u32));
                        RowSource::plan(chunk_rows, Vec::new(), segment, false)
                    })
                    .collect();
                ConcatPlan::try_new(row_dtype(), sources)
                    .expect("concat")
                    .into_plan()
            })
            .collect(),
        None,
    )
    .expect("pack")
    .into_plan();
    let segment = empty_segment();
    bencher
        .counter(ItemsCount::new(columns * chunks))
        .bench_local(|| {
            let produced = drive(&plan, vec![Split::all(0..rows as u64)], |_| segment.clone());
            assert_eq!(produced, rows);
        });
}

/// Column `column`'s value at `row` in the query fixture, as `scan_v1.rs` defines it.
fn value(column: usize, row: usize) -> i64 {
    let x = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (column as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    ((x >> 32) % 1000) as i64
}

/// Rows, chunks, and columns of the query fixture, as `scan_v1.rs` defines them.
const QUERY_ROWS: usize = 1 << 20;
const QUERY_CHUNKS: usize = 16;
const QUERY_COLUMNS: usize = 8;

/// The query fixture: the same struct of chunked columns `scan_v1.rs` builds, lowered to a
/// query of the filter over two columns and the projection of three, with the rows the filter
/// keeps.
fn query_fixture() -> (Arc<Store>, PlanRef, usize) {
    let mut store = Store::default();
    let mut layouts = Vec::with_capacity(QUERY_COLUMNS);
    for column in 0..QUERY_COLUMNS {
        let values =
            PrimitiveArray::from_iter((0..QUERY_ROWS).map(|row| value(column, row))).into_array();
        layouts.push(store.chunked(&values, QUERY_CHUNKS));
    }
    let dtype = DType::Struct(
        StructFields::from_iter((0..QUERY_COLUMNS).map(|i| {
            (
                format!("c{i}"),
                DType::Primitive(PType::I64, Nullability::NonNullable),
            )
        })),
        Nullability::NonNullable,
    );
    let layout = StructLayout::new(QUERY_ROWS as u64, dtype.clone(), layouts).into_layout();
    let plan = lower(&layout).expect("lower");
    let filter = and(
        lt(get_item("c0", root()), lit(200_i64)),
        gt(get_item("c1", root()), lit(499_i64)),
    )
    .bind(&dtype)
    .expect("bind filter");
    let projection = select(FieldNames::from(["c2", "c3", "c4"]), root())
        .bind(&dtype)
        .expect("bind projection");
    let query = QueryPlan::try_new(Some(filter), projection, plan)
        .expect("query")
        .into_plan();
    let expected = (0..QUERY_ROWS)
        .filter(|&row| value(0, row) < 200 && value(1, row) > 499)
        .count();
    (Arc::new(store), query, expected)
}

/// The filter-and-project query of `scan_v1.rs` as pipelines: one split per chunk, each
/// running the filter's conjuncts as stages under the rows the earlier ones kept, then the
/// projection under the rows they all kept, as the V1 scan does.
#[divan::bench]
fn query(bencher: Bencher) {
    let (store, query, expected) = query_fixture();
    let split_rows = (QUERY_ROWS / QUERY_CHUNKS) as u64;
    let splits: Vec<Split> = (0..QUERY_CHUNKS as u64)
        .map(|split| Split::all(split * split_rows..(split + 1) * split_rows))
        .collect();
    bencher.counter(ItemsCount::new(expected)).bench_local(|| {
        let rows = drive(&query, splits.clone(), |segment| {
            store.segments[segment].clone()
        });
        assert_eq!(rows, expected);
    });
}
