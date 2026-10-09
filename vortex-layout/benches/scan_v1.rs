// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The V1 scan (`ScanBuilder` over a `LayoutReader`) on a filter-and-project query, for
//! comparison with the same query on the pipeline executor in `pipeline.rs`.
//!
//! Both benchmarks build the same struct of chunked `i64` columns in in-memory segments, answer
//! every read from memory, split the scan at the chunk boundaries, evaluate the same predicate
//! over two columns, and project three others. Only the executor differs.

#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use futures::FutureExt;
use futures::future;
use mimalloc::MiMalloc;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
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
use vortex_error::vortex_err;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutRef;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
});

/// Rows in every column.
const ROWS: usize = 1 << 20;
/// Chunks per column, and therefore splits per scan.
const CHUNKS: usize = 16;
/// Columns in the struct.
const COLUMNS: usize = 8;

/// In-memory segments answering every read at once.
#[derive(Default)]
struct Store {
    segments: Vec<BufferHandle>,
}

impl SegmentSource for Store {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let segment = self
            .segments
            .get(*id as usize)
            .cloned()
            .ok_or_else(|| vortex_err!("Segment {id} not found"));
        future::ready(segment).boxed()
    }
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

    fn chunked(&mut self, array: &ArrayRef) -> LayoutRef {
        let size = array.len() / CHUNKS;
        let children = (0..CHUNKS)
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

/// Column `column`'s value at `row`: a cheap hash spread over `0..1000`, so predicates on it
/// have predictable selectivity and the columns are uncorrelated.
fn value(column: usize, row: usize) -> i64 {
    let x = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (column as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    ((x >> 32) % 1000) as i64
}

/// The query's filter and projection, bound to the struct's dtype.
///
/// `c0 < 200 and c1 > 499` keeps about a tenth of the rows; the projection reads three other
/// columns, so the filter columns are read for every row and the projected ones for the kept
/// rows only.
fn query(dtype: &DType) -> (BoundExpression, BoundExpression) {
    let filter = and(
        lt(get_item("c0", root()), lit(200_i64)),
        gt(get_item("c1", root()), lit(499_i64)),
    )
    .bind(dtype)
    .expect("bind filter");
    let projection = select(FieldNames::from(["c2", "c3", "c4"]), root())
        .bind(dtype)
        .expect("bind projection");
    (filter, projection)
}

/// The rows the query keeps, from the data itself.
fn expected_rows() -> usize {
    (0..ROWS)
        .filter(|&row| value(0, row) < 200 && value(1, row) > 499)
        .count()
}

/// The struct layout over in-memory segments, with its dtype.
fn fixture() -> (Arc<Store>, LayoutRef, DType) {
    let mut store = Store::default();
    let mut layouts = Vec::with_capacity(COLUMNS);
    for column in 0..COLUMNS {
        let values =
            PrimitiveArray::from_iter((0..ROWS).map(|row| value(column, row))).into_array();
        layouts.push(store.chunked(&values));
    }
    let dtype = DType::Struct(
        StructFields::from_iter((0..COLUMNS).map(|i| {
            (
                format!("c{i}"),
                DType::Primitive(PType::I64, Nullability::NonNullable),
            )
        })),
        Nullability::NonNullable,
    );
    let layout = StructLayout::new(ROWS as u64, dtype.clone(), layouts).into_layout();
    (Arc::new(store), layout, dtype)
}

/// One V1 scan of the query: a reader over the layout, a scan split at the chunk boundaries,
/// drained on a single-thread runtime.
#[divan::bench]
fn query_v1(bencher: Bencher) {
    let (store, layout, dtype) = fixture();
    let (filter, projection) = query(&dtype);
    let runtime = SingleThreadRuntime::default();
    let session = SESSION.clone().with_handle(runtime.handle());
    let expected = expected_rows();
    bencher.counter(ItemsCount::new(expected)).bench_local(|| {
        let reader = layout
            .new_reader(
                "bench".into(),
                Arc::<Store>::clone(&store),
                &session,
                &LayoutReaderContext::new(),
            )
            .expect("reader");
        let rows: usize = ScanBuilder::new(session.clone(), reader)
            .with_filter(filter.clone())
            .with_projection(projection.clone())
            .with_split_by(SplitBy::Layout)
            .into_array_iter(&runtime)
            .expect("scan")
            .map(|array| array.expect("batch").len())
            .sum();
        assert_eq!(rows, expected);
    });
}
