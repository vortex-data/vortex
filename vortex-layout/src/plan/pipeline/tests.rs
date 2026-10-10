// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Fixtures use a 20-row domain and name columns after their struct fields.
#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::executor::block_on;
use futures::future;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on as block_on_runtime;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use super::*;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::plan::EvalPlan;
use crate::plan::PlanRef;
use crate::plan::QueryPlan;
use crate::plan::SegmentScan;
use crate::plan::Take;
use crate::plan::lower;
use crate::plan::optimize;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::test::SESSION;
use crate::test::new_session;

const ROWS: u64 = 20;

/// In-memory segments, standing in for the IO service.
#[derive(Default)]
struct Store {
    segments: Vec<BufferHandle>,
}

impl Store {
    fn flat(&mut self, array: &ArrayRef) -> VortexResult<LayoutRef> {
        let ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &ctx,
            &SESSION,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let segment_id = SegmentId::from(u32::try_from(self.segments.len())?);
        self.segments.push(BufferHandle::new_host(bytes.freeze()));
        Ok(FlatLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.to_ids()),
        )
        .into_layout())
    }

    fn chunked(&mut self, array: &ArrayRef, sizes: &[usize]) -> VortexResult<LayoutRef> {
        let mut chunks = Vec::with_capacity(sizes.len());
        let mut start = 0;
        for size in sizes {
            chunks.push(self.flat(&array.slice(start..start + size)?)?);
            start += size;
        }
        assert_eq!(start, array.len(), "chunk sizes must cover the array");
        Ok(ChunkedLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            OwnedLayoutChildren::layout_children(chunks),
        )
        .into_layout())
    }

    fn read(&self, segment_id: SegmentId) -> BufferHandle {
        self.segments[*segment_id as usize].clone()
    }
}

/// A struct whose columns are chunked at different row boundaries.
///
/// ```text
/// a  i32        chunks [0,7) [7,12) [12,20)
/// b  i64        chunks [0,10) [10,20)
/// c  utf8       flat   [0,20)
/// d  i32?       chunks [0,1) [1,3) [3,6) [6,20)
/// e  struct     x: chunks [0,5) [5,20), y: flat
/// ```
fn fixture(store: &mut Store) -> VortexResult<(PlanRef, ArrayRef)> {
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let b = PrimitiveArray::from_iter((0..ROWS as i64).map(|v| v * 10)).into_array();
    let c = VarBinViewArray::from_iter_str((0..ROWS).map(|v| format!("row-{v}"))).into_array();
    let d = PrimitiveArray::from_option_iter((0..ROWS as i32).map(|v| (v % 3 != 0).then_some(v)))
        .into_array();
    let x = PrimitiveArray::from_iter((0..ROWS).map(|v| v as u8)).into_array();
    let y = BoolArray::from_iter((0..ROWS).map(|v| v % 2 == 0)).into_array();
    let e = StructArray::from_fields(&[("x", x.clone()), ("y", y.clone())])?.into_array();
    let expected = StructArray::from_fields(&[
        ("a", a.clone()),
        ("b", b.clone()),
        ("c", c.clone()),
        ("d", d.clone()),
        ("e", e.clone()),
    ])?
    .into_array();

    let e_layout = StructLayout::new(
        ROWS,
        e.dtype().clone(),
        vec![store.chunked(&x, &[5, 15])?, store.flat(&y)?],
    )
    .into_layout();
    let layout = StructLayout::new(
        ROWS,
        expected.dtype().clone(),
        vec![
            store.chunked(&a, &[7, 5, 8])?,
            store.chunked(&b, &[10, 10])?,
            store.flat(&c)?,
            store.chunked(&d, &[1, 2, 3, 14])?,
            e_layout,
        ],
    )
    .into_layout();
    Ok((lower(&layout)?, expected))
}

/// Which outstanding read the scripted IO service completes next.
#[derive(Clone, Copy, Debug)]
enum Delivery {
    /// Oldest read first.
    Fifo,
    /// Newest read first.
    Lifo,
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// Reads the scan asked for before it next waited, as segment ids.
    Io(Vec<u32>),
    /// A read delivered for this segment.
    Delivered(u32),
    /// An array of this many rows came out.
    Piece(usize),
}

struct Run {
    /// Every array, in the order it came out.
    arrays: Vec<ArrayRef>,
    /// The arrays of each split, in order.
    splits: Vec<Vec<ArrayRef>>,
    events: Vec<Event>,
    /// Ports and shared readers still held once the scan finished.
    leftover: (usize, usize),
}

/// Drives a scan the way an owner does, completing reads only while the scan waits.
fn drive(
    store: &Store,
    mut scan: Scan,
    mut pick: impl FnMut(&[ReadRequest]) -> usize,
) -> VortexResult<Run> {
    let mut inflight: Vec<ReadRequest> = Vec::new();
    let mut batch: Vec<u32> = Vec::new();
    let mut arrays = Vec::new();
    let mut splits: Vec<Vec<ArrayRef>> = Vec::new();
    let mut events = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => {
                batch.push(*read.segment_id);
                inflight.push(read);
            }
            Turn::Output(split, array) => {
                if !batch.is_empty() {
                    events.push(Event::Io(std::mem::take(&mut batch)));
                }
                assert!(!array.is_empty(), "a scan never emits an empty array");
                events.push(Event::Piece(array.len()));
                if split >= splits.len() {
                    splits.resize_with(split + 1, Vec::new);
                }
                splits[split].push(array.clone());
                arrays.push(array);
            }
            Turn::Waiting => {
                if !batch.is_empty() {
                    events.push(Event::Io(std::mem::take(&mut batch)));
                }
                if inflight.is_empty() {
                    return Err(vortex_err!("scan waits with no reads in flight"));
                }
                let read = inflight.remove(pick(&inflight));
                events.push(Event::Delivered(*read.segment_id));
                scan.deliver(read.id, store.read(read.segment_id))?;
            }
            Turn::Done => break,
        }
    }
    assert!(
        batch.is_empty() && inflight.is_empty(),
        "scan finished with reads in flight"
    );
    let leftover = (scan.live_ports(), scan.pending_shares());
    Ok(Run {
        arrays,
        splits,
        events,
        leftover,
    })
}

/// Runs `plan` over one split.
fn run(
    store: &Store,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    pick: impl FnMut(&[ReadRequest]) -> usize,
) -> VortexResult<Run> {
    let scan = Scan::try_new(SESSION.clone(), plan.clone(), vec![Split { rows, mask }])?;
    let run = drive(store, scan, pick)?;
    assert_eq!(run.leftover, (0, 0), "a finished scan holds no port");
    Ok(run)
}

fn delivery(order: Delivery) -> impl FnMut(&[ReadRequest]) -> usize {
    move |inflight| match order {
        Delivery::Fifo => 0,
        Delivery::Lifo => inflight.len() - 1,
    }
}

/// Delivers segments in the given order.
fn scripted(order: &[u32]) -> impl FnMut(&[ReadRequest]) -> usize + '_ {
    let mut next = order.iter();
    move |inflight| {
        let segment = *next.next().expect("script covers every read");
        inflight
            .iter()
            .position(|r| *r.segment_id == segment)
            .expect("scripted segment is in flight")
    }
}

fn reads(events: &[Event]) -> usize {
    events
        .iter()
        .map(|event| match event {
            Event::Io(batch) => batch.len(),
            _ => 0,
        })
        .sum()
}

/// Every segment read, sorted.
fn segments_read(events: &[Event]) -> Vec<u32> {
    let mut read: Vec<u32> = events
        .iter()
        .flat_map(|event| match event {
            Event::Io(batch) => batch.clone(),
            _ => Vec::new(),
        })
        .collect();
    read.sort_unstable();
    read
}

fn pieces(events: &[Event]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Piece(len) => Some(*len),
            _ => None,
        })
        .collect()
}

/// A view's selection over its rows.
#[derive(Clone, Copy, Debug)]
enum Sel {
    All,
    None,
    EveryOther,
    /// Rows relative to the view start.
    Rows(&'static [usize]),
}

impl Sel {
    fn mask(self, len: usize) -> Mask {
        match self {
            Sel::All => Mask::new_true(len),
            Sel::None => Mask::new_false(len),
            Sel::EveryOther => Mask::from_iter((0..len).map(|i| i % 2 == 0)),
            Sel::Rows(rows) => Mask::from_indices(len, rows.iter().copied()),
        }
    }
}

/// Joins arrays covering consecutive rows into one.
fn join(dtype: &vortex_array::dtype::DType, arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array())
}

/// Checks that the arrays, in the order they came out, are the selected rows of the view.
fn assert_view(
    expected: &ArrayRef,
    rows: &Range<u64>,
    mask: &Mask,
    arrays: Vec<ArrayRef>,
) -> VortexResult<()> {
    let actual = join(expected.dtype(), arrays)?;
    let expected = expected
        .slice(rows.start as usize..rows.end as usize)?
        .filter(mask.clone())?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::full(0..20, Sel::All)]
#[case::crosses_every_boundary(3..17, Sel::All)]
#[case::every_other(5..15, Sel::EveryOther)]
#[case::sparse(0..20, Sel::Rows(&[0, 6, 7, 19]))]
#[case::one_row(8..9, Sel::All)]
#[case::inside_one_chunk_per_column(13..16, Sel::All)]
#[case::nothing_selected(2..18, Sel::None)]
#[case::empty_range(5..5, Sel::All)]
fn views_of_one_plan(
    #[case] rows: Range<u64>,
    #[case] sel: Sel,
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = sel.mask((rows.end - rows.start) as usize);

    let run = run(&store, &plan, rows.clone(), mask.clone(), delivery(order))?;
    assert_view(&expected, &rows, &mask, run.arrays)
}

/// Splits cutting every column at different places, run as one scan, each produce their own
/// rows, and a segment several splits read is read and decoded once, then dropped.
#[rstest]
fn splits_of_one_scan_read_each_segment_once(
    #[values([0, 4, 11, 20], [0, 9, 13, 20], [0, 1, 2, 20])] cuts: [u64; 4],
    #[values(1, 3)] active: usize,
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let splits: Vec<Split> = cuts
        .windows(2)
        .map(|w| Split {
            rows: w[0]..w[1],
            mask: Sel::EveryOther.mask((w[1] - w[0]) as usize),
        })
        .collect();
    let scan = Scan::try_new(SESSION.clone(), plan, splits.clone())?.with_max_active(active);
    let run = drive(&store, scan, delivery(order))?;
    assert_eq!(run.leftover, (0, 0), "every shared segment was dropped");
    assert_eq!(
        segments_read(&run.events),
        (0..store.segments.len() as u32).collect::<Vec<_>>()
    );
    for (index, split) in splits.iter().enumerate() {
        assert_view(
            &expected,
            &split.rows,
            &split.mask,
            run.splits.get(index).cloned().unwrap_or_default(),
        )?;
    }
    Ok(())
}

#[test]
fn every_read_is_issued_before_any_is_delivered() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, _) = fixture(&mut store)?;

    let run = run(
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    let Some(Event::Io(first)) = run.events.first() else {
        return Err(vortex_err!("first event must be a read batch"));
    };
    assert_eq!(first.len(), store.segments.len());
    assert_eq!(reads(&run.events), store.segments.len());
    Ok(())
}

#[test]
fn unselected_chunks_are_never_read() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    // Rows 7..12 only: a reads one chunk, b both, c one, d one, e.x one, e.y one.
    let mask = Mask::from_indices(ROWS as usize, 7..12);

    let run = run(
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(reads(&run.events), 7);
    assert_view(&expected, &(0..ROWS), &mask, run.arrays)
}

#[test]
fn nothing_selected_issues_no_io_and_emits_nothing() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, _) = fixture(&mut store)?;
    let mask = Mask::new_false(ROWS as usize);

    let run = run(&store, &plan, 0..ROWS, mask, delivery(Delivery::Fifo))?;
    assert_eq!(reads(&run.events), 0);
    assert!(run.arrays.is_empty());
    Ok(())
}

/// `{a, b}` with `a` chunked `[0,7) [7,12) [12,20)` and `b` flat.
fn two_columns(store: &mut Store) -> VortexResult<(PlanRef, ArrayRef)> {
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let b = PrimitiveArray::from_iter((0..ROWS as i64).map(|v| v * 10)).into_array();
    let expected = StructArray::from_fields(&[("a", a.clone()), ("b", b.clone())])?.into_array();
    let layout = StructLayout::new(
        ROWS,
        expected.dtype().clone(),
        vec![store.chunked(&a, &[7, 5, 8])?, store.flat(&b)?],
    )
    .into_layout();
    Ok((lower(&layout)?, expected))
}

/// Fields chunked differently are joined, not cut at each other's boundaries: Pack waits for the
/// field with the fewest rows to fill or close, then emits every row all fields hold, in row
/// order, however the reads complete.
///
/// Segments: a0=0, a1=1, a2=2, b=3.
#[rstest]
#[case::b_last(&[1, 0, 2, 3])]
#[case::a_chunk_last(&[3, 0, 2, 1])]
#[case::in_order(&[3, 0, 1, 2])]
fn pack_joins_misaligned_fields(#[case] order: &[u32]) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(&store, &plan, 0..ROWS, mask.clone(), scripted(order))?;
    assert_eq!(pieces(&run.events), [ROWS as usize]);
    assert_view(&expected, &(0..ROWS), &mask, run.arrays)
}

/// A misaligned field that fills its inlet makes Pack emit what every field holds, so a struct
/// never waits for more rows than the inlets can queue.
#[test]
fn pack_emits_when_the_shortest_field_fills() -> VortexResult<()> {
    let mut store = Store::default();
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let b = PrimitiveArray::from_iter((0..ROWS as i64).map(|v| v * 10)).into_array();
    let expected = StructArray::from_fields(&[("a", a.clone()), ("b", b.clone())])?.into_array();
    let layout = StructLayout::new(
        ROWS,
        expected.dtype().clone(),
        vec![store.chunked(&a, &[1; ROWS as usize])?, store.flat(&b)?],
    )
    .into_layout();
    let plan = lower(&layout)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    let pieces = pieces(&run.events);
    assert!(pieces.len() > 1, "{pieces:?}");
    assert!(
        pieces.iter().all(|&piece| piece <= DEFAULT_CAPACITY),
        "{pieces:?}"
    );
    assert_view(&expected, &(0..ROWS), &mask, run.arrays)
}

/// Fields chunked alike stream one struct per chunk.
#[test]
fn aligned_fields_stream_one_struct_per_chunk() -> VortexResult<()> {
    let mut store = Store::default();
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let b = PrimitiveArray::from_iter((0..ROWS as i64).map(|v| v * 10)).into_array();
    let expected = StructArray::from_fields(&[("a", a.clone()), ("b", b.clone())])?.into_array();
    let layout = StructLayout::new(
        ROWS,
        expected.dtype().clone(),
        vec![
            store.chunked(&a, &[7, 5, 8])?,
            store.chunked(&b, &[7, 5, 8])?,
        ],
    )
    .into_layout();
    let plan = lower(&layout)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(pieces(&run.events), [7, 5, 8]);
    assert_view(&expected, &(0..ROWS), &mask, run.arrays)
}

/// A segment scan keeps the rows its split selects, and so does a filter over the same scan,
/// which runs as the same source.
#[rstest]
#[case::every_other(Sel::EveryOther)]
#[case::sparse(Sel::Rows(&[1, 4]))]
#[case::nothing(Sel::None)]
fn scan_keeps_the_selection_with_or_without_a_filter(#[case] sel: Sel) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let scan = lower(&store.flat(&values)?)?;
    assert!(scan.is::<SegmentScan>());
    let filtered = crate::plan::FilterPlan::new(scan.clone()).into_plan();

    let rows = 3..9;
    let mask = sel.mask(6);
    for plan in [&scan, &filtered] {
        let kept = run(
            &store,
            plan,
            rows.clone(),
            mask.clone(),
            delivery(Delivery::Fifo),
        )?;
        if mask.all_false() {
            assert_eq!(reads(&kept.events), 0, "nothing selected reads nothing");
        }
        assert_view(&values, &rows, &mask, kept.arrays)?;
    }
    Ok(())
}

/// A take reads its values over their whole domain and its codes over the selection, including
/// when a predicate has been pushed onto the values, and emits nothing before the values are
/// whole. The splits of one scan share the values, read once.
#[rstest]
#[case::values(false)]
#[case::predicate(true)]
fn take_waits_for_whole_values_and_shares_them(#[case] predicate: bool) -> VortexResult<()> {
    let mut store = Store::default();
    let values = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
    let codes = PrimitiveArray::from_iter((0..ROWS).map(|v| (v % 3) as u8)).into_array();
    let layout = DictLayout::new(store.flat(&values)?, store.flat(&codes)?).into_layout();
    let mut plan = lower(&layout)?;
    let mut expected = values.take(codes)?;
    if predicate {
        let expression = gt(root(), lit("a"))
            .bind(plan.dtype())?
            .optimize_recursive()?;
        expected = expected.apply_bound(&expression)?;
        plan = optimize(EvalPlan::try_new(expression, plan)?.into_plan())?;
    }
    assert!(plan.is::<Take>());

    let mask = Sel::EveryOther.mask(10);
    // Codes (segment 1) land first; nothing comes out until the values (segment 0) do.
    let first = run(&store, &plan, 0..10, mask.clone(), scripted(&[1, 0]))?;
    assert_eq!(reads(&first.events), 2);
    let first_piece = first
        .events
        .iter()
        .position(|e| matches!(e, Event::Piece(_)))
        .vortex_expect("a piece");
    assert!(
        first.events[first_piece..]
            .iter()
            .all(|e| !matches!(e, Event::Delivered(_))),
        "the only array must follow both deliveries"
    );
    assert_view(&expected, &(0..10), &mask, first.arrays)?;

    // The values are kept for the scan, not the plan: two splits of one scan read them once,
    // and a later scan reads them again.
    let splits = vec![
        Split {
            rows: 0..10,
            mask: mask.clone(),
        },
        Split {
            rows: 10..ROWS,
            mask: mask.clone(),
        },
    ];
    let scan = Scan::try_new(SESSION.clone(), plan, splits)?;
    let both = drive(&store, scan, delivery(Delivery::Fifo))?;
    assert_eq!(
        both.leftover,
        (0, 0),
        "the values are dropped with the last split"
    );
    // The codes are one segment the two splits share too.
    assert_eq!(segments_read(&both.events), [0, 1]);
    assert_view(&expected, &(10..ROWS), &mask, both.splits[1].clone())?;
    Ok(())
}

#[test]
fn eval_applies_expression_to_selected_rows() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let expression = gt(get_item("a", root()), lit(4_i32)).bind(plan.dtype())?;
    let expected = expected.apply_bound(&expression)?;
    let plan = EvalPlan::try_new(expression, plan)?.into_plan();

    let rows = 2..18;
    let mask = Sel::EveryOther.mask(16);
    let run = run(
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Lifo),
    )?;
    assert_view(&expected, &rows, &mask, run.arrays)
}

/// A column read by two fields of one struct is read and decoded once, and both fields get it.
#[test]
fn a_segment_two_readers_need_is_read_once() -> VortexResult<()> {
    let mut store = Store::default();
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let column = lower(&store.chunked(&a, &[7, 13])?)?;
    let expected = StructArray::from_fields(&[("x", a.clone()), ("y", a)])?.into_array();
    let fields = expected
        .dtype()
        .as_struct_fields_opt()
        .vortex_expect("struct")
        .clone();
    let plan = crate::plan::PackPlan::try_new(
        fields,
        vortex_array::dtype::Nullability::NonNullable,
        ROWS,
        vec![column.clone(), column],
        None,
    )?
    .into_plan();
    let rows = 3..17;
    let mask = Sel::EveryOther.mask(14);

    let run = run(
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Lifo),
    )?;
    assert_eq!(segments_read(&run.events), [0, 1]);
    assert_view(&expected, &rows, &mask, run.arrays)
}
#[rstest]
#[case::full(0..6, Sel::All)]
#[case::range(1..5, Sel::All)]
#[case::sparse(0..6, Sel::Rows(&[1, 3, 5]))]
#[case::empty_list(1..2, Sel::All)]
#[case::null_list(2..3, Sel::All)]
#[case::no_rows(1..5, Sel::None)]
#[case::empty_range(3..3, Sel::All)]
fn list_views(
    #[case] rows: Range<u64>,
    #[case] sel: Sel,
    #[values(false, true)] nullable: bool,
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
) -> VortexResult<()> {
    let mut store = Store::default();
    let elements = PrimitiveArray::from_option_iter([
        Some(0i32),
        None,
        Some(2),
        Some(3),
        None,
        Some(5),
        Some(6),
        Some(7),
    ])
    .into_array();
    // Nonzero first offsets also occur when a list's children are sliced independently.
    let offsets = PrimitiveArray::from_iter([1u64, 3, 3, 4, 6, 6, 8]).into_array();
    let validity = BoolArray::from_iter([true, true, false, true, true, true]).into_array();
    let expected = ListArray::try_new(
        elements.clone(),
        offsets.clone(),
        if nullable {
            Validity::Array(validity.clone())
        } else {
            Validity::NonNullable
        },
    )?
    .into_array();
    let layout = ListLayout::new(
        expected.dtype().clone(),
        store.chunked(&elements, &[2, 3, 3])?,
        store.chunked(&offsets, &[2, 2, 3])?,
        if nullable {
            Some(store.chunked(&validity, &[1, 5])?)
        } else {
            None
        },
    )
    .into_layout();
    let plan = lower(&layout)?;
    let mask = sel.mask((rows.end - rows.start) as usize);
    let output = run(&store, &plan, rows.clone(), mask.clone(), delivery(order))?;
    if mask.all_false() {
        assert_eq!(reads(&output.events), 0);
    }
    assert_view(&expected, &rows, &mask, output.arrays)
}

#[rstest]
fn nested_list_views(
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
) -> VortexResult<()> {
    let mut store = Store::default();
    let elements = PrimitiveArray::from_iter([10i32, 20, 30, 40]).into_array();
    let inner_offsets = PrimitiveArray::from_iter([0u32, 1, 1, 3, 4]).into_array();
    let inner = ListArray::try_new(
        elements.clone(),
        inner_offsets.clone(),
        Validity::NonNullable,
    )?
    .into_array();
    let inner_layout = ListLayout::new(
        inner.dtype().clone(),
        store.chunked(&elements, &[2, 2])?,
        store.chunked(&inner_offsets, &[2, 3])?,
        None,
    )
    .into_layout();
    let offsets = PrimitiveArray::from_iter([0u32, 1, 3, 4]).into_array();
    let expected = ListArray::try_new(inner, offsets.clone(), Validity::NonNullable)?.into_array();
    let layout = ListLayout::new(
        expected.dtype().clone(),
        inner_layout,
        store.flat(&offsets)?,
        None,
    )
    .into_layout();
    let plan = lower(&layout)?;
    let rows = 1..3;
    let mask = Mask::new_true(2);
    let output = run(&store, &plan, rows.clone(), mask.clone(), delivery(order))?;
    assert_view(&expected, &rows, &mask, output.arrays)
}

/// The segments of a [`Store`], served asynchronously.
struct StoreSource(Vec<BufferHandle>);

impl SegmentSource for StoreSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let segment = self
            .0
            .get(*id as usize)
            .cloned()
            .ok_or_else(|| vortex_err!("Segment {id} not found"));
        future::ready(segment).boxed()
    }
}

fn segment_source(store: &Store) -> Arc<dyn SegmentSource> {
    Arc::new(StoreSource(store.segments.clone()))
}

/// [`execute`] yields the selected rows in row order, as non-empty arrays with the plan's dtype.
#[rstest]
#[case::full(0..20, Sel::All)]
#[case::every_other(3..17, Sel::EveryOther)]
#[case::nothing_selected(0..20, Sel::None)]
#[case::empty_range(5..5, Sel::All)]
fn execute_streams_arrays_in_row_order(
    #[case] rows: Range<u64>,
    #[case] sel: Sel,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = sel.mask((rows.end - rows.start) as usize);

    let arrays: Vec<ArrayRef> = block_on(
        execute(
            SESSION.clone(),
            &plan,
            rows.clone(),
            mask.clone(),
            segment_source(&store),
        )?
        .try_collect(),
    )?;
    assert!(arrays.iter().all(|array| !array.is_empty()));
    assert!(arrays.iter().all(|array| array.dtype() == plan.dtype()));
    assert_view(&expected, &rows, &mask, arrays)
}

#[test]
fn execute_fails_when_a_segment_is_missing() -> VortexResult<()> {
    let mut store = Store::default();
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let plan = lower(&store.flat(&a)?)?;

    let result: VortexResult<Vec<ArrayRef>> = block_on(
        execute(
            SESSION.clone(),
            &plan,
            0..ROWS,
            Mask::new_true(ROWS as usize),
            Arc::new(StoreSource(Vec::new())),
        )?
        .try_collect(),
    );
    assert!(result.is_err());
    Ok(())
}

/// A query over `plan` selecting `c` and `e`, and a check of its output over one view.
fn query_over(
    plan: &PlanRef,
    filter: Option<vortex_array::expr::Expression>,
) -> VortexResult<(PlanRef, QueryCheck)> {
    let projection = select(FieldNames::from(["c", "e"]), root()).bind(plan.dtype())?;
    let filter = filter.map(|f| f.bind(plan.dtype())).transpose()?;
    let plan = QueryPlan::try_new(filter.clone(), projection.clone(), plan.clone())?.into_plan();
    Ok((plan, QueryCheck { filter, projection }))
}

struct QueryCheck {
    filter: Option<BoundExpression>,
    projection: BoundExpression,
}

impl QueryCheck {
    /// Checks that `arrays` are the view's selected rows that pass the filter, projected.
    fn assert_view(
        &self,
        source: &ArrayRef,
        rows: &Range<u64>,
        mask: &Mask,
        arrays: Vec<ArrayRef>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut expected = source
            .slice(rows.start as usize..rows.end as usize)?
            .filter(mask.clone())?;
        if let Some(filter) = &self.filter {
            let kept = expected
                .clone()
                .apply_bound(filter)?
                .fill_null(false)?
                .execute::<Mask>(&mut ctx)?;
            expected = expected.filter(kept)?;
        }
        let expected = expected.apply_bound(&self.projection)?;
        let actual = join(expected.dtype(), arrays)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}

/// A query evaluates its conjuncts one at a time and projects the rows that pass them all,
/// whichever path the mask's density sends each conjunct down.
#[rstest]
// Both conjuncts keep most rows: each runs over every row and is intersected.
#[case::dense(and(gt(get_item("a", root()), lit(1_i32)), lt(get_item("b", root()), lit(150_i64))), 2..18)]
// The first conjunct keeps two rows: the second runs under that sparse mask.
#[case::sparse(and(lt(get_item("a", root()), lit(2_i32)), lt(get_item("b", root()), lit(150_i64))), 0..20)]
// A conjunct that keeps nothing: the rest, and the projection, are never run.
#[case::empty(and(gt(get_item("a", root()), lit(100_i32)), lt(get_item("b", root()), lit(150_i64))), 0..20)]
// A nullable conjunct: nulls are kept out.
#[case::nullable(gt(get_item("d", root()), lit(10_i32)), 0..20)]
fn query_filters_then_projects(
    #[case] filter: vortex_array::expr::Expression,
    #[case] rows: Range<u64>,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (source, expected) = fixture(&mut store)?;
    let (plan, check) = query_over(&source, Some(filter))?;
    let mask = Sel::EveryOther.mask((rows.end - rows.start) as usize);

    let run = run(
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Lifo),
    )?;
    check.assert_view(&expected, &rows, &mask, run.arrays)
}

/// Without a filter a query is its projection.
#[test]
fn query_without_filter_is_the_projection() -> VortexResult<()> {
    let mut store = Store::default();
    let (source, expected) = fixture(&mut store)?;
    let (plan, check) = query_over(&source, None)?;
    let rows = 0..ROWS;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    check.assert_view(&expected, &rows, &mask, run.arrays)
}

/// A conjunct that keeps no rows stops the query: later conjuncts and the projection read
/// nothing.
#[test]
fn query_stops_reading_once_nothing_is_selected() -> VortexResult<()> {
    let mut store = Store::default();
    let (source, _) = fixture(&mut store)?;
    let filter = and(
        gt(get_item("a", root()), lit(100_i32)),
        lt(get_item("b", root()), lit(150_i64)),
    );
    let (plan, _) = query_over(&source, Some(filter))?;

    let run = run(
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    // Only the three chunks of `a`, the first conjunct's column, are read.
    assert_eq!(reads(&run.events), 3);
    assert!(pieces(&run.events).is_empty());
    Ok(())
}

/// A column both a conjunct and the projection read is read once per split, and a split the
/// conjuncts rule out releases what its projection would have read.
#[rstest]
fn query_reads_a_shared_column_once(
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
    #[values(1, 2)] active: usize,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (source, expected) = fixture(&mut store)?;
    // `c` is both filtered on and projected.
    let filter = gt(get_item("c", root()), lit("row-12"));
    let (plan, check) = query_over(&source, Some(filter))?;
    let splits = vec![Split::all(0..10), Split::all(10..ROWS)];
    let scan = Scan::try_new(SESSION.clone(), plan, splits.clone())?.with_max_active(active);
    let run = drive(&store, scan, delivery(order))?;
    assert_eq!(run.leftover, (0, 0));
    // `c` is one flat segment; every other column read is read once too.
    let read = segments_read(&run.events);
    let mut distinct = read.clone();
    distinct.dedup();
    assert_eq!(read, distinct, "no segment is read twice");
    for (index, split) in splits.iter().enumerate() {
        check.assert_view(
            &expected,
            &split.rows,
            &split.mask,
            run.splits.get(index).cloned().unwrap_or_default(),
        )?;
    }
    Ok(())
}
/// One `i32` column `1..=9`, written as three chunks of three rows with a zone map of
/// three-row zones, lowered to a plan over a store of its segments.
///
/// Segments: chunk 0 = 0, chunk 1 = 1, chunk 2 = 2, zone table = 3.
fn zoned_column() -> VortexResult<(Store, PlanRef)> {
    let segments = Arc::new(TestSegments::default());
    let (ptr, eof) = SequenceId::root().split();
    let strategy = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
            ..Default::default()
        },
    );
    let stream = ChunkedArray::from_iter([
        buffer![1_i32, 2, 3].into_array(),
        buffer![4_i32, 5, 6].into_array(),
        buffer![7_i32, 8, 9].into_array(),
    ])
    .into_array()
    .to_array_stream()
    .sequenced(ptr);
    let sink = Arc::clone(&segments);
    let layout = block_on_runtime(|handle| async move {
        let session = new_session().with_handle(handle);
        strategy
            .write_stream(ArrayContext::empty().into(), sink, stream, eof, &session)
            .await
    })?;
    let mut store = Store::default();
    while let Ok(segment) = block_on(segments.request(SegmentId::from(store.segments.len() as u32)))
    {
        store.segments.push(segment);
    }
    Ok((store, lower(&layout)?))
}

/// A query over a zoned column prunes the zones its conjunct cannot match before reading any
/// data, reads the zone table once per scan, and reads nothing for a split its zones rule out.
#[test]
fn query_prunes_zones_before_reading_data() -> VortexResult<()> {
    let (store, source) = zoned_column()?;
    let filter = gt(root(), lit(6_i32)).bind(source.dtype())?;
    let projection = root().bind(source.dtype())?;
    let plan = QueryPlan::try_new(Some(filter), projection, source)?.into_plan();
    let mut ctx = SESSION.create_execution_ctx();

    let first = run(
        &store,
        &plan,
        0..ROWS_ZONED,
        Mask::new_true(ROWS_ZONED as usize),
        delivery(Delivery::Fifo),
    )?;
    // The zone table, then only the chunk whose zone may hold a row above 6.
    assert_eq!(
        first.events,
        [
            Event::Io(vec![3]),
            Event::Delivered(3),
            Event::Io(vec![2]),
            Event::Delivered(2),
            Event::Piece(3),
        ]
    );
    let expected = buffer![7_i32, 8, 9].into_array();
    assert_arrays_eq!(join(expected.dtype(), first.arrays)?, expected, &mut ctx);

    // The zone table is kept for the scan, not the plan: a later scan reads it again, and the
    // splits of one scan read it once.
    let second = run(
        &store,
        &plan,
        0..ROWS_ZONED,
        Mask::new_true(ROWS_ZONED as usize),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(segments_read(&second.events), [2, 3]);
    let splits = vec![
        Split::all(0..3),
        Split::all(3..6),
        Split::all(6..ROWS_ZONED),
    ];
    let scan = Scan::try_new(SESSION.clone(), plan.clone(), splits)?;
    let three = drive(&store, scan, delivery(Delivery::Fifo))?;
    assert_eq!(three.leftover, (0, 0));
    assert_eq!(segments_read(&three.events), [2, 3]);

    // A split whose zones are all pruned reads the zone table and no data, and produces nothing.
    let pruned = run(
        &store,
        &plan,
        0..6,
        Mask::new_true(6),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(pruned.events, [Event::Io(vec![3]), Event::Delivered(3)]);
    Ok(())
}

const ROWS_ZONED: u64 = 9;

/// Pruning changes what is read, not what comes out: a view over zones partly pruned returns
/// the selected rows that pass the filter.
#[test]
fn query_over_pruned_zones_returns_the_passing_rows() -> VortexResult<()> {
    let (store, source) = zoned_column()?;
    let filter = lt(root(), lit(5_i32)).bind(source.dtype())?;
    let projection = root().bind(source.dtype())?;
    let plan = QueryPlan::try_new(Some(filter), projection, source)?.into_plan();
    let mut ctx = SESSION.create_execution_ctx();

    let rows = 2..8;
    let mask = Sel::EveryOther.mask(6);
    let run = run(&store, &plan, rows, mask, delivery(Delivery::Lifo))?;
    // Rows 2, 4 and 6 hold 3, 5 and 7; only 3 is below 5. The third chunk's zone is pruned.
    assert_eq!(reads(&run.events), 3);
    let expected = buffer![3_i32].into_array();
    assert_arrays_eq!(join(expected.dtype(), run.arrays)?, expected, &mut ctx);
    Ok(())
}

/// Once a split has gone on to project, later splits read their projection's segments ahead,
/// with their conjuncts', rather than after the conjuncts finish. The first split, with no
/// split seen yet, reads them only once its conjunct has run.
///
/// Segments: a0=0, a1=1, c0=2, c1=3.
#[test]
fn later_splits_read_their_projection_ahead() -> VortexResult<()> {
    let mut store = Store::default();
    let a = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let c = PrimitiveArray::from_iter((0..ROWS as i64).map(|v| v * 10)).into_array();
    let dtype = StructArray::from_fields(&[("a", a.clone()), ("c", c.clone())])?
        .dtype()
        .clone();
    let layout = StructLayout::new(
        ROWS,
        dtype,
        vec![store.chunked(&a, &[10, 10])?, store.chunked(&c, &[10, 10])?],
    )
    .into_layout();
    let source = lower(&layout)?;
    let filter = gt_eq(get_item("a", root()), lit(0_i32)).bind(source.dtype())?;
    let projection = select(FieldNames::from(["c"]), root()).bind(source.dtype())?;
    let plan = QueryPlan::try_new(Some(filter), projection, source)?.into_plan();

    let splits = vec![Split::all(0..10), Split::all(10..ROWS)];
    let scan = Scan::try_new(SESSION.clone(), plan, splits)?.with_max_active(1);
    let run = drive(&store, scan, delivery(Delivery::Fifo))?;
    let batches: Vec<Vec<u32>> = run
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Io(segments) => {
                let mut segments = segments.clone();
                segments.sort_unstable();
                Some(segments)
            }
            _ => None,
        })
        .collect();
    assert_eq!(batches, [vec![0], vec![2], vec![1, 3]]);
    Ok(())
}
