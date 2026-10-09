// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Fixtures use a 20-row domain and name columns after their struct fields.
#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]

use std::ops::Range;

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
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use super::*;
use crate::LayoutRef;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::list::ListLayout;
use crate::layouts::struct_::StructLayout;
use crate::plan::EvalPlan;
use crate::plan::Filter;
use crate::plan::PlanRef;
use crate::plan::SegmentScan;
use crate::plan::Take;
use crate::plan::lower;
use crate::plan::optimize;
use crate::segments::SegmentId;
use crate::test::SESSION;

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

/// Pack emits a struct as soon as every field has rows, however the reads complete, and the
/// structs come out in row order, one per chunk of `a`.
///
/// Segments: a0=0, a1=1, a2=2, b=3.
#[rstest]
// `b` arrives last: nothing can be emitted before it.
#[case::b_last(&[1, 0, 2, 3], 4)]
// `b` and `a0` first: rows 0..7 go out before `a2` and `a1` arrive.
#[case::a_chunk_last(&[3, 0, 2, 1], 2)]
// Chunks in order: each chunk's struct goes out as it lands.
#[case::in_order(&[3, 0, 1, 2], 2)]
fn pack_streams_in_row_order(
    #[case] order: &[u32],
    #[case] deliveries_before_first_piece: usize,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(&store, &plan, 0..ROWS, mask.clone(), scripted(order))?;
    assert_eq!(pieces(&run.events), [7, 5, 8]);
    let first_piece = run
        .events
        .iter()
        .position(|e| matches!(e, Event::Piece(_)))
        .vortex_expect("a piece");
    let delivered = run.events[..first_piece]
        .iter()
        .filter(|e| matches!(e, Event::Delivered(_)))
        .count();
    assert_eq!(delivered, deliveries_before_first_piece);
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

/// A bare segment scan returns every row of its range whatever it is told to care about, and a
/// filter over the same scan returns only the selected rows.
#[rstest]
#[case::every_other(Sel::EveryOther)]
#[case::sparse(Sel::Rows(&[1, 4]))]
#[case::nothing(Sel::None)]
fn bare_scan_is_dense_and_filter_keeps_the_selection(#[case] sel: Sel) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter(0..ROWS as i32).into_array();
    let filtered = lower(&store.flat(&values)?)?;
    assert!(filtered.is::<Filter>());
    let scan = filtered.child_required(0)?;
    assert!(scan.is::<SegmentScan>());

    let rows = 3..9;
    let mask = sel.mask(6);
    let mut ctx = SESSION.create_execution_ctx();

    let dense = run(
        &store,
        &scan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(dense.arrays.len(), 1);
    assert_arrays_eq!(dense.arrays[0], values.slice(3..9)?, &mut ctx);

    let kept = run(
        &store,
        &filtered,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_view(&values, &rows, &mask, kept.arrays)?;
    Ok(())
}

/// A take reads its values over their whole domain and its codes over the selection, including
/// when a predicate has been pushed onto the values, and emits nothing before the values are
/// whole. Later scans of the plan reuse the values and read only the codes.
#[rstest]
#[case::values(false)]
#[case::predicate(true)]
fn take_waits_for_whole_values_and_keeps_them(#[case] predicate: bool) -> VortexResult<()> {
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
