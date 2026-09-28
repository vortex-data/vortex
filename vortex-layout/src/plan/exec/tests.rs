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
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::serde::SerializeOptions;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
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
use crate::layouts::struct_::StructLayout;
use crate::plan::EvalPlan;
use crate::plan::Filter;
use crate::plan::SegmentScan;
use crate::plan::Take;
use crate::plan::lower;
use crate::plan::optimize;
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

/// Which outstanding request the scripted IO service completes next.
#[derive(Clone, Copy, Debug)]
enum Delivery {
    /// Oldest request first.
    Fifo,
    /// Newest request first.
    Lifo,
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// A batch returned by `compute`, as segment ids.
    Io(Vec<u32>),
    /// A result delivered for this segment.
    Delivered(u32),
    /// A piece reached the root.
    Piece(Range<u64>),
}

struct Run {
    pieces: Vec<Piece>,
    events: Vec<Event>,
}

/// Drives a graph the way an owner does, completing reads only while the graph waits.
fn run(
    store: &Store,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    mut pick: impl FnMut(&[IoRequest]) -> usize,
) -> VortexResult<Run> {
    let mut graph = ExecGraph::try_new(SESSION.clone(), plan, rows, mask, 0)?;
    let mut inflight: Vec<IoRequest> = Vec::new();
    let mut pieces = Vec::new();
    let mut events = Vec::new();
    loop {
        match graph.state() {
            ExecState::Done => break,
            ExecState::NeedsCompute => match graph.compute()? {
                ExecOutput::Piece(piece) => {
                    events.push(Event::Piece(piece.rows.clone()));
                    pieces.push(piece);
                }
                ExecOutput::NeedsIO(batch) => {
                    events.push(Event::Io(batch.iter().map(|r| *r.segment_id).collect()));
                    inflight.extend(batch);
                }
                ExecOutput::Yield => {}
            },
            ExecState::Waiting => {
                if inflight.is_empty() {
                    return Err(vortex_err!("graph waits with no reads in flight"));
                }
                let request = inflight.remove(pick(&inflight));
                events.push(Event::Delivered(*request.segment_id));
                graph.set_io_result(request.id, store.read(request.segment_id))?;
            }
        }
    }
    assert!(inflight.is_empty(), "graph finished with reads in flight");
    Ok(Run { pieces, events })
}

fn delivery(order: Delivery) -> impl FnMut(&[IoRequest]) -> usize {
    move |inflight| match order {
        Delivery::Fifo => 0,
        Delivery::Lifo => inflight.len() - 1,
    }
}

/// Delivers segments in the given order.
fn scripted(order: &[u32]) -> impl FnMut(&[IoRequest]) -> usize + '_ {
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

/// Checks that the pieces tile `rows` exactly and together hold the selected rows in order.
fn assert_view(
    expected: &ArrayRef,
    rows: &Range<u64>,
    mask: &Mask,
    mut pieces: Vec<Piece>,
) -> VortexResult<()> {
    pieces.sort_by_key(|piece| piece.rows.start);
    let mut cursor = rows.start;
    for piece in &pieces {
        assert_eq!(piece.rows.start, cursor, "pieces must tile the view");
        let selected = mask
            .slice((piece.rows.start - rows.start) as usize..(piece.rows.end - rows.start) as usize)
            .true_count();
        assert_eq!(piece.array.len(), selected, "piece {:?} length", piece.rows);
        cursor = piece.rows.end;
    }
    assert_eq!(cursor, rows.end, "pieces must tile the view");

    let actual = piece::join(
        expected.dtype(),
        pieces.into_iter().map(|piece| piece.array).collect(),
    )?;
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
fn views_of_one_plan(
    #[case] rows: Range<u64>,
    #[case] sel: Sel,
    #[values(Delivery::Fifo, Delivery::Lifo)] order: Delivery,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = sel.mask((rows.end - rows.start) as usize);

    let run = run(&store, &plan, rows.clone(), mask.clone(), delivery(order))?;
    assert_view(&expected, &rows, &mask, run.pieces)
}

#[test]
fn many_views_share_one_plan() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;

    // Split the row domain into views that cut through every column at different places.
    for split in [[0, 4, 11, 20], [0, 9, 13, 20], [0, 1, 2, 20]] {
        for window in split.windows(2) {
            let rows = window[0]..window[1];
            let mask = Sel::EveryOther.mask((rows.end - rows.start) as usize);
            let run = run(
                &store,
                &plan,
                rows.clone(),
                mask.clone(),
                delivery(Delivery::Lifo),
            )?;
            assert_view(&expected, &rows, &mask, run.pieces)?;
        }
    }
    Ok(())
}

#[test]
fn first_compute_publishes_every_read_in_one_batch() -> VortexResult<()> {
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
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
}

#[test]
fn nothing_selected_issues_no_io() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = Mask::new_false(ROWS as usize);

    let run = run(
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(reads(&run.events), 0);
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
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

#[test]
fn one_read_releases_several_pieces() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let mask = Mask::new_true(ROWS as usize);

    // Segments: a0=0, a1=1, a2=2, b=3. Hold a1 back so `a` has a hole at [7,12).
    let order = [0, 2, 3, 1];
    let run = run(&store, &plan, 0..ROWS, mask.clone(), scripted(&order))?;

    assert_eq!(
        run.events,
        vec![
            Event::Io(vec![0, 1, 2, 3]),
            Event::Delivered(0),
            Event::Delivered(2),
            // Nothing is emitted until `b` arrives, which then completes both covered ranges.
            Event::Delivered(3),
            Event::Piece(0..7),
            Event::Piece(12..20),
            Event::Delivered(1),
            Event::Piece(7..12),
        ]
    );
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
}

#[test]
fn reverse_delivery_emits_pieces_out_of_row_order() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Lifo),
    )?;
    let pieces = run
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Piece(rows) => Some(rows.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(pieces, vec![12..20, 7..12, 0..7]);
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
}

#[test]
fn state_is_side_effect_free() -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, _) = two_columns(&mut store)?;
    let graph = ExecGraph::try_new(
        SESSION.clone(),
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        0,
    )?;
    for _ in 0..3 {
        assert_eq!(graph.state(), ExecState::NeedsCompute);
    }
    Ok(())
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
    assert_eq!(dense.pieces.len(), 1);
    assert_eq!(dense.pieces[0].rows, rows);
    assert_arrays_eq!(dense.pieces[0].array, values.slice(3..9)?, &mut ctx);

    let kept = run(
        &store,
        &filtered,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_view(&values, &rows, &mask, kept.pieces)?;
    Ok(())
}

/// Executions of a take plan after the first reuse its values, reading only the codes, including
/// when a predicate has been pushed onto the values.
#[rstest]
#[case::values(false)]
#[case::predicate(true)]
fn take_values_are_read_once_per_plan(#[case] predicate: bool) -> VortexResult<()> {
    let mut store = Store::default();
    let values = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
    let codes = PrimitiveArray::from_iter((0..ROWS).map(|v| (v % 3) as u8)).into_array();
    let layout = DictLayout::new(store.flat(&values)?, store.flat(&codes)?).into_layout();
    let mut plan = lower(&layout)?;
    let mut expected = values.take(codes)?;
    if predicate {
        let expression = gt(root(), lit("a"))
            .optimize_recursive(plan.dtype())?
            .bind(plan.dtype())?;
        expected = expected.apply_bound(&expression)?;
        plan = optimize(EvalPlan::try_new(expression, plan)?.into_plan())?;
    }
    assert!(plan.is::<Take>());

    for (split, rows) in [0..10, 10..ROWS].into_iter().enumerate() {
        let mask = Mask::new_true(10);
        let run = run(
            &store,
            &plan,
            rows.clone(),
            mask.clone(),
            delivery(Delivery::Fifo),
        )?;
        assert_eq!(reads(&run.events), if split == 0 { 2 } else { 1 });
        assert_view(&expected, &rows, &mask, run.pieces)?;
    }
    Ok(())
}
