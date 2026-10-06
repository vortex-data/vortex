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
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::and;
use vortex_array::expr::binary;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::SerializeOptions;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
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
use crate::plan::SegmentScan;
use crate::plan::SegmentScanPlan;
use crate::plan::Take;
use crate::plan::lower;
use crate::plan::optimize;
use crate::plan::pipeline::PipelineGraph;
use crate::test::SESSION;

const ROWS: u64 = 20;

#[rstest]
#[case::whole(0..10)]
#[case::offset(2..9)]
#[case::single_run(4..6)]
#[case::empty(3..3)]
fn compound_encoded_predicate_preserves_nulls_and_slices(
    #[case] rows: Range<usize>,
    #[values(false, true)] disjunction: bool,
    #[values(false, true)] pco: bool,
) -> VortexResult<()> {
    let session = crate::test::new_session();
    vortex_runend::initialize(&session);
    session.arrays().register(Pco);
    let mut ctx = session.create_execution_ctx();
    let expected = PrimitiveArray::from_option_iter([
        Some(1_i32), Some(1), None, None, Some(3), Some(3), Some(3), Some(5), Some(5), None,
    ]);
    let encoded = if pco {
        Pco::from_primitive(expected.as_view(), 0, 128, &mut ctx)?.into_array()
    } else {
        RunEnd::encode(expected.clone().into_array(), &mut ctx)?.into_array()
    }
    .slice(rows.clone())?;
    let lhs = gt(root(), lit(1_i32));
    let rhs = lt(root(), lit(5_i32));
    let expression = if disjunction {
        or(lhs, rhs)
    } else {
        and(lhs, rhs)
    }
    .bind(encoded.dtype())?;
    let source = SegmentScanPlan::new(
        encoded.dtype().clone(),
        encoded.len() as u64,
        SegmentId::from(0),
        ReadContext::new([]),
        None,
    );
    let plan = EvalPlan::try_new(expression.clone(), source.into_plan())?;
    let actual = plan.apply(encoded, &session)?;
    let expected = expected.into_array().slice(rows)?.apply_bound(&expression)?;
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

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

/// The executor a test runs its plan on.
#[derive(Clone, Copy, Debug)]
enum Executor {
    /// [`ExecGraph`], which grows as its nodes run.
    Nodes,
    /// [`PipelineGraph`], which is compiled whole.
    Pipelines,
}

/// A graph of either executor. Both are driven the same way.
enum Graph {
    Nodes(ExecGraph),
    Pipelines(PipelineGraph),
}

impl Graph {
    fn try_new(
        executor: Executor,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        decoded: DecodeCache,
    ) -> VortexResult<Self> {
        let session = SESSION.clone();
        Ok(match executor {
            Executor::Nodes => {
                Self::Nodes(ExecGraph::try_new(session, plan, rows, mask, 0, decoded)?)
            }
            Executor::Pipelines => {
                Self::Pipelines(PipelineGraph::try_new(session, plan, rows, mask, 0, decoded)?)
            }
        })
    }

    fn state(&self) -> ExecState {
        match self {
            Self::Nodes(graph) => graph.state(),
            Self::Pipelines(graph) => graph.state(),
        }
    }

    fn compute(&mut self) -> VortexResult<ExecOutput> {
        match self {
            Self::Nodes(graph) => graph.compute(),
            Self::Pipelines(graph) => graph.compute(),
        }
    }

    fn set_io_result(&mut self, id: IoRequestId, result: BufferHandle) -> VortexResult<()> {
        match self {
            Self::Nodes(graph) => graph.set_io_result(id, result),
            Self::Pipelines(graph) => graph.set_io_result(id, result),
        }
    }
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
    executor: Executor,
    store: &Store,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    pick: impl FnMut(&[IoRequest]) -> usize,
) -> VortexResult<Run> {
    run_with(
        executor,
        store,
        plan,
        rows,
        mask,
        pick,
        DecodeCache::default(),
    )
}

/// Like [`run`], sharing `decoded` with other graphs.
fn run_with(
    executor: Executor,
    store: &Store,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    mut pick: impl FnMut(&[IoRequest]) -> usize,
    decoded: DecodeCache,
) -> VortexResult<Run> {
    let mut graph = Graph::try_new(executor, plan, rows, mask, decoded)?;
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

    let actual = join(
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
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = sel.mask((rows.end - rows.start) as usize);

    let run = run(executor, &store, &plan, rows.clone(), mask.clone(), delivery(order))?;
    assert_view(&expected, &rows, &mask, run.pieces)
}

#[rstest]
fn many_views_share_one_plan(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;

    // Split the row domain into views that cut through every column at different places.
    for split in [[0, 4, 11, 20], [0, 9, 13, 20], [0, 1, 2, 20]] {
        for window in split.windows(2) {
            let rows = window[0]..window[1];
            let mask = Sel::EveryOther.mask((rows.end - rows.start) as usize);
            let run = run(
                executor,
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

#[rstest]
fn first_compute_publishes_every_read_in_one_batch(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, _) = fixture(&mut store)?;

    let run = run(
        executor,
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

#[rstest]
fn unselected_chunks_are_never_read(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    // Rows 7..12 only: a reads one chunk, b both, c one, d one, e.x one, e.y one.
    let mask = Mask::from_indices(ROWS as usize, 7..12);

    let run = run(
        executor,
        &store,
        &plan,
        0..ROWS,
        mask.clone(),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(reads(&run.events), 7);
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
}

#[rstest]
fn nothing_selected_issues_no_io(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let mask = Mask::new_false(ROWS as usize);

    let run = run(
        executor,
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

/// A struct is emitted once, after the last of its fields' reads, whatever order they arrive in.
///
/// Segments: a0=0, a1=1, a2=2, b=3.
#[rstest]
// `b` is complete while `a` still has a hole at [7,12).
#[case::a_chunk_last(&[0, 2, 3, 1])]
#[case::reverse(&[3, 2, 1, 0])]
#[case::b_last(&[1, 0, 2, 3])]
fn pack_emits_once_every_field_has_closed(
    #[case] order: &[u32],
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = two_columns(&mut store)?;
    let mask = Mask::new_true(ROWS as usize);

    let run = run(executor, &store, &plan, 0..ROWS, mask.clone(), scripted(order))?;

    let mut events = vec![Event::Io(vec![0, 1, 2, 3])];
    events.extend(order.iter().map(|segment| Event::Delivered(*segment)));
    events.push(Event::Piece(0..ROWS));
    assert_eq!(run.events, events);
    assert_view(&expected, &(0..ROWS), &mask, run.pieces)
}

#[rstest]
fn state_is_side_effect_free(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, _) = two_columns(&mut store)?;
    let graph = Graph::try_new(
        executor,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        DecodeCache::default(),
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
fn bare_scan_is_dense_and_filter_keeps_the_selection(
    #[case] sel: Sel,
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
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
        executor,
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
        executor,
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
fn take_values_are_read_once_per_plan(
    #[case] predicate: bool,
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
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
            executor,
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

/// A graph sharing a decode cache with one that already ran over the same plan reads nothing and
/// returns the same rows, even under a different selection.
#[rstest]
fn shared_decode_cache_skips_reads(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let (plan, expected) = fixture(&mut store)?;
    let decoded = DecodeCache::default();

    let rows = 0..ROWS;
    let all = Mask::new_true(ROWS as usize);
    let first = run_with(
        executor,
        &store,
        &plan,
        rows.clone(),
        all,
        delivery(Delivery::Fifo),
        decoded.clone(),
    )?;
    assert!(reads(&first.events) > 0);

    let mask = Sel::EveryOther.mask(ROWS as usize);
    let second = run_with(
        executor,
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(Delivery::Fifo),
        decoded,
    )?;
    assert_eq!(reads(&second.events), 0);
    assert_view(&expected, &rows, &mask, second.pieces)?;
    Ok(())
}

#[rstest]
fn compound_dictionary_predicate_uses_one_lookup(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter([0i32, 2, 10]).into_array();
    let codes = PrimitiveArray::from_iter((0..ROWS).map(|row| (row % 3) as u8)).into_array();
    let array = DictArray::try_new(codes, values)?.into_array();
    let expression = or(
        and(gt(root(), lit(0i32)), lt(root(), lit(4i32))),
        gt(root(), lit(8i32)),
    )
    .bind(array.dtype())?;
    let plan = EvalPlan::try_new(expression, lower(&store.flat(&array)?)?)?.into_plan();
    let result = run(
        executor,
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    assert_eq!(result.pieces.len(), 1);
    assert!(result.pieces[0].array.is::<Dict>());
    let expected = BoolArray::from_iter((0..ROWS).map(|row| row % 3 != 0)).into_array();
    assert_view(
        &expected,
        &(0..ROWS),
        &Mask::new_true(ROWS as usize),
        result.pieces,
    )?;
    Ok(())
}

#[rstest]
fn dictionary_predicate_preserves_nullable_codes(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter([0i32, 2, 10]).into_array();
    let codes = PrimitiveArray::from_option_iter(
        (0..ROWS).map(|row| (row % 4 != 0).then_some((row % 3) as u8)),
    )
    .into_array();
    let array = DictArray::try_new(codes, values)?.into_array();
    let expression = and(gt(root(), lit(0i32)), lt(root(), lit(4i32))).bind(array.dtype())?;
    let expected = BoolArray::from_iter(
        (0..ROWS).map(|row| (row % 4 != 0).then_some(row % 3 == 1)),
    )
    .into_array();
    let plan = EvalPlan::try_new(expression, lower(&store.flat(&array)?)?)?.into_plan();
    let result = run(
        executor,
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    assert_view(
        &expected,
        &(0..ROWS),
        &Mask::new_true(ROWS as usize),
        result.pieces,
    )?;
    Ok(())
}

#[rstest]
fn dictionary_predicate_does_not_evaluate_unused_fallible_values(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter([0i32, 2, 10]).into_array();
    let codes = PrimitiveArray::from_iter((0..ROWS).map(|row| 1 + (row % 2) as u8)).into_array();
    let array = DictArray::try_new(codes, values)?.into_array();
    let quotient = binary(Operator::Div, lit(100i32), root());
    let expression =
        and(gt(quotient, lit(20i32)), lt(root(), lit(100i32))).bind(array.dtype())?;
    let plan = EvalPlan::try_new(expression, lower(&store.flat(&array)?)?)?.into_plan();
    let result = run(
        executor,
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    let expected = BoolArray::from_iter((0..ROWS).map(|row| row % 2 == 0)).into_array();
    assert_view(
        &expected,
        &(0..ROWS),
        &Mask::new_true(ROWS as usize),
        result.pieces,
    )?;
    Ok(())
}

#[rstest]
fn dictionary_predicate_preserves_non_strict_null_results(
    #[values(Executor::Nodes, Executor::Pipelines)] executor: Executor,
) -> VortexResult<()> {
    let mut store = Store::default();
    let values = PrimitiveArray::from_iter([0i32, 2, 10]).into_array();
    let codes = PrimitiveArray::from_option_iter(
        (0..ROWS).map(|row| (row % 4 != 0).then_some((row % 3) as u8)),
    )
    .into_array();
    let array = DictArray::try_new(codes, values)?.into_array();
    let expression = or(is_null(root()), gt(root(), lit(1i32))).bind(array.dtype())?;
    let plan = EvalPlan::try_new(expression, lower(&store.flat(&array)?)?)?.into_plan();
    let result = run(
        executor,
        &store,
        &plan,
        0..ROWS,
        Mask::new_true(ROWS as usize),
        delivery(Delivery::Fifo),
    )?;
    let expected = BoolArray::from_iter(
        (0..ROWS).map(|row| Some(row % 4 == 0 || row % 3 != 0)),
    )
    .into_array();
    assert_view(
        &expected,
        &(0..ROWS),
        &Mask::new_true(ROWS as usize),
        result.pieces,
    )
}

#[rstest]
fn dictionary_boolean_fusion_respects_code_identity(
    #[values(false, true)] shared_codes: bool,
    #[values(Operator::And, Operator::Or)] operator: Operator,
) -> VortexResult<()> {
    let indices = [
        Some(0u8), Some(1), Some(2), None, Some(2), Some(1), Some(0), None,
    ];
    let codes = PrimitiveArray::from_option_iter(indices).into_array();
    let other_codes = if shared_codes {
        codes.clone()
    } else {
        PrimitiveArray::from_option_iter(indices.into_iter().rev()).into_array()
    };
    let left = DictArray::try_new(
        codes,
        BoolArray::from_iter([Some(false), Some(true), None]).into_array(),
    )?
    .into_array();
    let right = DictArray::try_new(
        other_codes,
        BoolArray::from_iter([Some(true), None, Some(false)]).into_array(),
    )?
    .into_array();
    let original = left.binary(right, operator)?;
    let mut ctx = SESSION.create_execution_ctx();
    let fused = fuse_dictionary_predicate(original.clone(), &mut ctx)?;
    assert_eq!(fused.is::<Dict>(), shared_codes);
    assert_arrays_eq!(fused, original, &mut ctx);
    Ok(())
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
    let output = run(
        Executor::Nodes,
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(order),
    )?;
    if mask.all_false() {
        assert_eq!(reads(&output.events), 0);
    }
    assert_view(&expected, &rows, &mask, output.pieces)
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
    let output = run(
        Executor::Nodes,
        &store,
        &plan,
        rows.clone(),
        mask.clone(),
        delivery(order),
    )?;
    assert_view(&expected, &rows, &mask, output.pieces)
}
