// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of the graph's scheduling, driven through hand-built nodes over in-memory segments.
//!
//! Every source emits its own row indices, so what reaches the root can be checked against a
//! sequence.

#![allow(clippy::cast_possible_truncation)]

use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::synthetic::Probe;
use super::synthetic::RowSource;
use super::synthetic::empty_segment;
use super::*;
use crate::test::SESSION;

#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// A batch returned by `compute`, as segment ids.
    Io(Vec<u32>),
    /// A result delivered for this segment.
    Delivered(u32),
    /// An array of this many rows reached the root.
    Piece(usize),
    /// A node yielded to the owner.
    Yield,
}

struct Run {
    arrays: Vec<ArrayRef>,
    events: Vec<Event>,
}

/// Drives a graph the way an owner does, answering reads with empty segments in the order
/// `pick` chooses, only while the graph waits.
fn run(
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    mut pick: impl FnMut(&[IoRequest]) -> usize,
) -> VortexResult<Run> {
    let mut graph =
        ExecGraph::try_new(SESSION.clone(), plan, rows, mask, 0, DecodeCache::default())?;
    let mut inflight: Vec<IoRequest> = Vec::new();
    let mut arrays = Vec::new();
    let mut events = Vec::new();
    loop {
        match graph.state() {
            ExecState::Done => break,
            ExecState::NeedsCompute => match graph.compute()? {
                ExecOutput::Piece(array) => {
                    assert!(!array.is_empty(), "the root never emits an empty array");
                    events.push(Event::Piece(array.len()));
                    arrays.push(array);
                }
                ExecOutput::NeedsIO(batch) => {
                    events.push(Event::Io(batch.iter().map(|r| *r.segment_id).collect()));
                    inflight.extend(batch);
                }
                ExecOutput::Yield => events.push(Event::Yield),
            },
            ExecState::Waiting => {
                if inflight.is_empty() {
                    return Err(vortex_err!("graph waits with no reads in flight"));
                }
                let request = inflight.remove(pick(&inflight));
                events.push(Event::Delivered(*request.segment_id));
                graph.set_io_result(request.id, empty_segment())?;
            }
        }
    }
    assert!(inflight.is_empty(), "graph finished with reads in flight");
    Ok(Run { arrays, events })
}

fn fifo(_: &[IoRequest]) -> usize {
    0
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

fn pieces(events: &[Event]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Piece(len) => Some(*len),
            _ => None,
        })
        .collect()
}

fn indices(values: impl IntoIterator<Item = u64>) -> ArrayRef {
    Buffer::from_iter(values).into_array()
}

/// Joins the arrays a run produced, in order, into one array.
fn join(dtype: &DType, arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(arrays, dtype.clone())?.into_array())
}

fn assert_rows(expected: ArrayRef, arrays: Vec<ArrayRef>) -> VortexResult<()> {
    let actual = join(expected.dtype(), arrays)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

fn segment(id: u32) -> Option<SegmentId> {
    Some(SegmentId::from(id))
}

#[rstest]
#[case::one_piece(vec![], false)]
#[case::many_pieces(vec![3, 1, 4, 1, 5], false)]
#[case::yielding(vec![3, 1, 4, 1, 5], true)]
#[case::oversized_pieces(vec![100], true)]
fn source_pieces_come_out_in_order(
    #[case] cut: Vec<usize>,
    #[case] yielding: bool,
) -> VortexResult<()> {
    let plan = RowSource::plan(20, cut, None, yielding, false);
    let mask = Mask::from_iter((0..10).map(|i| i != 4));
    let run = run(&plan, 5..15, mask, fifo)?;
    assert_rows(indices((5..15).filter(|i| *i != 9)), run.arrays)
}

/// A node runs only when its readiness rule holds, whatever order its inputs finish in.
#[rstest]
#[case::any(Ready::Any)]
#[case::closed_second(Ready::Closed(&[1]))]
#[case::all_closed(Ready::AllClosed)]
fn probe_runs_only_when_ready(#[case] ready: Ready) -> VortexResult<()> {
    let runs = Arc::new(Mutex::new(Vec::new()));
    let children = vec![
        RowSource::plan(10, vec![2, 2, 2], segment(0), false, false),
        RowSource::plan(10, vec![5], segment(1), false, false),
    ];
    let plan = Probe::plan(children, ready, Arc::clone(&runs));
    // The first child lands first, so under `Any` the node runs while the second is open.
    let run = run(&plan, 0..10, Mask::new_true(10), scripted(&[0, 1]))?;
    assert_rows(indices(0..10), run.arrays)?;

    let runs = runs.lock();
    assert!(!runs.is_empty());
    match ready {
        Ready::Any => assert!(runs[0] == [true, false], "ran as soon as port 0 closed"),
        Ready::Closed(_) => assert!(runs.iter().all(|ports| ports[1])),
        Ready::AllClosed => {
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0], [true, true]);
        }
    }
    Ok(())
}

/// A source that yields between pieces is run once per piece, and its pieces reach the root in
/// order, one per run.
#[test]
fn yielding_source_is_run_once_per_piece() -> VortexResult<()> {
    let plan = RowSource::plan(12, vec![4, 4, 4], None, true, false);
    let run = run(&plan, 0..12, Mask::new_true(12), fifo)?;
    assert_eq!(pieces(&run.events), [4, 4, 4]);
    assert_rows(indices(0..12), run.arrays)
}

#[test]
fn input_take_slices_at_the_boundary() -> VortexResult<()> {
    let mut input = Input {
        spawned: true,
        ..Default::default()
    };
    input.push(PrimitiveArray::from_iter(0..4i32).into_array());
    input.push(PrimitiveArray::from_iter(4..10i32).into_array());
    assert_eq!(input.available(), 10);

    let first = input.take(6)?;
    assert_eq!(first.iter().map(|a| a.len()).collect::<Vec<_>>(), [4, 2]);
    assert_eq!(input.available(), 4);
    assert!(!input.finished());

    input.closed = true;
    let rest = input.take_all();
    assert_eq!(rest.iter().map(|a| a.len()).collect::<Vec<_>>(), [4]);
    assert!(input.finished());
    assert!(input.take(1).is_err());
    Ok(())
}
