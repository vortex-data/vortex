// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The runtime over hand-built sources: order, blocking, backpressure, and wakeups.

use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::synthetic::Probe;
use super::synthetic::ProbeRun;
use super::synthetic::RowSource;
use super::synthetic::SyntheticPlan;
use super::synthetic::empty_segment;
use super::synthetic::row_dtype;
use super::*;
use crate::plan::PlanRef;
use crate::segments::SegmentId;
use crate::test::SESSION;

/// The arrays a scan produced, the segments it asked for in order, and when each array came
/// out relative to the deliveries.
struct Run {
    arrays: Vec<ArrayRef>,
    /// `Some(segment)` for a delivery, `None` for an array.
    events: Vec<Option<u32>>,
}

/// Drives one split of `plan`, delivering reads in the order `pick` chooses, only while the
/// scan waits.
fn run(
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    mut pick: impl FnMut(&[ReadRequest]) -> usize,
) -> VortexResult<Run> {
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), vec![Split { rows, mask }])?;
    let mut inflight = Vec::new();
    let mut arrays = Vec::new();
    let mut events = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => inflight.push(read),
            Turn::Output(_, array) => {
                events.push(None);
                arrays.push(array);
            }
            Turn::Waiting => {
                if inflight.is_empty() {
                    return Err(vortex_err!("scan waits with no reads in flight"));
                }
                let read = inflight.remove(pick(&inflight));
                events.push(Some(*read.segment_id));
                scan.deliver(read.id, empty_segment())?;
            }
            Turn::Done => break,
        }
    }
    assert_eq!(scan.live_ports(), 0, "a finished scan holds no port");
    Ok(Run { arrays, events })
}

fn fifo(_: &[ReadRequest]) -> usize {
    0
}

/// Delivers segments in the given order.
fn scripted(order: &[u32]) -> impl FnMut(&[ReadRequest]) -> usize + '_ {
    let mut next = order.iter();
    move |inflight| {
        let segment = *next.next().expect("script covers every read");
        inflight
            .iter()
            .position(|read| *read.segment_id == segment)
            .expect("scripted segment is in flight")
    }
}

fn indices(values: impl IntoIterator<Item = u64>) -> ArrayRef {
    Buffer::from_iter(values).into_array()
}

fn assert_rows(expected: ArrayRef, arrays: Vec<ArrayRef>) -> VortexResult<()> {
    let actual = ChunkedArray::try_new(arrays, row_dtype())?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

fn segment(id: u32) -> Option<SegmentId> {
    Some(SegmentId::from(id))
}

/// A source's pieces come out in order, one array each, after its read if it has one.
#[rstest]
fn source_pieces_come_out_in_order(#[values(None, Some(7))] read: Option<u32>) -> VortexResult<()> {
    let plan = RowSource::plan(10, vec![3, 3, 4], read.and_then(segment), false);
    let run = run(&plan, 0..10, Mask::new_true(10), fifo)?;
    assert_eq!(
        run.arrays.iter().map(|a| a.len()).collect::<Vec<_>>(),
        [3, 3, 4]
    );
    if let Some(read) = read {
        assert_eq!(run.events.first(), Some(&Some(read)));
    }
    assert_rows(indices(0..10), run.arrays)
}

/// A selected source emits only the selected rows; a dense one emits every row.
#[rstest]
fn masks_are_applied_by_the_source(#[values(false, true)] dense: bool) -> VortexResult<()> {
    let plan = RowSource::plan(8, Vec::new(), None, dense);
    let mask = Mask::from_iter((0..8).map(|row| row % 2 == 0));
    let run = run(&plan, 0..8, mask, fifo)?;
    let expected = if dense {
        indices(0..8)
    } else {
        indices((0..8).step_by(2))
    };
    assert_rows(expected, run.arrays)
}

/// A reader runs only when an inlet has changed: every run of the probe after its first finds
/// a batch it has not seen, or an inlet newly closed.
#[rstest]
fn a_reader_runs_only_when_an_inlet_changed(
    #[values(&[1, 2], &[2, 1])] order: &[u32],
) -> VortexResult<()> {
    let runs: Arc<Mutex<Vec<ProbeRun>>> = Arc::default();
    let plan = Probe::plan(
        vec![
            RowSource::plan(6, vec![2, 2, 2], segment(1), false),
            RowSource::plan(6, vec![6], segment(2), false),
        ],
        8,
        Arc::clone(&runs),
    );
    let run = run(&plan, 0..6, Mask::new_true(6), scripted(order))?;
    assert_rows(indices(0..6), run.arrays)?;
    let runs = runs.lock();
    for pair in runs.windows(2) {
        assert_ne!(pair[0], pair[1], "a run saw nothing new: {runs:?}");
    }
    Ok(())
}

/// A writer is blocked once its reader's inlet holds as many batches as the reader allows, so
/// a fast source never runs far ahead of a slow reader.
#[test]
fn a_full_inlet_blocks_its_writer() -> VortexResult<()> {
    let runs: Arc<Mutex<Vec<ProbeRun>>> = Arc::default();
    let plan = Probe::plan(
        vec![RowSource::plan(20, vec![1; 20], None, false)],
        2,
        Arc::clone(&runs),
    );
    let run = run(&plan, 0..20, Mask::new_true(20), fifo)?;
    assert_rows(indices(0..20), run.arrays)?;
    // An empty inlet always has room, and the batch in flight may land in a full one.
    let deepest = runs.lock().iter().map(|run| run[0].0).max().unwrap_or(0);
    assert!(deepest <= 3, "inlet held {deepest} batches");
    Ok(())
}

/// Several splits of one scan each produce their own rows.
#[test]
fn splits_produce_their_own_rows() -> VortexResult<()> {
    let plan = RowSource::plan(12, vec![2; 6], segment(3), false);
    let splits = vec![Split::all(0..4), Split::all(4..9), Split::all(9..12)];
    let mut scan = Scan::try_new(SESSION.clone(), plan, splits)?.with_max_active(2);
    let mut by_split: Vec<Vec<ArrayRef>> = vec![Vec::new(); 3];
    let mut inflight = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => inflight.push(read),
            Turn::Output(split, array) => by_split[split].push(array),
            Turn::Waiting => {
                let read: ReadRequest = inflight.remove(0);
                scan.deliver(read.id, empty_segment())?;
            }
            Turn::Done => break,
        }
    }
    for (split, rows) in [(0, 0..4), (1, 4..9), (2, 9..12)] {
        assert_rows(indices(rows), std::mem::take(&mut by_split[split]))?;
    }
    Ok(())
}

/// Asks for every segment at once and emits each segment's id as it arrives.
struct ManyReads {
    wanted: Vec<SegmentId>,
    left: usize,
}

impl Operator for ManyReads {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if let Some((segment, _bytes)) = cx.take_segment() {
            self.left -= 1;
            return Ok(Step::More(indices([u64::from(*segment)])));
        }
        Ok(if self.left == 0 {
            Step::Finished
        } else {
            Step::Blocked(Blocked::Io)
        })
    }
}

impl Source for ManyReads {
    fn request(&mut self) -> Option<SegmentId> {
        self.wanted.pop()
    }
}

/// A source has all its reads in flight at once, and takes each segment's bytes as they arrive,
/// tagged with the segment.
#[test]
fn a_source_has_several_reads_in_flight() -> VortexResult<()> {
    let plan = SyntheticPlan::new(row_dtype(), 3, Vec::new(), |_, _, _| {
        Ok(Some(Chain::new(ManyReads {
            wanted: [3_u32, 2, 1].into_iter().map(SegmentId::from).collect(),
            left: 3,
        })))
    })
    .into_plan();
    let mut scan = Scan::try_new(SESSION.clone(), plan, vec![Split::all(0..3)])?;
    let mut inflight = Vec::new();
    let mut arrays = Vec::new();
    let mut first_wait = true;
    loop {
        match scan.step()? {
            Turn::Read(read) => inflight.push(read),
            Turn::Output(_, array) => arrays.push(array),
            Turn::Waiting => {
                if first_wait {
                    assert_eq!(
                        inflight.len(),
                        3,
                        "every read is asked for before any arrives"
                    );
                    first_wait = false;
                }
                // Deliver the last asked for first.
                let read = inflight
                    .pop()
                    .ok_or_else(|| vortex_err!("no read in flight"))?;
                scan.deliver(read.id, empty_segment())?;
            }
            Turn::Done => break,
        }
    }
    assert_rows(indices([3, 2, 1]), arrays)
}
