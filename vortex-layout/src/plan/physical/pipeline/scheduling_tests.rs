// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The runtime over hand-built sources: order, blocking, backpressure, and wakeups.

use std::collections::VecDeque;
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
use crate::plan::physical::PlanRef;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
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

impl Source for ManyReads {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if !self.wanted.is_empty() {
            while let Some(segment) = self.wanted.pop() {
                cx.request_read(segment);
            }
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if self.left == 0 || cx.has_bytes() {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Io)
        })
    }
}

impl Operator for ManyReads {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if let Some((segment, _bytes)) = cx.take_segment() {
            self.left -= 1;
            return Ok(Step::Final(indices([u64::from(*segment)])));
        }
        assert_eq!(self.left, 0, "computed while waiting for a read");
        Ok(Step::End)
    }
}

/// A source has all its reads in flight at once, and takes each segment's bytes as they arrive,
/// tagged with the segment.
#[rstest]
fn a_source_has_several_reads_in_flight(#[values(1, 3)] deliveries: usize) -> VortexResult<()> {
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
                assert!(
                    matches!(scan.step()?, Turn::Waiting),
                    "waiting for IO must not reissue reads or compute before bytes arrive"
                );
                // Deliver the last asked for first.
                for _ in 0..deliveries {
                    let read = inflight
                        .pop()
                        .ok_or_else(|| vortex_err!("no read in flight"))?;
                    scan.deliver(read.id, empty_segment())?;
                }
            }
            Turn::Done => break,
        }
    }
    assert_rows(indices([3, 2, 1]), arrays)
}

/// Holds an inlet batch until a read chosen from that batch has arrived.
struct ReadAfterInlet {
    batch: Option<ArrayRef>,
}

impl Source for ReadAfterInlet {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.batch.is_some() {
            return Ok(if cx.has_bytes() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Io)
            });
        }
        let mut inlet = cx.inlet(0);
        let Some(batch) = inlet.take() else {
            return Ok(if inlet.closed() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Inlet(0))
            });
        };
        cx.request_read(SegmentId::from(u32::try_from(batch.len())?));
        self.batch = Some(batch);
        Ok(SourceReady::Blocked(SourceBlocked::Io))
    }
}

impl Operator for ReadAfterInlet {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Some(batch) = self.batch.take() else {
            let inlet = cx.inlet(0);
            assert!(
                inlet.closed() && inlet.is_empty(),
                "computed before the inlet was ready"
            );
            return Ok(Step::End);
        };
        let (segment, _) = cx
            .take_segment()
            .ok_or_else(|| vortex_err!("computed before the read arrived"))?;
        assert_eq!(*segment, u32::try_from(batch.len())?);
        Ok(Step::Final(batch))
    }
}

#[test]
fn readiness_can_take_an_inlet_and_request_a_read() -> VortexResult<()> {
    let child = RowSource::plan(6, vec![1, 2, 3], segment(9), false);
    let plan = SyntheticPlan::new(
        row_dtype(),
        6,
        vec![child.clone()],
        move |rows, mask, compiler| {
            let chain = compiler
                .compile(&child, rows, mask)?
                .ok_or_else(|| vortex_err!("child has no rows"))?;
            Ok(Some(compiler.join_with(
                vec![chain],
                |_| 1,
                ReadAfterInlet { batch: None },
            )))
        },
    )
    .into_plan();
    let run = run(&plan, 0..6, Mask::new_true(6), fifo)?;
    assert_eq!(
        run.events,
        [Some(9), Some(1), None, Some(2), None, Some(3), None]
    );
    assert_rows(indices(0..6), run.arrays)
}

/// Asks for every segment at once, emits the id of the first to arrive, and finishes without
/// waiting for the rest.
struct FirstOf {
    wanted: Vec<SegmentId>,
    done: bool,
}

impl Source for FirstOf {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.done {
            return Ok(SourceReady::Ready);
        }
        if !self.wanted.is_empty() {
            while let Some(segment) = self.wanted.pop() {
                cx.request_read(segment);
            }
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if cx.has_bytes() {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Io)
        })
    }
}

impl Operator for FirstOf {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        let (segment, _) = cx
            .take_segment()
            .ok_or_else(|| vortex_err!("FirstOf computed before a read arrived"))?;
        self.done = true;
        Ok(Step::Final(indices([u64::from(*segment)])))
    }
}

/// A source that finishes with reads in flight has them cancelled: their bytes reach neither it
/// nor the next split's pipeline, which reuses its id, and the scan is done without them.
#[test]
fn finishing_cancels_reads_in_flight() -> VortexResult<()> {
    let plan = SyntheticPlan::new(row_dtype(), 2, Vec::new(), |rows, _, _| {
        let base = 10 * (u32::try_from(rows.start)? + 1);
        Ok(Some(Chain::new(FirstOf {
            wanted: (base..base + 3).map(SegmentId::from).collect(),
            done: false,
        })))
    })
    .into_plan();
    let splits = vec![Split::all(0..1), Split::all(1..2)];
    let mut scan = Scan::try_new(SESSION.clone(), plan, splits)?;
    let mut by_split: Vec<Vec<ArrayRef>> = vec![Vec::new(); 2];
    let mut inflight = Vec::new();
    loop {
        match scan.step()? {
            Turn::Read(read) => inflight.push(read),
            Turn::Output(split, array) => by_split[split].push(array),
            Turn::Waiting => {
                let read = inflight.remove(0);
                scan.deliver(read.id, empty_segment())?;
            }
            Turn::Done => break,
        }
    }
    assert!(
        !inflight.is_empty(),
        "the second split's other reads are never delivered"
    );
    assert_eq!(
        scan.awaited_reads(),
        0,
        "no pipeline waits for the reads left over"
    );
    assert_rows(indices([12]), std::mem::take(&mut by_split[0]))?;
    assert_rows(indices([22]), std::mem::take(&mut by_split[1]))?;
    assert_eq!(scan.live_ports(), 0);
    Ok(())
}

/// Emits pairs of batches, requiring readiness only between pairs.
#[derive(Default)]
struct Pairs {
    next: u64,
    ready: bool,
}

impl Source for Pairs {
    fn ready(&mut self, _cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        assert!(!self.ready, "More must skip readiness");
        self.ready = true;
        Ok(SourceReady::Ready)
    }
}

impl Operator for Pairs {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        assert!(matches!(input, Input::None));
        assert!(self.ready, "Final must check readiness again");
        if self.next == 6 {
            return Ok(Step::End);
        }
        let batch = indices([self.next]);
        self.next += 1;
        if self.next % 2 == 1 {
            Ok(Step::More(batch))
        } else {
            self.ready = false;
            Ok(Step::Final(batch))
        }
    }
}

struct Copies {
    copies: usize,
    left: usize,
    batch: Option<ArrayRef>,
    ended: bool,
}

impl Copies {
    fn new(copies: usize) -> Self {
        Self {
            copies,
            left: 0,
            batch: None,
            ended: false,
        }
    }
}

impl Operator for Copies {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        assert!(!self.ended, "a finished stage cannot run again");
        match input {
            Input::Batch(batch) => {
                assert!(
                    self.batch.is_none(),
                    "new input arrived before draining More"
                );
                self.batch = Some(batch);
                self.left = self.copies;
            }
            Input::None => assert!(self.batch.is_some(), "continued after Final"),
            Input::End => {
                assert!(self.batch.is_none(), "End arrived before draining More");
                self.ended = true;
                return Ok(Step::End);
            }
        }
        self.left -= 1;
        if self.left == 0 {
            Ok(Step::Final(self.batch.take().expect("batch is present")))
        } else {
            Ok(Step::More(
                self.batch.as_ref().expect("batch is present").clone(),
            ))
        }
    }
}

/// Buffers every input, then flushes with either Final or End terminating the continuation.
struct Buffered {
    batches: VecDeque<ArrayRef>,
    flushing: bool,
    final_batch: bool,
}

impl Operator for Buffered {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Batch(batch) => {
                assert!(!self.flushing);
                self.batches.push_back(batch);
                return Ok(Step::End);
            }
            Input::End => {
                assert!(!self.flushing, "End must be supplied exactly once");
                self.flushing = true;
            }
            Input::None => assert!(self.flushing),
        }
        Ok(match self.batches.pop_front() {
            Some(batch) if self.final_batch && self.batches.is_empty() => Step::Final(batch),
            Some(batch) => Step::More(batch),
            None => Step::End,
        })
    }
}

#[rstest]
fn more_drains_before_new_input(#[values(1, 2)] capacity: usize) -> VortexResult<()> {
    let plan = SyntheticPlan::new(row_dtype(), 36, Vec::new(), move |_, _, _| {
        let chain = Chain::new(Pairs::default())
            .with(Copies::new(2))
            .with(Copies::new(3));
        Ok(Some(chain))
    })
    .into_plan();
    let plan = Probe::plan(vec![plan], capacity, Arc::default());
    let result = run(&plan, 0..36, Mask::new_true(36), fifo)?;
    assert_eq!(result.arrays.len(), 36);
    assert_rows(indices((0..6).flat_map(|row| [row; 6])), result.arrays)
}

#[rstest]
fn more_flushes_before_ending_downstream(
    #[values(1, 2)] capacity: usize,
    #[values(false, true)] final_batch: bool,
) -> VortexResult<()> {
    let plan = SyntheticPlan::new(row_dtype(), 36, Vec::new(), move |_, _, _| {
        let chain = Chain::new(Pairs::default())
            .with(Buffered {
                batches: VecDeque::new(),
                flushing: false,
                final_batch,
            })
            .with(Copies::new(2))
            .with(Copies::new(3))
            .with(Buffered {
                batches: VecDeque::new(),
                flushing: false,
                final_batch,
            });
        Ok(Some(chain))
    })
    .into_plan();
    let plan = Probe::plan(vec![plan], capacity, Arc::default());
    let result = run(&plan, 0..36, Mask::new_true(36), fifo)?;
    assert_eq!(result.arrays.len(), 36);
    assert_rows(indices((0..6).flat_map(|row| [row; 6])), result.arrays)
}

/// Sources that break the scheduling contract, which debug builds catch.
#[cfg(debug_assertions)]
mod contract {
    use super::*;

    /// Emits `left` one-row batches.
    struct Repeat {
        left: usize,
    }

    impl Source for Repeat {}

    impl Operator for Repeat {
        fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
            Ok(match self.left.checked_sub(1) {
                Some(left) => {
                    self.left = left;
                    Step::Final(indices([0]))
                }
                None => Step::End,
            })
        }
    }

    /// Waits on its one-batch inlet whatever it holds, so once the inlet is full nothing wakes it.
    struct WaitsWhenFull;

    impl Source for WaitsWhenFull {
        fn ready(&mut self, _cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
            Ok(SourceReady::Blocked(SourceBlocked::Inlet(0)))
        }
    }

    impl Operator for WaitsWhenFull {
        fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
            Err(vortex_err!("a blocked source must not compute"))
        }
    }

    /// Debug builds catch a source waiting on an inlet its sink can no longer wake it through, at
    /// that source, instead of reporting the scan stuck.
    #[test]
    #[should_panic(expected = "which is full")]
    fn waiting_on_a_full_inlet_is_caught() {
        let plan = SyntheticPlan::new(row_dtype(), 2, Vec::new(), |_, _, compiler| {
            Ok(Some(compiler.join_with(
                vec![Chain::new(Repeat { left: 2 })],
                |_| 1,
                WaitsWhenFull,
            )))
        })
        .into_plan();
        let mut scan = Scan::try_new(SESSION.clone(), plan, vec![Split::all(0..2)]).unwrap();
        while !matches!(scan.step().unwrap(), Turn::Done) {}
    }
}
