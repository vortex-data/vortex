// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`ScanPool`] over one file's planner-driven scan.
//!
//! Opening the pool prepares the scan and runs its root planner until it has spawned the planner
//! of every part, the file's layout splits clipped to the scanned rows. The root only prunes the
//! scan, when its filter is dynamic, and cuts parts, so opening reads zone maps at most. Every
//! part then waits in a channel, and workers take them: each part runs on a planning run of its
//! own, inline on the worker's thread, which waits only for its reads. With a filter, a part's
//! pruning plans first, continuing into the chunks of the rows pruning kept; without, the chunks
//! directly.
//!
//! Parts are layout splits, so one worker exports a layout chunk, and the dictionary it carries,
//! on its own, which lets a host evaluate an expression over that dictionary once.

use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_channel::Receiver;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::pool::ScanPool;
use vortex_scan::selection::Selection;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ScanBuilder;

/// The parts each worker runs beside each other, so one part's reads overlap another's compute.
const PARTS_IN_FLIGHT_PER_WORKER: usize = 1;

/// A unit of a [`PartPool`].
enum Work {
    /// A part's planner, to run on a run of its own.
    Part(WorkScope, Box<dyn Planner>),
    /// A batch the root produced itself.
    Batch(ArrayRef),
}

/// A pool whose workers share the parts of one file's planner-driven scan through a channel.
pub struct PartPool {
    dtype: DType,
    io: Arc<dyn IoService>,
    receiver: Receiver<Work>,
    /// The rows of the scan, for progress.
    total_rows: u64,
    /// The rows of the parts workers have taken.
    rows_taken: AtomicU64,
}

impl PartPool {
    /// Prepares `builder` and runs its root until every part is in the pool.
    ///
    /// `splits` are the file's layout split boundaries, as a layout scan of the same fields cuts
    /// it. `dynamic` says the filter's value changes while the scan runs, as a top-N's bound
    /// does, in which case the root prunes the scan as a whole before cutting parts.
    pub async fn open(builder: ScanBuilder, splits: &[u64], dynamic: bool) -> VortexResult<Self> {
        let dtype = builder.dtype().clone();
        let scan = Arc::new(builder.prepare()?);
        let io = scan.io();
        let rows = scan.rows().clone();
        let (sender, receiver) = async_channel::unbounded();
        let mut root = Run::new();
        root.admit(
            compose(&scan, splits, dynamic),
            WorkScope {
                file_ordinal: 0,
                rows: rows.clone(),
            },
            io.session(),
        );
        loop {
            let work = match root.advance()? {
                Progress::Batch(batch) => Work::Batch(batch.array),
                Progress::Spawn(scope, part) => Work::Part(scope, part),
                Progress::RootDone(_) if root.live_roots() > 0 => continue,
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => {
                    future::poll_fn(|cx| root.poll_completion(cx)).await?;
                    continue;
                }
            };
            // The channel is unbounded and this holds a receiver, so sending cannot fail.
            drop(sender.try_send(work));
        }
        // Dropping the only sender closes the channel once workers have taken every part.
        drop(sender);
        Ok(Self {
            dtype,
            io,
            receiver,
            total_rows: rows.end - rows.start,
            rows_taken: AtomicU64::new(0),
        })
    }

    /// The arrays of one unit, taken by a worker.
    fn run(&self, work: Work) -> BoxStream<'static, VortexResult<ArrayRef>> {
        match work {
            Work::Batch(array) => stream::once(future::ready(Ok(array))).boxed(),
            Work::Part(scope, part) => {
                self.rows_taken
                    .fetch_add(scope.rows.end - scope.rows.start, Ordering::Relaxed);
                run_part(Arc::clone(&self.io), scope, part)
            }
        }
    }
}

impl ScanPool for PartPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let receiver = self.receiver.clone();
        // The worker holds the pool, so the pool is dropped only after its last worker.
        let arrays = receiver
            .map(move |work| self.run(work))
            .flatten_unordered(PARTS_IN_FLIGHT_PER_WORKER);
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }

    fn progress(&self) -> f64 {
        if self.total_rows == 0 || self.is_exhausted() {
            return 1.0;
        }
        (self.rows_taken.load(Ordering::Relaxed) as f64 / self.total_rows as f64).clamp(0.0, 1.0)
    }

    fn is_exhausted(&self) -> bool {
        self.receiver.is_empty()
    }
}

/// Runs one part on a run of its own, returning its batches in order.
fn run_part(
    io: Arc<dyn IoService>,
    scope: WorkScope,
    part: Box<dyn Planner>,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    let mut run = Run::new();
    run.admit(part, scope, io.session());
    stream::unfold(Some((run, io)), |state| async move {
        let (mut run, io) = state?;
        loop {
            let progress = match run.advance() {
                Ok(progress) => progress,
                Err(err) => return Some((Err(err), None)),
            };
            match progress {
                Progress::Batch(batch) => return Some((Ok(batch.array), Some((run, io)))),
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, io.session());
                }
                Progress::RootDone(_) => {}
                Progress::Idle => return None,
                Progress::Waiting => {
                    if let Err(err) = future::poll_fn(|cx| run.poll_completion(cx)).await {
                        return Some((Err(err), None));
                    }
                }
            }
        }
    })
    .boxed()
}

/// The planners of a scan, composed: with a dynamic filter, its pruning plans first over the
/// scan's rows, continuing into the parts of the rows they keep; otherwise, the parts directly.
fn compose(scan: &Arc<PreparedScan>, splits: &[u64], dynamic: bool) -> Box<dyn Planner> {
    let rows = scan.rows().clone();
    let parts = {
        let scan = Arc::clone(scan);
        let parts = part_ranges(rows.clone(), splits);
        let first_row = rows.start;
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            Box::new(PartPlanner::new(scan, parts, first_row, kept))
        }
    };
    if dynamic && scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(parts)))
    } else {
        parts(None)
    }
}

/// Spawns a planner per part of a scan's rows that zone pruning left something of, in row order.
///
/// With a filter, each part prunes over its own rows before its chunks, on the worker that takes
/// it, like the per-split pruning of a layout scan, so the pruning of a file's parts runs in
/// parallel. A dynamic filter, such as the bound a top-N pushes down, also prunes the scan as a
/// whole before any part is taken, so a file whose zones it rules out gives workers nothing. It
/// has tightened by the time a later part is taken, so pruning that part again skips more, from
/// the zone maps the scan-wide pruning read.
struct PartPlanner {
    scan: Arc<PreparedScan>,
    selection: Selection,
    parts: std::vec::IntoIter<Range<u64>>,
    /// The first row of `kept`.
    first_row: u64,
    /// The rows scan-wide pruning kept, from `first_row`, or `None` when nothing was pruned.
    kept: Option<Mask>,
}

impl PartPlanner {
    fn new(
        scan: Arc<PreparedScan>,
        parts: Vec<Range<u64>>,
        first_row: u64,
        kept: Option<Mask>,
    ) -> Self {
        Self {
            selection: scan.selection().clone(),
            scan,
            parts: parts.into_iter(),
            first_row,
            kept,
        }
    }

    /// The rows of `part` that scan-wide pruning kept, or `None` when nothing was pruned.
    fn kept_in(&self, part: &Range<u64>) -> Option<Mask> {
        let kept = self.kept.as_ref()?;
        let offset =
            |row: u64| usize::try_from(row - self.first_row).vortex_expect("a part fits usize");
        Some(kept.slice(offset(part.start)..offset(part.end)))
    }
}

impl IoConsumer for PartPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for PartPlanner {
    fn state(&self) -> State {
        if self.parts.as_slice().is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        while let Some(part) = self.parts.next() {
            let kept = self.kept_in(&part);
            if kept.as_ref().is_some_and(Mask::all_false) {
                continue;
            }
            let chunks = {
                let scan = Arc::clone(&self.scan);
                let selection = self.selection.clone();
                let part = part.clone();
                move |again: Option<Mask>| -> Box<dyn Planner> {
                    let kept = match (kept, again) {
                        (Some(kept), Some(again)) => Some(kept & &again),
                        (kept, again) => kept.or(again),
                    };
                    Box::new(ChunkPlanner::new(scan, part, selection, kept))
                }
            };
            let planner: Box<dyn Planner> = if self.scan.filter().is_some() {
                Box::new(PrunePlanner::new(
                    Arc::clone(&self.scan),
                    part.clone(),
                    Box::new(chunks),
                ))
            } else {
                chunks(None)
            };
            let scope = WorkScope {
                file_ordinal: 0,
                rows: part,
            };
            return Ok(PlannerOutput::Spawn(scope, planner));
        }
        Ok(PlannerOutput::Done)
    }
}

/// The ranges between adjacent `splits`, clipped to `rows`, leaving out empty ones.
fn part_ranges(rows: Range<u64>, splits: &[u64]) -> Vec<Range<u64>> {
    splits
        .windows(2)
        .map(|w| w[0].max(rows.start)..w[1].min(rows.end))
        .filter(|part| part.start < part.end)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::part_ranges;

    #[rstest]
    #[case::whole(0..30, &[0, 10, 20, 30], vec![0..10, 10..20, 20..30])]
    #[case::clipped(5..25, &[0, 10, 20, 30], vec![5..10, 10..20, 20..25])]
    #[case::inside_one_split(12..18, &[0, 10, 20, 30], vec![12..18])]
    #[case::empty(5..5, &[0, 10], vec![])]
    fn parts_follow_splits(
        #[case] rows: Range<u64>,
        #[case] splits: &[u64],
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(part_ranges(rows, splits), expected);
    }
}
