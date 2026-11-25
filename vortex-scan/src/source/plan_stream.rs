// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and [`Run`] driver
//! behind [`ScanExec`] and its [`ScanStream`](crate::stream::ScanStream)s.
//!
//! When a scan is made, it prepares its child once and cuts the child's rows into parts with
//! [`split_points`](crate::split::split_points), which the V1 executor shares. That reads nothing:
//! the child's zone maps are read by the parts, each over its own rows.
//! [`split`](crate::DataSourceScan::split) deals the parts out to the pieces, cutting the scan
//! into at least as many parts as pieces when a host has more threads than the scan has parts.
//!
//! Every stream of an opened scan takes parts from the one queue, and the scan is exhausted once
//! the last part is taken. A part is planned when a stream takes it: with a filter, it first
//! prunes its rows against the zone maps, applying each dynamic conjunct as it stands then, such
//! as the bound a top-N pushes down. Each stream owns a [`Run`], driven on the task that polls the
//! stream, and runs one part on it at a time: the part's chunks run side by side, so one chunk's
//! reads overlap another's compute, batches are returned as they are produced, and reads are
//! awaited with [`Run::poll_completion`]. Nothing is spawned, so a host that polls a stream with
//! `block_on` runs the scan's compute on its own thread.
//!
//! A limit is one [`RowLimit`] for the whole scan, shared by its pieces: streams stop taking parts
//! once it is spent, which also exhausts the scan, and each stream's output is trimmed to it.
//!
//! Every batch is cut at the chunk boundaries of its columns, by
//! [`ScanExec::with_batch_cuts`].

use std::collections::VecDeque;
use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoService;

use super::plan::ScanState;
use crate::execution::part_planner;
use crate::limit::RowLimit;
use crate::limit::limit_rows;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::WorkScope;
use crate::prepare::PreparedScan;
use crate::split::group_parts;
use crate::split::parts_between;
use crate::stream::Claims;
use crate::stream::ExhaustedFlag;
use crate::stream::ScanExec;

/// The parts of a scan, or of a piece of one, cut when the scan was made.
pub(super) struct Planned {
    /// The child's prepared scan, which every part plans over, or `None` for a scan that reads
    /// nothing.
    prepared: Option<Arc<Prepared>>,
    /// The parts, in row order.
    parts: Vec<Range<u64>>,
    /// Whether the parts cover all of the scan's rows, rather than being a piece of it.
    whole: bool,
}

/// A child's prepared scan and the I/O its parts read through.
struct Prepared {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
}

impl Planned {
    /// Prepares the scan of `rows` of the child, if it reads any, and cuts them into parts.
    /// Reads nothing.
    pub(super) fn new(state: &ScanState, rows: Option<Range<u64>>) -> VortexResult<Self> {
        let Some(rows) = rows.filter(|rows| !rows.is_empty()) else {
            return Ok(Self::empty());
        };
        let scan = Arc::new(state.builder(rows).prepare()?);
        let parts = parts_between(&scan.part_points(1)?);
        let io = scan.io();
        Ok(Self {
            prepared: Some(Arc::new(Prepared { scan, io })),
            parts,
            whole: true,
        })
    }

    fn empty() -> Self {
        Self {
            prepared: None,
            parts: Vec::new(),
            whole: true,
        }
    }

    /// The number of parts.
    pub(super) fn len(&self) -> usize {
        self.parts.len()
    }

    /// Deals the parts out to at most `n` pieces of contiguous parts. A whole scan with fewer
    /// than `n` parts is cut again into at least `n` first, where its rows hold enough blocks.
    /// Never empty.
    pub(super) fn split(self, n: usize) -> VortexResult<Vec<Self>> {
        let Some(prepared) = self.prepared else {
            return Ok(vec![Self::empty()]);
        };
        let parts = if self.whole && self.parts.len() < n {
            parts_between(&prepared.scan.part_points(n)?)
        } else {
            self.parts
        };
        Ok(group_parts(parts, n)
            .into_iter()
            .map(|parts| Self {
                prepared: Some(Arc::clone(&prepared)),
                parts,
                whole: false,
            })
            .collect())
    }
}

/// Opens a scan onto the parts it `planned`. A scan with no parts opens exhausted.
pub(super) fn open(state: &ScanState, planned: Planned) -> ScanExec {
    let dtype = state.dtype.clone();
    let limit = state.limit.clone();
    let exhausted = ExhaustedFlag::new();
    let queue = Arc::new(Queue {
        total: planned.parts.len(),
        parts: Mutex::new(planned.parts.into()),
        taken: AtomicUsize::new(0),
        exhausted: exhausted.clone(),
    });
    let progress = Arc::clone(&queue);
    let Some(prepared) = planned.prepared.filter(|_| queue.total > 0) else {
        exhausted.set();
        return ScanExec::new(dtype, exhausted, || stream::empty().boxed())
            .with_progress(move || progress.claims())
            .with_batch_cuts();
    };
    ScanExec::new(dtype, exhausted, move || {
        let worker = Worker {
            prepared: Arc::clone(&prepared),
            queue: Arc::clone(&queue),
            limit: limit.clone(),
            run: Run::new(),
        };
        let arrays = stream::unfold(Some(worker), |worker| async move {
            let mut worker = worker?;
            match worker.next_array().await {
                Ok(Some(array)) => Some((Ok(array), Some(worker))),
                Ok(None) => None,
                Err(err) => Some((Err(err), None)),
            }
        })
        .boxed();
        limited(arrays, limit.clone())
    })
    .with_progress(move || progress.claims())
    .with_batch_cuts()
}

/// The queue of a scan, which every stream of it takes from.
struct Queue {
    /// The parts no stream has taken yet, in row order.
    parts: Mutex<VecDeque<Range<u64>>>,
    /// The parts the queue started with.
    total: usize,
    /// The parts streams have taken.
    taken: AtomicUsize,
    /// Set once the last part is taken, or the scan's limit is spent.
    exhausted: ExhaustedFlag,
}

impl Queue {
    /// Takes the next part, or `None` once every one has been taken, and marks the scan exhausted
    /// when it takes the last.
    fn take(&self) -> Option<Range<u64>> {
        let mut parts = self.parts.lock();
        let next = parts.pop_front()?;
        if parts.is_empty() {
            self.exhausted.set();
        }
        drop(parts);
        self.taken.fetch_add(1, Ordering::Relaxed);
        Some(next)
    }

    /// How many of the queue's parts streams have taken, out of how many in all.
    fn claims(&self) -> Claims {
        Claims {
            claimed: self.taken.load(Ordering::Relaxed).min(self.total),
            total: self.total,
        }
    }
}

/// One stream's worker: a run of its own, fed one part at a time from the scan's queue, and driven
/// on the task that polls the stream.
struct Worker {
    prepared: Arc<Prepared>,
    queue: Arc<Queue>,
    limit: Option<RowLimit>,
    run: Run,
}

impl Worker {
    /// The worker's next array, or `None` once the scan has no more work for it.
    async fn next_array(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            if self.run.live_roots() == 0 {
                if self.limit.as_ref().is_some_and(RowLimit::is_exhausted) {
                    self.queue.exhausted.set();
                    return Ok(None);
                }
                let Some(part) = self.queue.take() else {
                    return Ok(None);
                };
                let planner = part_planner(&self.prepared.scan, part.clone());
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: part,
                };
                self.run.admit(planner, scope, self.prepared.io.session());
            }
            match self.run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(scope, planner) => {
                    self.run.admit(planner, scope, self.prepared.io.session());
                }
                Progress::RootDone(_) | Progress::Idle => {}
                Progress::Waiting => {
                    future::poll_fn(|cx| self.run.poll_completion(cx)).await?;
                }
            }
        }
    }
}

/// `arrays`, trimmed to the scan's limit.
fn limited(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: Option<RowLimit>,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    match limit {
        Some(limit) => limit_rows(arrays, limit),
        None => arrays,
    }
}
