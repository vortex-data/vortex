// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and [`Run`] driver
//! behind [`ScanExec`] and its [`ScanStream`](crate::stream::ScanStream)s.
//!
//! When a scan is made, it prepares its child once, prunes the child's rows by the zone maps of
//! its filter's static conjuncts, and cuts the rows into parts with
//! [`split_points`](crate::split::split_points), which the V1 executor shares, measuring each
//! part by the rows that survive. [`split`](crate::DataSourceScan::split) cuts the scan again into
//! at least as many parts as pieces and deals them out to the pieces.
//!
//! Every stream of an opened scan shares one [`Work`]: it takes a part from the queue and claims
//! its rows a few column chunks at a time, and once the queue is empty it steals the back half of
//! the unclaimed rest of another stream's part, cut at a column chunk boundary. The scan is
//! exhausted once nothing is left to take or steal. Claimed rows are planned when a stream claims
//! them, over the rows the static conjuncts kept: with a dynamic conjunct, they are first pruned
//! against the zone maps, applying the conjunct as it stands then, such as the bound a top-N
//! pushes down. Each stream owns a [`Run`], driven on the task that polls the stream: a claim's
//! chunks run side by side, so one chunk's reads overlap another's compute, the next claim is
//! admitted while the last chunk of the one before runs, batches are returned as they are
//! produced, and reads are awaited with [`Run::poll_completion`]. Nothing is spawned, so a host
//! that polls a stream with `block_on` runs the scan's compute on its own thread.
//!
//! A limit is one [`RowLimit`] for the whole scan, shared by its pieces: streams stop claiming rows
//! once it is spent, which also exhausts the scan, and each stream's output is trimmed to it.
//!
//! Every batch is cut at the chunk boundaries of its columns, by
//! [`ScanExec::with_batch_cuts`].

use std::future;
use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoService;
use vortex_mask::Mask;

use super::plan::ScanState;
use super::work::Owned;
use super::work::Work;
use crate::execution::chunk_boundaries;
use crate::execution::part_planner;
use crate::execution::static_kept;
use crate::limit::RowLimit;
use crate::limit::limit_rows;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::WorkScope;
use crate::prepare::PreparedScan;
use crate::split::group_parts;
use crate::split::kept_ranges;
use crate::split::parts_between;
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

/// A child's prepared scan, the I/O its parts read through, the rows of the scan its filter's
/// static conjuncts keep, and the chunk boundaries of the columns it reads, where its parts may
/// be cut while they run.
struct Prepared {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    kept: Option<Mask>,
    boundaries: Vec<u64>,
}

impl Prepared {
    /// The rows of `part` the static conjuncts keep, over the part.
    fn kept(&self, part: &Range<u64>) -> Option<Mask> {
        let start = self.scan.rows().start;
        self.kept.as_ref().map(|kept| {
            let offset = |row: u64| usize::try_from(row - start).unwrap_or(usize::MAX);
            kept.slice(offset(part.start)..offset(part.end))
        })
    }
}

impl Planned {
    /// Prepares the scan of `rows` of the child, if it reads any, prunes them by the zone maps of
    /// the filter's static conjuncts, and cuts them into parts.
    pub(super) async fn new(state: &ScanState, rows: Option<Range<u64>>) -> VortexResult<Self> {
        let Some(rows) = rows.filter(|rows| !rows.is_empty()) else {
            return Ok(Self::empty());
        };
        let scan = Arc::new(state.builder(rows).prepare()?);
        let io = scan.io();
        let kept = static_kept(&scan, &io).await?;
        if kept.as_ref().is_some_and(Mask::all_false) {
            return Ok(Self::empty());
        }
        let parts = parts_between(&scan.part_points(1, kept.as_ref())?);
        let boundaries = chunk_boundaries(&scan)?;
        Ok(Self {
            prepared: Some(Arc::new(Prepared {
                scan,
                io,
                kept,
                boundaries,
            })),
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

    /// Deals the parts out to at most `n` pieces of contiguous parts. A whole scan is cut again
    /// first, into at least `n` parts where its surviving rows and blocks allow. Never empty.
    pub(super) fn split(self, n: usize) -> VortexResult<Vec<Self>> {
        let Some(prepared) = self.prepared else {
            return Ok(vec![Self::empty()]);
        };
        let parts = if self.whole {
            parts_between(&prepared.scan.part_points(n, prepared.kept.as_ref())?)
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
    let session = state.session.clone();
    let exhausted = ExhaustedFlag::new();
    let Some(prepared) = planned.prepared.filter(|_| !planned.parts.is_empty()) else {
        let work = Arc::new(Work::new(
            Vec::new(),
            Vec::new(),
            None,
            1,
            exhausted.clone(),
        ));
        return ScanExec::new(dtype, exhausted, || stream::empty().boxed())
            .with_progress(move || work.claims())
            .with_batch_cuts(session);
    };
    let surviving = prepared
        .kept
        .as_ref()
        .map(|kept| kept_ranges(prepared.scan.rows(), kept));
    let work = Arc::new(Work::new(
        planned.parts,
        prepared.boundaries.clone(),
        surviving,
        (CLAIM_CHUNKS * prepared.scan.chunk_rows) as u64,
        exhausted.clone(),
    ));
    let progress = Arc::clone(&work);
    ScanExec::new(dtype, exhausted, move || {
        let worker = Worker {
            prepared: Arc::clone(&prepared),
            work: Arc::clone(&work),
            owned: None,
            drained: false,
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
    .with_batch_cuts(session)
}

/// The scan chunks of rows a stream claims at a time, at least: two, so that a stealing stream
/// finds work to cut off a part with as few as three chunks left.
const CLAIM_CHUNKS: usize = 2;

/// One stream's worker: a run of its own, fed rows claimed from the scan's work, and driven on
/// the task that polls the stream.
struct Worker {
    prepared: Arc<Prepared>,
    work: Arc<Work>,
    /// The running part the worker claims rows from.
    owned: Option<Owned>,
    /// Set once the worker found nothing left to claim, which stays so.
    drained: bool,
    limit: Option<RowLimit>,
    run: Run,
}

impl Worker {
    /// The worker's next array, or `None` once the scan has no more work for it.
    async fn next_array(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            // The next claim is admitted while the last chunk of the one before runs, so that
            // its reads overlap that chunk's compute.
            if self.run.live_roots() <= 1 && !self.admit_claim() && self.run.live_roots() == 0 {
                return Ok(None);
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

    /// Claims the worker's next rows and admits their planner, unless the scan's limit is spent
    /// or nothing is left to claim. Whether it admitted one.
    fn admit_claim(&mut self) -> bool {
        if self.drained {
            return false;
        }
        if self.limit.as_ref().is_some_and(RowLimit::is_exhausted) {
            self.work.exhaust();
            self.drained = true;
            return false;
        }
        let Some(rows) = self.work.claim(&mut self.owned) else {
            self.drained = true;
            return false;
        };
        let kept = self.prepared.kept(&rows);
        let planner = part_planner(&self.prepared.scan, rows.clone(), kept);
        let scope = WorkScope {
            file_ordinal: 0,
            rows,
        };
        self.run.admit(planner, scope, self.prepared.io.session());
        true
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
