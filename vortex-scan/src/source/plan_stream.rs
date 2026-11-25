// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An opened [`PlanDataSource`](super::PlanDataSource) scan: V2's planners and [`Run`] driver
//! behind [`ScanExec`] and its [`ScanStream`](crate::stream::ScanStream)s.
//!
//! When a scan is made, it prepares its child once, prunes the child's rows by the zone maps of
//! its filter's static conjuncts, and cuts the rows into [`units`](crate::split::units) at the
//! chunk boundaries of the columns it reads, leaving out the units pruning rules out entirely.
//! [`split`](crate::DataSourceScan::split) deals the same units out to the pieces in contiguous
//! groups.
//!
//! Every stream of an opened scan shares one [`Work`]: a stream claims one unit at a time from
//! the front of the units it runs, and once it has none left it steals the back half of another
//! stream's unclaimed units. The scan is exhausted once no unit is left unclaimed. Claimed rows
//! are planned when a stream claims
//! them, over the rows the static conjuncts kept: with a dynamic conjunct, they are first pruned
//! against the zone maps, applying the conjunct as it stands then, such as the bound a top-N
//! pushes down. Each stream owns a [`Run`], driven on the task that polls the stream: a claim's
//! chunks run side by side, so one chunk's reads overlap another's compute, until a stream has
//! had to steal, the next claim is admitted while the last chunk of the one before runs, batches are returned as they are
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
use crate::split::units;
use crate::stream::ExhaustedFlag;
use crate::stream::ScanExec;

/// The units of a scan, or of a piece of one, cut when the scan was made.
pub(super) struct Planned {
    /// The child's prepared scan, which every unit plans over, or `None` for a scan that reads
    /// nothing.
    prepared: Option<Arc<Prepared>>,
    /// The units that zone pruning left rows of, in row order.
    units: Vec<Range<u64>>,
}

/// A child's prepared scan, the I/O its units read through, and the rows of the scan its
/// filter's static conjuncts keep.
struct Prepared {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    kept: Option<Mask>,
}

impl Prepared {
    /// The rows of `unit` the static conjuncts keep, over the unit.
    fn kept(&self, unit: &Range<u64>) -> Option<Mask> {
        let start = self.scan.rows().start;
        self.kept.as_ref().map(|kept| {
            let offset = |row: u64| usize::try_from(row - start).unwrap_or(usize::MAX);
            kept.slice(offset(unit.start)..offset(unit.end))
        })
    }
}

impl Planned {
    /// Prepares the scan of `rows` of the child, if it reads any, prunes them by the zone maps of
    /// the filter's static conjuncts, and cuts them into units.
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
        let surviving = kept.as_ref().map(|kept| kept_ranges(scan.rows(), kept));
        let units = units(
            scan.rows().clone(),
            &chunk_boundaries(&scan)?,
            surviving.as_deref(),
        );
        Ok(Self {
            prepared: Some(Arc::new(Prepared { scan, io, kept })),
            units,
        })
    }

    fn empty() -> Self {
        Self {
            prepared: None,
            units: Vec::new(),
        }
    }

    /// The number of units.
    pub(super) fn len(&self) -> usize {
        self.units.len()
    }

    /// Deals the units out to at most `n` pieces of contiguous units. Never empty.
    pub(super) fn split(self, n: usize) -> Vec<Self> {
        let Some(prepared) = self.prepared else {
            return vec![Self::empty()];
        };
        group_parts(self.units, n)
            .into_iter()
            .map(|units| Self {
                prepared: Some(Arc::clone(&prepared)),
                units,
            })
            .collect()
    }
}

/// Opens a scan onto the units it `planned`. A scan with no units opens exhausted.
pub(super) fn open(state: &ScanState, planned: Planned) -> ScanExec {
    let dtype = state.dtype.clone();
    let limit = state.limit.clone();
    let session = state.session.clone();
    let exhausted = ExhaustedFlag::new();
    let Some(prepared) = planned.prepared.filter(|_| !planned.units.is_empty()) else {
        let work = Arc::new(Work::new(Vec::new(), None, exhausted.clone()));
        return ScanExec::new(dtype, exhausted, || stream::empty().boxed())
            .with_progress(move || work.claims())
            .with_batch_cuts(session);
    };
    let surviving = prepared
        .kept
        .as_ref()
        .map(|kept| kept_ranges(prepared.scan.rows(), kept));
    let work = Arc::new(Work::new(planned.units, surviving, exhausted.clone()));
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

/// One stream's worker: a run of its own, fed rows claimed from the scan's work, and driven on
/// the task that polls the stream.
struct Worker {
    prepared: Arc<Prepared>,
    work: Arc<Work>,
    /// The units the worker claims from.
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
            // Until a stream has had to steal, the next claim is admitted while the last chunk of
            // the one before runs, so that its reads overlap that chunk's compute. Once one has,
            // a claim waits for the last to finish, leaving the rest of the part to steal.
            let ahead = if self.work.stealing() { 0 } else { 1 };
            if self.run.live_roots() <= ahead && !self.admit_claim() && self.run.live_roots() == 0 {
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
