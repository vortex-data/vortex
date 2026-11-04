// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composes planners for a chunk using its filter and estimated physical reads.

use std::sync::Arc;

use once_cell::unsync::OnceCell;
use rustc_hash::FxHashSet;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use super::PreparedScan;
use super::SelectedRows;
use super::filter::FilterPlanner;
use super::prefetch::PrefetchPlanner;
use super::prefetch::hint;
use super::prefetch::segments_read;
use super::projection::ProjectionPlanner;
use crate::segments::SegmentId;

/// Composes the planners of one chunk. Called on the driver's compute path.
///
/// Implementations may run concurrently for different chunks. They must evaluate any pending
/// filter before producing output; selectivity estimates only guide scheduling. Closures with
/// this signature also implement the trait.
///
/// Use [`ChunkPlanningContext::filter_then`] to construct a planner from surviving rows, or
/// wrap a filter with [`ChunkPlanningContext::prefetch_projection`] to start projection reads
/// before those rows are known. Returning a custom [`Planner`] requires no driver changes.
pub trait ScanStrategy: Send + Sync {
    /// Returns the first planner of the chunk's execution, without performing I/O.
    fn plan(&self, context: &ChunkPlanningContext) -> VortexResult<Box<dyn Planner>>;
}

impl<F> ScanStrategy for F
where
    F: Fn(&ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> + Send + Sync,
{
    fn plan(&self, context: &ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> {
        self(context)
    }
}

/// Prefetches filter inputs and announces projection inputs before evaluating the filter.
/// Unfiltered chunks project directly. This is the default scan strategy.
#[derive(Clone, Copy, Debug, Default)]
pub struct EagerScanStrategy;

impl ScanStrategy for EagerScanStrategy {
    fn plan(&self, context: &ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> {
        if !context.has_filter() {
            return Ok(context.project());
        }
        context.prefetch_filter(context.filter()?)
    }
}

/// Evaluates filters on demand, then prefetches projection inputs for the surviving rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct FilterFirstStrategy;

impl ScanStrategy for FilterFirstStrategy {
    fn plan(&self, context: &ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> {
        if !context.has_filter() {
            return Ok(context.project());
        }
        let filter = context.filter_then(|kept| kept.prefetch_projection(kept.project()))?;
        context.announce_reads(filter)
    }
}

/// Overlaps projection reads with filtering when observed selectivity and read size permit it.
///
/// Unknown or low selectivity uses [`FilterFirstStrategy`]. Otherwise projection is prefetched
/// before filtering when the union of filter and projection segments fits the byte budget.
/// These estimates do not assume that row selectivity translates into proportional byte savings.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveScanStrategy {
    max_prefetch_bytes: u64,
    min_selectivity: f64,
}

impl AdaptiveScanStrategy {
    /// Sets the per-chunk read budget and minimum observed fraction of rows passing the filter.
    /// `min_selectivity` must be between zero and one, inclusive.
    pub fn new(max_prefetch_bytes: u64, min_selectivity: f64) -> VortexResult<Self> {
        vortex_ensure!(
            (0.0..=1.0).contains(&min_selectivity),
            "selectivity must be between zero and one"
        );
        Ok(Self {
            max_prefetch_bytes,
            min_selectivity,
        })
    }
}

impl Default for AdaptiveScanStrategy {
    /// Starts with an 8 MiB budget and requires at least half of observed rows to survive.
    fn default() -> Self {
        Self {
            max_prefetch_bytes: 8 * 1024 * 1024,
            min_selectivity: 0.5,
        }
    }
}

impl ScanStrategy for AdaptiveScanStrategy {
    fn plan(&self, context: &ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> {
        if !context.has_filter() {
            return Ok(context.project());
        }
        let estimate = context.estimate()?;
        let decision = match estimate.filter_selectivity {
            None => "filter_first:unknown_selectivity",
            Some(s) if s < self.min_selectivity => "filter_first:low_selectivity",
            Some(_)
                if estimate
                    .filter_bytes
                    .saturating_add(estimate.projection_extra_bytes)
                    > self.max_prefetch_bytes =>
            {
                "filter_first:over_budget"
            }
            Some(_) => "prefetch_projection",
        };
        tracing::debug!(
            target: "vx_strategy",
            decision,
            selectivity = ?estimate.filter_selectivity,
            filter_bytes = estimate.filter_bytes,
            projection_extra_bytes = estimate.projection_extra_bytes,
            rows = context.selection().true_count(),
            "adaptive"
        );
        if estimate
            .filter_selectivity
            .is_some_and(|selectivity| selectivity >= self.min_selectivity)
            && estimate
                .filter_bytes
                .saturating_add(estimate.projection_extra_bytes)
                <= self.max_prefetch_bytes
        {
            context.prefetch_projection(context.filter()?)
        } else {
            FilterFirstStrategy.plan(context)
        }
    }
}

/// Physical read estimates for the chunk's current selection, before cache hits or coalescing.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ChunkReadEstimate {
    /// Observed fraction of selected input rows surviving the whole filter, or no observations.
    pub filter_selectivity: Option<f64>,
    /// Stored bytes of distinct filter segments.
    pub filter_bytes: u64,
    /// Stored bytes of distinct projection segments.
    pub projection_bytes: u64,
    /// Projection bytes excluding segments also read by the filter.
    pub projection_extra_bytes: u64,
    /// Number of distinct filter segments.
    pub filter_segments: usize,
    /// Number of distinct projection segments.
    pub projection_segments: usize,
}

struct ChunkReads {
    filter: Vec<SegmentId>,
    projection: Vec<SegmentId>,
    filter_bytes: u64,
    projection_bytes: u64,
    projection_extra_bytes: u64,
}

/// A chunk's selection, lazy read estimates, and constructors for composing its planners.
///
/// After a [`filter_then`](Self::filter_then) continuation, the context contains the surviving
/// mask and no pending filter. Read estimates are recomputed for that selection when requested.
pub struct ChunkPlanningContext {
    scan: Arc<PreparedScan>,
    selected: SelectedRows,
    filter_pending: bool,
    reads: OnceCell<ChunkReads>,
}

impl ChunkPlanningContext {
    fn new(scan: Arc<PreparedScan>, selected: SelectedRows, filter_pending: bool) -> Self {
        Self {
            scan,
            selected,
            filter_pending,
            reads: OnceCell::new(),
        }
    }

    /// The chunk's file-local row range and ordering identity.
    pub fn scope(&self) -> &WorkScope {
        &self.selected.scope
    }

    /// Selected rows within the chunk's scope.
    pub fn selection(&self) -> &Mask {
        &self.selected.mask
    }

    /// Whether this context still requires filter evaluation.
    pub fn has_filter(&self) -> bool {
        self.filter_pending
    }

    /// Compiles read sets without I/O, caching them for subsequent planner construction.
    pub fn estimate(&self) -> VortexResult<ChunkReadEstimate> {
        let reads = self.reads()?;
        Ok(ChunkReadEstimate {
            filter_selectivity: self
                .scan
                .filter
                .as_ref()
                .filter(|_| self.filter_pending)
                .and_then(|filter| filter.selectivity()),
            filter_bytes: reads.filter_bytes,
            projection_bytes: reads.projection_bytes,
            projection_extra_bytes: reads.projection_extra_bytes,
            filter_segments: reads.filter.len(),
            projection_segments: reads.projection.len(),
        })
    }

    /// Projects this context's selected rows. Call after evaluating any pending filter.
    pub fn project(&self) -> Box<dyn Planner> {
        Box::new(ProjectionPlanner::new(
            Arc::clone(&self.scan),
            self.selected.clone(),
        ))
    }

    /// Evaluates the filter, then projects its surviving rows.
    pub fn filter(&self) -> VortexResult<Box<dyn Planner>> {
        self.filter_then(|kept| Ok(kept.project()))
    }

    /// Evaluates the filter and passes the surviving selection to a continuation.
    /// A chunk with no survivors finishes without calling the continuation.
    pub fn filter_then(
        &self,
        next: impl FnOnce(ChunkPlanningContext) -> VortexResult<Box<dyn Planner>> + Send + 'static,
    ) -> VortexResult<Box<dyn Planner>> {
        let scan = Arc::clone(&self.scan);
        if !self.filter_pending {
            return next(Self::new(scan, self.selected.clone(), false));
        }
        Ok(Box::new(FilterPlanner::new(
            Arc::clone(&scan),
            self.selected.clone(),
            Box::new(move |selected| next(Self::new(scan, selected, false))),
        )))
    }

    /// Announces potential filter and projection reads, then runs `next`.
    pub fn announce_reads(&self, next: Box<dyn Planner>) -> VortexResult<Box<dyn Planner>> {
        let reads = self.reads()?;
        Ok(self.with_hints(
            next,
            [
                (IoIntent::Announce, reads.filter.as_slice()),
                (IoIntent::Announce, reads.projection.as_slice()),
            ],
        ))
    }

    /// Prefetches filter inputs and announces projection inputs, then runs `next`.
    pub fn prefetch_filter(&self, next: Box<dyn Planner>) -> VortexResult<Box<dyn Planner>> {
        let reads = self.reads()?;
        Ok(self.with_hints(
            next,
            [
                (IoIntent::Prefetch, reads.filter.as_slice()),
                (IoIntent::Announce, reads.projection.as_slice()),
            ],
        ))
    }

    /// Prefetches projection inputs and announces pending filter inputs, then runs `next`.
    /// The continuation may evaluate a filter while these reads are in flight.
    pub fn prefetch_projection(&self, next: Box<dyn Planner>) -> VortexResult<Box<dyn Planner>> {
        let reads = self.reads()?;
        Ok(self.with_hints(
            next,
            [
                (IoIntent::Prefetch, reads.projection.as_slice()),
                (IoIntent::Announce, reads.filter.as_slice()),
            ],
        ))
    }

    fn reads(&self) -> VortexResult<&ChunkReads> {
        self.reads.get_or_try_init(|| {
            let filter = match self.scan.filter.as_ref().filter(|_| self.filter_pending) {
                Some(filter) => segments_read(
                    &self.scan.session,
                    filter.predicates(),
                    self.selected.scope.rows.clone(),
                    &self.selected.mask,
                )?,
                None => Vec::new(),
            };
            let projection = segments_read(
                &self.scan.session,
                std::iter::once(&self.scan.projection),
                self.selected.scope.rows.clone(),
                &self.selected.mask,
            )?;
            let bytes =
                |segment: &SegmentId| u64::from(self.scan.locations[**segment as usize].length);
            Ok(ChunkReads {
                filter_bytes: filter.iter().map(bytes).sum(),
                projection_bytes: projection.iter().map(bytes).sum(),
                projection_extra_bytes: projection
                    .iter()
                    .filter(|segment| filter.binary_search(segment).is_err())
                    .map(bytes)
                    .sum(),
                filter,
                projection,
            })
        })
    }

    fn with_hints<'a>(
        &self,
        next: Box<dyn Planner>,
        groups: impl IntoIterator<Item = (IoIntent, &'a [SegmentId])>,
    ) -> Box<dyn Planner> {
        let mut batch = Vec::new();
        let mut seen = FxHashSet::default();
        let mut next_hint = u32::MAX;
        for (intent, segments) in groups {
            hint(
                &self.scan.locations,
                &mut next_hint,
                intent,
                segments,
                &mut seen,
                &mut batch,
            );
        }
        Box::new(PrefetchPlanner::new(self.scope().clone(), batch, next))
    }
}

/// Invokes the strategy on the driver's compute path, then hands off to the chosen planner.
pub(super) struct StrategyPlanner(Option<ChunkPlanningContext>);

impl StrategyPlanner {
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let filter_pending = scan.filter.is_some();
        Self(Some(ChunkPlanningContext::new(
            scan,
            selected,
            filter_pending,
        )))
    }
}

impl IoConsumer for StrategyPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for StrategyPlanner {
    fn state(&self) -> State {
        if self.0.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(context) = self.0.take() else {
            return Ok(PlannerOutput::Done);
        };
        let planner = context.scan.strategy.plan(&context)?;
        Ok(PlannerOutput::Planner(context.scope().clone(), planner))
    }
}
