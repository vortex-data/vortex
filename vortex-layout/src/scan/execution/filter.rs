// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bit_vec::BitVec;
use rustc_hash::FxHashSet;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_expr::transform::find_between;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_session::VortexSession;

use super::PreparedScan;
use super::SegmentLocation;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::Piece;
use super::adaptor::ScanAdaptor;
use super::projection::ProjectionPlanner;
use crate::plan::physical::FilterPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScan;
use crate::plan::physical::Zoned;
use crate::plan::physical::optimize;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::filter::FilterExpr;
use crate::segments::SegmentId;

/// Plans and adaptive conjunct ordering shared by every chunk of a scan.
pub struct FilterPlans {
    expression: FilterExpr,
    predicates: Vec<PlanRef>,
    /// A metadata-only pruning plan per conjunct, where the conjunct has one.
    pruning: Vec<Option<PlanRef>>,
    /// What the scans of this filter learned about their projection reads, unless a scan was
    /// given feedback to share with other files' scans.
    pub(crate) feedback: PrefetchFeedback,
}

/// What filtered scans learn about how much of a chunk's projection its filtered rows read,
/// which decides whether later chunks prefetch their projection beside their conjuncts.
///
/// Every scan of a file's filter shares one by default. Scans of the same filter and projection
/// over other files, such as the files of one table scan, can share one through
/// [`ScanBuilder::with_prefetch_feedback`](crate::scan::v2::ScanBuilder::with_prefetch_feedback),
/// so a file's first chunks reuse what earlier files measured instead of measuring again.
#[derive(Clone, Default)]
pub struct PrefetchFeedback(Arc<Feedback>);

#[derive(Default)]
struct Feedback {
    /// Over the sampled chunks that finished filtering: bytes of the projection segments no
    /// conjunct reads, and the part of them the filtered rows did not need.
    projected: AtomicU64,
    unneeded: AtomicU64,
    /// Chunks planned so far. Finding the segments a chunk's filtered rows need is not free,
    /// so only every [`WASTE_SAMPLE`]th chunk measures it.
    chunks: AtomicU64,
}

const WASTE_SAMPLE: u64 = 8;

impl PrefetchFeedback {
    /// Whether a chunk should start its projection's reads beside its conjuncts', so they
    /// overlap filtering. Segments the filtered rows do not reach would be read for nothing, so
    /// scans prefetch only once sampled chunks show at most an eighth of those bytes were
    /// unneeded.
    fn prefetch(&self) -> bool {
        let projected = self.0.projected.load(Ordering::Relaxed);
        projected > 0 && self.0.unneeded.load(Ordering::Relaxed).saturating_mul(8) <= projected
    }

    /// Whether the chunk being planned should measure how much of its projection it needs.
    fn sample(&self) -> bool {
        self.0
            .chunks
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(WASTE_SAMPLE)
    }

    fn record(&self, projected: u64, unneeded: u64) {
        self.0.projected.fetch_add(projected, Ordering::Relaxed);
        self.0.unneeded.fetch_add(unneeded, Ordering::Relaxed);
    }
}

impl FilterPlans {
    pub(crate) fn new(
        expression: BoundExpression,
        root: &PlanRef,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let expression = FilterExpr::new(find_between(expression)?);
        let predicates = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                let plan = optimize(
                    plan_row_idx_expression(expr.clone(), root.clone())?,
                    session,
                )?;
                Ok(FilterPlan::new(plan).into_plan())
            })
            .collect::<VortexResult<_>>()?;
        let pruning = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                expr.falsify(session)?
                    .map(|pruning| {
                        optimize(plan_row_idx_expression(pruning, root.clone())?, session)
                    })
                    .transpose()
            })
            .map(|plan| {
                let plan = plan?;
                match plan {
                    Some(plan) if metadata_only(&plan)? => Ok(Some(plan)),
                    _ => Ok(None),
                }
            })
            .collect::<VortexResult<_>>()?;
        Ok(Self {
            expression,
            predicates,
            pruning,
            feedback: PrefetchFeedback::default(),
        })
    }

    /// The pruning plans, run once over a scan's rows before its chunks are planned.
    pub(crate) fn pruning(&self) -> impl Iterator<Item = &PlanRef> {
        self.pruning.iter().flatten()
    }
}

/// A pruning plan must not fetch data when its requested statistics are unavailable.
fn metadata_only(plan: &PlanRef) -> VortexResult<bool> {
    if plan
        .as_opt::<Zoned>()
        .is_some_and(|zoned| zoned.is_pruning())
    {
        return Ok(true);
    }
    if plan.is::<SegmentScan>() {
        return Ok(false);
    }
    for child in plan.children().iter() {
        if !metadata_only(&child?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Evaluates a chunk's conjuncts over its selected rows, then hands the mask to projection.
///
/// The first compute starts the reads of every conjunct's segments for the selected rows, so the
/// conjuncts run back to back. It starts the projection's reads too while the scan's filtered
/// rows still need nearly all of them, so they overlap filtering, and otherwise only announces
/// them so they can still coalesce with the conjuncts'. Conjuncts run in the order the shared
/// [`FilterExpr`] picks.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    active: Option<(usize, ScanAdaptor)>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Projection segments no conjunct reads, in id order, kept only when the chunk samples
    /// how many of them filtered rows need.
    sampled_projection: Vec<SegmentId>,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
}

/// The segments `plans` read over `rows` under `mask`, each once, in id order. Compiling finds
/// them without IO.
pub(super) fn segments_read<'a>(
    session: &VortexSession,
    plans: impl Iterator<Item = &'a PlanRef>,
    rows: Range<u64>,
    mask: &Mask,
) -> VortexResult<Vec<SegmentId>> {
    let mut segments = Vec::new();
    for plan in plans {
        segments.extend(Scan::segments(session, plan, rows.clone(), mask)?);
    }
    segments.sort_unstable();
    segments.dedup();
    Ok(segments)
}

impl FilterPlanner {
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let filter = Arc::clone(scan.filter.as_ref().vortex_expect("filter plans exist"));
        Self {
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            announced: false,
            sampled_projection: Vec::new(),
            next_hint: u32::MAX,
        }
    }
}

/// Publishes a hint for each of `segments` not in `seen`, and adds it there.
pub(super) fn hint(
    locations: &[SegmentLocation],
    next_hint: &mut u32,
    intent: IoIntent,
    segments: &[SegmentId],
    seen: &mut FxHashSet<SegmentId>,
    batch: &mut IoBatch,
) {
    for &segment in segments {
        if !seen.insert(segment) {
            continue;
        }
        batch.push(IoRequest {
            intent,
            request: IoRequestId(*next_hint),
            target: locations[*segment as usize].target(),
        });
        *next_hint -= 1;
    }
}

/// The total length of `segments`.
fn segment_bytes(locations: &[SegmentLocation], segments: &[SegmentId]) -> u64 {
    segments
        .iter()
        .map(|&segment| u64::from(locations[*segment as usize].length))
        .sum()
}

/// Executes a scan's boolean piece to a mask, nulls as false.
pub(super) fn piece_mask(session: &VortexSession, piece: Piece) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some((_, adaptor)) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        match &self.active {
            Some((_, adaptor)) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let predicates = segments_read(
                session,
                self.filter.predicates.iter(),
                rows.clone(),
                &selected.mask,
            )?;
            let projection = segments_read(
                session,
                std::iter::once(&self.scan.projection),
                rows,
                &selected.mask,
            )?;
            let mut batch = Vec::new();
            let mut seen = FxHashSet::default();
            let locations = &self.scan.locations;
            hint(
                locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &predicates,
                &mut seen,
                &mut batch,
            );
            if self.scan.prefetch.sample() {
                self.sampled_projection = projection
                    .iter()
                    .filter(|segment| predicates.binary_search(segment).is_err())
                    .copied()
                    .collect();
            }
            let projection_intent = if self.scan.prefetch.prefetch() {
                IoIntent::Prefetch
            } else {
                IoIntent::Announce
            };
            hint(
                locations,
                &mut self.next_hint,
                projection_intent,
                &projection,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some((index, adaptor)) = &mut self.active {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let input = selected.mask.true_count();
            let result = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(result.len(), input);
            selected.mask = selected.mask.intersect_by_rank(&result);
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = selected.mask.true_count(),
                conjunct = *index,
                "v2 filter conjunct evaluated"
            );
            self.filter
                .expression
                .report_selectivity(*index, selected.mask.true_count() as f64 / input as f64);
            self.active = None;
            self.pieces.clear();
        }
        if selected.mask.all_false() {
            let bytes = segment_bytes(&self.scan.locations, &self.sampled_projection);
            self.scan.prefetch.record(bytes, bytes);
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        if let Some(index) = self.filter.expression.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            let scan = Scan::try_new(
                self.scan.session.clone(),
                self.filter.predicates[index].clone(),
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some((
                index,
                ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
            ));
            return Ok(PlannerOutput::Continue);
        }
        if !self.sampled_projection.is_empty() {
            let locations = &self.scan.locations;
            let projected = segment_bytes(locations, &self.sampled_projection);
            let needed = segments_read(
                &self.scan.session,
                std::iter::once(&self.scan.projection),
                selected.scope.rows.clone(),
                &selected.mask,
            )?;
            self.sampled_projection
                .retain(|segment| needed.binary_search(segment).is_err());
            self.scan.prefetch.record(
                projected,
                segment_bytes(locations, &self.sampled_projection),
            );
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
