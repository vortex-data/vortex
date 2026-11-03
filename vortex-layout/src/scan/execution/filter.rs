// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod ordering;

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

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

use self::ordering::AdaptiveOrder;
use self::ordering::ChunkOrder;
use self::ordering::Observation;
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
    ordering: AdaptiveOrder,
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
                // Leaf filters already apply the demand. Wrapping the predicate in another
                // Filter would reset it to all rows before evaluating the expression.
                optimize(
                    plan_row_idx_expression(expr.clone(), root.clone())?,
                    session,
                )
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
            ordering: AdaptiveOrder::new(expression.conjuncts().len()),
            expression,
            predicates,
            pruning,
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
/// The first compute starts the reads of every conjunct's segments for the selected rows and
/// announces the projection's, so the conjuncts run back to back and the projection's reads
/// coalesce with them. Each chunk takes its conjunct order from the cost estimates its file's
/// scans share, occasionally exploring a neighbouring order.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    order: ChunkOrder,
    active: Option<ActiveConjunct>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
}

/// A conjunct being evaluated, with the CPU time it has taken so far.
struct ActiveConjunct {
    index: usize,
    /// The dynamic version sampled before evaluation, so an update during IO is not credited.
    version: Option<u64>,
    adaptor: ScanAdaptor,
    cpu_ns: u128,
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
        let order = filter.ordering.begin_chunk();
        tracing::debug!(
            row_start = selected.scope.rows.start,
            row_end = selected.scope.rows.end,
            order = ?order.indices,
            exploration = order.exploration,
            "v2 predicate order selected"
        );
        Self {
            order,
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            announced: false,
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

/// Executes a scan's boolean piece to a mask, nulls as false.
pub(super) fn piece_mask(session: &VortexSession, piece: Piece) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(active) = &mut self.active {
            active.adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        match &self.active {
            Some(active) if active.adaptor.state() != State::Done => active.adaptor.state(),
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
            hint(
                locations,
                &mut self.next_hint,
                IoIntent::Announce,
                &projection,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        // Only the order among several conjuncts needs their costs.
        let timed = self.filter.predicates.len() > 1;
        if let Some(active) = &mut self.active {
            let started = timed.then(Instant::now);
            if active.adaptor.state() != State::Done {
                let output = match active.adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                };
                if let Some(started) = started {
                    active.cpu_ns += started.elapsed().as_nanos();
                }
                return Ok(output);
            }
            let input = selected.mask.true_count();
            let result = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(result.len(), input);
            selected.mask = selected.mask.intersect_by_rank(&result);
            if let Some(started) = started {
                active.cpu_ns += started.elapsed().as_nanos();
            }
            let output = selected.mask.true_count();
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = output,
                conjunct = active.index,
                cpu_ns = active.cpu_ns,
                "v2 filter conjunct evaluated"
            );
            self.filter.ordering.report(
                active.index,
                Observation {
                    version: active.version.unwrap_or(0),
                    input_rows: input,
                    output_rows: output,
                    compute_ns: active.cpu_ns,
                    recheck: false,
                },
            );
            self.active = None;
            self.pieces.clear();
        }
        if selected.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        if let Some(index) = self.order.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            let started = timed.then(Instant::now);
            let version = self
                .filter
                .expression
                .dynamic_updates(index)
                .map(|updates| updates.version());
            if let Some(version) = version {
                // A new bound changes this conjunct's cost and selectivity, so earlier
                // estimates no longer describe it.
                self.filter.ordering.version_changed(index, version);
            }
            let plan = self.filter.predicates[index].clone();
            // Dense demand keeps vectorized evaluation cheaper than gathering almost every row.
            // Selective demand is kept all the way to the predicate's leaves.
            let plan = if !selected.mask.all_true()
                && selected.mask.true_count().saturating_mul(2) >= selected.mask.len()
            {
                FilterPlan::new(plan).into_plan()
            } else {
                plan
            };
            let scan = Scan::try_new(
                self.scan.session.clone(),
                plan,
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some(ActiveConjunct {
                index,
                version,
                adaptor: ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
                cpu_ns: started.map_or(0, |started| started.elapsed().as_nanos()),
            });
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
