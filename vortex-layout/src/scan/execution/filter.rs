// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

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
/// coalesce with them. Conjuncts run in the order the shared [`FilterExpr`] picks.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    active: Option<(usize, ScanAdaptor)>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Whether a conjunct has filtered the chunk's rows.
    evaluated: bool,
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
            evaluated: false,
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

impl Drop for FilterPlanner {
    fn drop(&mut self) {
        // A conjunct still evaluating did not report, so another evaluation may measure it.
        if let Some((index, _)) = &self.active {
            self.filter.expression.abandon_evaluation(*index);
        }
    }
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
            // Conjuncts the chunk skips are decided before any read, so their segments are not
            // fetched.
            let input_rows = selected.mask.true_count();
            for index in 0..self.filter.predicates.len() {
                if !self.filter.expression.should_evaluate(index, input_rows) {
                    self.remaining.set(index, false);
                }
            }
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let predicates = segments_read(
                session,
                self.filter
                    .predicates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| self.remaining[*index])
                    .map(|(_, plan)| plan),
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
            self.filter.expression.report_evaluation(
                *index,
                input,
                selected.mask.true_count(),
                !self.evaluated,
            );
            self.evaluated = true;
            self.active = None;
            self.pieces.clear();
        }
        if selected.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        while let Some(index) = self.filter.expression.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            if !self.filter.expression.begin_evaluation(index) {
                continue;
            }
            let scan = match Scan::try_new(
                self.scan.session.clone(),
                self.filter.predicates[index].clone(),
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            ) {
                Ok(scan) => scan,
                Err(err) => {
                    self.filter.expression.abandon_evaluation(index);
                    return Err(err);
                }
            }
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some((
                index,
                ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
            ));
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
