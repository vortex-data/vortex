// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use bit_vec::BitVec;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_functions::between::find_between;
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

use super::ScanPlans;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
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

/// Plans and adaptive conjunct ordering shared by all splits of a scan.
pub(in crate::scan) struct FilterPlans {
    expression: FilterExpr,
    predicates: Vec<PlanRef>,
    pruning: Vec<Option<PlanRef>>,
}

impl FilterPlans {
    pub(in crate::scan) fn new(
        expression: BoundExpression,
        root: &PlanRef,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let expression = FilterExpr::new(find_between(expression)?);
        let predicates = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                let plan = optimize(plan_row_idx_expression(expr.clone(), root.clone())?)?;
                Ok(FilterPlan::new(plan).into_plan())
            })
            .collect::<VortexResult<_>>()?;
        let pruning = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                expr.falsify(session)?
                    .map(|proof| optimize(plan_row_idx_expression(proof, root.clone())?))
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
}

/// A pruning proof must not fetch data when its requested statistics are unavailable.
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

/// Proves zone exclusions, evaluates conjuncts over survivors, then hands the mask to projection.
pub(super) struct FilterPlanner {
    plans: ScanPlans,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    prune_index: usize,
    active: Option<(usize, bool, ScanAdaptor)>,
    pieces: Vec<Mask>,
    announced: bool,
}

impl FilterPlanner {
    pub(super) fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        let filter = Arc::clone(plans.filter.as_ref().vortex_expect("filter plans exist"));
        Self {
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            plans,
            selected: Some(selected),
            filter,
            prune_index: 0,
            active: None,
            pieces: Vec::new(),
            announced: false,
        }
    }
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some((_, _, adaptor)) = &mut self.active {
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
            Some((_, _, adaptor)) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if !self.announced {
            self.announced = true;
            let mut batch = Vec::new();
            for plan in self
                .filter
                .pruning
                .iter()
                .flatten()
                .chain(self.filter.predicates.iter())
                .chain(std::iter::once(&self.plans.projection))
            {
                let segments = Scan::segments(
                    &self.plans.session,
                    plan,
                    selected.scope.rows.clone(),
                    &selected.mask,
                )?;
                for segment in segments {
                    batch.push(IoRequest {
                        intent: IoIntent::Announce,
                        request: IoRequestId(u32::MAX - u32::try_from(batch.len())?),
                        target: self.plans.locations[*segment as usize].target(),
                    });
                }
            }
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some((index, pruning, adaptor)) = &mut self.active {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        let mut ctx = self.plans.session.create_execution_ctx();
                        let mask = piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)?;
                        self.pieces.push(mask);
                        PlannerOutput::Continue
                    }
                });
            }
            let input = selected.mask.true_count();
            let result = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(result.len(), input);
            let result = if *pruning { !result } else { result };
            selected.mask = selected.mask.intersect_by_rank(&result);
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = selected.mask.true_count(),
                pruning = *pruning,
                conjunct = *index,
                "v2 filter conjunct evaluated"
            );
            if !*pruning {
                self.filter
                    .expression
                    .report_selectivity(*index, selected.mask.true_count() as f64 / input as f64);
            }
            self.active = None;
            self.pieces.clear();
        }
        if selected.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }

        let mut next = None;
        while self.prune_index < self.filter.pruning.len() {
            let index = self.prune_index;
            self.prune_index += 1;
            if let Some(plan) = &self.filter.pruning[index] {
                next = Some((index, true, plan.clone()));
                break;
            }
        }
        if next.is_none()
            && let Some(index) = self.filter.expression.next_conjunct(&self.remaining)
        {
            self.remaining.set(index, false);
            next = Some((index, false, self.filter.predicates[index].clone()));
        }
        if let Some((index, pruning, plan)) = next {
            let scan = Scan::try_new(
                self.plans.session.clone(),
                plan,
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.plans.row_offset)
            .with_value_cache(self.plans.values.clone());
            self.active = Some((
                index,
                pruning,
                ScanAdaptor::new(scan, Arc::clone(&self.plans.locations)),
            ));
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(self.plans.clone(), selected)),
        ))
    }
}
