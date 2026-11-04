// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use bit_vec::BitVec;
use parking_lot::Mutex;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_expr::transform::find_between;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_session::VortexSession;

use super::PreparedScan;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::Piece;
use super::adaptor::ScanAdaptor;
use crate::plan::physical::FilterPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScan;
use crate::plan::physical::Zoned;
use crate::plan::physical::optimize;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::filter::FilterExpr;

/// Plans and adaptive conjunct ordering shared by every chunk of a scan.
pub struct FilterPlans {
    expression: FilterExpr,
    predicates: Vec<PlanRef>,
    /// A metadata-only pruning plan per conjunct, where the conjunct has one.
    pruning: Vec<Option<PlanRef>>,
    selectivity: Mutex<FilterSelectivity>,
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
            selectivity: Mutex::default(),
        })
    }

    pub(super) fn predicates(&self) -> impl Iterator<Item = &PlanRef> {
        self.predicates.iter()
    }

    pub(super) fn selectivity(&self) -> Option<f64> {
        let observed = self.selectivity.lock();
        (observed.input_rows > 0).then(|| observed.output_rows as f64 / observed.input_rows as f64)
    }

    fn record_selectivity(&self, input_rows: usize, output_rows: usize) {
        let mut observed = self.selectivity.lock();
        observed.input_rows = observed.input_rows.saturating_add(input_rows as u64);
        observed.output_rows = observed.output_rows.saturating_add(output_rows as u64);
    }

    /// The pruning plans, run once over a scan's rows before its chunks are planned.
    pub(crate) fn pruning(&self) -> impl Iterator<Item = &PlanRef> {
        self.pruning.iter().flatten()
    }
}

/// Whole-filter observations avoid assuming independence between conjuncts. Shared by chunks
/// using the same cached filter plans; these statistics only affect scheduling.
#[derive(Default)]
struct FilterSelectivity {
    input_rows: u64,
    output_rows: u64,
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

/// Fetches and evaluates a chunk's conjuncts, then passes surviving rows to its continuation.
/// Conjuncts run in the order the shared [`FilterExpr`] picks. Optional reads are composed by
/// the scan strategy in separate planners.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    active: Option<(usize, ScanAdaptor)>,
    pieces: Vec<Mask>,
    input_rows: usize,
    next: Option<FilterNext>,
}

pub(super) type FilterNext = Box<dyn FnOnce(SelectedRows) -> VortexResult<Box<dyn Planner>> + Send>;

impl FilterPlanner {
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows, next: FilterNext) -> Self {
        let filter = Arc::clone(scan.filter.as_ref().vortex_expect("filter plans exist"));
        Self {
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            scan,
            input_rows: selected.mask.true_count(),
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            next: Some(next),
        }
    }
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
            self.filter.record_selectivity(self.input_rows, 0);
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
        let selected = self.selected.take().vortex_expect("selected rows exist");
        self.filter
            .record_selectivity(self.input_rows, selected.mask.true_count());
        let scope = selected.scope.clone();
        let next = self.next.take().vortex_expect("filter continuation exists");
        Ok(PlannerOutput::Planner(scope, next(selected)?))
    }
}
