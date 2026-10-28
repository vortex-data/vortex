// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use bit_vec::BitVec;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;

use super::ScanPlans;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use super::projection::ProjectionPlanner;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::selection::selected;
use crate::scan::filter::FilterExpr;

/// Physical predicates and the selectivity order shared by this scan's splits.
pub(in crate::scan) struct FilterPlans {
    pub(in crate::scan) expressions: FilterExpr,
    pub(in crate::scan) plans: Vec<PlanRef>,
}

/// Evaluates each conjunct only on the rows earlier conjuncts retained. The rank mapping
/// keeps sparse input selections in file coordinates before projection reads are requested.
pub(super) struct FilterPlanner {
    plans: ScanPlans,
    selected: Option<SelectedRows>,
    remaining: BitVec,
    current: Option<usize>,
    adaptor: Option<ScanAdaptor>,
    masks: Vec<Mask>,
    next_request: u32,
}

impl FilterPlanner {
    pub(super) fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        let count = plans.filter.as_ref().map_or(0, |filter| filter.plans.len());
        Self {
            plans,
            selected: Some(selected),
            remaining: BitVec::from_elem(count, true),
            current: None,
            adaptor: None,
            masks: Vec::new(),
            next_request: 0,
        }
    }
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = &mut self.adaptor {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        match &self.adaptor {
            Some(adaptor) if adaptor.state() == State::Waiting => State::Waiting,
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(rows) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if rows.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        let filter = self.plans.filter.as_ref()
            .ok_or_else(|| vortex_err!("Filter planner has no filter plans"))?;
        if let Some(adaptor) = &mut self.adaptor {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        let mut ctx = self.plans.session.create_execution_ctx();
                        self.masks.push(piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let mask = Mask::concat(self.masks.iter())?;
            vortex_ensure_eq!(mask.len(), rows.mask.true_count());
            let index = self.current.take()
                .ok_or_else(|| vortex_err!("Completed filter has no conjunct"))?;
            tracing::debug!(
                target: "vortex_layout::scan::execution::filter",
                row_start = rows.scope.rows.start,
                row_end = rows.scope.rows.end,
                conjunct = index,
                input_rows = mask.len(),
                output_rows = mask.true_count(),
                "completed V2 conjunct",
            );
            filter.expressions.report_row_counts(index, mask.len(), mask.true_count());
            rows.mask = rows.mask.intersect_by_rank(&mask);
            self.remaining.set(index, false);
            self.next_request = adaptor.next_request_id();
            self.adaptor = None;
            self.masks.clear();
            return Ok(PlannerOutput::Continue);
        }
        let Some(index) = filter.expressions.next_conjunct(&self.remaining) else {
            let rows = self.selected.take()
                .ok_or_else(|| vortex_err!("Filter lost its selected rows"))?;
            return Ok(PlannerOutput::Planner(
                rows.scope.clone(),
                Box::new(ProjectionPlanner::new(self.plans.clone(), rows)),
            ));
        };
        tracing::debug!(
            target: "vortex_layout::scan::execution::filter",
            row_start = rows.scope.rows.start,
            row_end = rows.scope.rows.end,
            conjunct = index,
            input_rows = rows.mask.true_count(),
            "starting V2 conjunct",
        );
        let scan = Scan::try_new(
            self.plans.session.clone(),
            selected(filter.plans[index].clone(), rows.scope.rows.clone(), rows.mask.clone()),
            vec![Split {
                rows: rows.scope.rows.clone(),
                mask: Mask::new_true(rows.mask.len()),
            }],
        )?
        .with_row_offset(self.plans.row_offset)
        .with_value_cache(self.plans.values.clone());
        self.current = Some(index);
        self.adaptor = Some(ScanAdaptor::new(
            scan, Arc::clone(&self.plans.locations), self.plans.subsegment_reads,
        ).with_request_start(self.next_request));
        Ok(PlannerOutput::Continue)
    }
}
