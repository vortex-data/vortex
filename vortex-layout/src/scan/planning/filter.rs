// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::plan::exec::ExecGraph;
use crate::plan::exec::Piece;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;

/// The rows of a split that survived its filter.
pub struct SelectedRows {
    /// The split the rows belong to.
    pub scope: WorkScope,
    /// Which rows of `scope.rows` are selected.
    pub mask: Mask,
}

/// Evaluates a split's filter to a selection, then hands the selection to `next`.
///
/// The filter plan runs over the split's rows and incoming mask, producing one boolean per
/// selected row. A split where no row survives finishes without a child.
pub struct FilterPlanner {
    plans: ScanPlans,
    filter: PlanRef,
    scope: WorkScope,
    mask: Mask,
    next: Next<SelectedRows>,
    graph: Option<ProtocolGraph>,
    pieces: Vec<Piece>,
    done: bool,
}

impl FilterPlanner {
    /// Creates a planner that filters the rows of `scope` selected by `mask`.
    pub fn new(
        plans: ScanPlans,
        filter: PlanRef,
        scope: WorkScope,
        mask: Mask,
        next: Next<SelectedRows>,
    ) -> Self {
        Self {
            plans,
            filter,
            scope,
            mask,
            next,
            graph: None,
            pieces: Vec::new(),
            done: false,
        }
    }

    /// The incoming mask narrowed to the rows whose filter value is true.
    fn selected(&mut self) -> VortexResult<Mask> {
        self.pieces.sort_by_key(|piece| piece.rows.start);
        let values: Vec<ArrayRef> = std::mem::take(&mut self.pieces)
            .into_iter()
            .map(|piece| piece.array)
            .collect();
        let values = ChunkedArray::try_new(values, self.filter.dtype().clone())?.into_array();
        let mut ctx = self.plans.session.create_execution_ctx();
        let values: Mask = values.null_as_false().execute(&mut ctx)?;
        Ok(self.mask.intersect_by_rank(&values))
    }
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(graph) = self.graph.as_mut() {
            graph.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        match &self.graph {
            _ if self.done => State::Done,
            None => State::NeedsCompute,
            Some(graph) => match graph.state() {
                State::Done => State::NeedsCompute,
                state => state,
            },
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            vortex_bail!("FilterPlanner: compute called after Done");
        }
        let Some(graph) = self.graph.as_mut() else {
            let graph = ExecGraph::try_new(
                self.plans.session.clone(),
                &self.filter,
                self.scope.rows.clone(),
                self.mask.clone(),
                self.plans.row_offset,
            )?;
            self.graph = Some(ProtocolGraph::new(graph, Arc::clone(&self.plans.locations)));
            return Ok(PlannerOutput::Continue);
        };
        if graph.state() == State::Done {
            self.done = true;
            let mask = self.selected()?;
            if mask.all_false() {
                return Ok(PlannerOutput::Done);
            }
            let scope = self.scope.clone();
            let child = (self.next)(SelectedRows {
                scope: scope.clone(),
                mask,
            })?;
            return Ok(PlannerOutput::Planner(scope, child));
        }
        Ok(match graph.compute()? {
            GraphStep::Yield => PlannerOutput::Continue,
            GraphStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
            GraphStep::Piece(piece) => {
                self.pieces.push(piece);
                PlannerOutput::Continue
            }
        })
    }
}
