// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::exec::ExecGraph;
use crate::plan::exec::Piece;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::SelectedRows;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;
use crate::scan::v2::splits::projection_splits;

/// Turns a filter split's selected rows into the morsels that project them.
///
/// The rows are cut into projection splits where the projection's chunks start, and each split
/// with a selected row becomes one [`ProjectionMorsel`], so a morsel reads at most one chunk of
/// every column. The planner finishes once it has handed out every morsel.
pub struct ProjectionPlanner {
    plans: ScanPlans,
    /// The projection splits not yet handed out, last first.
    pending: Vec<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        let SelectedRows { scope, mask } = selected;
        let start = scope.rows.start;
        let index = |row: u64| usize::try_from(row - start).vortex_expect("split row fits usize");
        let mut pending = projection_splits(&plans.projection_starts, scope.rows.clone())
            .into_iter()
            .filter_map(|rows| {
                let mask = mask.slice(index(rows.start)..index(rows.end));
                (!mask.all_false()).then(|| SelectedRows {
                    scope: WorkScope {
                        file_ordinal: scope.file_ordinal,
                        rows,
                    },
                    mask,
                })
            })
            .collect::<Vec<_>>();
        pending.reverse();
        Self { plans, pending }
    }
}

impl IoConsumer for ProjectionPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for ProjectionPlanner {
    fn state(&self) -> State {
        if self.pending.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = self.pending.pop() else {
            vortex_bail!("ProjectionPlanner: compute called after Done");
        };
        let scope = selected.scope.clone();
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(ProjectionMorsel::new(self.plans.clone(), selected)),
        ))
    }
}

/// Runs the projection plan over a split's selected rows and returns them as one batch, in row
/// order. A selection that projects to no rows finishes without a batch.
pub struct ProjectionMorsel {
    plans: ScanPlans,
    selected: SelectedRows,
    graph: Option<ProtocolGraph>,
    pieces: Vec<Piece>,
    done: bool,
}

impl ProjectionMorsel {
    /// Creates a morsel projecting `selected`.
    pub fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        Self {
            plans,
            selected,
            graph: None,
            pieces: Vec::new(),
            done: false,
        }
    }

    fn finish(&mut self) -> VortexResult<MorselOutput> {
        self.done = true;
        self.pieces.sort_by_key(|piece| piece.rows.start);
        let mut arrays: Vec<_> = std::mem::take(&mut self.pieces)
            .into_iter()
            .map(|piece| piece.array)
            .filter(|array| !array.is_empty())
            .collect();
        let array = match arrays.len() {
            0 => return Ok(MorselOutput::Done),
            1 => arrays.remove(0),
            _ => ChunkedArray::try_new(arrays, self.plans.projection.dtype().clone())?.into_array(),
        };
        Ok(MorselOutput::Batch(array))
    }
}

impl IoConsumer for ProjectionMorsel {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(graph) = self.graph.as_mut() {
            graph.set_io_result(request, result);
        }
    }
}

impl Morsel for ProjectionMorsel {
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

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("ProjectionMorsel: compute called after Done");
        }
        let Some(graph) = self.graph.as_mut() else {
            let graph = ExecGraph::try_new(
                self.plans.session.clone(),
                &self.plans.projection,
                self.selected.scope.rows.clone(),
                self.selected.mask.clone(),
                self.plans.row_offset,
                self.plans.decoded.clone(),
            )?;
            self.graph = Some(ProtocolGraph::new(
                graph,
                Arc::clone(&self.plans.locations),
                0,
            ));
            return Ok(MorselOutput::Continue);
        };
        if graph.state() == State::Done {
            return self.finish();
        }
        Ok(match graph.compute()? {
            GraphStep::Yield => MorselOutput::Continue,
            GraphStep::NeedsIO(batch) => MorselOutput::NeedsIO(batch),
            GraphStep::Piece(piece) => {
                self.pieces.push(piece);
                MorselOutput::Continue
            }
        })
    }
}
