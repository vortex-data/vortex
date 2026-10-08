// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::exec::Piece;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::SelectedRows;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;
use crate::scan::planning::graph::ScanGraph;
use crate::scan::v2::prefetch::plan_segments;
use crate::scan::v2::splits::projection_splits;

/// Turns a filter split's selected rows into the morsels that project them.
///
/// The rows are cut into projection splits where the projection's chunks start, and each split
/// with a selected row becomes one [`ProjectionMorsel`], so a morsel reads at most one chunk of
/// every column. The planner finishes once it has handed out every morsel.
///
/// When there are multiple morsels, the planner first prefetches every segment the projection
/// reads over the selected rows, so their reads start together. Nothing asks for projection
/// segments earlier by default. Opt-in projection read-ahead can start a bounded set during
/// filtering instead.
pub struct ProjectionPlanner {
    plans: ScanPlans,
    /// The filter split's selected rows, until the first compute cuts them.
    selected: Option<SelectedRows>,
    /// The projection splits not yet handed out, last first.
    pending: Vec<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        Self {
            plans,
            selected: Some(selected),
            pending: Vec::new(),
        }
    }

    /// Prefetches the nonempty projection chunks, which are already cut for execution.
    fn prefetch(&self) -> VortexResult<IoBatch> {
        if self.pending.len() == 1 {
            // One morsel publishes its own reads before waiting; there are no later morsels
            // whose reads need to start ahead of it.
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for selected in &self.pending {
            plan_segments(
                &self.plans.projection,
                selected.scope.rows.clone(),
                &mut ids,
            )?;
        }
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter()
            .enumerate()
            .map(|(index, id)| {
                let location = self
                    .plans
                    .locations
                    .get(*id as usize)
                    .ok_or_else(|| vortex_err!("segment {id} has no known location"))?;
                Ok(IoRequest {
                    intent: IoIntent::Prefetch,
                    request: IoRequestId(u32::try_from(index)?),
                    target: location.target(),
                })
            })
            .collect()
    }

    /// The projection splits of `selected` that have a selected row, last first.
    fn cut(&self, selected: SelectedRows) -> Vec<SelectedRows> {
        let SelectedRows { scope, mask } = selected;
        let start = scope.rows.start;
        let index = |row: u64| usize::try_from(row - start).vortex_expect("split row fits usize");
        let mut splits = projection_splits(&self.plans.projection_starts, scope.rows.clone())
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
        splits.reverse();
        splits
    }
}

impl IoConsumer for ProjectionPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for ProjectionPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() && self.pending.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(selected) = self.selected.take() {
            self.pending = self.cut(selected);
            // No projection split has a selected row.
            if self.pending.is_empty() {
                return Ok(PlannerOutput::Done);
            }
            let prefetch = self.prefetch()?;
            if !prefetch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(prefetch));
            }
        }
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
            let graph = ScanGraph::try_new(
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

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_buffer::Alignment;
    use vortex_mask::Mask;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::plan::ConcatPlan;
    use crate::plan::FilterPlan;
    use crate::plan::SegmentScanPlan;
    use crate::plan::exec::DecodeCache;
    use crate::scan::planning::SegmentLocation;
    use crate::segments::SegmentId;
    use crate::test::new_session;

    #[test]
    fn prefetch_skips_empty_chunks_without_enumerating_row_runs() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let chunks = (0..3)
            .map(|segment| {
                FilterPlan::new(
                    SegmentScanPlan::new(
                        dtype.clone(),
                        1000,
                        SegmentId::from(segment),
                        ReadContext::new([]),
                        None,
                    )
                    .into_plan(),
                )
                .into_plan()
            })
            .collect();
        let locations: Arc<[_]> = (0..3)
            .map(|offset| SegmentLocation {
                offset,
                length: 1,
                alignment: Alignment::none(),
            })
            .collect();
        // More than 64 runs used to fall back to prefetching the entire bounding range,
        // including the empty middle chunk, after enumerating every run.
        let mask =
            Mask::from_iter((0..3000).map(|row| row % 2 == 0 && !(1000..2000).contains(&row)));
        let mut planner = ProjectionPlanner::new(
            ScanPlans {
                session: new_session(),
                locations: Arc::clone(&locations),
                projection: ConcatPlan::try_new(dtype, chunks)?.into_plan(),
                projection_starts: Arc::from([0, 1000, 2000]),
                row_offset: 0,
                decoded: DecodeCache::default(),
            },
            SelectedRows {
                scope: WorkScope {
                    file_ordinal: 0,
                    rows: 0..3000,
                },
                mask: mask.clone(),
            },
        );
        let PlannerOutput::NeedsIO(requests) = planner.compute()? else {
            vortex_bail!("expected projection prefetch");
        };
        let targets: Vec<_> = requests.iter().map(|request| request.target).collect();
        assert_eq!(targets, vec![locations[0].target(), locations[2].target()]);
        let Mask::Values(values) = mask else {
            vortex_bail!("expected a partial mask");
        };
        assert!(values.cached_slices().is_none());
        Ok(())
    }
}
