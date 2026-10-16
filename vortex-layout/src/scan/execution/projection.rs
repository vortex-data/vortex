// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_io::segments::IoKey;
use vortex_io::segments::SourceResult;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;

use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::scan::execution::ScanPlans;
use crate::scan::execution::SelectedRows;
use crate::scan::execution::adaptor::AdaptorStep;
use crate::scan::execution::adaptor::Piece;
use crate::scan::execution::adaptor::ScanAdaptor;

/// Hands one fixed-size work chunk to a projection morsel.
pub(super) struct ProjectionPlanner {
    work: Option<(ScanPlans, SelectedRows)>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub(super) fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        Self {
            work: Some((plans, selected)),
        }
    }
}

impl Planner for ProjectionPlanner {
    fn deliver(&mut self, key: IoKey, _result: SourceResult) {
        vortex_panic!("ProjectionPlanner: delivery of {key:?}, though it never fetches");
    }

    fn state(&self) -> State {
        if self.work.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some((plans, selected)) = self.work.take() else {
            return Ok(PlannerOutput::Done);
        };
        if selected.mask.all_false() {
            return Ok(PlannerOutput::Done);
        }
        Ok(PlannerOutput::Morsel(
            selected.scope.clone(),
            Box::new(ProjectionMorsel::new(plans, selected)),
        ))
    }
}

/// Runs the projection plan over a split's selected rows and returns them as one batch, in row
/// order. A selection that projects to no rows finishes without a batch.
struct ProjectionMorsel {
    plans: Option<ScanPlans>,
    dtype: DType,
    selected: SelectedRows,
    adaptor: Option<ScanAdaptor>,
    pieces: Vec<Piece>,
    done: bool,
}

impl ProjectionMorsel {
    /// Creates a morsel projecting `selected`.
    fn new(plans: ScanPlans, selected: SelectedRows) -> Self {
        Self {
            dtype: plans.projection.dtype().clone(),
            plans: Some(plans),
            selected,
            adaptor: None,
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
            _ => ChunkedArray::try_new(arrays, self.dtype.clone())?.into_array(),
        };
        Ok(MorselOutput::Batch(array))
    }
}

impl Morsel for ProjectionMorsel {
    fn deliver(&mut self, key: IoKey, result: SourceResult) {
        if let Some(adaptor) = self.adaptor.as_mut() {
            adaptor.set_result(key, result);
        }
    }

    fn state(&self) -> State {
        match &self.adaptor {
            _ if self.done => State::Done,
            None => State::NeedsCompute,
            Some(adaptor) => match adaptor.state() {
                State::Done => State::NeedsCompute,
                state => state,
            },
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("ProjectionMorsel: compute called after Done");
        }
        let Some(adaptor) = self.adaptor.as_mut() else {
            let plans = self
                .plans
                .take()
                .vortex_expect("an uncompiled projection still has its plans");
            let scan = Scan::try_new(
                plans.session,
                plans.projection,
                vec![Split {
                    rows: self.selected.scope.rows.clone(),
                    mask: self.selected.mask.clone(),
                }],
            )?
            .with_row_offset(plans.row_offset)
            .with_value_cache(plans.values);
            self.adaptor = Some(ScanAdaptor::new(scan, plans.locations, plans.source));
            return Ok(MorselOutput::Continue);
        };
        if adaptor.state() == State::Done {
            return self.finish();
        }
        Ok(match adaptor.compute()? {
            AdaptorStep::Yield => MorselOutput::Continue,
            AdaptorStep::NeedsIO(batch) => MorselOutput::NeedsIO(batch),
            AdaptorStep::Piece(piece) => {
                self.pieces.push(piece);
                MorselOutput::Continue
            }
        })
    }
}
