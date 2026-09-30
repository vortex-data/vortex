// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;

use crate::scan::planning::FilterPlans;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::SelectedRows;
use crate::scan::v2::prefetch::plan_segments;

/// Announces the segments a split is likely to read, then hands the split to `next`.
///
/// An announcement reads nothing itself, but the IO service can coalesce every announced segment
/// near one that it does read, as the layout reader's splits do by registering their reads when
/// their futures are built.
pub struct AnnouncePlanner {
    plans: ScanPlans,
    filter: Option<FilterPlans>,
    /// The split, until it is handed to `next`.
    selected: Option<SelectedRows>,
    announced: bool,
    next: Next<SelectedRows>,
}

impl AnnouncePlanner {
    /// Creates a planner announcing the reads of `filter` and the projection over `selected`.
    pub fn new(
        plans: ScanPlans,
        filter: Option<FilterPlans>,
        selected: SelectedRows,
        next: Next<SelectedRows>,
    ) -> Self {
        Self {
            plans,
            filter,
            selected: Some(selected),
            announced: false,
            next,
        }
    }

    fn announce(&self, selected: &SelectedRows) -> VortexResult<IoBatch> {
        let rows = &selected.scope.rows;
        let mut ids = Vec::new();
        for filter in self.filter.iter().flat_map(FilterPlans::plans) {
            plan_segments(filter, rows.clone(), &mut ids)?;
        }
        plan_segments(&self.plans.projection, rows.clone(), &mut ids)?;
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
                    intent: IoIntent::Announce,
                    request: IoRequestId(u32::try_from(index)?),
                    target: location.target(),
                })
            })
            .collect()
    }
}

impl IoConsumer for AnnouncePlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for AnnouncePlanner {
    fn state(&self) -> State {
        if self.selected.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.announced {
            self.announced = true;
            let batch = match &self.selected {
                Some(selected) => self.announce(selected)?,
                None => Vec::new(),
            };
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        let Some(selected) = self.selected.take() else {
            vortex_bail!("AnnouncePlanner: compute called after Done");
        };
        let scope = selected.scope.clone();
        Ok(PlannerOutput::Planner(scope, (self.next)(selected)?))
    }
}
