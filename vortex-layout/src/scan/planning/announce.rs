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

use crate::plan::PlanRef;
use crate::scan::planning::FilterPlans;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::SelectedRows;
use crate::scan::planning::plan_selected;
use crate::scan::v2::prefetch::plan_segments;
use crate::scan::v2::splits::projection_splits;

/// Announces the segments a split is likely to read, then hands the split to `next`.
///
/// An announcement reads nothing itself, but the IO service can coalesce every announced segment
/// near one that it does read, as the layout reader's splits do by registering their reads when
/// their futures are built.
pub struct AnnouncePlanner {
    plans: Option<ScanPlans>,
    filter: Option<FilterPlans>,
    /// The split, until it is handed to `next`.
    selected: Option<SelectedRows>,
    announced: bool,
    next: Option<Continuation>,
}

enum Continuation {
    Next(Next<SelectedRows>),
    Split(Option<PlanRef>),
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
            plans: Some(plans),
            filter,
            selected: Some(selected),
            announced: false,
            next: Some(Continuation::Next(next)),
        }
    }

    pub(super) fn for_split(
        plans: ScanPlans,
        pruning: Option<PlanRef>,
        filter: Option<FilterPlans>,
        selected: SelectedRows,
    ) -> Self {
        Self {
            plans: Some(plans),
            filter,
            selected: Some(selected),
            announced: false,
            next: Some(Continuation::Split(pruning)),
        }
    }

    fn announce(&self, selected: &SelectedRows) -> VortexResult<IoBatch> {
        let plans = self
            .plans
            .as_ref()
            .ok_or_else(|| vortex_err!("AnnouncePlanner has no plans"))?;
        let rows = &selected.scope.rows;
        let mut ids = Vec::new();
        let mut consumer = Vec::new();
        let mut announce = |plan: &PlanRef, rows| -> VortexResult<()> {
            consumer.clear();
            plan_segments(plan, rows, &mut consumer)?;
            consumer.sort_unstable();
            consumer.dedup();
            ids.extend(consumer.iter().copied());
            Ok(())
        };
        for filter in self.filter.iter().flat_map(FilterPlans::plans) {
            announce(filter, rows.clone())?;
        }
        // V1 constructs projection futures before filtering, so a shared segment read remains
        // live for each pending projection. Keep one registration per potential consumer.
        for rows in projection_splits(&plans.projection_starts, rows.clone()) {
            // The input selection is already known, even before predicates run. Empty chunks
            // cannot produce a projection consumer and must not keep read registrations alive.
            let start = usize::try_from(rows.start - selected.scope.rows.start)?;
            let end = usize::try_from(rows.end - selected.scope.rows.start)?;
            if selected.mask.slice(start..end).all_false() {
                continue;
            }
            announce(&plans.projection, rows)?;
        }
        ids.into_iter()
            .enumerate()
            .map(|(index, id)| {
                let location = plans
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
        let next = match self.next.take() {
            Some(Continuation::Next(next)) => next(selected)?,
            Some(Continuation::Split(pruning)) => {
                let plans = self
                    .plans
                    .take()
                    .ok_or_else(|| vortex_err!("AnnouncePlanner has no plans"))?;
                plan_selected(plans, pruning, self.filter.take(), selected)?
            }
            None => vortex_bail!("AnnouncePlanner has no continuation"),
        };
        Ok(PlannerOutput::Planner(scope, next))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_buffer::Alignment;
    use vortex_mask::Mask;
    use vortex_scan::planning::planner::WorkScope;
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
    fn announcements_skip_chunks_without_selected_rows() -> VortexResult<()> {
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
        let selected = SelectedRows {
            scope: WorkScope {
                file_ordinal: 0,
                rows: 100..2900,
            },
            mask: Mask::from_iter((100..2900).map(|row| row == 150 || row == 2800)),
        };
        let mut planner = AnnouncePlanner::for_split(
            ScanPlans {
                session: new_session(),
                locations: Arc::clone(&locations),
                projection: ConcatPlan::try_new(dtype, chunks)?.into_plan(),
                projection_starts: Arc::from([0, 1000, 2000]),
                row_offset: 0,
                decoded: DecodeCache::disabled(),
            },
            None,
            None,
            selected,
        );
        let PlannerOutput::NeedsIO(requests) = planner.compute()? else {
            vortex_bail!("expected announcements");
        };
        assert!(
            requests
                .iter()
                .all(|request| request.intent == IoIntent::Announce)
        );
        let targets: Vec<_> = requests.iter().map(|request| request.target).collect();
        assert_eq!(targets, vec![locations[0].target(), locations[2].target()]);
        Ok(())
    }
}
