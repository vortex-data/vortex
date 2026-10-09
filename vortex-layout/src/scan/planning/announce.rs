// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

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
use crate::scan::planning::ProjectionPlanner;
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
    SparseProjection { rows: Option<Range<u64>> },
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

    pub(super) fn for_sparse_projection(plans: ScanPlans, selected: SelectedRows) -> Self {
        // Input index masks retain their indices unless limited. Keep contiguous ranges on
        // the projection path that preserves their chunk slices.
        let has_gaps = selected
            .mask
            .first()
            .zip(selected.mask.last())
            .is_some_and(|(first, last)| last - first + 1 > selected.mask.true_count());
        if !has_gaps {
            return Self::for_split(plans, None, None, selected);
        }
        Self {
            next: Some(Continuation::SparseProjection { rows: None }),
            ..Self::for_split(plans, None, None, selected)
        }
    }

    fn announce(
        &self,
        selected: &SelectedRows,
        project_before_filter: bool,
    ) -> VortexResult<(IoBatch, Option<Range<u64>>)> {
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
        let mut projection_rows: Option<Range<u64>> = None;
        if self.filter.is_none() || project_before_filter {
            for rows in projection_splits(&plans.projection_starts, rows.clone()) {
                // The input selection is already known, even before predicates run. Empty chunks
                // cannot produce a projection consumer and must not keep read registrations alive.
                let start = usize::try_from(rows.start - selected.scope.rows.start)?;
                let end = usize::try_from(rows.end - selected.scope.rows.start)?;
                if selected.mask.slice(start..end).all_false() {
                    continue;
                }
                projection_rows.get_or_insert_with(|| rows.clone()).end = rows.end;
                announce(&plans.projection, rows)?;
            }
        }
        let batch = ids
            .into_iter()
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
            .collect::<VortexResult<IoBatch>>()?;
        Ok((batch, projection_rows))
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
        static PROJECT_BEFORE_FILTER: LazyLock<bool> = LazyLock::new(|| {
            !std::env::var("VORTEX_SCAN_PROJECT_ANNOUNCE").is_ok_and(|value| value == "0")
        });
        if !self.announced {
            self.announced = true;
            let (batch, rows) = match &self.selected {
                Some(selected) => self.announce(selected, *PROJECT_BEFORE_FILTER)?,
                None => (Vec::new(), None),
            };
            if let Some(Continuation::SparseProjection { rows: selected }) = &mut self.next {
                *selected = rows;
            }
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
            Some(Continuation::SparseProjection { rows }) => {
                let mut plans = self
                    .plans
                    .take()
                    .ok_or_else(|| vortex_err!("AnnouncePlanner has no plans"))?;
                let mut selected = selected;
                if let Some(rows) = rows {
                    // Reuse the populated cuts found by announcements to trim empty edges,
                    // especially when only one cut contains rows.
                    let offset = selected.scope.rows.start;
                    let start = usize::try_from(rows.start - offset)?;
                    let end = usize::try_from(rows.end - offset)?;
                    selected.mask = selected.mask.slice(start..end);
                    selected.scope.rows = rows;
                }
                if selected.mask.true_count() <= selected.mask.len() / 8 {
                    // The graph follows each column's own chunk boundaries. Keep the original
                    // boundaries for announcements, which must still skip unselected chunks.
                    plans.projection_starts = Default::default();
                }
                Box::new(ProjectionPlanner::new(plans, selected))
            }
            None => vortex_bail!("AnnouncePlanner has no continuation"),
        };
        Ok(PlannerOutput::Planner(scope, next))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use rstest::rstest;
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

    #[rstest]
    #[case::default(false, &[150, 2800], None)]
    #[case::one_cut(true, &[150], Some(vec![100..1000]))]
    #[case::two_rows_in_one_cut(true, &[150, 160], Some(vec![100..1000]))]
    #[case::two_cuts(true, &[150, 2800], Some(vec![100..2900]))]
    #[case::contiguous_across_cuts(true, &[999, 1000], Some(vec![100..1000, 1000..2000]))]
    fn announcements_skip_chunks_without_selected_rows(
        #[case] sparse: bool,
        #[case] indices: &[usize],
        #[case] morsel_rows: Option<Vec<Range<u64>>>,
    ) -> VortexResult<()> {
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
            mask: Mask::from_indices(2800, indices.iter().map(|row| row - 100)),
        };
        let plans = ScanPlans {
            session: new_session(),
            locations: Arc::clone(&locations),
            projection: ConcatPlan::try_new(dtype, chunks)?.into_plan(),
            projection_starts: Arc::from([0, 1000, 2000]),
            row_offset: 0,
            decoded: DecodeCache::disabled(),
        };
        let mut planner = if sparse {
            AnnouncePlanner::for_sparse_projection(plans, selected)
        } else {
            AnnouncePlanner::for_split(plans, None, None, selected)
        };
        let PlannerOutput::NeedsIO(requests) = planner.compute()? else {
            vortex_bail!("expected announcements");
        };
        assert!(
            requests
                .iter()
                .all(|request| request.intent == IoIntent::Announce)
        );
        let targets: Vec<_> = requests.iter().map(|request| request.target).collect();
        let mut expected: Vec<_> = indices
            .iter()
            .map(|row| locations[row / 1000].target())
            .collect();
        expected.dedup();
        assert_eq!(targets, expected);
        if let Some(morsel_rows) = morsel_rows {
            let PlannerOutput::Planner(_, mut projection) = planner.compute()? else {
                vortex_bail!("expected projection planner");
            };
            for rows in morsel_rows {
                let mut output = projection.compute()?;
                if matches!(output, PlannerOutput::NeedsIO(_)) {
                    output = projection.compute()?;
                }
                let PlannerOutput::Morsel(scope, _) = output else {
                    vortex_bail!("expected a projection morsel");
                };
                assert_eq!(scope.rows, rows);
            }
            assert_eq!(projection.state(), State::Done);
        }
        Ok(())
    }

    #[rstest]
    #[case::early_projection(true, true, &[0, 0, 1])]
    #[case::deferred_projection(true, false, &[0])]
    #[case::unfiltered(false, false, &[0, 1])]
    fn announcements_keep_filter_segments_and_unfiltered_projection(
        #[case] filtered: bool,
        #[case] early_projection: bool,
        #[case] expected: &[usize],
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let chunks: Vec<_> = (0..2)
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
        let filter = filtered.then(|| FilterPlans::single(chunks[0].clone()));
        let locations: Arc<[_]> = (0..2)
            .map(|offset| SegmentLocation {
                offset,
                length: 1,
                alignment: Alignment::none(),
            })
            .collect();
        let selected = SelectedRows {
            scope: WorkScope {
                file_ordinal: 0,
                rows: 0..2000,
            },
            mask: Mask::new_true(2000),
        };
        let planner = AnnouncePlanner::for_split(
            ScanPlans {
                session: new_session(),
                locations: Arc::clone(&locations),
                projection: ConcatPlan::try_new(dtype, chunks)?.into_plan(),
                projection_starts: Arc::from([0, 1000]),
                row_offset: 0,
                decoded: DecodeCache::disabled(),
            },
            None,
            filter,
            selected,
        );
        let (batch, _) = planner.announce(planner.selected.as_ref().unwrap(), early_projection)?;
        let actual: Vec<_> = batch.iter().map(|request| request.target).collect();
        let expected: Vec<_> = expected
            .iter()
            .map(|&index| locations[index].target())
            .collect();
        assert_eq!(actual, expected);
        assert!(
            batch
                .iter()
                .all(|request| request.intent == IoIntent::Announce)
        );
        Ok(())
    }
}
