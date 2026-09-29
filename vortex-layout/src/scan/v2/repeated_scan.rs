// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::Zoned;
use crate::plan::exec::DecodeCache;
use crate::plan::optimize;
use crate::plan::plan_row_idx_expression;
use crate::scan::filter::FilterExpr;
use crate::scan::planning::FilterPlans;
use crate::scan::planning::ScanPlans;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::scan_builder::referenced_field_masks;
use crate::scan::splits::Splits;
use crate::scan::splits::attempt_split_ranges;
use crate::scan::v2::ScanFile;
use crate::scan::v2::conjuncts::group_conjuncts;
use crate::scan::v2::io::SegmentRanges;
use crate::scan::v2::io::segment_ranges;
use crate::scan::v2::lower::lower;
use crate::scan::v2::lower::lower_with_zones;
use crate::scan::v2::prefetch::plan_segments;
use crate::scan::v2::split::SplitTask;
use crate::segments::SegmentFuture;
use crate::segments::SegmentSource;

/// Computes split ranges for `builder` and returns an executable scan over `file`, the file the
/// builder's reader was opened over.
///
/// The file's layout is lowered once into a physical plan, and the filter and projection are
/// planned and optimized over it. Every split then runs those plans through the planning
/// protocol, and never calls the layout reader's evaluation methods. The builder's reader is only
/// used to choose the splits.
///
/// The replacement for [`ScanBuilder::prepare`].
pub fn prepare<A: 'static + Send>(
    builder: ScanBuilder<A>,
    file: ScanFile,
) -> VortexResult<RepeatedScanV2<A>> {
    let dtype = builder.dtype()?;
    let parts = builder.into_parts();

    if parts.filter.is_some() && parts.limit.is_some() {
        vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
    }

    let layout_reader = parts.layout_reader;
    let root = lower(&file.layout)?;
    let plan = |expression| optimize(plan_row_idx_expression(expression, root.clone())?);
    let filter = parts
        .filter
        .clone()
        .map(|filter| {
            let conjuncts = group_conjuncts(FilterExpr::new(filter).conjuncts())?;
            let filter = Arc::new(FilterExpr::from_conjuncts(conjuncts));
            let plans = filter
                .conjuncts()
                .iter()
                .cloned()
                .map(plan)
                .collect::<VortexResult<Vec<_>>>()?;
            VortexResult::Ok(FilterPlans::conjuncts(filter, plans))
        })
        .transpose()?;
    let pruning = parts
        .filter
        .as_ref()
        .map(|filter| pruning_plan(filter, &file, &parts.session))
        .transpose()?
        .flatten();
    let plans = ScanPlans {
        session: parts.session,
        locations: Arc::clone(&file.locations),
        projection: plan(parts.projection.clone())?,
        row_offset: parts.row_offset,
        decoded: DecodeCache::default(),
    };

    let splits =
        if let Some(ranges) = attempt_split_ranges(&parts.selection, parts.row_range.as_ref()) {
            Splits::Ranges(ranges)
        } else if let Some(boundaries) = parts.natural_splits {
            Splits::Natural(boundaries)
        } else {
            let field_mask = referenced_field_masks(&parts.projection, parts.filter.as_ref())?;
            let split_range = parts
                .row_range
                .clone()
                .unwrap_or_else(|| 0..layout_reader.row_count());
            Splits::Natural(
                parts
                    .split_by
                    .splits(layout_reader.as_ref(), &split_range, &field_mask)?
                    .into(),
            )
        };

    Ok(RepeatedScanV2 {
        pruning,
        plans,
        filter,
        ranges: segment_ranges(&file.locations),
        segments: file.segments,
        row_range: parts.row_range,
        selection: parts.selection,
        splits,
        map_fn: parts.map_fn,
        limit: parts.limit,
        dtype,
    })
}

/// A prepared scan that turns row ranges into one task per split.
///
/// The replacement for [`RepeatedScan`](crate::scan::repeated_scan::RepeatedScan).
pub struct RepeatedScanV2<A: 'static + Send> {
    /// Proves rows can't match the filter from zone statistics, when the filter allows it.
    pruning: Option<PlanRef>,
    plans: ScanPlans,
    filter: Option<FilterPlans>,
    ranges: SegmentRanges,
    segments: Arc<dyn SegmentSource>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    splits: Splits,
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    limit: Option<u64>,
    dtype: DType,
}

impl<A: 'static + Send> RepeatedScanV2<A> {
    /// The dtype of the projected arrays.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns one task per split of `row_range` that has selected rows.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        let ranges = match &self.splits {
            Splits::Natural(vec) => {
                let splits_iter = match row_range {
                    None => Either::Left(vec.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Ok(Vec::new());
                        }
                        let lo = vec.partition_point(|&x| x <= range.start);
                        let hi = vec.partition_point(|&x| x < range.end);
                        Either::Right(
                            iter::once(range.start)
                                .chain(vec[lo..hi].iter().copied())
                                .chain(iter::once(range.end)),
                        )
                    }
                };
                Either::Left(splits_iter.tuple_windows().map(|(start, end)| start..end))
            }
            Splits::Ranges(ranges) => Either::Right(match row_range {
                None => Either::Left(ranges.iter().cloned()),
                Some(range) => {
                    if range.is_empty() {
                        return Ok(Vec::new());
                    }
                    Either::Right(ranges.iter().filter_map(move |r| {
                        let start = cmp::max(r.start, range.start);
                        let end = cmp::min(r.end, range.end);
                        (start < end).then_some(start..end)
                    }))
                }
            }),
        };

        let mut limit = self.limit;
        let mut tasks = Vec::new();
        for range in ranges {
            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }
            let mask = match (&self.filter, limit.as_mut()) {
                (None, Some(0)) => Mask::new_false(row_mask.mask().len()),
                (None, Some(l)) => {
                    let true_count = row_mask.mask().true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    *l -= mask_limit as u64;
                    row_mask.mask().clone().limit(mask_limit)
                }
                _ => row_mask.mask().clone(),
            };
            let range = row_mask.row_range();
            let task = SplitTask {
                plans: self.plans.clone(),
                pruning: self.pruning.clone(),
                filter: self.filter.clone(),
                segments: Arc::clone(&self.segments),
                ranges: Arc::clone(&self.ranges),
                registered: self.register(&range)?,
                range,
                mask,
                map_fn: Arc::clone(&self.map_fn),
            };
            tasks.push(task.run().boxed());
            if limit.is_some_and(|l| l == 0) {
                break;
            }
        }

        Ok(tasks)
    }
}

/// Plans the zone-statistics proof that `filter` is false, over a copy of the file's plan that keeps
/// its zones.
///
/// The filter is falsified into a predicate over statistics, which the optimizer pushes down to
/// the zoned columns it reads and rewrites into pruning plans over their zone tables. Returns
/// `None` when the filter cannot be falsified, or when any part of the proof would still read
/// column data rather than zone statistics.
fn pruning_plan(
    filter: &BoundExpression,
    file: &ScanFile,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    let Some(predicate) = filter.falsify(session)? else {
        return Ok(None);
    };
    let root = lower_with_zones(&file.layout)?;
    let plan = optimize(EvalPlan::try_new(predicate, root)?.into_plan())?;
    Ok(reads_only_zones(&plan)?.then_some(plan))
}

/// Whether every leaf of `plan` is a pruning plan over zone statistics.
fn reads_only_zones(plan: &PlanRef) -> VortexResult<bool> {
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        return Ok(zoned.is_pruning());
    }
    if !plan.is::<Eval>() && !plan.is::<Pack>() {
        return Ok(false);
    }
    for child in plan.children().iter() {
        if !reads_only_zones(&child?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<A: 'static + Send> RepeatedScanV2<A> {
    /// Registers the segments a split over `range` is likely to read, as the layout reader does
    /// when it builds a split's futures. Nothing is read until the split asks for a segment, but
    /// the source can coalesce every registered segment near one that it does read.
    fn register(&self, range: &Range<u64>) -> VortexResult<Vec<SegmentFuture>> {
        let mut ids = Vec::new();
        for filter in self.filter.iter().flat_map(FilterPlans::plans) {
            plan_segments(filter, range.clone(), &mut ids)?;
        }
        plan_segments(&self.plans.projection, range.clone(), &mut ids)?;
        ids.sort_unstable();
        ids.dedup();
        Ok(ids
            .into_iter()
            .map(|id| self.segments.request(id))
            .collect())
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}
