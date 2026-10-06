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
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoService;
use vortex_mask::Mask;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

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
use crate::scan::planning::plan_split;
use crate::scan::scan_builder;
use crate::scan::splits::Splits;
use crate::scan::splits::attempt_split_ranges;
use crate::scan::v2::ScanBuilder;
use crate::scan::v2::ScanFile;
use crate::scan::v2::conjuncts::filter_after_eval;
use crate::scan::v2::conjuncts::group_conjuncts;
use crate::scan::v2::file::SharedFile;
use crate::scan::v2::file::shared_file;
use crate::scan::v2::io::SegmentScanIo;
use crate::scan::v2::io::segment_ranges;
use crate::scan::v2::pruning::PrunedFile;
use crate::scan::v2::pruning::file_pruning_enabled;
use crate::scan::v2::pruning::prune_file;
use crate::scan::v2::share::unshare_unread;
use crate::scan::v2::split::SplitPlan;
use crate::scan::v2::splits::chunk_starts;
use crate::scan::v2::splits::filter_split_boundaries;
use crate::scan::v2::splits::max_split_rows;

/// Computes split ranges for `builder` and returns an executable scan over `file`, the file the
/// builder's reader was opened over.
///
/// The file's layout is lowered once into a physical plan, and the filter and projection are
/// planned and optimized over it. Every split then runs those plans through the planning
/// protocol, and never calls the layout reader's evaluation methods. The builder's reader is only
/// used to choose the splits.
///
/// The replacement for the default [`ScanBuilder::prepare`](scan_builder::ScanBuilder::prepare).
/// Equivalent to copying `builder` into a [`ScanBuilder`] and preparing that.
pub fn prepare<A: 'static + Send>(
    builder: scan_builder::ScanBuilder<A>,
    file: ScanFile,
) -> VortexResult<RepeatedScanV2<A>> {
    ScanBuilder::from_default(builder, file).prepare()
}

/// Plans `builder`'s scan over its file. See [`prepare`].
pub(super) fn prepare_scan<A: 'static + Send>(
    builder: ScanBuilder<A>,
) -> VortexResult<RepeatedScanV2<A>> {
    let dtype = builder.dtype()?;

    if builder.filter.is_some() && builder.limit.is_some() {
        vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
    }

    let shared = shared_file(&builder.layout_reader, builder.file)?;
    let prepared = shared.prepare_plans(&builder.projection, &builder.filter, &builder.session)?;
    let plans = ScanPlans {
        session: builder.session,
        locations: Arc::clone(&shared.file.locations),
        projection: prepared.projection.clone(),
        projection_starts: Arc::clone(&prepared.projection_starts),
        row_offset: builder.row_offset,
        decoded: DecodeCache::default(),
    };

    let splits = match attempt_split_ranges(&builder.selection, builder.row_range.as_ref()) {
        Some(ranges) => Splits::Ranges(ranges),
        None => Splits::Natural(
            filter_split_boundaries(
                &prepared.filter_starts,
                &prepared.all_starts,
                0..shared.root.row_count(),
                max_split_rows(
                    shared.root.row_count(),
                    get_available_parallelism().unwrap_or(1),
                ),
            )
            .into(),
        ),
    };

    Ok(RepeatedScanV2 {
        pruning: prepared.pruning.clone(),
        plans,
        filter: prepared.filter.clone(),
        io: shared.file.io.clone().unwrap_or_else(|| {
            Arc::new(SegmentScanIo::new(
                Arc::clone(&shared.file.segments),
                segment_ranges(&shared.file.locations),
            ))
        }),
        row_range: builder.row_range,
        selection: builder.selection,
        splits,
        map_fn: builder.map_fn,
        limit: builder.limit,
        dtype,
    })
}

/// Expression plans and chunk boundaries, independent of a partition's row range and selection.
pub(super) struct PreparedPlans {
    projection: PlanRef,
    filter: Option<FilterPlans>,
    pruning: Option<PlanRef>,
    projection_starts: Arc<[u64]>,
    filter_starts: Arc<[u64]>,
    all_starts: Arc<[u64]>,
}

impl PreparedPlans {
    pub(super) fn new(
        shared: &SharedFile,
        projection: &BoundExpression,
        filter: &Option<BoundExpression>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let root = shared.root.clone();
        let plan = |expression| optimize(plan_row_idx_expression(expression, root.clone())?);
        let filter_expression = filter;
        let filter = filter
            .clone()
            .map(|filter| {
                let conjuncts = group_conjuncts(FilterExpr::new(filter).conjuncts())?;
                VortexResult::Ok(Arc::new(FilterExpr::from_conjuncts(conjuncts)))
            })
            .transpose()?;
        // The projection comes first, then one plan per conjunct.
        let mut all_plans = vec![plan(projection.clone())?];
        for conjunct in filter.iter().flat_map(|filter| filter.conjuncts()) {
            all_plans.push(filter_after_eval(plan(conjunct.clone())?)?);
        }
        let mut all_plans = unshare_unread(all_plans)?;
        let conjunct_plans = all_plans.split_off(1);
        let projection = all_plans.remove(0);
        // Filter splits are sized by the chunks of the columns the filter reads, or of those the
        // projection reads when there is no filter, and each is cut into projection splits.
        let projection_starts: Arc<[u64]> = chunk_starts([&projection])?.into();
        let filter_starts = if conjunct_plans.is_empty() {
            projection_starts.to_vec()
        } else {
            chunk_starts(&conjunct_plans)?
        };
        let mut all_starts = filter_starts.clone();
        all_starts.extend_from_slice(&projection_starts);
        all_starts.sort_unstable();
        all_starts.dedup();
        let filter = filter
            .map(|filter| FilterPlans::conjuncts(filter, conjunct_plans))
            .transpose()?;
        let pruning = filter_expression
            .as_ref()
            .map(|filter| pruning_plan(filter, &shared.zones, session))
            .transpose()?
            .flatten();
        Ok(Self {
            projection,
            filter,
            pruning,
            projection_starts,
            filter_starts: filter_starts.into(),
            all_starts: all_starts.into(),
        })
    }
}

/// Joins a filter split's batches into one array without copying them: a struct whose fields are
/// chunked over the batches when every batch is a struct of the scan's non-nullable struct dtype,
/// and a chunked array otherwise.
fn join_batches(mut batches: Vec<ArrayRef>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    if batches.len() <= 1 {
        return Ok(batches.pop());
    }
    if let DType::Struct(fields, Nullability::NonNullable) = dtype
        && let Some(structs) = batches
            .iter()
            .map(|batch| batch.as_opt::<Struct>().map(|batch| batch.into_owned()))
            .collect::<Option<Vec<StructArray>>>()
    {
        let len = structs.iter().map(|batch| batch.len()).sum();
        let columns = (0..fields.nfields())
            .map(|index| {
                let chunks = structs
                    .iter()
                    .map(|batch| batch.unmasked_field(index).clone())
                    .collect::<Vec<_>>();
                let dtype = chunks[0].dtype().clone();
                Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let array =
            StructArray::try_new(fields.names().clone(), columns, len, Validity::NonNullable)?;
        return Ok(Some(array.into_array()));
    }
    Ok(Some(
        ChunkedArray::try_new(batches, dtype.clone())?.into_array(),
    ))
}

/// A prepared scan that turns row ranges into one task per filter split.
///
/// The replacement for [`RepeatedScan`](crate::scan::repeated_scan::RepeatedScan).
pub struct RepeatedScanV2<A: 'static + Send> {
    /// Proves rows can't match the filter from zone statistics, when the filter allows it.
    pruning: Option<PlanRef>,
    plans: ScanPlans,
    filter: Option<FilterPlans>,
    /// Serves the splits' reads.
    io: Arc<dyn IoService>,
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

    /// The service the scan's splits read through: one session per planning run root.
    pub fn io(&self) -> &Arc<dyn IoService> {
        &self.io
    }

    /// Converts a projected array into the scan's output.
    pub fn map(&self, array: ArrayRef) -> VortexResult<A> {
        (self.map_fn)(array)
    }

    /// Joins the batches of one filter split, in row order, into one array without copying them.
    /// Returns `None` when there are none.
    pub fn join(&self, batches: Vec<ArrayRef>) -> VortexResult<Option<ArrayRef>> {
        join_batches(batches, self.plans.projection.dtype())
    }

    /// Returns one task per filter split of `row_range` that has selected rows.
    ///
    /// A task returns its projection splits' batches as one array, whose fields are chunked over
    /// them, so nothing is copied.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        self.execute_splits(self.split_plans(row_range)?)
    }

    /// Like [`execute`](Self::execute), with file pruning before split creation when
    /// `VORTEX_SCAN_FILE_PRUNING=1`. Statistics IO completes before this returns the data tasks.
    pub async fn execute_pruned(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        self.execute_splits(self.execution_splits(row_range).await?)
    }

    fn execute_splits(
        &self,
        splits: Vec<SplitPlan>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let dtype = self.plans.projection.dtype().clone();
        Ok(splits
            .into_iter()
            .map(|split| {
                let io = Arc::clone(&self.io);
                let map_fn = Arc::clone(&self.map_fn);
                let dtype = dtype.clone();
                let run = split.run(io);
                async move {
                    join_batches(run.await?, &dtype)?
                        .map(|array| map_fn(array))
                        .transpose()
                }
                .boxed()
            })
            .collect())
    }

    /// Returns one task per filter split of `row_range` that has selected rows, each returning
    /// one batch per projection split with selected rows, in row order.
    pub fn execute_batches(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Vec<A>>>>> {
        self.execute_batch_splits(self.split_plans(row_range)?)
    }

    pub(super) async fn execute_batches_pruned(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Vec<A>>>>> {
        self.execute_batch_splits(self.execution_splits(row_range).await?)
    }

    fn execute_batch_splits(
        &self,
        splits: Vec<SplitPlan>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Vec<A>>>>> {
        Ok(splits
            .into_iter()
            .map(|split| {
                let io = Arc::clone(&self.io);
                let map_fn = Arc::clone(&self.map_fn);
                let run = split.run(io);
                async move { run.await?.into_iter().map(|array| map_fn(array)).collect() }.boxed()
            })
            .collect())
    }

    async fn execution_splits(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<SplitPlan>> {
        if file_pruning_enabled() {
            self.pruned_split_plans(row_range).await
        } else {
            self.split_plans(row_range)
        }
    }

    /// Prunes the whole file's zone statistics before constructing or announcing data splits.
    /// Surviving splits retain pruning only when dynamic bounds can change during execution.
    pub async fn pruned_split_plans(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<SplitPlan>> {
        let pruned = match &self.pruning {
            Some(proof) => Some(prune_file(&self.plans, proof.clone(), &self.io).await?),
            None => None,
        };
        self.split_plans_with_pruning(row_range, pruned.as_ref())
    }

    /// The filter splits of `row_range` that have selected rows, in row order, each ready to be
    /// admitted to a planning run that reads through a session of [`io`](Self::io).
    pub fn split_plans(&self, row_range: Option<Range<u64>>) -> VortexResult<Vec<SplitPlan>> {
        self.split_plans_with_pruning(row_range, None)
    }

    fn split_plans_with_pruning(
        &self,
        row_range: Option<Range<u64>>,
        pruned: Option<&PrunedFile>,
    ) -> VortexResult<Vec<SplitPlan>> {
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

        let pruning = if pruned.is_some_and(|pruned| !pruned.dynamic) {
            None
        } else {
            self.pruning.clone()
        };
        let mut limit = self.limit;
        let mut splits = Vec::new();
        for range in ranges {
            let (range, surviving) = match pruned {
                Some(pruned) => match pruned.select(range)? {
                    Some((range, mask)) => (range, Some(mask)),
                    None => continue,
                },
                None => (range, None),
            };
            let row_mask = self.selection.row_mask(&range);
            let selected = match surviving {
                Some(surviving) => row_mask.mask() & &surviving,
                None => row_mask.mask().clone(),
            };
            if selected.all_false() {
                continue;
            }
            let mask = match (&self.filter, limit.as_mut()) {
                (None, Some(0)) => Mask::new_false(selected.len()),
                (None, Some(l)) => {
                    let true_count = selected.true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    *l -= mask_limit as u64;
                    selected.limit(mask_limit)
                }
                _ => selected,
            };
            if !mask.all_false() {
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: row_mask.row_range(),
                };
                splits.push(SplitPlan {
                    root: plan_split(
                        self.plans.clone(),
                        pruning.clone(),
                        self.filter.clone(),
                        scope.clone(),
                        mask,
                    )?,
                    scope,
                });
            }
            if limit.is_some_and(|l| l == 0) {
                break;
            }
        }

        tracing::debug!(
            target: "vortex_layout::scan::v2::pruning",
            splits = splits.len(),
            file_pruned = pruned.is_some(),
            "filter splits prepared"
        );
        Ok(splits)
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
    zones: &PlanRef,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    let Some(predicate) = filter.falsify(session)? else {
        return Ok(None);
    };
    let plan = optimize(EvalPlan::try_new(predicate, zones.clone())?.into_plan())?;
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

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}
