// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::metrics::ScanCounters;
use crate::scan::metrics::ScanMetrics;

/// This future returns the projected rows of one split or `None`.
pub(crate) type SplitFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// The caller must exclude empty masks before task construction.
pub(crate) fn split_exec(ctx: &Arc<TaskContext>, read_mask: RowMask) -> VortexResult<SplitFuture> {
    if ctx.counters.is_some() {
        split_exec_inner::<true>(ctx, read_mask)
    } else {
        split_exec_inner::<false>(ctx, read_mask)
    }
}

fn split_exec_inner<const TRACK_COUNTERS: bool>(
    ctx: &Arc<TaskContext>,
    read_mask: RowMask,
) -> VortexResult<SplitFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if ctx.filter.is_none() {
        return split_projection::<TRACK_COUNTERS>(ctx, &row_range, row_mask);
    }
    let filter_mask = build_filter_mask::<TRACK_COUNTERS>(ctx, &row_range, row_mask);

    // Construct the projection before the filter has run so the reader can prefetch its I/O.
    let projection =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;
    Ok(attempted::<TRACK_COUNTERS>(ctx, async move {
        if filter_mask.await?.all_false() {
            return Ok(None);
        }
        projection.await.map(Some)
    }))
}

/// Project the rows selected by an already-evaluated `mask`.
fn split_projection<const TRACK_COUNTERS: bool>(
    ctx: &Arc<TaskContext>,
    row_range: &Range<u64>,
    mask: Mask,
) -> VortexResult<SplitFuture> {
    if mask.all_false() {
        return Ok(futures::future::ready(Ok(None)).boxed());
    }
    let projection =
        ctx.reader
            .projection_evaluation(row_range, &ctx.projection, MaskFuture::ready(mask))?;
    Ok(attempted::<TRACK_COUNTERS>(
        ctx,
        projection.map(|array| array.map(Some)),
    ))
}

/// With counters enabled, this future counts the split when its task first runs.
/// It holds the context until it stops, so registry totals include its counts.
fn attempted<const TRACK_COUNTERS: bool>(
    ctx: &Arc<TaskContext>,
    split: impl Future<Output = VortexResult<Option<ArrayRef>>> + Send + 'static,
) -> SplitFuture {
    if !TRACK_COUNTERS {
        return split.boxed();
    }
    let ctx = Arc::clone(ctx);
    async move {
        ctx.count::<TRACK_COUNTERS>(ScanCounters::attempt_split);
        split.await
    }
    .boxed()
}

fn build_filter_mask<const TRACK_COUNTERS: bool>(
    ctx: &Arc<TaskContext>,
    row_range: &Range<u64>,
    row_mask: Mask,
) -> MaskFuture {
    let ctx = Arc::clone(ctx);
    let row_range = row_range.clone();
    MaskFuture::new(row_mask.len(), async move {
        let reader = &ctx.reader;
        let filter = ctx.filter.as_ref().vortex_expect("filter exists");
        let mut mask = row_mask;
        let mut dynamic_versions = vec![None; filter.conjuncts().len()];

        // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
        for (idx, version) in dynamic_versions.iter_mut().enumerate() {
            // Store the latest version of the dynamic expression prior to pruning.
            // We will re-run the pruning later if the version has changed in the meantime.
            *version = filter.dynamic_updates(idx).map(|du| du.version());

            mask = prune_conjunct::<TRACK_COUNTERS>(&ctx, filter, idx, *version, &row_range, mask)
                .await?;
            if mask.all_false() {
                return Ok(mask);
            }
        }

        // Now we loop through the conjuncts in the preferred order and evaluate them.
        let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
        while let Some(idx) = filter.next_conjunct(&remaining) {
            remaining.set(idx, false);
            let conjunct = &filter.conjuncts()[idx];

            // If the dynamic expression has changed since pruning, re-run the pruning.
            // Store the dynamic update once to avoid TOCTOU race condition
            let current_version = filter.dynamic_updates(idx).map(|du| du.version());
            if let Some(dv) = current_version
                && dynamic_versions[idx].is_none_or(|v| v < dv)
            {
                // The dynamic expression has been updated, re-run the pruning.
                dynamic_versions[idx] = Some(dv);
                mask =
                    prune_conjunct::<TRACK_COUNTERS>(&ctx, filter, idx, Some(dv), &row_range, mask)
                        .await?;
                if mask.all_false() {
                    return Ok(mask);
                }
            }

            let input_true_count = mask.true_count();
            let conjunct_mask = reader
                .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            filter.report_selectivity(
                idx,
                conditional_selectivity(input_true_count, conjunct_mask.true_count()),
            );

            // Filter evaluations return a mask already intersected with the input mask.
            mask = conjunct_mask;
            if mask.all_false() {
                ctx.count::<TRACK_COUNTERS>(ScanCounters::empty_split_after_predicate);
                return Ok(mask);
            }
        }

        Ok(mask)
    })
}

/// Applies whole-file statistics for a dynamic conjunct, then split statistics.
async fn prune_conjunct<const TRACK_COUNTERS: bool>(
    ctx: &TaskContext,
    filter: &FilterExpr,
    idx: usize,
    dynamic_version: Option<u64>,
    row_range: &Range<u64>,
    mask: Mask,
) -> VortexResult<Mask> {
    // Static conjuncts are checked against whole-file statistics once per execution.
    if dynamic_version.is_some()
        && filter.can_prune_file(idx, dynamic_version, ctx.reader.as_ref())?
    {
        ctx.count::<TRACK_COUNTERS>(ScanCounters::prune_file);
        return Ok(Mask::new_false(mask.len()));
    }

    let conjunct = &filter.conjuncts()[idx];
    let conjunct_mask = ctx
        .reader
        .split_pruning_evaluation(row_range, conjunct, mask.clone())?
        .await?;
    let mask = mask.bitand(&conjunct_mask);
    if TRACK_COUNTERS && mask.all_false() {
        if ctx.reader.split_pruning_is_statistics(conjunct)? {
            ctx.count::<TRACK_COUNTERS>(ScanCounters::prune_split);
        } else {
            ctx.count::<TRACK_COUNTERS>(ScanCounters::empty_split_after_predicate);
        }
    }
    Ok(mask)
}

fn conditional_selectivity(input_true_count: usize, output_true_count: usize) -> f64 {
    debug_assert!(input_true_count > 0);
    debug_assert!(output_true_count <= input_true_count);
    output_true_count as f64 / input_true_count as f64
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included
pub(crate) struct TaskContext {
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: LayoutReaderRef,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: BoundExpression,
    /// Counters shared by the tasks of this execution.
    pub(super) counters: Option<Arc<ScanCounters>>,
    /// Registry totals that receive one update after this execution stops.
    pub(super) metrics: Option<ScanMetrics>,
}

impl TaskContext {
    #[inline]
    pub(super) fn count<const TRACK_COUNTERS: bool>(&self, update: impl FnOnce(&ScanCounters)) {
        if TRACK_COUNTERS && let Some(counters) = &self.counters {
            update(counters);
        }
    }
}

impl Drop for TaskContext {
    fn drop(&mut self) {
        if let Some(metrics) = &self.metrics
            && let Some(counters) = &self.counters
        {
            metrics.record(counters.snapshot());
        }
    }
}

#[cfg(test)]
mod tests;
