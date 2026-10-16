// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;

/// A future resolving to the projected rows of one split, or `None` when it selects no rows.
pub(crate) type SplitFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// Logic for executing a single split reading task.
/// N.B. read_mask should be evaluated against all_false() before calling this
/// method to avoid creating an empty task.
///
/// # Task execution flow
///
/// First, the task's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// The intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// This mask is then provided to the reader to perform a filtered projection over the split data,
/// yielding the projected array (or `None` when the split selects no rows).
pub(crate) fn split_exec(ctx: &TaskContext, read_mask: RowMask) -> VortexResult<SplitFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    let Some(filter) = ctx.filter.as_ref() else {
        return split_projection(ctx, &row_range, row_mask);
    };
    let filter_mask = build_filter_mask(&ctx.reader, filter, &row_range, row_mask);

    // Construct the projection before the filter has run so the reader can prefetch its I/O.
    let projection =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;
    Ok(async move {
        if filter_mask.await?.all_false() {
            return Ok(None);
        }
        projection.await.map(Some)
    }
    .boxed())
}

/// Project the rows selected by an already-evaluated `mask`.
pub(crate) fn split_projection(
    ctx: &TaskContext,
    row_range: &Range<u64>,
    mask: Mask,
) -> VortexResult<SplitFuture> {
    if mask.all_false() {
        return Ok(futures::future::ready(Ok(None)).boxed());
    }
    let projection =
        ctx.reader
            .projection_evaluation(row_range, &ctx.projection, MaskFuture::ready(mask))?;
    Ok(projection.map(|array| array.map(Some)).boxed())
}

/// Build the filtered mask for a split.
///
/// The pruning and filter evaluations are constructed OUTSIDE the returned future on purpose:
/// registering these row ranges eagerly is a hint to the IO system that we want to start
/// prefetching the IO for this split.
fn build_filter_mask(
    reader: &LayoutReaderRef,
    filter: &Arc<FilterExpr>,
    row_range: &Range<u64>,
    row_mask: Mask,
) -> MaskFuture {
    let reader = Arc::clone(reader);
    let filter = Arc::clone(filter);
    let filter_row_range = row_range.clone();
    MaskFuture::new(row_mask.len(), async move {
        let mut mask = row_mask;
        if mask.all_false() {
            return Ok(mask);
        }

        // Store the latest version of each dynamic expression prior to pruning.
        // We will re-run the pruning later if the version has changed in the meantime.
        let mut dynamic_versions: Vec<_> = (0..filter.conjuncts().len())
            .map(|idx| filter.dynamic_updates(idx).map(|du| du.version()))
            .collect();

        // Prune with every conjunct concurrently, intersecting the masks as they resolve. Each
        // pruning sees only the input mask, and the remaining evaluations are dropped once the
        // intersection rules out every row.
        let mut pruning: FuturesUnordered<_> = filter
            .conjuncts()
            .iter()
            .map(|conjunct| reader.pruning_evaluation(&filter_row_range, conjunct, mask.clone()))
            .collect::<VortexResult<_>>()?;
        while let Some(conjunct_mask) = pruning.try_next().await? {
            mask = mask.bitand(&conjunct_mask);
            if mask.all_false() {
                return Ok(mask);
            }
        }

        // Now we loop through the conjuncts in the preferred order and evaluate them.
        let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
        while let Some(idx) = filter.next_conjunct(&remaining) {
            remaining.set(idx, false);
            if mask.all_false() {
                return Ok(mask);
            }

            let conjunct = &filter.conjuncts()[idx];

            // If the dynamic expression has changed since pruning, re-run the pruning.
            // Store the dynamic update once to avoid TOCTOU race condition.
            let current_version = filter.dynamic_updates(idx).map(|du| du.version());
            if let Some(dv) = current_version
                && dynamic_versions[idx].is_none_or(|v| v < dv)
            {
                // The dynamic expression has changed, re-run the pruning.
                dynamic_versions[idx] = Some(dv);
                let conjunct_mask = reader
                    .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
                    .await?;
                mask = mask.bitand(&conjunct_mask);
            }
            if mask.all_false() {
                return Ok(mask);
            }

            let input_true_count = mask.true_count();
            let conjunct_mask = reader
                .filter_evaluation(&filter_row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            filter.report_selectivity(
                idx,
                conditional_selectivity(input_true_count, conjunct_mask.true_count()),
            );

            // Filter evaluations return a mask already intersected with the input mask.
            mask = conjunct_mask;
        }

        Ok(mask)
    })
}

fn conditional_selectivity(input_true_count: usize, output_true_count: usize) -> f64 {
    debug_assert!(input_true_count > 0);
    debug_assert!(output_true_count <= input_true_count);
    output_true_count as f64 / input_true_count as f64
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included.
pub(crate) struct TaskContext {
    /// The shared filter expression.
    pub(crate) filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub(crate) reader: LayoutReaderRef,
    /// The projection expression to apply to gather the scanned rows.
    pub(crate) projection: BoundExpression,
}

#[cfg(test)]
mod tests {
    use super::conditional_selectivity;

    #[test]
    fn selectivity_is_relative_to_the_input_mask() {
        assert_eq!(conditional_selectivity(20, 5), 0.25);
    }
}
