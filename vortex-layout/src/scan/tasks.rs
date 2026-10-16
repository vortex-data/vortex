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
        let mut dynamic_versions = vec![None; filter.conjuncts().len()];
        let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);

        // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
        for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
            if mask.all_false() {
                return Ok(mask);
            }

            if filter.is_unbounded(idx) {
                remaining.set(idx, false);
                filter.report_selectivity(idx, 1.0);
                continue;
            }

            // Store the latest version of the dynamic expression prior to pruning.
            // We will re-run the pruning later if the version has changed in the meantime.
            dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

            let conjunct_mask = reader
                .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
                .await?;
            mask = mask.bitand(&conjunct_mask);
        }

        // Now we loop through the conjuncts in the preferred order and evaluate them.
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
            // A dynamic conjunct that keeps almost every row is skipped until its values change.
            if let Some(version) = current_version
                && filter.skips_dynamic_conjunct(idx, version)
            {
                continue;
            }

            let input_true_count = mask.true_count();
            let conjunct_mask = reader
                .filter_evaluation(&filter_row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            let selectivity = conditional_selectivity(input_true_count, conjunct_mask.true_count());
            filter.report_selectivity(idx, selectivity);
            if let Some(version) = current_version {
                filter.report_dynamic_selectivity(idx, version, selectivity);
            }

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
    use std::sync::Arc;

    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::dynamic;
    use vortex_array::expr::is_null;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::build_filter_mask;
    use super::conditional_selectivity;
    use crate::LayoutReaderRef;
    use crate::scan::filter::FilterExpr;
    use crate::scan::test::TestLayoutReader;

    #[rstest]
    #[tokio::test]
    async fn unbounded_filter_skips_reads_until_updated(
        #[values(false, true)] keep_nulls: bool,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let bound = Arc::new(Mutex::new(None::<i32>));
        let value = Arc::clone(&bound);
        let expression = dynamic(
            CompareOperator::Gt,
            move || value.lock().map(Into::into),
            dtype.clone(),
            true,
            root(),
        );
        let expression = if keep_nulls {
            or(is_null(root()), expression)
        } else {
            expression
        };
        let filter = Arc::new(FilterExpr::new(expression.bind(&dtype)?));
        let reader: LayoutReaderRef = Arc::new(TestLayoutReader::new(8).with_fail_first_filter());
        let mask = Mask::from_iter([true, false, true, false, true, true, false, true]);

        let actual = build_filter_mask(&reader, &filter, &(0..8), mask.clone()).await?;
        assert_eq!(actual, mask);

        *bound.lock() = Some(3);
        assert!(
            build_filter_mask(&reader, &filter, &(0..8), mask)
                .await
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn selectivity_is_relative_to_the_input_mask() {
        assert_eq!(conditional_selectivity(20, 5), 0.25);
    }
}
