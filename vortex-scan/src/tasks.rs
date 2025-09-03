// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::{BitAnd, Range};
use std::sync::Arc;

use futures::future::{ok, BoxFuture};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::pipeline::operators::MaskFuture;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_layout::{LayoutReader, LayoutReaderRef};
use vortex_mask::Mask;

use crate::filter::FilterExpr;
use crate::Selection;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Information needed to execute a single split task.
pub(super) struct TaskContext<A> {
    /// A caller-provided range of the file to read. All tasks should intersect their reads
    /// with this range to ensure that they are split as well.
    pub(super) row_range: Option<Range<u64>>,
    /// A row selection to apply.
    pub(super) selection: Selection,
    /// The shared filter expression.
    pub(super) filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub(super) reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: ExprRef,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

/// Logic for executing a single split reading task.
///
/// # Task execution flow
///
/// First, the tasks's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// Then intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// This mask is then provided to the reader to perform a filtered projection over the split data,
/// finally mapping the Vortex columnar record batches into some result type `A`.
pub(super) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<&mut usize>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Step 1: using the caller-provided row range and selection, attempt to disregard this split.
    let read_range = match &ctx.row_range {
        None => split,
        Some(row_range) => {
            if row_range.start >= split.end || row_range.end < split.start {
                // No overlap for this task
                return Ok(ok(None).boxed());
            }

            let intersect_start = row_range.start.max(split.start);
            let intersect_end = row_range.end.min(split.end);
            intersect_start..intersect_end
        }
    };

    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&read_range);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
        // No filter == immediate task
        None => {
            let row_mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(row_mask.len()),
                Some(l) => {
                    let true_count = row_mask.true_count();
                    let row_mask = row_mask.limit(*l);
                    *l -= usize::min(*l, true_count);
                    row_mask
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => filter_mask(
            row_mask,
            row_range.clone(),
            filter.clone(),
            ctx.reader.clone(),
        ),
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let projection_future =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;

    let mapper = ctx.mapper.clone();
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        let array = projection_future.await?;
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

fn filter_mask(
    mask: Mask,
    row_range: Range<u64>,
    filter: Arc<FilterExpr>,
    reader: LayoutReaderRef,
) -> MaskFuture {
    MaskFuture::new(mask.len(), async move {
        // We collect together the dynamic version for each conjunct prior to evaluation.
        // This allows us to detect if a dynamic expression has changed prior to actual evaluation
        // and to re-run the pruning step if needed.
        let mut dynamic_versions = Vec::with_capacity(filter.conjuncts().len());
        for idx in 0..filter.conjuncts().len() {
            dynamic_versions.push(Arc::new(Mutex::new(
                filter.dynamic_updates(idx).map(|du| du.version()),
            )))
        }

        // Now we create a MaskFuture to perform pruning concurrently.
        let mask = MaskFuture::intersect(
            MaskFuture::ready(mask.clone()),
            filter
                .conjuncts()
                .iter()
                .map(|conjunct| {
                    reader.pruning_evaluation(&row_range, conjunct, Mask::new_true(mask.len()))
                })
                .try_collect()?,
        );

        // Now we loop through the conjuncts in the preferred order and evaluate them.
        let idxs = filter.order().into_iter();

        // For each conjunct, we create a function that takes an input MaskFuture, and returns a
        // MaskFuture that performs the following steps:
        // 1. If the current mask is all false, return early.
        // 2. If the dynamic expression has changed since pruning, re-run the pruning step
        //    and update the mask.
        // 3. Run the full filter evaluation and update the mask.
        // 4. Report the selectivity of the conjunct.
        let fns = idxs.map(|idx| {
            // For each filter, we re-run the pruning step _just before_ the mask
            // is awaited.
            let conjunct = filter.conjuncts()[idx].clone();
            let dynamic_version = dynamic_versions[idx].clone();
            let filter = filter.clone();
            let reader = reader.clone();
            let row_range = row_range.clone();

            move |mask: MaskFuture| {
                MaskFuture::new(
                    mask.len(),
                    mask.and_then(move |mut mask| async move {
                        if mask.all_false() {
                            // No need to re-run pruning if the mask is already all false.
                            return Ok(mask);
                        }

                        // If the dynamic expression has changed since pruning, re-run the pruning.
                        // Store the dynamic update once to avoid TOCTOU race condition
                        let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                        let reprune = {
                            let mut dv = dynamic_version.lock();
                            match current_version {
                                None => dv.is_some(),
                                Some(current_version) => {
                                    let reprune = dv.is_none_or(|v| v < current_version);
                                    *dv = Some(current_version);
                                    reprune
                                }
                            }
                        };

                        if reprune {
                            mask = mask.bitand(
                                &reader
                                    .pruning_evaluation(&row_range, &conjunct, mask.clone())?
                                    .await?,
                            );
                        }
                        if mask.all_false() {
                            return Ok(mask);
                        }

                        // Otherwise, run the full filter evaluation.
                        let mask = reader
                            .filter_evaluation(
                                &row_range,
                                &conjunct,
                                MaskFuture::ready(mask.clone()),
                            )?
                            .await?;
                        filter.report_selectivity(idx, mask.density());

                        Ok(mask)
                    }),
                )
            }
        });

        // Finally, we create a future that runs each conjunct future concurrently but passes the
        // output mask of each conjunct to the next one in order.
        MaskFuture::fold_intersect(mask, fns).await
    })
}
