// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::{BitAnd, Range};
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::{BoxFuture, ok};
use vortex_array::{ArrayRef, MaskFuture};
use vortex_error::VortexResult;
use vortex_expr::Expression;
use vortex_layout::LayoutReader;
use vortex_mask::Mask;

use crate::filter::FilterExpr;
use crate::selection::Selection;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Logic for executing a single split reading task.
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
/// finally mapping the Vortex columnar record batches into some result type `A`.
pub(super) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<&mut usize>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
        // No filter == immediate mask
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
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = ctx.reader.clone();
            let filter = filter.clone();
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
                for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    // Store the latest version of the dynamic expression prior to pruning.
                    // We will re-run the pruning later if the version has changed in the meantime.
                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
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
                    // Store the dynamic update once to avoid TOCTOU race condition
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        // The dynamic expression has been updated, re-run the pruning.
                        dynamic_versions[idx] = Some(dv);
                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct_mask = reader
                        .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(idx, conjunct_mask.density());

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
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

/// Information needed to execute a single split task.
pub(super) struct TaskContext<A> {
    /// A row selection to apply.
    pub(super) selection: Selection,
    /// The shared filter expression.
    pub(super) filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub(super) reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: Expression,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
