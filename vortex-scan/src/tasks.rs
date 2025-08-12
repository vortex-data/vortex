// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::{BitAnd, Range};
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::{BoxFuture, ok};
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_error::{VortexError, VortexResult};
use vortex_expr::ExprRef;
use vortex_layout::LayoutReader;
use vortex_mask::Mask;

use crate::Selection;
use crate::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

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

    let filter = match ctx.filter.as_ref() {
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

            ok(row_mask).boxed()
        }
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.

            // Create one pruning task per conjunct.
            let pruning_conjuncts: Vec<_> = filter
                .conjuncts()
                .iter()
                .map(|expr| ctx.reader.pruning_evaluation(&row_range, expr))
                .try_collect()?;

            // And one projection task per conjunct.
            let conjuncts: Vec<_> = filter
                .conjuncts()
                .iter()
                .map(|expr| ctx.reader.filter_evaluation(&row_range, expr))
                .try_collect()?;

            let filter = filter.clone();

            async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                // Debug assertion to ensure the pruning_conjuncts and conjuncts have the same length
                // as filter.conjuncts() to prevent index out of bounds
                assert_eq!(
                    pruning_conjuncts.len(),
                    filter.conjuncts().len(),
                    "pruning_conjuncts length ({}) != filter.conjuncts().len() ({})",
                    pruning_conjuncts.len(),
                    filter.conjuncts().len()
                );
                assert_eq!(
                    conjuncts.len(),
                    filter.conjuncts().len(),
                    "conjuncts length ({}) != filter.conjuncts().len() ({})",
                    conjuncts.len(),
                    filter.conjuncts().len()
                );

                // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
                for (idx, conjunct) in pruning_conjuncts.iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    // Store the latest version of the dynamic expression prior to pruning.
                    // We will re-run the pruning later if the version has changed in the meantime.
                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                    let conjunct_mask = conjunct.invoke(mask.clone()).await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                // Now we loop through the conjuncts in the preferred order and evaluate them.
                let mut remaining = BitVec::from_elem(conjuncts.len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);

                    if mask.all_false() {
                        return Ok(mask);
                    }

                    // If the dynamic expression has changed since pruning, re-run the pruning.
                    // Store the dynamic update once to avoid TOCTOU race condition
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        // The dynamic expression has been updated, re-run the pruning.
                        dynamic_versions[idx] = Some(dv);
                        let conjunct_mask = pruning_conjuncts[idx].invoke(mask.clone()).await?;
                        mask = mask.bitand(&conjunct_mask);
                    }

                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct_mask = conjuncts[idx].invoke(mask.clone()).await?;

                    // TODO(ngates): what selectivity should we report?
                    let selectivity = conjunct_mask.true_count() as f64 / mask.len() as f64;
                    //let selectivity = conjunct_mask.true_count() as f64 / mask.true_count() as f64;
                    filter.report_selectivity(idx, selectivity);

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                Ok::<_, VortexError>(mask)
            }
            .boxed()
        }
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let exec = ctx
        .reader
        .projection_evaluation(&row_range, &ctx.projection)?;
    let mapper = ctx.mapper.clone();
    let array_fut = async move {
        let filtered_mask = filter.await?;
        if filtered_mask.all_false() {
            return Ok(None);
        }
        let array_ref = exec.invoke(filtered_mask).await?;
        mapper(array_ref).map(Some)
    };

    Ok(array_fut.boxed())
}

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
