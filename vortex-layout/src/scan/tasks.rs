// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
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

use crate::LayoutReader;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// The result of metadata pruning for one filter-native row range.
pub(crate) struct PrunedMask {
    row_mask: RowMask,
    dynamic_versions: Vec<Option<u64>>,
}

impl PrunedMask {
    pub(crate) fn row_range(&self) -> std::ops::Range<u64> {
        self.row_mask.row_range()
    }

    pub(crate) fn mask(&self) -> &Mask {
        self.row_mask.mask()
    }
}

/// Logic for executing a single split reading task.
/// N.B. read_mask should be evaluated against all_false() before calling this
/// method to avoid creating an empty TaskFuture.
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
pub fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let row_range = read_mask.row_range();
    let filter_mask = filter_mask(Arc::clone(&ctx), &read_mask, limit)?;
    project_exec_with_mask(ctx, row_range, filter_mask)
}

/// Resolve metadata pruning without evaluating the exact filter.
pub(crate) fn prune_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
) -> VortexResult<TaskFuture<PrunedMask>> {
    let row_range = read_mask.row_range();
    let row_offset = row_range.start;
    let row_mask = read_mask.mask().clone();

    let Some(filter) = ctx.filter.as_ref().cloned() else {
        return Ok(async move {
            Ok(PrunedMask {
                row_mask: RowMask::new(row_offset, row_mask),
                dynamic_versions: Vec::new(),
            })
        }
        .boxed());
    };

    let reader = Arc::clone(&ctx.reader);
    Ok(async move {
        let mut mask = row_mask;
        let mut dynamic_versions = vec![None; filter.conjuncts().len()];

        for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
            if mask.all_false() {
                break;
            }

            dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());
            let conjunct_mask = reader
                .pruning_evaluation(&row_range, conjunct, mask.clone())?
                .await?;
            mask = mask.bitand(&conjunct_mask);
        }

        Ok(PrunedMask {
            row_mask: RowMask::new(row_offset, mask),
            dynamic_versions,
        })
    }
    .boxed())
}

/// Evaluate the exact filter starting from a resolved pruning mask.
pub(crate) fn filter_after_pruning<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    pruned: PrunedMask,
) -> VortexResult<MaskFuture> {
    let row_range = pruned.row_mask.row_range();
    let row_mask = pruned.row_mask.mask().clone();

    let Some(filter) = ctx.filter.as_ref().cloned() else {
        return Ok(MaskFuture::ready(row_mask));
    };

    let reader = Arc::clone(&ctx.reader);
    Ok(MaskFuture::new(row_mask.len(), async move {
        let mut mask = row_mask;
        let mut dynamic_versions = pruned.dynamic_versions;

        // Evaluate conjuncts in their learned order, preserving the dynamic-pruning refresh from
        // the coupled scan path.
        let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
        while let Some(idx) = filter.next_conjunct(&remaining) {
            remaining.set(idx, false);
            if mask.all_false() {
                return Ok(mask);
            }

            let conjunct = &filter.conjuncts()[idx];
            let current_version = filter.dynamic_updates(idx).map(|du| du.version());
            if let Some(dv) = current_version
                && dynamic_versions[idx].is_none_or(|v| v < dv)
            {
                dynamic_versions[idx] = Some(dv);
                let conjunct_mask = reader
                    .pruning_evaluation(&row_range, conjunct, mask.clone())?
                    .await?;
                mask = mask.bitand(&conjunct_mask);
            }
            if mask.all_false() {
                return Ok(mask);
            }

            let input_true_count = mask.true_count();
            let conjunct_mask = reader
                .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            filter.report_selectivity(
                idx,
                conditional_selectivity(input_true_count, conjunct_mask.true_count()),
            );
            mask = conjunct_mask;
        }

        Ok(mask)
    }))
}

fn filter_mask<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: &RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<MaskFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    Ok(match ctx.filter.as_ref() {
        // No filter == immediate mask
        None => {
            let row_mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(row_mask.len()),
                Some(l) => {
                    let true_count = row_mask.true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    let row_mask = row_mask.limit(mask_limit);
                    *l -= mask_limit as u64;
                    row_mask
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            // Keep the coupled path's original scheduling shape. The pruning-first barrier is
            // intentionally limited to the experimental separate-splits stream.
            let reader = Arc::clone(&ctx.reader);
            let filter = Arc::clone(filter);
            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());
                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct = &filter.conjuncts()[idx];
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        dynamic_versions[idx] = Some(dv);
                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let input_true_count = mask.true_count();
                    let conjunct_mask = reader
                        .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(
                        idx,
                        conditional_selectivity(input_true_count, conjunct_mask.true_count()),
                    );
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    })
}

pub(crate) fn project_exec_with_mask<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    row_range: std::ops::Range<u64>,
    filter_mask: MaskFuture,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Step 4: execute the projection, only at the mask for rows which match the filter
    let projection_future =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
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

fn conditional_selectivity(input_true_count: usize, output_true_count: usize) -> f64 {
    debug_assert!(input_true_count > 0);
    debug_assert!(output_true_count <= input_true_count);
    output_true_count as f64 / input_true_count as f64
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included
pub struct TaskContext<A> {
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: BoundExpression,
    /// Function that maps into an A.
    pub mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

#[cfg(test)]
mod tests {
    use super::conditional_selectivity;

    #[test]
    fn selectivity_is_relative_to_the_input_mask() {
        assert_eq!(conditional_selectivity(20, 5), 0.25);
    }
}
