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

use crate::LayoutReader;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

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
    let row_mask = read_mask.mask().clone();

    let filter_mask = match ctx.filter.as_ref() {
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
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = Arc::clone(&ctx.reader);
            let filter = Arc::clone(filter);
            let row_range = row_range.clone();

            chained_filter_mask(reader, filter, row_range, row_mask)?
        }
    };

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

/// Builds the filter mask by chaining every conjunct's evaluation at task-construction time.
///
/// [`LayoutReader::filter_evaluation`] registers its segment reads when it is *called*, but only
/// awaits its input mask when it is *polled*. Building the evaluations one at a time — awaiting
/// each before constructing the next — therefore trickles reads in one conjunct at a time, per
/// split. Feeding each conjunct's output [`MaskFuture`] straight into the next instead registers
/// the reads for the whole chain up front, so the IO system can coalesce them, while each
/// conjunct still receives the mask its predecessor refined.
///
/// This matters most for filter columns that are not projected. The projection evaluation is
/// already built eagerly, so a filter over a projected column has its segments registered either
/// way; a filter over an unprojected column otherwise has nothing registering them ahead of time.
///
/// The evaluation order is taken from [`FilterExpr::next_conjunct`] up front rather than being
/// re-queried between conjuncts. That ordering is recomputed only when a *completed* conjunct
/// reports its selectivity, so within a single split it was already fixed; draining it here gives
/// up nothing but lets the chain be built before anything is awaited. Ordering still adapts
/// across splits.
fn chained_filter_mask(
    reader: Arc<dyn LayoutReader>,
    filter: Arc<FilterExpr>,
    row_range: Range<u64>,
    row_mask: Mask,
) -> VortexResult<MaskFuture> {
    let len = row_mask.len();
    let conjunct_count = filter.conjuncts().len();

    // Each pruning evaluation is fed the original split mask rather than the mask accumulated by
    // the preceding conjuncts. Pruning masks are folded together with `bitand`, and intersection
    // is associative and commutative, so the final mask is unchanged.
    let mut dynamic_versions = Vec::with_capacity(conjunct_count);
    let mut pruning_evals = Vec::with_capacity(conjunct_count);
    for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
        // Store the latest version of the dynamic expression prior to pruning. We re-run the
        // pruning if the version has changed by the time the task is polled.
        dynamic_versions.push(filter.dynamic_updates(idx).map(|du| du.version()));
        pruning_evals.push(reader.pruning_evaluation(&row_range, conjunct, row_mask.clone())?);
    }

    let pruned = MaskFuture::new(len, {
        let reader = Arc::clone(&reader);
        let filter = Arc::clone(&filter);
        let row_range = row_range.clone();
        async move {
            let mut mask = row_mask;

            for pruning_eval in pruning_evals {
                if mask.all_false() {
                    // Dropping the remaining evaluations cancels their outstanding reads.
                    return Ok(mask);
                }
                mask = mask.bitand(&pruning_eval.await?);
            }

            // Re-run the pruning for any conjunct whose dynamic expression has changed since.
            for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                if mask.all_false() {
                    return Ok(mask);
                }

                let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                if let Some(dv) = current_version
                    && dynamic_versions[idx].is_none_or(|v| v < dv)
                {
                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }
            }

            Ok(mask)
        }
    });

    let mut remaining = BitVec::from_elem(conjunct_count, true);
    let mut chain = Vec::with_capacity(conjunct_count);
    let mut mask_fut = pruned;
    while let Some(idx) = filter.next_conjunct(&remaining) {
        remaining.set(idx, false);
        mask_fut = reader.filter_evaluation(&row_range, &filter.conjuncts()[idx], mask_fut)?;
        chain.push((idx, mask_fut.clone()));
    }

    Ok(MaskFuture::new(len, async move {
        // Filter evaluations return a mask already intersected with the input mask, so the tail
        // of the chain is the fully refined mask.
        let mask = mask_fut.await?;

        // Every link has resolved by the time the tail has, so these awaits are already complete.
        for (idx, link) in chain {
            filter.report_selectivity(idx, link.await?.density());
        }

        Ok(mask)
    }))
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
