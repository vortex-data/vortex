// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::{BoxFuture, ok};
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_layout::{LayoutReader, TaskExecutor, TaskExecutorExt};
use vortex_mask::Mask;

use crate::row_mask::RowMask;

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
pub(super) fn split_exec<A: 'static + Send + Sync>(
    ctx: Arc<TaskContext<A>>,
    split_mask: RowMask,
    limit: Option<&mut usize>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // println!("split_exec {:?}", split_mask);
    // // Step 1: using the caller-provided row range and selection, attempt to disregard this split.
    let read_range = split_mask.row_range();
    let read_mask = split_mask.into_mask();

    // Early exit if the read mask is empty.
    if read_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter = match ctx.filter.as_ref() {
        // No filter == immediate task
        None => {
            let mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(read_mask.len()),
                Some(l) => {
                    let true_count = read_mask.true_count();
                    let row_mask = read_mask.limit(*l);
                    *l -= usize::min(*l, true_count);
                    row_mask
                }
                None => read_mask,
            };

            ok(mask).boxed()
        }
        Some(filter) => {
            // Step 2: if there is a filter provided, attempt to prune this range based on the filter.
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // of the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let prune = ctx.reader.pruning_evaluation(&read_range, filter)?;
            let eval = ctx.reader.filter_evaluation(&read_range, filter)?;

            async move {
                let pruned_mask = prune.invoke(read_mask).await?;

                // Step 3: apply exact filtering. The pruning step has already eliminated entire blocks
                // where we know the filter won't match any rows, so the amount of work to do here
                // should be a lot less.
                eval.invoke(pruned_mask).await
            }
            .boxed()
        }
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let exec = ctx
        .reader
        .projection_evaluation(&read_range, &ctx.projection)?;
    let mapper = ctx.mapper.clone();
    let array_fut = async move {
        let filtered_mask = filter.await?;
        let array_ref = exec.invoke(filtered_mask).await?;
        mapper(array_ref).map(Some)
    };

    match &ctx.task_executor {
        None => Ok(array_fut.boxed()),
        // If caller provided an executor for the CPU work, spawn onto that and await the result
        Some(executor) => Ok(executor.clone().spawn(array_fut.boxed())),
    }
}

/// Information needed to execute a single split task.
pub(super) struct TaskContext<A> {
    /// The filter expression for the current task.
    pub(super) filter: Option<ExprRef>,

    /// The layout reader.
    pub(super) reader: Arc<dyn LayoutReader>,

    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: ExprRef,

    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,

    pub(super) task_executor: Option<Arc<dyn TaskExecutor>>,
}
