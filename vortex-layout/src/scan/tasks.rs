//! Split scanning task implementation.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::LayoutReader;
use crate::scan::{Selection, TaskExecutor, TaskExecutorExt};

/// Bails immediately if there is an empty mask
macro_rules! nonempty {
    ($mask:expr) => {
        if $mask.all_false() {
            return Ok(None);
        } else {
            $mask
        }
    };
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
pub(super) async fn split_exec<A: 'static + Send + Sync>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
) -> VortexResult<Option<A>> {
    // Step 1: using the caller-provided row range and selection, attempt to disregard this split.
    let read_range = match &ctx.row_range {
        None => split,
        Some(row_range) => {
            if row_range.start >= split.end || row_range.end < split.start {
                // No overlap for this task
                return Ok(None);
            }

            let intersect_start = row_range.start.max(split.start);
            let intersect_end = row_range.end.min(split.end);
            intersect_start..intersect_end
        }
    };

    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&read_range);
    let row_range = read_mask.row_range();
    let row_mask = nonempty!(read_mask.mask().clone());

    let filtered_mask = match ctx.filter.as_ref() {
        None => row_mask,
        Some(filter) => {
            // Step 2: if there is a filter provided, attempt to prune this range based on the filter.
            let prune = ctx.reader.pruning_evaluation(&row_range, filter)?;
            let pruned_mask = prune.invoke(row_mask).await?;

            // Step 3: apply exact filtering. The pruning step has already eliminated entire blocks
            // where we know the filter won't match any rows, so the amount of work to do here
            // should be a lot less.
            let eval = ctx.reader.filter_evaluation(&row_range, filter)?;
            eval.invoke(pruned_mask).await?
        }
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let filtered_mask = nonempty!(filtered_mask);
    let exec = ctx
        .reader
        .projection_evaluation(&row_range, &ctx.projection)?;
    let mapper = ctx.mapper.clone();
    let array_fut = async move {
        let array_ref = exec.invoke(filtered_mask).await?;
        mapper(array_ref).map(Some)
    };

    match &ctx.task_executor {
        None => array_fut.await,
        // If caller provided an executor for the CPU work, spawn onto that and await the result
        Some(executor) => executor.clone().spawn(array_fut.boxed()).await,
    }
}

/// Information needed to execute a single split task.
pub(super) struct TaskContext<A> {
    /// A caller-provided range of the file to read. All tasks should intersect their reads
    /// with this range to ensure that they are split as well.
    pub(super) row_range: Option<Range<u64>>,

    /// A row selection to apply.
    pub(super) selection: Selection,

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
