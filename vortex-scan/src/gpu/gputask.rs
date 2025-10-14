// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::{BoxFuture, ok};
use vortex_array::{ArrayRef, MaskFuture};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_layout::LayoutReader;
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
pub(super) fn gpu_split_exec<A: 'static + Send>(
    ctx: Arc<GpuTaskContext<A>>,
    split: Range<u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Apply the selection to calculate a read mask
    let len =
        usize::try_from(split.end - split.start).vortex_expect("Range length must fit in usize");
    let read_mask = RowMask::new(split.start, Mask::new_true(len));
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter_mask = MaskFuture::ready(row_mask);

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
pub(super) struct GpuTaskContext<A> {
    /// The layout reader.
    pub(super) reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: ExprRef,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
