// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_gpu::GpuVector;
use vortex_layout::GpuLayoutReader;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

pub(super) fn gpu_split_exec<A: 'static + Send>(
    ctx: Arc<GpuTaskContext<A>>,
    split: Range<u64>,
) -> VortexResult<TaskFuture<Option<Vec<A>>>> {
    let projection_future = ctx.reader.projection_evaluation(&split, &ctx.projection)?;

    let mapper = ctx.mapper.clone();
    let array_fut = async move {
        let array = projection_future.await?;
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

/// Information needed to execute a single split task.
pub(super) struct GpuTaskContext<A> {
    /// The layout reader.
    pub(super) reader: Arc<dyn GpuLayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: Expression,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(Vec<GpuVector>) -> VortexResult<Vec<A>> + Send + Sync>,
}
