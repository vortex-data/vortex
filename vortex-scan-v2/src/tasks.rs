// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

pub(crate) type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

pub(crate) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    tracing::trace!(
        target: "vortex_scan_v2::execution",
        ?row_range,
        selected_rows = row_mask.true_count(),
        has_filter = ctx.filter.is_some(),
        "executing a plan scan split"
    );

    let filter_mask = match &ctx.filter {
        None => {
            let row_mask = match limit {
                Some(limit) if *limit == 0 => Mask::new_false(row_mask.len()),
                Some(limit) => {
                    let true_count = row_mask.true_count();
                    let mask_limit = usize::try_from(*limit)
                        .map(|limit| limit.min(true_count))
                        .unwrap_or(true_count);
                    let row_mask = row_mask.limit(mask_limit);
                    *limit -= mask_limit as u64;
                    row_mask
                }
                None => row_mask,
            };
            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            let predicate = filter.execute(
                &ctx.execution,
                &row_range,
                MaskFuture::ready(row_mask.clone()),
            )?;
            let session = ctx.execution.session().clone();
            MaskFuture::new(row_mask.len(), async move {
                let predicate = predicate.await?;
                let mut execution = session.create_execution_ctx();
                let predicate = predicate.null_as_false().execute(&mut execution)?;
                Ok(row_mask.intersect_by_rank(&predicate))
            })
        }
    };

    let projection = ctx
        .projection
        .execute(&ctx.execution, &row_range, filter_mask.clone())?;
    let mapper = Arc::clone(&ctx.mapper);
    Ok(async move {
        if filter_mask.await?.all_false() {
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                "plan scan split produced no matching rows"
            );
            return Ok(None);
        }
        let array = projection.await?;
        tracing::trace!(
            target: "vortex_scan_v2::execution",
            ?row_range,
            output_rows = array.len(),
            dtype = %array.dtype(),
            "completed a plan scan split"
        );
        mapper(array).map(Some)
    }
    .boxed())
}

pub(crate) struct TaskContext<A> {
    pub(crate) execution: PlanExecutionContext,
    pub(crate) filter: Option<PlanRef>,
    pub(crate) projection: PlanRef,
    pub(crate) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
