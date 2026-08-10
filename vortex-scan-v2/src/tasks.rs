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
        has_pruning = ctx.pruning.is_some(),
        has_filter = ctx.filter.is_some(),
        "executing a plan scan split"
    );

    let row_mask = match (&ctx.filter, limit) {
        (None, Some(limit)) if *limit == 0 => Mask::new_false(row_mask.len()),
        (None, Some(limit)) => {
            let true_count = row_mask.true_count();
            let mask_limit = usize::try_from(*limit)
                .map(|limit| limit.min(true_count))
                .unwrap_or(true_count);
            let row_mask = row_mask.limit(mask_limit);
            *limit -= mask_limit as u64;
            row_mask
        }
        _ => row_mask,
    };

    Ok(async move {
        let mut row_mask = row_mask;
        if let Some(pruning) = &ctx.pruning {
            let proof = pruning.execute(
                &ctx.execution,
                &row_range,
                MaskFuture::ready(row_mask.clone()),
            )?;
            let proof = proof.await?;
            let mut execution = ctx.execution.session().create_execution_ctx();
            let pruned: Mask = proof.null_as_false().execute(&mut execution)?;
            let pruned_rows = pruned.true_count();
            row_mask = row_mask.intersect_by_rank(&!pruned);
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                pruned_rows,
                remaining_rows = row_mask.true_count(),
                "applied the plan pruning proof"
            );
        }

        if row_mask.all_false() {
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                "plan pruning skipped the scan split"
            );
            return Ok(None);
        }

        let filter_mask = if let Some(filter) = &ctx.filter {
            let predicate = filter.execute(
                &ctx.execution,
                &row_range,
                MaskFuture::ready(row_mask.clone()),
            )?;
            let session = ctx.execution.session().clone();
            MaskFuture::new(row_mask.len(), async move {
                let predicate = predicate.await?;
                let mut execution = session.create_execution_ctx();
                let predicate: Mask = predicate.null_as_false().execute(&mut execution)?;
                Ok(row_mask.intersect_by_rank(&predicate))
            })
        } else {
            MaskFuture::ready(row_mask)
        };

        // Register projection reads before resolving the filter mask so segments used by both
        // expressions can share the same in-flight request.
        let projection = ctx
            .projection
            .execute(&ctx.execution, &row_range, filter_mask.clone())?;
        let row_mask = filter_mask.await?;

        if row_mask.all_false() {
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
        (ctx.mapper)(array).map(Some)
    }
    .boxed())
}

pub(crate) struct TaskContext<A> {
    pub(crate) execution: PlanExecutionContext,
    pub(crate) pruning: Option<PlanRef>,
    pub(crate) filter: Option<PlanRef>,
    pub(crate) projection: PlanRef,
    pub(crate) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
