// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::future::Future;
use std::ops::BitAnd;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::ArrayFuture;
use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::limit::RowLimit;

/// The result of a split task.
///
/// Filter errors happen before a row limit reserves any rows, so callers may report them and
/// continue with later splits. Projection errors after reservation cannot safely release rows back
/// to a concurrent limit, so callers must report them and terminate the limited scan.
pub(crate) enum TaskResult {
    /// A completed projection, or an empty split.
    Array(Option<ArrayRef>),
    /// An error that occurred before a row limit reserved rows.
    Recoverable(VortexError),
    /// An error that occurred after a row limit reserved rows.
    Terminal(VortexError),
}

/// A future that executes one split and classifies any failure by whether it happened before or
/// after a row-limit reservation.
#[must_use = "split tasks must be scheduled or awaited"]
pub(crate) struct TaskFuture {
    inner: BoxFuture<'static, TaskResult>,
}

impl TaskFuture {
    pub(crate) fn new(future: impl Future<Output = TaskResult> + Send + 'static) -> Self {
        Self {
            inner: future.boxed(),
        }
    }

    fn ready(result: TaskResult) -> Self {
        Self::new(futures::future::ready(result))
    }

    pub(crate) fn empty() -> Self {
        Self::ready(TaskResult::Array(None))
    }

    pub(crate) fn recoverable(error: VortexError) -> Self {
        Self::ready(TaskResult::Recoverable(error))
    }

    pub(crate) fn terminal(error: VortexError) -> Self {
        Self::ready(TaskResult::Terminal(error))
    }

    /// Project a split whose mask needed no filtering.
    ///
    /// `terminal` marks that the mask already reserved rows against a limit, so a failure cannot
    /// be reported without ending the scan.
    pub(crate) fn projection(projection: ArrayFuture, terminal: bool) -> Self {
        Self::new(async move {
            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) if terminal => TaskResult::Terminal(error),
                Err(error) => TaskResult::Recoverable(error),
            }
        })
    }

    /// Project the rows a filter matched, skipping projection entirely for an empty split.
    ///
    /// The projection is constructed by the caller, before the filter has run, so that the reader
    /// can prefetch its I/O.
    pub(crate) fn filtered_projection(filter_mask: MaskFuture, projection: ArrayFuture) -> Self {
        Self::new(async move {
            let mask = match filter_mask.await {
                Ok(mask) => mask,
                Err(error) => return TaskResult::Recoverable(error),
            };
            if mask.all_false() {
                return TaskResult::Array(None);
            }

            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) => TaskResult::Recoverable(error),
            }
        })
    }

    /// Filter, reserve the matching rows against `row_limit`, then project only what was granted.
    ///
    /// Projection work is constructed after reservation, so rows the limit cannot grant are never
    /// decoded. Once rows have been reserved they cannot be released back to a concurrent limit,
    /// so any projection failure is terminal.
    fn limited_filtered_projection(
        ctx: Arc<TaskContext>,
        row_range: Range<u64>,
        filter_mask: MaskFuture,
        row_limit: RowLimit,
    ) -> Self {
        Self::new(async move {
            let mask = match filter_mask.await {
                Ok(mask) => mask,
                Err(error) => return TaskResult::Recoverable(error),
            };
            // A filter error above returns before reserving any rows.
            let mask = row_limit.limit(mask);
            if mask.all_false() {
                return TaskResult::Array(None);
            }

            let projection = match ctx.reader.projection_evaluation(
                &row_range,
                &ctx.projection,
                MaskFuture::ready(mask),
            ) {
                Ok(projection) => projection,
                Err(error) => return TaskResult::Terminal(error),
            };
            match projection.await {
                Ok(array) => TaskResult::Array(Some(array)),
                Err(error) => TaskResult::Terminal(error),
            }
        })
    }
}

impl Future for TaskFuture {
    type Output = TaskResult;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
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
/// The final mask is limited before it is given to the reader to perform a filtered projection
/// over the split data, yielding the projected array (or `None` when the split selects no rows).
/// Limiting before projection prevents decode work for rows that the scan cannot return.
pub(crate) fn split_exec(
    ctx: Arc<TaskContext>,
    read_mask: RowMask,
    row_limit: Option<RowLimit>,
) -> VortexResult<TaskFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    let Some(filter) = ctx.filter.as_ref() else {
        let limited = row_limit.is_some();
        let row_mask = if let Some(limit) = row_limit {
            limit.limit(row_mask)
        } else {
            row_mask
        };
        if row_mask.all_false() {
            return Ok(TaskFuture::empty());
        }

        // With no filter, limit the selection before constructing projection work.
        let projection = match ctx.reader.projection_evaluation(
            &row_range,
            &ctx.projection,
            MaskFuture::ready(row_mask),
        ) {
            Ok(projection) => projection,
            Err(err) if limited => return Ok(TaskFuture::terminal(err)),
            Err(err) => return Err(err),
        };
        return Ok(TaskFuture::projection(projection, limited));
    };

    let filter_mask = build_filter_mask(&ctx.reader, filter, &row_range, row_mask);

    let Some(row_limit) = row_limit else {
        // Without a limit, retain the existing eager projection setup so readers can prefetch
        // projection work while the filter is being evaluated.
        let projection =
            ctx.reader
                .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;
        return Ok(TaskFuture::filtered_projection(filter_mask, projection));
    };

    Ok(TaskFuture::limited_filtered_projection(
        ctx,
        row_range,
        filter_mask,
        row_limit,
    ))
}

/// Build the filtered mask for a split.
///
/// The pruning and filter evaluations are constructed OUTSIDE the returned future on purpose:
/// registering these row ranges eagerly is a hint to the IO system that we want to start
/// prefetching the IO for this split.
fn build_filter_mask(
    reader: &LayoutReaderRef,
    filter: &Arc<FilterExpr>,
    row_range: &Range<u64>,
    row_mask: Mask,
) -> MaskFuture {
    let reader = Arc::clone(reader);
    let filter = Arc::clone(filter);
    let filter_row_range = row_range.clone();
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
                .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
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
            // Store the dynamic update once to avoid TOCTOU race condition.
            let current_version = filter.dynamic_updates(idx).map(|du| du.version());
            if let Some(dv) = current_version
                && dynamic_versions[idx].is_none_or(|v| v < dv)
            {
                // The dynamic expression has changed, re-run the pruning.
                dynamic_versions[idx] = Some(dv);
                let conjunct_mask = reader
                    .pruning_evaluation(&filter_row_range, conjunct, mask.clone())?
                    .await?;
                mask = mask.bitand(&conjunct_mask);
            }
            if mask.all_false() {
                return Ok(mask);
            }

            let conjunct_mask = reader
                .filter_evaluation(&filter_row_range, conjunct, MaskFuture::ready(mask))?
                .await?;
            filter.report_selectivity(idx, conjunct_mask.density());

            // Filter evaluations return a mask already intersected with the input mask.
            mask = conjunct_mask;
        }

        Ok(mask)
    })
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included.
pub(crate) struct TaskContext {
    /// The shared filter expression.
    pub(crate) filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub(crate) reader: LayoutReaderRef,
    /// The projection expression to apply to gather the scanned rows.
    pub(crate) projection: BoundExpression,
}
