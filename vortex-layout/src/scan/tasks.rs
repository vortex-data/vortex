// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::ok;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;

use crate::LayoutReader;
use crate::mask_debug::log_mask_batch;
use crate::mask_debug::mask_coordinate_summary;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Logic for executing a single split reading task.
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
    split: Range<u64>,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

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
            let debug_label = ctx.debug_label.clone();

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

                    let input_mask = mask.clone();
                    let input_rows = input_mask.true_count();
                    let start = Instant::now();
                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, input_mask.clone())?
                        .await?;
                    let output_mask = input_mask.bitand(&conjunct_mask);
                    log_conjunct_eval(
                        "v1 pruning conjunct evaluated",
                        idx,
                        conjunct,
                        &row_range,
                        input_rows,
                        output_mask.true_count(),
                        start.elapsed(),
                        &input_mask,
                        &output_mask,
                        debug_label.as_deref(),
                    );
                    mask = output_mask;
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
                    // Store the dynamic update once to avoid TOCTOU race condition
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        // The dynamic expression has been updated, re-run the pruning.
                        dynamic_versions[idx] = Some(dv);
                        let input_mask = mask.clone();
                        let input_rows = input_mask.true_count();
                        let start = Instant::now();
                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, input_mask.clone())?
                            .await?;
                        let output_mask = input_mask.bitand(&conjunct_mask);
                        log_conjunct_eval(
                            "v1 pruning conjunct evaluated",
                            idx,
                            conjunct,
                            &row_range,
                            input_rows,
                            output_mask.true_count(),
                            start.elapsed(),
                            &input_mask,
                            &output_mask,
                            debug_label.as_deref(),
                        );
                        mask = output_mask;
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let input_mask = mask;
                    let input_rows = input_mask.true_count();
                    let start = Instant::now();
                    let conjunct_mask = reader
                        .filter_evaluation(
                            &row_range,
                            conjunct,
                            MaskFuture::ready(input_mask.clone()),
                        )?
                        .await?;
                    log_conjunct_eval(
                        "v1 filter conjunct evaluated",
                        idx,
                        conjunct,
                        &row_range,
                        input_rows,
                        conjunct_mask.true_count(),
                        start.elapsed(),
                        &input_mask,
                        &conjunct_mask,
                        debug_label.as_deref(),
                    );
                    filter.report_selectivity(idx, conjunct_mask.density());

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    };

    // Step 4: execute the projection, only at the mask for rows which match the filter
    let projection_future =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
    let projection_row_range = row_range.clone();
    let projection_debug_label = ctx.debug_label.clone();
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            log_mask_batch(
                "v1 scan batch skipped",
                projection_debug_label.as_deref(),
                &projection_row_range,
                &projection_row_range,
                &mask,
                None,
                None,
            );
            return Ok(None);
        }

        let start = Instant::now();
        let array = projection_future.await?;
        log_mask_batch(
            "v1 scan batch projected",
            projection_debug_label.as_deref(),
            &projection_row_range,
            &projection_row_range,
            &mask,
            Some(start.elapsed()),
            Some(array.len()),
        );
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

#[allow(clippy::too_many_arguments)]
fn log_conjunct_eval(
    message: &'static str,
    conjunct_idx: usize,
    conjunct: &Expression,
    row_range: &Range<u64>,
    input_rows: usize,
    output_rows: usize,
    elapsed: Duration,
    input_mask: &Mask,
    output_mask: &Mask,
    debug_label: Option<&str>,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let selectivity = if input_rows == 0 {
        0.0
    } else {
        output_rows as f64 / input_rows as f64
    };
    let input_coords = mask_coordinate_summary(input_mask, row_range);
    let output_coords = mask_coordinate_summary(output_mask, row_range);
    tracing::debug!(
        conjunct_idx,
        scan_label = debug_label.unwrap_or(""),
        conjunct = %conjunct,
        row_start = row_range.start,
        row_end = row_range.end,
        input_rows,
        output_rows,
        selectivity,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        input_coord_rows = input_coords.rows,
        input_coord_true_rows = input_coords.true_rows,
        input_coord_density = input_coords.density,
        input_coord_first_row = ?input_coords.first_row,
        input_coord_last_row = ?input_coords.last_row,
        input_coord_hash = input_coords.coord_hash,
        input_coord_sum = input_coords.coord_sum,
        input_coord_xor = input_coords.coord_xor,
        input_coord_sample = input_coords.sample_ranges.as_str(),
        output_coord_rows = output_coords.rows,
        output_coord_true_rows = output_coords.true_rows,
        output_coord_density = output_coords.density,
        output_coord_first_row = ?output_coords.first_row,
        output_coord_last_row = ?output_coords.last_row,
        output_coord_hash = output_coords.coord_hash,
        output_coord_sum = output_coords.coord_sum,
        output_coord_xor = output_coords.coord_xor,
        output_coord_sample = output_coords.sample_ranges.as_str(),
        message
    );
}

/// Information needed to execute a single split task.
pub struct TaskContext<A> {
    /// A row selection to apply.
    pub selection: Selection,
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: Expression,
    /// Function that maps into an A.
    pub mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Optional label included in debug/trace logs for correlating scan work.
    pub debug_label: Option<Arc<str>>,
}
