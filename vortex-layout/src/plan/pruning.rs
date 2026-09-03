// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::RowIdx;
use crate::plan::Zoned;

/// Returns whether `plan` can execute over `row_range` without reading data values.
///
/// Pruning plans can retain fallback branches that evaluate the original expression against data.
/// This predicate lets pruning operators compose proofs from children only when optimization has
/// replaced every relevant fallback with a pruning-only source.
pub fn uses_only_pruning_sources(plan: &PlanRef, row_range: &Range<u64>) -> VortexResult<bool> {
    vortex_ensure!(
        row_range.start <= row_range.end && row_range.end <= plan.row_count(),
        "Pruning row range {row_range:?} is outside 0..{}",
        plan.row_count()
    );
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        return Ok(zoned.is_pruning());
    }
    if plan.is::<RowIdx>() || (plan.is::<Pack>() && plan.children().is_empty()) {
        return Ok(true);
    }
    if let Some(concat) = plan.as_opt::<Concat>() {
        if row_range.is_empty() {
            return Ok(true);
        }
        let first = concat
            .row_offsets()
            .partition_point(|&offset| offset <= row_range.start)
            .saturating_sub(1);
        let end = concat
            .row_offsets()
            .partition_point(|&offset| offset < row_range.end);
        for child_index in first..end {
            let child_start = concat.row_offsets()[child_index];
            let child_end = concat
                .row_offsets()
                .get(child_index + 1)
                .copied()
                .unwrap_or_else(|| concat.row_count());
            let parent_start = row_range.start.max(child_start);
            let parent_end = row_range.end.min(child_end);
            if parent_start >= parent_end {
                continue;
            }
            let start = parent_start - child_start;
            let end = parent_end - child_start;
            if !uses_only_pruning_sources(&concat.child_required(child_index)?, &(start..end))? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if plan.is::<Eval>() || plan.is::<Pack>() {
        for child_index in 0..plan.child_count() {
            if !uses_only_pruning_sources(&plan.child_required(child_index)?, row_range)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if plan.children().is_empty() {
        return Ok(false);
    }
    for child in plan.children().iter() {
        let child = child?;
        let child_range = 0..child.row_count();
        if !uses_only_pruning_sources(&child, &child_range)? {
            return Ok(false);
        }
    }
    Ok(true)
}
