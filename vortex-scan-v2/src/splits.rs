// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_layout::plan::Concat;
use vortex_layout::plan::Eval;
use vortex_layout::plan::Pack;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::RowIdx;
use vortex_layout::plan::RowIdxPartition;
use vortex_layout::plan::Take;
use vortex_layout::plan::Zoned;
use vortex_scan::selection::Selection;

const IDEAL_SPLIT_SIZE: u64 = 100_000;
const MAX_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;
const MIN_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;

/// Defines how a plan scan is divided into independently executable row ranges.
#[derive(Default, Copy, Clone, Debug)]
pub enum SplitBy {
    /// Uses boundaries exposed by the optimized physical plan.
    #[default]
    Layout,
    /// Splits every `n` rows.
    RowCount(usize),
}

impl SplitBy {
    pub(crate) fn splits(
        &self,
        plans: &[&PlanRef],
        row_range: &Range<u64>,
    ) -> VortexResult<Vec<u64>> {
        let mut boundaries = match *self {
            Self::Layout => {
                let mut boundaries = vec![row_range.start];
                for plan in plans {
                    collect_plan_splits(plan, 0, row_range, &mut boundaries)?;
                }
                boundaries
            }
            Self::RowCount(row_count) => {
                vortex_ensure!(row_count > 0, "Row-count split size must be non-zero");
                row_range
                    .clone()
                    .step_by(row_count)
                    .chain(once(row_range.end))
                    .collect()
            }
        };
        boundaries.sort_unstable();
        boundaries.dedup();
        Ok(subdivide_large_spans(boundaries, IDEAL_SPLIT_SIZE))
    }
}

fn collect_plan_splits(
    plan: &PlanRef,
    row_offset: u64,
    row_range: &Range<u64>,
    boundaries: &mut Vec<u64>,
) -> VortexResult<()> {
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        if let Some(data) = zoned.data_plan()? {
            collect_plan_splits(&data, row_offset, row_range, boundaries)?;
        }
        return Ok(());
    }

    if plan.is::<Eval>() || plan.is::<RowIdx>() {
        if let Some(child) = plan.child(0)? {
            collect_plan_splits(&child, row_offset, row_range, boundaries)?;
        }
        return Ok(());
    }

    if plan.is::<Take>() {
        if let Some(codes) = plan.child(0)? {
            collect_plan_splits(&codes, row_offset, row_range, boundaries)?;
        }
        return Ok(());
    }

    // A childless Pack preserves row count even though it exposes no leaf boundaries.
    if plan.is::<Pack>() && plan.child_count() == 0 {
        boundaries.push(row_offset + row_range.end);
        return Ok(());
    }

    if plan.is::<Pack>() || plan.is::<RowIdxPartition>() {
        for index in 0..plan.child_count() {
            if let Some(child) = plan.child(index)?
                && child.row_count() == plan.row_count()
            {
                collect_plan_splits(&child, row_offset, row_range, boundaries)?;
            }
        }
        return Ok(());
    }

    if plan.is::<Concat>() {
        let mut chunk_offset = 0_u64;
        for index in 0..plan.child_count() {
            let Some(chunk) = plan.child(index)? else {
                continue;
            };
            let chunk_end = chunk_offset
                .checked_add(chunk.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Chunk row offset overflow"))?;
            let start = row_range.start.max(chunk_offset);
            let end = row_range.end.min(chunk_end);
            if start < end {
                let child_range = start - chunk_offset..end - chunk_offset;
                collect_plan_splits(&chunk, row_offset + chunk_offset, &child_range, boundaries)?;
                boundaries.push(row_offset + end);
            }
            chunk_offset = chunk_end;
        }
        return Ok(());
    }

    boundaries.push(row_offset + row_range.end);
    Ok(())
}

fn subdivide_large_spans(boundaries: Vec<u64>, max_span: u64) -> Vec<u64> {
    if boundaries.len() < 2
        || boundaries
            .windows(2)
            .all(|window| window[1] - window[0] <= max_span)
    {
        return boundaries;
    }

    let mut output = Vec::with_capacity(boundaries.len() * 2);
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        output.push(start);
        let span = end - start;
        if span > max_span {
            let split_count = span.div_ceil(max_span);
            let split_size = span.div_ceil(split_count);
            let mut point = start + split_size;
            while point < end {
                output.push(point);
                point = point.saturating_add(split_size);
            }
        }
    }
    if let Some(&last) = boundaries.last() {
        output.push(last);
    }
    output
}

pub(crate) enum Splits {
    Natural(Vec<u64>),
    Ranges(Vec<Range<u64>>),
}

pub(crate) fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };
    if row_range.is_some() {
        return None;
    }
    let indices = buffer.as_slice();
    if indices.is_empty() {
        return Some(Vec::new());
    }

    let mut ranges = Vec::with_capacity((indices.len() as u64 / MAX_RANGE_SIZE) as usize);
    let mut current_start = indices[0];
    let mut current_end = indices[0] + 1;
    for &index in &indices[1..] {
        let new_range_size = (index + 1) - current_start;
        let gap = (index + 1) - current_end;
        if new_range_size >= MAX_RANGE_SIZE {
            if gap < MIN_GAP_BETWEEN_RANGES {
                return None;
            }
            ranges.push(current_start..current_end);
            current_start = index;
        }
        current_end = index + 1;
    }
    ranges.push(current_start..current_end);
    Some(ranges)
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::StructFields;
    use vortex_error::VortexResult;
    use vortex_layout::plan::PackPlan;

    use super::SplitBy;

    #[test]
    fn childless_pack_preserves_its_row_count() -> VortexResult<()> {
        let plan = PackPlan::try_new(
            StructFields::empty(),
            Nullability::NonNullable,
            6,
            Vec::new(),
            None,
        )?
        .into_plan();

        assert_eq!(SplitBy::Layout.splits(&[&plan], &(0..6))?, vec![0, 6]);
        Ok(())
    }
}
