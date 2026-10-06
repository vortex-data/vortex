// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registers a split's reads with the segment source before the split runs.
//!
//! A registered read does no IO by itself, but a file's segment source coalesces it into any
//! nearby read that is issued, so reads registered for every split up front are served in a few
//! large IOs rather than many small ones. The layout reader registers its reads the same way.

use std::cmp;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::SegmentScan;
use crate::plan::Share;
use crate::plan::Take;
use crate::segments::SegmentId;

/// The most row ranges reads are asked for ahead of time one by one; a more scattered selection
/// is asked for as the range spanning it.
const MAX_PREFETCH_RANGES: usize = 64;

/// The row ranges to ask for ahead of time when `mask` selects rows of `rows`.
pub(crate) fn selected_ranges(rows: &Range<u64>, mask: &Mask) -> Vec<Range<u64>> {
    let start = rows.start;
    match mask.slices() {
        AllOr::All => vec![rows.clone()],
        AllOr::None => Vec::new(),
        AllOr::Some(slices) if slices.len() <= MAX_PREFETCH_RANGES => slices
            .iter()
            .map(|&(begin, end)| start + begin as u64..start + end as u64)
            .collect(),
        AllOr::Some(slices) => {
            let first = slices.first().map_or(0, |slice| slice.0);
            let last = slices.last().map_or(0, |slice| slice.1);
            vec![start + first as u64..start + last as u64]
        }
    }
}

/// Adds the segments `plan` is likely to read over `rows` to `segments`.
///
/// This is a hint, not a promise: it follows plans whose children cover the same rows, or a known
/// part of them, and stops at any other plan, such as a list whose elements cover other rows.
pub(crate) fn plan_segments(
    plan: &PlanRef,
    rows: Range<u64>,
    segments: &mut Vec<SegmentId>,
) -> VortexResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(scan) = plan.as_opt::<SegmentScan>() {
        segments.push(scan.segment_id());
    } else if let Some(concat) = plan.as_opt::<Concat>() {
        let offsets = concat.row_offsets();
        let first = offsets.partition_point(|&offset| offset <= rows.start);
        for (index, &offset) in offsets
            .iter()
            .enumerate()
            .skip(first.saturating_sub(1))
            .take_while(|&(_, &offset)| offset < rows.end)
        {
            let child = concat.child_required(index)?;
            let start = cmp::max(rows.start, offset);
            let end = cmp::min(rows.end, offset + child.row_count());
            plan_segments(&child, start - offset..end - offset, segments)?;
        }
    } else if let Some(take) = plan.as_opt::<Take>() {
        plan_segments(&take.codes()?, rows, segments)?;
        if take.cached_values().is_none() {
            let values = take.values()?;
            let len = values.row_count();
            plan_segments(&values, 0..len, segments)?;
        }
    } else if let Some(share) = plan.as_opt::<Share>() {
        if share.cached().is_none() {
            plan_segments(&share.child_plan()?, rows, segments)?;
        }
    } else if plan.is::<Pack>() || plan.is::<Filter>() || plan.is::<Eval>() {
        for child in plan.children().iter() {
            plan_segments(&child?, rows.clone(), segments)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::plan::ConcatPlan;
    use crate::plan::FilterPlan;
    use crate::plan::SegmentScanPlan;

    const DTYPE: DType = DType::Primitive(PType::I32, NonNullable);

    /// Four flat chunks of 1000 rows, in segments 0 to 3.
    fn chunked() -> VortexResult<PlanRef> {
        let chunks = (0..4)
            .map(|segment| {
                let scan = SegmentScanPlan::new(
                    DTYPE,
                    1000,
                    SegmentId::from(segment),
                    ReadContext::new(ArrayContext::empty().to_ids()),
                    None,
                );
                FilterPlan::new(scan.into_plan()).into_plan()
            })
            .collect();
        Ok(ConcatPlan::try_new(DTYPE, chunks)?.into_plan())
    }

    #[rstest]
    #[case::everything(0..4000, &[0, 1, 2, 3])]
    #[case::inside_one_chunk(1200..1800, &[1])]
    #[case::whole_chunk(1000..2000, &[1])]
    #[case::across_a_boundary(999..1001, &[0, 1])]
    #[case::several_chunks(500..2500, &[0, 1, 2])]
    #[case::last_row(3999..4000, &[3])]
    #[case::empty(1500..1500, &[])]
    fn concat_reads_only_the_chunks_it_overlaps(
        #[case] rows: Range<u64>,
        #[case] expected: &[u32],
    ) -> VortexResult<()> {
        let mut segments = Vec::new();
        plan_segments(&chunked()?, rows, &mut segments)?;
        let expected: Vec<SegmentId> = expected.iter().copied().map(SegmentId::from).collect();
        assert_eq!(segments, expected);
        Ok(())
    }
}
