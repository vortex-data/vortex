// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Admission of overlapping list ranges, charged for whole stored element chunks.

use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::plan::physical::Concat;
use crate::plan::physical::Filter;
use crate::plan::physical::LegacyStats;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScan;
use crate::plan::physical::Zoned;
use crate::plan::physical::pipeline::overlapping;

const MAX_ELEMENT_BYTES: u64 = 8 * 1024 * 1024;

/// Overlap grouped rows only when all touched chunks fit the decoded-element estimate.
/// This limits additional range concurrency, not total heap use or coalesced IO buffers.
pub(super) fn can_overlap(
    elements: &PlanRef,
    ranges: &[Range<u64>],
    selected: usize,
    parallelism: usize,
) -> VortexResult<bool> {
    if parallelism <= 1 || ranges.len() <= 1 || selected / ranges.len() < 2 {
        return Ok(false);
    }
    let mut bytes = 0_u64;
    for range in ranges {
        let Some(estimate) = range_bytes(elements, range)? else {
            return Ok(false);
        };
        // Charging a shared chunk again is conservative and avoids a per-scan set.
        bytes = bytes.saturating_add(estimate);
        if bytes > MAX_ELEMENT_BYTES {
            return Ok(false);
        }
    }
    Ok(true)
}

fn range_bytes(plan: &PlanRef, rows: &Range<u64>) -> VortexResult<Option<u64>> {
    if let Some(filter) = plan.as_opt::<Filter>() {
        return range_bytes(&filter.child_plan()?, rows);
    }
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        return match zoned.data_plan()? {
            Some(data) => range_bytes(&data, rows),
            None => Ok(None),
        };
    }
    if let Some(stats) = plan.as_opt::<LegacyStats>() {
        return range_bytes(&stats.data_plan()?, rows);
    }
    if let Some(concat) = plan.as_opt::<Concat>() {
        let mut bytes = 0_u64;
        for (index, start, local) in overlapping(concat, rows) {
            let Some(estimate) = range_bytes(
                &concat.child_required(index)?,
                &(local.start - start..local.end - start),
            )?
            else {
                return Ok(None);
            };
            bytes = bytes.saturating_add(estimate);
            if bytes > MAX_ELEMENT_BYTES {
                break;
            }
        }
        return Ok(Some(bytes));
    }
    if !plan.is::<SegmentScan>() {
        return Ok(None);
    }
    let width = match plan.dtype() {
        DType::Primitive(ptype, _) => ptype.byte_width() as u64,
        DType::Bool(_) => 1,
        _ => return Ok(None),
    };
    let validity = if plan.dtype().is_nullable() {
        plan.row_count().div_ceil(8)
    } else {
        0
    };
    Ok(Some(
        plan.row_count()
            .saturating_mul(width)
            .saturating_add(validity),
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::PType;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::plan::physical::ConcatPlan;
    use crate::plan::physical::LegacyStatsPlan;
    use crate::plan::physical::SegmentScanPlan;
    use crate::segments::SegmentId;

    fn segment(rows: u64, dtype: DType) -> PlanRef {
        SegmentScanPlan::new(dtype, rows, SegmentId::from(0), ReadContext::new([]), None)
            .into_plan()
    }

    #[rstest]
    #[case::grouped(4, 30, true)]
    #[case::isolated(2, 30, false)]
    #[case::single_worker(4, 1, false)]
    fn overlap_requires_grouped_parallel_work(
        #[case] selected: usize,
        #[case] workers: usize,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let elements = segment(1024, DType::Primitive(PType::I64, false.into()));
        assert_eq!(
            can_overlap(&elements, &[0..2, 10..12], selected, workers)?,
            expected
        );
        Ok(())
    }

    #[rstest]
    #[case::within_limit(MAX_ELEMENT_BYTES / 16, false, true)]
    #[case::large_chunk(MAX_ELEMENT_BYTES / 8, false, false)]
    #[case::validity_counts(MAX_ELEMENT_BYTES / 16, true, false)]
    fn charges_whole_chunks_and_validity(
        #[case] rows: u64,
        #[case] nullable: bool,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let elements = segment(rows, DType::Primitive(PType::I64, nullable.into()));
        assert_eq!(can_overlap(&elements, &[0..2, 10..12], 4, 30)?, expected);
        Ok(())
    }

    #[test]
    fn skips_untouched_chunks_and_zone_data() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, false.into());
        let data = ConcatPlan::try_new(
            dtype.clone(),
            vec![
                segment(1024, dtype.clone()),
                segment(10_000_000, dtype.clone()),
            ],
        )?
        .into_plan();
        let elements =
            LegacyStatsPlan::new(data, segment(10_000_000, dtype), 1024, [].into()).into_plan();
        assert!(can_overlap(&elements, &[0..2, 10..12], 4, 30)?);
        assert!(!can_overlap(&elements, &[0..2, 1024..1026], 4, 30)?);
        Ok(())
    }

    #[test]
    fn unknown_width_and_overflow_stay_sequential() -> VortexResult<()> {
        let unknown = segment(10, DType::Utf8(false.into()));
        let overflow = segment(u64::MAX, DType::Primitive(PType::I64, true.into()));
        for plan in [unknown, overflow] {
            assert!(!can_overlap(&plan, &[0..2, 5..7], 4, 30)?);
        }
        Ok(())
    }
}
