// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::plan::physical::Concat;
use crate::plan::physical::Eval;
use crate::plan::physical::Filter;
use crate::plan::physical::LegacyStats;
use crate::plan::physical::Pack;
use crate::plan::physical::PlanRef;
use crate::plan::physical::Take;
use crate::plan::physical::Zoned;

/// Finds dictionary regions of a projection's only string column without visiting values or
/// unrelated fields. Other projection shapes leave partitioning to the driver.
pub(super) fn string_dictionary_chunks(
    projection: &PlanRef,
    rows: &Range<u64>,
) -> VortexResult<Option<Vec<Range<u64>>>> {
    let strings = match projection.dtype() {
        DType::Struct(fields, _) => fields
            .fields()
            .filter(|dtype| dtype.is_utf8() || dtype.is_binary())
            .take(2)
            .count(),
        dtype => usize::from(dtype.is_utf8() || dtype.is_binary()),
    };
    if strings != 1 {
        return Ok(None);
    }
    let mut pending = vec![(projection.clone(), 0)];
    let mut chunks = Vec::new();
    while let Some((plan, offset)) = pending.pop() {
        let start = offset.max(rows.start);
        let end = (offset + plan.row_count()).min(rows.end);
        if start >= end {
            continue;
        }
        if plan.is::<Take>() && (plan.dtype().is_utf8() || plan.dtype().is_binary()) {
            chunks.push(start..end);
        } else if let Some(concat) = plan.as_opt::<Concat>() {
            let offsets = concat.row_offsets();
            for index in (0..offsets.len()).rev() {
                let child_start = offset + offsets[index];
                let child_end = offset
                    + offsets
                        .get(index + 1)
                        .copied()
                        .unwrap_or_else(|| plan.row_count());
                if child_start < rows.end && rows.start < child_end {
                    pending.push((concat.child_required(index)?, child_start));
                }
            }
        } else if let Some(pack) = plan.as_opt::<Pack>() {
            let mut strings = pack
                .fields()
                .fields()
                .enumerate()
                .filter(|(_, dtype)| dtype.is_utf8() || dtype.is_binary());
            let Some((index, _)) = strings.next() else {
                return Ok(None);
            };
            if strings.next().is_some() {
                return Ok(None);
            }
            pending.push((pack.child_required(index)?, offset));
        } else if let Some(filter) = plan.as_opt::<Filter>() {
            pending.push((filter.child_plan()?, offset));
        } else if let Some(zoned) = plan.as_opt::<Zoned>() {
            let Some(data) = zoned.data_plan()? else {
                return Ok(None);
            };
            pending.push((data, offset));
        } else if let Some(stats) = plan.as_opt::<LegacyStats>() {
            pending.push((stats.data_plan()?, offset));
        } else if let Some(eval) = plan.as_opt::<Eval>() {
            let child = eval.child_plan()?;
            if child.dtype() != plan.dtype() {
                return Ok(None);
            }
            pending.push((child, offset));
        } else {
            return Ok(None);
        }
    }
    Ok(Some(chunks))
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_error::VortexResult;
    use vortex_session::registry::CachedId;
    use vortex_session::registry::ReadContext;

    use super::string_dictionary_chunks;
    use crate::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::layouts::dict::DictLayout;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::foreign::new_foreign_layout;
    use crate::layouts::struct_::StructLayout;
    use crate::plan::physical::ConcatPlan;
    use crate::plan::physical::FilterPlan;
    use crate::plan::physical::PackPlan;
    use crate::plan::physical::PlanRef;
    use crate::plan::physical::SegmentScanPlan;
    use crate::plan::physical::TakePlan;
    use crate::plan::physical::lower;
    use crate::segments::SegmentId;

    fn segment(dtype: DType, rows: u64) -> PlanRef {
        SegmentScanPlan::new(dtype, rows, SegmentId::from(0), ReadContext::new([]), None)
            .into_plan()
    }

    fn dictionary(rows: u64) -> PlanRef {
        TakePlan::new(
            segment(DType::Primitive(PType::U32, Nullability::NonNullable), rows),
            segment(DType::Utf8(Nullability::Nullable), 5),
        )
        .into_plan()
    }

    #[test]
    fn nested_dictionary_regions_are_clipped_in_file_coordinates() -> VortexResult<()> {
        let dtype = DType::Utf8(Nullability::Nullable);
        let inner = ConcatPlan::try_new(dtype.clone(), vec![dictionary(20), dictionary(30)])?;
        let plan = ConcatPlan::try_new(dtype, vec![dictionary(10), inner.into_plan()])?;
        let plan = FilterPlan::new(plan.into_plan()).into_plan();
        assert_eq!(
            string_dictionary_chunks(&plan, &(7..47))?,
            Some(vec![7..10, 10..30, 30..47])
        );
        Ok(())
    }

    #[test]
    fn a_numeric_field_does_not_add_dictionary_boundaries() -> VortexResult<()> {
        let strings = dictionary(50);
        let numbers = segment(DType::Primitive(PType::U64, Nullability::NonNullable), 50);
        let fields = StructFields::from_iter([
            ("strings", strings.dtype().clone()),
            ("numbers", numbers.dtype().clone()),
        ]);
        let plan = PackPlan::try_new(
            fields,
            Nullability::NonNullable,
            50,
            vec![strings, numbers],
            None,
        )?
        .into_plan();
        assert_eq!(
            string_dictionary_chunks(&plan, &(0..50))?,
            Some(vec![0..50])
        );
        Ok(())
    }

    #[test]
    fn multiple_string_columns_leave_partitioning_to_the_driver() -> VortexResult<()> {
        let strings = dictionary(50);
        let fields = StructFields::from_iter([
            ("left", strings.dtype().clone()),
            ("right", strings.dtype().clone()),
        ]);
        let plan = PackPlan::try_new(
            fields,
            Nullability::NonNullable,
            50,
            vec![strings.clone(), strings],
            None,
        )?
        .into_plan();
        assert!(string_dictionary_chunks(&plan, &(0..50))?.is_none());
        Ok(())
    }

    #[test]
    fn mixed_flat_and_dictionary_strings_leave_partitioning_to_the_driver() -> VortexResult<()> {
        let dtype = DType::Utf8(Nullability::Nullable);
        let plan = ConcatPlan::try_new(dtype.clone(), vec![dictionary(10), segment(dtype, 20)])?
            .into_plan();
        assert!(string_dictionary_chunks(&plan, &(0..30))?.is_none());
        assert_eq!(
            string_dictionary_chunks(&plan, &(0..10))?,
            Some(vec![0..10])
        );
        Ok(())
    }

    #[test]
    fn dictionary_values_and_unrelated_fields_are_not_lowered() -> VortexResult<()> {
        static ID: CachedId = CachedId::new("vortex.test.unsupported");
        let strings = DType::Utf8(Nullability::Nullable);
        let numbers = DType::Primitive(PType::U32, Nullability::NonNullable);
        let values = new_foreign_layout(*ID, strings.clone(), 5, vec![], vec![], vec![]);
        let codes = FlatLayout::new(
            50,
            numbers.clone(),
            SegmentId::from(0),
            ReadContext::new([]),
        )
        .into_layout();
        let layout = StructLayout::new(
            50,
            DType::Struct(
                StructFields::from_iter([("strings", strings), ("numbers", numbers.clone())]),
                Nullability::NonNullable,
            ),
            vec![
                DictLayout::new(values, codes).into_layout(),
                new_foreign_layout(*ID, numbers, 50, vec![], vec![], vec![]),
            ],
        )
        .into_layout();
        assert_eq!(
            string_dictionary_chunks(&lower(&layout)?, &(0..50))?,
            Some(vec![0..50])
        );
        Ok(())
    }

    #[test]
    fn numeric_projection_does_not_initialize_chunk_plans() -> VortexResult<()> {
        static ID: CachedId = CachedId::new("vortex.test.unsupported");
        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let layout = ChunkedLayout::new(
            50,
            dtype.clone(),
            OwnedLayoutChildren::layout_children(vec![new_foreign_layout(
                *ID,
                dtype,
                50,
                vec![],
                vec![],
                vec![],
            )]),
        )
        .into_layout();
        assert!(string_dictionary_chunks(&lower(&layout)?, &(0..50))?.is_none());
        Ok(())
    }
}
