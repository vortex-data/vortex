// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, fill_null, take};
use vortex_array::stats::{Stat, StatsProvider};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_dtype::Nullability;
use vortex_error::{VortexResult, vortex_panic};
use vortex_scalar::Scalar;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl TakeKernel for DateTimePartsVTable {
    fn take(&self, array: &DateTimePartsArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // we go ahead and canonicalize here to avoid worst-case canonicalizing 3 separate times
        let indices = indices.to_primitive()?;

        let taken_days = take(array.days(), indices.as_ref())?;
        let taken_seconds = take(array.seconds(), indices.as_ref())?;
        let taken_subseconds = take(array.subseconds(), indices.as_ref())?;

        // Update the dtype if the nullability changed due to nullable indices
        let dtype = if taken_days.dtype().is_nullable() != array.dtype().is_nullable() {
            array
                .dtype()
                .with_nullability(taken_days.dtype().nullability())
        } else {
            array.dtype().clone()
        };

        if !taken_seconds.dtype().is_nullable() && !taken_subseconds.dtype().is_nullable() {
            return Ok(DateTimePartsArray::try_new(
                dtype,
                taken_days,
                taken_seconds,
                taken_subseconds,
            )?
            .into_array());
        }

        // DateTimePartsArray requires seconds and subseconds to be non-nullable.
        // If they became nullable due to nullable indices, we need to fill nulls.
        // But first, we need to check that the types are consistent.
        if !taken_days.dtype().is_nullable() {
            vortex_panic!("Mismatched types: days is not nullable, seconds is nullable");
        }
        if !taken_seconds.dtype().is_nullable() {
            vortex_panic!("Mismatched types: seconds is not nullable, days is nullable");
        }
        if !taken_subseconds.dtype().is_nullable() {
            vortex_panic!(
                "Mismatched types: subseconds is not nullable, days & seconds are nullable"
            );
        }
        if !indices.dtype().is_nullable() {
            vortex_panic!(
                "Mismatched types: indices are not nullable, days & seconds are nullable"
            );
        }

        let seconds_fill = array
            .seconds()
            .statistics()
            .get(Stat::Min)
            .map(|s| s.into_inner())
            .unwrap_or_else(|| Scalar::primitive(0i64, Nullability::NonNullable))
            .cast(array.seconds().dtype())?;
        let taken_seconds = fill_null(taken_seconds.as_ref(), &seconds_fill)?;

        let subseconds_fill = array
            .subseconds()
            .statistics()
            .get(Stat::Min)
            .map(|s| s.into_inner())
            .unwrap_or_else(|| Scalar::primitive(0i64, Nullability::NonNullable))
            .cast(array.subseconds().dtype())?;
        let taken_subseconds = fill_null(taken_subseconds.as_ref(), &subseconds_fill)?;

        Ok(
            DateTimePartsArray::try_new(dtype, taken_days, taken_seconds, taken_subseconds)?
                .into_array(),
        )
    }
}

register_kernel!(TakeKernelAdapter(DateTimePartsVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_dtype::datetime::TimeUnit;

    use crate::DateTimePartsArray;

    #[rstest]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter([
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".to_string())
    )).unwrap())]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".to_string())
    )).unwrap())]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter([86_400_000i64]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".to_string())
    )).unwrap())]
    fn test_take_datetime_parts_conformance(#[case] array: DateTimePartsArray) {
        test_take_conformance(array.as_ref());
    }
}
