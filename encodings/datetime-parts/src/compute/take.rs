// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, fill_null, take};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl TakeKernel for DateTimePartsVTable {
    fn take(&self, array: &DateTimePartsArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // we go ahead and canonicalize here to avoid worst-case canonicalizing 3 separate times
        let indices = indices.to_primitive()?;

        let taken_days = take(array.days(), indices)?;
        let taken_seconds = take(array.seconds(), indices)?;
        let taken_subseconds = take(array.subseconds(), indices)?;

        // Update the dtype if the nullability changed due to nullable indices
        let dtype = if taken_days.dtype().is_nullable() != array.dtype().is_nullable() {
            array
                .dtype()
                .with_nullability(taken_days.dtype().nullability())
        } else {
            array.dtype().clone()
        };

        // DateTimePartsArray requires seconds and subseconds to be non-nullable.
        // If they became nullable due to nullable indices, we need to fill nulls.
        let (taken_seconds, taken_subseconds) = if taken_seconds.dtype().is_nullable() {
            if !taken_days.dtype().is_nullable() {
                vortex_panic!("Mismatched types: days is not nullable, seconds is nullable");
            }
            if !taken_subseconds.dtype().is_nullable() {
                vortex_panic!("Mismatched types: seconds is nullable, subseconds is not");
            }
            if !indices.dtype().is_nullable() {
                vortex_panic!("Mismatched types: indices are not nullable, seconds is nullable");
            }

            // get the first non-null index, and use that to fill both seconds and subseconds
            let fill_index: Option<usize> = find_first_non_null_value(indices).map(|i| i.as_());

            let seconds_fill = fill_index
                .map(|i| taken_seconds.scalar_at(i))
                .transpose()?
                .unwrap_or_else(|| {
                    Scalar::primitive(0i64, Nullability::NonNull).cast(taken_seconds.dtype())
                });

            let subseconds_fill = fill_index
                .map(|i| taken_subseconds.scalar_at(i))
                .transpose()?
                .unwrap_or_else(|| {
                    Scalar::primitive(0i64, Nullability::NonNull).cast(taken_subseconds.dtype())
                });

            let taken_seconds = fill_null(taken_seconds.as_ref(), &seconds_fill)?;
            let taken_subseconds = fill_null(taken_subseconds.as_ref(), &subseconds_fill)?;
            (taken_seconds, taken_subseconds)
        } else {
            (taken_seconds, taken_subseconds)
        };

        Ok(
            DateTimePartsArray::try_new(dtype, taken_days, taken_seconds, taken_subseconds)?
                .into_array(),
        )
    }
}

fn find_first_non_null_value(array: &dyn Array) -> Option<Scalar> {}

register_kernel!(TakeKernelAdapter(DateTimePartsVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_dtype::datetime::TimeUnit;

    use crate::DateTimePartsArray;

    #[test]
    fn test_take_datetime_parts_conformance() {
        // Test with non-nullable datetime parts
        let timestamps = PrimitiveArray::from_iter([
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ])
        .into_array();

        let temporal =
            TemporalArray::new_timestamp(timestamps, TimeUnit::Ms, Some("UTC".to_string()));

        let array = DateTimePartsArray::try_from(temporal).unwrap();
        test_take_conformance(array.as_ref());

        // Test with nullable datetime parts
        let timestamps = PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ])
        .into_array();

        let temporal =
            TemporalArray::new_timestamp(timestamps, TimeUnit::Ms, Some("UTC".to_string()));

        let array = DateTimePartsArray::try_from(temporal).unwrap();
        test_take_conformance(array.as_ref());

        // Test with single element
        let timestamps = PrimitiveArray::from_iter([86_400_000i64]).into_array();
        let temporal =
            TemporalArray::new_timestamp(timestamps, TimeUnit::Ms, Some("UTC".to_string()));
        let array = DateTimePartsArray::try_from(temporal).unwrap();
        test_take_conformance(array.as_ref());
    }
}
