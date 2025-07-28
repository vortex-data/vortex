// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl TakeKernel for DateTimePartsVTable {
    fn take(&self, array: &DateTimePartsArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            take(array.days(), indices)?,
            take(array.seconds(), indices)?,
            take(array.subseconds(), indices)?,
        )?
        .into_array())
    }
}

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
