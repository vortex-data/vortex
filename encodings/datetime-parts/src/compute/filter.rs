// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl FilterKernel for DateTimePartsVTable {
    fn filter(&self, array: &DateTimePartsArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            filter(array.days().as_ref(), mask)?,
            filter(array.seconds().as_ref(), mask)?,
            filter(array.subseconds().as_ref(), mask)?,
        )?
        .into_array())
    }
}
register_kernel!(FilterKernelAdapter(DateTimePartsVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_dtype::datetime::TimeUnit;

    use crate::DateTimePartsArray;

    #[test]
    fn test_filter_datetime_parts() {
        // Create temporal arrays and convert to DateTimePartsArray
        let timestamps = PrimitiveArray::from_iter([
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ])
        .into_array();

        let temporal = TemporalArray::new_timestamp(
            timestamps,
            TimeUnit::Milliseconds,
            Some("UTC".to_string()),
        );

        let array = DateTimePartsArray::try_from(temporal).unwrap();
        test_filter_conformance(array.as_ref());

        // Test with nullable values
        let timestamps = PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ])
        .into_array();

        let temporal = TemporalArray::new_timestamp(
            timestamps,
            TimeUnit::Milliseconds,
            Some("UTC".to_string()),
        );

        let array = DateTimePartsArray::try_from(temporal).unwrap();
        test_filter_conformance(array.as_ref());
    }
}
