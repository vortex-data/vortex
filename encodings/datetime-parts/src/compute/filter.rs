// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DateTimeParts;
use crate::array::DateTimePartsArraySlotsExt;
impl FilterReduce for DateTimeParts {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            DateTimeParts::try_new(
                array.dtype().clone(),
                array.days().filter(mask.clone())?,
                array.seconds().filter(mask.clone())?,
                array.subseconds().filter(mask.clone())?,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;

    use crate::DateTimeParts;

    #[test]
    fn test_filter_datetime_parts() {
        let mut ctx = array_session().create_execution_ctx();
        // Create temporal arrays and convert to DateTimePartsArray
        let timestamps = buffer![
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ]
        .into_array();

        let temporal =
            TemporalArray::new_timestamp(timestamps, TimeUnit::Milliseconds, Some("UTC".into()));

        let array = DateTimeParts::try_from_temporal(temporal, &mut ctx).unwrap();
        test_filter_conformance(&array.into_array());

        // Test with nullable values
        let timestamps = PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ])
        .into_array();

        let temporal =
            TemporalArray::new_timestamp(timestamps, TimeUnit::Milliseconds, Some("UTC".into()));

        let array = DateTimeParts::try_from_temporal(temporal, &mut ctx).unwrap();
        test_filter_conformance(&array.into_array());
    }
}
