use vortex_array::compute::{mask, FilterMask, MaskFn};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl MaskFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn mask(&self, array: &DateTimePartsArray, filter_mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone().as_nullable(),
            mask(array.days().as_ref(), filter_mask)?,
            array.seconds(),
            array.subsecond(),
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::TemporalArray;
    use vortex_array::compute::test_harness::test_mask;
    use vortex_array::IntoArrayData as _;
    use vortex_buffer::buffer;
    use vortex_datetime_dtype::TimeUnit;
    use vortex_dtype::DType;

    use crate::{split_temporal, DateTimePartsArray, TemporalParts};

    #[test]
    fn test_mask_datetime_parts_array() {
        let raw_millis = buffer![
            86_400i64,             // element with only day component
            86_400i64 + 1000,      // element with day + second components
            86_400i64 + 1000 + 1,  // element with day + second + sub-second components
            86_400i64 + 1000 + 5,  // element with day + second + sub-second components
            86_400i64 + 1000 + 55, // element with day + second + sub-second components
        ]
        .into_array();
        let temporal_array =
            TemporalArray::new_timestamp(raw_millis, TimeUnit::Ms, Some("UTC".to_string()));
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array.clone()).unwrap();
        let date_times = DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            days,
            seconds,
            subseconds,
        )
        .unwrap()
        .into_array();

        test_mask(date_times);
    }
}
