// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;

impl FilterReduce for DecimalByteParts {
    fn filter(array: &DecimalBytePartsArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        DecimalBytePartsArray::try_new(array.msp().filter(mask.clone())?, *array.decimal_dtype())
            .map(|d| Some(d.into_array()))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::DecimalDType;
    use vortex_buffer::buffer;

    use crate::DecimalBytePartsArray;

    #[test]
    fn test_filter_decimal_byte_parts() {
        // Create test data with 5 signed integer values
        let msp = buffer![100i32, 200, 300, 400, 500].into_array();

        let decimal_dtype = DecimalDType::new(8, 2);
        let array = DecimalBytePartsArray::try_new(msp, decimal_dtype).unwrap();
        test_filter_conformance(&array.into_array());

        // Test with nullable values
        let msp = PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), Some(40), None])
            .into_array();

        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalBytePartsArray::try_new(msp, decimal_dtype).unwrap();
        test_filter_conformance(&array.into_array());
    }
}
