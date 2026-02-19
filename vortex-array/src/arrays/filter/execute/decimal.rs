// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_mask::MaskValues;

use crate::arrays::DecimalArray;
use crate::arrays::filter::execute::buffer;
use crate::arrays::filter::execute::filter_validity;
use crate::match_each_decimal_value_type;
use crate::vtable::ValidityHelper;

pub fn filter_decimal(array: &DecimalArray, mask: &Arc<MaskValues>) -> DecimalArray {
    let filtered_validity = filter_validity(array.validity().clone(), mask);

    match_each_decimal_value_type!(array.values_type(), |T| {
        let filtered_buffer = buffer::filter_buffer(array.buffer::<T>(), mask.as_ref());

        // SAFETY: We filter both the validity and the buffer with the same mask, so they must have
        // the same length, and since the buffer came from an existing and valid `DecimalArray` the
        // values must all be be representable by the decimal type.
        unsafe {
            DecimalArray::new_unchecked(filtered_buffer, array.decimal_dtype(), filtered_validity)
        }
    })
}

#[cfg(test)]
mod test {
    use crate::arrays::DecimalArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DecimalDType;

    #[test]
    fn test_filter_decimal128_conformance() {
        let decimal_dtype = DecimalDType::new(38, 2);
        let values = vec![
            Some(12345i128),
            Some(67890),
            Some(-12345),
            Some(0),
            Some(99999),
        ];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(array.as_ref());
    }

    #[test]
    fn test_filter_decimal128_with_nulls_conformance() {
        let decimal_dtype = DecimalDType::new(38, 4);
        let values = vec![Some(12345i128), None, Some(-12345), Some(0), None];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(array.as_ref());
    }
}
